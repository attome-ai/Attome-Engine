#include "ATMEngine.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <new>

#ifdef EMSCRIPTEN
#include <emscripten.h>
#endif

// Implementation of EntityChunk constructor
EntityChunk::EntityChunk(int chunk_capacity) {
    capacity = chunk_capacity;
    count = 0;

    // Allocate memory for all arrays with proper alignment
    size_t float_size = capacity * sizeof(float);
    size_t int_size = capacity * sizeof(int);
    size_t bool_size = capacity * sizeof(bool);
    size_t ptr_size = capacity * sizeof(int*);

    // Use aligned allocation for better SIMD performance
    x = static_cast<float*>(SDL_aligned_alloc(float_size, CACHE_LINE_SIZE));
    y = static_cast<float*>(SDL_aligned_alloc(float_size, CACHE_LINE_SIZE));
    right = static_cast<float*>(SDL_aligned_alloc(float_size, CACHE_LINE_SIZE));
    bottom = static_cast<float*>(SDL_aligned_alloc(float_size, CACHE_LINE_SIZE));
    width = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    height = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    texture_id = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    layer = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    active = static_cast<bool*>(SDL_aligned_alloc(bool_size, CACHE_LINE_SIZE));
    grid_cell = static_cast<int**>(SDL_aligned_alloc(ptr_size, CACHE_LINE_SIZE));

    // Initialize all memory to zero
    memset(x, 0, float_size);
    memset(y, 0, float_size);
    memset(right, 0, float_size);
    memset(bottom, 0, float_size);
    memset(width, 0, int_size);
    memset(height, 0, int_size);
    memset(texture_id, 0, int_size);
    memset(layer, 0, int_size);
    memset(active, 0, bool_size);
    memset(grid_cell, 0, ptr_size);
}

// Implementation of EntityChunk destructor
EntityChunk::~EntityChunk() {
    SDL_aligned_free(x);
    SDL_aligned_free(y);
    SDL_aligned_free(right);
    SDL_aligned_free(bottom);
    SDL_aligned_free(width);
    SDL_aligned_free(height);
    SDL_aligned_free(texture_id);
    SDL_aligned_free(layer);
    SDL_aligned_free(active);
    SDL_aligned_free(grid_cell);
}

// Implementation of EntityManager constructor
EntityManager::EntityManager() {
    chunk_count = 0;
    chunks_capacity = 4; // Start with space for 4 chunks
    total_count = 0;
    free_count = 0;
    free_capacity = 1024; // Initial capacity for free indices

    // Allocate memory for chunks array
    chunks = static_cast<EntityChunk**>(malloc(chunks_capacity * sizeof(EntityChunk*)));

    // Allocate memory for free indices
    free_indices = static_cast<int*>(malloc(free_capacity * sizeof(int)));
}

// Implementation of EntityManager destructor
EntityManager::~EntityManager() {
    // Free all chunks
    for (int i = 0; i < chunk_count; i++) {
        delete chunks[i];
    }

    // Free chunks array and free indices
    free(chunks);
    free(free_indices);
}

// Helper to get chunk and local indices from entity index
void EntityManager::getChunkIndices(int entity_idx, int* chunk_idx, int* local_idx) const {
    *chunk_idx = entity_idx / ENTITY_CHUNK_SIZE;
    *local_idx = entity_idx % ENTITY_CHUNK_SIZE;
}

// Add a new entity to the entity manager
int EntityManager::addEntity() {
    int entity_idx;

    // If there are free indices, reuse one
    if (free_count > 0) {
        entity_idx = free_indices[--free_count];
    }
    else {
        // No free indices, create a new entity
        entity_idx = total_count++;

        // Calculate which chunk this entity belongs to
        int chunk_idx, local_idx;
        getChunkIndices(entity_idx, &chunk_idx, &local_idx);

        // Ensure we have enough chunks
        if (chunk_idx >= chunk_count) {
            // Need to create a new chunk
            if (chunk_idx >= chunks_capacity) {
                // Need to resize chunks array
                chunks_capacity *= 2;
                chunks = static_cast<EntityChunk**>(realloc(chunks, chunks_capacity * sizeof(EntityChunk*)));
            }

            // Create new chunk(s) as needed
            while (chunk_count <= chunk_idx) {
                chunks[chunk_count] = new EntityChunk(ENTITY_CHUNK_SIZE);
                chunk_count++;
            }
        }

        // Increment count in the appropriate chunk
        chunks[chunk_idx]->count++;
    }

    return entity_idx;
}

// Remove an entity from the entity manager
void EntityManager::removeEntity(int entity_idx) {
    if (!isValidEntity(entity_idx)) {
        return;
    }

    // Calculate which chunk this entity belongs to
    int chunk_idx, local_idx;
    getChunkIndices(entity_idx, &chunk_idx, &local_idx);

    // Mark as inactive
    chunks[chunk_idx]->active[local_idx] = false;

    // Add to free indices
    if (free_count >= free_capacity) {
        // Need to resize free indices array
        free_capacity *= 2;
        free_indices = static_cast<int*>(realloc(free_indices, free_capacity * sizeof(int)));
    }

    free_indices[free_count++] = entity_idx;
}

// Check if an entity is valid
bool EntityManager::isValidEntity(int entity_idx) const {
    if (entity_idx < 0 || entity_idx >= total_count) {
        return false;
    }

    int chunk_idx, local_idx;
    getChunkIndices(entity_idx, &chunk_idx, &local_idx);

    if (chunk_idx >= chunk_count) {
        return false;
    }

    return local_idx < chunks[chunk_idx]->count;
}

// Initialize spatial grid for entity queries
void init_spatial_grid(SpatialGrid* grid, float world_width, float world_height, float cell_size) {
    grid->cell_size = cell_size;
    grid->width = (int)ceil(world_width / cell_size);
    grid->height = (int)ceil(world_height / cell_size);
    grid->total_cells = grid->width * grid->height;

    // Allocate cells array using a flat array for better cache locality
    grid->cells = static_cast<int***>(SDL_aligned_alloc(grid->height * sizeof(int**), CACHE_LINE_SIZE));
    grid->cell_counts = static_cast<int**>(SDL_aligned_alloc(grid->height * sizeof(int*), CACHE_LINE_SIZE));
    grid->cell_capacities = static_cast<int**>(SDL_aligned_alloc(grid->height * sizeof(int*), CACHE_LINE_SIZE));

    for (int y = 0; y < grid->height; y++) {
        grid->cells[y] = static_cast<int**>(SDL_aligned_alloc(grid->width * sizeof(int*), CACHE_LINE_SIZE));
        grid->cell_counts[y] = static_cast<int*>(SDL_aligned_alloc(grid->width * sizeof(int), CACHE_LINE_SIZE));
        grid->cell_capacities[y] = static_cast<int*>(SDL_aligned_alloc(grid->width * sizeof(int), CACHE_LINE_SIZE));

        for (int x = 0; x < grid->width; x++) {
            int initial_capacity = 32; // Start with space for 32 entities per cell
            grid->cell_counts[y][x] = 0;
            grid->cell_capacities[y][x] = initial_capacity;
            grid->cells[y][x] = static_cast<int*>(SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE));
        }
    }
}

// Free spatial grid
void free_spatial_grid(SpatialGrid* grid) {
    for (int y = 0; y < grid->height; y++) {
        for (int x = 0; x < grid->width; x++) {
            SDL_aligned_free(grid->cells[y][x]);
        }
        SDL_aligned_free(grid->cells[y]);
        SDL_aligned_free(grid->cell_counts[y]);
        SDL_aligned_free(grid->cell_capacities[y]);
    }

    SDL_aligned_free(grid->cells);
    SDL_aligned_free(grid->cell_counts);
    SDL_aligned_free(grid->cell_capacities);
}

// Clear all cells in the spatial grid
void clear_spatial_grid(SpatialGrid* grid) {
    for (int y = 0; y < grid->height; y++) {
        for (int x = 0; x < grid->width; x++) {
            grid->cell_counts[y][x] = 0;
        }
    }
}

// Add entity to spatial grid
void spatial_grid_add(SpatialGrid* grid, int entity_idx, float x, float y) {
    int grid_x = (int)(x / grid->cell_size);
    int grid_y = (int)(y / grid->cell_size);

    // Clamp to grid bounds
    grid_x = std::max(0, std::min(grid_x, grid->width - 1));
    grid_y = std::max(0, std::min(grid_y, grid->height - 1));

    // Ensure capacity
    int& count = grid->cell_counts[grid_y][grid_x];
    int& capacity = grid->cell_capacities[grid_y][grid_x];

    if (count >= capacity) {
        int new_capacity = capacity * 2;
        int* new_cell = static_cast<int*>(SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE));

        // Copy existing data
        memcpy(new_cell, grid->cells[grid_y][grid_x], count * sizeof(int));

        // Free old memory and update
        SDL_aligned_free(grid->cells[grid_y][grid_x]);
        grid->cells[grid_y][grid_x] = new_cell;
        capacity = new_capacity;
    }

    // Add entity to cell
    grid->cells[grid_y][grid_x][count++] = entity_idx;
}

// Query entities in a region
void spatial_grid_query(SpatialGrid* grid, SDL_FRect query_rect,
    int* result_indices, int* result_count, int max_results,
    EntityManager* entities) {
    // Calculate grid cells that overlap with query rect
    int start_x = (int)(query_rect.x / grid->cell_size);
    int start_y = (int)(query_rect.y / grid->cell_size);
    int end_x = (int)ceil((query_rect.x + query_rect.w) / grid->cell_size);
    int end_y = (int)ceil((query_rect.y + query_rect.h) / grid->cell_size);

    // Clamp to grid bounds
    start_x = std::max(0, std::min(start_x, grid->width - 1));
    start_y = std::max(0, std::min(start_y, grid->height - 1));
    end_x = std::max(0, std::min(end_x, grid->width - 1));
    end_y = std::max(0, std::min(end_y, grid->height - 1));

    *result_count = 0;

    // Iterate through cells that overlap the query rect
    for (int y = start_y; y <= end_y && *result_count < max_results; y++) {
        for (int x = start_x; x <= end_x && *result_count < max_results; x++) {
            int cell_entity_count = grid->cell_counts[y][x];
            int* cell_entities = grid->cells[y][x];

            // Add visible entities to result
            for (int i = 0; i < cell_entity_count && *result_count < max_results; i++) {
                int entity_idx = cell_entities[i];

                // Check if entity is valid and active
                if (!entities->isValidEntity(entity_idx)) {
                    continue;
                }

                // Get chunk and local index
                int chunk_idx, local_idx;
                entities->getChunkIndices(entity_idx, &chunk_idx, &local_idx);

                EntityChunk* chunk = entities->chunks[chunk_idx];

                if (!chunk->active[local_idx]) {
                    continue;
                }

                // Get entity bounds
                float ex = chunk->x[local_idx];
                float ey = chunk->y[local_idx];
                float er = chunk->right[local_idx];
                float eb = chunk->bottom[local_idx];

                // Check if entity is in view
                if (!(er <= query_rect.x || ex >= query_rect.x + query_rect.w ||
                    eb <= query_rect.y || ey >= query_rect.y + query_rect.h)) {
                    result_indices[(*result_count)++] = entity_idx;
                }
            }
        }
    }
}

// Modified radix sort for entity indices
void radix_sort_entities(int* indices, int count, EntityManager* entities) {
    if (count <= 1) return;

    // Allocate temporary arrays
    int* temp_indices = static_cast<int*>(SDL_aligned_alloc(count * sizeof(int), CACHE_LINE_SIZE));

    // Using 8 bits per pass (256 buckets)
    const int RADIX_BITS = 8;
    const int RADIX_SIZE = 1 << RADIX_BITS;
    const int RADIX_MASK = RADIX_SIZE - 1;

    // First sort by texture_id
    int count_texture[RADIX_SIZE] = { 0 };

    // Count frequencies for texture_id
    for (int i = 0; i < count; i++) {
        int entity_idx = indices[i];

        // Get chunk and local index
        int chunk_idx, local_idx;
        entities->getChunkIndices(entity_idx, &chunk_idx, &local_idx);

        EntityChunk* chunk = entities->chunks[chunk_idx];
        int texture_id = chunk->texture_id[local_idx] & RADIX_MASK;
        count_texture[texture_id]++;
    }

    // Compute prefix sum
    int total = 0;
    for (int i = 0; i < RADIX_SIZE; i++) {
        int temp = count_texture[i];
        count_texture[i] = total;
        total += temp;
    }

    // Sort by texture_id
    for (int i = 0; i < count; i++) {
        int entity_idx = indices[i];

        // Get chunk and local index
        int chunk_idx, local_idx;
        entities->getChunkIndices(entity_idx, &chunk_idx, &local_idx);

        EntityChunk* chunk = entities->chunks[chunk_idx];
        int texture_id = chunk->texture_id[local_idx] & RADIX_MASK;
        temp_indices[count_texture[texture_id]++] = entity_idx;
    }

    // Copy back to original array
    memcpy(indices, temp_indices, count * sizeof(int));

    // Now sort by layer within each texture group
    // First identify texture groups
    int texture_start[RADIX_SIZE];
    int texture_count[RADIX_SIZE];

    for (int i = 0; i < RADIX_SIZE; i++) {
        texture_start[i] = 0;
        texture_count[i] = 0;
    }

    // Count entities per texture
    for (int i = 0; i < count; i++) {
        int entity_idx = indices[i];

        // Get chunk and local index
        int chunk_idx, local_idx;
        entities->getChunkIndices(entity_idx, &chunk_idx, &local_idx);

        EntityChunk* chunk = entities->chunks[chunk_idx];
        int texture_id = chunk->texture_id[local_idx] & RADIX_MASK;
        texture_count[texture_id]++;
    }

    // Calculate start positions
    int pos = 0;
    for (int i = 0; i < RADIX_SIZE; i++) {
        texture_start[i] = pos;
        pos += texture_count[i];
    }

    // Sort each texture group by layer
    for (int tex = 0; tex < RADIX_SIZE; tex++) {
        int start = texture_start[tex];
        int size = texture_count[tex];

        if (size <= 1) continue; // Skip single-element groups

        int count_layer[RADIX_SIZE] = { 0 };

        // Count frequencies for this group
        for (int i = 0; i < size; i++) {
            int entity_idx = indices[start + i];

            // Get chunk and local index
            int chunk_idx, local_idx;
            entities->getChunkIndices(entity_idx, &chunk_idx, &local_idx);

            EntityChunk* chunk = entities->chunks[chunk_idx];
            int layer = chunk->layer[local_idx] & RADIX_MASK;
            count_layer[layer]++;
        }

        // Compute prefix sum for this group
        int group_total = 0;
        for (int i = 0; i < RADIX_SIZE; i++) {
            int temp = count_layer[i];
            count_layer[i] = group_total;
            group_total += temp;
        }

        // Sort this group by layer
        for (int i = 0; i < size; i++) {
            int entity_idx = indices[start + i];

            // Get chunk and local index
            int chunk_idx, local_idx;
            entities->getChunkIndices(entity_idx, &chunk_idx, &local_idx);

            EntityChunk* chunk = entities->chunks[chunk_idx];
            int layer = chunk->layer[local_idx] & RADIX_MASK;
            temp_indices[start + count_layer[layer]++] = entity_idx;
        }

        // Copy back to original array just for this texture group
        memcpy(&indices[start], &temp_indices[start], size * sizeof(int));
    }

    // Cleanup
    SDL_aligned_free(temp_indices);
}

// Initialize texture atlas
void init_texture_atlas(TextureAtlas* atlas, SDL_Renderer* renderer, int width, int height) {
    atlas->texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_TARGET, width, height);
    SDL_SetTextureBlendMode(atlas->texture, SDL_BLENDMODE_BLEND);
    atlas->region_capacity = 32;
    atlas->region_count = 0;
    atlas->regions = static_cast<SDL_FRect*>(SDL_aligned_alloc(
        atlas->region_capacity * sizeof(SDL_FRect), CACHE_LINE_SIZE));
}

// Get visible rect based on camera
SDL_FRect get_visible_rect(Camera* camera) {
    // Calculate world-space visible region
    float half_width = camera->width / (2.0f * camera->zoom);
    float half_height = camera->height / (2.0f * camera->zoom);

    SDL_FRect visible_rect = {
        camera->x - half_width,
        camera->y - half_height,
        camera->width / camera->zoom,
        camera->height / camera->zoom
    };

    return visible_rect;
}

// Add a quad to a render batch with efficient buffer usage
void add_to_batch(RenderBatch* batch, float x, float y, float w, float h,
    SDL_FRect tex_region, SDL_FColor color) {
    // Ensure we have enough space for vertices
    if (batch->vertex_count + 4 > batch->vertex_capacity) {
        int new_capacity = batch->vertex_capacity * 2;
        SDL_Vertex* new_vertices = static_cast<SDL_Vertex*>(SDL_aligned_alloc(
            new_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE));

        // Copy existing data
        memcpy(new_vertices, batch->vertices, batch->vertex_count * sizeof(SDL_Vertex));
        SDL_aligned_free(batch->vertices);
        batch->vertices = new_vertices;
        batch->vertex_capacity = new_capacity;
    }

    // Ensure we have enough space for indices
    if (batch->index_count + 6 > batch->index_capacity) {
        int new_capacity = batch->index_capacity * 2;
        int* new_indices = static_cast<int*>(SDL_aligned_alloc(
            new_capacity * sizeof(int), CACHE_LINE_SIZE));

        // Copy existing data
        memcpy(new_indices, batch->indices, batch->index_count * sizeof(int));
        SDL_aligned_free(batch->indices);
        batch->indices = new_indices;
        batch->index_capacity = new_capacity;
    }

    // Add vertices
    int base_vertex = batch->vertex_count;

    // Top-left
    batch->vertices[base_vertex].position.x = x;
    batch->vertices[base_vertex].position.y = y;
    batch->vertices[base_vertex].color = color;
    batch->vertices[base_vertex].tex_coord.x = tex_region.x;
    batch->vertices[base_vertex].tex_coord.y = tex_region.y;

    // Top-right
    batch->vertices[base_vertex + 1].position.x = x + w;
    batch->vertices[base_vertex + 1].position.y = y;
    batch->vertices[base_vertex + 1].color = color;
    batch->vertices[base_vertex + 1].tex_coord.x = tex_region.x + tex_region.w;
    batch->vertices[base_vertex + 1].tex_coord.y = tex_region.y;

    // Bottom-right
    batch->vertices[base_vertex + 2].position.x = x + w;
    batch->vertices[base_vertex + 2].position.y = y + h;
    batch->vertices[base_vertex + 2].color = color;
    batch->vertices[base_vertex + 2].tex_coord.x = tex_region.x + tex_region.w;
    batch->vertices[base_vertex + 2].tex_coord.y = tex_region.y + tex_region.h;

    // Bottom-left
    batch->vertices[base_vertex + 3].position.x = x;
    batch->vertices[base_vertex + 3].position.y = y + h;
    batch->vertices[base_vertex + 3].color = color;
    batch->vertices[base_vertex + 3].tex_coord.x = tex_region.x;
    batch->vertices[base_vertex + 3].tex_coord.y = tex_region.y + tex_region.h;

    // Add indices
    batch->indices[batch->index_count++] = base_vertex;
    batch->indices[batch->index_count++] = base_vertex + 1;
    batch->indices[batch->index_count++] = base_vertex + 2;
    batch->indices[batch->index_count++] = base_vertex;
    batch->indices[batch->index_count++] = base_vertex + 2;
    batch->indices[batch->index_count++] = base_vertex + 3;

    batch->vertex_count += 4;
}

// Find batch index for texture/layer combination
int find_batch_index(RenderBatch* batches, int batch_count, int texture_id, int layer) {
    for (int i = 0; i < batch_count; i++) {
        if (batches[i].texture_id == texture_id && batches[i].layer == layer) {
            return i;
        }
    }
    return -1; // Not found
}

// Create a new batch for a texture/layer combination - FIXED FROM ORIGINAL
void create_batch(RenderBatch** batches, int* batch_count, int texture_id, int layer) {
    // Allocate new memory for batches array
    *batch_count = *batch_count + 1;
    *batches = static_cast<RenderBatch*>(realloc(*batches, (*batch_count) * sizeof(RenderBatch)));

    // Initialize the new batch
    int new_idx = *batch_count - 1;
    (*batches)[new_idx].texture_id = texture_id;
    (*batches)[new_idx].layer = layer;
    (*batches)[new_idx].vertex_capacity = 4096; // Larger initial capacity
    (*batches)[new_idx].index_capacity = 6144; // Larger initial capacity
    (*batches)[new_idx].vertex_count = 0;
    (*batches)[new_idx].index_count = 0;
    (*batches)[new_idx].vertices = static_cast<SDL_Vertex*>(SDL_aligned_alloc(
        (*batches)[new_idx].vertex_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE));
    (*batches)[new_idx].indices = static_cast<int*>(SDL_aligned_alloc(
        (*batches)[new_idx].index_capacity * sizeof(int), CACHE_LINE_SIZE));
}

// Calculate screen coordinates
void calculate_screen_coordinates(float* world_x, float* world_y, float* width, float* height,
    float* screen_x, float* screen_y, float* screen_w, float* screen_h,
    int count, float visible_rect_x, float visible_rect_y, float zoom) {
    for (int i = 0; i < count; i++) {
        screen_x[i] = (world_x[i] - visible_rect_x) * zoom;
        screen_y[i] = (world_y[i] - visible_rect_y) * zoom;
        screen_w[i] = width[i] * zoom;
        screen_h[i] = height[i] * zoom;
    }
}

// Update which grid cells are visible and load/unload as needed
void update_dynamic_loading(Engine* engine) {
    SDL_FRect visible_rect = get_visible_rect(&engine->camera);

    // Add padding to avoid pop-in at edges (1 cell padding)
    visible_rect.x -= engine->grid_cell_size;
    visible_rect.y -= engine->grid_cell_size;
    visible_rect.w += engine->grid_cell_size * 2;
    visible_rect.h += engine->grid_cell_size * 2;

    // Compute grid cells that should be loaded
    int start_x = (int)(visible_rect.x / engine->grid_cell_size);
    int start_y = (int)(visible_rect.y / engine->grid_cell_size);
    int end_x = (int)ceil((visible_rect.x + visible_rect.w) / engine->grid_cell_size);
    int end_y = (int)ceil((visible_rect.y + visible_rect.h) / engine->grid_cell_size);

    // Clamp to world bounds
    start_x = std::max(0, std::min(start_x, engine->grid_width - 1));
    start_y = std::max(0, std::min(start_y, engine->grid_height - 1));
    end_x = std::max(0, std::min(end_x, engine->grid_width - 1));
    end_y = std::max(0, std::min(end_y, engine->grid_height - 1));

    // First mark all cells as unloaded
    for (int x = 0; x < engine->grid_width; x++) {
        for (int y = 0; y < engine->grid_height; y++) {
            engine->grid_loaded[x][y] = false;
        }
    }

    // Then mark visible cells as loaded
    for (int x = start_x; x <= end_x; x++) {
        for (int y = start_y; y <= end_y; y++) {
            engine->grid_loaded[x][y] = true;
        }
    }

    // Update entity active states based on grid loading
    for (int chunk_idx = 0; chunk_idx < engine->entities.chunk_count; chunk_idx++) {
        EntityChunk* chunk = engine->entities.chunks[chunk_idx];

        for (int local_idx = 0; local_idx < chunk->count; local_idx++) {
            float entity_x = chunk->x[local_idx];
            float entity_y = chunk->y[local_idx];

            int grid_x = (int)(entity_x / engine->grid_cell_size);
            int grid_y = (int)(entity_y / engine->grid_cell_size);

            if (grid_x >= 0 && grid_x < engine->grid_width &&
                grid_y >= 0 && grid_y < engine->grid_height) {
                chunk->active[local_idx] = engine->grid_loaded[grid_x][grid_y];
            }
            else {
                chunk->active[local_idx] = false;
            }
        }
    }
}

// Engine API implementations
Engine* engine_create(int window_width, int window_height, int world_width, int world_height, int cell_size) {
    Engine* engine = static_cast<Engine*>(malloc(sizeof(Engine)));
    if (!engine) return NULL;

    // Use placement new to properly initialize C++ members
    new (&engine->entity_indices_pool) FixedBufferPool(100000 * sizeof(int), 4);
    new (&engine->screen_coords_pool) FixedBufferPool(100000 * sizeof(float) * 4, 4);
    new (&engine->entities) EntityManager();

    // Create window and renderer
    engine->window = SDL_CreateWindow("2D Game Engine - SDL3", window_width, window_height, 0);
    if (!engine->window) {
        engine->entity_indices_pool.~FixedBufferPool();
        engine->screen_coords_pool.~FixedBufferPool();
        engine->entities.~EntityManager();
        free(engine);
        return NULL;
    }

    engine->renderer = SDL_CreateRenderer(engine->window, NULL);
    if (!engine->renderer) {
        SDL_DestroyWindow(engine->window);
        engine->entity_indices_pool.~FixedBufferPool();
        engine->screen_coords_pool.~FixedBufferPool();
        engine->entities.~EntityManager();
        free(engine);
        return NULL;
    }

    // Init world bounds
    engine->world_bounds.x = 0;
    engine->world_bounds.y = 0;
    engine->world_bounds.w = world_width;
    engine->world_bounds.h = world_height;

    // Init spatial grid (replaces quadtree)
    init_spatial_grid(&engine->grid, world_width, world_height, cell_size);

    // Init texture atlas
    init_texture_atlas(&engine->atlas, engine->renderer, 2048, 2048);

    // Init camera
    engine->camera.x = 0;
    engine->camera.y = 0;
    engine->camera.width = window_width;
    engine->camera.height = window_height;
    engine->camera.zoom = 1.0f;

    // Init render batches (for batch rendering)
    engine->batch_count = 8;
    engine->batches = static_cast<RenderBatch*>(malloc(engine->batch_count * sizeof(RenderBatch)));
    for (int i = 0; i < engine->batch_count; i++) {
        engine->batches[i].texture_id = i;
        engine->batches[i].layer = 0;
        engine->batches[i].vertex_capacity = 4096;
        engine->batches[i].index_capacity = 6144;
        engine->batches[i].vertex_count = 0;
        engine->batches[i].index_count = 0;
        engine->batches[i].vertices = static_cast<SDL_Vertex*>(SDL_aligned_alloc(
            engine->batches[i].vertex_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE));
        engine->batches[i].indices = static_cast<int*>(SDL_aligned_alloc(
            engine->batches[i].index_capacity * sizeof(int), CACHE_LINE_SIZE));
    }

    // Init dynamic grid loading
    engine->grid_cell_size = 256.0f;
    engine->grid_width = (int)ceil(world_width / engine->grid_cell_size);
    engine->grid_height = (int)ceil(world_height / engine->grid_cell_size);

    // Allocate grid loaded array
    engine->grid_loaded = static_cast<bool**>(malloc(engine->grid_width * sizeof(bool*)));
    for (int x = 0; x < engine->grid_width; x++) {
        engine->grid_loaded[x] = static_cast<bool*>(malloc(engine->grid_height * sizeof(bool)));
        for (int y = 0; y < engine->grid_height; y++) {
            engine->grid_loaded[x][y] = false;
        }
    }

    // Init timing
    engine->last_frame_time = SDL_GetTicks();
    engine->fps = 0.0f;

    return engine;
}

void engine_destroy(Engine* engine) {
    if (!engine) return;

    // Call destructors for C++ members
    engine->entities.~EntityManager();
    engine->entity_indices_pool.~FixedBufferPool();
    engine->screen_coords_pool.~FixedBufferPool();

    // Free spatial grid
    free_spatial_grid(&engine->grid);

    // Free batches
    for (int i = 0; i < engine->batch_count; i++) {
        SDL_aligned_free(engine->batches[i].vertices);
        SDL_aligned_free(engine->batches[i].indices);
    }
    free(engine->batches);

    // Free atlas
    SDL_DestroyTexture(engine->atlas.texture);
    SDL_aligned_free(engine->atlas.regions);

    // Free grid
    for (int x = 0; x < engine->grid_width; x++) {
        free(engine->grid_loaded[x]);
    }
    free(engine->grid_loaded);

    // Free SDL resources
    SDL_DestroyRenderer(engine->renderer);
    SDL_DestroyWindow(engine->window);

    free(engine);
}

void engine_update(Engine* engine) {
    // Calculate delta time
    Uint64 current_time = SDL_GetTicks();
    float delta_time = (current_time - engine->last_frame_time) / 1000.0f;
    engine->last_frame_time = current_time;

    // Smooth FPS calculation
    engine->fps = 0.95f * engine->fps + 0.05f * (1.0f / delta_time);

    // Update dynamic grid loading based on camera position
    update_dynamic_loading(engine);

    // Clear spatial grid
    clear_spatial_grid(&engine->grid);

    // Rebuild spatial grid with active entities
    for (int chunk_idx = 0; chunk_idx < engine->entities.chunk_count; chunk_idx++) {
        EntityChunk* chunk = engine->entities.chunks[chunk_idx];

        for (int local_idx = 0; local_idx < chunk->count; local_idx++) {
            if (chunk->active[local_idx]) {
                int entity_idx = chunk_idx * ENTITY_CHUNK_SIZE + local_idx;
                spatial_grid_add(&engine->grid, entity_idx, chunk->x[local_idx], chunk->y[local_idx]);
            }
        }
    }
}

void engine_render(Engine* engine) {
    // Clear screen
    SDL_SetRenderDrawColor(engine->renderer, 0, 0, 0, 255);
    SDL_RenderClear(engine->renderer);

    // Get visible area for culling
    SDL_FRect visible_rect = get_visible_rect(&engine->camera);

    // Get reusable buffer for visible entities
    auto indices_handle = engine->entity_indices_pool.getBuffer();
    int* visible_indices = reinterpret_cast<int*>(indices_handle.data());
    int visible_count = 0;

    // Query visible entities from spatial grid
    spatial_grid_query(&engine->grid, visible_rect, visible_indices, &visible_count,
        engine->entities.total_count, &engine->entities);

    // Sort visible entities by texture and z-order
    radix_sort_entities(visible_indices, visible_count, &engine->entities);

    // Clear batches and reuse them
    for (int i = 0; i < engine->batch_count; i++) {
        engine->batches[i].vertex_count = 0;
        engine->batches[i].index_count = 0;
    }

    // Get reusable buffers for transformation
    auto coords_handle = engine->screen_coords_pool.getBuffer();
    float* screen_coords = reinterpret_cast<float*>(coords_handle.data());
    float* screen_x = &screen_coords[0];
    float* screen_y = &screen_coords[visible_count];
    float* screen_w = &screen_coords[visible_count * 2];
    float* screen_h = &screen_coords[visible_count * 3];

    // Extract entity data to contiguous arrays for processing
    std::vector<float> world_x(visible_count);
    std::vector<float> world_y(visible_count);
    std::vector<float> width_f(visible_count);
    std::vector<float> height_f(visible_count);

    // Copy entity data to contiguous arrays for processing
    for (int i = 0; i < visible_count; i++) {
        int entity_idx = visible_indices[i];

        // Get chunk and local index
        int chunk_idx, local_idx;
        engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);

        EntityChunk* chunk = engine->entities.chunks[chunk_idx];
        world_x[i] = chunk->x[local_idx];
        world_y[i] = chunk->y[local_idx];
        width_f[i] = static_cast<float>(chunk->width[local_idx]);
        height_f[i] = static_cast<float>(chunk->height[local_idx]);
    }

    // Calculate screen coordinates
    calculate_screen_coordinates(world_x.data(), world_y.data(), width_f.data(), height_f.data(),
        screen_x, screen_y, screen_w, screen_h,
        visible_count, visible_rect.x, visible_rect.y, engine->camera.zoom);

    // Track last texture/layer to batch together
    int last_texture_id = -1;
    int last_layer = -1;
    int current_batch_idx = -1;

    // Add visible entities to batches
    for (int i = 0; i < visible_count; i++) {
        int entity_idx = visible_indices[i];

        // Get chunk and local index
        int chunk_idx, local_idx;
        engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);

        EntityChunk* chunk = engine->entities.chunks[chunk_idx];
        int texture_id = chunk->texture_id[local_idx];
        int layer = chunk->layer[local_idx];

        // If texture or layer changed, get/create appropriate batch
        if (texture_id != last_texture_id || layer != last_layer) {
            // Find existing batch for this texture/layer
            current_batch_idx = find_batch_index(engine->batches, engine->batch_count, texture_id, layer);

            // If no batch exists, create one
            if (current_batch_idx == -1) {
                create_batch(&engine->batches, &engine->batch_count, texture_id, layer);
                current_batch_idx = engine->batch_count - 1;
            }

            last_texture_id = texture_id;
            last_layer = layer;
        }

        // Use pre-calculated screen coordinates
        float screen_pos_x = screen_x[i];
        float screen_pos_y = screen_y[i];
        float screen_pos_w = screen_w[i];
        float screen_pos_h = screen_h[i];

        // Get texture region from atlas
        SDL_FRect tex_region = engine->atlas.regions[texture_id];

        // Add to appropriate batch
        SDL_FColor color = { 1.0f, 1.0f, 1.0f, 1.0f };
        add_to_batch(&engine->batches[current_batch_idx],
            screen_pos_x, screen_pos_y,
            screen_pos_w, screen_pos_h,
            tex_region, color);
    }

    // Render batches
    for (int i = 0; i < engine->batch_count; i++) {
        if (engine->batches[i].vertex_count > 0) {
            // Single draw call per batch!
            SDL_RenderGeometry(engine->renderer, engine->atlas.texture,
                engine->batches[i].vertices, engine->batches[i].vertex_count,
                engine->batches[i].indices, engine->batches[i].index_count);
        }
    }

    // Present renderer
    SDL_RenderPresent(engine->renderer);
}

int engine_add_entity(Engine* engine, float x, float y, int width, int height,
    int texture_id, int layer) {
    // Add entity to manager
    int entity_idx = engine->entities.addEntity();

    // Get chunk and local index
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);

    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    // Set entity properties
    chunk->x[local_idx] = x;
    chunk->y[local_idx] = y;
    chunk->width[local_idx] = width;
    chunk->height[local_idx] = height;

    // Precompute right and bottom for faster culling
    chunk->right[local_idx] = x + width;
    chunk->bottom[local_idx] = y + height;

    chunk->texture_id[local_idx] = texture_id;
    chunk->layer[local_idx] = layer;

    // Determine if entity should be active based on grid loading
    int grid_x = (int)(x / engine->grid_cell_size);
    int grid_y = (int)(y / engine->grid_cell_size);

    if (grid_x >= 0 && grid_x < engine->grid_width &&
        grid_y >= 0 && grid_y < engine->grid_height) {
        chunk->active[local_idx] = engine->grid_loaded[grid_x][grid_y];
    }
    else {
        chunk->active[local_idx] = false;
    }

    // Add to spatial grid
    spatial_grid_add(&engine->grid, entity_idx, x, y);

    return entity_idx;
}

int engine_add_texture(Engine* engine, SDL_Surface* surface, int x, int y) {
    int texture_id = engine->atlas.region_count;

    // Ensure capacity
    if (texture_id >= engine->atlas.region_capacity) {
        int new_capacity = engine->atlas.region_capacity * 2;
        SDL_FRect* new_regions = static_cast<SDL_FRect*>(SDL_aligned_alloc(
            new_capacity * sizeof(SDL_FRect), CACHE_LINE_SIZE));

        // Copy existing regions
        memcpy(new_regions, engine->atlas.regions,
            engine->atlas.region_count * sizeof(SDL_FRect));

        SDL_aligned_free(engine->atlas.regions);
        engine->atlas.regions = new_regions;
        engine->atlas.region_capacity = new_capacity;
    }

    // Calculate normalized UV coordinates
    float atlas_width, atlas_height;
    SDL_GetTextureSize(engine->atlas.texture, &atlas_width, &atlas_height);

    SDL_FRect region = {
        (float)x / atlas_width,
        (float)y / atlas_height,
        (float)surface->w / atlas_width,
        (float)surface->h / atlas_height
    };

    engine->atlas.regions[texture_id] = region;
    engine->atlas.region_count++;

    // Copy surface to atlas texture
    SDL_Texture* temp = SDL_CreateTextureFromSurface(engine->renderer, surface);

    // Set render target to atlas
    SDL_Texture* old_target = SDL_GetRenderTarget(engine->renderer);
    SDL_SetRenderTarget(engine->renderer, engine->atlas.texture);

    // Copy texture to atlas
    SDL_FRect dest = { (float)x, (float)y, (float)surface->w, (float)surface->h };
    SDL_RenderTexture(engine->renderer, temp, NULL, &dest);

    // Reset render target
    SDL_SetRenderTarget(engine->renderer, old_target);

    // Clean up
    SDL_DestroyTexture(temp);

    return texture_id;
}

void engine_set_camera_position(Engine* engine, float x, float y) {
    engine->camera.x = x;
    engine->camera.y = y;
}

void engine_set_camera_zoom(Engine* engine, float zoom) {
    engine->camera.zoom = zoom;
}

void engine_set_entity_position(Engine* engine, int entity_idx, float x, float y) {
    if (!engine->entities.isValidEntity(entity_idx)) {
        return;
    }

    // Get chunk and local index
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);

    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    chunk->x[local_idx] = x;
    chunk->y[local_idx] = y;

    // Update precomputed values
    chunk->right[local_idx] = x + chunk->width[local_idx];
    chunk->bottom[local_idx] = y + chunk->height[local_idx];
}

void engine_set_entity_active(Engine* engine, int entity_idx, bool active) {
    if (!engine->entities.isValidEntity(entity_idx)) {
        return;
    }

    // Get chunk and local index
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);

    EntityChunk* chunk = engine->entities.chunks[chunk_idx];
    chunk->active[local_idx] = active;
}

SDL_FRect engine_get_visible_rect(Engine* engine) {
    return get_visible_rect(&engine->camera);
}

int engine_get_visible_entities_count(Engine* engine) {
    SDL_FRect visible_rect = get_visible_rect(&engine->camera);

    auto handle = engine->entity_indices_pool.getBuffer();
    int* visible_indices = reinterpret_cast<int*>(handle.data());
    int visible_count = 0;

    spatial_grid_query(&engine->grid, visible_rect, visible_indices, &visible_count,
        engine->entities.total_count, &engine->entities);

    return visible_count;
}