#include "ATMEngine.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <new>

#ifdef EMSCRIPTEN
#include <emscripten.h>
#endif


// In the implementation file (ATMEngine.cpp)
EntityChunk::EntityChunk(int chunk_type_id, int chunk_capacity, size_t extra_data_size) {
    type_id = chunk_type_id;
    capacity = chunk_capacity;
    count = 0;
    next_chunk_of_type = -1; // Initialize with no next chunk

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
    visible = static_cast<bool*>(SDL_aligned_alloc(bool_size, CACHE_LINE_SIZE)); // New visibility field
    grid_cell = static_cast<int**>(SDL_aligned_alloc(ptr_size, CACHE_LINE_SIZE));

    // Hierarchy arrays
    parent_id = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    first_child_id = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    next_sibling_id = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    local_x = static_cast<float*>(SDL_aligned_alloc(float_size, CACHE_LINE_SIZE));
    local_y = static_cast<float*>(SDL_aligned_alloc(float_size, CACHE_LINE_SIZE));

    // Allocate type-specific data if needed
    if (extra_data_size > 0) {
        type_data = SDL_aligned_alloc(capacity * extra_data_size, CACHE_LINE_SIZE);
        memset(type_data, 0, capacity * extra_data_size);
    }
    else {
        type_data = nullptr;
    }

    // Initialize all memory to zero
    memset(x, 0, float_size);
    memset(y, 0, float_size);
    memset(right, 0, float_size);
    memset(bottom, 0, float_size);
    memset(width, 0, int_size);
    memset(height, 0, int_size);
    memset(texture_id, 0, int_size);
    memset(layer, 0, int_size);
    memset(active, 1, bool_size); // Default to active
    memset(visible, 0, bool_size); // Default to not visible
    memset(grid_cell, 0, ptr_size);

    // Initialize hierarchy arrays with -1 (no parent/child/sibling)
    for (int i = 0; i < capacity; i++) {
        parent_id[i] = -1;
        first_child_id[i] = -1;
        next_sibling_id[i] = -1;
    }

    memset(local_x, 0, float_size);
    memset(local_y, 0, float_size);
}

// In the implementation file (ATMEngine.cpp)
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
    SDL_aligned_free(visible); // Free the visibility array
    SDL_aligned_free(grid_cell);

    // Free hierarchy arrays
    SDL_aligned_free(parent_id);
    SDL_aligned_free(first_child_id);
    SDL_aligned_free(next_sibling_id);
    SDL_aligned_free(local_x);
    SDL_aligned_free(local_y);

    // Free type-specific data if allocated
    if (type_data) {
        SDL_aligned_free(type_data);
    }
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
                chunks[chunk_count] = new EntityChunk(-1, ENTITY_CHUNK_SIZE);
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

                // Check if entity is valid
                if (!entities->isValidEntity(entity_idx)) {
                    continue;
                }

                // Get chunk and local index
                int chunk_idx, local_idx;
                entities->getChunkIndices(entity_idx, &chunk_idx, &local_idx);

                EntityChunk* chunk = entities->chunks[chunk_idx];

                // Check visibility instead of active status
                if (!chunk->visible[local_idx]) {
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
        screen_x[i] = roundf((world_x[i] - visible_rect_x) * zoom);
        screen_y[i] = roundf((world_y[i] - visible_rect_y) * zoom);
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

    // Update entity visible states based on grid loading (not active states)
    for (int chunk_idx = 0; chunk_idx < engine->entities.chunk_count; chunk_idx++) {
        EntityChunk* chunk = engine->entities.chunks[chunk_idx];

        for (int local_idx = 0; local_idx < chunk->count; local_idx++) {
            float entity_x = chunk->x[local_idx];
            float entity_y = chunk->y[local_idx];

            int grid_x = (int)(entity_x / engine->grid_cell_size);
            int grid_y = (int)(entity_y / engine->grid_cell_size);

            if (grid_x >= 0 && grid_x < engine->grid_width &&
                grid_y >= 0 && grid_y < engine->grid_height) {
                chunk->visible[local_idx] = engine->grid_loaded[grid_x][grid_y];
            }
            else {
                chunk->visible[local_idx] = false;
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
    new (&engine->type_id_to_index) std::unordered_map<int, int>();


    // Initialize entity type system with capacity for 200K+ types
    engine->entity_type_count = 0;
    engine->entity_type_capacity = 262144;  // 256K initial capacity
    engine->entity_types = static_cast<EntityTypeConfig*>(
        malloc(engine->entity_type_capacity * sizeof(EntityTypeConfig))
        );

    // Initialize performance tracking
    engine->active_entity_count = 0;
    engine->update_time = 0.0f;
    engine->render_time = 0.0f;

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


    // Free entity type system
    free(engine->entity_types);
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

// Modified engine_update to include entity type updates
void engine_update(Engine* engine) {
    // Calculate delta time
    Uint64 current_time = SDL_GetTicks();
    float delta_time = (current_time - engine->last_frame_time) / 1000.0f;
    engine->last_frame_time = current_time;

    // Smooth FPS calculation
    engine->fps = 0.95f * engine->fps + 0.05f * (1.0f / delta_time);

    // Update dynamic loading based on camera position (updates visibility)
    update_dynamic_loading(engine);

    // Update transforms for hierarchy
    engine_update_entity_transforms(engine);

    // Update all entity types (now regardless of visibility)
    engine_update_entity_types(engine, delta_time);

    // Clear spatial grid
    clear_spatial_grid(&engine->grid);

    // Rebuild spatial grid with visible entities (not just active)
    for (int chunk_idx = 0; chunk_idx < engine->entities.chunk_count; chunk_idx++) {
        EntityChunk* chunk = engine->entities.chunks[chunk_idx];

        for (int local_idx = 0; local_idx < chunk->count; local_idx++) {
            if (chunk->visible[local_idx]) {  // Changed from active to visible
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

    // Initialize hierarchy properties
    chunk->parent_id[local_idx] = -1;
    chunk->first_child_id[local_idx] = -1;
    chunk->next_sibling_id[local_idx] = -1;
    chunk->local_x[local_idx] = x;
    chunk->local_y[local_idx] = y;

    // Set active by default
    chunk->active[local_idx] = true;

    // Determine if entity should be visible based on grid loading
    int grid_x = (int)(x / engine->grid_cell_size);
    int grid_y = (int)(y / engine->grid_cell_size);

    if (grid_x >= 0 && grid_x < engine->grid_width &&
        grid_y >= 0 && grid_y < engine->grid_height) {
        chunk->visible[local_idx] = engine->grid_loaded[grid_x][grid_y];
    }
    else {
        chunk->visible[local_idx] = false;
    }

    // Add to spatial grid if visible
    if (chunk->visible[local_idx]) {
        spatial_grid_add(&engine->grid, entity_idx, x, y);
    }

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
// Set entity's parent
void engine_set_parent(Engine* engine, int entity_id, int parent_id) {
    if (!engine->entities.isValidEntity(entity_id)) {
        return;
    }

    // Don't allow circular references
    int p = parent_id;
    while (p != -1) {
        if (p == entity_id) {
            return; // Would create a cycle
        }
        p = engine_get_parent(engine, p);
    }

    // Get chunk and local indices for entity
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    // Store current world position as local position if changing parent
    int old_parent = chunk->parent_id[local_idx];
    if (old_parent != parent_id) {
        // Store current world position in local coordinates
        chunk->local_x[local_idx] = chunk->x[local_idx];
        chunk->local_y[local_idx] = chunk->y[local_idx];

        // If had previous parent, detach from old parent's child list
        if (old_parent != -1) {
            engine_remove_parent(engine, entity_id);
        }

        // Set new parent
        chunk->parent_id[local_idx] = parent_id;

        // If has new parent, add to new parent's child list
        if (parent_id != -1 && engine->entities.isValidEntity(parent_id)) {
            int parent_chunk_idx, parent_local_idx;
            engine->entities.getChunkIndices(parent_id, &parent_chunk_idx, &parent_local_idx);
            EntityChunk* parent_chunk = engine->entities.chunks[parent_chunk_idx];

            // Get parent's first child
            int first_child = parent_chunk->first_child_id[parent_local_idx];

            // Make this entity the first child and link to old first child
            parent_chunk->first_child_id[parent_local_idx] = entity_id;
            chunk->next_sibling_id[local_idx] = first_child;

            // Update entity's local position relative to parent
            if (first_child != -1) {
                chunk->local_x[local_idx] = chunk->x[local_idx] - parent_chunk->x[parent_local_idx];
                chunk->local_y[local_idx] = chunk->y[local_idx] - parent_chunk->y[parent_local_idx];
            }
        }

        // Update world transforms for this entity and its children
        engine_update_entity_transforms(engine);
    }
}

// Remove entity from its parent
void engine_remove_parent(Engine* engine, int entity_id) {
    if (!engine->entities.isValidEntity(entity_id)) {
        return;
    }

    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    int parent_id = chunk->parent_id[local_idx];
    if (parent_id == -1) {
        return; // No parent to remove
    }

    // Find parent
    int parent_chunk_idx, parent_local_idx;
    engine->entities.getChunkIndices(parent_id, &parent_chunk_idx, &parent_local_idx);
    EntityChunk* parent_chunk = engine->entities.chunks[parent_chunk_idx];

    // Find and remove this entity from parent's child list
    int* child_id_ptr = &parent_chunk->first_child_id[parent_local_idx];
    while (*child_id_ptr != -1) {
        if (*child_id_ptr == entity_id) {
            // Get the sibling that will replace this entity in the list
            int next_sibling = chunk->next_sibling_id[local_idx];
            *child_id_ptr = next_sibling;
            break;
        }

        // Move to next sibling
        int sibling_chunk_idx, sibling_local_idx;
        engine->entities.getChunkIndices(*child_id_ptr, &sibling_chunk_idx, &sibling_local_idx);
        EntityChunk* sibling_chunk = engine->entities.chunks[sibling_chunk_idx];
        child_id_ptr = &sibling_chunk->next_sibling_id[sibling_local_idx];
    }

    // Reset parent relationship
    chunk->parent_id[local_idx] = -1;
    chunk->next_sibling_id[local_idx] = -1;

    // Set world position to local position (becoming a root entity)
    // No need to modify local coordinates as they become the world coordinates
}

// Get entity's parent
int engine_get_parent(Engine* engine, int entity_id) {
    if (!engine->entities.isValidEntity(entity_id)) {
        return -1;
    }

    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    return chunk->parent_id[local_idx];
}

// Get entity's first child
int engine_get_first_child(Engine* engine, int entity_id) {
    if (!engine->entities.isValidEntity(entity_id)) {
        return -1;
    }

    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    return chunk->first_child_id[local_idx];
}

// Get entity's next sibling
int engine_get_next_sibling(Engine* engine, int entity_id) {
    if (!engine->entities.isValidEntity(entity_id)) {
        return -1;
    }

    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    return chunk->next_sibling_id[local_idx];
}

// Set entity's local position (relative to parent)
void engine_set_entity_local_position(Engine* engine, int entity_id, float x, float y) {
    if (!engine->entities.isValidEntity(entity_id)) {
        return;
    }

    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    chunk->local_x[local_idx] = x;
    chunk->local_y[local_idx] = y;

    // Update world transforms for this entity and its children
    engine_update_entity_transforms(engine);
}

// Update entity transforms (recursive in batches for better cache coherence)
void engine_update_entity_transforms(Engine* engine) {

    // Get temporary buffer for entity queue from the engine's buffer pool
    // This avoids expensive allocation on every call
    const int MAX_QUEUE_SIZE = 100000;
    auto queue_handle = engine->entity_indices_pool.getBuffer();
    int* queue = reinterpret_cast<int*>(queue_handle.data());
    int queue_size = engine->entity_indices_pool.getBufferSize() / sizeof(int);

    if (!queue || queue_size < 1000) {
        // Fallback if buffer is too small
        static std::vector<int> fallback_queue(MAX_QUEUE_SIZE);
        queue = fallback_queue.data();
        queue_size = MAX_QUEUE_SIZE;
    }

    int queue_head = 0;
    int queue_tail = 0;

    // First pass: Process root entities and queue those with children
    // Use chunk-based processing for better cache locality
    for (int chunk_idx = 0; chunk_idx < engine->entities.chunk_count; chunk_idx++) {
        EntityChunk* chunk = engine->entities.chunks[chunk_idx];
        if (!chunk || chunk->count == 0) continue;

        // Find and update root entities
        for (int local_idx = 0; local_idx < chunk->count; local_idx++) {
            if (!chunk->active[local_idx]) continue;

            if (chunk->parent_id[local_idx] == -1) {
                // This is a root entity
                chunk->x[local_idx] = chunk->local_x[local_idx];
                chunk->y[local_idx] = chunk->local_y[local_idx];

                // Update precomputed bounds
                chunk->right[local_idx] = chunk->x[local_idx] + chunk->width[local_idx];
                chunk->bottom[local_idx] = chunk->y[local_idx] + chunk->height[local_idx];

                // If this root has children, queue it
                if (chunk->first_child_id[local_idx] != -1) {
                    int entity_id = chunk_idx * ENTITY_CHUNK_SIZE + local_idx;
                    queue[queue_tail] = entity_id;
                    queue_tail = (queue_tail + 1) % queue_size;

                    // Safety check for queue overflow
                    if (queue_tail == queue_head) {
                        // Process the queue now to avoid overflow
                        break;
                    }
                }
            }
        }
    }

    // Second pass: Process hierarchy in breadth-first order
    // This avoids recursion and improves cache locality
    while (queue_head != queue_tail) {
        int parent_id = queue[queue_head];
        queue_head = (queue_head + 1) % queue_size;

        int parent_chunk_idx, parent_local_idx;
        engine->entities.getChunkIndices(parent_id, &parent_chunk_idx, &parent_local_idx);

        EntityChunk* parent_chunk = engine->entities.chunks[parent_chunk_idx];
        if (!parent_chunk || !parent_chunk->active[parent_local_idx]) continue;

        // Cache parent's world position
        float parent_x = parent_chunk->x[parent_local_idx];
        float parent_y = parent_chunk->y[parent_local_idx];

        // Process all children of this parent
        int child_id = parent_chunk->first_child_id[parent_local_idx];

        // Use a batch approach for siblings to improve cache locality
        int sibling_batch[16]; // Small batch size for better cache performance
        int batch_size = 0;

        // Collect siblings in batches
        while (child_id != -1 && batch_size < 16) {
            sibling_batch[batch_size++] = child_id;

            int child_chunk_idx, child_local_idx;
            engine->entities.getChunkIndices(child_id, &child_chunk_idx, &child_local_idx);
            EntityChunk* child_chunk = engine->entities.chunks[child_chunk_idx];

            // Move to next sibling
            child_id = child_chunk->next_sibling_id[child_local_idx];
        }

        // Process the batch of siblings
        for (int i = 0; i < batch_size; i++) {
            int current_child_id = sibling_batch[i];
            int child_chunk_idx, child_local_idx;
            engine->entities.getChunkIndices(current_child_id, &child_chunk_idx, &child_local_idx);
            EntityChunk* child_chunk = engine->entities.chunks[child_chunk_idx];

            if (!child_chunk || !child_chunk->active[child_local_idx]) continue;

            // Update child's world position based on parent
            child_chunk->x[child_local_idx] = parent_x + child_chunk->local_x[child_local_idx];
            child_chunk->y[child_local_idx] = parent_y + child_chunk->local_y[child_local_idx];

            // Update precomputed bounds
            child_chunk->right[child_local_idx] = child_chunk->x[child_local_idx] + child_chunk->width[child_local_idx];
            child_chunk->bottom[child_local_idx] = child_chunk->y[child_local_idx] + child_chunk->height[child_local_idx];

            // If this child has children, enqueue it
            if (child_chunk->first_child_id[child_local_idx] != -1) {
                queue[queue_tail] = current_child_id;
                queue_tail = (queue_tail + 1) % queue_size;

                // Safety check for queue overflow
                if (queue_tail == queue_head) {
                    // Queue is full, we'll process what we have so far
                    break;
                }
            }
        }
    }

    // Buffer is automatically returned to the pool when queue_handle goes out of scope
}

void engine_set_entity_position(Engine* engine, int entity_idx, float x, float y) {
    if (!engine->entities.isValidEntity(entity_idx)) {
        return;
    }

    // Get chunk and local index
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    // Check if entity has a parent
    if (chunk->parent_id[local_idx] != -1) {
        // For entities with parents, we're setting world position, so we need
        // to calculate local position based on parent's world position
        int parent_id = chunk->parent_id[local_idx];
        int parent_chunk_idx, parent_local_idx;
        engine->entities.getChunkIndices(parent_id, &parent_chunk_idx, &parent_local_idx);
        EntityChunk* parent_chunk = engine->entities.chunks[parent_chunk_idx];

        // Set local position relative to parent
        chunk->local_x[local_idx] = x - parent_chunk->x[parent_local_idx];
        chunk->local_y[local_idx] = y - parent_chunk->y[parent_local_idx];
    }
    else {
        // For root entities, local position is the same as world position
        chunk->local_x[local_idx] = x;
        chunk->local_y[local_idx] = y;
    }

    chunk->x[local_idx] = x;
    chunk->y[local_idx] = y;

    // Update precomputed values
    chunk->right[local_idx] = x + chunk->width[local_idx];
    chunk->bottom[local_idx] = y + chunk->height[local_idx];

    // Update all child entities recursively
    if (chunk->first_child_id[local_idx] != -1) {
        engine_update_entity_transforms(engine);
    }
}

int engine_add_child_entity(Engine* engine, int parent_id, float local_x, float local_y,
    int width, int height, int texture_id, int layer) {
    if (!engine->entities.isValidEntity(parent_id)) {
        return -1;
    }

    // Get parent's world position
    int parent_chunk_idx, parent_local_idx;
    engine->entities.getChunkIndices(parent_id, &parent_chunk_idx, &parent_local_idx);
    EntityChunk* parent_chunk = engine->entities.chunks[parent_chunk_idx];

    float parent_x = parent_chunk->x[parent_local_idx];
    float parent_y = parent_chunk->y[parent_local_idx];

    // Create entity at parent's position + local offset
    int entity_id = engine_add_entity(engine, parent_x + local_x, parent_y + local_y,
        width, height, texture_id, layer);

    // Set parent relationship
    engine_set_parent(engine, entity_id, parent_id);

    // Make sure local position is set correctly
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];
    chunk->local_x[local_idx] = local_x;
    chunk->local_y[local_idx] = local_y;

    return entity_id;
}

// Register an entity type with the engine
void engine_register_entity_type(Engine* engine, int type_id, EntityTypeUpdateFunc update_func, size_t extra_data_size) {
    // Ensure capacity for new type
    if (engine->entity_type_count >= engine->entity_type_capacity) {
        int new_capacity = engine->entity_type_capacity * 2;
        engine->entity_types = static_cast<EntityTypeConfig*>(realloc(
            engine->entity_types,
            new_capacity * sizeof(EntityTypeConfig)
        ));
        engine->entity_type_capacity = new_capacity;
    }

    // Add new type
    int type_index = engine->entity_type_count++;
    engine->type_id_to_index[type_id] = type_index;  // Store mapping for O(1) lookup

    EntityTypeConfig* config = &engine->entity_types[type_index];
    config->type_id = type_id;
    config->update_func = update_func;
    config->extra_data_size = extra_data_size;
    config->instance_count = 0;
    config->last_update_time = 0.0f;
    config->first_chunk_idx = -1;  // No chunks of this type yet
}

// Find or create a chunk for a specific entity type
int find_or_create_chunk_for_type(Engine* engine, int type_id) {
    // Get the type index from the mapping
    auto it = engine->type_id_to_index.find(type_id);
    if (it == engine->type_id_to_index.end()) {
        // Type not registered
        return -1;
    }

    int type_index = it->second;
    EntityTypeConfig* config = &engine->entity_types[type_index];

    // First, look for an existing chunk of this type with available space
    int chunk_idx = config->first_chunk_idx;
    int prev_chunk_idx = -1;

    while (chunk_idx != -1) {
        EntityChunk* chunk = engine->entities.chunks[chunk_idx];
        if (chunk->count < chunk->capacity) {
            return chunk_idx;
        }
        prev_chunk_idx = chunk_idx;
        chunk_idx = chunk->next_chunk_of_type;
    }

    // No chunk found with space, create a new one
    if (engine->entities.chunk_count >= engine->entities.chunks_capacity) {
        engine->entities.chunks_capacity *= 2;
        engine->entities.chunks = static_cast<EntityChunk**>(realloc(
            engine->entities.chunks,
            engine->entities.chunks_capacity * sizeof(EntityChunk*)
        ));
    }

    // Create new chunk
    int new_chunk_idx = engine->entities.chunk_count++;
    engine->entities.chunks[new_chunk_idx] = new EntityChunk(
        type_id,
        ENTITY_CHUNK_SIZE,
        config->extra_data_size
    );

    // Add to the linked list of chunks for this type
    if (prev_chunk_idx != -1) {
        engine->entities.chunks[prev_chunk_idx]->next_chunk_of_type = new_chunk_idx;
    }
    else {
        config->first_chunk_idx = new_chunk_idx;
    }

    config->instance_count += ENTITY_CHUNK_SIZE; // Account for max possible entities

    return new_chunk_idx;
}

// Add entity with a specific type
int engine_add_entity_with_type(Engine* engine, int type_id, float x, float y,
    int width, int height, int texture_id, int layer) {
    // Find or create appropriate chunk
    int chunk_idx = find_or_create_chunk_for_type(engine, type_id);
    if (chunk_idx == -1) return -1; // Type not registered

    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    // Local index in the chunk
    int local_idx = chunk->count++;
    int entity_idx = chunk_idx * ENTITY_CHUNK_SIZE + local_idx;

    // Ensure total_count is updated
    if (entity_idx >= engine->entities.total_count) {
        engine->entities.total_count = entity_idx + 1;
    }

    // Initialize entity data
    chunk->x[local_idx] = x;
    chunk->y[local_idx] = y;
    chunk->width[local_idx] = width;
    chunk->height[local_idx] = height;
    chunk->texture_id[local_idx] = texture_id;
    chunk->layer[local_idx] = layer;

    // Precompute right and bottom for faster culling
    chunk->right[local_idx] = x + width;
    chunk->bottom[local_idx] = y + height;

    // Initialize hierarchy properties to default values
    chunk->parent_id[local_idx] = -1;
    chunk->first_child_id[local_idx] = -1;
    chunk->next_sibling_id[local_idx] = -1;
    chunk->local_x[local_idx] = x;
    chunk->local_y[local_idx] = y;

    // Set active by default
    chunk->active[local_idx] = true;

    // Set visible status based on grid loading
    int grid_x = (int)(x / engine->grid_cell_size);
    int grid_y = (int)(y / engine->grid_cell_size);

    if (grid_x >= 0 && grid_x < engine->grid_width &&
        grid_y >= 0 && grid_y < engine->grid_height) {
        chunk->visible[local_idx] = engine->grid_loaded[grid_x][grid_y];
    }
    else {
        chunk->visible[local_idx] = false;
    }

    // Add to spatial grid if visible
    if (chunk->visible[local_idx]) {
        spatial_grid_add(&engine->grid, entity_idx, x, y);
    }

    return entity_idx;
}


// Get type-specific data for an entity
void* engine_get_entity_type_data(Engine* engine, int entity_idx) {
    if (!engine->entities.isValidEntity(entity_idx)) {
        return nullptr;
    }

    // Get chunk and local index
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    // If chunk has no type-specific data, return nullptr
    if (!chunk->type_data) {
        return nullptr;
    }

    // Find the type config
    size_t extra_data_size = 0;
    for (int i = 0; i < engine->entity_type_count; i++) {
        if (engine->entity_types[i].type_id == chunk->type_id) {
            extra_data_size = engine->entity_types[i].extra_data_size;
            break;
        }
    }

    if (extra_data_size == 0) {
        return nullptr;
    }

    // Calculate the pointer to this entity's extra data
    uint8_t* typed_data = static_cast<uint8_t*>(chunk->type_data);
    return typed_data + (local_idx * extra_data_size);
}

void engine_update_entity_types(Engine* engine, float delta_time) {

    Uint64 start_time = SDL_GetTicks();
    int total_active_entities = 0;

    // Pre-allocate buffers once instead of per-chunk
    // Using 256 as a reasonable upper limit for active entities in a chunk
    float original_x_stack[ENTITY_CHUNK_SIZE];
    float original_y_stack[ENTITY_CHUNK_SIZE];
    int active_to_chunk_map_stack[ENTITY_CHUNK_SIZE];

    float* original_x_heap = nullptr;
    float* original_y_heap = nullptr;
    int* active_to_chunk_map_heap = nullptr;
    int original_x_heap_size = 0;

    // Using the type-based chunk linking for much faster updates
    for (int type_idx = 0; type_idx < engine->entity_type_count; type_idx++) {
        EntityTypeConfig* config = &engine->entity_types[type_idx];

        // Skip if no update function or no chunks
        if (!config->update_func || config->first_chunk_idx == -1) {
            continue;
        }

        Uint64 type_start_time = SDL_GetTicks();
        int type_active_count = 0;

        // Process all chunks of this type without searching
        int chunk_idx = config->first_chunk_idx;
        while (chunk_idx != -1) {
            EntityChunk* chunk = engine->entities.chunks[chunk_idx];

            // Count active entities in this chunk (based on active flag, not visibility)
            int active_count = 0;
            for (int j = 0; j < chunk->count; j++) {
                active_count += chunk->active[j] ? 1 : 0;
            }

            // Only update if there are active entities
            if (active_count > 0) {
                // Select stack or heap based arrays
                float* original_x;
                float* original_y;
                int* active_to_chunk_map;

                if (active_count <= ENTITY_CHUNK_SIZE) {
                    original_x = original_x_stack;
                    original_y = original_y_stack;
                    active_to_chunk_map = active_to_chunk_map_stack;
                }
                else {
                    // Allocate or resize heap arrays as needed
                    if (!original_x_heap || original_x_heap_size < active_count) {
                        delete[] original_x_heap;
                        delete[] original_y_heap;
                        delete[] active_to_chunk_map_heap;

                        original_x_heap = new float[active_count];
                        original_y_heap = new float[active_count];
                        active_to_chunk_map_heap = new int[active_count];
                        original_x_heap_size = active_count;
                    }

                    original_x = original_x_heap;
                    original_y = original_y_heap;
                    active_to_chunk_map = active_to_chunk_map_heap;
                }

                // Collect original positions of active entities only - optimized loop
                int active_idx = 0;
                for (int j = 0; j < chunk->count; j++) {
                    if (chunk->active[j]) {
                        original_x[active_idx] = chunk->x[j];
                        original_y[active_idx] = chunk->y[j];
                        active_to_chunk_map[active_idx] = j;
                        active_idx++;
                    }
                }

                // Call the update function
                config->update_func(chunk, chunk->count, delta_time);

                // Efficiently update only positions that changed
                for (int j = 0; j < active_count; j++) {
                    int entity_idx = active_to_chunk_map[j];
                    float new_x = chunk->x[entity_idx];
                    float new_y = chunk->y[entity_idx];

                    // Skip if position didn't change
                    if (new_x == original_x[j] && new_y == original_y[j]) {
                        continue;
                    }

                    // Update dependent values
                    chunk->right[entity_idx] = new_x + static_cast<float>(chunk->width[entity_idx]);
                    chunk->bottom[entity_idx] = new_y + static_cast<float>(chunk->height[entity_idx]);

                    // Update hierarchy relationships if needed
                    if (chunk->parent_id[entity_idx] != -1) {
                        int parent_id = chunk->parent_id[entity_idx];
                        int parent_chunk_idx, parent_local_idx;
                        engine->entities.getChunkIndices(parent_id, &parent_chunk_idx, &parent_local_idx);
                        EntityChunk* parent_chunk = engine->entities.chunks[parent_chunk_idx];

                        chunk->local_x[entity_idx] = new_x - parent_chunk->x[parent_local_idx];
                        chunk->local_y[entity_idx] = new_y - parent_chunk->y[parent_local_idx];
                    }
                    else {
                        chunk->local_x[entity_idx] = new_x;
                        chunk->local_y[entity_idx] = new_y;
                    }
                }

                type_active_count += active_count;
            }

            // Move to next chunk of the same type
            chunk_idx = chunk->next_chunk_of_type;
        }

        // Update stats
        config->last_update_time = (SDL_GetTicks() - type_start_time);
        total_active_entities += type_active_count;
    }

    // Clean up any heap-allocated arrays
    delete[] original_x_heap;
    delete[] original_y_heap;
    delete[] active_to_chunk_map_heap;

    // Update engine stats
    engine->active_entity_count = total_active_entities;
    engine->update_time = (SDL_GetTicks() - start_time);
}

void engine_set_entity_visible(Engine* engine, int entity_idx, bool visible) {
    if (!engine->entities.isValidEntity(entity_idx)) {
        return;
    }

    // Get chunk and local index
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);

    EntityChunk* chunk = engine->entities.chunks[chunk_idx];
    chunk->visible[local_idx] = visible;
}