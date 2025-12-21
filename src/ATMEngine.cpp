#include "ATMEngine.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#include "ATMProfiler.h"
#ifdef EMSCRIPTEN
#include <emscripten.h>
#endif
// In the implementation file (ATMEngine.cpp)
EntityChunk::EntityChunk(int chunk_type_id, int chunk_capacity, size_t extra_data_size) {
    PROFILE_SCOPE("EntityChunk::Constructor");

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
    x = static_cast<float*>(SDL_aligned_alloc(CACHE_LINE_SIZE, float_size));
    y = static_cast<float*>(SDL_aligned_alloc(CACHE_LINE_SIZE, float_size));
    right = static_cast<float*>(SDL_aligned_alloc(CACHE_LINE_SIZE, float_size));
    bottom = static_cast<float*>(SDL_aligned_alloc(CACHE_LINE_SIZE, float_size));
    width = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE, int_size));
    height = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE, int_size));
    texture_id = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE, int_size));
    layer = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE, int_size));
    z_index = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE, int_size)); // New z_index field
    visible = static_cast<bool*>(SDL_aligned_alloc(CACHE_LINE_SIZE, bool_size));
    grid_cell = static_cast<int**>(SDL_aligned_alloc(CACHE_LINE_SIZE, ptr_size));

    // Hierarchy arrays
    parent_id = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE, int_size));
    first_child_id = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE, int_size));
    next_sibling_id = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE, int_size));
    local_x = static_cast<float*>(SDL_aligned_alloc(CACHE_LINE_SIZE, float_size));
    local_y = static_cast<float*>(SDL_aligned_alloc(CACHE_LINE_SIZE, float_size));

    // Allocate type-specific data if needed
    if (extra_data_size > 0) {
        type_data = SDL_aligned_alloc(CACHE_LINE_SIZE, capacity * extra_data_size);
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
    memset(z_index, 0, int_size); // Initialize z_index to 0
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
    PROFILE_SCOPE("EntityChunk::Destructor");

    SDL_aligned_free(x);
    SDL_aligned_free(y);
    SDL_aligned_free(right);
    SDL_aligned_free(bottom);
    SDL_aligned_free(width);
    SDL_aligned_free(height);
    SDL_aligned_free(texture_id);
    SDL_aligned_free(layer);
    SDL_aligned_free(z_index); // Free the z_index array
    SDL_aligned_free(visible);
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
    PROFILE_SCOPE("EntityManager::Constructor");

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
    PROFILE_SCOPE("EntityManager::Destructor");

    // Free all chunks
    for (int i = 0; i < chunk_count; i++) {
        delete chunks[i];
    }

    // Free chunks array and free indices
    free(chunks);
    free(free_indices);
}



// Initialize spatial grid for entity queries
void init_spatial_grid(SpatialGrid* grid, float world_width, float world_height, float cell_size) {
    PROFILE_FUNCTION();

    grid->cell_size = cell_size;
    grid->width = (int)ceil(world_width / cell_size);
    grid->height = (int)ceil(world_height / cell_size);
    grid->total_cells = grid->width * grid->height;

    // Allocate cells array using a flat array for better cache locality
    grid->cells = static_cast<int***>(SDL_aligned_alloc(CACHE_LINE_SIZE, grid->height * sizeof(int**)));
    grid->cell_counts = static_cast<int**>(SDL_aligned_alloc(CACHE_LINE_SIZE, grid->height * sizeof(int*)));
    grid->cell_capacities = static_cast<int**>(SDL_aligned_alloc(CACHE_LINE_SIZE, grid->height * sizeof(int*)));

    for (int y = 0; y < grid->height; y++) {
        grid->cells[y] = static_cast<int**>(SDL_aligned_alloc(CACHE_LINE_SIZE, grid->width * sizeof(int*)));
        grid->cell_counts[y] = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE, grid->width * sizeof(int)));
        grid->cell_capacities[y] = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE, grid->width * sizeof(int)));

        for (int x = 0; x < grid->width; x++) {
            int initial_capacity = 32; // Start with space for 32 entities per cell
            grid->cell_counts[y][x] = 0;
            grid->cell_capacities[y][x] = initial_capacity;
            grid->cells[y][x] = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE, initial_capacity * sizeof(int)));
        }
    }
}

// Free spatial grid
void free_spatial_grid(SpatialGrid* grid) {
    PROFILE_FUNCTION();

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
    PROFILE_FUNCTION();

    for (int y = 0; y < grid->height; y++) {
        for (int x = 0; x < grid->width; x++) {
            grid->cell_counts[y][x] = 0;
        }
    }
}







void init_texture_atlas(TextureAtlas* atlas, SDL_Renderer* renderer, int width, int height) {
    PROFILE_FUNCTION();

    atlas->texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_TARGET, width, height);
    SDL_SetTextureBlendMode(atlas->texture, SDL_BLENDMODE_BLEND);
    atlas->region_capacity = 32;
    atlas->region_count = 0;
    atlas->regions = static_cast<SDL_FRect*>(SDL_aligned_alloc(
        CACHE_LINE_SIZE, atlas->region_capacity * sizeof(SDL_FRect)));
}

// Get visible rect based on camera
SDL_FRect get_visible_rect(Camera* camera) {
    PROFILE_FUNCTION();

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
    PROFILE_FUNCTION();

    // Ensure we have enough space for vertices
    if (batch->vertex_count + 4 > batch->vertex_capacity) {
        int new_capacity = batch->vertex_capacity * 2;
        SDL_Vertex* new_vertices = static_cast<SDL_Vertex*>(SDL_aligned_alloc(
            CACHE_LINE_SIZE, new_capacity * sizeof(SDL_Vertex)));

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
            CACHE_LINE_SIZE, new_capacity * sizeof(int)));

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

// Find batch index for texture/z_index combination (changed from layer to z_index)
int find_batch_index(RenderBatch* batches, int batch_count, int texture_id, int z_index) {
    PROFILE_FUNCTION();

    for (int i = 0; i < batch_count; i++) {
        if (batches[i].texture_id == texture_id && batches[i].z_index == z_index) {
            return i;
        }
    }
    return -1; // Not found
}

// Create a new batch for a texture/z_index combination (changed from layer to z_index)
void create_batch(RenderBatch** batches, int* batch_count, int texture_id, int z_index) {
    PROFILE_FUNCTION();

    // Allocate new memory for batches array
    *batch_count = *batch_count + 1;
    *batches = static_cast<RenderBatch*>(realloc(*batches, (*batch_count) * sizeof(RenderBatch)));

    // Initialize the new batch
    int new_idx = *batch_count - 1;
    (*batches)[new_idx].texture_id = texture_id;
    (*batches)[new_idx].z_index = z_index; // Store z_index instead of layer
    (*batches)[new_idx].vertex_capacity = ENTITY_CHUNK_SIZE * 4; // Larger initial capacity
    (*batches)[new_idx].index_capacity = ENTITY_CHUNK_SIZE * 6; // Larger initial capacity
    (*batches)[new_idx].vertex_count = 0;
    (*batches)[new_idx].index_count = 0;
    (*batches)[new_idx].vertices = static_cast<SDL_Vertex*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
        (*batches)[new_idx].vertex_capacity * sizeof(SDL_Vertex)));
    (*batches)[new_idx].indices = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
        (*batches)[new_idx].index_capacity * sizeof(int)));
}

// Calculate screen coordinates
void calculate_screen_coordinates(float* world_x, float* world_y, float* width, float* height,
    float* screen_x, float* screen_y, float* screen_w, float* screen_h,
    int count, float visible_rect_x, float visible_rect_y, float zoom) {
    PROFILE_FUNCTION();

    for (int i = 0; i < count; i++) {
        screen_x[i] = roundf((world_x[i] - visible_rect_x) * zoom);
        screen_y[i] = roundf((world_y[i] - visible_rect_y) * zoom);
        screen_w[i] = width[i] * zoom;
        screen_h[i] = height[i] * zoom;
    }
}

// Update which grid cells are visible
void update_dynamic_loading(Engine* engine) {
    PROFILE_FUNCTION();

    // Cache frequently used values to reduce repeated calculations
    const float cell_size = engine->grid_cell_size;
    const int grid_width = engine->grid_width;
    const int grid_height = engine->grid_height;

    SDL_FRect visible_rect = get_visible_rect(&engine->camera);

    // Add padding to avoid pop-in at edges (1 cell padding)
    visible_rect.x -= cell_size;
    visible_rect.y -= cell_size;
    visible_rect.w += cell_size * 2;
    visible_rect.h += cell_size * 2;

    // Pre-compute and clamp grid cell boundaries (avoiding redundant calculations)
    const int start_x = std::max(0, (int)(visible_rect.x / cell_size));
    const int start_y = std::max(0, (int)(visible_rect.y / cell_size));
    const int end_x = std::min(grid_width - 1, (int)ceil((visible_rect.x + visible_rect.w) / cell_size));
    const int end_y = std::min(grid_height - 1, (int)ceil((visible_rect.y + visible_rect.h) / cell_size));

    // First mark all cells as unloaded - use memset for efficiency (much faster than nested loops)
    for (int x = 0; x < grid_width; x++) {
        memset(engine->grid_loaded[x], 0, grid_height * sizeof(bool));
    }

    // Then mark visible cells as loaded - use better memory access pattern by accessing row-by-row
    for (int x = start_x; x <= end_x; x++) {
        bool* row = engine->grid_loaded[x]; // Get pointer to row once
        for (int y = start_y; y <= end_y; y++) {
            row[y] = true;
        }
    }

    // For entities that need visibility check - calculate once
    const float min_x = visible_rect.x - cell_size; // Additional padding for fast check
    const float min_y = visible_rect.y - cell_size;
    const float max_x = visible_rect.x + visible_rect.w + cell_size;
    const float max_y = visible_rect.y + visible_rect.h + cell_size;

    // Update entity visible states based on grid loading
    for (int chunk_idx = 0; chunk_idx < engine->entities.chunk_count; chunk_idx++) {
        EntityChunk* chunk = engine->entities.chunks[chunk_idx];

        // Skip empty chunks
        if (!chunk || chunk->count == 0) continue;

        // Process all entities in the chunk
        for (int local_idx = 0; local_idx < chunk->count; local_idx++) {
            const float entity_x = chunk->x[local_idx];
            const float entity_y = chunk->y[local_idx];

            // Quick rejection test - outside extended visible area
            if (entity_x < min_x || entity_x > max_x || entity_y < min_y || entity_y > max_y) {
                chunk->visible[local_idx] = false;
                continue;
            }

            // Now do the detailed grid-based test
            const int grid_x = (int)(entity_x / cell_size);
            const int grid_y = (int)(entity_y / cell_size);

            // Combined bounds check and grid lookup
            chunk->visible[local_idx] =
                (grid_x >= 0 && grid_x < grid_width && grid_y >= 0 && grid_y < grid_height) ?
                engine->grid_loaded[grid_x][grid_y] : false;
        }
    }
}



Engine* engine_create(int window_width, int window_height, int world_width, int world_height, int cell_size) {
    PROFILE_FUNCTION();

    Engine* engine = static_cast<Engine*>(malloc(sizeof(Engine)));
    if (!engine) return NULL;

    // Use placement new to properly initialize C++ members
    new (&engine->entity_indices_pool) FixedBufferPool(100000 * sizeof(int), 4);
    new (&engine->screen_coords_pool) FixedBufferPool(100000 * sizeof(float) * 4, 4);
    new (&engine->entities) EntityManager();

    // Initialize vector instead of unordered_map for type_id lookup
    new (&engine->type_id_to_index) std::vector<int>();

    // Pre-allocate with a reasonable capacity and fill with -1 (no mapping)
    engine->type_id_to_index.reserve(1024);

    // Initialize next_type_id for auto-generation
    engine->next_type_id = 0;

    // Initialize entity type system with capacity for 200K+ types
    engine->entity_type_count = 0;
    engine->entity_type_capacity = 262144;  // 256K initial capacity
    engine->entity_types = static_cast<EntityTypeConfig*>(
        malloc(engine->entity_type_capacity * sizeof(EntityTypeConfig))
        );



    // Create window and renderer
    engine->window = SDL_CreateWindow("2D Game Engine - SDL3", window_width, window_height, 0);
    if (!engine->window) {
        engine->entity_indices_pool.~FixedBufferPool();
        engine->screen_coords_pool.~FixedBufferPool();
        engine->entities.~EntityManager();
        engine->type_id_to_index.~vector();
        free(engine);
        return NULL;
    }

    engine->renderer = SDL_CreateRenderer(engine->window, NULL);
    if (!engine->renderer) {
        SDL_DestroyWindow(engine->window);
        engine->entity_indices_pool.~FixedBufferPool();
        engine->screen_coords_pool.~FixedBufferPool();
        engine->entities.~EntityManager();
        engine->type_id_to_index.~vector();
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

    // Init render batches (for batch rendering) - use z_index instead of layer
    engine->batch_count = 8;
    engine->batches = static_cast<RenderBatch*>(malloc(engine->batch_count * sizeof(RenderBatch)));
    for (int i = 0; i < engine->batch_count; i++) {
        engine->batches[i].texture_id = i;
        engine->batches[i].z_index = 0;  // Initialize with z_index 0 instead of layer
        engine->batches[i].vertex_capacity = 4096;
        engine->batches[i].index_capacity = 6144;
        engine->batches[i].vertex_count = 0;
        engine->batches[i].index_count = 0;
        engine->batches[i].vertices = static_cast<SDL_Vertex*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
            engine->batches[i].vertex_capacity * sizeof(SDL_Vertex)));
        engine->batches[i].indices = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
            engine->batches[i].index_capacity * sizeof(int)));
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
    PROFILE_FUNCTION();

    if (!engine) return;

    // Free entity type system
    free(engine->entity_types);
    // Call destructors for C++ members
    engine->entities.~EntityManager();
    engine->entity_indices_pool.~FixedBufferPool();
    engine->screen_coords_pool.~FixedBufferPool();
    engine->type_id_to_index.~vector();

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
    PROFILE_FUNCTION();

    // Calculate delta time
    Uint64 current_time = SDL_GetTicks();
    float delta_time = (current_time - engine->last_frame_time) / 1000.0f;
    engine->last_frame_time = current_time;

    // Smooth FPS calculation
    engine->fps = 0.95f * engine->fps + 0.05f * (1.0f / delta_time);

    // Update dynamic loading based on camera position (updates visibility)
    {
        PROFILE_SCOPE("update_dynamic_loading");
        update_dynamic_loading(engine);
    }



    engine_update_entity_types(engine, delta_time);


    // Print both grid views
    print_active_entities_grid(engine);
    print_visible_entities_grid(engine);

}




void engine_render(Engine* engine) {
    PROFILE_FUNCTION();

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
    {
        PROFILE_SCOPE("spatial_grid_query");
        spatial_grid_query(&engine->grid, visible_rect, visible_indices, &visible_count,
            engine->entities.total_count, &engine->entities);
    }

    // Sort visible entities by z_index (changed from texture and layer)
    {
        PROFILE_SCOPE("radix_sort_entities");
        radix_sort_entities(visible_indices, visible_count, &engine->entities);
    }

    // Clear batches and reuse them
    {
        PROFILE_SCOPE("clear_batches");
        for (int i = 0; i < engine->batch_count; i++) {
            engine->batches[i].vertex_count = 0;
            engine->batches[i].index_count = 0;
        }
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
    {
        PROFILE_SCOPE("collect_entity_data");
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
    }

    // Calculate screen coordinates
    {
        PROFILE_SCOPE("calculate_screen_coordinates");
        calculate_screen_coordinates(world_x.data(), world_y.data(), width_f.data(), height_f.data(),
            screen_x, screen_y, screen_w, screen_h,
            visible_count, visible_rect.x, visible_rect.y, engine->camera.zoom);
    }

    // Track last texture/z_index to batch together (changed from layer to z_index)
    int last_texture_id = -1;
    int last_z_index = -1;
    int current_batch_idx = -1;

    // Add visible entities to batches
    {
        PROFILE_SCOPE("add_to_batches");
        for (int i = 0; i < visible_count; i++) {
            int entity_idx = visible_indices[i];

            // Get chunk and local index
            int chunk_idx, local_idx;
            engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);

            EntityChunk* chunk = engine->entities.chunks[chunk_idx];
            int texture_id = chunk->texture_id[local_idx];
            int z_index = chunk->z_index[local_idx];  // Use z_index instead of layer

            // If texture or z_index changed, get/create appropriate batch
            if (texture_id != last_texture_id || z_index != last_z_index) {
                // Find existing batch for this texture/z_index
                current_batch_idx = find_batch_index(engine->batches, engine->batch_count, texture_id, z_index);

                // If no batch exists, create one
                if (current_batch_idx == -1) {
                    create_batch(&engine->batches, &engine->batch_count, texture_id, z_index);
                    current_batch_idx = engine->batch_count - 1;
                }

                last_texture_id = texture_id;
                last_z_index = z_index;  // Track z_index instead of layer
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
    }

    // Render batches
    {
        PROFILE_SCOPE("render_batches");
        for (int i = 0; i < engine->batch_count; i++) {
            if (engine->batches[i].vertex_count > 0) {
                // Single draw call per batch!
                SDL_RenderGeometry(engine->renderer, engine->atlas.texture,
                    engine->batches[i].vertices, engine->batches[i].vertex_count,
                    engine->batches[i].indices, engine->batches[i].index_count);
            }
        }
    }
}

int engine_add_texture(Engine* engine, SDL_Surface* surface, int x, int y) {
    PROFILE_FUNCTION();

    int texture_id = engine->atlas.region_count;

    // Ensure capacity
    if (texture_id >= engine->atlas.region_capacity) {
        int new_capacity = engine->atlas.region_capacity * 2;
        SDL_FRect* new_regions = static_cast<SDL_FRect*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
            new_capacity * sizeof(SDL_FRect)));

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
    PROFILE_FUNCTION();

    engine->camera.x = x;
    engine->camera.y = y;
}

void engine_set_camera_zoom(Engine* engine, float zoom) {
    PROFILE_FUNCTION();

    engine->camera.zoom = zoom;
}

