#define SDL_MAIN_USE_CALLBACKS 1
//#define MY_RYZEN_CPU 1
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ATMCommon.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <immintrin.h>


// SIMD includes
#if defined(__x86_64__) || defined(_M_X64)
#define HAVE_SIMD 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define HAVE_SIMD 1
#elif defined(MY_RYZEN_CPU)  // Add your own definition in compiler flags
#include <immintrin.h>
#define HAVE_SIMD 1
#else
#define HAVE_SIMD 0
#endif


// Alignment for memory
#define CACHE_LINE_SIZE 64

// Forward declarations
typedef struct GameState GameState;

// Entity manager using Structure of Arrays (SoA) for cache efficiency
typedef struct {
    float* x;              // x positions - aligned
    float* y;              // y positions - aligned
    float* right;          // x + width (precomputed) - aligned
    float* bottom;         // y + height (precomputed) - aligned
    int* width;            // widths - aligned
    int* height;           // heights - aligned
    int* texture_id;       // texture IDs - aligned
    int* layer;            // z-ordering/layers - aligned
    bool* active;          // is entity active/visible - aligned
    int** grid_cell;       // grid cell references for each entity - aligned
    int capacity;          // allocated capacity
    int count;             // actual count of entities
} EntityManager;

// Spatial grid for efficient entity queries (replaces quadtree)
typedef struct {
    int** cells;           // 2D array of entity index arrays
    int* cell_counts;      // Number of entities in each cell
    int* cell_capacities;  // Capacity of each cell
    float cell_size;       // Size of each cell
    int width, height;     // Grid dimensions
    int total_cells;       // Total number of cells
} SpatialGrid;

// Rendering batch (groups by texture and layer)
typedef struct {
    int texture_id;
    int layer;             // Added to combine texture + layer
    SDL_Vertex* vertices;  // Vertex data for batch
    int* indices;          // Index data for batch
    int vertex_count;
    int index_count;
    int vertex_capacity;
    int index_capacity;
} RenderBatch;

// Texture atlas
typedef struct {
    SDL_Texture* texture;
    SDL_FRect* regions;    // UV regions for each subtexture
    int region_count;
    int region_capacity;
} TextureAtlas;

// Camera for culling
typedef struct {
    float x, y;            // Position
    float width, height;   // Viewport dimensions
    float zoom;            // Zoom level
} Camera;

// Reusable buffer pool for temporary allocations
typedef struct {
    void** buffers;
    int count;
    int capacity;
    size_t buffer_size;
} BufferPool;

// Game state
typedef struct GameState {
    SDL_Window* window;
    SDL_Renderer* renderer;
    EntityManager entities;
    SpatialGrid grid;      // Spatial grid replaces quadtree
    RenderBatch* batches;
    int batch_count;
    TextureAtlas atlas;
    Camera camera;
    SDL_FRect world_bounds;  // Total world size
    float grid_cell_size;    // Size of dynamic loading grid cells
    bool** grid_loaded;      // 2D array tracking loaded grid cells
    int grid_width, grid_height;  // Grid dimensions
    Uint64 last_frame_time;
    float fps;
    int benchmark_frame;    // For benchmark tracking

    // Reusable buffer pools
    BufferPool entity_indices_pool;
    BufferPool screen_coords_pool;

    // Added for Emscripten
    bool benchmark_running;
    Uint64 benchmark_start_time;
    Uint64 last_report_time;
    double total_ms;
    int total_frames;
    int frames_since_report;
    int target_entity_count;
    int current_entity_count;
    int benchmark_duration_seconds;
    int entities_per_frame;
} GameState;



// Initialize buffer pool
void init_buffer_pool(BufferPool* pool, size_t buffer_size, int initial_capacity) {
    pool->buffer_size = buffer_size;
    pool->capacity = initial_capacity;
    pool->count = 0;
    pool->buffers = (void**)SDL_aligned_alloc(initial_capacity * sizeof(void*), CACHE_LINE_SIZE);
}

// Get buffer from pool (or allocate new one)
void* get_buffer(BufferPool* pool) {
    if (pool->count > 0) {
        return pool->buffers[--pool->count];
    }

    // Allocate new buffer with proper alignment
    return SDL_aligned_alloc(pool->buffer_size, 32); // 32-byte alignment for AVX
}

// Return buffer to pool
void return_buffer(BufferPool* pool, void* buffer) {
    if (pool->count >= pool->capacity) {
        // Pool is full, resize
        int new_capacity = pool->capacity * 2;
        void** new_buffers = (void**) SDL_aligned_alloc(new_capacity * sizeof(void*), CACHE_LINE_SIZE);
        memcpy(new_buffers, pool->buffers, pool->count * sizeof(void*));
        SDL_aligned_free(pool->buffers);
        pool->buffers = new_buffers;
        pool->capacity = new_capacity;
    }

    pool->buffers[pool->count++] = buffer;
}

// Free buffer pool
void free_buffer_pool(BufferPool* pool) {
    for (int i = 0; i < pool->count; i++) {
        SDL_aligned_free(pool->buffers[i]);
    }
    SDL_aligned_free(pool->buffers);
    pool->count = 0;
    pool->capacity = 0;
}

// Initialize spatial grid for entity queries
void init_spatial_grid(SpatialGrid* grid, float world_width, float world_height, float cell_size) {
    grid->cell_size = cell_size;
    grid->width = (int)ceil(world_width / cell_size);
    grid->height = (int)ceil(world_height / cell_size);
    grid->total_cells = grid->width * grid->height;

    // Allocate cells array as a flat array for better cache locality
    grid->cells = (int**)SDL_aligned_alloc(grid->total_cells * sizeof(int*), CACHE_LINE_SIZE);
    grid->cell_counts = (int*)SDL_aligned_alloc(grid->total_cells * sizeof(int), CACHE_LINE_SIZE);
    grid->cell_capacities = (int*)SDL_aligned_alloc(grid->total_cells * sizeof(int), CACHE_LINE_SIZE);

    // Initialize cells
    for (int i = 0; i < grid->total_cells; i++) {
        grid->cell_counts[i] = 0;
        grid->cell_capacities[i] = 32; // Increased initial capacity
        grid->cells[i] = (int*)SDL_aligned_alloc(grid->cell_capacities[i] * sizeof(int), CACHE_LINE_SIZE);
    }
}

// Free spatial grid
void free_spatial_grid(SpatialGrid* grid) {
    for (int i = 0; i < grid->total_cells; i++) {
        SDL_aligned_free(grid->cells[i]);
    }
    SDL_aligned_free(grid->cells);
    SDL_aligned_free(grid->cell_counts);
    SDL_aligned_free(grid->cell_capacities);
}

// Clear all cells in the spatial grid
void clear_spatial_grid(SpatialGrid* grid) {
    for (int i = 0; i < grid->total_cells; i++) {
        grid->cell_counts[i] = 0;
    }
}

// Get cell index from world coordinates
int get_cell_index(SpatialGrid* grid, float x, float y) {
    int grid_x = (int)(x / grid->cell_size);
    int grid_y = (int)(y / grid->cell_size);

    // Clamp to grid bounds
    grid_x = (grid_x < 0) ? 0 : ((grid_x >= grid->width) ? grid->width - 1 : grid_x);
    grid_y = (grid_y < 0) ? 0 : ((grid_y >= grid->height) ? grid->height - 1 : grid_y);

    return grid_y * grid->width + grid_x;
}

void spatial_grid_add(SpatialGrid* grid, int entity_idx, float x, float y) {
    int cell_idx = get_cell_index(grid, x, y);

    // Ensure capacity
    if (grid->cell_counts[cell_idx] >= grid->cell_capacities[cell_idx]) {
        int new_capacity = grid->cell_capacities[cell_idx] * 2;
        int* new_cell = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);

        // Copy existing data to new array
        for (int i = 0; i < grid->cell_counts[cell_idx]; i++) {
            new_cell[i] = grid->cells[cell_idx][i];
        }

        // Free old array and update pointers/capacity
        SDL_aligned_free(grid->cells[cell_idx]);
        grid->cells[cell_idx] = new_cell;
        grid->cell_capacities[cell_idx] = new_capacity;
    }

    // Add entity to cell
    grid->cells[cell_idx][grid->cell_counts[cell_idx]++] = entity_idx;
}

// SIMD-based frustum culling for entities
int simd_frustum_cull(EntityManager* entities, SDL_FRect view_rect, int* cell_entities,
    int cell_entity_count, int* result_indices) {
    int result_count = 0;

#if HAVE_SIMD
    // Process multiple entities at once with SIMD
#if defined(__x86_64__) || defined(_M_X64)
    // SSE/AVX implementation for x86
    __m128 view_min_x = _mm_set1_ps(view_rect.x);
    __m128 view_min_y = _mm_set1_ps(view_rect.y);
    __m128 view_max_x = _mm_set1_ps(view_rect.x + view_rect.w);
    __m128 view_max_y = _mm_set1_ps(view_rect.y + view_rect.h);

    for (int i = 0; i < cell_entity_count; i += 4) {
        int batch_size = (cell_entity_count - i < 4) ? cell_entity_count - i : 4;

        // Load entity indices for this batch
        int indices_batch[4] = { 0 };
        for (int j = 0; j < batch_size; j++) {
            indices_batch[j] = cell_entities[i + j];
        }

        // Using x, y, right, bottom for each entity
        __m128 entity_x = _mm_set_ps(
            batch_size > 3 ? entities->x[indices_batch[3]] : 0.0f,
            batch_size > 2 ? entities->x[indices_batch[2]] : 0.0f,
            batch_size > 1 ? entities->x[indices_batch[1]] : 0.0f,
            entities->x[indices_batch[0]]
        );

        __m128 entity_y = _mm_set_ps(
            batch_size > 3 ? entities->y[indices_batch[3]] : 0.0f,
            batch_size > 2 ? entities->y[indices_batch[2]] : 0.0f,
            batch_size > 1 ? entities->y[indices_batch[1]] : 0.0f,
            entities->y[indices_batch[0]]
        );

        __m128 entity_right = _mm_set_ps(
            batch_size > 3 ? entities->right[indices_batch[3]] : 0.0f,
            batch_size > 2 ? entities->right[indices_batch[2]] : 0.0f,
            batch_size > 1 ? entities->right[indices_batch[1]] : 0.0f,
            entities->right[indices_batch[0]]
        );

        __m128 entity_bottom = _mm_set_ps(
            batch_size > 3 ? entities->bottom[indices_batch[3]] : 0.0f,
            batch_size > 2 ? entities->bottom[indices_batch[2]] : 0.0f,
            batch_size > 1 ? entities->bottom[indices_batch[1]] : 0.0f,
            entities->bottom[indices_batch[0]]
        );

        // Check if entities are in view
        __m128 test1 = _mm_cmpge_ps(entity_x, view_max_x);      // entity_x >= view_max_x
        __m128 test2 = _mm_cmpge_ps(view_min_x, entity_right);  // view_min_x >= entity_right
        __m128 test3 = _mm_cmpge_ps(entity_y, view_max_y);      // entity_y >= view_max_y
        __m128 test4 = _mm_cmpge_ps(view_min_y, entity_bottom); // view_min_y >= entity_bottom

        // Combine tests: if any test is true, entity is outside view
        __m128 or1 = _mm_or_ps(test1, test2);
        __m128 or2 = _mm_or_ps(test3, test4);
        __m128 outside = _mm_or_ps(or1, or2);

        // Convert to integer mask
        int mask = _mm_movemask_ps(outside);

        // Add visible entities to results
        for (int j = 0; j < batch_size; j++) {
            if (!(mask & (1 << j)) && entities->active[indices_batch[j]]) {
                result_indices[result_count++] = indices_batch[j];
            }
        }
    }

#elif defined(__ARM_NEON)
    // NEON implementation for ARM
    float32x4_t view_min_x = vdupq_n_f32(view_rect.x);
    float32x4_t view_min_y = vdupq_n_f32(view_rect.y);
    float32x4_t view_max_x = vdupq_n_f32(view_rect.x + view_rect.w);
    float32x4_t view_max_y = vdupq_n_f32(view_rect.y + view_rect.h);

    for (int i = 0; i < cell_entity_count; i += 4) {
        int batch_size = (cell_entity_count - i < 4) ? cell_entity_count - i : 4;

        // Load entity indices for this batch
        int indices_batch[4] = { 0 };
        for (int j = 0; j < batch_size; j++) {
            indices_batch[j] = cell_entities[i + j];
        }

        // Prepare arrays for loading entity coordinates
        float entity_x_arr[4] = { 0.0f };
        float entity_y_arr[4] = { 0.0f };
        float entity_right_arr[4] = { 0.0f };
        float entity_bottom_arr[4] = { 0.0f };

        // Load entity coordinates
        for (int j = 0; j < batch_size; j++) {
            entity_x_arr[j] = entities->x[indices_batch[j]];
            entity_y_arr[j] = entities->y[indices_batch[j]];
            entity_right_arr[j] = entities->right[indices_batch[j]];
            entity_bottom_arr[j] = entities->bottom[indices_batch[j]];
        }

        float32x4_t entity_x = vld1q_f32(entity_x_arr);
        float32x4_t entity_y = vld1q_f32(entity_y_arr);
        float32x4_t entity_right = vld1q_f32(entity_right_arr);
        float32x4_t entity_bottom = vld1q_f32(entity_bottom_arr);

        // Check if entities are in view
        uint32x4_t test1 = vcgeq_f32(entity_x, view_max_x);      // entity_x >= view_max_x
        uint32x4_t test2 = vcgeq_f32(view_min_x, entity_right);  // view_min_x >= entity_right
        uint32x4_t test3 = vcgeq_f32(entity_y, view_max_y);      // entity_y >= view_max_y
        uint32x4_t test4 = vcgeq_f32(view_min_y, entity_bottom); // view_min_y >= entity_bottom

        // Combine tests using OR operations
        uint32x4_t or1 = vorrq_u32(test1, test2);
        uint32x4_t or2 = vorrq_u32(test3, test4);
        uint32x4_t outside = vorrq_u32(or1, or2);

        // Extract results
        uint32_t mask[4];
        vst1q_u32(mask, outside);

        // Add visible entities to results
        for (int j = 0; j < batch_size; j++) {
            if (!mask[j] && entities->active[indices_batch[j]]) {
                result_indices[result_count++] = indices_batch[j];
            }
        }
    }
#endif
#else
    // Fallback without SIMD
    for (int i = 0; i < cell_entity_count; i++) {
        int entity_idx = cell_entities[i];
        if (entities->active[entity_idx]) {
            float x = entities->x[entity_idx];
            float y = entities->y[entity_idx];
            float right = entities->right[entity_idx];
            float bottom = entities->bottom[entity_idx];

            // Check if entity is in view
            if (!(right <= view_rect.x || x >= view_rect.x + view_rect.w ||
                bottom <= view_rect.y || y >= view_rect.y + view_rect.h)) {
                result_indices[result_count++] = entity_idx;
            }
            }
}
#endif

    return result_count;
}


// Query entities in a region using SIMD-based frustum culling
void spatial_grid_query(SpatialGrid* grid, SDL_FRect query_rect,
    int* result_indices, int* result_count, int max_results,
    EntityManager* entities) {
    // Calculate grid cells that overlap with query rect
    int start_x = (int)(query_rect.x / grid->cell_size);
    int start_y = (int)(query_rect.y / grid->cell_size);
    int end_x = (int)ceil((query_rect.x + query_rect.w) / grid->cell_size);
    int end_y = (int)ceil((query_rect.y + query_rect.h) / grid->cell_size);

    // Clamp to grid bounds
    start_x = (start_x < 0) ? 0 : ((start_x >= grid->width) ? grid->width - 1 : start_x);
    start_y = (start_y < 0) ? 0 : ((start_y >= grid->height) ? grid->height - 1 : start_y);
    end_x = (end_x < 0) ? 0 : ((end_x >= grid->width) ? grid->width - 1 : end_x);
    end_y = (end_y < 0) ? 0 : ((end_y >= grid->height) ? grid->height - 1 : end_y);

    *result_count = 0;

    // Iterate through cells that overlap the query rect
    for (int y = start_y; y <= end_y && *result_count < max_results; y++) {
        for (int x = start_x; x <= end_x && *result_count < max_results; x++) {
            int cell_idx = y * grid->width + x;
            int* cell_entities = grid->cells[cell_idx];
            int cell_entity_count = grid->cell_counts[cell_idx];

            // Use SIMD-based frustum culling
            int visible_count = simd_frustum_cull(entities, query_rect,
                cell_entities, cell_entity_count,
                &result_indices[*result_count]);

            *result_count += visible_count;

            // Ensure we don't exceed the max results
            if (*result_count > max_results) {
                *result_count = max_results;
                return;
            }
        }
    }
}

// Radix sort optimization for entities (process more bits per pass)
// Radix sort optimization for entities (process more bits per pass)
void radix_sort_entities(int* indices, int count, EntityManager* entities) {
    if (count <= 1) return;

    // Allocate temporary arrays
    int* temp_indices = (int*)SDL_aligned_alloc(count * sizeof(int), CACHE_LINE_SIZE);

    // Using 8 bits per pass (256 buckets)
    const int RADIX_BITS = 8;
    const int RADIX_SIZE = 1 << RADIX_BITS;
    const int RADIX_MASK = RADIX_SIZE - 1;

    // First sort by texture_id
    int count_texture[RADIX_SIZE] = { 0 };

    // Count frequencies
    for (int i = 0; i < count; i++) {
        int entity_idx = indices[i];
        int texture_id = entities->texture_id[entity_idx] & RADIX_MASK;
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
        int texture_id = entities->texture_id[entity_idx] & RADIX_MASK;
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
        int texture_id = entities->texture_id[entity_idx] & RADIX_MASK;
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
            int layer = entities->layer[entity_idx] & RADIX_MASK;
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
            int layer = entities->layer[entity_idx] & RADIX_MASK;
            temp_indices[start + count_layer[layer]++] = entity_idx;
        }

        // Copy back to original array just for this texture group
        memcpy(&indices[start], &temp_indices[start], size * sizeof(int));
    }

    // Cleanup
    SDL_aligned_free(temp_indices);
}

// Initialize entity manager with SoA design (aligned memory)
void init_entity_manager(EntityManager* manager, int initial_capacity) {
    manager->capacity = initial_capacity;
    manager->count = 0;

    // Allocate aligned arrays for better SIMD performance
    manager->x = (float*)SDL_aligned_alloc(initial_capacity * sizeof(float), CACHE_LINE_SIZE);
    manager->y = (float*)SDL_aligned_alloc(initial_capacity * sizeof(float), CACHE_LINE_SIZE);
    manager->right = (float*)SDL_aligned_alloc(initial_capacity * sizeof(float), CACHE_LINE_SIZE);
    manager->bottom = (float*)SDL_aligned_alloc(initial_capacity * sizeof(float), CACHE_LINE_SIZE);
    manager->width = (int*)SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->height = (int*)SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->texture_id = (int*)SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->layer = (int*)SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->active = (bool*)SDL_aligned_alloc(initial_capacity * sizeof(bool), CACHE_LINE_SIZE);
    manager->grid_cell = (int**)SDL_aligned_alloc(initial_capacity * sizeof(int*), CACHE_LINE_SIZE);
}

// Free entity manager
void free_entity_manager(EntityManager* manager) {
    SDL_aligned_free(manager->x);
    SDL_aligned_free(manager->y);
    SDL_aligned_free(manager->right);
    SDL_aligned_free(manager->bottom);
    SDL_aligned_free(manager->width);
    SDL_aligned_free(manager->height);
    SDL_aligned_free(manager->texture_id);
    SDL_aligned_free(manager->layer);
    SDL_aligned_free(manager->active);
    SDL_aligned_free(manager->grid_cell);
}

// Check if a point is inside a rectangle
bool point_in_rect(float x, float y, SDL_FRect rect) {
    return x >= rect.x && x < rect.x + rect.w &&
        y >= rect.y && y < rect.y + rect.h;
}

// Check if two rectangles intersect
bool rects_intersect(SDL_FRect a, SDL_FRect b) {
    return !(a.x + a.w <= b.x || a.x >= b.x + b.w ||
        a.y + a.h <= b.y || a.y >= b.y + b.h);
}

// Initialize a texture atlas
void init_texture_atlas(TextureAtlas* atlas, SDL_Renderer* renderer, int width, int height) {
    atlas->texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_TARGET, width, height);
    SDL_SetTextureBlendMode(atlas->texture, SDL_BLENDMODE_BLEND);
    atlas->region_capacity = 32;
    atlas->region_count = 0; 
    atlas->regions = (SDL_FRect*)SDL_aligned_alloc(atlas->region_capacity * sizeof(SDL_FRect), CACHE_LINE_SIZE);
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
    // Ensure we have enough space
    if (batch->vertex_count + 4 > batch->vertex_capacity) {
        int new_capacity = batch->vertex_capacity * 2;
        SDL_Vertex* new_vertices = (SDL_Vertex*)SDL_aligned_alloc(
            new_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE);
       
        // Copy existing data
        memcpy(new_vertices, batch->vertices, batch->vertex_count * sizeof(SDL_Vertex));
        SDL_aligned_free(batch->vertices);
        batch->vertices = new_vertices;
        batch->vertex_capacity = new_capacity;
    }

    if (batch->index_count + 6 > batch->index_capacity) {
        int new_capacity = batch->index_capacity * 2;
        int* new_indices = (int*)SDL_aligned_alloc(
            new_capacity * sizeof(int), CACHE_LINE_SIZE);

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

// Update which grid cells are visible and load/unload as needed
void update_dynamic_loading(GameState* state) {
    SDL_FRect visible_rect = get_visible_rect(&state->camera);

    // Add padding to avoid pop-in at edges (1 cell padding)
    visible_rect.x -= state->grid_cell_size;
    visible_rect.y -= state->grid_cell_size;
    visible_rect.w += state->grid_cell_size * 2;
    visible_rect.h += state->grid_cell_size * 2;

    // Compute grid cells that should be loaded
    int start_x = (int)(visible_rect.x / state->grid_cell_size);
    int start_y = (int)(visible_rect.y / state->grid_cell_size);
    int end_x = (int)ceil((visible_rect.x + visible_rect.w) / state->grid_cell_size);
    int end_y = (int)ceil((visible_rect.y + visible_rect.h) / state->grid_cell_size);

    // Clamp to world bounds
    start_x = (start_x < 0) ? 0 : ((start_x >= state->grid_width) ? state->grid_width - 1 : start_x);
    start_y = (start_y < 0) ? 0 : ((start_y >= state->grid_height) ? state->grid_height - 1 : start_y);
    end_x = (end_x < 0) ? 0 : ((end_x >= state->grid_width) ? state->grid_width - 1 : end_x);
    end_y = (end_y < 0) ? 0 : ((end_y >= state->grid_height) ? state->grid_height - 1 : end_y);

    // First unload all cells
    for (int x = 0; x < state->grid_width; x++) {
        for (int y = 0; y < state->grid_height; y++) {
            state->grid_loaded[x][y] = false;
        }
    }

    // Then load visible cells
    for (int x = start_x; x <= end_x; x++) {
        for (int y = start_y; y <= end_y; y++) {
            state->grid_loaded[x][y] = true;
        }
    }

    // Update entity active states based on grid loading
    for (int i = 0; i < state->entities.count; i++) {
        int grid_x = (int)(state->entities.x[i] / state->grid_cell_size);
        int grid_y = (int)(state->entities.y[i] / state->grid_cell_size);

        if (grid_x >= 0 && grid_x < state->grid_width &&
            grid_y >= 0 && grid_y < state->grid_height) {
            state->entities.active[i] = state->grid_loaded[grid_x][grid_y];
        }
        else {
            state->entities.active[i] = false;
        }
    }
}

// Initialize game state
GameState* init_game_state(int window_width, int window_height, int world_width, int world_height) {
    GameState* state = (GameState*)SDL_malloc(sizeof(GameState));
    if (!state) return NULL;
    memset(state, 0, sizeof(GameState));

    // Create window and renderer
    state->window = SDL_CreateWindow("Building War - SDL3", window_width, window_height, 0);
    if (!state->window) {
        free(state);
        return NULL;
    }

    state->renderer = SDL_CreateRenderer(state->window, NULL);
    if (!state->renderer) {
        SDL_DestroyWindow(state->window);
        free(state);
        return NULL;
    }

    // Init entity manager (SoA design)
    init_entity_manager(&state->entities, 100000); // Increased initial capacity

    // Init world bounds
    state->world_bounds.x = 0;
    state->world_bounds.y = 0;
    state->world_bounds.w = world_width;
    state->world_bounds.h = world_height;

    // Init spatial grid (replaces quadtree) - smaller cell size for better performance
    init_spatial_grid(&state->grid, world_width, world_height, 128.0f);

    // Init texture atlas
    init_texture_atlas(&state->atlas, state->renderer, 2048, 2048);

    // Init camera
    state->camera.x = 0;
    state->camera.y = 0;
    state->camera.width = window_width;
    state->camera.height = window_height;
    state->camera.zoom = 1.0f;

    // Init render batches (for batch rendering) with larger initial capacities
    state->batch_count = 8;
    state->batches = (RenderBatch*)SDL_malloc(state->batch_count * sizeof(RenderBatch));
    for (int i = 0; i < state->batch_count; i++) {
        state->batches[i].texture_id = i;
        state->batches[i].layer = 0;
        state->batches[i].vertex_capacity = 4096;  // Increased capacity
        state->batches[i].index_capacity = 6144;   // Increased capacity
        state->batches[i].vertex_count = 0;
        state->batches[i].index_count = 0;
        state->batches[i].vertices = (SDL_Vertex*)SDL_aligned_alloc(
            state->batches[i].vertex_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE);
        state->batches[i].indices = (int*)SDL_aligned_alloc(
            state->batches[i].index_capacity * sizeof(int), CACHE_LINE_SIZE);
    }

    // Init dynamic grid loading
    state->grid_cell_size = 256.0f;
    state->grid_width = (int)ceil(world_width / state->grid_cell_size);
    state->grid_height = (int)ceil(world_height / state->grid_cell_size);

    // Allocate grid loaded array
    state->grid_loaded = (bool**)SDL_malloc(state->grid_width * sizeof(bool*));
    for (int x = 0; x < state->grid_width; x++) {
        state->grid_loaded[x] = (bool*)SDL_malloc(state->grid_height * sizeof(bool));
        for (int y = 0; y < state->grid_height; y++) {
            state->grid_loaded[x][y] = false;
        }
    }

    // Initialize buffer pools for reuse
    init_buffer_pool(&state->entity_indices_pool, 100000 * sizeof(int), 4);
    init_buffer_pool(&state->screen_coords_pool, 100000 * sizeof(float) * 4, 4);

    // Init timing
    state->last_frame_time = SDL_GetTicks();
    state->fps = 0.0f;
    state->benchmark_frame = 0;

    // Emscripten benchmark settings - much more modest for web
    state->benchmark_running = false;
    state->target_entity_count = 5000; // Reduce for Emscripten
    state->current_entity_count = 0;
    state->benchmark_duration_seconds = 10;
    state->entities_per_frame = 200; // Add entities gradually

    return state;
}

// Free game state resources
void free_game_state(GameState* state) {
    if (!state) return;

    // Free entity manager
    free_entity_manager(&state->entities);

    // Free spatial grid
    free_spatial_grid(&state->grid);

    // Free batches
    for (int i = 0; i < state->batch_count; i++) {
        SDL_aligned_free(state->batches[i].vertices);
        SDL_aligned_free(state->batches[i].indices);
    }
    free(state->batches);

    // Free atlas
    SDL_DestroyTexture(state->atlas.texture);
    SDL_aligned_free(state->atlas.regions);

    // Free grid
    for (int x = 0; x < state->grid_width; x++) {
        free(state->grid_loaded[x]);
    }
    free(state->grid_loaded);

    // Free buffer pools
    free_buffer_pool(&state->entity_indices_pool);
    free_buffer_pool(&state->screen_coords_pool);

    // Free SDL resources
    SDL_DestroyRenderer(state->renderer);
    SDL_DestroyWindow(state->window);

    free(state);
}

// Add an entity to the game world
int add_entity(GameState* state, float x, float y, int width, int height,
    int texture_id, int layer) {
    EntityManager* em = &state->entities;

    // Ensure we have capacity (SoA growth pattern)
    if (em->count >= em->capacity) {
        int new_capacity = em->capacity * 2;

        // Allocate new arrays
        float* new_x = (float*)SDL_aligned_alloc(new_capacity * sizeof(float), CACHE_LINE_SIZE);
        float* new_y = (float*)SDL_aligned_alloc(new_capacity * sizeof(float), CACHE_LINE_SIZE);
        float* new_right = (float*)SDL_aligned_alloc(new_capacity * sizeof(float), CACHE_LINE_SIZE);
        float* new_bottom = (float*)SDL_aligned_alloc(new_capacity * sizeof(float), CACHE_LINE_SIZE);
        int* new_width = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        int* new_height = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        int* new_texture_id = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        int* new_layer = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        bool* new_active = (bool*)SDL_aligned_alloc(new_capacity * sizeof(bool), CACHE_LINE_SIZE);
        int** new_grid_cell = (int**)SDL_aligned_alloc(new_capacity * sizeof(int*), CACHE_LINE_SIZE);

        // Copy existing data
        memcpy(new_x, em->x, em->count * sizeof(float));
        memcpy(new_y, em->y, em->count * sizeof(float));
        memcpy(new_right, em->right, em->count * sizeof(float));
        memcpy(new_bottom, em->bottom, em->count * sizeof(float));
        memcpy(new_width, em->width, em->count * sizeof(int));
        memcpy(new_height, em->height, em->count * sizeof(int));
        memcpy(new_texture_id, em->texture_id, em->count * sizeof(int));
        memcpy(new_layer, em->layer, em->count * sizeof(int));
        memcpy(new_active, em->active, em->count * sizeof(bool));
        memcpy(new_grid_cell, em->grid_cell, em->count * sizeof(int*));

        // Free old arrays
        SDL_aligned_free(em->x);
        SDL_aligned_free(em->y);
        SDL_aligned_free(em->right);
        SDL_aligned_free(em->bottom);
        SDL_aligned_free(em->width);
        SDL_aligned_free(em->height);
        SDL_aligned_free(em->texture_id);
        SDL_aligned_free(em->layer);
        SDL_aligned_free(em->active);
        SDL_aligned_free(em->grid_cell);

        // Update pointers
        em->x = new_x;
        em->y = new_y;
        em->right = new_right;
        em->bottom = new_bottom;
        em->width = new_width;
        em->height = new_height;
        em->texture_id = new_texture_id;
        em->layer = new_layer;
        em->active = new_active;
        em->grid_cell = new_grid_cell;
        em->capacity = new_capacity;
    }

    int entity_idx = em->count++;

    // Set entity properties
    em->x[entity_idx] = x;
    em->y[entity_idx] = y;
    em->width[entity_idx] = width;
    em->height[entity_idx] = height;

    // Precompute right and bottom for faster culling
    em->right[entity_idx] = x + width;
    em->bottom[entity_idx] = y + height;

    em->texture_id[entity_idx] = texture_id;
    em->layer[entity_idx] = layer;

    // Determine if entity should be active based on grid loading
    int grid_x = (int)(x / state->grid_cell_size);
    int grid_y = (int)(y / state->grid_cell_size);

    if (grid_x >= 0 && grid_x < state->grid_width &&
        grid_y >= 0 && grid_y < state->grid_height) {
        em->active[entity_idx] = state->grid_loaded[grid_x][grid_y];
    }
    else {
        em->active[entity_idx] = false;
    }

    // Add to spatial grid instead of quadtree
    spatial_grid_add(&state->grid, entity_idx, x, y);

    return entity_idx;
}

// Add a texture to the atlas
int add_texture_to_atlas(GameState* state, SDL_Surface* surface, int x, int y) {
    int texture_id = state->atlas.region_count;

    // Ensure capacity
    if (texture_id >= state->atlas.region_capacity) {
        state->atlas.region_capacity *= 2;
        state->atlas.regions = (SDL_FRect*)SDL_aligned_alloc(
            state->atlas.region_capacity * sizeof(SDL_FRect), CACHE_LINE_SIZE);
    }

    // Calculate normalized UV coordinates
    float atlas_width, atlas_height;
    SDL_GetTextureSize(state->atlas.texture, &atlas_width, &atlas_height);

    SDL_FRect region = {
        (float)x / atlas_width,
        (float)y / atlas_height,
        (float)surface->w / atlas_width,
        (float)surface->h / atlas_height
    };

    state->atlas.regions[texture_id] = region;
    state->atlas.region_count++;

    // Copy surface to atlas texture
    SDL_Texture* temp = SDL_CreateTextureFromSurface(state->renderer, surface);

    // Set render target to atlas
    SDL_Texture* old_target = SDL_GetRenderTarget(state->renderer);
    SDL_SetRenderTarget(state->renderer, state->atlas.texture);

    // Copy texture to atlas
    SDL_FRect dest = { (float)x, (float)y, (float)surface->w, (float)surface->h };
    SDL_RenderTexture(state->renderer, temp, NULL, &dest);

    // Reset render target
    SDL_SetRenderTarget(state->renderer, old_target);

    // Clean up
    SDL_DestroyTexture(temp);

    return texture_id;
}

// Update game state
void update_game(GameState* state) {
    // Calculate delta time
    Uint64 current_time = SDL_GetTicks();
    float delta_time = (current_time - state->last_frame_time) / 1000.0f;
    state->last_frame_time = current_time;

    // Smooth FPS calculation
    state->fps = 0.95f * state->fps + 0.05f * (1.0f / delta_time);

    // Update dynamic grid loading based on camera position
    update_dynamic_loading(state);

    // Clear spatial grid
    clear_spatial_grid(&state->grid);

    // Rebuild spatial grid with active entities
    for (int i = 0; i < state->entities.count; i++) {
        if (state->entities.active[i]) {
            spatial_grid_add(&state->grid, i, state->entities.x[i], state->entities.y[i]);
        }
    }
}

// Calculate screen coordinates using SIMD if available
void calculate_screen_coordinates(float* world_x, float* world_y, float* width, float* height,
    float* screen_x, float* screen_y, float* screen_w, float* screen_h,
    int count, float visible_rect_x, float visible_rect_y, float zoom) {
#if true
    // Process multiple entities at once with SIMD
#if defined(__x86_64__) || defined(_M_X64) || true
    // SSE implementation for x86
    __m128 vrect_x = _mm_set1_ps(visible_rect_x);
    __m128 vrect_y = _mm_set1_ps(visible_rect_y);
    __m128 vzoom = _mm_set1_ps(zoom);

    for (int i = 0; i < count; i += 4) {
        int remaining = (count - i < 4) ? count - i : 4;

        if (remaining == 4) {
            // Load 4 world_x, world_y values
            __m128 vx = _mm_load_ps(&world_x[i]);
            __m128 vy = _mm_load_ps(&world_y[i]);
            __m128 vw = _mm_load_ps(&width[i]);
            __m128 vh = _mm_load_ps(&height[i]);

            // Calculate screen coordinates
            __m128 vscr_x = _mm_mul_ps(_mm_sub_ps(vx, vrect_x), vzoom);
            __m128 vscr_y = _mm_mul_ps(_mm_sub_ps(vy, vrect_y), vzoom);
            __m128 vscr_w = _mm_mul_ps(vw, vzoom);
            __m128 vscr_h = _mm_mul_ps(vh, vzoom);

            // Store results
            _mm_store_ps(&screen_x[i], vscr_x);
            _mm_store_ps(&screen_y[i], vscr_y);
            _mm_store_ps(&screen_w[i], vscr_w);
            _mm_store_ps(&screen_h[i], vscr_h);
        }
        else {
            // Fallback for partial vectors
            for (int j = 0; j < remaining; j++) {
                screen_x[i + j] = (world_x[i + j] - visible_rect_x) * zoom;
                screen_y[i + j] = (world_y[i + j] - visible_rect_y) * zoom;
                screen_w[i + j] = width[i + j] * zoom;
                screen_h[i + j] = height[i + j] * zoom;
            }
        }
    }
#elif defined(__ARM_NEON)
    // NEON implementation for ARM
    float32x4_t vrect_x = vdupq_n_f32(visible_rect_x);
    float32x4_t vrect_y = vdupq_n_f32(visible_rect_y);
    float32x4_t vzoom = vdupq_n_f32(zoom);

    for (int i = 0; i < count; i += 4) {
        int remaining = (count - i < 4) ? count - i : 4;

        if (remaining == 4) {
            // Load 4 world_x, world_y values
            float32x4_t vx = vld1q_f32(&world_x[i]);
            float32x4_t vy = vld1q_f32(&world_y[i]);
            float32x4_t vw = vld1q_f32(&width[i]);
            float32x4_t vh = vld1q_f32(&height[i]);

            // Calculate screen coordinates
            float32x4_t vscr_x = vmulq_f32(vsubq_f32(vx, vrect_x), vzoom);
            float32x4_t vscr_y = vmulq_f32(vsubq_f32(vy, vrect_y), vzoom);
            float32x4_t vscr_w = vmulq_f32(vw, vzoom);
            float32x4_t vscr_h = vmulq_f32(vh, vzoom);

            // Store results
            vst1q_f32(&screen_x[i], vscr_x);
            vst1q_f32(&screen_y[i], vscr_y);
            vst1q_f32(&screen_w[i], vscr_w);
            vst1q_f32(&screen_h[i], vscr_h);
        }
        else {
            // Fallback for partial vectors
            for (int j = 0; j < remaining; j++) {
                screen_x[i + j] = (world_x[i + j] - visible_rect_x) * zoom;
                screen_y[i + j] = (world_y[i + j] - visible_rect_y) * zoom;
                screen_w[i + j] = width[i + j] * zoom;
                screen_h[i + j] = height[i + j] * zoom;
            }
        }
    }
#endif
#else
    // Fallback without SIMD
    for (int i = 0; i < count; i++) {
        screen_x[i] = (world_x[i] - visible_rect_x) * zoom;
        screen_y[i] = (world_y[i] - visible_rect_y) * zoom;
        screen_w[i] = width[i] * zoom;
        screen_h[i] = height[i] * zoom;
    }
#endif
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

// Create a new batch for a texture/layer combination
void create_batch(RenderBatch** batches, int* batch_count, int texture_id, int layer) {
    *batch_count = *batch_count + 1;
    *batches = (RenderBatch*)realloc(*batches, (*batch_count) * sizeof(RenderBatch));

    // Initialize the new batch
    int new_idx = *batch_count - 1;
    (*batches)[new_idx].texture_id = texture_id;
    (*batches)[new_idx].layer = layer;
    (*batches)[new_idx].vertex_capacity = 4096;  // Larger initial capacity
    (*batches)[new_idx].index_capacity = 6144;   // Larger initial capacity
    (*batches)[new_idx].vertex_count = 0;
    (*batches)[new_idx].index_count = 0;
    (*batches)[new_idx].vertices = (SDL_Vertex*)SDL_aligned_alloc(
        (*batches)[new_idx].vertex_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE);
    (*batches)[new_idx].indices = (int*)SDL_aligned_alloc(
        (*batches)[new_idx].index_capacity * sizeof(int), CACHE_LINE_SIZE);
}









// Render the game
void render_game(GameState* state) {
    // Clear screen
    SDL_SetRenderDrawColor(state->renderer, 0, 0, 0, 255);
    SDL_RenderClear(state->renderer);

    // Get visible area for culling
    SDL_FRect visible_rect = get_visible_rect(&state->camera);

    // Get reusable buffer for visible entities
    int* visible_indices = (int*)get_buffer(&state->entity_indices_pool);
    int visible_count = 0;

    // Query visible entities from spatial grid using SIMD-based frustum culling
    spatial_grid_query(&state->grid, visible_rect, visible_indices, &visible_count,
        state->entities.count, &state->entities);

    // Sort visible entities by texture and z-order using improved radix sort
    radix_sort_entities(visible_indices, visible_count, &state->entities);

    // Clear batches and reuse them
    for (int i = 0; i < state->batch_count; i++) {
        state->batches[i].vertex_count = 0;
        state->batches[i].index_count = 0;
    }

    // Get reusable buffers for SIMD transformation
    float* screen_coords = (float*)get_buffer(&state->screen_coords_pool);
    float* screen_x = &screen_coords[0];
    float* screen_y = &screen_coords[visible_count];
    float* screen_w = &screen_coords[visible_count * 2];
    float* screen_h = &screen_coords[visible_count * 3];

    // Extract entity data to contiguous arrays for SIMD processing
    float* world_x = (float*)SDL_aligned_alloc(visible_count * sizeof(float), 32); // AVX alignment
    float* world_y = (float*)SDL_aligned_alloc(visible_count * sizeof(float), 32);
    float* width_f = (float*)SDL_aligned_alloc(visible_count * sizeof(float), 32);
    float* height_f = (float*)SDL_aligned_alloc(visible_count * sizeof(float), 32);

    // Copy entity data to contiguous arrays for SIMD processing
    for (int i = 0; i < visible_count; i++) {
        int entity_idx = visible_indices[i];
        world_x[i] = state->entities.x[entity_idx];
        world_y[i] = state->entities.y[entity_idx];
        width_f[i] = (float)state->entities.width[entity_idx];
        height_f[i] = (float)state->entities.height[entity_idx];
    }

    // Calculate screen coordinates using SIMD
    calculate_screen_coordinates(world_x, world_y, width_f, height_f,
        screen_x, screen_y, screen_w, screen_h,
        visible_count, visible_rect.x, visible_rect.y, state->camera.zoom);

    // Free temporary arrays
    SDL_aligned_free(world_x);
    SDL_aligned_free(world_y);
    SDL_aligned_free(width_f);
    SDL_aligned_free(height_f);

    // Track last texture/layer to batch together
    int last_texture_id = -1;
    int last_layer = -1;
    int current_batch_idx = -1;

    // Add visible entities to batches with merged batching
    for (int i = 0; i < visible_count; i++) {
        int entity_idx = visible_indices[i];
        int texture_id = state->entities.texture_id[entity_idx];
        int layer = state->entities.layer[entity_idx];

        // If texture or layer changed, get/create appropriate batch
        if (texture_id != last_texture_id || layer != last_layer) {
            // Find existing batch for this texture/layer
            current_batch_idx = find_batch_index(state->batches, state->batch_count, texture_id, layer);

            // If no batch exists, create one
            if (current_batch_idx == -1) {
                create_batch(&state->batches, &state->batch_count, texture_id, layer);
                current_batch_idx = state->batch_count - 1;
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
        SDL_FRect tex_region = state->atlas.regions[texture_id];

        // Add to appropriate batch
        SDL_FColor color = { 1.0f, 1.0f, 1.0f, 1.0f };
        add_to_batch(&state->batches[current_batch_idx],
            screen_pos_x, screen_pos_y,
            screen_pos_w, screen_pos_h,
            tex_region, color);
    }

    // Render batches
    for (int i = 0; i < state->batch_count; i++) {
        if (state->batches[i].vertex_count > 0) {
            // Single draw call per batch!
            SDL_RenderGeometry(state->renderer, state->atlas.texture,
                state->batches[i].vertices, state->batches[i].vertex_count,
                state->batches[i].indices, state->batches[i].index_count);
        }
    }

    // Return buffers to pools
    return_buffer(&state->entity_indices_pool, visible_indices);
    return_buffer(&state->screen_coords_pool, screen_coords);

    // Show benchmark info
    char fps_text[64];
    sprintf(fps_text, "FPS: %.1f - Entities: %d - Visible: %d",
        state->fps, state->entities.count, visible_count);

    // Would render text here with proper SDL_ttf
}

// Generate random entity for benchmark
void generate_random_entity(GameState* state) {
    float x = (float)(rand() % (int)state->world_bounds.w);
    float y = (float)(rand() % (int)state->world_bounds.h);
    int width = 16 + rand() % 48;
    int height = 16 + rand() % 48;
    int texture_id = rand() % 8;  // Assuming 8 textures
    int layer = rand() % 5;       // 5 layers

    add_entity(state, x, y, width, height, texture_id, layer);
}

// Start benchmark - Emscripten friendly
void start_benchmark(GameState* state, int entity_count, int duration_seconds) {
    ATMLOG("Starting benchmark with target %d entities for %d seconds...",
        entity_count, duration_seconds);

    // Reset entity count
    state->entities.count = 0;

    // Set benchmark parameters
    state->benchmark_running = true;
    state->target_entity_count = entity_count;
    state->current_entity_count = 0;
    state->benchmark_duration_seconds = duration_seconds;
    state->benchmark_start_time = SDL_GetTicks();
    state->last_report_time = state->benchmark_start_time;
    state->total_ms = 0.0;
    state->total_frames = 0;
    state->frames_since_report = 0;
}

// Update benchmark state per frame
void update_benchmark(GameState* state) {
    if (!state->benchmark_running) {
        return;
    }

    // Add entities gradually
    int entities_to_add = state->entities_per_frame;
    if (state->current_entity_count + entities_to_add > state->target_entity_count) {
        entities_to_add = state->target_entity_count - state->current_entity_count;
    }

    for (int i = 0; i < entities_to_add; i++) {
        generate_random_entity(state);
        state->current_entity_count++;
    }

    // Move camera to simulate world exploration
    state->camera.x += 2.0f;
    if (state->camera.x > state->world_bounds.w) {
        state->camera.x = 0.0f;
        state->camera.y += 200.0f;
        if (state->camera.y > state->world_bounds.h) {
            state->camera.y = 0.0f;
        }
    }

    // Time frame execution
    Uint64 frame_start = SDL_GetPerformanceCounter();

    // Update and render happen in main loop

    // Calculate frame time
    Uint64 frame_end = SDL_GetPerformanceCounter();
    double frame_ms = (frame_end - frame_start) * 1000.0 / SDL_GetPerformanceFrequency();

    // Skip first 100 frames (warmup)
    state->total_frames++;
    if (state->total_frames > 100) {
        state->total_ms += frame_ms;
        state->frames_since_report++;
    }

    // Report FPS every second
    Uint64 current_time = SDL_GetTicks();
    if (current_time - state->last_report_time >= 1000) {
        double elapsed_seconds = (current_time - state->last_report_time) / 1000.0;
        double current_fps = state->frames_since_report / elapsed_seconds;

        // Get visible entities count
        SDL_FRect visible_rect = get_visible_rect(&state->camera);
        int* visible_indices = (int*)get_buffer(&state->entity_indices_pool);
        int visible_count = 0;

        spatial_grid_query(&state->grid, visible_rect,
            visible_indices, &visible_count, state->entities.count, &state->entities);

        return_buffer(&state->entity_indices_pool, visible_indices);

        ATMLOG("Entities: %d/%d - FPS: %.2f (%.2f ms/frame) - Visible: %d",
            state->current_entity_count, state->target_entity_count,
            current_fps,
            state->frames_since_report > 0 ? (state->total_ms / state->frames_since_report) : 0,
            visible_count);

        state->frames_since_report = 0;
        state->last_report_time = current_time;
    }

    // Check if benchmark is complete
    if (state->current_entity_count >= state->target_entity_count &&
        current_time - state->benchmark_start_time >= state->benchmark_duration_seconds * 1000) {

        // Calculate final results
        double avg_ms = state->total_ms / (state->total_frames - 100); // Exclude warmup frames
        double avg_fps = 1000.0 / avg_ms;

        ATMLOG("Benchmark Complete:");
        ATMLOG("- Entities: %d", state->current_entity_count);
        ATMLOG("- Duration: %d seconds", (int)((current_time - state->benchmark_start_time) / 1000));
        ATMLOG("- Total Frames: %d", state->total_frames);
        ATMLOG("- Average Frame Time: %.2f ms", avg_ms);
        ATMLOG("- Average FPS: %.2f", avg_fps);

        // End benchmark
        state->benchmark_running = false;
    }
}

// SDL callbacks
SDL_AppResult SDL_AppInit(void** appstate, int argc, char* argv[]) {


    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        SDL_Log("Could not initialize SDL: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    // Create game state with large world
    GameState* state = init_game_state(800, 600, 8000, 8000);
    if (!state) {
        SDL_Log("Failed to initialize game state");
        SDL_Quit();
        return SDL_APP_FAILURE;
    }

    // Create some dummy textures for benchmark
    for (int i = 0; i < 8; i++) {
        // Create a colored surface
        SDL_Surface* surface = SDL_CreateSurface(64, 64, SDL_PIXELFORMAT_RGBA8888);

        // Fill with a color
        Uint32 color = SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format), 0,
            rand() % 255, rand() % 255, rand() % 255, 255);
        SDL_FillSurfaceRect(surface, NULL, color);

        // Add to atlas
        add_texture_to_atlas(state, surface, (i % 4) * 64, (i / 4) * 64);

        // Clean up
        SDL_DestroySurface(surface);
    }

    // Set app state
    *appstate = state;

    // Start benchmark with fewer entities for web
    start_benchmark(state, 200000, 30);

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appstate) {
    GameState* state = (GameState*)appstate;

    // Update benchmark state
    update_benchmark(state);

    // Update game
    update_game(state);

    // Render
    render_game(state);
    SDL_RenderPresent(state->renderer);

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event)
{
    GameState* state = (GameState*)appstate;

    switch (event->type) {
    case SDL_EVENT_QUIT:
        return SDL_APP_FAILURE;

    case SDL_EVENT_KEY_DOWN:
        if (event->key.scancode == SDL_SCANCODE_ESCAPE) {
            return SDL_APP_FAILURE;
        }
        // Camera controls
        else if (event->key.scancode == SDL_SCANCODE_W) {
            state->camera.y -= 100.0f;
        }
        else if (event->key.scancode == SDL_SCANCODE_S) {
            state->camera.y += 100.0f;
        }
        else if (event->key.scancode == SDL_SCANCODE_A) {
            state->camera.x -= 100.0f;
        }
        else if (event->key.scancode == SDL_SCANCODE_D) {
            state->camera.x += 100.0f;
        }
        else if (event->key.scancode == SDL_SCANCODE_Q) {
            state->camera.zoom *= 0.9f;
        }
        else if (event->key.scancode == SDL_SCANCODE_E) {
            state->camera.zoom *= 1.1f;
        }
        // Start benchmark
        else if (event->key.scancode == SDL_SCANCODE_B) {
            if (!state->benchmark_running) {
                start_benchmark(state, 5000, 10);
            }
        }
        break;
    }

    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult result) {
    GameState* state = (GameState*)appstate;

    if (state) {
        free_game_state(state);
    }

    SDL_Quit();
}