#include "ATMEngine.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

// SIMD includes
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HAVE_SIMD 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define HAVE_SIMD 1
#else
#define HAVE_SIMD 0
#endif

// Initialize buffer pool with maximum size limitation
void init_buffer_pool(BufferPool* pool, size_t buffer_size, int initial_capacity) {
    pool->buffer_size = buffer_size;
    pool->capacity = initial_capacity;
    pool->count = 0;
    pool->max_buffers = 64; // Limit maximum buffers to prevent memory bloat
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
        // Pool is full, check if we should resize
        if (pool->capacity < pool->max_buffers) {
            // Resize the pool
            int new_capacity = pool->capacity * 2;
            if (new_capacity > pool->max_buffers) new_capacity = pool->max_buffers;

            void** new_buffers = (void**)SDL_aligned_alloc(new_capacity * sizeof(void*), CACHE_LINE_SIZE);
            memcpy(new_buffers, pool->buffers, pool->count * sizeof(void*));
            SDL_aligned_free(pool->buffers);
            pool->buffers = new_buffers;
            pool->capacity = new_capacity;
        }
        else {
            // We've reached the max pool size, free this buffer instead of storing it
            SDL_aligned_free(buffer);
            return;
        }
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

// Allocate a new entity chunk
EntityChunk* allocate_entity_chunk() {
    EntityChunk* chunk = (EntityChunk*)SDL_malloc(sizeof(EntityChunk));
    chunk->count = 0;

    // Allocate memory for all entity data in this chunk
    chunk->x = (float*)SDL_aligned_alloc(ENTITY_CHUNK_SIZE * sizeof(float), CACHE_LINE_SIZE);
    chunk->y = (float*)SDL_aligned_alloc(ENTITY_CHUNK_SIZE * sizeof(float), CACHE_LINE_SIZE);
    chunk->right = (float*)SDL_aligned_alloc(ENTITY_CHUNK_SIZE * sizeof(float), CACHE_LINE_SIZE);
    chunk->bottom = (float*)SDL_aligned_alloc(ENTITY_CHUNK_SIZE * sizeof(float), CACHE_LINE_SIZE);
    chunk->width = (int*)SDL_aligned_alloc(ENTITY_CHUNK_SIZE * sizeof(int), CACHE_LINE_SIZE);
    chunk->height = (int*)SDL_aligned_alloc(ENTITY_CHUNK_SIZE * sizeof(int), CACHE_LINE_SIZE);
    chunk->texture_id = (int*)SDL_aligned_alloc(ENTITY_CHUNK_SIZE * sizeof(int), CACHE_LINE_SIZE);
    chunk->layer = (int*)SDL_aligned_alloc(ENTITY_CHUNK_SIZE * sizeof(int), CACHE_LINE_SIZE);
    chunk->active = (bool*)SDL_aligned_alloc(ENTITY_CHUNK_SIZE * sizeof(bool), CACHE_LINE_SIZE);
    chunk->grid_cell = (int**)SDL_aligned_alloc(ENTITY_CHUNK_SIZE * sizeof(int*), CACHE_LINE_SIZE);

    return chunk;
}

// Free an entity chunk
void free_entity_chunk(EntityChunk* chunk) {
    if (!chunk) return;

    SDL_aligned_free(chunk->x);
    SDL_aligned_free(chunk->y);
    SDL_aligned_free(chunk->right);
    SDL_aligned_free(chunk->bottom);
    SDL_aligned_free(chunk->width);
    SDL_aligned_free(chunk->height);
    SDL_aligned_free(chunk->texture_id);
    SDL_aligned_free(chunk->layer);
    SDL_aligned_free(chunk->active);
    SDL_aligned_free(chunk->grid_cell);

    SDL_free(chunk);
}

// Initialize chunked entity manager
void init_entity_manager(EntityManager* manager, int initial_chunk_count) {
    manager->chunks_capacity = initial_chunk_count;
    manager->chunk_count = 1; // Start with one chunk
    manager->total_count = 0;

    // Allocate array of chunk pointers
    manager->chunks = (EntityChunk**)SDL_malloc(initial_chunk_count * sizeof(EntityChunk*));

    // Allocate first chunk
    manager->chunks[0] = allocate_entity_chunk();

    // Setup free indices storage for entity recycling
    manager->free_capacity = 1024;
    manager->free_count = 0;
    manager->free_indices = (int*)SDL_malloc(manager->free_capacity * sizeof(int));
}

// Get chunk and local index from global entity index
void get_chunk_indices(int entity_idx, int* chunk_idx, int* local_idx) {
    *chunk_idx = entity_idx / ENTITY_CHUNK_SIZE;
    *local_idx = entity_idx % ENTITY_CHUNK_SIZE;
}

// Free entity manager
void free_entity_manager(EntityManager* manager) {
    for (int i = 0; i < manager->chunk_count; i++) {
        free_entity_chunk(manager->chunks[i]);
    }
    SDL_free(manager->chunks);
    SDL_free(manager->free_indices);

    manager->chunks = NULL;
    manager->free_indices = NULL;
    manager->chunk_count = 0;
    manager->total_count = 0;
}

// Get entity data using global index
float* get_entity_x(EntityManager* manager, int entity_idx) {
    int chunk_idx, local_idx;
    get_chunk_indices(entity_idx, &chunk_idx, &local_idx);
    return &manager->chunks[chunk_idx]->x[local_idx];
}

float* get_entity_y(EntityManager* manager, int entity_idx) {
    int chunk_idx, local_idx;
    get_chunk_indices(entity_idx, &chunk_idx, &local_idx);
    return &manager->chunks[chunk_idx]->y[local_idx];
}

float* get_entity_right(EntityManager* manager, int entity_idx) {
    int chunk_idx, local_idx;
    get_chunk_indices(entity_idx, &chunk_idx, &local_idx);
    return &manager->chunks[chunk_idx]->right[local_idx];
}

float* get_entity_bottom(EntityManager* manager, int entity_idx) {
    int chunk_idx, local_idx;
    get_chunk_indices(entity_idx, &chunk_idx, &local_idx);
    return &manager->chunks[chunk_idx]->bottom[local_idx];
}

int* get_entity_width(EntityManager* manager, int entity_idx) {
    int chunk_idx, local_idx;
    get_chunk_indices(entity_idx, &chunk_idx, &local_idx);
    return &manager->chunks[chunk_idx]->width[local_idx];
}

int* get_entity_height(EntityManager* manager, int entity_idx) {
    int chunk_idx, local_idx;
    get_chunk_indices(entity_idx, &chunk_idx, &local_idx);
    return &manager->chunks[chunk_idx]->height[local_idx];
}

int* get_entity_texture_id(EntityManager* manager, int entity_idx) {
    int chunk_idx, local_idx;
    get_chunk_indices(entity_idx, &chunk_idx, &local_idx);
    return &manager->chunks[chunk_idx]->texture_id[local_idx];
}

int* get_entity_layer(EntityManager* manager, int entity_idx) {
    int chunk_idx, local_idx;
    get_chunk_indices(entity_idx, &chunk_idx, &local_idx);
    return &manager->chunks[chunk_idx]->layer[local_idx];
}

bool* get_entity_active(EntityManager* manager, int entity_idx) {
    int chunk_idx, local_idx;
    get_chunk_indices(entity_idx, &chunk_idx, &local_idx);
    return &manager->chunks[chunk_idx]->active[local_idx];
}

int** get_entity_grid_cell(EntityManager* manager, int entity_idx) {
    int chunk_idx, local_idx;
    get_chunk_indices(entity_idx, &chunk_idx, &local_idx);
    return &manager->chunks[chunk_idx]->grid_cell[local_idx];
}

// Get a new entity index (reuse freed index or allocate new)
int get_new_entity_index(EntityManager* manager) {
    // Check if we have free indices to reuse
    if (manager->free_count > 0) {
        return manager->free_indices[--manager->free_count];
    }

    // No free indices, create a new entity
    int entity_idx = manager->total_count++;

    // Check if we need a new chunk
    int chunk_idx = entity_idx / ENTITY_CHUNK_SIZE;
    if (chunk_idx >= manager->chunk_count) {
        // Need to allocate a new chunk
        if (manager->chunk_count >= manager->chunks_capacity) {
            // Resize chunks array
            int new_capacity = manager->chunks_capacity * 2;
            EntityChunk** new_chunks = (EntityChunk**)SDL_malloc(new_capacity * sizeof(EntityChunk*));
            memcpy(new_chunks, manager->chunks, manager->chunk_count * sizeof(EntityChunk*));
            SDL_free(manager->chunks);
            manager->chunks = new_chunks;
            manager->chunks_capacity = new_capacity;
        }

        // Allocate new chunk
        manager->chunks[manager->chunk_count++] = allocate_entity_chunk();
    }

    // Increment the count in the chunk
    int local_idx = entity_idx % ENTITY_CHUNK_SIZE;
    manager->chunks[chunk_idx]->count = local_idx + 1;

    return entity_idx;
}

// Free an entity index for reuse
void free_entity_index(EntityManager* manager, int entity_idx) {
    // Set entity as inactive
    *get_entity_active(manager, entity_idx) = false;

    // Add to free list
    if (manager->free_count >= manager->free_capacity) {
        // Resize free indices array
        int new_capacity = manager->free_capacity * 2;
        int* new_indices = (int*)SDL_malloc(new_capacity * sizeof(int));
        memcpy(new_indices, manager->free_indices, manager->free_count * sizeof(int));
        SDL_free(manager->free_indices);
        manager->free_indices = new_indices;
        manager->free_capacity = new_capacity;
    }

    manager->free_indices[manager->free_count++] = entity_idx;
}

// Initialize optimized spatial grid for entity queries
void init_spatial_grid(SpatialGrid* grid, float world_width, float world_height, float cell_size) {
    grid->cell_size = cell_size;
    grid->width = (int)ceil(world_width / cell_size);
    grid->height = (int)ceil(world_height / cell_size);

    // Allocate sparse arrays - we only allocate row pointers first
    grid->cells = (int***)SDL_aligned_alloc(grid->height * sizeof(int**), CACHE_LINE_SIZE);
    grid->cell_counts = (int**)SDL_aligned_alloc(grid->height * sizeof(int*), CACHE_LINE_SIZE);
    grid->cell_capacities = (int**)SDL_aligned_alloc(grid->height * sizeof(int*), CACHE_LINE_SIZE);

    // Initialize all rows to NULL (lazy allocation)
    for (int y = 0; y < grid->height; y++) {
        grid->cells[y] = NULL;
        grid->cell_counts[y] = NULL;
        grid->cell_capacities[y] = NULL;
    }
}

// Get or create a cell in the sparse spatial grid
int* get_or_create_cell(SpatialGrid* grid, int grid_x, int grid_y) {
    if (grid_x < 0 || grid_x >= grid->width || grid_y < 0 || grid_y >= grid->height) {
        return NULL; // Out of bounds
    }

    // Check if row is allocated
    if (grid->cells[grid_y] == NULL) {
        // Allocate row
        grid->cells[grid_y] = (int**)SDL_aligned_alloc(grid->width * sizeof(int*), CACHE_LINE_SIZE);
        grid->cell_counts[grid_y] = (int*)SDL_aligned_alloc(grid->width * sizeof(int), CACHE_LINE_SIZE);
        grid->cell_capacities[grid_y] = (int*)SDL_aligned_alloc(grid->width * sizeof(int), CACHE_LINE_SIZE);

        // Initialize all cells in this row to NULL
        for (int x = 0; x < grid->width; x++) {
            grid->cells[grid_y][x] = NULL;
            grid->cell_counts[grid_y][x] = 0;
            grid->cell_capacities[grid_y][x] = 0;
        }
    }

    // Check if cell is allocated
    if (grid->cells[grid_y][grid_x] == NULL) {
        // Allocate cell
        int initial_capacity = 32;
        grid->cells[grid_y][grid_x] = (int*)SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
        grid->cell_counts[grid_y][grid_x] = 0;
        grid->cell_capacities[grid_y][grid_x] = initial_capacity;
    }

    return grid->cells[grid_y][grid_x];
}

// Free spatial grid with sparse storage
void free_spatial_grid(SpatialGrid* grid) {
    for (int y = 0; y < grid->height; y++) {
        if (grid->cells[y]) {
            for (int x = 0; x < grid->width; x++) {
                if (grid->cells[y][x]) {
                    SDL_aligned_free(grid->cells[y][x]);
                }
            }
            SDL_aligned_free(grid->cells[y]);
            SDL_aligned_free(grid->cell_counts[y]);
            SDL_aligned_free(grid->cell_capacities[y]);
        }
    }
    SDL_aligned_free(grid->cells);
    SDL_aligned_free(grid->cell_counts);
    SDL_aligned_free(grid->cell_capacities);
}

// Clear all cells in the spatial grid
void clear_spatial_grid(SpatialGrid* grid) {
    for (int y = 0; y < grid->height; y++) {
        if (grid->cells[y]) {
            for (int x = 0; x < grid->width; x++) {
                if (grid->cells[y][x]) {
                    grid->cell_counts[y][x] = 0;
                }
            }
        }
    }
}

// Add entity to spatial grid
void spatial_grid_add(SpatialGrid* grid, int entity_idx, float x, float y) {
    int grid_x = (int)(x / grid->cell_size);
    int grid_y = (int)(y / grid->cell_size);

    // Clamp to grid bounds
    grid_x = (grid_x < 0) ? 0 : ((grid_x >= grid->width) ? grid->width - 1 : grid_x);
    grid_y = (grid_y < 0) ? 0 : ((grid_y >= grid->height) ? grid->height - 1 : grid_y);

    // Get or create the cell
    int* cell = get_or_create_cell(grid, grid_x, grid_y);
    if (!cell) return; // Out of bounds

    // Ensure capacity
    int cell_count = grid->cell_counts[grid_y][grid_x];
    int cell_capacity = grid->cell_capacities[grid_y][grid_x];

    if (cell_count >= cell_capacity) {
        int new_capacity = cell_capacity * 2;
        int* new_cell = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);

        // Copy existing data
        memcpy(new_cell, cell, cell_count * sizeof(int));

        // Free old cell and update
        SDL_aligned_free(cell);
        grid->cells[grid_y][grid_x] = new_cell;
        grid->cell_capacities[grid_y][grid_x] = new_capacity;
    }

    // Add entity to cell
    grid->cells[grid_y][grid_x][cell_count] = entity_idx;
    grid->cell_counts[grid_y][grid_x]++;
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

        // Using chunked entity access
        float entity_x_values[4] = { 0.0f };
        float entity_y_values[4] = { 0.0f };
        float entity_right_values[4] = { 0.0f };
        float entity_bottom_values[4] = { 0.0f };
        bool entity_active_values[4] = { false };

        // Get values from chunks
        for (int j = 0; j < batch_size; j++) {
            int entity_idx = indices_batch[j];
            int chunk_idx, local_idx;
            get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

            entity_x_values[j] = entities->chunks[chunk_idx]->x[local_idx];
            entity_y_values[j] = entities->chunks[chunk_idx]->y[local_idx];
            entity_right_values[j] = entities->chunks[chunk_idx]->right[local_idx];
            entity_bottom_values[j] = entities->chunks[chunk_idx]->bottom[local_idx];
            entity_active_values[j] = entities->chunks[chunk_idx]->active[local_idx];
        }

        __m128 entity_x = _mm_set_ps(
            batch_size > 3 ? entity_x_values[3] : 0.0f,
            batch_size > 2 ? entity_x_values[2] : 0.0f,
            batch_size > 1 ? entity_x_values[1] : 0.0f,
            entity_x_values[0]
        );

        __m128 entity_y = _mm_set_ps(
            batch_size > 3 ? entity_y_values[3] : 0.0f,
            batch_size > 2 ? entity_y_values[2] : 0.0f,
            batch_size > 1 ? entity_y_values[1] : 0.0f,
            entity_y_values[0]
        );

        __m128 entity_right = _mm_set_ps(
            batch_size > 3 ? entity_right_values[3] : 0.0f,
            batch_size > 2 ? entity_right_values[2] : 0.0f,
            batch_size > 1 ? entity_right_values[1] : 0.0f,
            entity_right_values[0]
        );

        __m128 entity_bottom = _mm_set_ps(
            batch_size > 3 ? entity_bottom_values[3] : 0.0f,
            batch_size > 2 ? entity_bottom_values[2] : 0.0f,
            batch_size > 1 ? entity_bottom_values[1] : 0.0f,
            entity_bottom_values[0]
        );

        // Check if entities are in view
        __m128 test1 = _mm_cmpge_ps(entity_x, view_max_x); // entity_x >= view_max_x
        __m128 test2 = _mm_cmpge_ps(view_min_x, entity_right); // view_min_x >= entity_right
        __m128 test3 = _mm_cmpge_ps(entity_y, view_max_y); // entity_y >= view_max_y
        __m128 test4 = _mm_cmpge_ps(view_min_y, entity_bottom); // view_min_y >= entity_bottom

        // Combine tests: if any test is true, entity is outside view
        __m128 or1 = _mm_or_ps(test1, test2);
        __m128 or2 = _mm_or_ps(test3, test4);
        __m128 outside = _mm_or_ps(or1, or2);

        // Convert to integer mask
        int mask = _mm_movemask_ps(outside);

        // Add visible entities to results
        for (int j = 0; j < batch_size; j++) {
            if (!(mask & (1 << j)) && entity_active_values[j]) {
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

        // Prepare arrays for loading entity coordinates from chunks
        float entity_x_arr[4] = { 0.0f };
        float entity_y_arr[4] = { 0.0f };
        float entity_right_arr[4] = { 0.0f };
        float entity_bottom_arr[4] = { 0.0f };
        bool entity_active_arr[4] = { false };

        // Get values from chunks
        for (int j = 0; j < batch_size; j++) {
            int entity_idx = indices_batch[j];
            int chunk_idx, local_idx;
            get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

            entity_x_arr[j] = entities->chunks[chunk_idx]->x[local_idx];
            entity_y_arr[j] = entities->chunks[chunk_idx]->y[local_idx];
            entity_right_arr[j] = entities->chunks[chunk_idx]->right[local_idx];
            entity_bottom_arr[j] = entities->chunks[chunk_idx]->bottom[local_idx];
            entity_active_arr[j] = entities->chunks[chunk_idx]->active[local_idx];
        }

        float32x4_t entity_x = vld1q_f32(entity_x_arr);
        float32x4_t entity_y = vld1q_f32(entity_y_arr);
        float32x4_t entity_right = vld1q_f32(entity_right_arr);
        float32x4_t entity_bottom = vld1q_f32(entity_bottom_arr);

        // Check if entities are in view
        uint32x4_t test1 = vcgeq_f32(entity_x, view_max_x); // entity_x >= view_max_x
        uint32x4_t test2 = vcgeq_f32(view_min_x, entity_right); // view_min_x >= entity_right
        uint32x4_t test3 = vcgeq_f32(entity_y, view_max_y); // entity_y >= view_max_y
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
            if (!mask[j] && entity_active_arr[j]) {
                result_indices[result_count++] = indices_batch[j];
            }
        }
    }
#endif
#else
    // Fallback without SIMD
    for (int i = 0; i < cell_entity_count; i++) {
        int entity_idx = cell_entities[i];
        int chunk_idx, local_idx;
        get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

        bool active = entities->chunks[chunk_idx]->active[local_idx];

        if (active) {
            float x = entities->chunks[chunk_idx]->x[local_idx];
            float y = entities->chunks[chunk_idx]->y[local_idx];
            float right = entities->chunks[chunk_idx]->right[local_idx];
            float bottom = entities->chunks[chunk_idx]->bottom[local_idx];

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

// Query entities in a region using SIMD-based frustum culling with sparse grid
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
        // Skip rows that aren't allocated
        if (!grid->cells[y]) continue;

        for (int x = start_x; x <= end_x && *result_count < max_results; x++) {
            // Skip cells that aren't allocated
            if (!grid->cells[y][x]) continue;

            int cell_entity_count = grid->cell_counts[y][x];
            if (cell_entity_count == 0) continue;

            // Use SIMD-based frustum culling
            int visible_count = simd_frustum_cull(entities, query_rect,
                grid->cells[y][x], cell_entity_count,
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
        int chunk_idx, local_idx;
        get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

        int texture_id = entities->chunks[chunk_idx]->texture_id[local_idx] & RADIX_MASK;
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
        int chunk_idx, local_idx;
        get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

        int texture_id = entities->chunks[chunk_idx]->texture_id[local_idx] & RADIX_MASK;
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
        int chunk_idx, local_idx;
        get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

        int texture_id = entities->chunks[chunk_idx]->texture_id[local_idx] & RADIX_MASK;
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
            int chunk_idx, local_idx;
            get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

            int layer = entities->chunks[chunk_idx]->layer[local_idx] & RADIX_MASK;
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
            int chunk_idx, local_idx;
            get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

            int layer = entities->chunks[chunk_idx]->layer[local_idx] & RADIX_MASK;
            temp_indices[start + count_layer[layer]++] = entity_idx;
        }

        // Copy back to original array just for this texture group
        memcpy(&indices[start], &temp_indices[start], size * sizeof(int));
    }

    // Cleanup
    SDL_aligned_free(temp_indices);
}

// Initialize texture atlas (unchanged)
void init_texture_atlas(TextureAtlas* atlas, SDL_Renderer* renderer, int width, int height) {
    atlas->texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_TARGET, width, height);
    SDL_SetTextureBlendMode(atlas->texture, SDL_BLENDMODE_BLEND);
    atlas->region_capacity = 32;
    atlas->region_count = 0;
    atlas->regions = (SDL_FRect*)SDL_aligned_alloc(atlas->region_capacity * sizeof(SDL_FRect), CACHE_LINE_SIZE);
}

// Get visible rect based on camera (unchanged)
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

// Add a quad to a render batch (unchanged)
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

// Find batch index for texture/layer combination (unchanged)
int find_batch_index(RenderBatch* batches, int batch_count, int texture_id, int layer) {
    for (int i = 0; i < batch_count; i++) {
        if (batches[i].texture_id == texture_id && batches[i].layer == layer) {
            return i;
        }
    }
    return -1; // Not found
}

// Create a new batch for a texture/layer combination (unchanged)
void create_batch(RenderBatch** batches, int* batch_count, int texture_id, int layer) {
    *batch_count = *batch_count + 1;
    *batches = (RenderBatch*)realloc(*batches, (*batch_count) * sizeof(RenderBatch));

    // Initialize the new batch
    int new_idx = *batch_count - 1;
    (*batches)[new_idx].texture_id = texture_id;
    (*batches)[new_idx].layer = layer;
    (*batches)[new_idx].vertex_capacity = 4096; // Larger initial capacity
    (*batches)[new_idx].index_capacity = 6144; // Larger initial capacity
    (*batches)[new_idx].vertex_count = 0;
    (*batches)[new_idx].index_count = 0;
    (*batches)[new_idx].vertices = (SDL_Vertex*)SDL_aligned_alloc(
        (*batches)[new_idx].vertex_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE);
    (*batches)[new_idx].indices = (int*)SDL_aligned_alloc(
        (*batches)[new_idx].index_capacity * sizeof(int), CACHE_LINE_SIZE);
}

// Calculate screen coordinates using SIMD for chunked entities
void calculate_screen_coordinates(EntityManager* entities, int* entity_indices,
    float* screen_x, float* screen_y, float* screen_w, float* screen_h,
    int count, float visible_rect_x, float visible_rect_y, float zoom) {
#if HAVE_SIMD
    // Process multiple entities at once with SIMD
#if defined(__x86_64__) || defined(_M_X64)
    // SSE implementation for x86
    __m128 vrect_x = _mm_set1_ps(visible_rect_x);
    __m128 vrect_y = _mm_set1_ps(visible_rect_y);
    __m128 vzoom = _mm_set1_ps(zoom);

    for (int i = 0; i < count; i += 4) {
        int remaining = (count - i < 4) ? count - i : 4;

        // Prepare arrays for loading entity data
        float world_x_arr[4] = { 0.0f };
        float world_y_arr[4] = { 0.0f };
        float width_arr[4] = { 0.0f };
        float height_arr[4] = { 0.0f };

        // Get data from entity chunks
        for (int j = 0; j < remaining; j++) {
            int entity_idx = entity_indices[i + j];
            int chunk_idx, local_idx;
            get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

            world_x_arr[j] = entities->chunks[chunk_idx]->x[local_idx];
            world_y_arr[j] = entities->chunks[chunk_idx]->y[local_idx];
            width_arr[j] = (float)entities->chunks[chunk_idx]->width[local_idx];
            height_arr[j] = (float)entities->chunks[chunk_idx]->height[local_idx];
        }

        if (remaining == 4) {
            // Load values into SIMD registers
            __m128 vx = _mm_loadu_ps(world_x_arr);
            __m128 vy = _mm_loadu_ps(world_y_arr);
            __m128 vw = _mm_loadu_ps(width_arr);
            __m128 vh = _mm_loadu_ps(height_arr);

            // Calculate screen coordinates
            __m128 vscr_x = _mm_mul_ps(_mm_sub_ps(vx, vrect_x), vzoom);
            __m128 vscr_y = _mm_mul_ps(_mm_sub_ps(vy, vrect_y), vzoom);
            __m128 vscr_w = _mm_mul_ps(vw, vzoom);
            __m128 vscr_h = _mm_mul_ps(vh, vzoom);

            // Store results
            _mm_storeu_ps(&screen_x[i], vscr_x);
            _mm_storeu_ps(&screen_y[i], vscr_y);
            _mm_storeu_ps(&screen_w[i], vscr_w);
            _mm_storeu_ps(&screen_h[i], vscr_h);
        }
        else {
            // Fallback for partial vectors
            for (int j = 0; j < remaining; j++) {
                screen_x[i + j] = (world_x_arr[j] - visible_rect_x) * zoom;
                screen_y[i + j] = (world_y_arr[j] - visible_rect_y) * zoom;
                screen_w[i + j] = width_arr[j] * zoom;
                screen_h[i + j] = height_arr[j] * zoom;
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

        // Prepare arrays for loading entity data
        float world_x_arr[4] = { 0.0f };
        float world_y_arr[4] = { 0.0f };
        float width_arr[4] = { 0.0f };
        float height_arr[4] = { 0.0f };

        // Get data from entity chunks
        for (int j = 0; j < remaining; j++) {
            int entity_idx = entity_indices[i + j];
            int chunk_idx, local_idx;
            get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

            world_x_arr[j] = entities->chunks[chunk_idx]->x[local_idx];
            world_y_arr[j] = entities->chunks[chunk_idx]->y[local_idx];
            width_arr[j] = (float)entities->chunks[chunk_idx]->width[local_idx];
            height_arr[j] = (float)entities->chunks[chunk_idx]->height[local_idx];
        }

        if (remaining == 4) {
            // Load values into SIMD registers
            float32x4_t vx = vld1q_f32(world_x_arr);
            float32x4_t vy = vld1q_f32(world_y_arr);
            float32x4_t vw = vld1q_f32(width_arr);
            float32x4_t vh = vld1q_f32(height_arr);

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
                screen_x[i + j] = (world_x_arr[j] - visible_rect_x) * zoom;
                screen_y[i + j] = (world_y_arr[j] - visible_rect_y) * zoom;
                screen_w[i + j] = width_arr[j] * zoom;
                screen_h[i + j] = height_arr[j] * zoom;
            }
        }
    }
#endif
#else
    // Fallback without SIMD
    for (int i = 0; i < count; i++) {
        int entity_idx = entity_indices[i];
        int chunk_idx, local_idx;
        get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

        float world_x = entities->chunks[chunk_idx]->x[local_idx];
        float world_y = entities->chunks[chunk_idx]->y[local_idx];
        float width = (float)entities->chunks[chunk_idx]->width[local_idx];
        float height = (float)entities->chunks[chunk_idx]->height[local_idx];

        screen_x[i] = (world_x - visible_rect_x) * zoom;
        screen_y[i] = (world_y - visible_rect_y) * zoom;
        screen_w[i] = width * zoom;
        screen_h[i] = height * zoom;
    }
#endif
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
    start_x = (start_x < 0) ? 0 : ((start_x >= engine->grid_width) ? engine->grid_width - 1 : start_x);
    start_y = (start_y < 0) ? 0 : ((start_y >= engine->grid_height) ? engine->grid_height - 1 : start_y);
    end_x = (end_x < 0) ? 0 : ((end_x >= engine->grid_width) ? engine->grid_width - 1 : end_x);
    end_y = (end_y < 0) ? 0 : ((end_y >= engine->grid_height) ? engine->grid_height - 1 : end_y);

    // First unload all cells
    for (int x = 0; x < engine->grid_width; x++) {
        for (int y = 0; y < engine->grid_height; y++) {
            engine->grid_loaded[x][y] = false;
        }
    }

    // Then load visible cells
    for (int x = start_x; x <= end_x; x++) {
        for (int y = start_y; y <= end_y; y++) {
            engine->grid_loaded[x][y] = true;
        }
    }

    // Update entity active states based on grid loading
    for (int c = 0; c < engine->entities.chunk_count; c++) {
        EntityChunk* chunk = engine->entities.chunks[c];
        for (int i = 0; i < chunk->count; i++) {
            float entity_x = chunk->x[i];
            float entity_y = chunk->y[i];

            int grid_x = (int)(entity_x / engine->grid_cell_size);
            int grid_y = (int)(entity_y / engine->grid_cell_size);

            if (grid_x >= 0 && grid_x < engine->grid_width &&
                grid_y >= 0 && grid_y < engine->grid_height) {
                chunk->active[i] = engine->grid_loaded[grid_x][grid_y];
            }
            else {
                chunk->active[i] = false;
            }
        }
    }
}

// Engine API implementations
Engine* engine_create(int window_width, int window_height, int world_width, int world_height, int cell_size) {
    Engine* engine = (Engine*)SDL_malloc(sizeof(Engine));
    if (!engine) return NULL;
    memset(engine, 0, sizeof(Engine));

    // Create window and renderer
    engine->window = SDL_CreateWindow("2D Game Engine - SDL3", window_width, window_height, 0);
    if (!engine->window) {
        free(engine);
        return NULL;
    }

    engine->renderer = SDL_CreateRenderer(engine->window, NULL);
    if (!engine->renderer) {
        SDL_DestroyWindow(engine->window);
        free(engine);
        return NULL;
    }

    // Init entity manager (Chunked SoA design)
    init_entity_manager(&engine->entities, 4); // Start with few chunks, expand as needed

    // Init world bounds
    engine->world_bounds.x = 0;
    engine->world_bounds.y = 0;
    engine->world_bounds.w = world_width;
    engine->world_bounds.h = world_height;

    // Init spatial grid (optimized sparse implementation)
    init_spatial_grid(&engine->grid, world_width, world_height, cell_size);

    // Init texture atlas
    init_texture_atlas(&engine->atlas, engine->renderer, 2048, 2048);

    // Init camera
    engine->camera.x = 0;
    engine->camera.y = 0;
    engine->camera.width = window_width;
    engine->camera.height = window_height;
    engine->camera.zoom = 1.0f;

    // Init render batches (for batch rendering) with larger initial capacities
    engine->batch_count = 8;
    engine->batches = (RenderBatch*)SDL_malloc(engine->batch_count * sizeof(RenderBatch));
    for (int i = 0; i < engine->batch_count; i++) {
        engine->batches[i].texture_id = i;
        engine->batches[i].layer = 0;
        engine->batches[i].vertex_capacity = 4096; // Increased capacity
        engine->batches[i].index_capacity = 6144; // Increased capacity
        engine->batches[i].vertex_count = 0;
        engine->batches[i].index_count = 0;
        engine->batches[i].vertices = (SDL_Vertex*)SDL_aligned_alloc(
            engine->batches[i].vertex_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE);
        engine->batches[i].indices = (int*)SDL_aligned_alloc(
            engine->batches[i].index_capacity * sizeof(int), CACHE_LINE_SIZE);
    }

    // Init dynamic grid loading
    engine->grid_cell_size = 256.0f;
    engine->grid_width = (int)ceil(world_width / engine->grid_cell_size);
    engine->grid_height = (int)ceil(world_height / engine->grid_cell_size);

    // Allocate grid loaded array
    engine->grid_loaded = (bool**)SDL_malloc(engine->grid_width * sizeof(bool*));
    for (int x = 0; x < engine->grid_width; x++) {
        engine->grid_loaded[x] = (bool*)SDL_malloc(engine->grid_height * sizeof(bool));
        for (int y = 0; y < engine->grid_height; y++) {
            engine->grid_loaded[x][y] = false;
        }
    }

    // Initialize buffer pools for reuse with size limits
    init_buffer_pool(&engine->entity_indices_pool, 65536 * sizeof(int), 4);  // Use smaller pools
    init_buffer_pool(&engine->screen_coords_pool, 65536 * sizeof(float) * 4, 4);

    // Init timing
    engine->last_frame_time = SDL_GetTicks();
    engine->fps = 0.0f;

    return engine;
}

void engine_destroy(Engine* engine) {
    if (!engine) return;

    // Free entity manager (chunked implementation)
    free_entity_manager(&engine->entities);

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

    // Free buffer pools
    free_buffer_pool(&engine->entity_indices_pool);
    free_buffer_pool(&engine->screen_coords_pool);

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

    // Rebuild spatial grid with active entities using chunked access
    for (int c = 0; c < engine->entities.chunk_count; c++) {
        EntityChunk* chunk = engine->entities.chunks[c];
        for (int i = 0; i < chunk->count; i++) {
            if (chunk->active[i]) {
                int entity_idx = c * ENTITY_CHUNK_SIZE + i;
                spatial_grid_add(&engine->grid, entity_idx, chunk->x[i], chunk->y[i]);
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

    // Get reusable buffer for visible entities - use a reasonable size for efficiency
    int max_visible = 65536; // Maximum entities to process at once
    int* visible_indices = (int*)get_buffer(&engine->entity_indices_pool);
    int visible_count = 0;

    // Query visible entities from spatial grid using SIMD-based frustum culling
    spatial_grid_query(&engine->grid, visible_rect, visible_indices, &visible_count,
        max_visible, &engine->entities);

    // Sort visible entities by texture and z-order using improved radix sort
    radix_sort_entities(visible_indices, visible_count, &engine->entities);

    // Clear batches and reuse them
    for (int i = 0; i < engine->batch_count; i++) {
        engine->batches[i].vertex_count = 0;
        engine->batches[i].index_count = 0;
    }

    // Get reusable buffers for SIMD transformation
    float* screen_coords = (float*)get_buffer(&engine->screen_coords_pool);
    float* screen_x = &screen_coords[0];
    float* screen_y = &screen_coords[max_visible];
    float* screen_w = &screen_coords[max_visible * 2];
    float* screen_h = &screen_coords[max_visible * 3];

    // Calculate screen coordinates using SIMD for chunked entity data
    calculate_screen_coordinates(&engine->entities, visible_indices,
        screen_x, screen_y, screen_w, screen_h,
        visible_count, visible_rect.x, visible_rect.y, engine->camera.zoom);

    // Track last texture/layer to batch together
    int last_texture_id = -1;
    int last_layer = -1;
    int current_batch_idx = -1;

    // Add visible entities to batches with merged batching
    for (int i = 0; i < visible_count; i++) {
        int entity_idx = visible_indices[i];
        int chunk_idx, local_idx;
        get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

        int texture_id = engine->entities.chunks[chunk_idx]->texture_id[local_idx];
        int layer = engine->entities.chunks[chunk_idx]->layer[local_idx];

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

    // Return buffers to pools
    return_buffer(&engine->entity_indices_pool, visible_indices);
    return_buffer(&engine->screen_coords_pool, screen_coords);

    // Present renderer
    SDL_RenderPresent(engine->renderer);
}

int engine_add_entity(Engine* engine, float x, float y, int width, int height,
    int texture_id, int layer) {
    // Get a new entity index from the entity manager
    int entity_idx = get_new_entity_index(&engine->entities);

    // Get chunk and local indices
    int chunk_idx, local_idx;
    get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

    // Set entity properties in the chunk
    engine->entities.chunks[chunk_idx]->x[local_idx] = x;
    engine->entities.chunks[chunk_idx]->y[local_idx] = y;
    engine->entities.chunks[chunk_idx]->width[local_idx] = width;
    engine->entities.chunks[chunk_idx]->height[local_idx] = height;

    // Precompute right and bottom for faster culling
    engine->entities.chunks[chunk_idx]->right[local_idx] = x + width;
    engine->entities.chunks[chunk_idx]->bottom[local_idx] = y + height;

    engine->entities.chunks[chunk_idx]->texture_id[local_idx] = texture_id;
    engine->entities.chunks[chunk_idx]->layer[local_idx] = layer;

    // Determine if entity should be active based on grid loading
    int grid_x = (int)(x / engine->grid_cell_size);
    int grid_y = (int)(y / engine->grid_cell_size);

    if (grid_x >= 0 && grid_x < engine->grid_width &&
        grid_y >= 0 && grid_y < engine->grid_height) {
        engine->entities.chunks[chunk_idx]->active[local_idx] = engine->grid_loaded[grid_x][grid_y];
    }
    else {
        engine->entities.chunks[chunk_idx]->active[local_idx] = false;
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
        SDL_FRect* new_regions = (SDL_FRect*)SDL_aligned_alloc(
            new_capacity * sizeof(SDL_FRect), CACHE_LINE_SIZE);

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
    if (entity_idx < 0) return;

    // Get chunk and local indices
    int chunk_idx, local_idx;
    get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

    // Check if valid entity
    if (chunk_idx >= engine->entities.chunk_count) return;
    if (local_idx >= engine->entities.chunks[chunk_idx]->count) return;

    // Update position
    engine->entities.chunks[chunk_idx]->x[local_idx] = x;
    engine->entities.chunks[chunk_idx]->y[local_idx] = y;

    // Update precomputed values
    engine->entities.chunks[chunk_idx]->right[local_idx] = x + engine->entities.chunks[chunk_idx]->width[local_idx];
    engine->entities.chunks[chunk_idx]->bottom[local_idx] = y + engine->entities.chunks[chunk_idx]->height[local_idx];
}

void engine_set_entity_active(Engine* engine, int entity_idx, bool active) {
    if (entity_idx < 0) return;

    // Get chunk and local indices
    int chunk_idx, local_idx;
    get_chunk_indices(entity_idx, &chunk_idx, &local_idx);

    // Check if valid entity
    if (chunk_idx >= engine->entities.chunk_count) return;
    if (local_idx >= engine->entities.chunks[chunk_idx]->count) return;

    // Update active state
    engine->entities.chunks[chunk_idx]->active[local_idx] = active;
}

SDL_FRect engine_get_visible_rect(Engine* engine) {
    return get_visible_rect(&engine->camera);
}

int engine_get_visible_entities_count(Engine* engine) {
    SDL_FRect visible_rect = get_visible_rect(&engine->camera);
    int* visible_indices = (int*)get_buffer(&engine->entity_indices_pool);
    int visible_count = 0;
    int max_visible = 65536; // Set a reasonable maximum

    spatial_grid_query(&engine->grid, visible_rect, visible_indices, &visible_count,
        max_visible, &engine->entities);

    return_buffer(&engine->entity_indices_pool, visible_indices);
    return visible_count;
}