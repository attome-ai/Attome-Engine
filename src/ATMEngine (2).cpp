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
        void** new_buffers = (void**)SDL_aligned_alloc(new_capacity * sizeof(void*), CACHE_LINE_SIZE);
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
    (*batches)[new_idx].vertex_capacity = 4096; // Larger initial capacity
    (*batches)[new_idx].index_capacity = 6144; // Larger initial capacity
    (*batches)[new_idx].vertex_count = 0;
    (*batches)[new_idx].index_count = 0;
    (*batches)[new_idx].vertices = (SDL_Vertex*)SDL_aligned_alloc(
        (*batches)[new_idx].vertex_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE);
    (*batches)[new_idx].indices = (int*)SDL_aligned_alloc(
        (*batches)[new_idx].index_capacity * sizeof(int), CACHE_LINE_SIZE);
}

// Calculate screen coordinates using SIMD if available
void calculate_screen_coordinates(float* world_x, float* world_y, float* width, float* height,
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
    for (int i = 0; i < engine->entities.count; i++) {
        int grid_x = (int)(engine->entities.x[i] / engine->grid_cell_size);
        int grid_y = (int)(engine->entities.y[i] / engine->grid_cell_size);

        if (grid_x >= 0 && grid_x < engine->grid_width &&
            grid_y >= 0 && grid_y < engine->grid_height) {
            engine->entities.active[i] = engine->grid_loaded[grid_x][grid_y];
        }
        else {
            engine->entities.active[i] = false;
        }
    }
}

// Modified entity manager initialization to support hierarchies
void init_entity_manager(EntityManager* manager, int initial_capacity) {
    manager->capacity = initial_capacity;
    manager->count = 0;

    // Allocate aligned arrays for better SIMD performance (original fields)
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

    // New fields for entity composition
    manager->local_x = (float*)SDL_aligned_alloc(initial_capacity * sizeof(float), CACHE_LINE_SIZE);
    manager->local_y = (float*)SDL_aligned_alloc(initial_capacity * sizeof(float), CACHE_LINE_SIZE);
    manager->scale_x = (float*)SDL_aligned_alloc(initial_capacity * sizeof(float), CACHE_LINE_SIZE);
    manager->scale_y = (float*)SDL_aligned_alloc(initial_capacity * sizeof(float), CACHE_LINE_SIZE);
    manager->rotation = (float*)SDL_aligned_alloc(initial_capacity * sizeof(float), CACHE_LINE_SIZE);
    manager->parent = (int*)SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->first_child = (int*)SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->next_sibling = (int*)SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->child_count = (int*)SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->local_transform = (Transform*)SDL_aligned_alloc(initial_capacity * sizeof(Transform), CACHE_LINE_SIZE);
    manager->world_transform = (Transform*)SDL_aligned_alloc(initial_capacity * sizeof(Transform), CACHE_LINE_SIZE);
    manager->transform_dirty = (bool*)SDL_aligned_alloc(initial_capacity * sizeof(bool), CACHE_LINE_SIZE);

    // Initialize all entities with default values
    for (int i = 0; i < initial_capacity; i++) {
        manager->local_x[i] = 0.0f;
        manager->local_y[i] = 0.0f;
        manager->scale_x[i] = 1.0f;
        manager->scale_y[i] = 1.0f;
        manager->rotation[i] = 0.0f;
        manager->parent[i] = -1;
        manager->first_child[i] = -1;
        manager->next_sibling[i] = -1;
        manager->child_count[i] = 0;
        transform_identity(&manager->local_transform[i]);
        transform_identity(&manager->world_transform[i]);
        manager->transform_dirty[i] = true;
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

    // Init entity manager (SoA design)
    init_entity_manager(&engine->entities, 100000); // Increased initial capacity

    // Init world bounds
    engine->world_bounds.x = 0;
    engine->world_bounds.y = 0;
    engine->world_bounds.w = world_width;
    engine->world_bounds.h = world_height;

    // Init spatial grid (replaces quadtree) - smaller cell size for better performance
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

    // Initialize buffer pools for reuse
    init_buffer_pool(&engine->entity_indices_pool, 100000 * sizeof(int), 4);
    init_buffer_pool(&engine->screen_coords_pool, 100000 * sizeof(float) * 4, 4);

    // Init timing
    engine->last_frame_time = SDL_GetTicks();
    engine->fps = 0.0f;

    return engine;
}
// Modified free_entity_manager to release new arrays
void free_entity_manager(EntityManager* manager) {
    // Free original arrays
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

    // Free new arrays for entity composition
    SDL_aligned_free(manager->local_x);
    SDL_aligned_free(manager->local_y);
    SDL_aligned_free(manager->scale_x);
    SDL_aligned_free(manager->scale_y);
    SDL_aligned_free(manager->rotation);
    SDL_aligned_free(manager->parent);
    SDL_aligned_free(manager->first_child);
    SDL_aligned_free(manager->next_sibling);
    SDL_aligned_free(manager->child_count);
    SDL_aligned_free(manager->local_transform);
    SDL_aligned_free(manager->world_transform);
    SDL_aligned_free(manager->transform_dirty);
}

void engine_destroy(Engine* engine) {
    if (!engine) return;

    // Free entity manager
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


void engine_set_entity_active(Engine* engine, int entity_idx, bool active) {
    if (entity_idx >= 0 && entity_idx < engine->entities.count) {
        engine->entities.active[entity_idx] = active;
    }
}

SDL_FRect engine_get_visible_rect(Engine* engine) {
    return get_visible_rect(&engine->camera);
}

int engine_get_visible_entities_count(Engine* engine) {
    SDL_FRect visible_rect = get_visible_rect(&engine->camera);
    int* visible_indices = (int*)get_buffer(&engine->entity_indices_pool);
    int visible_count = 0;

    spatial_grid_query(&engine->grid, visible_rect, visible_indices, &visible_count,
        engine->entities.count, &engine->entities);

    return_buffer(&engine->entity_indices_pool, visible_indices);
    return visible_count;
}



// Initialize transform to identity matrix
void transform_identity(Transform* transform) {
    transform->m[0] = 1.0f; // a
    transform->m[1] = 0.0f; // c
    transform->m[2] = 0.0f; // e (x translation)
    transform->m[3] = 0.0f; // b
    transform->m[4] = 1.0f; // d
    transform->m[5] = 0.0f; // f (y translation)
}

// Set transform values directly
void transform_set(Transform* transform, float x, float y, float rotation, float scale_x, float scale_y) {
    float cos_r = cosf(rotation);
    float sin_r = sinf(rotation);

    transform->m[0] = cos_r * scale_x; // a
    transform->m[1] = sin_r * scale_x; // c
    transform->m[2] = x;               // e (x translation)
    transform->m[3] = -sin_r * scale_y; // b
    transform->m[4] = cos_r * scale_y;  // d
    transform->m[5] = y;                // f (y translation)
}

// Combine two transforms (matrix multiplication: result = a * b)
void transform_combine(Transform* result, const Transform* a, const Transform* b) {
    float a0 = a->m[0], a1 = a->m[1], a2 = a->m[2];
    float a3 = a->m[3], a4 = a->m[4], a5 = a->m[5];

    float b0 = b->m[0], b1 = b->m[1], b2 = b->m[2];
    float b3 = b->m[3], b4 = b->m[4], b5 = b->m[5];

    result->m[0] = a0 * b0 + a1 * b3;
    result->m[1] = a0 * b1 + a1 * b4;
    result->m[2] = a0 * b2 + a1 * b5 + a2;
    result->m[3] = a3 * b0 + a4 * b3;
    result->m[4] = a3 * b1 + a4 * b4;
    result->m[5] = a3 * b2 + a4 * b5 + a5;
}

// Apply transform to a point
void transform_apply(const Transform* transform, float x, float y, float* out_x, float* out_y) {
    *out_x = transform->m[0] * x + transform->m[1] * y + transform->m[2];
    *out_y = transform->m[3] * x + transform->m[4] * y + transform->m[5];
}

// SIMD-optimized batch transform application
void transform_batch_apply(const Transform* transform, float* x, float* y, float* out_x, float* out_y, int count) {
#if HAVE_SIMD
#if defined(x86_64) || defined(_M_X64)
    // SSE implementation for x86
    __m128 m0 = _mm_set1_ps(transform->m[0]);
    __m128 m1 = _mm_set1_ps(transform->m[1]);
    __m128 m2 = _mm_set1_ps(transform->m[2]);
    __m128 m3 = _mm_set1_ps(transform->m[3]);
    __m128 m4 = _mm_set1_ps(transform->m[4]);
    __m128 m5 = _mm_set1_ps(transform->m[5]);

    for (int i = 0; i < count; i += 4) {
        int remaining = (count - i < 4) ? count - i : 4;

        if (remaining == 4) {
            // Load 4 points
            __m128 vx = _mm_load_ps(&x[i]);
            __m128 vy = _mm_load_ps(&y[i]);

            // Calculate transformed points
            __m128 tx = _mm_add_ps(_mm_add_ps(_mm_mul_ps(m0, vx), _mm_mul_ps(m1, vy)), m2);
            __m128 ty = _mm_add_ps(_mm_add_ps(_mm_mul_ps(m3, vx), _mm_mul_ps(m4, vy)), m5);

            // Store results
            _mm_store_ps(&out_x[i], tx);
            _mm_store_ps(&out_y[i], ty);
        }
        else {
            // Fallback for partial vectors
            for (int j = 0; j < remaining; j++) {
                transform_apply(transform, x[i + j], y[i + j], &out_x[i + j], &out_y[i + j]);
            }
        }
    }
#elif defined(__ARM_NEON)
    // NEON implementation for ARM
    float32x4_t m0 = vdupq_n_f32(transform->m[0]);
    float32x4_t m1 = vdupq_n_f32(transform->m[1]);
    float32x4_t m2 = vdupq_n_f32(transform->m[2]);
    float32x4_t m3 = vdupq_n_f32(transform->m[3]);
    float32x4_t m4 = vdupq_n_f32(transform->m[4]);
    float32x4_t m5 = vdupq_n_f32(transform->m[5]);

    for (int i = 0; i < count; i += 4) {
        int remaining = (count - i < 4) ? count - i : 4;

        if (remaining == 4) {
            // Load 4 points
            float32x4_t vx = vld1q_f32(&x[i]);
            float32x4_t vy = vld1q_f32(&y[i]);

            // Calculate transformed points
            float32x4_t tx = vaddq_f32(vaddq_f32(vmulq_f32(m0, vx), vmulq_f32(m1, vy)), m2);
            float32x4_t ty = vaddq_f32(vaddq_f32(vmulq_f32(m3, vx), vmulq_f32(m4, vy)), m5);

            // Store results
            vst1q_f32(&out_x[i], tx);
            vst1q_f32(&out_y[i], ty);
        }
        else {
            // Fallback for partial vectors
            for (int j = 0; j < remaining; j++) {
                transform_apply(transform, x[i + j], y[i + j], &out_x[i + j], &out_y[i + j]);
            }
        }
    }
#endif
#else
    // Fallback without SIMD
    for (int i = 0; i < count; i++) {
        transform_apply(transform, x[i], y[i], &out_x[i], &out_y[i]);
    }
#endif
}





// Add a child entity to a parent entity
int engine_add_child_entity(Engine* engine, int parent_idx, float local_x, float local_y, int width, int height, int texture_id, int layer) {
    EntityManager* em = &engine->entities;

    // Create the entity with default world position (will be updated by transform hierarchy)
    int entity_idx = engine_add_entity(engine, 0, 0, width, height, texture_id, layer);

    // Set local transform
    em->local_x[entity_idx] = local_x;
    em->local_y[entity_idx] = local_y;
    em->scale_x[entity_idx] = 1.0f;
    em->scale_y[entity_idx] = 1.0f;
    em->rotation[entity_idx] = 0.0f;

    // Configure local transform matrix
    transform_set(&em->local_transform[entity_idx], local_x, local_y, 0, 1.0f, 1.0f);
    em->transform_dirty[entity_idx] = true;

    // Set parent-child relationship
    engine_set_parent(engine, entity_idx, parent_idx);

    return entity_idx;
}

// Set an entity's parent
void engine_set_parent(Engine* engine, int entity_idx, int parent_idx) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) return;
    if (parent_idx >= engine->entities.count) return;

    EntityManager* em = &engine->entities;

    // If the entity already has a parent, remove it from that parent's children list
    if (em->parent[entity_idx] != -1) {
        engine_remove_parent(engine, entity_idx);
    }

    // Set the new parent
    em->parent[entity_idx] = parent_idx;

    if (parent_idx >= 0) {
        // Add to parent's children list (at the beginning for efficiency)
        int old_first_child = em->first_child[parent_idx];
        em->first_child[parent_idx] = entity_idx;
        em->next_sibling[entity_idx] = old_first_child;
        em->child_count[parent_idx]++;

        // Mark transform as dirty
        em->transform_dirty[entity_idx] = true;
    }

    // Update the world transform immediately
    engine_update_entity_transform(engine, entity_idx);
}

// Remove parent-child relationship
void engine_remove_parent(Engine* engine, int entity_idx) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) return;

    EntityManager* em = &engine->entities;
    int parent_idx = em->parent[entity_idx];

    if (parent_idx == -1) return; // No parent to remove

    // Remove from parent's children list
    if (em->first_child[parent_idx] == entity_idx) {
        // This entity is the first child, update the parent's first_child pointer
        em->first_child[parent_idx] = em->next_sibling[entity_idx];
    }
    else {
        // Find this entity in the parent's children list
        int prev_sibling = em->first_child[parent_idx];
        while (prev_sibling != -1 && em->next_sibling[prev_sibling] != entity_idx) {
            prev_sibling = em->next_sibling[prev_sibling];
        }

        if (prev_sibling != -1) {
            // Remove from sibling chain
            em->next_sibling[prev_sibling] = em->next_sibling[entity_idx];
        }
    }

    // Update parent's child count
    em->child_count[parent_idx]--;

    // Reset this entity's parent and sibling info
    em->parent[entity_idx] = -1;
    em->next_sibling[entity_idx] = -1;

    // Preserve the entity's world position by setting its local position to its current world position
    em->local_x[entity_idx] = em->x[entity_idx];
    em->local_y[entity_idx] = em->y[entity_idx];

    // Update transform matrix
    transform_set(&em->local_transform[entity_idx],
        em->local_x[entity_idx],
        em->local_y[entity_idx],
        em->rotation[entity_idx],
        em->scale_x[entity_idx],
        em->scale_y[entity_idx]);

    // Copy local to world since it's now a root entity
    em->world_transform[entity_idx] = em->local_transform[entity_idx];
    em->transform_dirty[entity_idx] = false;

    // Update position
    em->x[entity_idx] = em->local_x[entity_idx];
    em->y[entity_idx] = em->local_y[entity_idx];
    em->right[entity_idx] = em->x[entity_idx] + em->width[entity_idx] * em->scale_x[entity_idx];
    em->bottom[entity_idx] = em->y[entity_idx] + em->height[entity_idx] * em->scale_y[entity_idx];
}

// Get number of children for an entity
int engine_get_child_count(Engine* engine, int entity_idx) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) return 0;
    return engine->entities.child_count[entity_idx];
}

// Get child at a specific index
int engine_get_child_at_index(Engine* engine, int entity_idx, int child_index) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) return -1;
    if (child_index < 0 || child_index >= engine->entities.child_count[entity_idx]) return -1;

    // Traverse the linked list of children to find the child at the specified index
    int current_child = engine->entities.first_child[entity_idx];
    int current_index = 0;

    while (current_child != -1 && current_index < child_index) {
        current_child = engine->entities.next_sibling[current_child];
        current_index++;
    }

    return current_child;
}

// Set entity's local transform
void engine_set_entity_local_transform(Engine* engine, int entity_idx, float x, float y, float rotation, float scale_x, float scale_y) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) return;

    EntityManager* em = &engine->entities;

    em->local_x[entity_idx] = x;
    em->local_y[entity_idx] = y;
    em->rotation[entity_idx] = rotation;
    em->scale_x[entity_idx] = scale_x;
    em->scale_y[entity_idx] = scale_y;

    // Update local transform matrix
    transform_set(&em->local_transform[entity_idx], x, y, rotation, scale_x, scale_y);

    // Mark as dirty
    em->transform_dirty[entity_idx] = true;

    // Update world transform
    engine_update_entity_transform(engine, entity_idx);
}

// Set entity rotation
void engine_set_entity_rotation(Engine* engine, int entity_idx, float rotation) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) return;

    EntityManager* em = &engine->entities;
    engine_set_entity_local_transform(engine, entity_idx,
        em->local_x[entity_idx],
        em->local_y[entity_idx],
        rotation,
        em->scale_x[entity_idx],
        em->scale_y[entity_idx]);
}

// Set entity scale
void engine_set_entity_scale(Engine* engine, int entity_idx, float scale_x, float scale_y) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) return;

    EntityManager* em = &engine->entities;
    engine_set_entity_local_transform(engine, entity_idx,
        em->local_x[entity_idx],
        em->local_y[entity_idx],
        em->rotation[entity_idx],
        scale_x,
        scale_y);
}

// Update world transforms for all entities
void engine_update_world_transforms(Engine* engine) {
    EntityManager* em = &engine->entities;

    // Process entities in parent-first order (breadth-first traversal)
    // Start with root entities (those without parents)
    for (int i = 0; i < em->count; i++) {
        if (em->parent[i] == -1) {
            engine_update_entity_transform(engine, i);
        }
    }
}

// Update transform for a specific entity and its children
void engine_update_entity_transform(Engine* engine, int entity_idx) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) return;

    EntityManager* em = &engine->entities;

    // Update local transform if dirty
    if (em->transform_dirty[entity_idx]) {
        transform_set(&em->local_transform[entity_idx],
            em->local_x[entity_idx],
            em->local_y[entity_idx],
            em->rotation[entity_idx],
            em->scale_x[entity_idx],
            em->scale_y[entity_idx]);
        em->transform_dirty[entity_idx] = false;
    }

    // Calculate world transform based on parent
    if (em->parent[entity_idx] == -1) {
        // This is a root entity, world transform = local transform
        em->world_transform[entity_idx] = em->local_transform[entity_idx];
    }
    else {
        // Combine parent's world transform with this entity's local transform
        transform_combine(&em->world_transform[entity_idx],
            &em->world_transform[em->parent[entity_idx]],
            &em->local_transform[entity_idx]);
    }

    // Update world position from world transform
    em->x[entity_idx] = em->world_transform[entity_idx].m[2]; // Extract x translation
    em->y[entity_idx] = em->world_transform[entity_idx].m[5]; // Extract y translation

    // Update bounds (approximate for rotated/scaled entities)
    float scaled_width = em->width[entity_idx] * fabsf(em->scale_x[entity_idx]);
    float scaled_height = em->height[entity_idx] * fabsf(em->scale_y[entity_idx]);

    if (em->rotation[entity_idx] == 0.0f) {
        // No rotation, simple bounds
        em->right[entity_idx] = em->x[entity_idx] + scaled_width;
        em->bottom[entity_idx] = em->y[entity_idx] + scaled_height;
    }
    else {
        // With rotation, use a conservative bounding box
        float sin_r = fabsf(sinf(em->rotation[entity_idx]));
        float cos_r = fabsf(cosf(em->rotation[entity_idx]));
        float rotated_width = scaled_width * cos_r + scaled_height * sin_r;
        float rotated_height = scaled_width * sin_r + scaled_height * cos_r;

        em->right[entity_idx] = em->x[entity_idx] + rotated_width;
        em->bottom[entity_idx] = em->y[entity_idx] + rotated_height;
    }

    // Recursively update all children
    int child = em->first_child[entity_idx];
    while (child != -1) {
        engine_update_entity_transform(engine, child);
        child = em->next_sibling[child];
    }
}

// Modified engine_update to handle transforms
void engine_update(Engine* engine) {
    // Calculate delta time
    Uint64 current_time = SDL_GetTicks();
    float delta_time = (current_time - engine->last_frame_time) / 1000.0f;
    engine->last_frame_time = current_time;

    // Smooth FPS calculation
    engine->fps = 0.95f * engine->fps + 0.05f * (1.0f / delta_time);

    // Update all entity transforms in the hierarchy
    engine_update_world_transforms(engine);

    // Update dynamic grid loading based on camera position
    update_dynamic_loading(engine);

    // Clear spatial grid
    clear_spatial_grid(&engine->grid);

    // Rebuild spatial grid with active entities
    for (int i = 0; i < engine->entities.count; i++) {
        if (engine->entities.active[i]) {
            spatial_grid_add(&engine->grid, i, engine->entities.x[i], engine->entities.y[i]);
        }
    }
}

// Modified engine_add_entity to handle transforms
int engine_add_entity(Engine* engine, float x, float y, int width, int height, int texture_id, int layer) {
    EntityManager* em = &engine->entities;

    // Ensure we have capacity (SoA growth pattern)
    if (em->count >= em->capacity) {
        int new_capacity = em->capacity * 2;

        // Allocate and copy original arrays
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

        // Allocate and copy new arrays for entity composition
        float* new_local_x = (float*)SDL_aligned_alloc(new_capacity * sizeof(float), CACHE_LINE_SIZE);
        float* new_local_y = (float*)SDL_aligned_alloc(new_capacity * sizeof(float), CACHE_LINE_SIZE);
        float* new_scale_x = (float*)SDL_aligned_alloc(new_capacity * sizeof(float), CACHE_LINE_SIZE);
        float* new_scale_y = (float*)SDL_aligned_alloc(new_capacity * sizeof(float), CACHE_LINE_SIZE);
        float* new_rotation = (float*)SDL_aligned_alloc(new_capacity * sizeof(float), CACHE_LINE_SIZE);
        int* new_parent = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        int* new_first_child = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        int* new_next_sibling = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        int* new_child_count = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        Transform* new_local_transform = (Transform*)SDL_aligned_alloc(new_capacity * sizeof(Transform), CACHE_LINE_SIZE);
        Transform* new_world_transform = (Transform*)SDL_aligned_alloc(new_capacity * sizeof(Transform), CACHE_LINE_SIZE);
        bool* new_transform_dirty = (bool*)SDL_aligned_alloc(new_capacity * sizeof(bool), CACHE_LINE_SIZE);

        // Copy existing data (original arrays)
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

        // Copy existing data (new arrays)
        memcpy(new_local_x, em->local_x, em->count * sizeof(float));
        memcpy(new_local_y, em->local_y, em->count * sizeof(float));
        memcpy(new_scale_x, em->scale_x, em->count * sizeof(float));
        memcpy(new_scale_y, em->scale_y, em->count * sizeof(float));
        memcpy(new_rotation, em->rotation, em->count * sizeof(float));
        memcpy(new_parent, em->parent, em->count * sizeof(int));
        memcpy(new_first_child, em->first_child, em->count * sizeof(int));
        memcpy(new_next_sibling, em->next_sibling, em->count * sizeof(int));
        memcpy(new_child_count, em->child_count, em->count * sizeof(int));
        memcpy(new_local_transform, em->local_transform, em->count * sizeof(Transform));
        memcpy(new_world_transform, em->world_transform, em->count * sizeof(Transform));
        memcpy(new_transform_dirty, em->transform_dirty, em->count * sizeof(bool));

        // Initialize new entities
        for (int i = em->count; i < new_capacity; i++) {
            new_local_x[i] = 0.0f;
            new_local_y[i] = 0.0f;
            new_scale_x[i] = 1.0f;
            new_scale_y[i] = 1.0f;
            new_rotation[i] = 0.0f;
            new_parent[i] = -1;
            new_first_child[i] = -1;
            new_next_sibling[i] = -1;
            new_child_count[i] = 0;
            transform_identity(&new_local_transform[i]);
            transform_identity(&new_world_transform[i]);
            new_transform_dirty[i] = true;
        }

        // Free old arrays (original)
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

        // Free old arrays (new)
        SDL_aligned_free(em->local_x);
        SDL_aligned_free(em->local_y);
        SDL_aligned_free(em->scale_x);
        SDL_aligned_free(em->scale_y);
        SDL_aligned_free(em->rotation);
        SDL_aligned_free(em->parent);
        SDL_aligned_free(em->first_child);
        SDL_aligned_free(em->next_sibling);
        SDL_aligned_free(em->child_count);
        SDL_aligned_free(em->local_transform);
        SDL_aligned_free(em->world_transform);
        SDL_aligned_free(em->transform_dirty);

        // Update pointers (original)
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

        // Update pointers (new)
        em->local_x = new_local_x;
        em->local_y = new_local_y;
        em->scale_x = new_scale_x;
        em->scale_y = new_scale_y;
        em->rotation = new_rotation;
        em->parent = new_parent;
        em->first_child = new_first_child;
        em->next_sibling = new_next_sibling;
        em->child_count = new_child_count;
        em->local_transform = new_local_transform;
        em->world_transform = new_world_transform;
        em->transform_dirty = new_transform_dirty;

        em->capacity = new_capacity;
    }

    int entity_idx = em->count++;

    // Set entity properties
    em->x[entity_idx] = x;
    em->y[entity_idx] = y;
    em->width[entity_idx] = width;
    em->height[entity_idx] = height;

    // Set transform properties
    em->local_x[entity_idx] = x;
    em->local_y[entity_idx] = y;
    em->scale_x[entity_idx] = 1.0f;
    em->scale_y[entity_idx] = 1.0f;
    em->rotation[entity_idx] = 0.0f;

    // Precompute right and bottom for faster culling
    em->right[entity_idx] = x + width;
    em->bottom[entity_idx] = y + height;

    em->texture_id[entity_idx] = texture_id;
    em->layer[entity_idx] = layer;

    // Initialize transforms
    transform_set(&em->local_transform[entity_idx], x, y, 0.0f, 1.0f, 1.0f);
    em->world_transform[entity_idx] = em->local_transform[entity_idx]; // Initially same as local

    // Determine if entity should be active based on grid loading
    int grid_x = (int)(x / engine->grid_cell_size);
    int grid_y = (int)(y / engine->grid_cell_size);

    if (grid_x >= 0 && grid_x < engine->grid_width &&
        grid_y >= 0 && grid_y < engine->grid_height) {
        em->active[entity_idx] = engine->grid_loaded[grid_x][grid_y];
    }
    else {
        em->active[entity_idx] = false;
    }

    // Add to spatial grid
    spatial_grid_add(&engine->grid, entity_idx, x, y);

    return entity_idx;
}

// Modified engine_set_entity_position for transform hierarchy
void engine_set_entity_position(Engine* engine, int entity_idx, float x, float y) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) return;

    EntityManager* em = &engine->entities;

    if (em->parent[entity_idx] == -1) {
        // Root entity: directly set position
        em->x[entity_idx] = x;
        em->y[entity_idx] = y;
        em->local_x[entity_idx] = x;
        em->local_y[entity_idx] = y;

        // Update precomputed values
        em->right[entity_idx] = x + em->width[entity_idx] * em->scale_x[entity_idx];
        em->bottom[entity_idx] = y + em->height[entity_idx] * em->scale_y[entity_idx];

        // Update transform
        transform_set(&em->local_transform[entity_idx],
            x, y,
            em->rotation[entity_idx],
            em->scale_x[entity_idx],
            em->scale_y[entity_idx]);

        em->world_transform[entity_idx] = em->local_transform[entity_idx];

        // Recursively update children
        int child = em->first_child[entity_idx];
        while (child != -1) {
            engine_update_entity_transform(engine, child);
            child = em->next_sibling[child];
        }
    }
    else {
        // Child entity: convert to local space
        // Get inverse of parent's world transform
        int parent_idx = em->parent[entity_idx];

        // Calculate the inverse transform (approximate for 2D transforms)
        Transform parent_world_inv;
        float a = em->world_transform[parent_idx].m[0];
        float b = em->world_transform[parent_idx].m[3];
        float c = em->world_transform[parent_idx].m[1];
        float d = em->world_transform[parent_idx].m[4];
        float e = em->world_transform[parent_idx].m[2];
        float f = em->world_transform[parent_idx].m[5];

        // Determinant
        float det = a * d - b * c;
        if (fabsf(det) < 0.0001f) {
            // Singular matrix, can't invert
            return;
        }

        // Inverse elements
        float inv_det = 1.0f / det;
        parent_world_inv.m[0] = d * inv_det;
        parent_world_inv.m[1] = -c * inv_det;
        parent_world_inv.m[2] = (c * f - d * e) * inv_det;
        parent_world_inv.m[3] = -b * inv_det;
        parent_world_inv.m[4] = a * inv_det;
        parent_world_inv.m[5] = (b * e - a * f) * inv_det;

        // Convert world position to local space
        float local_x, local_y;
        transform_apply(&parent_world_inv, x, y, &local_x, &local_y);

        // Set local position
        em->local_x[entity_idx] = local_x;
        em->local_y[entity_idx] = local_y;

        // Update transform
        transform_set(&em->local_transform[entity_idx],
            local_x, local_y,
            em->rotation[entity_idx],
            em->scale_x[entity_idx],
            em->scale_y[entity_idx]);

        // Update world transform and position
        engine_update_entity_transform(engine, entity_idx);
    }
}

// Modified engine_render to handle transformed entities
void engine_render(Engine* engine) {
    // Clear screen
    SDL_SetRenderDrawColor(engine->renderer, 0, 0, 0, 255);
    SDL_RenderClear(engine->renderer);

    // Get visible area for culling
    SDL_FRect visible_rect = get_visible_rect(&engine->camera);

    // Get reusable buffer for visible entities
    int* visible_indices = (int*)get_buffer(&engine->entity_indices_pool);
    int visible_count = 0;

    // Query visible entities from spatial grid using SIMD-based frustum culling
    spatial_grid_query(&engine->grid, visible_rect, visible_indices, &visible_count,
        engine->entities.count, &engine->entities);

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
        world_x[i] = engine->entities.x[entity_idx];
        world_y[i] = engine->entities.y[entity_idx];
        width_f[i] = (float)engine->entities.width[entity_idx] * engine->entities.scale_x[entity_idx];
        height_f[i] = (float)engine->entities.height[entity_idx] * engine->entities.scale_y[entity_idx];
    }

    // Calculate screen coordinates using SIMD
    calculate_screen_coordinates(world_x, world_y, width_f, height_f,
        screen_x, screen_y, screen_w, screen_h,
        visible_count, visible_rect.x, visible_rect.y, engine->camera.zoom);

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
        int texture_id = engine->entities.texture_id[entity_idx];
        int layer = engine->entities.layer[entity_idx];

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

        // Add to appropriate batch, handling rotation if needed
        SDL_FColor color = { 1.0f, 1.0f, 1.0f, 1.0f };

        if (engine->entities.rotation[entity_idx] == 0.0f) {
            // No rotation, simple quad
            add_to_batch(&engine->batches[current_batch_idx],
                screen_pos_x, screen_pos_y,
                screen_pos_w, screen_pos_h,
                tex_region, color);
        }
        else {
            // Rotated quad - we need to create rotated vertices
            // Get the batch
            RenderBatch* batch = &engine->batches[current_batch_idx];

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

            // Calculate center of rotation
            float center_x = screen_pos_x + screen_pos_w / 2.0f;
            float center_y = screen_pos_y + screen_pos_h / 2.0f;

            // Calculate half-dimensions
            float half_w = screen_pos_w / 2.0f;
            float half_h = screen_pos_h / 2.0f;

            // Calculate sin and cos of rotation
            float rotation = engine->entities.rotation[entity_idx] * engine->camera.zoom;
            float sin_rot = sinf(rotation);
            float cos_rot = cosf(rotation);

            // Calculate vertex positions relative to center, then rotate
            int base_vertex = batch->vertex_count;

            // Top-left
            float x = -half_w;
            float y = -half_h;
            batch->vertices[base_vertex].position.x = center_x + x * cos_rot - y * sin_rot;
            batch->vertices[base_vertex].position.y = center_y + x * sin_rot + y * cos_rot;
            batch->vertices[base_vertex].color = color;
            batch->vertices[base_vertex].tex_coord.x = tex_region.x;
            batch->vertices[base_vertex].tex_coord.y = tex_region.y;

            // Top-right
            x = half_w;
            y = -half_h;
            batch->vertices[base_vertex + 1].position.x = center_x + x * cos_rot - y * sin_rot;
            batch->vertices[base_vertex + 1].position.y = center_y + x * sin_rot + y * cos_rot;
            batch->vertices[base_vertex + 1].color = color;
            batch->vertices[base_vertex + 1].tex_coord.x = tex_region.x + tex_region.w;
            batch->vertices[base_vertex + 1].tex_coord.y = tex_region.y;

            // Bottom-right
            x = half_w;
            y = half_h;
            batch->vertices[base_vertex + 2].position.x = center_x + x * cos_rot - y * sin_rot;
            batch->vertices[base_vertex + 2].position.y = center_y + x * sin_rot + y * cos_rot;
            batch->vertices[base_vertex + 2].color = color;
            batch->vertices[base_vertex + 2].tex_coord.x = tex_region.x + tex_region.w;
            batch->vertices[base_vertex + 2].tex_coord.y = tex_region.y + tex_region.h;

            // Bottom-left
            x = -half_w;
            y = half_h;
            batch->vertices[base_vertex + 3].position.x = center_x + x * cos_rot - y * sin_rot;
            batch->vertices[base_vertex + 3].position.y = center_y + x * sin_rot + y * cos_rot;
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
