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

    // Allocate child entity system arrays
    manager->parent_id = (int*)SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->first_child_id = (int*)SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->next_sibling_id = (int*)SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->child_count = (int*)SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->local_x = (float*)SDL_aligned_alloc(initial_capacity * sizeof(float), CACHE_LINE_SIZE);
    manager->local_y = (float*)SDL_aligned_alloc(initial_capacity * sizeof(float), CACHE_LINE_SIZE);

    // Initialize all to default values
    for (int i = 0; i < initial_capacity; i++) {
        manager->parent_id[i] = -1;
        manager->first_child_id[i] = -1;
        manager->next_sibling_id[i] = -1;
        manager->child_count[i] = 0;
        manager->local_x[i] = 0.0f;
        manager->local_y[i] = 0.0f;
    }
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

    // Free child entity system arrays
    SDL_aligned_free(manager->parent_id);
    SDL_aligned_free(manager->first_child_id);
    SDL_aligned_free(manager->next_sibling_id);
    SDL_aligned_free(manager->child_count);
    SDL_aligned_free(manager->local_x);
    SDL_aligned_free(manager->local_y);
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

// Modified engine_update function to include hierarchy updates
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

    // Process root entities only (those without parents)
    EntityManager* em = &engine->entities;
    for (int i = 0; i < em->count; i++) {
        if (em->active[i] && em->parent_id[i] == -1) {
            // Add root entity to spatial grid
            spatial_grid_add(&engine->grid, i, em->x[i], em->y[i]);

            // Process its children
            int child_idx = em->first_child_id[i];
            while (child_idx != -1) {
                if (em->active[child_idx]) {
                    spatial_grid_add(&engine->grid, child_idx, em->x[child_idx], em->y[child_idx]);
                }
                child_idx = em->next_sibling_id[child_idx];
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
        width_f[i] = (float)engine->entities.width[entity_idx];
        height_f[i] = (float)engine->entities.height[entity_idx];
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
    EntityManager* em = &engine->entities;

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

        // Allocate new child entity system arrays
        int* new_parent_id = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        int* new_first_child_id = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        int* new_next_sibling_id = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        int* new_child_count = (int*)SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        float* new_local_x = (float*)SDL_aligned_alloc(new_capacity * sizeof(float), CACHE_LINE_SIZE);
        float* new_local_y = (float*)SDL_aligned_alloc(new_capacity * sizeof(float), CACHE_LINE_SIZE);

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

        // Copy existing child entity data
        memcpy(new_parent_id, em->parent_id, em->count * sizeof(int));
        memcpy(new_first_child_id, em->first_child_id, em->count * sizeof(int));
        memcpy(new_next_sibling_id, em->next_sibling_id, em->count * sizeof(int));
        memcpy(new_child_count, em->child_count, em->count * sizeof(int));
        memcpy(new_local_x, em->local_x, em->count * sizeof(float));
        memcpy(new_local_y, em->local_y, em->count * sizeof(float));

        // Initialize new elements
        for (int i = em->count; i < new_capacity; i++) {
            new_parent_id[i] = -1;
            new_first_child_id[i] = -1;
            new_next_sibling_id[i] = -1;
            new_child_count[i] = 0;
            new_local_x[i] = 0.0f;
            new_local_y[i] = 0.0f;
        }

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

        // Free old child entity arrays
        SDL_aligned_free(em->parent_id);
        SDL_aligned_free(em->first_child_id);
        SDL_aligned_free(em->next_sibling_id);
        SDL_aligned_free(em->child_count);
        SDL_aligned_free(em->local_x);
        SDL_aligned_free(em->local_y);

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

        // Update child entity pointers
        em->parent_id = new_parent_id;
        em->first_child_id = new_first_child_id;
        em->next_sibling_id = new_next_sibling_id;
        em->child_count = new_child_count;
        em->local_x = new_local_x;
        em->local_y = new_local_y;

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

    // Initialize child entity data
    em->parent_id[entity_idx] = -1;
    em->first_child_id[entity_idx] = -1;
    em->next_sibling_id[entity_idx] = -1;
    em->child_count[entity_idx] = 0;
    em->local_x[entity_idx] = 0.0f;
    em->local_y[entity_idx] = 0.0f;

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

// Add a child entity with local position relative to parent
int engine_add_child_entity(Engine* engine, int parent_idx, float local_x, float local_y,
    int width, int height, int texture_id, int layer) {
    if (parent_idx < 0 || parent_idx >= engine->entities.count) {
        return -1; // Invalid parent
    }

    EntityManager* em = &engine->entities;

    // Calculate world position based on parent's position
    float world_x = em->x[parent_idx] + local_x;
    float world_y = em->y[parent_idx] + local_y;

    // Create the entity
    int child_idx = engine_add_entity(engine, world_x, world_y, width, height, texture_id, layer);
    if (child_idx < 0) {
        return -1; // Failed to create entity
    }

    // Set up parent-child relationship
    em->local_x[child_idx] = local_x;
    em->local_y[child_idx] = local_y;

    // Attach the child to the parent
    engine_attach_entity(engine, parent_idx, child_idx);

    return child_idx;
}
// Attach an existing entity to a parent
void engine_attach_entity(Engine* engine, int parent_idx, int child_idx) {
    if (parent_idx < 0 || parent_idx >= engine->entities.count ||
        child_idx < 0 || child_idx >= engine->entities.count ||
        parent_idx == child_idx) {
        return; // Invalid indices or trying to attach to self
    }

    EntityManager* em = &engine->entities;

    // If child already has a parent, detach it first
    if (em->parent_id[child_idx] != -1) {
        engine_detach_entity(engine, child_idx);
    }

    // Store the parent-child relationship
    em->parent_id[child_idx] = parent_idx;

    // Update child count for parent
    em->child_count[parent_idx]++;

    // Add child to parent's child list (as first child)
    int old_first_child = em->first_child_id[parent_idx];
    em->first_child_id[parent_idx] = child_idx;
    em->next_sibling_id[child_idx] = old_first_child;

    // Calculate and store the local coordinates
    em->local_x[child_idx] = em->x[child_idx] - em->x[parent_idx];
    em->local_y[child_idx] = em->y[child_idx] - em->y[parent_idx];
}

// Detach an entity from its parent
void engine_detach_entity(Engine* engine, int entity_idx) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) {
        return; // Invalid entity
    }

    EntityManager* em = &engine->entities;

    // Check if entity has a parent
    int parent_idx = em->parent_id[entity_idx];
    if (parent_idx == -1) {
        return; // Entity has no parent
    }

    // Update child list of parent
    if (em->first_child_id[parent_idx] == entity_idx) {
        // Entity is the first child
        em->first_child_id[parent_idx] = em->next_sibling_id[entity_idx];
    }
    else {
        // Find the previous sibling
        int prev_sibling = em->first_child_id[parent_idx];
        while (prev_sibling != -1 && em->next_sibling_id[prev_sibling] != entity_idx) {
            prev_sibling = em->next_sibling_id[prev_sibling];
        }

        if (prev_sibling != -1) {
            // Update next sibling link
            em->next_sibling_id[prev_sibling] = em->next_sibling_id[entity_idx];
        }
    }

    // Reset entity's parent and sibling data
    em->parent_id[entity_idx] = -1;
    em->next_sibling_id[entity_idx] = -1;

    // Decrement parent's child count
    em->child_count[parent_idx]--;

    // World position remains the same, but local coordinates are reset
    em->local_x[entity_idx] = 0.0f;
    em->local_y[entity_idx] = 0.0f;
}

// Get all children of an entity
int engine_get_children(Engine* engine, int entity_idx, int* child_indices, int max_children) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) {
        return 0; // Invalid entity
    }

    EntityManager* em = &engine->entities;
    int count = 0;

    // Traverse the linked list of children
    int child_idx = em->first_child_id[entity_idx];
    while (child_idx != -1 && count < max_children) {
        child_indices[count++] = child_idx;
        child_idx = em->next_sibling_id[child_idx];
    }

    return count;
}

// Get the number of children an entity has
int engine_get_child_count(Engine* engine, int entity_idx) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) {
        return 0; // Invalid entity
    }

    return engine->entities.child_count[entity_idx];
}
// Set the local position of an entity relative to its parent
void engine_set_local_position(Engine* engine, int entity_idx, float local_x, float local_y) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) {
        return; // Invalid entity
    }

    EntityManager* em = &engine->entities;

    // Store the local position
    em->local_x[entity_idx] = local_x;
    em->local_y[entity_idx] = local_y;

    // Update world position if entity has a parent
    int parent_idx = em->parent_id[entity_idx];
    if (parent_idx != -1) {
        float world_x = em->x[parent_idx] + local_x;
        float world_y = em->y[parent_idx] + local_y;

        // Update world position
        engine_set_entity_position(engine, entity_idx, world_x, world_y);
    }
}

// Update world position based on parent's position and local coordinates
void engine_update_entity_world_position(Engine* engine, int entity_idx) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) {
        return; // Invalid entity
    }

    EntityManager* em = &engine->entities;

    // Only update if entity has a parent
    int parent_idx = em->parent_id[entity_idx];
    if (parent_idx != -1) {
        float world_x = em->x[parent_idx] + em->local_x[entity_idx];
        float world_y = em->y[parent_idx] + em->local_y[entity_idx];

        // Update entity position
        em->x[entity_idx] = world_x;
        em->y[entity_idx] = world_y;

        // Update precomputed values
        em->right[entity_idx] = world_x + em->width[entity_idx];
        em->bottom[entity_idx] = world_y + em->height[entity_idx];

        // Update spatial grid
        spatial_grid_add(&engine->grid, entity_idx, world_x, world_y);

        // Recursively update all children
        int child_idx = em->first_child_id[entity_idx];
        while (child_idx != -1) {
            engine_update_entity_world_position(engine, child_idx);
            child_idx = em->next_sibling_id[child_idx];
        }
    }
}
// Update existing set_entity_position to handle child entities
void engine_set_entity_position(Engine* engine, int entity_idx, float x, float y) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) {
        return;
    }

    EntityManager* em = &engine->entities;

    // Store old position to calculate delta
    float old_x = em->x[entity_idx];
    float old_y = em->y[entity_idx];

    // Update position
    em->x[entity_idx] = x;
    em->y[entity_idx] = y;

    // Update precomputed values
    em->right[entity_idx] = x + em->width[entity_idx];
    em->bottom[entity_idx] = y + em->height[entity_idx];

    // Update spatial grid
    spatial_grid_add(&engine->grid, entity_idx, x, y);

    // If this entity has a parent, update its local coordinates
    int parent_idx = em->parent_id[entity_idx];
    if (parent_idx != -1) {
        em->local_x[entity_idx] = x - em->x[parent_idx];
        em->local_y[entity_idx] = y - em->y[parent_idx];
    }

    // Update all children recursively (using SIMD if available)
#if HAVE_SIMD
    // Fast path: Process multiple children at once using SIMD
    float delta_x = x - old_x;
    float delta_y = y - old_y;

    // Process children in batches for SIMD efficiency
    const int batch_size = 4; // SIMD register size (4 floats)
    int child_count = em->child_count[entity_idx];
    int* child_indices = (int*)SDL_aligned_alloc(child_count * sizeof(int), CACHE_LINE_SIZE);

    // Collect all children first for cache-friendly processing
    int collected = 0;
    int child_idx = em->first_child_id[entity_idx];
    while (child_idx != -1 && collected < child_count) {
        child_indices[collected++] = child_idx;
        child_idx = em->next_sibling_id[child_idx];
    }

    // Process in SIMD batches
    for (int i = 0; i < collected; i += batch_size) {
        int remaining = (collected - i < batch_size) ? collected - i : batch_size;

        // Load child positions
        for (int j = 0; j < remaining; j++) {
            int idx = child_indices[i + j];
            // Update child position
            float child_x = em->x[idx] + delta_x;
            float child_y = em->y[idx] + delta_y;
            engine_set_entity_position(engine, idx, child_x, child_y);
        }
    }

    SDL_aligned_free(child_indices);
#else
    // Slower but simpler approach - process children one by one
    int child_idx = em->first_child_id[entity_idx];
    while (child_idx != -1) {
        float child_local_x = em->local_x[child_idx];
        float child_local_y = em->local_y[child_idx];
        float child_world_x = x + child_local_x;
        float child_world_y = y + child_local_y;

        engine_set_entity_position(engine, child_idx, child_world_x, child_world_y);
        child_idx = em->next_sibling_id[child_idx];
    }
#endif
}



// Update entire entity hierarchy starting from a root entity
void engine_update_entity_hierarchy(Engine* engine, int entity_idx) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) {
        return; // Invalid entity
    }

    // Only proceed if this is a root entity (no parent)
    if (engine->entities.parent_id[entity_idx] == -1) {
        // Update all children recursively
        int child_idx = engine->entities.first_child_id[entity_idx];
        while (child_idx != -1) {
            engine_update_entity_world_position(engine, child_idx);
            child_idx = engine->entities.next_sibling_id[child_idx];
        }
    }
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



// Update active state to handle child entities
void engine_set_entity_active(Engine* engine, int entity_idx, bool active) {
    if (entity_idx < 0 || entity_idx >= engine->entities.count) {
        return;
    }

    EntityManager* em = &engine->entities;
    em->active[entity_idx] = active;

    // Propagate active state to all children
    int child_idx = em->first_child_id[entity_idx];
    while (child_idx != -1) {
        engine_set_entity_active(engine, child_idx, active);
        child_idx = em->next_sibling_id[child_idx];
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

