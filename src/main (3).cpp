#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ATMCommon.h>
#include <ATMBufferPool.h>
#include <ATMByteBuffer.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef EMSCRIPTEN
#include <emscripten.h>
#endif

// SIMD includes
#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX 1
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__SSE4_1__) || defined(__SSE4_2__)
#include <xmmintrin.h>
#include <emmintrin.h>
#define USE_SSE 1
#endif

#ifdef _MSC_VER
// MSVC (Visual Studio)
#define CACHE_LINE_SIZE 64
#define ALIGN_TO_CACHE_LINE __declspec(align(CACHE_LINE_SIZE))
#elif defined(__GNUC__) || defined(__clang__)
// GCC, Clang, or compatible compilers
#define CACHE_LINE_SIZE 64
#define ALIGN_TO_CACHE_LINE __attribute__((aligned(CACHE_LINE_SIZE)))
#else
// Fallback for other compilers (no alignment)
#define CACHE_LINE_SIZE 64
#define ALIGN_TO_CACHE_LINE
#endif

// SIMD-friendly chunk size for AoSoA (optimized for AVX/SSE)
#ifdef USE_AVX
#define CHUNK_SIZE 8
#elif defined(USE_SSE)
#define CHUNK_SIZE 4
#else
#define CHUNK_SIZE 8  // Default
#endif

// Maximum entity count supported
#define MAX_ENTITIES 1000000

// Performance monitoring macros
#ifdef ENABLE_PROFILING
#define PROFILE_START(name) Uint64 profile_start_##name = SDL_GetPerformanceCounter()
#define PROFILE_END(name) { \
    Uint64 profile_end_##name = SDL_GetPerformanceCounter(); \
    double elapsed_##name = (profile_end_##name - profile_start_##name) * 1000.0 / SDL_GetPerformanceFrequency(); \
    ATMLOG("PROFILE: %s took %.3f ms", #name, elapsed_##name); \
}
#else
#define PROFILE_START(name)
#define PROFILE_END(name)
#endif

// Forward declarations
typedef struct GameState GameState;

// Array of Structures of Arrays (AoSoA) for better SIMD utilization
typedef struct ALIGN_TO_CACHE_LINE EntityChunk {
    float x[CHUNK_SIZE];
    float y[CHUNK_SIZE];
    int width[CHUNK_SIZE];
    int height[CHUNK_SIZE];
    int texture_id[CHUNK_SIZE];
    int layer[CHUNK_SIZE];
    float min_depth[CHUNK_SIZE];
    float max_depth[CHUNK_SIZE];
} EntityChunk;

// Hot/cold data splitting
typedef struct EntityColdData {
    // Rarely accessed data
    int creation_time;
    int last_update_time;
    int flags;
} EntityColdData;

// Entity manager using AoSoA design
typedef struct {
    EntityChunk* chunks;         // Array of chunks (hot data)
    EntityColdData* cold_data;   // Cold data (less frequently accessed)
    int chunk_count;             // Number of chunks
    int chunk_capacity;          // Allocated chunk capacity
    int entity_count;            // Total entity count

    // Active entity tracking (optimization #5)
    int* active_list;            // List of active entity indices
    int active_count;            // Number of active entities
    uint64_t* active_bitset;     // Bitset for O(1) active lookup
} EntityManager;

// Optimized spatial grid (optimization #1)
typedef struct {
    int cell_size;               // Size of each grid cell
    int grid_width;              // Width of grid (cells)
    int grid_height;             // Height of grid (cells)

    // Flat arrays for better cache locality
    int* cell_entity_indices;    // Flat array of entity indices [cell_idx][entity_idx]
    int* cell_start_indices;     // Starting index in cell_entity_indices for each cell
    int* cell_counts;            // Number of entities in each cell

    float* cell_min_depth;       // Min depth per cell (flat array)
    float* cell_max_depth;       // Max depth per cell (flat array)

    bool* cell_dirty;            // Track which cells need updating (flat array)

    // Preallocated buffer for entity indices
    int entity_indices_capacity;
} SpatialGrid;

// Occlusion query result structure
typedef struct {
    uint32_t query_id;
    bool is_visible;
    SDL_FRect bounds;
} OcclusionQueryResult;

// Rendering batch with preallocated buffers (optimization #3)
typedef struct {
    int texture_id;
    SDL_Vertex* vertices;        // Vertex data for batch
    int* indices;                // Index data for batch
    int vertex_count;
    int index_count;
    int vertex_capacity;
    int index_capacity;

    // Matrix transformations for batch
    float view_matrix[16];       // View matrix for the batch
} RenderBatch;

// Texture atlas (optimization #7)
typedef struct {
    SDL_Texture* texture;
    SDL_FRect* regions;          // UV regions for each subtexture
    int region_count;
    int region_capacity;

    // Dynamic packing information
    int next_x;
    int next_y;
    int current_row_height;
} TextureAtlas;

// Camera for culling
typedef struct {
    float x, y;                  // Position
    float width, height;         // Viewport dimensions
    float zoom;                  // Zoom level
    float near_plane, far_plane; // For depth calculations
    float frustum[6][4];         // Frustum planes for SIMD culling
    float view_matrix[16];       // View matrix for transformations
} Camera;

// Arena allocator for temporary allocations (optimization #8)
typedef struct {
    void* memory;
    size_t size;
    size_t used;
    size_t high_water_mark;
    size_t allocation_count;     // For monitoring
    bool auto_expand;            // Flag to enable auto-expansion
} ArenaAllocator;

// World sector for parallel processing
typedef struct {
    SDL_FRect bounds;
    int* entity_indices;
    int entity_count;
    int entity_capacity;
} WorldSector;

// Game state
typedef struct GameState {
    SDL_Window* window;
    SDL_Renderer* renderer;
    EntityManager entities;
    SpatialGrid grid;
    WorldSector* sectors;
    int sector_count;
    int sector_rows;
    int sector_cols;
    RenderBatch* batches;
    int batch_count;
    TextureAtlas atlas;
    Camera camera;
    SDL_FRect world_bounds;
    float grid_cell_size;
    bool** grid_loaded;
    int grid_width, grid_height;
    ArenaAllocator frame_arena;
    ArenaAllocator persistent_arena;  // For longer-lived allocations
    OcclusionQueryResult* occlusion_queries;
    int occlusion_query_count;
    int occlusion_query_capacity;
    Uint64 last_frame_time;
    float fps;

    // Benchmarking
    int benchmark_frame;
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

    // Fixed buffer pools for temporary allocations
    FixedBufferPool vertex_buffer_pool = FixedBufferPool(sizeof(SDL_Vertex) * 64, 32);
    FixedBufferPool index_buffer_pool = FixedBufferPool(sizeof(int) * 96, 32);

    // Timing data for profiling (optimization #10)
    struct {
        double update_time;
        double render_time;
        double grid_update_time;
        double frustum_culling_time;
        double batch_preparation_time;
        double draw_time;
    } frame_stats;
} GameState;

// Initialize arena allocator with auto-expand (optimization #8)
void init_arena_allocator(ArenaAllocator* arena, size_t size, bool auto_expand) {
    arena->memory = SDL_malloc(size);
    arena->size = size;
    arena->used = 0;
    arena->high_water_mark = 0;
    arena->allocation_count = 0;
    arena->auto_expand = auto_expand;
}

// Expand arena if needed
void* arena_expand_if_needed(ArenaAllocator* arena, size_t size_needed) {
    if (!arena->auto_expand) return NULL;

    size_t new_size = arena->size * 2;
    while (new_size < arena->used + size_needed) {
        new_size *= 2;
    }

    void* new_memory = SDL_malloc(new_size);
    if (!new_memory) return NULL;

    // Copy existing data
    SDL_memcpy(new_memory, arena->memory, arena->used);
    SDL_free(arena->memory);

    void* result = (char*)new_memory + arena->used;
    arena->memory = new_memory;
    arena->size = new_size;

    ATMLOG("Arena expanded from %zu to %zu bytes", arena->size / 2, arena->size);
    return result;
}

// Reset arena for next frame (doesn't free memory)
void reset_arena(ArenaAllocator* arena) {
    if (arena->used > arena->high_water_mark) {
        arena->high_water_mark = arena->used;
    }
    arena->used = 0;
    arena->allocation_count = 0;
}

// Allocate from arena with auto-expansion
void* arena_alloc(ArenaAllocator* arena, size_t size) {
    // Align size to avoid misalignment issues
    size = (size + 7) & ~7; // Align to 8 bytes

    if (arena->used + size > arena->size) {
        if (arena->auto_expand) {
            void* ptr = arena_expand_if_needed(arena, size);
            if (ptr) {
                arena->used += size;
                arena->allocation_count++;
                return ptr;
            }
        }

        ATMLOG("Arena allocator out of memory! Used: %zu, Size: %zu, Requested: %zu",
            arena->used, arena->size, size);
        return NULL;
    }

    void* ptr = (char*)arena->memory + arena->used;
    arena->used += size;
    arena->allocation_count++;
    return ptr;
}

// Free arena allocator
void free_arena_allocator(ArenaAllocator* arena) {
    SDL_free(arena->memory);
    arena->memory = NULL;
    arena->size = 0;
    arena->used = 0;
}

// Initialize entity manager with AoSoA design and active list/bitset
void init_entity_manager(EntityManager* manager, int initial_capacity) {
    // Ensure capacity is sufficient for maximum entities
    if (initial_capacity > MAX_ENTITIES) {
        initial_capacity = MAX_ENTITIES;
    }

    // Calculate initial chunk capacity
    int initial_chunks = (initial_capacity + CHUNK_SIZE - 1) / CHUNK_SIZE;

    manager->chunk_capacity = initial_chunks;
    manager->chunk_count = 0;
    manager->entity_count = 0;

    // Allocate chunks and cold data
    manager->chunks = (EntityChunk*)SDL_malloc(initial_chunks * sizeof(EntityChunk));
    manager->cold_data = (EntityColdData*)SDL_malloc(initial_capacity * sizeof(EntityColdData));

    // Initialize all memory to 0
    SDL_memset(manager->chunks, 0, initial_chunks * sizeof(EntityChunk));
    SDL_memset(manager->cold_data, 0, initial_capacity * sizeof(EntityColdData));

    // Initialize active entity tracking (optimization #5)
    manager->active_list = (int*)SDL_malloc(initial_capacity * sizeof(int));
    manager->active_count = 0;

    // Initialize active bitset (optimization #5)
    int bitset_size = (initial_capacity + 63) / 64;
    manager->active_bitset = (uint64_t*)SDL_calloc(bitset_size, sizeof(uint64_t));
}

// Set entity active state with bitset optimization
void set_entity_active(EntityManager* manager, int entity_idx, bool active) {
    int chunk_idx = entity_idx / CHUNK_SIZE;
    int idx_in_chunk = entity_idx % CHUNK_SIZE;

    // Get current active state
    bool currently_active = manager->chunks[chunk_idx].x[idx_in_chunk] != 0;

    // If state is changing
    if (currently_active != active) {
        if (active) {
            // Add to active list
            manager->active_list[manager->active_count++] = entity_idx;

            // Set bit in active bitset
            manager->active_bitset[entity_idx / 64] |= (1ULL << (entity_idx % 64));
        }
        else {
            // Remove from active list
            for (int i = 0; i < manager->active_count; i++) {
                if (manager->active_list[i] == entity_idx) {
                    // Swap with last element and decrement count
                    manager->active_list[i] = manager->active_list[--manager->active_count];
                    break;
                }
            }

            // Clear bit in active bitset
            manager->active_bitset[entity_idx / 64] &= ~(1ULL << (entity_idx % 64));
        }
    }
}

// Check if entity is active using bitset (O(1) lookup)
bool is_entity_active(EntityManager* manager, int entity_idx) {
    return (manager->active_bitset[entity_idx / 64] & (1ULL << (entity_idx % 64))) != 0;
}

// Free entity manager
void free_entity_manager(EntityManager* manager) {
    SDL_free(manager->chunks);
    SDL_free(manager->cold_data);
    SDL_free(manager->active_list);
    SDL_free(manager->active_bitset);

    manager->chunks = NULL;
    manager->cold_data = NULL;
    manager->active_list = NULL;
    manager->active_bitset = NULL;
}

// Initialize optimized spatial grid (flat arrays)
void init_spatial_grid(SpatialGrid* grid, int world_width, int world_height, int cell_size) {
    grid->cell_size = cell_size;
    grid->grid_width = (world_width + cell_size - 1) / cell_size;
    grid->grid_height = (world_height + cell_size - 1) / cell_size;

    int total_cells = grid->grid_width * grid->grid_height;

    // Allocate flat arrays for better cache coherence
    grid->cell_start_indices = (int*)SDL_malloc(total_cells * sizeof(int));
    grid->cell_counts = (int*)SDL_calloc(total_cells, sizeof(int));
    grid->cell_dirty = (bool*)SDL_calloc(total_cells, sizeof(bool));
    grid->cell_min_depth = (float*)SDL_malloc(total_cells * sizeof(float));
    grid->cell_max_depth = (float*)SDL_malloc(total_cells * sizeof(float));

    // Initialize depth values
    for (int i = 0; i < total_cells; i++) {
        grid->cell_min_depth[i] = INFINITY;
        grid->cell_max_depth[i] = -INFINITY;
        grid->cell_start_indices[i] = 0;
    }

    // Pre-allocate buffer for entity indices (10% of MAX_ENTITIES as initial capacity)
    grid->entity_indices_capacity = MAX_ENTITIES / 10;
    grid->cell_entity_indices = (int*)SDL_malloc(grid->entity_indices_capacity * sizeof(int));
}

// Free spatial grid
void free_spatial_grid(SpatialGrid* grid) {
    SDL_free(grid->cell_entity_indices);
    SDL_free(grid->cell_start_indices);
    SDL_free(grid->cell_counts);
    SDL_free(grid->cell_dirty);
    SDL_free(grid->cell_min_depth);
    SDL_free(grid->cell_max_depth);
}

// Get cell index from grid coordinates
inline int get_cell_index(SpatialGrid* grid, int x, int y) {
    return y * grid->grid_width + x;
}

// Clear all cells in spatial grid
void clear_spatial_grid(SpatialGrid* grid) {
    int total_cells = grid->grid_width * grid->grid_height;
    SDL_memset(grid->cell_counts, 0, total_cells * sizeof(int));

    // Reset depth values
    for (int i = 0; i < total_cells; i++) {
        grid->cell_min_depth[i] = INFINITY;
        grid->cell_max_depth[i] = -INFINITY;
    }
}

// Rebuild the spatial grid using compact storage
void rebuild_spatial_grid(SpatialGrid* grid, EntityManager* entities) {
    PROFILE_START(rebuild_grid);

    // First count total entities per cell to determine offsets
    int total_entities = 0;
    int total_cells = grid->grid_width * grid->grid_height;

    // Mark all cells as clean
    SDL_memset(grid->cell_dirty, 0, total_cells * sizeof(bool));

    // Reset cell counts
    SDL_memset(grid->cell_counts, 0, total_cells * sizeof(int));

    // Count entities per cell using active list
    for (int i = 0; i < entities->active_count; i++) {
        int entity_idx = entities->active_list[i];
        int chunk_idx = entity_idx / CHUNK_SIZE;
        int idx_in_chunk = entity_idx % CHUNK_SIZE;

        EntityChunk* chunk = &entities->chunks[chunk_idx];

        // Calculate grid cells that contain this entity
        int start_x = (int)(chunk->x[idx_in_chunk] / grid->cell_size);
        int start_y = (int)(chunk->y[idx_in_chunk] / grid->cell_size);
        int end_x = (int)((chunk->x[idx_in_chunk] + chunk->width[idx_in_chunk]) / grid->cell_size);
        int end_y = (int)((chunk->y[idx_in_chunk] + chunk->height[idx_in_chunk]) / grid->cell_size);

        // Clamp to grid bounds
        start_x = SDL_clamp(start_x, 0, grid->grid_width - 1);
        start_y = SDL_clamp(start_y, 0, grid->grid_height - 1);
        end_x = SDL_clamp(end_x, 0, grid->grid_width - 1);
        end_y = SDL_clamp(end_y, 0, grid->grid_height - 1);

        // Count cell occupancy
        for (int cx = start_x; cx <= end_x; cx++) {
            for (int cy = start_y; cy <= end_y; cy++) {
                int cell_idx = get_cell_index(grid, cx, cy);
                grid->cell_counts[cell_idx]++;
                total_entities++;
            }
        }
    }

    // If we need more capacity, reallocate
    if (total_entities > grid->entity_indices_capacity) {
        grid->entity_indices_capacity = total_entities * 2; // Double for future growth
        SDL_free(grid->cell_entity_indices);
        grid->cell_entity_indices = (int*)SDL_malloc(grid->entity_indices_capacity * sizeof(int));
    }

    // Calculate start indices
    int current_offset = 0;
    for (int i = 0; i < total_cells; i++) {
        grid->cell_start_indices[i] = current_offset;
        current_offset += grid->cell_counts[i];

        // Reset counts for reuse when adding entities
        grid->cell_counts[i] = 0;
    }

    // Now add entities to their cells
    for (int i = 0; i < entities->active_count; i++) {
        int entity_idx = entities->active_list[i];
        int chunk_idx = entity_idx / CHUNK_SIZE;
        int idx_in_chunk = entity_idx % CHUNK_SIZE;

        EntityChunk* chunk = &entities->chunks[chunk_idx];

        // Calculate grid cells that contain this entity
        int start_x = (int)(chunk->x[idx_in_chunk] / grid->cell_size);
        int start_y = (int)(chunk->y[idx_in_chunk] / grid->cell_size);
        int end_x = (int)((chunk->x[idx_in_chunk] + chunk->width[idx_in_chunk]) / grid->cell_size);
        int end_y = (int)((chunk->y[idx_in_chunk] + chunk->height[idx_in_chunk]) / grid->cell_size);

        // Clamp to grid bounds
        start_x = SDL_clamp(start_x, 0, grid->grid_width - 1);
        start_y = SDL_clamp(start_y, 0, grid->grid_height - 1);
        end_x = SDL_clamp(end_x, 0, grid->grid_width - 1);
        end_y = SDL_clamp(end_y, 0, grid->grid_height - 1);

        float depth = chunk->min_depth[idx_in_chunk];

        // Add entity to each cell
        for (int cx = start_x; cx <= end_x; cx++) {
            for (int cy = start_y; cy <= end_y; cy++) {
                int cell_idx = get_cell_index(grid, cx, cy);
                int index = grid->cell_start_indices[cell_idx] + grid->cell_counts[cell_idx]++;
                grid->cell_entity_indices[index] = entity_idx;

                // Update depth range
                if (depth < grid->cell_min_depth[cell_idx]) grid->cell_min_depth[cell_idx] = depth;
                if (depth > grid->cell_max_depth[cell_idx]) grid->cell_max_depth[cell_idx] = depth;
            }
        }
    }

    PROFILE_END(rebuild_grid);
}

// Optimized spatial query with bitset tracking (optimization #4)
void spatial_grid_query(SpatialGrid* grid, SDL_FRect query_rect, float min_depth, float max_depth,
    int* result_indices, int* result_count, int max_results,
    uint64_t* encountered_bitset) {
    // Calculate grid cells that overlap the query rect
    int start_x = (int)(query_rect.x / grid->cell_size);
    int start_y = (int)(query_rect.y / grid->cell_size);
    int end_x = (int)((query_rect.x + query_rect.w) / grid->cell_size);
    int end_y = (int)((query_rect.y + query_rect.h) / grid->cell_size);

    // Clamp to grid bounds
    start_x = SDL_clamp(start_x, 0, grid->grid_width - 1);
    start_y = SDL_clamp(start_y, 0, grid->grid_height - 1);
    end_x = SDL_clamp(end_x, 0, grid->grid_width - 1);
    end_y = SDL_clamp(end_y, 0, grid->grid_height - 1);

    *result_count = 0;

    // Process each overlapping cell
    for (int cy = start_y; cy <= end_y; cy++) {
        for (int cx = start_x; cx <= end_x; cx++) {
            int cell_idx = get_cell_index(grid, cx, cy);

            // Hierarchical Z-buffer culling
            if (grid->cell_max_depth[cell_idx] < min_depth ||
                grid->cell_min_depth[cell_idx] > max_depth) {
                continue;
            }

            // Get entities from this cell
            int start_idx = grid->cell_start_indices[cell_idx];
            int count = grid->cell_counts[cell_idx];

            // Add entities to result
            for (int i = 0; i < count && *result_count < max_results; i++) {
                int entity_idx = grid->cell_entity_indices[start_idx + i];

                // Check if already encountered using bitset
                if ((encountered_bitset[entity_idx / 64] & (1ULL << (entity_idx % 64))) == 0) {
                    result_indices[(*result_count)++] = entity_idx;

                    // Mark as encountered
                    encountered_bitset[entity_idx / 64] |= (1ULL << (entity_idx % 64));
                }
            }
        }
    }
}

// Initialize sectors for parallel processing
void init_sectors(GameState* state, int rows, int cols) {
    state->sector_rows = rows;
    state->sector_cols = cols;
    state->sector_count = rows * cols;

    // Allocate sectors
    state->sectors = (WorldSector*)SDL_malloc(state->sector_count * sizeof(WorldSector));

    // Calculate sector dimensions
    float sector_width = state->world_bounds.w / cols;
    float sector_height = state->world_bounds.h / rows;

    // Initialize each sector
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int idx = y * cols + x;
            state->sectors[idx].bounds.x = x * sector_width;
            state->sectors[idx].bounds.y = y * sector_height;
            state->sectors[idx].bounds.w = sector_width;
            state->sectors[idx].bounds.h = sector_height;
            state->sectors[idx].entity_capacity = 1024;
            state->sectors[idx].entity_count = 0;
            state->sectors[idx].entity_indices = (int*)SDL_malloc(state->sectors[idx].entity_capacity * sizeof(int));
        }
    }
}

// Free sectors
void free_sectors(GameState* state) {
    for (int i = 0; i < state->sector_count; i++) {
        SDL_free(state->sectors[i].entity_indices);
    }
    SDL_free(state->sectors);
}

// Get chunk and index within chunk for an entity
inline void get_entity_location(int entity_idx, int* chunk_idx, int* index_in_chunk) {
    *chunk_idx = entity_idx / CHUNK_SIZE;
    *index_in_chunk = entity_idx % CHUNK_SIZE;
}

// Check if two rectangles intersect
inline bool rects_intersect(SDL_FRect a, SDL_FRect b) {
    return !(a.x + a.w <= b.x || a.x >= b.x + b.w ||
        a.y + a.h <= b.y || a.y >= b.y + b.h);
}

// Initialize a texture atlas with dynamic packing (optimization #7)
void init_texture_atlas(TextureAtlas* atlas, SDL_Renderer* renderer, int width, int height) {
    atlas->texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_TARGET, width, height);
    if (!atlas->texture) {
        ATMLOG("Failed to create texture atlas: %s", SDL_GetError());
        return;
    }

    SDL_SetTextureBlendMode(atlas->texture, SDL_BLENDMODE_BLEND);
    atlas->region_capacity = 32;
    atlas->region_count = 0;
    atlas->regions = (SDL_FRect*)SDL_malloc(atlas->region_capacity * sizeof(SDL_FRect));

    // Initialize packing state
    atlas->next_x = 0;
    atlas->next_y = 0;
    atlas->current_row_height = 0;
}

// Add texture to atlas with dynamic packing
int add_texture_to_atlas_dynamic(TextureAtlas* atlas, SDL_Renderer* renderer, SDL_Surface* surface) {
    int texture_id = atlas->region_count;
    float width, height;

    SDL_GetTextureSize(atlas->texture, &width, &height);

    // Check if we need to move to next row
    if (atlas->next_x + surface->w > width) {
        atlas->next_x = 0;
        atlas->next_y += atlas->current_row_height;
        atlas->current_row_height = 0;
    }

    // Check if we're out of space
    if (atlas->next_y + surface->h > height) {
        ATMLOG("Texture atlas full!");
        return -1;
    }

    // Ensure capacity
    if (texture_id >= atlas->region_capacity) {
        atlas->region_capacity *= 2;
        atlas->regions = (SDL_FRect*)SDL_realloc(atlas->regions,
            atlas->region_capacity * sizeof(SDL_FRect));
    }

    // Calculate normalized UV coordinates
    SDL_FRect region = {
        (float)atlas->next_x / width,
        (float)atlas->next_y / height,
        (float)surface->w / width,
        (float)surface->h / height
    };

    atlas->regions[texture_id] = region;
    atlas->region_count++;

    // Copy surface to atlas texture
    SDL_Texture* temp = SDL_CreateTextureFromSurface(renderer, surface);

    // Set render target to atlas
    SDL_Texture* old_target = SDL_GetRenderTarget(renderer);
    SDL_SetRenderTarget(renderer, atlas->texture);

    // Copy texture to atlas
    SDL_FRect dest = { (float)atlas->next_x, (float)atlas->next_y,
                      (float)surface->w, (float)surface->h };
    SDL_RenderTexture(renderer, temp, NULL, &dest);

    // Reset render target
    SDL_SetRenderTarget(renderer, old_target);

    // Clean up
    SDL_DestroyTexture(temp);

    // Update position
    atlas->next_x += surface->w;
    atlas->current_row_height = SDL_max(atlas->current_row_height, surface->h);

    return texture_id;
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

// SIMD frustum culling (optimization #2)
#ifdef USE_AVX
// AVX version of frustum culling
bool is_aabb_in_frustum_simd(float* frustum, float x, float y, float width, float height) {
    // Convert AABB to center-radius form
    float center_x = x + width * 0.5f;
    float center_y = y + height * 0.5f;
    float extent_x = width * 0.5f;
    float extent_y = height * 0.5f;

    // Test against each frustum plane
    for (int i = 0; i < 6; i += 2) {
        // Load two planes at once
        __m256 plane0 = _mm256_loadu_ps(&frustum[i * 4]);
        __m256 plane1 = _mm256_loadu_ps(&frustum[(i + 1) * 4]);

        // Broadcast center coordinates
        __m256 center_x8 = _mm256_set1_ps(center_x);
        __m256 center_y8 = _mm256_set1_ps(center_y);
        __m256 zeros = _mm256_setzero_ps();

        // Compute dot product for both planes
        __m256 dot0 = _mm256_mul_ps(plane0, center_x8);
        __m256 dot1 = _mm256_mul_ps(plane1, center_x8);

        dot0 = _mm256_fmadd_ps(plane0, center_y8, dot0);
        dot1 = _mm256_fmadd_ps(plane1, center_y8, dot1);

        // Compute dot product for plane normal
        __m256 radius = _mm256_mul_ps(_mm256_set1_ps(extent_x), _mm256_abs_ps(plane0));
        radius = _mm256_add_ps(radius, _mm256_mul_ps(_mm256_set1_ps(extent_y), _mm256_abs_ps(plane0)));

        __m256 radius1 = _mm256_mul_ps(_mm256_set1_ps(extent_x), _mm256_abs_ps(plane1));
        radius1 = _mm256_add_ps(radius1, _mm256_mul_ps(_mm256_set1_ps(extent_y), _mm256_abs_ps(plane1)));

        // Check if center is further than radius
        __m256 test0 = _mm256_cmp_ps(dot0, _mm256_sub_ps(zeros, radius), _MM_CMPINT_LT);
        __m256 test1 = _mm256_cmp_ps(dot1, _mm256_sub_ps(zeros, radius1), _MM_CMPINT_LT);

        // If any plane test fails, object is outside
        int mask0 = _mm256_movemask_ps(test0);
        int mask1 = _mm256_movemask_ps(test1);

        if (mask0 || mask1) {
            return false;
        }
    }

    return true;
}
#elif defined(USE_SSE)
// SSE version of frustum culling
bool is_aabb_in_frustum_simd(float* frustum, float x, float y, float width, float height) {
    // Convert AABB to center-radius form
    float center_x = x + width * 0.5f;
    float center_y = y + height * 0.5f;
    float extent_x = width * 0.5f;
    float extent_y = height * 0.5f;

    // Test against each frustum plane
    for (int i = 0; i < 6; i++) {
        __m128 plane = _mm_loadu_ps(&frustum[i * 4]);
        __m128 center = _mm_set_ps(1.0f, 0.0f, center_y, center_x);

        // Compute dot product for center
        __m128 dot = _mm_mul_ps(plane, center);
        dot = _mm_hadd_ps(dot, dot);
        dot = _mm_hadd_ps(dot, dot);

        // Compute radius
        __m128 abs_plane = _mm_abs_ps(plane);
        __m128 extents = _mm_set_ps(0.0f, 0.0f, extent_y, extent_x);
        __m128 radius = _mm_mul_ps(abs_plane, extents);
        radius = _mm_hadd_ps(radius, radius);
        radius = _mm_hadd_ps(radius, radius);

        // Compare dot + radius < 0
        float dot_value = _mm_cvtss_f32(dot);
        float radius_value = _mm_cvtss_f32(radius);

        if (dot_value < -radius_value) {
            return false;
        }
    }

    return true;
}
#else
// Scalar version as fallback
bool is_aabb_in_frustum_simd(float* frustum, float x, float y, float width, float height) {
    // Convert AABB to center-radius form for more efficient testing
    float center_x = x + width * 0.5f;
    float center_y = y + height * 0.5f;
    float extent_x = width * 0.5f;
    float extent_y = height * 0.5f;

    // Test against each frustum plane
    for (int i = 0; i < 6; i++) {
        float plane_x = frustum[i * 4 + 0];
        float plane_y = frustum[i * 4 + 1];
        float plane_z = frustum[i * 4 + 2];
        float plane_d = frustum[i * 4 + 3];

        // Project box extents onto plane normal
        float radius = extent_x * fabsf(plane_x) + extent_y * fabsf(plane_y);

        // Distance from center to plane
        float distance = plane_x * center_x + plane_y * center_y + plane_z * 0 + plane_d;

        // If center is further from plane than radius, box is outside
        if (distance < -radius) {
            return false;
        }
    }

    return true;
}
#endif

// Set up frustum planes for the camera
void setup_camera_frustum(Camera* camera) {
    // Get visible rect
    SDL_FRect visible = get_visible_rect(camera);

    // Left plane
    camera->frustum[0][0] = 1.0f;  // Normal x
    camera->frustum[0][1] = 0.0f;  // Normal y
    camera->frustum[0][2] = 0.0f;  // Normal z
    camera->frustum[0][3] = -visible.x; // Distance

    // Right plane
    camera->frustum[1][0] = -1.0f;
    camera->frustum[1][1] = 0.0f;
    camera->frustum[1][2] = 0.0f;
    camera->frustum[1][3] = visible.x + visible.w;

    // Top plane
    camera->frustum[2][0] = 0.0f;
    camera->frustum[2][1] = 1.0f;
    camera->frustum[2][2] = 0.0f;
    camera->frustum[2][3] = -visible.y;

    // Bottom plane
    camera->frustum[3][0] = 0.0f;
    camera->frustum[3][1] = -1.0f;
    camera->frustum[3][2] = 0.0f;
    camera->frustum[3][3] = visible.y + visible.h;

    // Near plane
    camera->frustum[4][0] = 0.0f;
    camera->frustum[4][1] = 0.0f;
    camera->frustum[4][2] = 1.0f;
    camera->frustum[4][3] = -camera->near_plane;

    // Far plane
    camera->frustum[5][0] = 0.0f;
    camera->frustum[5][1] = 0.0f;
    camera->frustum[5][2] = -1.0f;
    camera->frustum[5][3] = camera->far_plane;

    // Set up view matrix for transformations
    float cos_zoom = cosf(0.0f);
    float sin_zoom = sinf(0.0f);

    // Row-major view matrix (rotation + translation + scale)
    camera->view_matrix[0] = cos_zoom * camera->zoom;
    camera->view_matrix[1] = sin_zoom * camera->zoom;
    camera->view_matrix[2] = 0.0f;
    camera->view_matrix[3] = 0.0f;

    camera->view_matrix[4] = -sin_zoom * camera->zoom;
    camera->view_matrix[5] = cos_zoom * camera->zoom;
    camera->view_matrix[6] = 0.0f;
    camera->view_matrix[7] = 0.0f;

    camera->view_matrix[8] = 0.0f;
    camera->view_matrix[9] = 0.0f;
    camera->view_matrix[10] = 1.0f;
    camera->view_matrix[11] = 0.0f;

    camera->view_matrix[12] = -camera->x * camera->zoom;
    camera->view_matrix[13] = -camera->y * camera->zoom;
    camera->view_matrix[14] = 0.0f;
    camera->view_matrix[15] = 1.0f;
}

// Compare function for z-order and texture sorting (batching optimization)
int compare_entities_for_rendering(void* user_data, const void* a, const void* b) {
    int* idx_a = (int*)a;
    int* idx_b = (int*)b;
    EntityManager* entities = (EntityManager*)user_data;

    // Get chunk and index information
    int chunk_a, index_a, chunk_b, index_b;
    get_entity_location(*idx_a, &chunk_a, &index_a);
    get_entity_location(*idx_b, &chunk_b, &index_b);

    // First sort by texture (for batch efficiency)
    if (entities->chunks[chunk_a].texture_id[index_a] != entities->chunks[chunk_b].texture_id[index_b]) {
        return entities->chunks[chunk_a].texture_id[index_a] - entities->chunks[chunk_b].texture_id[index_b];
    }

    // Then by z-layer
    return entities->chunks[chunk_a].layer[index_a] - entities->chunks[chunk_b].layer[index_b];
}

// Add a quad to a render batch with matrix transformation
void add_to_batch(RenderBatch* batch, float x, float y, float w, float h,
    SDL_FRect tex_region, SDL_FColor color, float* transform_matrix) {
    // Ensure we have enough space
    if (batch->vertex_count + 4 > batch->vertex_capacity) {
        batch->vertex_capacity *= 2;
        batch->vertices = (SDL_Vertex*)SDL_realloc(batch->vertices,
            batch->vertex_capacity * sizeof(SDL_Vertex));
    }

    if (batch->index_count + 6 > batch->index_capacity) {
        batch->index_capacity *= 2;
        batch->indices = (int*)SDL_realloc(batch->indices,
            batch->index_capacity * sizeof(int));
    }

    // Add vertices
    int base_vertex = batch->vertex_count;

    // Define quad corners
    float corners[4][3] = {
        { x, y, 1.0f },           // Top-left
        { x + w, y, 1.0f },       // Top-right
        { x + w, y + h, 1.0f },   // Bottom-right
        { x, y + h, 1.0f }        // Bottom-left
    };

    // UV coordinates
    float uvs[4][2] = {
        { tex_region.x, tex_region.y },                       // Top-left
        { tex_region.x + tex_region.w, tex_region.y },        // Top-right
        { tex_region.x + tex_region.w, tex_region.y + tex_region.h },  // Bottom-right
        { tex_region.x, tex_region.y + tex_region.h }         // Bottom-left
    };

    // Transform and add each vertex
    for (int i = 0; i < 4; i++) {
        // Apply matrix transformation
        float tx = corners[i][0] * transform_matrix[0] +
            corners[i][1] * transform_matrix[4] +
            corners[i][2] * transform_matrix[12];

        float ty = corners[i][0] * transform_matrix[1] +
            corners[i][1] * transform_matrix[5] +
            corners[i][2] * transform_matrix[13];

        batch->vertices[base_vertex + i].position.x = tx;
        batch->vertices[base_vertex + i].position.y = ty;
        batch->vertices[base_vertex + i].color = color;
        batch->vertices[base_vertex + i].tex_coord.x = uvs[i][0];
        batch->vertices[base_vertex + i].tex_coord.y = uvs[i][1];
    }

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
    start_x = SDL_clamp(start_x, 0, state->grid_width - 1);
    start_y = SDL_clamp(start_y, 0, state->grid_height - 1);
    end_x = SDL_clamp(end_x, 0, state->grid_width - 1);
    end_y = SDL_clamp(end_y, 0, state->grid_height - 1);

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
    state->entities.active_count = 0;
    int bitset_size = (state->entities.entity_count + 63) / 64;
    SDL_memset(state->entities.active_bitset, 0, bitset_size * sizeof(uint64_t));

    // Process only entities in loaded cells
    for (int c = 0; c < state->entities.chunk_count; c++) {
        EntityChunk* chunk = &state->entities.chunks[c];
        for (int i = 0; i < CHUNK_SIZE; i++) {
            int entity_idx = c * CHUNK_SIZE + i;
            if (entity_idx < state->entities.entity_count) {
                int grid_x = (int)(chunk->x[i] / state->grid_cell_size);
                int grid_y = (int)(chunk->y[i] / state->grid_cell_size);

                bool should_be_active = false;
                if (grid_x >= 0 && grid_x < state->grid_width &&
                    grid_y >= 0 && grid_y < state->grid_height) {
                    should_be_active = state->grid_loaded[grid_x][grid_y];
                }

                if (should_be_active) {
                    // Add to active list and set bit
                    state->entities.active_list[state->entities.active_count++] = entity_idx;
                    state->entities.active_bitset[entity_idx / 64] |= (1ULL << (entity_idx % 64));
                }
            }
        }
    }
}

// Process sector visibility for parallel processing with optimized bitset (optimization #6)
void process_sector_visibility(GameState* state, int sector_idx, uint64_t* local_encountered) {
    PROFILE_START(process_sector);

    WorldSector* sector = &state->sectors[sector_idx];

    // Clear current entities
    sector->entity_count = 0;

    // Skip if sector is outside view frustum
    if (!is_aabb_in_frustum_simd((float*)state->camera.frustum,
        sector->bounds.x, sector->bounds.y,
        sector->bounds.w, sector->bounds.h)) {
        PROFILE_END(process_sector);
        return;
    }

    // Query spatial grid for entities in this sector with bitset for duplicate tracking
    int* result_indices = (int*)arena_alloc(&state->frame_arena, state->entities.entity_count * sizeof(int));
    int result_count = 0;

    // Reset the local encountered bitset
    int bitset_size = (MAX_ENTITIES + 63) / 64;
    SDL_memset(local_encountered, 0, bitset_size * sizeof(uint64_t));

    // Query with depth range for hierarchical Z-buffering
    spatial_grid_query(&state->grid, sector->bounds,
        state->camera.near_plane, state->camera.far_plane,
        result_indices, &result_count, state->entities.entity_count,
        local_encountered);

    // Ensure sector has enough capacity
    if (result_count > sector->entity_capacity) {
        sector->entity_capacity = result_count * 2;
        sector->entity_indices = (int*)SDL_realloc(sector->entity_indices,
            sector->entity_capacity * sizeof(int));
    }

    // Filter for frustum and copy to sector list
    for (int i = 0; i < result_count; i++) {
        int entity_idx = result_indices[i];

        // Skip if not active
        if (!is_entity_active(&state->entities, entity_idx)) {
            continue;
        }

        int chunk_idx, idx_in_chunk;
        get_entity_location(entity_idx, &chunk_idx, &idx_in_chunk);

        EntityChunk* chunk = &state->entities.chunks[chunk_idx];

        if (is_aabb_in_frustum_simd((float*)state->camera.frustum,
            chunk->x[idx_in_chunk],
            chunk->y[idx_in_chunk],
            (float)chunk->width[idx_in_chunk],
            (float)chunk->height[idx_in_chunk])) {
            sector->entity_indices[sector->entity_count++] = entity_idx;
        }
    }

    PROFILE_END(process_sector);
}

void parallel_collect_visible_entities(GameState* state, int* visible_indices, int* visible_count) {
    PROFILE_START(collect_visible);

    // Initialize scan with prefix sum
    int* prefix_sum = (int*)arena_alloc(&state->frame_arena, (state->sector_count + 1) * sizeof(int));
    prefix_sum[0] = 0;

    // Calculate prefix sum (exclusive scan)
    for (int i = 0; i < state->sector_count; i++) {
        prefix_sum[i + 1] = prefix_sum[i] + state->sectors[i].entity_count;
    }

    // Total count
    *visible_count = prefix_sum[state->sector_count];

    // Create bitset to track duplicates
    int bitset_size = (MAX_ENTITIES + 63) / 64;
    uint64_t* encountered = (uint64_t*)arena_alloc(&state->frame_arena, bitset_size * sizeof(uint64_t));
    SDL_memset(encountered, 0, bitset_size * sizeof(uint64_t));

    // Parallel copy from sectors to result array
#pragma omp parallel for
    for (int i = 0; i < state->sector_count; i++) {
        WorldSector* sector = &state->sectors[i];
        int start_idx = prefix_sum[i];

        for (int j = 0; j < sector->entity_count; j++) {
            int entity_idx = sector->entity_indices[j];

            // Use atomic operation to check/set bit to avoid duplicates
            bool is_duplicate = false;
#pragma omp critical
            {
                if (encountered[entity_idx / 64] & (1ULL << (entity_idx % 64))) {
                    is_duplicate = true;
                }
                else {
                    encountered[entity_idx / 64] |= (1ULL << (entity_idx % 64));
                }
            }

            if (!is_duplicate) {
                visible_indices[start_idx + j] = entity_idx;
            }
        }
    }

    // Count unique entities using a cross-platform approach
    int unique_count = 0;
    for (int i = 0; i < bitset_size; i++) {
        uint64_t v = encountered[i];
        // Count bits using Brian Kernighan's algorithm
        while (v) {
            v &= v - 1; // Clear the least significant bit set
            unique_count++;
        }
    }
    *visible_count = unique_count;

    PROFILE_END(collect_visible);
}

// Initialize game state
GameState* init_game_state(int window_width, int window_height, int world_width, int world_height) {
    GameState* state = (GameState*)SDL_malloc(sizeof(GameState));
    if (!state) return NULL;
    SDL_memset(state, 0, sizeof(GameState));

    // Create window and renderer
    state->window = SDL_CreateWindow("Building War - SDL3", window_width, window_height, 0);
    if (!state->window) {
        SDL_free(state);
        return NULL;
    }

    state->renderer = SDL_CreateRenderer(state->window, NULL);
    if (!state->renderer) {
        SDL_DestroyWindow(state->window);
        SDL_free(state);
        return NULL;
    }

    // Init entity manager (AoSoA design)
    init_entity_manager(&state->entities, MAX_ENTITIES);

    // Init world bounds
    state->world_bounds.x = 0;
    state->world_bounds.y = 0;
    state->world_bounds.w = world_width;
    state->world_bounds.h = world_height;

    // Initialize spatial grid with flat arrays
    init_spatial_grid(&state->grid, world_width, world_height, 256);

    // Init world sectors for parallel processing
    init_sectors(state, 8, 8); // Divide world into 8x8 sectors

    // Initialize arena allocators with auto-expand
    init_arena_allocator(&state->frame_arena, 16 * 1024 * 1024, true); // 16MB frame arena
    init_arena_allocator(&state->persistent_arena, 8 * 1024 * 1024, true); // 8MB persistent arena

    // Init texture atlas with dynamic packing
    init_texture_atlas(&state->atlas, state->renderer, 4096, 4096);

    // Init camera
    state->camera.x = 0;
    state->camera.y = 0;
    state->camera.width = window_width;
    state->camera.height = window_height;
    state->camera.zoom = 1.0f;
    state->camera.near_plane = 0.1f;
    state->camera.far_plane = 1000.0f;
    setup_camera_frustum(&state->camera);

    // Init render batches with preallocated buffers
    state->batch_count = 8;
    state->batches = (RenderBatch*)SDL_malloc(state->batch_count * sizeof(RenderBatch));
    for (int i = 0; i < state->batch_count; i++) {
        state->batches[i].texture_id = i;
        state->batches[i].vertex_capacity = 16384; // Preallocate larger buffers
        state->batches[i].index_capacity = 24576;
        state->batches[i].vertex_count = 0;
        state->batches[i].index_count = 0;
        state->batches[i].vertices = (SDL_Vertex*)SDL_malloc(state->batches[i].vertex_capacity * sizeof(SDL_Vertex));
        state->batches[i].indices = (int*)SDL_malloc(state->batches[i].index_capacity * sizeof(int));

        // Initialize view matrix
        SDL_memset(state->batches[i].view_matrix, 0, 16 * sizeof(float));
        state->batches[i].view_matrix[0] = 1.0f;
        state->batches[i].view_matrix[5] = 1.0f;
        state->batches[i].view_matrix[10] = 1.0f;
        state->batches[i].view_matrix[15] = 1.0f;
    }

    // Init occlusion queries
    state->occlusion_query_capacity = 128;
    state->occlusion_query_count = 0;
    state->occlusion_queries = (OcclusionQueryResult*)SDL_malloc(
        state->occlusion_query_capacity * sizeof(OcclusionQueryResult));

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

    // Init timing
    state->last_frame_time = SDL_GetTicks();
    state->fps = 0.0f;
    state->benchmark_frame = 0;

    // Emscripten benchmark settings
    state->benchmark_running = false;
    state->target_entity_count = 5000;
    state->current_entity_count = 0;
    state->benchmark_duration_seconds = 10;
    state->entities_per_frame = 200;




    // Initialize frame stats
    SDL_memset(&state->frame_stats, 0, sizeof(state->frame_stats));

    return state;
}

// Free game state resources
void free_game_state(GameState* state) {
    if (!state) return;

    // Free entity manager
    free_entity_manager(&state->entities);

    // Free spatial grid
    free_spatial_grid(&state->grid);

    // Free sectors
    free_sectors(state);

    // Free arena allocators
    free_arena_allocator(&state->frame_arena);
    free_arena_allocator(&state->persistent_arena);

    // Free batches
    for (int i = 0; i < state->batch_count; i++) {
        SDL_free(state->batches[i].vertices);
        SDL_free(state->batches[i].indices);
    }
    SDL_free(state->batches);

    // Free atlas
    SDL_DestroyTexture(state->atlas.texture);
    SDL_free(state->atlas.regions);

    // Free grid
    for (int x = 0; x < state->grid_width; x++) {
        SDL_free(state->grid_loaded[x]);
    }
    SDL_free(state->grid_loaded);

    // Free occlusion queries
    SDL_free(state->occlusion_queries);

    // Free SDL resources
    SDL_DestroyRenderer(state->renderer);
    SDL_DestroyWindow(state->window);

    SDL_free(state);
}

// Add an entity to the game world
int add_entity(GameState* state, float x, float y, int width, int height,
    int texture_id, int layer) {
    EntityManager* em = &state->entities;

    // Get chunk and index
    int entity_idx = em->entity_count;
    int chunk_idx = entity_idx / CHUNK_SIZE;
    int idx_in_chunk = entity_idx % CHUNK_SIZE;

    // Ensure we have capacity
    if (chunk_idx >= em->chunk_capacity) {
        em->chunk_capacity *= 2;
        em->chunks = (EntityChunk*)SDL_realloc(em->chunks, em->chunk_capacity * sizeof(EntityChunk));

        // Zero initialize new chunks
        for (int i = em->chunk_count; i < em->chunk_capacity; i++) {
            SDL_memset(&em->chunks[i], 0, sizeof(EntityChunk));
        }
    }

    // Ensure we don't exceed MAX_ENTITIES
    if (entity_idx >= MAX_ENTITIES) {
        ATMLOG("Maximum entity count reached!");
        return -1;
    }

    // Ensure cold data capacity
    if (entity_idx >= em->chunk_capacity * CHUNK_SIZE) {
        int new_capacity = em->chunk_capacity * CHUNK_SIZE;
        em->cold_data = (EntityColdData*)SDL_realloc(em->cold_data, new_capacity * sizeof(EntityColdData));

        // Zero initialize new cold data
        for (int i = em->entity_count; i < new_capacity; i++) {
            SDL_memset(&em->cold_data[i], 0, sizeof(EntityColdData));
        }
    }

    // Update chunk count if needed
    if (chunk_idx >= em->chunk_count) {
        em->chunk_count = chunk_idx + 1;
    }

    // Set entity properties
    EntityChunk* chunk = &em->chunks[chunk_idx];
    chunk->x[idx_in_chunk] = x;
    chunk->y[idx_in_chunk] = y;
    chunk->width[idx_in_chunk] = width;
    chunk->height[idx_in_chunk] = height;
    chunk->texture_id[idx_in_chunk] = texture_id;
    chunk->layer[idx_in_chunk] = layer;

    // Calculate depth for Z-buffer
    float depth = (float)layer; // Simple depth based on layer
    chunk->min_depth[idx_in_chunk] = depth;
    chunk->max_depth[idx_in_chunk] = depth;

    // Determine if entity should be active based on grid loading
    int grid_x = (int)(x / state->grid_cell_size);
    int grid_y = (int)(y / state->grid_cell_size);

    bool should_be_active = false;
    if (grid_x >= 0 && grid_x < state->grid_width &&
        grid_y >= 0 && grid_y < state->grid_height) {
        should_be_active = state->grid_loaded[grid_x][grid_y];
    }

    // Set active state using optimized function
    if (should_be_active) {
        em->active_list[em->active_count++] = entity_idx;
        em->active_bitset[entity_idx / 64] |= (1ULL << (entity_idx % 64));
    }

    // Set creation time in cold data
    em->cold_data[entity_idx].creation_time = (int)SDL_GetTicks();

    // Increment entity count
    em->entity_count++;

    return entity_idx;
}

// Add a texture to the atlas
int add_texture_to_atlas(GameState* state, SDL_Surface* surface, int x, int y) {
    int texture_id = state->atlas.region_count;

    // Ensure capacity
    if (texture_id >= state->atlas.region_capacity) {
        state->atlas.region_capacity *= 2;
        state->atlas.regions = (SDL_FRect*)SDL_realloc(state->atlas.regions,
            state->atlas.region_capacity * sizeof(SDL_FRect));
    }

    // Calculate normalized UV coordinates
    float  atlas_width, atlas_height;
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
    PROFILE_START(update_game);

    // Calculate delta time
    Uint64 current_time = SDL_GetTicks();
    float delta_time = (current_time - state->last_frame_time) / 1000.0f;
    state->last_frame_time = current_time;

    // Smooth FPS calculation
    state->fps = 0.95f * state->fps + 0.05f * (1.0f / delta_time);

    // Reset arena allocator for this frame
    reset_arena(&state->frame_arena);

    // Update dynamic grid loading based on camera position
    update_dynamic_loading(state);

    // Update spatial grid using bitset-based active list
    PROFILE_START(rebuild_grid);
    rebuild_spatial_grid(&state->grid, &state->entities);
    PROFILE_END(rebuild_grid);

    // Update camera frustum planes
    setup_camera_frustum(&state->camera);

    // Allocate encountered bitsets for each thread
    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif

    uint64_t** thread_bitsets = (uint64_t**)arena_alloc(&state->frame_arena,
        num_threads * sizeof(uint64_t*));

    int bitset_size = (MAX_ENTITIES + 63) / 64;
    for (int i = 0; i < num_threads; i++) {
        thread_bitsets[i] = (uint64_t*)arena_alloc(&state->frame_arena,
            bitset_size * sizeof(uint64_t));
        SDL_memset(thread_bitsets[i], 0, bitset_size * sizeof(uint64_t));
    }

    // Process each sector in parallel
    PROFILE_START(process_sectors);
#pragma omp parallel for
    for (int i = 0; i < state->sector_count; i++) {
#ifdef _OPENMP
        int thread_id = omp_get_thread_num();
#else
        int thread_id = 0;
#endif
        process_sector_visibility(state, i, thread_bitsets[thread_id]);
    }
    PROFILE_END(process_sectors);

    PROFILE_END(update_game);

    // Store timing for profiling
    state->frame_stats.update_time = delta_time * 1000.0f;
}

// Render the game
void render_game(GameState* state) {
    PROFILE_START(render_game);

    // Clear screen
    SDL_SetRenderDrawColor(state->renderer, 0, 0, 0, 255);
    SDL_RenderClear(state->renderer);

    // Collect visible entities from all sectors using parallel prefix sum
    int* visible_indices = (int*)arena_alloc(&state->frame_arena, state->entities.entity_count * sizeof(int));
    int visible_count = 0;

    PROFILE_START(collect_visible);
    parallel_collect_visible_entities(state, visible_indices, &visible_count);
    PROFILE_END(collect_visible);

    // Sort visible entities by texture and z-order for batch efficiency
    PROFILE_START(sort_entities);
    SDL_qsort_r(visible_indices, visible_count, sizeof(int),
        compare_entities_for_rendering, &state->entities);
    PROFILE_END(sort_entities);

    // Clear batches
    PROFILE_START(prepare_batches);
    for (int i = 0; i < state->batch_count; i++) {
        state->batches[i].vertex_count = 0;
        state->batches[i].index_count = 0;

        // Copy camera view matrix to batch
        SDL_memcpy(state->batches[i].view_matrix, state->camera.view_matrix, 16 * sizeof(float));
    }

    // Calculate visible rect for screen-space coordinate conversion
    SDL_FRect visible_rect = get_visible_rect(&state->camera);

    // Add visible entities to batches using matrix transformations
    for (int i = 0; i < visible_count; i++) {
        int entity_idx = visible_indices[i];
        int chunk_idx, idx_in_chunk;
        get_entity_location(entity_idx, &chunk_idx, &idx_in_chunk);

        EntityChunk* chunk = &state->entities.chunks[chunk_idx];
        int texture_id = chunk->texture_id[idx_in_chunk];

        // Ensure we have a batch for this texture
        if (texture_id >= state->batch_count) {
            int old_count = state->batch_count;
            state->batch_count = texture_id + 1;
            state->batches = (RenderBatch*)SDL_realloc(state->batches,
                state->batch_count * sizeof(RenderBatch));

            // Init new batches
            for (int j = old_count; j < state->batch_count; j++) {
                state->batches[j].texture_id = j;
                state->batches[j].vertex_capacity = 16384;
                state->batches[j].index_capacity = 24576;
                state->batches[j].vertex_count = 0;
                state->batches[j].index_count = 0;
                state->batches[j].vertices = (SDL_Vertex*)SDL_malloc(
                    state->batches[j].vertex_capacity * sizeof(SDL_Vertex));
                state->batches[j].indices = (int*)SDL_malloc(
                    state->batches[j].index_capacity * sizeof(int));

                // Copy camera view matrix
                SDL_memcpy(state->batches[j].view_matrix, state->camera.view_matrix, 16 * sizeof(float));
            }
        }

        // Use world coordinates directly with view matrix transformation
        float world_x = chunk->x[idx_in_chunk];
        float world_y = chunk->y[idx_in_chunk];
        float world_w = (float)chunk->width[idx_in_chunk];
        float world_h = (float)chunk->height[idx_in_chunk];

        // Get texture region from atlas
        SDL_FRect tex_region = state->atlas.regions[texture_id];

        // Add to batch with view matrix transformation
        SDL_FColor color = { 1.0f, 1.0f, 1.0f, 1.0f };
        add_to_batch(&state->batches[texture_id], world_x, world_y, world_w, world_h,
            tex_region, color, state->camera.view_matrix);
    }
    PROFILE_END(prepare_batches);

    // Render all batches
    PROFILE_START(draw_batches);
    for (int i = 0; i < state->batch_count; i++) {
        if (state->batches[i].vertex_count > 0) {
            // Single draw call per batch!
            SDL_RenderGeometry(state->renderer, state->atlas.texture,
                state->batches[i].vertices, state->batches[i].vertex_count,
                state->batches[i].indices, state->batches[i].index_count);
        }
    }
    PROFILE_END(draw_batches);

    PROFILE_END(render_game);

    // Store timing stats
    state->frame_stats.render_time = state->frame_stats.draw_time;
}

// Generate random entity for benchmark
void generate_random_entity(GameState* state) {
    float x, y;

    if (state->benchmark_running) {
        // Place entities within initial camera view
        SDL_FRect visible = get_visible_rect(&state->camera);
        x = visible.x + (rand() % (int)visible.w);
        y = visible.y + (rand() % (int)visible.h);
    }
    else {
        x = (float)(rand() % (int)state->world_bounds.w);
        y = (float)(rand() % (int)state->world_bounds.h);
    }

    // Clamp to ensure within world bounds
    x = SDL_clamp(x, 0.0f, state->world_bounds.w - 1);
    y = SDL_clamp(y, 0.0f, state->world_bounds.h - 1);

    int width = 16 + rand() % 48;
    int height = 16 + rand() % 48;
    int texture_id = rand() % 8;
    int layer = rand() % 5;

    add_entity(state, x, y, width, height, texture_id, layer);
}

// Start benchmark - Emscripten friendly
void start_benchmark(GameState* state, int entity_count, int duration_seconds) {
    ATMLOG("Starting benchmark with target %d entities for %d seconds...",
        entity_count, duration_seconds);

    // Reset entity count
    state->entities.entity_count = 0;
    state->entities.chunk_count = 0;
    state->entities.active_count = 0;

    // Reset active bitset
    int bitset_size = (MAX_ENTITIES + 63) / 64;
    SDL_memset(state->entities.active_bitset, 0, bitset_size * sizeof(uint64_t));

    // Clear spatial grid
    clear_spatial_grid(&state->grid);

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
    state->camera.x += 1.0f;
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

        // Calculate visible entities count
        int visible_count = 0;
        for (int i = 0; i < state->sector_count; i++) {
            visible_count += state->sectors[i].entity_count;
        }

        ATMLOG("Entities: %d/%d - FPS: %.2f (%.2f ms/frame) - Visible: %d - Active: %d",
            state->current_entity_count, state->target_entity_count,
            current_fps,
            state->frames_since_report > 0 ? (state->total_ms / state->frames_since_report) : 0,
            visible_count, state->entities.active_count);

#ifdef ENABLE_PROFILING
        ATMLOG("  Update: %.2f ms, Render: %.2f ms, Grid: %.2f ms, Frustum: %.2f ms, Batching: %.2f ms",
            state->frame_stats.update_time,
            state->frame_stats.render_time,
            state->frame_stats.grid_update_time,
            state->frame_stats.frustum_culling_time,
            state->frame_stats.batch_preparation_time);

        ATMLOG("  Arena high-water: %zu KB, Allocations: %zu",
            state->frame_arena.high_water_mark / 1024,
            state->frame_arena.allocation_count);
#endif

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

        // Add to atlas with dynamic packing
        add_texture_to_atlas_dynamic(&state->atlas, state->renderer, surface);

        // Clean up
        SDL_DestroySurface(surface);
    }

    // Set app state
    *appstate = state;

    // Start benchmark with appropriate entity count
    int entity_count = 200000;

#ifdef EMSCRIPTEN
    // Use a much smaller count for web
    entity_count = 5000;
#endif

    start_benchmark(state, entity_count, 30);

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

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event) {
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
#ifdef EMSCRIPTEN
                start_benchmark(state, 5000, 10);
#else
                start_benchmark(state, 200000, 30);
#endif
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