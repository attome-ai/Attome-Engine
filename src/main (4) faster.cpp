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

// Alignment for memory
#define CACHE_LINE_SIZE 64

// Forward declarations
typedef struct GameState GameState;

// Entity manager using Structure of Arrays (SoA) for cache efficiency
typedef struct {
    float* x;              // x positions - aligned
    float* y;              // y positions - aligned
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

// Rendering batch (groups by texture)
typedef struct {
    int texture_id;
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

// Aligned memory allocation
void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = NULL;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = NULL;
    }
#endif
    return ptr;
}

// Aligned memory free
void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Initialize spatial grid for entity queries
void init_spatial_grid(SpatialGrid* grid, float world_width, float world_height, float cell_size) {
    grid->cell_size = cell_size;
    grid->width = (int)ceil(world_width / cell_size);
    grid->height = (int)ceil(world_height / cell_size);
    grid->total_cells = grid->width * grid->height;

    // Allocate cells array as a flat array for better cache locality
    grid->cells = (int**)aligned_malloc(grid->total_cells * sizeof(int*), CACHE_LINE_SIZE);
    grid->cell_counts = (int*)aligned_malloc(grid->total_cells * sizeof(int), CACHE_LINE_SIZE);
    grid->cell_capacities = (int*)aligned_malloc(grid->total_cells * sizeof(int), CACHE_LINE_SIZE);

    // Initialize cells
    for (int i = 0; i < grid->total_cells; i++) {
        grid->cell_counts[i] = 0;
        grid->cell_capacities[i] = 16; // Initial capacity
        grid->cells[i] = (int*)aligned_malloc(grid->cell_capacities[i] * sizeof(int), CACHE_LINE_SIZE);
    }
}

// Free spatial grid
void free_spatial_grid(SpatialGrid* grid) {
    for (int i = 0; i < grid->total_cells; i++) {
        aligned_free(grid->cells[i]);
    }
    aligned_free(grid->cells);
    aligned_free(grid->cell_counts);
    aligned_free(grid->cell_capacities);
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
        int* new_cell = (int*)aligned_malloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);

        // Copy existing data to new array
        for (int i = 0; i < grid->cell_counts[cell_idx]; i++) {
            new_cell[i] = grid->cells[cell_idx][i];
        }

        // Free old array and update pointers/capacity
        aligned_free(grid->cells[cell_idx]);
        grid->cells[cell_idx] = new_cell;
        grid->cell_capacities[cell_idx] = new_capacity;
    }

    // Add entity to cell
    grid->cells[cell_idx][grid->cell_counts[cell_idx]++] = entity_idx;
}

// Query entities in a region
void spatial_grid_query(SpatialGrid* grid, SDL_FRect query_rect,
    int* result_indices, int* result_count, int max_results) {
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
    for (int y = start_y; y <= end_y; y++) {
        for (int x = start_x; x <= end_x; x++) {
            int cell_idx = y * grid->width + x;

            // Add all entities from this cell
            for (int i = 0; i < grid->cell_counts[cell_idx] && *result_count < max_results; i++) {
                int entity_idx = grid->cells[cell_idx][i];
                // Only add valid indices
                if (entity_idx >= 0) {
                    result_indices[(*result_count)++] = entity_idx;
                }
            }
        }
    }
}

// Radix sort for entities (two-pass: sort by texture_id then layer)
void radix_sort_entities(int* indices, int count, EntityManager* entities) {
    // Allocate temporary arrays
    int* temp_indices = (int*)aligned_malloc(count * sizeof(int), CACHE_LINE_SIZE);
    int* count_texture = (int*)aligned_malloc(256 * sizeof(int), CACHE_LINE_SIZE); // Assuming texture_id < 256
    int* count_layer = (int*)aligned_malloc(256 * sizeof(int), CACHE_LINE_SIZE);   // Assuming layer < 256
    
    // Reset counters
    memset(count_texture, 0, 256 * sizeof(int));
    memset(count_layer, 0, 256 * sizeof(int));
    
    // Count textures
    for (int i = 0; i < count; i++) {
        int texture_id = entities->texture_id[indices[i]] & 0xFF;
        count_texture[texture_id]++;
    }

    // Compute prefix sum for textures
    for (int i = 1; i < 256; i++) {
        count_texture[i] += count_texture[i - 1];
    }

    // Sort by texture_id
    for (int i = count - 1; i >= 0; i--) {
        int texture_id = entities->texture_id[indices[i]] & 0xFF;
        temp_indices[--count_texture[texture_id]] = indices[i];
    }

    // For each texture group, sort by layer
    int texture_start = 0;
    for (int t = 0; t < 256; t++) {
        if (t > 0) {
            texture_start = count_texture[t - 1];
        }
        int texture_end = count_texture[t];
        int texture_count = texture_end - texture_start;

        if (texture_count > 1) {
            // Count layers for this texture group
            memset(count_layer, 0, 256 * sizeof(int));
            for (int i = texture_start; i < texture_end; i++) {
                int layer = entities->layer[temp_indices[i]] & 0xFF;
                count_layer[layer]++;
            }

            // Compute prefix sum for layers
            for (int i = 1; i < 256; i++) {
                count_layer[i] += count_layer[i - 1];
            }

            // Sort by layer within this texture group
            for (int i = texture_end - 1; i >= texture_start; i--) {
                int layer = entities->layer[temp_indices[i]] & 0xFF;
                indices[texture_start + (--count_layer[layer])] = temp_indices[i];
            }
        }
        else if (texture_count == 1) {
            // Just copy the single entity
            indices[texture_start] = temp_indices[texture_start];
        }
    }

    // Cleanup
    aligned_free(temp_indices);
    aligned_free(count_texture);
    aligned_free(count_layer);
}

// Initialize entity manager with SoA design (aligned memory)
void init_entity_manager(EntityManager* manager, int initial_capacity) {
    manager->capacity = initial_capacity;
    manager->count = 0;

    // Allocate aligned arrays for better SIMD performance
    manager->x = (float*)aligned_malloc(initial_capacity * sizeof(float), CACHE_LINE_SIZE);
    manager->y = (float*)aligned_malloc(initial_capacity * sizeof(float), CACHE_LINE_SIZE);
    manager->width = (int*)aligned_malloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->height = (int*)aligned_malloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->texture_id = (int*)aligned_malloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->layer = (int*)aligned_malloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE);
    manager->active = (bool*)aligned_malloc(initial_capacity * sizeof(bool), CACHE_LINE_SIZE);
    manager->grid_cell = (int**)aligned_malloc(initial_capacity * sizeof(int*), CACHE_LINE_SIZE);
}

// Free entity manager
void free_entity_manager(EntityManager* manager) {
    aligned_free(manager->x);
    aligned_free(manager->y);
    aligned_free(manager->width);
    aligned_free(manager->height);
    aligned_free(manager->texture_id);
    aligned_free(manager->layer);
    aligned_free(manager->active);
    aligned_free(manager->grid_cell);
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
    atlas->regions = (SDL_FRect*)aligned_malloc(atlas->region_capacity * sizeof(SDL_FRect), CACHE_LINE_SIZE);
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

// Add a quad to a render batch
void add_to_batch(RenderBatch* batch, float x, float y, float w, float h,
    SDL_FRect tex_region, SDL_FColor color) {
    // Ensure we have enough space
    if (batch->vertex_count + 4 > batch->vertex_capacity) {
        batch->vertex_capacity *= 2;
        batch->vertices = (SDL_Vertex*)aligned_malloc(
            batch->vertex_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE);
    }

    if (batch->index_count + 6 > batch->index_capacity) {
        batch->index_capacity *= 2;
        batch->indices = (int*)aligned_malloc(
            batch->index_capacity * sizeof(int), CACHE_LINE_SIZE);
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
    GameState* state = (GameState*)malloc(sizeof(GameState));
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
    init_entity_manager(&state->entities, 10000);

    // Init world bounds
    state->world_bounds.x = 0;
    state->world_bounds.y = 0;
    state->world_bounds.w = world_width;
    state->world_bounds.h = world_height;

    // Init spatial grid (replaces quadtree)
    init_spatial_grid(&state->grid, world_width, world_height, 256.0f);

    // Init texture atlas
    init_texture_atlas(&state->atlas, state->renderer, 2048, 2048);

    // Init camera
    state->camera.x = 0;
    state->camera.y = 0;
    state->camera.width = window_width;
    state->camera.height = window_height;
    state->camera.zoom = 1.0f;

    // Init render batches (for batch rendering)
    state->batch_count = 8;
    state->batches = (RenderBatch*)malloc(state->batch_count * sizeof(RenderBatch));
    for (int i = 0; i < state->batch_count; i++) {
        state->batches[i].texture_id = i;
        state->batches[i].vertex_capacity = 1024;
        state->batches[i].index_capacity = 1536;
        state->batches[i].vertex_count = 0;
        state->batches[i].index_count = 0;
        state->batches[i].vertices = (SDL_Vertex*)aligned_malloc(
            state->batches[i].vertex_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE);
        state->batches[i].indices = (int*)aligned_malloc(
            state->batches[i].index_capacity * sizeof(int), CACHE_LINE_SIZE);
    }

    // Init dynamic grid loading
    state->grid_cell_size = 256.0f;
    state->grid_width = (int)ceil(world_width / state->grid_cell_size);
    state->grid_height = (int)ceil(world_height / state->grid_cell_size);

    // Allocate grid loaded array
    state->grid_loaded = (bool**)malloc(state->grid_width * sizeof(bool*));
    for (int x = 0; x < state->grid_width; x++) {
        state->grid_loaded[x] = (bool*)malloc(state->grid_height * sizeof(bool));
        for (int y = 0; y < state->grid_height; y++) {
            state->grid_loaded[x][y] = false;
        }
    }

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
        aligned_free(state->batches[i].vertices);
        aligned_free(state->batches[i].indices);
    }
    free(state->batches);

    // Free atlas
    SDL_DestroyTexture(state->atlas.texture);
    aligned_free(state->atlas.regions);

    // Free grid
    for (int x = 0; x < state->grid_width; x++) {
        free(state->grid_loaded[x]);
    }
    free(state->grid_loaded);

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
        float* new_x = (float*)aligned_malloc(new_capacity * sizeof(float), CACHE_LINE_SIZE);
        float* new_y = (float*)aligned_malloc(new_capacity * sizeof(float), CACHE_LINE_SIZE);
        int* new_width = (int*)aligned_malloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        int* new_height = (int*)aligned_malloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        int* new_texture_id = (int*)aligned_malloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        int* new_layer = (int*)aligned_malloc(new_capacity * sizeof(int), CACHE_LINE_SIZE);
        bool* new_active = (bool*)aligned_malloc(new_capacity * sizeof(bool), CACHE_LINE_SIZE);
        int** new_grid_cell = (int**)aligned_malloc(new_capacity * sizeof(int*), CACHE_LINE_SIZE);

        // Copy existing data
        memcpy(new_x, em->x, em->count * sizeof(float));
        memcpy(new_y, em->y, em->count * sizeof(float));
        memcpy(new_width, em->width, em->count * sizeof(int));
        memcpy(new_height, em->height, em->count * sizeof(int));
        memcpy(new_texture_id, em->texture_id, em->count * sizeof(int));
        memcpy(new_layer, em->layer, em->count * sizeof(int));
        memcpy(new_active, em->active, em->count * sizeof(bool));
        memcpy(new_grid_cell, em->grid_cell, em->count * sizeof(int*));

        // Free old arrays
        aligned_free(em->x);
        aligned_free(em->y);
        aligned_free(em->width);
        aligned_free(em->height);
        aligned_free(em->texture_id);
        aligned_free(em->layer);
        aligned_free(em->active);
        aligned_free(em->grid_cell);

        // Update pointers
        em->x = new_x;
        em->y = new_y;
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
        state->atlas.regions = (SDL_FRect*)aligned_malloc(
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
#if HAVE_SIMD
    // Process multiple entities at once with SIMD
#if defined(__x86_64__) || defined(_M_X64)
// SSE implementation for x86
    for (int i = 0; i < count; i += 4) {
        int remaining = (count - i < 4) ? count - i : 4;

        if (remaining == 4) {
            // Load 4 world_x, world_y values
            __m128 vx = _mm_load_ps(&world_x[i]);
            __m128 vy = _mm_load_ps(&world_y[i]);
            __m128 vw = _mm_cvtepi32_ps(_mm_load_si128((__m128i*) & width[i]));
            __m128 vh = _mm_cvtepi32_ps(_mm_load_si128((__m128i*) & height[i]));

            // Calculate screen coordinates
            __m128 vrect_x = _mm_set1_ps(visible_rect_x);
            __m128 vrect_y = _mm_set1_ps(visible_rect_y);
            __m128 vzoom = _mm_set1_ps(zoom);

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
    for (int i = 0; i < count; i += 4) {
        int remaining = (count - i < 4) ? count - i : 4;

        if (remaining == 4) {
            // Load 4 world_x, world_y values
            float32x4_t vx = vld1q_f32(&world_x[i]);
            float32x4_t vy = vld1q_f32(&world_y[i]);
            float32x4_t vw = vcvtq_f32_s32(vld1q_s32(&width[i]));
            float32x4_t vh = vcvtq_f32_s32(vld1q_s32(&height[i]));

            // Calculate screen coordinates
            float32x4_t vrect_x = vdupq_n_f32(visible_rect_x);
            float32x4_t vrect_y = vdupq_n_f32(visible_rect_y);
            float32x4_t vzoom = vdupq_n_f32(zoom);

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

// Merge consecutive batches with the same texture
void merge_batches(RenderBatch* batches, int* batch_count) {
    int write_idx = 0;

    for (int read_idx = 1; read_idx < *batch_count; read_idx++) {
        // If current batch uses same texture as previous one, merge them
        if (batches[write_idx].texture_id == batches[read_idx].texture_id) {
            // Ensure enough capacity in the destination batch
            if (batches[write_idx].vertex_count + batches[read_idx].vertex_count > batches[write_idx].vertex_capacity) {
                batches[write_idx].vertex_capacity = batches[write_idx].vertex_count + batches[read_idx].vertex_count;
                batches[write_idx].vertices = (SDL_Vertex*)aligned_malloc(
                    batches[write_idx].vertex_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE);
            }

            if (batches[write_idx].index_count + batches[read_idx].index_count > batches[write_idx].index_capacity) {
                batches[write_idx].index_capacity = batches[write_idx].index_count + batches[read_idx].index_count;
                batches[write_idx].indices = (int*)aligned_malloc(
                    batches[write_idx].index_capacity * sizeof(int), CACHE_LINE_SIZE);
            }

            // Adjust indices for the merged batch
            int base_vertex = batches[write_idx].vertex_count;
            for (int i = 0; i < batches[read_idx].index_count; i++) {
                batches[write_idx].indices[batches[write_idx].index_count++] =
                    batches[read_idx].indices[i] + base_vertex;
            }

            // Copy vertices
            for (int i = 0; i < batches[read_idx].vertex_count; i++) {
                batches[write_idx].vertices[batches[write_idx].vertex_count++] =
                    batches[read_idx].vertices[i];
            }
        }
        else {
            // Different texture, move to next write position
            write_idx++;
            if (write_idx != read_idx) {
                batches[write_idx] = batches[read_idx];
            }
        }
    }

    *batch_count = write_idx + 1;
}

// Render the game
void render_game(GameState* state) {
    // Clear screen
    SDL_SetRenderDrawColor(state->renderer, 0, 0, 0, 255);
    SDL_RenderClear(state->renderer);

    // Get visible area for culling
    SDL_FRect visible_rect = get_visible_rect(&state->camera);

    // Query visible entities from spatial grid
    int* visible_indices = (int*)aligned_malloc(state->entities.count * sizeof(int), CACHE_LINE_SIZE);
    int visible_count = 0;

    spatial_grid_query(&state->grid, visible_rect, visible_indices, &visible_count, state->entities.count);

    // Sort visible entities by texture and z-order using radix sort
    radix_sort_entities(visible_indices, visible_count, &state->entities);

    // Clear batches
    for (int i = 0; i < state->batch_count; i++) {
        state->batches[i].vertex_count = 0;
        state->batches[i].index_count = 0;
    }

    // Temporary arrays for SIMD transformation
    float* screen_x = (float*)aligned_malloc(visible_count * sizeof(float), CACHE_LINE_SIZE);
    float* screen_y = (float*)aligned_malloc(visible_count * sizeof(float), CACHE_LINE_SIZE);
    float* screen_w = (float*)aligned_malloc(visible_count * sizeof(float), CACHE_LINE_SIZE);
    float* screen_h = (float*)aligned_malloc(visible_count * sizeof(float), CACHE_LINE_SIZE);
    float* world_x = (float*)aligned_malloc(visible_count * sizeof(float), CACHE_LINE_SIZE);
    float* world_y = (float*)aligned_malloc(visible_count * sizeof(float), CACHE_LINE_SIZE);
    float* width = (float*)aligned_malloc(visible_count * sizeof(int), CACHE_LINE_SIZE);
    float* height = (float*)aligned_malloc(visible_count * sizeof(int), CACHE_LINE_SIZE);

    // Copy entity data to contiguous arrays for SIMD processing
    for (int i = 0; i < visible_count; i++) {
        int entity_idx = visible_indices[i];
        world_x[i] = state->entities.x[entity_idx];
        world_y[i] = state->entities.y[entity_idx];
        width[i] = state->entities.width[entity_idx];
        height[i] = state->entities.height[entity_idx];
    }

    // Calculate screen coordinates using SIMD
    calculate_screen_coordinates(world_x, world_y, width, height,
        screen_x, screen_y, screen_w, screen_h,
        visible_count, visible_rect.x, visible_rect.y, state->camera.zoom);

    // Add visible entities to batches (no need to check active flag - spatial grid has only active entities)
    for (int i = 0; i < visible_count; i++) {
        int entity_idx = visible_indices[i];
        int texture_id = state->entities.texture_id[entity_idx];

        // Ensure we have a batch for this texture
        if (texture_id >= state->batch_count) {
            int old_count = state->batch_count;
            int new_count = texture_id + 1;

            // Use safe reallocation approach
            RenderBatch* new_batches = (RenderBatch*)malloc(new_count * sizeof(RenderBatch));
            if (!new_batches) {
                // Handle allocation failure
                ATMLOG("Failed to allocate memory for new batches");
                return; // Or handle error appropriately
            }

            // Copy existing batches
            for (int j = 0; j < old_count; j++) {
                new_batches[j] = state->batches[j];
            }

            // Init new batches
            for (int j = old_count; j < new_count; j++) {
                new_batches[j].texture_id = j;
                new_batches[j].vertex_capacity = 1024;
                new_batches[j].index_capacity = 1536;
                new_batches[j].vertex_count = 0;
                new_batches[j].index_count = 0;
                new_batches[j].vertices = (SDL_Vertex*)aligned_malloc(
                    new_batches[j].vertex_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE);
                new_batches[j].indices = (int*)aligned_malloc(
                    new_batches[j].index_capacity * sizeof(int), CACHE_LINE_SIZE);
            }

            // Free old batches array and update state
            free(state->batches);
            state->batches = new_batches;
            state->batch_count = new_count;
        }
        // Use pre-calculated screen coordinates
        float screen_pos_x = screen_x[i];
        float screen_pos_y = screen_y[i];
        float screen_pos_w = screen_w[i];
        float screen_pos_h = screen_h[i];

        // Get texture region from atlas
        SDL_FRect tex_region = state->atlas.regions[texture_id];

        // Add to batch
        SDL_FColor color = { 1.0f, 1.0f, 1.0f, 1.0f };
        add_to_batch(&state->batches[texture_id], screen_pos_x, screen_pos_y, screen_pos_w, screen_pos_h,
            tex_region, color);
    }

    // Free temporary arrays
    aligned_free(screen_x);
    aligned_free(screen_y);
    aligned_free(screen_w);
    aligned_free(screen_h);
    aligned_free(world_x);
    aligned_free(world_y);
    aligned_free(width);
    aligned_free(height);

    // Merge batches with the same texture
    int active_batch_count = state->batch_count;
    merge_batches(state->batches, &active_batch_count);

    // Render batches
    for (int i = 0; i < active_batch_count; i++) {
        if (state->batches[i].vertex_count > 0) {
            // Single draw call per batch!
            SDL_RenderGeometry(state->renderer, state->atlas.texture,
                state->batches[i].vertices, state->batches[i].vertex_count,
                state->batches[i].indices, state->batches[i].index_count);
        }
    }

    // Show benchmark info
    char fps_text[64];
    sprintf(fps_text, "FPS: %.1f - Entities: %d - Visible: %d",
        state->fps, state->entities.count, visible_count);

    // Would render text here with proper SDL_ttf

    aligned_free(visible_indices);
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
        int* visible_indices = (int*)malloc(state->entities.count * sizeof(int));
        int visible_count = 0;
        spatial_grid_query(&state->grid, visible_rect,
            visible_indices, &visible_count, state->entities.count);
        free(visible_indices);

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
        Uint32 color = SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format),0,
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