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



// Forward declarations
typedef struct GameState GameState;

// Entity manager using Structure of Arrays (SoA) for cache efficiency
typedef struct {
    float* x;              // x positions
    float* y;              // y positions
    int* width;            // widths
    int* height;           // heights
    int* texture_id;       // texture IDs
    int* layer;            // z-ordering/layers
    bool* active;          // is entity active/visible
    int capacity;          // allocated capacity
    int count;             // actual count of entities
} EntityManager;

// Quadtree for spatial partitioning
typedef struct QuadTreeNode {
    SDL_FRect bounds;
    int* entity_indices;   // Indices to entities in this node
    int entity_count;
    int entity_capacity;
    struct QuadTreeNode* children[4];
    bool is_leaf;
} QuadTreeNode;

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
    QuadTreeNode* world_quadtree;
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

// Initialize entity manager with SoA design
void init_entity_manager(EntityManager* manager, int initial_capacity) {
    manager->capacity = initial_capacity;
    manager->count = 0;

    // Allocate arrays
    manager->x = (float*)malloc(initial_capacity * sizeof(float));
    manager->y = (float*)malloc(initial_capacity * sizeof(float));
    manager->width = (int*)malloc(initial_capacity * sizeof(int));
    manager->height = (int*)malloc(initial_capacity * sizeof(int));
    manager->texture_id = (int*)malloc(initial_capacity * sizeof(int));
    manager->layer = (int*)malloc(initial_capacity * sizeof(int));
    manager->active = (bool*)malloc(initial_capacity * sizeof(bool));
}

// Free entity manager
void free_entity_manager(EntityManager* manager) {
    free(manager->x);
    free(manager->y);
    free(manager->width);
    free(manager->height);
    free(manager->texture_id);
    free(manager->layer);
    free(manager->active);
}

// Create a quadtree node
QuadTreeNode* create_quadtree_node(SDL_FRect bounds) {
    QuadTreeNode* node = (QuadTreeNode*)malloc(sizeof(QuadTreeNode));
    node->bounds = bounds;
    node->is_leaf = true;
    node->entity_capacity = 16;  // Initial capacity
    node->entity_count = 0;
    node->entity_indices = (int*)malloc(node->entity_capacity * sizeof(int));

    // Initialize children to NULL
    for (int i = 0; i < 4; i++) {
        node->children[i] = NULL;
    }

    return node;
}

// Free quadtree
void free_quadtree(QuadTreeNode* node) {
    if (!node) return;

    for (int i = 0; i < 4; i++) {
        if (node->children[i]) {
            free_quadtree(node->children[i]);
        }
    }

    free(node->entity_indices);
    free(node);
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

// Split a quadtree node
void split_node(QuadTreeNode* node) {
    if (!node->is_leaf) return;

    float x = node->bounds.x;
    float y = node->bounds.y;
    float w = node->bounds.w / 2;
    float h = node->bounds.h / 2;

    // Create four children nodes
    SDL_FRect rect;

    // Top-left
    rect.x = x;
    rect.y = y;
    rect.w = w;
    rect.h = h;
    node->children[0] = create_quadtree_node(rect);

    // Top-right
    rect.x = x + w;
    rect.y = y;
    rect.w = w;
    rect.h = h;
    node->children[1] = create_quadtree_node(rect);

    // Bottom-left
    rect.x = x;
    rect.y = y + h;
    rect.w = w;
    rect.h = h;
    node->children[2] = create_quadtree_node(rect);

    // Bottom-right
    rect.x = x + w;
    rect.y = y + h;
    rect.w = w;
    rect.h = h;
    node->children[3] = create_quadtree_node(rect);

    node->is_leaf = false;
}

// Insert an entity into the quadtree
void quadtree_insert(QuadTreeNode* node, int entity_idx, EntityManager* entities) {
    // Get entity bounds
    float x = entities->x[entity_idx];
    float y = entities->y[entity_idx];
    float w = (float)entities->width[entity_idx];
    float h = (float)entities->height[entity_idx];
    SDL_FRect entity_bounds = { x, y, w, h };

    // Check if entity is in this node
    if (!rects_intersect(node->bounds, entity_bounds)) {
        return;
    }

    // If this is a leaf node, add the entity
    if (node->is_leaf) {
        // Split if needed and entity count is high
        if (node->entity_count >= node->entity_capacity &&
            node->bounds.w > 64 && node->bounds.h > 64) {  // Don't split tiny nodes
            split_node(node);

            // Try to insert into children
            for (int i = 0; i < 4; i++) {
                quadtree_insert(node->children[i], entity_idx, entities);
            }
            return;
        }

        // Make sure we have capacity
        if (node->entity_count >= node->entity_capacity) {
            node->entity_capacity *= 2;
            node->entity_indices = (int*)realloc(node->entity_indices,
                node->entity_capacity * sizeof(int));
        }

        // Add entity to this node
        node->entity_indices[node->entity_count++] = entity_idx;
    }
    else {
        // Not a leaf, so insert into appropriate children
        for (int i = 0; i < 4; i++) {
            quadtree_insert(node->children[i], entity_idx, entities);
        }
    }
}

// Query entities in a region
void quadtree_query(QuadTreeNode* node, SDL_FRect query_rect,
    int* result_indices, int* result_count, int max_results,
    EntityManager* entities) {
    // Check if this node intersects the query rect
    if (!rects_intersect(node->bounds, query_rect)) {
        return;
    }

    // Add entities from this node that intersect the query rect
    for (int i = 0; i < node->entity_count; i++) {
        int entity_idx = node->entity_indices[i];
        SDL_FRect entity_bounds = {
            entities->x[entity_idx],
            entities->y[entity_idx],
            (float)entities->width[entity_idx],
            (float)entities->height[entity_idx]
        };

        if (rects_intersect(entity_bounds, query_rect) && *result_count < max_results) {
            result_indices[(*result_count)++] = entity_idx;
        }
    }

    // If this is not a leaf, query children
    if (!node->is_leaf) {
        for (int i = 0; i < 4; i++) {
            quadtree_query(node->children[i], query_rect,
                result_indices, result_count, max_results, entities);
        }
    }
}

// Clear the quadtree for next frame
void quadtree_clear(QuadTreeNode* node) {
    node->entity_count = 0;

    if (!node->is_leaf) {
        for (int i = 0; i < 4; i++) {
            quadtree_clear(node->children[i]);
        }
    }
}

// Initialize a texture atlas
void init_texture_atlas(TextureAtlas* atlas, SDL_Renderer* renderer, int width, int height) {
    atlas->texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_TARGET, width, height);
    SDL_SetTextureBlendMode(atlas->texture, SDL_BLENDMODE_BLEND);
    atlas->region_capacity = 32;
    atlas->region_count = 0;
    atlas->regions = (SDL_FRect*)malloc(atlas->region_capacity * sizeof(SDL_FRect));
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

// Compare function for z-order and texture sorting
int compare_entities_for_rendering(const void* a, const void* b) {
    int* idx_a = (int*)a;
    int* idx_b = (int*)b;
    EntityManager* entities = (EntityManager*)SDL_GetPointerProperty(SDL_GetGlobalProperties(), "entities", NULL);

    // First sort by texture (for batch efficiency)
    if (entities->texture_id[*idx_a] != entities->texture_id[*idx_b]) {
        return entities->texture_id[*idx_a] - entities->texture_id[*idx_b];
    }

    // Then by z-layer
    return entities->layer[*idx_a] - entities->layer[*idx_b];
}

// Add a quad to a render batch
void add_to_batch(RenderBatch* batch, float x, float y, float w, float h,
    SDL_FRect tex_region, SDL_FColor color) {
    // Ensure we have enough space
    if (batch->vertex_count + 4 > batch->vertex_capacity) {
        batch->vertex_capacity *= 2;
        batch->vertices = (SDL_Vertex*)realloc(batch->vertices,
            batch->vertex_capacity * sizeof(SDL_Vertex));
    }

    if (batch->index_count + 6 > batch->index_capacity) {
        batch->index_capacity *= 2;
        batch->indices = (int*)realloc(batch->indices,
            batch->index_capacity * sizeof(int));
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

    // Create world quadtree for spatial partitioning
    state->world_quadtree = create_quadtree_node(state->world_bounds);

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
        state->batches[i].vertices = (SDL_Vertex*)malloc(state->batches[i].vertex_capacity * sizeof(SDL_Vertex));
        state->batches[i].indices = (int*)malloc(state->batches[i].index_capacity * sizeof(int));
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

    // Store entities pointer in global properties for sorting
    SDL_SetPointerProperty(SDL_GetGlobalProperties(), "entities", &state->entities);

    return state;
}

// Free game state resources
void free_game_state(GameState* state) {
    if (!state) return;

    // Free entity manager
    free_entity_manager(&state->entities);

    // Free quadtree
    free_quadtree(state->world_quadtree);

    // Free batches
    for (int i = 0; i < state->batch_count; i++) {
        free(state->batches[i].vertices);
        free(state->batches[i].indices);
    }
    free(state->batches);

    // Free atlas
    SDL_DestroyTexture(state->atlas.texture);
    free(state->atlas.regions);

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
        em->capacity *= 2;
        em->x = (float*)realloc(em->x, em->capacity * sizeof(float));
        em->y = (float*)realloc(em->y, em->capacity * sizeof(float));
        em->width = (int*)realloc(em->width, em->capacity * sizeof(int));
        em->height = (int*)realloc(em->height, em->capacity * sizeof(int));
        em->texture_id = (int*)realloc(em->texture_id, em->capacity * sizeof(int));
        em->layer = (int*)realloc(em->layer, em->capacity * sizeof(int));
        em->active = (bool*)realloc(em->active, em->capacity * sizeof(bool));
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

    // Add to quadtree for spatial partitioning
    quadtree_insert(state->world_quadtree, entity_idx, em);

    return entity_idx;
}

// Add a texture to the atlas
int add_texture_to_atlas(GameState* state, SDL_Surface* surface, int x, int y) {
    int texture_id = state->atlas.region_count;

    // Ensure capacity
    if (texture_id >= state->atlas.region_capacity) {
        state->atlas.region_capacity *= 2;
        state->atlas.regions = (SDL_FRect*)realloc(state->atlas.regions,
            state->atlas.region_capacity * sizeof(SDL_FRect));
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

    // Clear and rebuild quadtree
    quadtree_clear(state->world_quadtree);

    // Add all active entities to quadtree
    for (int i = 0; i < state->entities.count; i++) {
        if (state->entities.active[i]) {
            quadtree_insert(state->world_quadtree, i, &state->entities);
        }
    }
}

// Render the game
void render_game(GameState* state) {
    // Clear screen
    SDL_SetRenderDrawColor(state->renderer, 0, 0, 0, 255);
    SDL_RenderClear(state->renderer);

    // Get visible area for culling
    SDL_FRect visible_rect = get_visible_rect(&state->camera);

    // Query visible entities from quadtree (spatial partitioning)
    int* visible_indices = (int*)malloc(state->entities.count * sizeof(int));
    int visible_count = 0;
    quadtree_query(state->world_quadtree, visible_rect,
        visible_indices, &visible_count, state->entities.count,
        &state->entities);

    // Sort visible entities by texture and z-order
    qsort(visible_indices, visible_count, sizeof(int), compare_entities_for_rendering);

    // Clear batches
    for (int i = 0; i < state->batch_count; i++) {
        state->batches[i].vertex_count = 0;
        state->batches[i].index_count = 0;
    }

    // Add visible entities to batches
    for (int i = 0; i < visible_count; i++) {
        int entity_idx = visible_indices[i];

        // Skip inactive entities
        if (!state->entities.active[entity_idx]) {
            continue;
        }

        int texture_id = state->entities.texture_id[entity_idx];

        // Ensure we have a batch for this texture
        if (texture_id >= state->batch_count) {
            int old_count = state->batch_count;
            state->batch_count = texture_id + 1;
            state->batches = (RenderBatch*)realloc(state->batches,
                state->batch_count * sizeof(RenderBatch));

            // Init new batches
            for (int j = old_count; j < state->batch_count; j++) {
                state->batches[j].texture_id = j;
                state->batches[j].vertex_capacity = 1024;
                state->batches[j].index_capacity = 1536;
                state->batches[j].vertex_count = 0;
                state->batches[j].index_count = 0;
                state->batches[j].vertices = (SDL_Vertex*)malloc(
                    state->batches[j].vertex_capacity * sizeof(SDL_Vertex));
                state->batches[j].indices = (int*)malloc(
                    state->batches[j].index_capacity * sizeof(int));
            }
        }

        // Convert to screen-space coordinates
        float screen_x = (state->entities.x[entity_idx] - visible_rect.x) * state->camera.zoom;
        float screen_y = (state->entities.y[entity_idx] - visible_rect.y) * state->camera.zoom;
        float screen_w = state->entities.width[entity_idx] * state->camera.zoom;
        float screen_h = state->entities.height[entity_idx] * state->camera.zoom;

        // Get texture region from atlas
        SDL_FRect tex_region = state->atlas.regions[texture_id];

        // Add to batch
        SDL_FColor color = { 1.0f, 1.0f, 1.0f, 1.0f };
        add_to_batch(&state->batches[texture_id], screen_x, screen_y, screen_w, screen_h,
            tex_region, color);
    }

    for (int i = 0; i < state->batch_count; i++) {
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

    free(visible_indices);
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
        quadtree_query(state->world_quadtree, visible_rect,
            visible_indices, &visible_count, state->entities.count,
            &state->entities);
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


    ATMBufferPoolRunAllTests();
    ATMByteBufferRunAllTests();

    ATMBufferPoolRunAllBenchmarks();
    ATMByteBufferRunAllBenchmarks();
    
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