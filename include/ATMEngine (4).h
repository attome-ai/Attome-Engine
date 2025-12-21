// ====== ENGINE.H ======
#ifndef ENGINE_H
#define ENGINE_H

#include <SDL3/SDL.h>
#include <stdbool.h>

// Forward declarations
typedef struct Engine Engine;
typedef struct EntityManager EntityManager;
typedef struct SpatialGrid SpatialGrid;
typedef struct RenderBatch RenderBatch;
typedef struct TextureAtlas TextureAtlas;
typedef struct Camera Camera;
typedef struct BufferPool BufferPool;

// Alignment for memory
#define CACHE_LINE_SIZE 64

// Ultra-optimized child system
#define MAX_CHILDREN 8  // Fixed number of children slots per entity

// Entity manager using Structure of Arrays (SoA) for cache efficiency
typedef struct EntityManager {
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

    // Minimal child system with fixed arrays (extreme performance)
    int* parent;           // Parent entity ID (-1 for none) - aligned
    int* child_count;      // Number of children (up to MAX_CHILDREN) - aligned
    int(*children)[MAX_CHILDREN]; // Fixed array of child IDs - aligned (ultra cache-friendly)
    float(*rel_x)[MAX_CHILDREN];  // Relative x offsets - aligned (cache-friendly)
    float(*rel_y)[MAX_CHILDREN];  // Relative y offsets - aligned (cache-friendly)

    // Transform update tracking
    bool* is_transform_dirty; // Tracks which entities need transform updates
    int* update_queue;       // Queue of entities needing updates
    int update_queue_count;  // Number of entities in update queue
    int update_queue_capacity; // Capacity of update queue

    int capacity;          // allocated capacity
    int count;             // actual count of entities
} EntityManager;
\

// Spatial grid for efficient entity queries (replaces quadtree)
typedef struct SpatialGrid {
    int** cells;           // 2D array of entity index arrays
    int* cell_counts;      // Number of entities in each cell
    int* cell_capacities;  // Capacity of each cell
    float cell_size;       // Size of each cell
    int width, height;     // Grid dimensions
    int total_cells;       // Total number of cells
} SpatialGrid;

// Rendering batch (groups by texture and layer)
typedef struct RenderBatch {
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
typedef struct TextureAtlas {
    SDL_Texture* texture;
    SDL_FRect* regions;    // UV regions for each subtexture
    int region_count;
    int region_capacity;
} TextureAtlas;

// Camera for culling
typedef struct Camera {
    float x, y;            // Position
    float width, height;   // Viewport dimensions
    float zoom;            // Zoom level
} Camera;

// Reusable buffer pool for temporary allocations
typedef struct BufferPool {
    void** buffers;
    int count;
    int capacity;
    size_t buffer_size;
} BufferPool;

// Main engine struct
typedef struct Engine {
    SDL_Window* window;
    SDL_Renderer* renderer;
    EntityManager entities;
    SpatialGrid grid;
    RenderBatch* batches;
    int batch_count;
    TextureAtlas atlas;
    Camera camera;
    SDL_FRect world_bounds;
    float grid_cell_size;
    bool** grid_loaded;
    int grid_width, grid_height;
    Uint64 last_frame_time;
    float fps;

    // Reusable buffer pools
    BufferPool entity_indices_pool;
    BufferPool screen_coords_pool;

    // Transform buffers for hierarchy updates
    float* transform_buffer_x;
    float* transform_buffer_y;
    int transform_buffer_capacity;
} Engine;






// Engine initialization and management
Engine* engine_create(int window_width, int window_height, int world_width, int world_height, int cell_size = 128 * 8);
void engine_destroy(Engine* engine);
void engine_update(Engine* engine);
void engine_render(Engine* engine);
void engine_set_camera_position(Engine* engine, float x, float y);
void engine_set_camera_zoom(Engine* engine, float zoom);

// Entity management
int engine_add_entity(Engine* engine, float x, float y, int width, int height,
    int texture_id, int layer);
void engine_set_entity_position(Engine* engine, int entity_idx, float x, float y);
void engine_set_entity_active(Engine* engine, int entity_idx, bool active);

// Ultra-minimal child entity system
int engine_add_child(Engine* engine, int parent_idx, float rel_x, float rel_y,
    int width, int height, int texture_id, int layer);
bool engine_attach_child(Engine* engine, int parent_idx, int child_idx);
void engine_detach_child(Engine* engine, int child_idx);
int engine_get_children(Engine* engine, int entity_idx, int* children_out, int max_count);
int engine_get_child_count(Engine* engine, int entity_idx);

// Resource management
int engine_add_texture(Engine* engine, SDL_Surface* surface, int x, int y);

// Helper functions
SDL_FRect engine_get_visible_rect(Engine* engine);
int engine_get_visible_entities_count(Engine* engine);


// Queue an entity for transform updates
void queue_entity_for_update(EntityManager* em, int entity_idx);

#endif // ENGINE_H