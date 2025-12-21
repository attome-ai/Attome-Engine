// ====== ENGINE.H ======
#ifndef ENGINE_H
#define ENGINE_H

#include <SDL3/SDL.h>
#include <stdbool.h>

// Forward declarations
// Forward declarations
typedef struct Engine Engine;
typedef struct EntityManager EntityManager;
typedef struct SpatialGrid SpatialGrid;
typedef struct RenderBatch RenderBatch;
typedef struct TextureAtlas TextureAtlas;
typedef struct Camera Camera;
typedef struct BufferPool BufferPool;
typedef struct Transform Transform;

// Alignment for memory
#define CACHE_LINE_SIZE 64
#define MAX_CHILDREN_PER_ENTITY 16

// 2D transform for efficient matrix operations
typedef struct Transform {
    float m[6]; // 2D affine transformation matrix (column-major): [a, c, e, b, d, f, 0, 0]
} Transform;



// Entity manager using Structure of Arrays (SoA) for cache efficiency
// Entity manager using Structure of Arrays (SoA) for cache efficiency
typedef struct EntityManager {
    float* x;            // world x positions - aligned
    float* y;            // world y positions - aligned
    float* right;        // x + width (precomputed) - aligned
    float* bottom;       // y + height (precomputed) - aligned
    float* local_x;      // local x positions relative to parent - aligned
    float* local_y;      // local y positions relative to parent - aligned
    float* scale_x;      // x scale - aligned
    float* scale_y;      // y scale - aligned
    float* rotation;     // rotation in radians - aligned
    int* width;          // widths - aligned
    int* height;         // heights - aligned
    int* texture_id;     // texture IDs - aligned
    int* layer;          // z-ordering/layers - aligned
    bool* active;        // is entity active/visible - aligned
    int* parent;         // parent entity index (-1 for root entities) - aligned
    int* first_child;    // index of first child (-1 if none) - aligned
    int* next_sibling;   // index of next sibling (-1 if none) - aligned
    int* child_count;    // number of children - aligned
    Transform* local_transform;  // cached local transform - aligned
    Transform* world_transform;  // cached world transform - aligned
    int** grid_cell;     // grid cell references for each entity - aligned
    bool* transform_dirty; // whether local transform needs updating - aligned
    int capacity;        // allocated capacity
    int count;           // actual count of entities
} EntityManager;


// Spatial grid for efficient entity queries (replaces quadtree)
typedef struct SpatialGrid {
    int** cells;        // 2D array of entity index arrays
    int* cell_counts;   // Number of entities in each cell
    int* cell_capacities; // Capacity of each cell
    float cell_size;    // Size of each cell
    int width, height;  // Grid dimensions
    int total_cells;    // Total number of cells
} SpatialGrid;

// Rendering batch (groups by texture and layer)
typedef struct RenderBatch {
    int texture_id;
    int layer;          // Added to combine texture + layer
    SDL_Vertex* vertices; // Vertex data for batch
    int* indices;       // Index data for batch
    int vertex_count;
    int index_count;
    int vertex_capacity;
    int index_capacity;
} RenderBatch;

// Texture atlas
typedef struct TextureAtlas {
    SDL_Texture* texture;
    SDL_FRect* regions;  // UV regions for each subtexture
    int region_count;
    int region_capacity;
} TextureAtlas;

// Camera for culling
typedef struct Camera {
    float x, y;          // Position
    float width, height; // Viewport dimensions
    float zoom;          // Zoom level
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
    BufferPool transform_pool;
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

// Resource management
int engine_add_texture(Engine* engine, SDL_Surface* surface, int x, int y);

// Helper functions
SDL_FRect engine_get_visible_rect(Engine* engine);
int engine_get_visible_entities_count(Engine* engine);


// Transform functions
void transform_identity(Transform* transform);
void transform_set(Transform* transform, float x, float y, float rotation, float scale_x, float scale_y);
void transform_combine(Transform* result, const Transform* a, const Transform* b);
void transform_apply(const Transform* transform, float x, float y, float* out_x, float* out_y);
void transform_batch_apply(const Transform* transform, float* x, float* y, float* out_x, float* out_y, int count);

// Engine initialization and management
Engine* engine_create(int window_width, int window_height, int world_width, int world_height, int cell_size);
void engine_destroy(Engine* engine);
void engine_update(Engine* engine);
void engine_render(Engine* engine);
void engine_set_camera_position(Engine* engine, float x, float y);
void engine_set_camera_zoom(Engine* engine, float zoom);

// Entity management
int engine_add_entity(Engine* engine, float x, float y, int width, int height, int texture_id, int layer);
void engine_set_entity_position(Engine* engine, int entity_idx, float x, float y);
void engine_set_entity_active(Engine* engine, int entity_idx, bool active);
void engine_set_entity_local_transform(Engine* engine, int entity_idx, float x, float y, float rotation, float scale_x, float scale_y);
void engine_set_entity_rotation(Engine* engine, int entity_idx, float rotation);
void engine_set_entity_scale(Engine* engine, int entity_idx, float scale_x, float scale_y);

// Entity composition
int engine_add_child_entity(Engine* engine, int parent_idx, float local_x, float local_y, int width, int height, int texture_id, int layer);
void engine_set_parent(Engine* engine, int entity_idx, int parent_idx);
void engine_remove_parent(Engine* engine, int entity_idx);
int engine_get_child_count(Engine* engine, int entity_idx);
int engine_get_child_at_index(Engine* engine, int entity_idx, int child_index);
void engine_update_world_transforms(Engine* engine);
void engine_update_entity_transform(Engine* engine, int entity_idx);

// Resource management
int engine_add_texture(Engine* engine, SDL_Surface* surface, int x, int y);

// Helper functions
SDL_FRect engine_get_visible_rect(Engine* engine);
int engine_get_visible_entities_count(Engine* engine);

#endif // ENGINE_H