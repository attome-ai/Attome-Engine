#ifndef ENGINE_H
#define ENGINE_H

#include <SDL3/SDL.h>
#include <stdbool.h>

// Forward declarations
typedef struct Engine Engine;
typedef struct EntityManager EntityManager;
typedef struct SpatialGrid SpatialGrid;
typedef struct SpatialHash SpatialHash;
typedef struct Quadtree Quadtree;
typedef struct QuadtreeNode QuadtreeNode;
typedef struct RenderBatch RenderBatch;
typedef struct TextureAtlas TextureAtlas;
typedef struct Camera Camera;
typedef struct BufferPool BufferPool;

// Spatial partitioning methods
typedef enum {
    SPATIAL_PARTITIONING_GRID,
    SPATIAL_PARTITIONING_HASH,
    SPATIAL_PARTITIONING_QUADTREE
} SpatialPartitioningMethod;

// Alignment for memory
#define CACHE_LINE_SIZE 64

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
    bool* visible;         // NEW: is entity should be rendered - aligned
    int** grid_cell;       // grid cell references for each entity - aligned
    int capacity;          // allocated capacity
    int count;             // actual count of entities
} EntityManager;

// Spatial grid for efficient entity queries
typedef struct SpatialGrid {
    int** cells;           // 2D array of entity index arrays
    int* cell_counts;      // Number of entities in each cell
    int* cell_capacities;  // Capacity of each cell
    float cell_size;       // Size of each cell
    int width, height;     // Grid dimensions
    int total_cells;       // Total number of cells
} SpatialGrid;

// Spatial hash for efficient entity queries
typedef struct SpatialHash {
    int** buckets;         // Hash buckets of entity index arrays
    int* bucket_counts;    // Number of entities in each bucket
    int* bucket_capacities;// Capacity of each bucket
    int bucket_count;      // Total number of buckets
    float cell_size;       // Size of each cell for hashing
    int prime;             // Prime number for hashing
} SpatialHash;

// Quadtree for dynamic region-based entity queries
typedef struct QuadtreeNode {
    SDL_FRect bounds;      // Boundary of this node
    int capacity;          // Capacity before splitting
    int count;             // Current entity count
    int* entities;         // Array of entity indices
    bool is_leaf;          // Is this a leaf node?
    struct QuadtreeNode* children[4]; // Children nodes (NW, NE, SW, SE)
} QuadtreeNode;

typedef struct Quadtree {
    QuadtreeNode* root;    // Root node of the quadtree
    int max_depth;         // Maximum depth of the tree
    int node_capacity;     // Maximum entities before splitting
    SDL_FRect bounds;      // World bounds
    bool needs_rebuild;    // Flag to indicate if tree needs rebuilding
} Quadtree;

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

    // Spatial partitioning structures
    SpatialGrid grid;
    SpatialHash hash;
    Quadtree quadtree;
    SpatialPartitioningMethod spatial_method; // Current method in use

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
} Engine;

// Engine initialization and management
Engine* engine_create(int window_width, int window_height, int world_width, int world_height);
void engine_destroy(Engine* engine);
void engine_update(Engine* engine);
void engine_render(Engine* engine);
void engine_set_camera_position(Engine* engine, float x, float y);
void engine_set_camera_zoom(Engine* engine, float zoom);
void engine_set_spatial_method(Engine* engine, SpatialPartitioningMethod method);

// Entity management
int engine_add_entity(Engine* engine, float x, float y, int width, int height,
    int texture_id, int layer);
void engine_set_entity_position(Engine* engine, int entity_idx, float x, float y);
void engine_set_entity_active(Engine* engine, int entity_idx, bool active);
void engine_set_entity_visible(Engine* engine, int entity_idx, bool visible);

// Resource management
int engine_add_texture(Engine* engine, SDL_Surface* surface, int x, int y);

// Helper functions
SDL_FRect engine_get_visible_rect(Engine* engine);
int engine_get_visible_entities_count(Engine* engine);

#endif // ENGINE_H