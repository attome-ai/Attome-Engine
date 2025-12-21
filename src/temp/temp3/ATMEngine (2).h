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
typedef struct EntityChunk EntityChunk;

// Alignment for memory
#define CACHE_LINE_SIZE 64
#define ENTITY_CHUNK_SIZE 16384  // 16K entities per chunk for better memory management

// Entity chunk for improved memory management
typedef struct EntityChunk {
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
    int count;             // number of entities in this chunk
} EntityChunk;

// Entity manager using chunked Structure of Arrays (SoA) for better memory management
typedef struct EntityManager {
    EntityChunk** chunks;  // Array of entity chunks
    int chunk_count;       // Number of chunks
    int chunks_capacity;   // Total capacity of chunks array
    int total_count;       // Total entity count
    int* free_indices;     // Pool of free entity indices for reuse
    int free_count;        // Number of free indices
    int free_capacity;     // Capacity of free indices array
} EntityManager;

// Optimized spatial grid for efficient entity queries with sparse storage
typedef struct SpatialGrid {
    int*** cells;          // 3D array: [cell_y][cell_x][entity_indices]
    int** cell_counts;     // Counts per cell
    int** cell_capacities; // Capacities per cell
    float cell_size;       // Size of each cell
    int width, height;     // Grid dimensions
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

// Improved buffer pool for temporary allocations
typedef struct BufferPool {
    void** buffers;
    int count;
    int capacity;
    size_t buffer_size;
    int max_buffers;       // Limit on pool size to prevent memory bloat
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
} Engine;

// Engine initialization and management
Engine* engine_create(int window_width, int window_height, int world_width, int world_height, int cell_size);
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
void get_chunk_indices(int entity_idx, int* chunk_idx, int* local_idx);
#endif // ENGINE_H