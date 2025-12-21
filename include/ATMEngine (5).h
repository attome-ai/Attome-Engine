#ifndef ENGINE_H
#define ENGINE_H

#include <SDL3/SDL.h>
#include <stdbool.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cassert>

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
#define ENTITY_CHUNK_SIZE (16384/2) // 16K entities per chunk for better memory management

// Entity chunk for improved memory management
typedef struct EntityChunk {
    float* x;          // x positions - aligned
    float* y;          // y positions - aligned
    float* right;      // x + width (precomputed) - aligned
    float* bottom;     // y + height (precomputed) - aligned
    int* width;        // widths - aligned
    int* height;       // heights - aligned
    int* texture_id;   // texture IDs - aligned
    int* layer;        // z-ordering/layers - aligned
    bool* active;      // is entity active/visible - aligned
    int** grid_cell;   // grid cell references for each entity - aligned
    int count;         // number of entities in this chunk
    int capacity;      // capacity of this chunk

    EntityChunk(int capacity);
    ~EntityChunk();
} EntityChunk;

// Entity manager using chunked Structure of Arrays (SoA) for better memory management
typedef struct EntityManager {
    EntityChunk** chunks;      // Array of entity chunks
    int chunk_count;           // Number of chunks
    int chunks_capacity;       // Total capacity of chunks array
    int total_count;           // Total entity count
    int* free_indices;         // Pool of free entity indices for reuse
    int free_count;            // Number of free indices
    int free_capacity;         // Capacity of free indices array

    EntityManager();
    ~EntityManager();
    int addEntity();
    void removeEntity(int entity_idx);
    bool isValidEntity(int entity_idx) const;
    // Helpers to get chunk and local index from entity index
    void getChunkIndices(int entity_idx, int* chunk_idx, int* local_idx) const;
} EntityManager;

// Optimized spatial grid for efficient entity queries with sparse storage
typedef struct SpatialGrid {
    int*** cells;              // 3D array: [cell_y][cell_x][entity_indices]
    int** cell_counts;         // Counts per cell
    int** cell_capacities;     // Capacities per cell
    float cell_size;           // Size of each cell
    int width, height;         // Grid dimensions
    int total_cells;           // Total number of cells
} SpatialGrid;

// Rendering batch (groups by texture and layer)
typedef struct RenderBatch {
    int texture_id;
    int layer;                 // Added to combine texture + layer
    SDL_Vertex* vertices;      // Vertex data for batch
    int* indices;              // Index data for batch
    int vertex_count;
    int index_count;
    int vertex_capacity;
    int index_capacity;
} RenderBatch;

// Texture atlas
typedef struct TextureAtlas {
    SDL_Texture* texture;
    SDL_FRect* regions;        // UV regions for each subtexture
    int region_count;
    int region_capacity;
} TextureAtlas;

// Camera for culling
typedef struct Camera {
    float x, y;                // Position
    float width, height;       // Viewport dimensions
    float zoom;                // Zoom level
} Camera;

// Improved buffer pool for temporary allocations
class FixedBufferPool {
public:
    class BufferHandle {
    private:
        uint8_t* buffer_;
        FixedBufferPool* pool_;

    public:
        BufferHandle() noexcept : buffer_(nullptr), pool_(nullptr) {}

        inline BufferHandle(uint8_t* buffer, FixedBufferPool* pool) noexcept
            : buffer_(buffer), pool_(pool) {}

        inline BufferHandle(BufferHandle&& other) noexcept
            : buffer_(other.buffer_), pool_(other.pool_) {
            other.buffer_ = nullptr;
            other.pool_ = nullptr;
        }

        inline BufferHandle& operator=(BufferHandle&& other) noexcept {
            if (this != &other) {
                release();
                buffer_ = other.buffer_;
                pool_ = other.pool_;
                other.buffer_ = nullptr;
                other.pool_ = nullptr;
            }
            return *this;
        }

        BufferHandle(const BufferHandle&) = delete;
        BufferHandle& operator=(const BufferHandle&) = delete;

        inline ~BufferHandle() {
            release();
        }

        inline void release() {
            if (buffer_ && pool_) {
                pool_->returnBuffer(buffer_);
                buffer_ = nullptr;
                pool_ = nullptr;
            }
        }

        inline uint8_t* data() const noexcept {
            return buffer_;
        }

        inline bool valid() const noexcept {
            return buffer_ != nullptr;
        }

        inline operator bool() const noexcept {
            return valid();
        }
    };

private:
    struct MemoryChunk {
        std::unique_ptr<uint8_t[]> memory;
        size_t buffer_count;

        MemoryChunk(size_t total_size, size_t count)
            : memory(new uint8_t[total_size]), buffer_count(count) {}
    };

    const size_t buffer_size_;
    std::vector<MemoryChunk> memory_chunks_;
    std::vector<uint8_t*> free_buffers_;

    static constexpr size_t DEFAULT_CAPACITY = 16;
    static constexpr size_t CHUNK_SIZE = 64; // Allocate 64 buffers per chunk

public:
    explicit FixedBufferPool(size_t buffer_size, size_t initial_count = 0)
        : buffer_size_(buffer_size) {
        // Reserve capacity to avoid reallocations
        free_buffers_.reserve(initial_count > 0 ? initial_count : DEFAULT_CAPACITY);

        // Calculate initial chunks needed
        size_t initial_chunks = (initial_count + CHUNK_SIZE - 1) / CHUNK_SIZE;
        if (initial_chunks == 0 && initial_count > 0) initial_chunks = 1;

        // Pre-allocate chunks
        for (size_t i = 0; i < initial_chunks; ++i) {
            allocateChunk();
        }
    }

    inline BufferHandle getBuffer() {
        if (free_buffers_.empty()) {
            allocateChunk();
        }

        uint8_t* buffer = free_buffers_.back();
        free_buffers_.pop_back();
        return BufferHandle(buffer, this);
    }

    inline size_t getTotalBuffers() const noexcept {
        size_t total = 0;
        for (const auto& chunk : memory_chunks_) {
            total += chunk.buffer_count;
        }
        return total;
    }

    inline size_t getAvailableBuffers() const noexcept {
        return free_buffers_.size();
    }

    inline size_t getBufferSize() const noexcept {
        return buffer_size_;
    }

private:
    void allocateChunk() {
        // Calculate total memory needed for the chunk
        size_t total_size = buffer_size_ * CHUNK_SIZE;

        // Allocate a new chunk
        memory_chunks_.emplace_back(total_size, CHUNK_SIZE);
        uint8_t* base_ptr = memory_chunks_.back().memory.get();

        // Add all buffers from the chunk to the free list
        for (size_t i = 0; i < CHUNK_SIZE; ++i) {
            uint8_t* buffer = base_ptr + (i * buffer_size_);
            free_buffers_.push_back(buffer);
        }
    }

    inline void returnBuffer(uint8_t* buffer) {
        free_buffers_.push_back(buffer);
    }

    friend class BufferHandle;
};

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
    FixedBufferPool entity_indices_pool;
    FixedBufferPool screen_coords_pool;
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

#endif // ENGINE_H

