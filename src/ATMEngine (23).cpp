
#ifndef ENGINE_H
#define ENGINE_H

#include <SDL3/SDL.h>
#include <stdbool.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <unordered_map>
#include <immintrin.h> // For SIMD intrinsics

// Forward declarations
typedef struct Engine Engine;
typedef struct EntityManager EntityManager;
typedef struct SpatialGrid SpatialGrid;
typedef struct RenderBatch RenderBatch;
typedef struct TextureAtlas TextureAtlas;
typedef struct Camera Camera;
typedef struct BufferPool BufferPool;
typedef struct EntityChunk EntityChunk;
typedef struct CellMemoryPool CellMemoryPool;

// Alignment for memory
#define CACHE_LINE_SIZE 64
#define ENTITY_CHUNK_SIZE (512) // 16K entities per chunk for better memory management

// Entity type update function typedef
typedef void (*EntityTypeUpdateFunc)(EntityChunk* chunk, int count, float delta_time);

// Configuration for each entity type
typedef struct EntityTypeConfig {
    int type_id;
    EntityTypeUpdateFunc update_func;
    size_t extra_data_size;  // Size of any type-specific data per entity

    // Performance tracking
    int instance_count;
    float last_update_time;

    // Linked list of chunks containing this type (for faster updates)
    int first_chunk_idx;
} EntityTypeConfig;

// Memory pool for efficient cell allocation
typedef struct CellMemoryPool {
    uint8_t* memory;
    size_t block_size;
    size_t capacity;
    int* free_indices;
    int free_count;
} CellMemoryPool;

// Modify EntityChunk structure to include optimizations
typedef struct EntityChunk {
    int type_id;            // Type of entities in this chunk
    int next_chunk_of_type; // Next chunk with the same type (-1 if none)

    // Data arrays with SIMD-friendly alignment
    float* x;               // x positions - aligned
    float* y;               // y positions - aligned
    float* right;           // x + width (precomputed) - aligned
    float* bottom;          // y + height (precomputed) - aligned
    int* width;             // widths - aligned
    int* height;            // heights - aligned
    int* texture_id;        // texture IDs - aligned
    int* layer;             // z-ordering/layers - aligned
    bool* active;           // is entity active for updates - aligned
    bool* visible;          // is entity visible for rendering - aligned 
    int** grid_cell;        // grid cell references for each entity - aligned

    // Hierarchy fields
    int* parent_id;         // parent entity id (-1 for root entities) - aligned
    int* first_child_id;    // id of first child (-1 if no children) - aligned
    int* next_sibling_id;   // id of next sibling (-1 if last child) - aligned
    float* local_x;         // local x position - aligned
    float* local_y;         // local y position - aligned

    // Type-specific data (optional, dynamically sized)
    void* type_data;        // Array of type-specific component data

    // Optimization: Active entity tracking
    int* active_indices;    // Packed array of indices for active entities
    int active_count;       // Number of active entities

    // Optimization: Transform dirty flags (1 bit per entity)
    uint64_t* transform_dirty_flags; // Bitmask of entities with dirty transforms

    // Existing fields
    int count;              // number of entities in this chunk
    int capacity;           // capacity of this chunk

    EntityChunk(int type_id, int capacity, size_t extra_data_size = 0);
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

    // Optimization: Temporal coherence tracking
    int*** last_frame_cells;   // Previous frame cell locations for each entity
    CellMemoryPool*** cell_pools; // Pre-allocated memory pools per cell
    uint32_t* cell_modified;   // Bitset for tracking modified cells
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

// Main engine struct with optimizations
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

    // Entity type system optimized for direct access
    std::unordered_map<int, int> type_id_to_index; // Maps type_id to index in entity_types array
    EntityTypeConfig* entity_types;
    int entity_type_count;
    int entity_type_capacity;

    // Performance tracking
    int active_entity_count;
    float update_time;
    float render_time;

    // Reusable buffer pools
    FixedBufferPool entity_indices_pool;
    FixedBufferPool screen_coords_pool;

    // Optimization: Sorting keys buffer pool
    FixedBufferPool sort_keys_pool;
} Engine;

// Engine initialization and management
Engine* engine_create(int window_width, int window_height, int world_width, int world_height, int cell_size);
void engine_destroy(Engine* engine);
void engine_update(Engine* engine);
void engine_render(Engine* engine);
inline void engine_present(Engine* engine) {
    // Present renderer
    SDL_RenderPresent(engine->renderer);
}
void engine_set_camera_position(Engine* engine, float x, float y);
void engine_set_camera_zoom(Engine* engine, float zoom);

// Entity management
int engine_add_entity(Engine* engine, float x, float y, int width, int height,
    int texture_id, int layer);
void engine_set_entity_position(Engine* engine, int entity_idx, float x, float y);
void engine_set_entity_active(Engine* engine, int entity_idx, bool active);
void engine_set_entity_visible(Engine* engine, int entity_idx, bool visible);

// Resource management
int engine_add_texture(Engine* engine, SDL_Surface* surface, int x, int y);

SDL_FRect engine_get_visible_rect(Engine* engine);
int engine_get_visible_entities_count(Engine* engine);

void engine_set_parent(Engine* engine, int entity_id, int parent_id);
void engine_remove_parent(Engine* engine, int entity_id);
void engine_update_entity_transforms(Engine* engine);
int engine_get_parent(Engine* engine, int entity_id);
int engine_get_first_child(Engine* engine, int entity_id);
int engine_get_next_sibling(Engine* engine, int entity_id);
void engine_set_entity_local_position(Engine* engine, int entity_id, float x, float y);
int engine_add_child_entity(Engine* engine, int parent_id, float local_x, float local_y,
    int width, int height, int texture_id, int layer);
void spatial_grid_query(SpatialGrid* grid, SDL_FRect query_rect,
    int* result_indices, int* result_count, int max_results,
    EntityManager* entities);

void engine_register_entity_type(Engine* engine, int type_id, EntityTypeUpdateFunc update_func, size_t extra_data_size = 0);
int engine_add_entity_with_type(Engine* engine, int type_id, float x, float y, int width, int height, int texture_id, int layer);
void engine_update_entity_types(Engine* engine, float delta_time);
void* engine_get_entity_type_data(Engine* engine, int entity_idx);

// Optimization: Memory pool operations
CellMemoryPool* create_cell_pool(size_t block_size, size_t initial_capacity);
void* pool_get_block(CellMemoryPool* pool);
void pool_return_block(CellMemoryPool* pool, void* block);

// Optimization: Batch operations
void spatial_grid_add_batch(SpatialGrid* grid, int* entity_indices, float* x_positions, float* y_positions, int count, EntityManager* entities);
void update_dirty_transforms(Engine* engine);
void mark_entity_transform_dirty(EntityChunk* chunk, int local_idx);

// Optimization: SIMD operations
void transform_entity_batch_simd(float* src_x, float* src_y, float* src_width, float* src_height,
    float* dst_x, float* dst_y, float* dst_width, float* dst_height,
    float scale_x, float scale_y, float translate_x, float translate_y,
    int count);
uint64_t make_render_sort_key(int texture_id, int layer);
void radix_sort_by_key(uint64_t* keys, int* values, int* output, int count);

#endif // ENGINE_H

// Implementation file
#include "ATMEngine.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#include "ATMProfiler.h"
#ifdef EMSCRIPTEN
#include <emscripten.h>
#endif

// Memory pool implementation
CellMemoryPool* create_cell_pool(size_t block_size, size_t initial_capacity) {
    PROFILE_FUNCTION();

    CellMemoryPool* pool = static_cast<CellMemoryPool*>(SDL_aligned_alloc(sizeof(CellMemoryPool), CACHE_LINE_SIZE));
    pool->memory = static_cast<uint8_t*>(SDL_aligned_alloc(block_size * initial_capacity, CACHE_LINE_SIZE));
    pool->block_size = block_size;
    pool->capacity = initial_capacity;
    pool->free_indices = static_cast<int*>(SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE));
    pool->free_count = initial_capacity;

    // Initialize free indices in reverse order for LIFO allocation
    for (int i = 0; i < initial_capacity; i++) {
        pool->free_indices[i] = initial_capacity - i - 1;
    }

    return pool;
}

void* pool_get_block(CellMemoryPool* pool) {
    PROFILE_FUNCTION();

    if (pool->free_count == 0) {
        // Double capacity when out of blocks
        size_t new_capacity = pool->capacity * 2;
        uint8_t* new_memory = static_cast<uint8_t*>(SDL_aligned_alloc(pool->block_size * new_capacity, CACHE_LINE_SIZE));
        memcpy(new_memory, pool->memory, pool->block_size * pool->capacity);

        int* new_free_indices = static_cast<int*>(SDL_aligned_alloc(new_capacity * sizeof(int), CACHE_LINE_SIZE));
        memcpy(new_free_indices, pool->free_indices, pool->capacity * sizeof(int));

        // Add new blocks to free list
        for (int i = 0; i < new_capacity - pool->capacity; i++) {
            new_free_indices[pool->capacity + i] = pool->capacity + i;
        }

        SDL_aligned_free(pool->memory);
        SDL_aligned_free(pool->free_indices);

        pool->memory = new_memory;
        pool->free_indices = new_free_indices;
        pool->free_count += (new_capacity - pool->capacity);
        pool->capacity = new_capacity;
    }

    int block_idx = pool->free_indices[--pool->free_count];
    return pool->memory + (block_idx * pool->block_size);
}

void pool_return_block(CellMemoryPool* pool, void* block) {
    PROFILE_FUNCTION();

    size_t offset = static_cast<uint8_t*>(block) - pool->memory;
    int block_idx = offset / pool->block_size;
    pool->free_indices[pool->free_count++] = block_idx;
}

void free_memory_pool(CellMemoryPool* pool) {
    PROFILE_FUNCTION();

    if (pool) {
        SDL_aligned_free(pool->memory);
        SDL_aligned_free(pool->free_indices);
        SDL_aligned_free(pool);
    }
}

// Make render sort key for texture and layer
uint64_t make_render_sort_key(int texture_id, int layer) {
    return ((uint64_t)texture_id << 32) | (uint64_t)layer;
}

// Mark entity transform as dirty
void mark_entity_transform_dirty(EntityChunk* chunk, int local_idx) {
    int flag_idx = local_idx / 64;
    int bit_pos = local_idx % 64;
    chunk->transform_dirty_flags[flag_idx] |= (1ULL << bit_pos);
}

// Modified EntityChunk constructor with optimizations
EntityChunk::EntityChunk(int chunk_type_id, int chunk_capacity, size_t extra_data_size) {
    PROFILE_SCOPE("EntityChunk::Constructor");

    type_id = chunk_type_id;
    capacity = chunk_capacity;
    count = 0;
    next_chunk_of_type = -1; // Initialize with no next chunk
    active_count = 0;

    // Allocate memory for all arrays with proper alignment
    size_t float_size = capacity * sizeof(float);
    size_t int_size = capacity * sizeof(int);
    size_t bool_size = capacity * sizeof(bool);
    size_t ptr_size = capacity * sizeof(int*);

    // Use aligned allocation for better SIMD performance
    x = static_cast<float*>(SDL_aligned_alloc(float_size, CACHE_LINE_SIZE));
    y = static_cast<float*>(SDL_aligned_alloc(float_size, CACHE_LINE_SIZE));
    right = static_cast<float*>(SDL_aligned_alloc(float_size, CACHE_LINE_SIZE));
    bottom = static_cast<float*>(SDL_aligned_alloc(float_size, CACHE_LINE_SIZE));
    width = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    height = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    texture_id = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    layer = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    active = static_cast<bool*>(SDL_aligned_alloc(bool_size, CACHE_LINE_SIZE));
    visible = static_cast<bool*>(SDL_aligned_alloc(bool_size, CACHE_LINE_SIZE));
    grid_cell = static_cast<int**>(SDL_aligned_alloc(ptr_size, CACHE_LINE_SIZE));

    // Hierarchy arrays
    parent_id = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    first_child_id = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    next_sibling_id = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));
    local_x = static_cast<float*>(SDL_aligned_alloc(float_size, CACHE_LINE_SIZE));
    local_y = static_cast<float*>(SDL_aligned_alloc(float_size, CACHE_LINE_SIZE));

    // Optimization: Active entity tracking
    active_indices = static_cast<int*>(SDL_aligned_alloc(int_size, CACHE_LINE_SIZE));

    // Optimization: Transform dirty flags (1 bit per entity, packed into 64-bit chunks)
    int dirty_flags_size = (capacity + 63) / 64; // Rounded up to nearest 64
    transform_dirty_flags = static_cast<uint64_t*>(SDL_aligned_alloc(dirty_flags_size * sizeof(uint64_t), CACHE_LINE_SIZE));
    memset(transform_dirty_flags, 0, dirty_flags_size * sizeof(uint64_t));

    // Allocate type-specific data if needed
    if (extra_data_size > 0) {
        type_data = SDL_aligned_alloc(capacity * extra_data_size, CACHE_LINE_SIZE);
        memset(type_data, 0, capacity * extra_data_size);
    }
    else {
        type_data = nullptr;
    }

    // Initialize all memory to zero
    memset(x, 0, float_size);
    memset(y, 0, float_size);
    memset(right, 0, float_size);
    memset(bottom, 0, float_size);
    memset(width, 0, int_size);
    memset(height, 0, int_size);
    memset(texture_id, 0, int_size);
    memset(layer, 0, int_size);
    memset(active, 1, bool_size); // Default to active
    memset(visible, 0, bool_size); // Default to not visible
    memset(grid_cell, 0, ptr_size);
    memset(active_indices, 0, int_size);

    // Initialize hierarchy arrays with -1 (no parent/child/sibling)
    for (int i = 0; i < capacity; i++) {
        parent_id[i] = -1;
        first_child_id[i] = -1;
        next_sibling_id[i] = -1;
    }

    memset(local_x, 0, float_size);
    memset(local_y, 0, float_size);
}

// EntityChunk destructor with optimizations
EntityChunk::~EntityChunk() {
    PROFILE_SCOPE("EntityChunk::Destructor");

    SDL_aligned_free(x);
    SDL_aligned_free(y);
    SDL_aligned_free(right);
    SDL_aligned_free(bottom);
    SDL_aligned_free(width);
    SDL_aligned_free(height);
    SDL_aligned_free(texture_id);
    SDL_aligned_free(layer);
    SDL_aligned_free(active);
    SDL_aligned_free(visible);
    SDL_aligned_free(grid_cell);

    // Free hierarchy arrays
    SDL_aligned_free(parent_id);
    SDL_aligned_free(first_child_id);
    SDL_aligned_free(next_sibling_id);
    SDL_aligned_free(local_x);
    SDL_aligned_free(local_y);

    // Free optimization arrays
    SDL_aligned_free(active_indices);
    SDL_aligned_free(transform_dirty_flags);

    // Free type-specific data if allocated
    if (type_data) {
        SDL_aligned_free(type_data);
    }
}

// Implementation of EntityManager constructor
EntityManager::EntityManager() {
    PROFILE_SCOPE("EntityManager::Constructor");

    chunk_count = 0;
    chunks_capacity = 4; // Start with space for 4 chunks
    total_count = 0;
    free_count = 0;
    free_capacity = 1024; // Initial capacity for free indices

    // Allocate memory for chunks array
    chunks = static_cast<EntityChunk**>(malloc(chunks_capacity * sizeof(EntityChunk*)));

    // Allocate memory for free indices
    free_indices = static_cast<int*>(malloc(free_capacity * sizeof(int)));
}

// Implementation of EntityManager destructor
EntityManager::~EntityManager() {
    PROFILE_SCOPE("EntityManager::Destructor");

    // Free all chunks
    for (int i = 0; i < chunk_count; i++) {
        delete chunks[i];
    }

    // Free chunks array and free indices
    free(chunks);
    free(free_indices);
}

// Helper to get chunk and local indices from entity index
void EntityManager::getChunkIndices(int entity_idx, int* chunk_idx, int* local_idx) const {
    PROFILE_FUNCTION();

    *chunk_idx = entity_idx / ENTITY_CHUNK_SIZE;
    *local_idx = entity_idx % ENTITY_CHUNK_SIZE;
}

// Add a new entity to the entity manager
int EntityManager::addEntity() {
    PROFILE_FUNCTION();

    int entity_idx;

    // If there are free indices, reuse one
    if (free_count > 0) {
        entity_idx = free_indices[--free_count];
    }
    else {
        // No free indices, create a new entity
        entity_idx = total_count++;

        // Calculate which chunk this entity belongs to
        int chunk_idx, local_idx;
        getChunkIndices(entity_idx, &chunk_idx, &local_idx);

        // Ensure we have enough chunks
        if (chunk_idx >= chunk_count) {
            // Need to create a new chunk
            if (chunk_idx >= chunks_capacity) {
                // Need to resize chunks array
                chunks_capacity *= 2;
                chunks = static_cast<EntityChunk**>(realloc(chunks, chunks_capacity * sizeof(EntityChunk*)));
            }

            // Create new chunk(s) as needed
            while (chunk_count <= chunk_idx) {
                chunks[chunk_count] = new EntityChunk(-1, ENTITY_CHUNK_SIZE);
                chunk_count++;
            }
        }

        // Increment count in the appropriate chunk
        chunks[chunk_idx]->count++;
    }

    return entity_idx;
}

// Remove an entity from the entity manager
void EntityManager::removeEntity(int entity_idx) {
    PROFILE_FUNCTION();

    if (!isValidEntity(entity_idx)) {
        return;
    }

    // Calculate which chunk this entity belongs to
    int chunk_idx, local_idx;
    getChunkIndices(entity_idx, &chunk_idx, &local_idx);

    // Mark as inactive
    chunks[chunk_idx]->active[local_idx] = false;

    // Update active indices list
    EntityChunk* chunk = chunks[chunk_idx];
    for (int i = 0; i < chunk->active_count; i++) {
        if (chunk->active_indices[i] == local_idx) {
            // Remove from active list by swapping with last element
            chunk->active_indices[i] = chunk->active_indices[--chunk->active_count];
            break;
        }
    }

    // Add to free indices
    if (free_count >= free_capacity) {
        // Need to resize free indices array
        free_capacity *= 2;
        free_indices = static_cast<int*>(realloc(free_indices, free_capacity * sizeof(int)));
    }

    free_indices[free_count++] = entity_idx;
}

// Check if an entity is valid
bool EntityManager::isValidEntity(int entity_idx) const {
    PROFILE_FUNCTION();

    if (entity_idx < 0 || entity_idx >= total_count) {
        return false;
    }

    int chunk_idx, local_idx;
    getChunkIndices(entity_idx, &chunk_idx, &local_idx);

    if (chunk_idx >= chunk_count) {
        return false;
    }

    return local_idx < chunks[chunk_idx]->count;
}

// Initialize spatial grid for entity queries
void init_spatial_grid(SpatialGrid* grid, float world_width, float world_height, float cell_size) {
    PROFILE_FUNCTION();

    grid->cell_size = cell_size;
    grid->width = (int)ceil(world_width / cell_size);
    grid->height = (int)ceil(world_height / cell_size);
    grid->total_cells = grid->width * grid->height;

    // Allocate cells array using a flat array for better cache locality
    grid->cells = static_cast<int***>(SDL_aligned_alloc(grid->height * sizeof(int**), CACHE_LINE_SIZE));
    grid->cell_counts = static_cast<int**>(SDL_aligned_alloc(grid->height * sizeof(int*), CACHE_LINE_SIZE));
    grid->cell_capacities = static_cast<int**>(SDL_aligned_alloc(grid->height * sizeof(int*), CACHE_LINE_SIZE));

    // Allocate temporal coherence tracking arrays
    grid->last_frame_cells = static_cast<int***>(SDL_aligned_alloc(grid->height * sizeof(int**), CACHE_LINE_SIZE));
    grid->cell_pools = static_cast<CellMemoryPool***>(SDL_aligned_alloc(grid->height * sizeof(CellMemoryPool**), CACHE_LINE_SIZE));

    // Cell modified bitset (1 bit per cell)
    int bitset_size = (grid->total_cells + 31) / 32; // Rounded up to nearest 32
    grid->cell_modified = static_cast<uint32_t*>(SDL_aligned_alloc(bitset_size * sizeof(uint32_t), CACHE_LINE_SIZE));
    memset(grid->cell_modified, 0, bitset_size * sizeof(uint32_t));

    for (int y = 0; y < grid->height; y++) {
        grid->cells[y] = static_cast<int**>(SDL_aligned_alloc(grid->width * sizeof(int*), CACHE_LINE_SIZE));
        grid->cell_counts[y] = static_cast<int*>(SDL_aligned_alloc(grid->width * sizeof(int), CACHE_LINE_SIZE));
        grid->cell_capacities[y] = static_cast<int*>(SDL_aligned_alloc(grid->width * sizeof(int), CACHE_LINE_SIZE));
        grid->last_frame_cells[y] = static_cast<int**>(SDL_aligned_alloc(grid->width * sizeof(int*), CACHE_LINE_SIZE));
        grid->cell_pools[y] = static_cast<CellMemoryPool**>(SDL_aligned_alloc(grid->width * sizeof(CellMemoryPool*), CACHE_LINE_SIZE));

        for (int x = 0; x < grid->width; x++) {
            int initial_capacity = 32; // Start with space for 32 entities per cell
            grid->cell_counts[y][x] = 0;
            grid->cell_capacities[y][x] = initial_capacity;

            // Create pool for this cell
            grid->cell_pools[y][x] = create_cell_pool(sizeof(int), initial_capacity);

            // Allocate cell from pool
            grid->cells[y][x] = static_cast<int*>(pool_get_block(grid->cell_pools[y][x]));
            grid->last_frame_cells[y][x] = static_cast<int*>(SDL_aligned_alloc(initial_capacity * sizeof(int), CACHE_LINE_SIZE));
            memset(grid->last_frame_cells[y][x], -1, initial_capacity * sizeof(int));
        }
    }
}

// Free spatial grid with memory pool cleanup
void free_spatial_grid(SpatialGrid* grid) {
    PROFILE_FUNCTION();

    for (int y = 0; y < grid->height; y++) {
        for (int x = 0; x < grid->width; x++) {
            pool_return_block(grid->cell_pools[y][x], grid->cells[y][x]);
            free_memory_pool(grid->cell_pools[y][x]);
            SDL_aligned_free(grid->last_frame_cells[y][x]);
        }
        SDL_aligned_free(grid->cells[y]);
        SDL_aligned_free(grid->cell_counts[y]);
        SDL_aligned_free(grid->cell_capacities[y]);
        SDL_aligned_free(grid->last_frame_cells[y]);
        SDL_aligned_free(grid->cell_pools[y]);
    }

    SDL_aligned_free(grid->cells);
    SDL_aligned_free(grid->cell_counts);
    SDL_aligned_free(grid->cell_capacities);
    SDL_aligned_free(grid->last_frame_cells);
    SDL_aligned_free(grid->cell_pools);
    SDL_aligned_free(grid->cell_modified);
}

// Clear all cells in the spatial grid
void clear_spatial_grid(SpatialGrid* grid) {
    PROFILE_FUNCTION();

    // Only clear cells that were modified last frame for better performance
    int bitset_size = (grid->total_cells + 31) / 32;
    for (int i = 0; i < bitset_size; i++) {
        uint32_t modified_mask = grid->cell_modified[i];
        if (!modified_mask) continue;

        while (modified_mask) {
            int bit_pos = _tzcnt_u32(modified_mask);
            int cell_index = i * 32 + bit_pos;
            int cell_y = cell_index / grid->width;
            int cell_x = cell_index % grid->width;

            // Clear cell count
            grid->cell_counts[cell_y][cell_x] = 0;

            // Clear the bit
            modified_mask &= ~(1 << bit_pos);
        }
    }

    // Reset modified flags for next frame
    memset(grid->cell_modified, 0, bitset_size * sizeof(uint32_t));
}

// Add entity to spatial grid with cell tracking
void spatial_grid_add(SpatialGrid* grid, int entity_idx, float x, float y, EntityManager* entities) {
    PROFILE_FUNCTION();

    int grid_x = (int)(x / grid->cell_size);
    int grid_y = (int)(y / grid->cell_size);

    // Clamp to grid bounds
    grid_x = std::max(0, std::min(grid_x, grid->width - 1));
    grid_y = std::max(0, std::min(grid_y, grid->height - 1));

    // Get chunk and local index
    int chunk_idx, local_idx;
    entities->getChunkIndices(entity_idx, &chunk_idx, &local_idx);
    EntityChunk* chunk = entities->chunks[chunk_idx];

    // Update entity's grid cell reference
    int* previous_cell = chunk->grid_cell[local_idx];

    // Check if entity moved to a different cell
    if (previous_cell != &grid->cell_counts[grid_y][grid_x]) {
        // Ensure capacity in new cell
        int& count = grid->cell_counts[grid_y][grid_x];
        int& capacity = grid->cell_capacities[grid_y][grid_x];

        if (count >= capacity) {
            int new_capacity = capacity * 2;
            int* new_cell = static_cast<int*>(pool_get_block(grid->cell_pools[grid_y][grid_x]));

            // Copy existing data
            memcpy(new_cell, grid->cells[grid_y][grid_x], count * sizeof(int));

            // Return old cell to pool
            pool_return_block(grid->cell_pools[grid_y][grid_x], grid->cells[grid_y][grid_x]);

            // Update with new cell
            grid->cells[grid_y][grid_x] = new_cell;
            capacity = new_capacity;
        }

        // Add entity to new cell
        grid->cells[grid_y][grid_x][count++] = entity_idx;

        // Update entity's cell reference
        chunk->grid_cell[local_idx] = &grid->cell_counts[grid_y][grid_x];

        // Mark cell as modified
        int cell_index = grid_y * grid->width + grid_x;
        grid->cell_modified[cell_index / 32] |= (1 << (cell_index % 32));
    }
}

// Optimized batch operation for adding entities to spatial grid
void spatial_grid_add_batch(SpatialGrid* grid, int* entity_indices, float* x_positions, float* y_positions, int count, EntityManager* entities) {
    PROFILE_FUNCTION();

    // Temporary buffers for cell indices
    int* grid_x = static_cast<int*>(SDL_aligned_alloc(count * sizeof(int), CACHE_LINE_SIZE));
    int* grid_y = static_cast<int*>(SDL_aligned_alloc(count * sizeof(int), CACHE_LINE_SIZE));

    // Calculate cell indices for all entities
#pragma omp simd
    for (int i = 0; i < count; i++) {
        grid_x[i] = (int)(x_positions[i] / grid->cell_size);
        grid_y[i] = (int)(y_positions[i] / grid->cell_size);

        // Clamp to grid bounds
        grid_x[i] = std::max(0, std::min(grid_x[i], grid->width - 1));
        grid_y[i] = std::max(0, std::min(grid_y[i], grid->height - 1));
    }

    // Create cell-based sort keys (y in high bits, x in low bits)
    uint32_t* sort_keys = static_cast<uint32_t*>(SDL_aligned_alloc(count * sizeof(uint32_t), CACHE_LINE_SIZE));

    for (int i = 0; i < count; i++) {
        sort_keys[i] = ((uint32_t)grid_y[i] << 16) | grid_x[i];
    }

    // Create sorted copy of entity indices
    int* sorted_entities = static_cast<int*>(SDL_aligned_alloc(count * sizeof(int), CACHE_LINE_SIZE));
    int* sorted_grid_x = static_cast<int*>(SDL_aligned_alloc(count * sizeof(int), CACHE_LINE_SIZE));
    int* sorted_grid_y = static_cast<int*>(SDL_aligned_alloc(count * sizeof(int), CACHE_LINE_SIZE));

    // Sort entities by their cell indices (simple counting sort for demo)
    int* count_by_key = static_cast<int*>(SDL_aligned_alloc(grid->width * grid->height * sizeof(int), CACHE_LINE_SIZE));
    memset(count_by_key, 0, grid->width * grid->height * sizeof(int));

    // Count occurrences of each key
    for (int i = 0; i < count; i++) {
        count_by_key[sort_keys[i]]++;
    }

    // Calculate starting positions
    int total = 0;
    for (int i = 0; i < grid->width * grid->height; i++) {
        int old_count = count_by_key[i];
        count_by_key[i] = total;
        total += old_count;
    }

    // Place elements in sorted order
    for (int i = 0; i < count; i++) {
        int pos = count_by_key[sort_keys[i]]++;
        sorted_entities[pos] = entity_indices[i];
        sorted_grid_x[pos] = grid_x[i];
        sorted_grid_y[pos] = grid_y[i];
    }

    SDL_aligned_free(count_by_key);

    // Now add entities to grid in sorted order for better cache locality
    int current_cell_x = -1;
    int current_cell_y = -1;
    int* current_cell = nullptr;
    int* current_count = nullptr;

    for (int i = 0; i < count; i++) {
        int entity_idx = sorted_entities[i];
        int cell_x = sorted_grid_x[i];
        int cell_y = sorted_grid_y[i];

        // Check if this is a new cell
        if (cell_x != current_cell_x || cell_y != current_cell_y) {
            current_cell_x = cell_x;
            current_cell_y = cell_y;
            current_cell = grid->cells[cell_y][cell_x];
            current_count = &grid->cell_counts[cell_y][cell_x];

            // Mark cell as modified
            int cell_index = cell_y * grid->width + cell_x;
            grid->cell_modified[cell_index / 32] |= (1 << (cell_index % 32));
        }

        // Get chunk and local index
        int chunk_idx, local_idx;
        entities->getChunkIndices(entity_idx, &chunk_idx, &local_idx);
        EntityChunk* chunk = entities->chunks[chunk_idx];

        // Update entity's grid cell reference
        chunk->grid_cell[local_idx] = &grid->cell_counts[cell_y][cell_x];

        // Add entity to current cell
        if (*current_count < grid->cell_capacities[cell_y][cell_x]) {
            current_cell[(*current_count)++] = entity_idx;
        }
        else {
            // Reallocate cell if needed
            int new_capacity = grid->cell_capacities[cell_y][cell_x] * 2;
            int* new_cell = static_cast<int*>(pool_get_block(grid->cell_pools[cell_y][cell_x]));

            // Copy existing data
            memcpy(new_cell, current_cell, (*current_count) * sizeof(int));

            // Return old cell to pool
            pool_return_block(grid->cell_pools[cell_y][cell_x], grid->cells[cell_y][cell_x]);

            // Update grid and pointers
            grid->cells[cell_y][cell_x] = new_cell;
            grid->cell_capacities[cell_y][cell_x] = new_capacity;
            current_cell = new_cell;

            // Add entity
            current_cell[(*current_count)++] = entity_idx;
        }
    }

    // Cleanup
    SDL_aligned_free(grid_x);
    SDL_aligned_free(grid_y);
    SDL_aligned_free(sort_keys);
    SDL_aligned_free(sorted_entities);
    SDL_aligned_free(sorted_grid_x);
    SDL_aligned_free(sorted_grid_y);
}

// Optimized spatial grid query with frustum culling
void spatial_grid_query(SpatialGrid* grid, SDL_FRect query_rect,
    int* result_indices, int* result_count, int max_results,
    EntityManager* entities) {
    PROFILE_FUNCTION();

    // Calculate grid cells that overlap with query rect
    int start_x = (int)(query_rect.x / grid->cell_size);
    int start_y = (int)(query_rect.y / grid->cell_size);
    int end_x = (int)ceil((query_rect.x + query_rect.w) / grid->cell_size);
    int end_y = (int)ceil((query_rect.y + query_rect.h) / grid->cell_size);

    // Clamp to grid bounds
    start_x = std::max(0, std::min(start_x, grid->width - 1));
    start_y = std::max(0, std::min(start_y, grid->height - 1));
    end_x = std::max(0, std::min(end_x, grid->width - 1));
    end_y = std::max(0, std::min(end_y, grid->height - 1));

    *result_count = 0;

    // Pre-allocate batch buffers for entity data
    const int MAX_BATCH_SIZE = 256;
    float* entity_x = static_cast<float*>(SDL_aligned_alloc(MAX_BATCH_SIZE * sizeof(float), CACHE_LINE_SIZE));
    float* entity_y = static_cast<float*>(SDL_aligned_alloc(MAX_BATCH_SIZE * sizeof(float), CACHE_LINE_SIZE));
    float* entity_right = static_cast<float*>(SDL_aligned_alloc(MAX_BATCH_SIZE * sizeof(float), CACHE_LINE_SIZE));
    float* entity_bottom = static_cast<float*>(SDL_aligned_alloc(MAX_BATCH_SIZE * sizeof(float), CACHE_LINE_SIZE));
    int* entity_indices_batch = static_cast<int*>(SDL_aligned_alloc(MAX_BATCH_SIZE * sizeof(int), CACHE_LINE_SIZE));
    bool* entity_visible = static_cast<bool*>(SDL_aligned_alloc(MAX_BATCH_SIZE * sizeof(bool), CACHE_LINE_SIZE));

    // Iterate through cells that overlap the query rect
    for (int y = start_y; y <= end_y && *result_count < max_results; y++) {
        for (int x = start_x; x <= end_x && *result_count < max_results; x++) {
            int cell_entity_count = grid->cell_counts[y][x];
            int* cell_entities = grid->cells[y][x];

            // Process cell entities in batches for better cache usage
            for (int cell_offset = 0; cell_offset < cell_entity_count && *result_count < max_results; cell_offset += MAX_BATCH_SIZE) {
                int batch_size = std::min(MAX_BATCH_SIZE, cell_entity_count - cell_offset);

                // Gather entity data in batches
                for (int i = 0; i < batch_size; i++) {
                    int entity_idx = cell_entities[cell_offset + i];
                    entity_indices_batch[i] = entity_idx;

                    // Get chunk and local index
                    int chunk_idx, local_idx;
                    entities->getChunkIndices(entity_idx, &chunk_idx, &local_idx);
                    EntityChunk* chunk = entities->chunks[chunk_idx];

                    // Extract entity data in batch
                    entity_x[i] = chunk->x[local_idx];
                    entity_y[i] = chunk->y[local_idx];
                    entity_right[i] = chunk->right[local_idx];
                    entity_bottom[i] = chunk->bottom[local_idx];
                    entity_visible[i] = chunk->visible[local_idx];
                }

                // Batch frustum culling test
#pragma omp simd
                for (int i = 0; i < batch_size; i++) {
                    // AABB overlap test with query rect (frustum culling)
                    bool in_frustum = !(
                        entity_right[i] <= query_rect.x ||
                        entity_x[i] >= query_rect.x + query_rect.w ||
                        entity_bottom[i] <= query_rect.y ||
                        entity_y[i] >= query_rect.y + query_rect.h
                        );

                    // Only consider visible entities
                    entity_visible[i] = entity_visible[i] && in_frustum;
                }

                // Add visible entities to result
                for (int i = 0; i < batch_size && *result_count < max_results; i++) {
                    if (entity_visible[i]) {
                        result_indices[(*result_count)++] = entity_indices_batch[i];
                    }
                }
            }
        }
    }

    // Cleanup
    SDL_aligned_free(entity_x);
    SDL_aligned_free(entity_y);
    SDL_aligned_free(entity_right);
    SDL_aligned_free(entity_bottom);
    SDL_aligned_free(entity_indices_batch);
    SDL_aligned_free(entity_visible);
}

// Improved radix sort for entity batches
void radix_sort_by_keys(uint64_t* keys, int* indices, int count) {
    PROFILE_FUNCTION();

    if (count <= 1) return;

    // Allocate temporary arrays with alignment
    uint64_t* temp_keys = static_cast<uint64_t*>(SDL_aligned_alloc(count * sizeof(uint64_t), CACHE_LINE_SIZE));
    int* temp_indices = static_cast<int*>(SDL_aligned_alloc(count * sizeof(int), CACHE_LINE_SIZE));

    // Radix sort using 8 bits per pass (256 buckets)
    const int RADIX_BITS = 8;
    const int RADIX_SIZE = 1 << RADIX_BITS;
    const int RADIX_MASK = RADIX_SIZE - 1;

    // Number of passes needed for 64-bit keys
    const int NUM_PASSES = 8; // 8 passes * 8 bits = 64 bits

    // Initialize source and destination pointers
    uint64_t* src_keys = keys;
    int* src_indices = indices;
    uint64_t* dst_keys = temp_keys;
    int* dst_indices = temp_indices;

    // For each pass (process 8 bits at a time)
    for (int pass = 0; pass < NUM_PASSES; pass++) {
        int shift = pass * RADIX_BITS;

        // Count array for current digit
        int count_array[RADIX_SIZE] = { 0 };

        // Count frequencies
        for (int i = 0; i < count; i++) {
            uint64_t key = src_keys[i];
            int digit = (key >> shift) & RADIX_MASK;
            count_array[digit]++;
        }

        // Compute prefix sum for counting sort
        int start_pos[RADIX_SIZE];
        int total = 0;
        for (int i = 0; i < RADIX_SIZE; i++) {
            start_pos[i] = total;
            total += count_array[i];
        }

        // Place elements into correct positions
        for (int i = 0; i < count; i++) {
            uint64_t key = src_keys[i];
            int digit = (key >> shift) & RADIX_MASK;
            int pos = start_pos[digit]++;

            dst_keys[pos] = key;
            dst_indices[pos] = src_indices[i];
        }

        // Swap source and destination for next iteration
        std::swap(src_keys, dst_keys);
        std::swap(src_indices, dst_indices);
    }

    // If the result ends up in the temp array, copy back to the original arrays
    if (src_keys == temp_keys) {
        memcpy(keys, temp_keys, count * sizeof(uint64_t));
        memcpy(indices, temp_indices, count * sizeof(int));
    }

    // Free temporary arrays
    SDL_aligned_free(temp_keys);
    SDL_aligned_free(temp_indices);
}

// SIMD-optimized batch transform
void transform_entity_batch_simd(float* src_x, float* src_y, float* src_width, float* src_height,
    float* dst_x, float* dst_y, float* dst_width, float* dst_height,
    float scale_x, float scale_y, float translate_x, float translate_y,
    int count) {
    PROFILE_FUNCTION();

    // Process in batches of 8 for AVX
    int i = 0;
    for (; i <= count - 8; i += 8) {
        // Load source data
        __m256 vx = _mm256_loadu_ps(&src_x[i]);
        __m256 vy = _mm256_loadu_ps(&src_y[i]);
        __m256 vw = _mm256_loadu_ps(&src_width[i]);
        __m256 vh = _mm256_loadu_ps(&src_height[i]);

        // Create scale and translate vectors
        __m256 vscale_x = _mm256_set1_ps(scale_x);
        __m256 vscale_y = _mm256_set1_ps(scale_y);
        __m256 vtranslate_x = _mm256_set1_ps(translate_x);
        __m256 vtranslate_y = _mm256_set1_ps(translate_y);

        // Perform transformations
        __m256 vdst_x = _mm256_add_ps(_mm256_mul_ps(vx, vscale_x), vtranslate_x);
        __m256 vdst_y = _mm256_add_ps(_mm256_mul_ps(vy, vscale_y), vtranslate_y);
        __m256 vdst_w = _mm256_mul_ps(vw, vscale_x);
        __m256 vdst_h = _mm256_mul_ps(vh, vscale_y);

        // Store results
        _mm256_storeu_ps(&dst_x[i], vdst_x);
        _mm256_storeu_ps(&dst_y[i], vdst_y);
        _mm256_storeu_ps(&dst_width[i], vdst_w);
        _mm256_storeu_ps(&dst_height[i], vdst_h);
    }

    // Process remaining elements
    for (; i < count; i++) {
        dst_x[i] = src_x[i] * scale_x + translate_x;
        dst_y[i] = src_y[i] * scale_y + translate_y;
        dst_width[i] = src_width[i] * scale_x;
        dst_height[i] = src_height[i] * scale_y;
    }
}

// Initialize texture atlas
void init_texture_atlas(TextureAtlas* atlas, SDL_Renderer* renderer, int width, int height) {
    PROFILE_FUNCTION();

    atlas->texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_TARGET, width, height);
    SDL_SetTextureBlendMode(atlas->texture, SDL_BLENDMODE_BLEND);
    atlas->region_capacity = 32;
    atlas->region_count = 0;
    atlas->regions = static_cast<SDL_FRect*>(SDL_aligned_alloc(
        atlas->region_capacity * sizeof(SDL_FRect), CACHE_LINE_SIZE));
}

// Get visible rect based on camera
SDL_FRect get_visible_rect(Camera* camera) {
    PROFILE_FUNCTION();

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
    PROFILE_FUNCTION();

    // Ensure we have enough space for vertices
    if (batch->vertex_count + 4 > batch->vertex_capacity) {
        int new_capacity = batch->vertex_capacity * 2;
        SDL_Vertex* new_vertices = static_cast<SDL_Vertex*>(SDL_aligned_alloc(
            new_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE));

        // Copy existing data
        memcpy(new_vertices, batch->vertices, batch->vertex_count * sizeof(SDL_Vertex));
        SDL_aligned_free(batch->vertices);
        batch->vertices = new_vertices;
        batch->vertex_capacity = new_capacity;
    }

    // Ensure we have enough space for indices
    if (batch->index_count + 6 > batch->index_capacity) {
        int new_capacity = batch->index_capacity * 2;
        int* new_indices = static_cast<int*>(SDL_aligned_alloc(
            new_capacity * sizeof(int), CACHE_LINE_SIZE));

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
    PROFILE_FUNCTION();

    // Binary search for better performance on many batches
    int left = 0;
    int right = batch_count - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        // Create keys for comparison
        uint64_t mid_key = make_render_sort_key(batches[mid].texture_id, batches[mid].layer);
        uint64_t search_key = make_render_sort_key(texture_id, layer);

        if (mid_key == search_key) {
            return mid;
        }

        if (mid_key < search_key) {
            left = mid + 1;
        }
        else {
            right = mid - 1;
        }
    }

    return -1; // Not found
}

// Create a new batch for a texture/layer combination
void create_batch(RenderBatch** batches, int* batch_count, int texture_id, int layer) {
    PROFILE_FUNCTION();

    // Allocate new memory for batches array
    *batch_count = *batch_count + 1;
    *batches = static_cast<RenderBatch*>(realloc(*batches, (*batch_count) * sizeof(RenderBatch)));

    // Initialize the new batch
    int new_idx = *batch_count - 1;
    (*batches)[new_idx].texture_id = texture_id;
    (*batches)[new_idx].layer = layer;
    (*batches)[new_idx].vertex_capacity = 4096; // Larger initial capacity
    (*batches)[new_idx].index_capacity = 6144; // Larger initial capacity
    (*batches)[new_idx].vertex_count = 0;
    (*batches)[new_idx].index_count = 0;
    (*batches)[new_idx].vertices = static_cast<SDL_Vertex*>(SDL_aligned_alloc(
        (*batches)[new_idx].vertex_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE));
    (*batches)[new_idx].indices = static_cast<int*>(SDL_aligned_alloc(
        (*batches)[new_idx].index_capacity * sizeof(int), CACHE_LINE_SIZE));

    // Sort batches by texture_id and layer for better performance
    // This allows binary search to work
    for (int i = *batch_count - 1; i > 0; i--) {
        uint64_t current_key = make_render_sort_key((*batches)[i].texture_id, (*batches)[i].layer);
        uint64_t prev_key = make_render_sort_key((*batches)[i - 1].texture_id, (*batches)[i - 1].layer);

        if (current_key < prev_key) {
            // Swap batches
            RenderBatch temp = (*batches)[i];
            (*batches)[i] = (*batches)[i - 1];
            (*batches)[i - 1] = temp;
        }
        else {
            break;
        }
    }
}

// Update which grid cells are visible and load/unload as needed
void update_dynamic_loading(Engine* engine) {
    PROFILE_FUNCTION();

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
    start_x = std::max(0, std::min(start_x, engine->grid_width - 1));
    start_y = std::max(0, std::min(start_y, engine->grid_height - 1));
    end_x = std::max(0, std::min(end_x, engine->grid_width - 1));
    end_y = std::max(0, std::min(end_y, engine->grid_height - 1));

    // Track changes to visibility state
    bool visibility_changed = false;

    // Use a bitmap to mark visible cells (1 = visible, 0 = not visible)
    int grid_size = engine->grid_width * engine->grid_height;
    int bitset_size = (grid_size + 31) / 32;
    uint32_t* visibility_bitmap = static_cast<uint32_t*>(SDL_aligned_alloc(bitset_size * sizeof(uint32_t), CACHE_LINE_SIZE));
    memset(visibility_bitmap, 0, bitset_size * sizeof(uint32_t));

    // Mark visible cells in bitmap
    for (int y = start_y; y <= end_y; y++) {
        for (int x = start_x; x <= end_x; x++) {
            int index = y * engine->grid_width + x;
            int bit_index = index / 32;
            int bit_offset = index % 32;

            visibility_bitmap[bit_index] |= (1 << bit_offset);

            // Check if visibility changed
            if (!engine->grid_loaded[x][y]) {
                engine->grid_loaded[x][y] = true;
                visibility_changed = true;
            }
        }
    }

    // Now check cells that should be unloaded
    for (int y = 0; y < engine->grid_height; y++) {
        for (int x = 0; x < engine->grid_width; x++) {
            int index = y * engine->grid_width + x;
            int bit_index = index / 32;
            int bit_offset = index % 32;

            // If cell is marked as loaded but not in the visibility bitmap
            if (engine->grid_loaded[x][y] && !(visibility_bitmap[bit_index] & (1 << bit_offset))) {
                engine->grid_loaded[x][y] = false;
                visibility_changed = true;
            }
        }
    }

    // Update entity visible states if visibility changed
    if (visibility_changed) {
        // Process in chunks for better cache locality
        const int ENTITY_BATCH_SIZE = 512;
        int* chunk_indices = static_cast<int*>(SDL_aligned_alloc(ENTITY_BATCH_SIZE * sizeof(int), CACHE_LINE_SIZE));
        int* local_indices = static_cast<int*>(SDL_aligned_alloc(ENTITY_BATCH_SIZE * sizeof(int), CACHE_LINE_SIZE));
        float* entity_x = static_cast<float*>(SDL_aligned_alloc(ENTITY_BATCH_SIZE * sizeof(float), CACHE_LINE_SIZE));
        float* entity_y = static_cast<float*>(SDL_aligned_alloc(ENTITY_BATCH_SIZE * sizeof(float), CACHE_LINE_SIZE));
        bool* entity_active = static_cast<bool*>(SDL_aligned_alloc(ENTITY_BATCH_SIZE * sizeof(bool), CACHE_LINE_SIZE));
        bool* cell_loaded = static_cast<bool*>(SDL_aligned_alloc(ENTITY_BATCH_SIZE * sizeof(bool), CACHE_LINE_SIZE));

        // Process all entities
        int total_entities = 0;
        for (int chunk_idx = 0; chunk_idx < engine->entities.chunk_count; chunk_idx++) {
            EntityChunk* chunk = engine->entities.chunks[chunk_idx];

            for (int batch_start = 0; batch_start < chunk->count; batch_start += ENTITY_BATCH_SIZE) {
                int batch_size = std::min(ENTITY_BATCH_SIZE, chunk->count - batch_start);

                // Extract batch data
                for (int i = 0; i < batch_size; i++) {
                    int local_idx = batch_start + i;
                    chunk_indices[i] = chunk_idx;
                    local_indices[i] = local_idx;
                    entity_x[i] = chunk->x[local_idx];
                    entity_y[i] = chunk->y[local_idx];
                    entity_active[i] = chunk->active[local_idx];
                }

                // Calculate grid cells for all entities in batch
#pragma omp simd
                for (int i = 0; i < batch_size; i++) {
                    int grid_x = (int)(entity_x[i] / engine->grid_cell_size);
                    int grid_y = (int)(entity_y[i] / engine->grid_cell_size);

                    // Clamp to grid bounds
                    grid_x = std::max(0, std::min(grid_x, engine->grid_width - 1));
                    grid_y = std::max(0, std::min(grid_y, engine->grid_height - 1));

                    // Check if cell is loaded
                    cell_loaded[i] = engine->grid_loaded[grid_x][grid_y];
                }

                // Update visibility state - only active entities can be visible
                for (int i = 0; i < batch_size; i++) {
                    EntityChunk* batch_chunk = engine->entities.chunks[chunk_indices[i]];
                    bool was_visible = batch_chunk->visible[local_indices[i]];
                    bool should_be_visible = entity_active[i] && cell_loaded[i];

                    // Only update if visibility changed
                    if (was_visible != should_be_visible) {
                        batch_chunk->visible[local_indices[i]] = should_be_visible;
                    }
                }

                total_entities += batch_size;
            }
        }

        // Cleanup
        SDL_aligned_free(chunk_indices);
        SDL_aligned_free(local_indices);
        SDL_aligned_free(entity_x);
        SDL_aligned_free(entity_y);
        SDL_aligned_free(entity_active);
        SDL_aligned_free(cell_loaded);
    }

    SDL_aligned_free(visibility_bitmap);
}

// Update transforms for entities with dirty flags
void update_dirty_transforms(Engine* engine) {
    PROFILE_FUNCTION();

    // First update root entities with dirty transforms
    for (int chunk_idx = 0; chunk_idx < engine->entities.chunk_count; chunk_idx++) {
        EntityChunk* chunk = engine->entities.chunks[chunk_idx];

        if (!chunk->transform_dirty_flags) continue;

        // Process each 64-bit chunk of the dirty flags
        for (int flag_idx = 0; flag_idx < (chunk->capacity + 63) / 64; flag_idx++) {
            uint64_t dirty_mask = chunk->transform_dirty_flags[flag_idx];
            if (!dirty_mask) continue;

            // Process each bit in the mask
            while (dirty_mask) {
                // Find the index of the lowest set bit
                int bit_pos = _tzcnt_u64(dirty_mask);
                int local_idx = flag_idx * 64 + bit_pos;

                // Skip if not a root entity
                if (local_idx >= chunk->count || chunk->parent_id[local_idx] != -1) {
                    // Clear the bit and continue
                    dirty_mask &= ~(1ULL << bit_pos);
                    continue;
                }

                // Update root entity transform
                chunk->x[local_idx] = chunk->local_x[local_idx];
                chunk->y[local_idx] = chunk->local_y[local_idx];

                // Update precomputed bounds
                chunk->right[local_idx] = chunk->x[local_idx] + chunk->width[local_idx];
                chunk->bottom[local_idx] = chunk->y[local_idx] + chunk->height[local_idx];

                // Clear the dirty bit
                dirty_mask &= ~(1ULL << bit_pos);
            }
        }
    }

    // Use BFS for hierarchy updates (better cache efficiency)
    const int MAX_QUEUE_SIZE = 100000;
    int* queue = static_cast<int*>(SDL_aligned_alloc(MAX_QUEUE_SIZE * sizeof(int), CACHE_LINE_SIZE));
    int queue_head = 0;
    int queue_tail = 0;

    // Enqueue all root entities with dirty transforms that have children
    for (int chunk_idx = 0; chunk_idx < engine->entities.chunk_count; chunk_idx++) {
        EntityChunk* chunk = engine->entities.chunks[chunk_idx];

        if (!chunk->transform_dirty_flags) continue;

        for (int flag_idx = 0; flag_idx < (chunk->capacity + 63) / 64; flag_idx++) {
            uint64_t dirty_mask = chunk->transform_dirty_flags[flag_idx];
            if (!dirty_mask) continue;

            while (dirty_mask) {
                int bit_pos = _tzcnt_u64(dirty_mask);
                int local_idx = flag_idx * 64 + bit_pos;

                if (local_idx < chunk->count &&
                    chunk->parent_id[local_idx] == -1 &&
                    chunk->first_child_id[local_idx] != -1) {

                    int entity_id = chunk_idx * ENTITY_CHUNK_SIZE + local_idx;
                    queue[queue_tail++] = entity_id;
                    if (queue_tail >= MAX_QUEUE_SIZE) queue_tail = 0;
                }

                // Clear the bit
                dirty_mask &= ~(1ULL << bit_pos);
            }
        }
    }

    // Process queue
    while (queue_head != queue_tail) {
        int parent_id = queue[queue_head++];
        if (queue_head >= MAX_QUEUE_SIZE) queue_head = 0;

        int parent_chunk_idx, parent_local_idx;
        engine->entities.getChunkIndices(parent_id, &parent_chunk_idx, &parent_local_idx);
        EntityChunk* parent_chunk = engine->entities.chunks[parent_chunk_idx];

        // Get parent position
        float parent_x = parent_chunk->x[parent_local_idx];
        float parent_y = parent_chunk->y[parent_local_idx];

        // Process all children
        int child_id = parent_chunk->first_child_id[parent_local_idx];
        while (child_id != -1) {
            int child_chunk_idx, child_local_idx;
            engine->entities.getChunkIndices(child_id, &child_chunk_idx, &child_local_idx);
            EntityChunk* child_chunk = engine->entities.chunks[child_chunk_idx];

            // Update child position based on parent
            child_chunk->x[child_local_idx] = parent_x + child_chunk->local_x[child_local_idx];
            child_chunk->y[child_local_idx] = parent_y + child_chunk->local_y[child_local_idx];

            // Update precomputed bounds
            child_chunk->right[child_local_idx] = child_chunk->x[child_local_idx] + child_chunk->width[child_local_idx];
            child_chunk->bottom[child_local_idx] = child_chunk->y[child_local_idx] + child_chunk->height[child_local_idx];

            // Mark transform as clean for this entity
            int flag_idx = child_local_idx / 64;
            int bit_pos = child_local_idx % 64;
            child_chunk->transform_dirty_flags[flag_idx] &= ~(1ULL << bit_pos);

            // If this child has children, enqueue it
            if (child_chunk->first_child_id[child_local_idx] != -1) {
                queue[queue_tail++] = child_id;
                if (queue_tail >= MAX_QUEUE_SIZE) queue_tail = 0;
            }

            // Move to next sibling
            child_id = child_chunk->next_sibling_id[child_local_idx];
        }
    }

    SDL_aligned_free(queue);
}

// Engine API implementations
Engine* engine_create(int window_width, int window_height, int world_width, int world_height, int cell_size) {
    PROFILE_FUNCTION();

    Engine* engine = static_cast<Engine*>(malloc(sizeof(Engine)));
    if (!engine) return NULL;

    // Use placement new to properly initialize C++ members
    new (&engine->entity_indices_pool) FixedBufferPool(100000 * sizeof(int), 4);
    new (&engine->screen_coords_pool) FixedBufferPool(100000 * sizeof(float) * 4, 4);
    new (&engine->sort_keys_pool) FixedBufferPool(100000 * sizeof(uint64_t), 4);
    new (&engine->entities) EntityManager();
    new (&engine->type_id_to_index) std::unordered_map<int, int>();

    // Initialize entity type system with capacity for 256K+ types
    engine->entity_type_count = 0;
    engine->entity_type_capacity = 262144;  // 256K initial capacity
    engine->entity_types = static_cast<EntityTypeConfig*>(
        malloc(engine->entity_type_capacity * sizeof(EntityTypeConfig))
        );

    // Initialize performance tracking
    engine->active_entity_count = 0;
    engine->update_time = 0.0f;
    engine->render_time = 0.0f;

    // Create window and renderer
    engine->window = SDL_CreateWindow("2D Game Engine - SDL3", window_width, window_height, 0);
    if (!engine->window) {
        engine->entity_indices_pool.~FixedBufferPool();
        engine->screen_coords_pool.~FixedBufferPool();
        engine->sort_keys_pool.~FixedBufferPool();
        engine->entities.~EntityManager();
        free(engine);
        return NULL;
    }

    engine->renderer = SDL_CreateRenderer(engine->window, NULL);
    if (!engine->renderer) {
        SDL_DestroyWindow(engine->window);
        engine->entity_indices_pool.~FixedBufferPool();
        engine->screen_coords_pool.~FixedBufferPool();
        engine->sort_keys_pool.~FixedBufferPool();
        engine->entities.~EntityManager();
        free(engine);
        return NULL;
    }

    // Init world bounds
    engine->world_bounds.x = 0;
    engine->world_bounds.y = 0;
    engine->world_bounds.w = world_width;
    engine->world_bounds.h = world_height;

    // Init spatial grid (replaces quadtree)
    init_spatial_grid(&engine->grid, world_width, world_height, cell_size);

    // Init texture atlas
    init_texture_atlas(&engine->atlas, engine->renderer, 2048, 2048);

    // Init camera
    engine->camera.x = 0;
    engine->camera.y = 0;
    engine->camera.width = window_width;
    engine->camera.height = window_height;
    engine->camera.zoom = 1.0f;

    // Init render batches (for batch rendering)
    engine->batch_count = 8;
    engine->batches = static_cast<RenderBatch*>(malloc(engine->batch_count * sizeof(RenderBatch)));
    for (int i = 0; i < engine->batch_count; i++) {
        engine->batches[i].texture_id = i;
        engine->batches[i].layer = 0;
        engine->batches[i].vertex_capacity = 4096;
        engine->batches[i].index_capacity = 6144;
        engine->batches[i].vertex_count = 0;
        engine->batches[i].index_count = 0;
        engine->batches[i].vertices = static_cast<SDL_Vertex*>(SDL_aligned_alloc(
            engine->batches[i].vertex_capacity * sizeof(SDL_Vertex), CACHE_LINE_SIZE));
        engine->batches[i].indices = static_cast<int*>(SDL_aligned_alloc(
            engine->batches[i].index_capacity * sizeof(int), CACHE_LINE_SIZE));
    }

    // Init dynamic grid loading
    engine->grid_cell_size = 256.0f;
    engine->grid_width = (int)ceil(world_width / engine->grid_cell_size);
    engine->grid_height = (int)ceil(world_height / engine->grid_cell_size);

    // Allocate grid loaded array
    engine->grid_loaded = static_cast<bool**>(malloc(engine->grid_width * sizeof(bool*)));
    for (int x = 0; x < engine->grid_width; x++) {
        engine->grid_loaded[x] = static_cast<bool*>(malloc(engine->grid_height * sizeof(bool)));
        for (int y = 0; y < engine->grid_height; y++) {
            engine->grid_loaded[x][y] = false;
        }
    }

    // Init timing
    engine->last_frame_time = SDL_GetTicks();
    engine->fps = 0.0f;

    return engine;
}

void engine_destroy(Engine* engine) {
    PROFILE_FUNCTION();

    if (!engine) return;

    // Free entity type system
    free(engine->entity_types);

    // Call destructors for C++ members
    engine->entities.~EntityManager();
    engine->entity_indices_pool.~FixedBufferPool();
    engine->screen_coords_pool.~FixedBufferPool();
    engine->sort_keys_pool.~FixedBufferPool();

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

    // Free SDL resources
    SDL_DestroyRenderer(engine->renderer);
    SDL_DestroyWindow(engine->window);

    free(engine);
}

// Modified engine_update to include entity type updates
void engine_update(Engine* engine) {
    PROFILE_FUNCTION();

    // Calculate delta time
    Uint64 current_time = SDL_GetTicks();
    float delta_time = (current_time - engine->last_frame_time) / 1000.0f;
    engine->last_frame_time = current_time;

    // Smooth FPS calculation
    engine->fps = 0.95f * engine->fps + 0.05f * (1.0f / delta_time);

    // Update dynamic loading based on camera position (updates visibility)
    {
        PROFILE_SCOPE("update_dynamic_loading");
        update_dynamic_loading(engine);
    }

    // Update transforms for hierarchy with optimized dirty flag tracking
    {
        PROFILE_SCOPE("update_dirty_transforms");
        update_dirty_transforms(engine);
    }

    // Update all entity types (now regardless of visibility)
    {
        PROFILE_SCOPE("update_entity_types");
        engine_update_entity_types(engine, delta_time);
    }

    // Clear spatial grid
    {
        PROFILE_SCOPE("clear_spatial_grid");
        clear_spatial_grid(&engine->grid);
    }

    // Rebuild spatial grid with visible entities (not just active)
    // Use batch processing for better cache coherence
    {
        PROFILE_SCOPE("rebuild_spatial_grid");

        // Extract active entity data for batch processing
        const int MAX_BATCH_SIZE = 1024;
        int* batch_entities = static_cast<int*>(SDL_aligned_alloc(MAX_BATCH_SIZE * sizeof(int), CACHE_LINE_SIZE));
        float* batch_x = static_cast<float*>(SDL_aligned_alloc(MAX_BATCH_SIZE * sizeof(float), CACHE_LINE_SIZE));
        float* batch_y = static_cast<float*>(SDL_aligned_alloc(MAX_BATCH_SIZE * sizeof(float), CACHE_LINE_SIZE));

        for (int chunk_idx = 0; chunk_idx < engine->entities.chunk_count; chunk_idx++) {
            EntityChunk* chunk = engine->entities.chunks[chunk_idx];

            int batch_count = 0;
            for (int local_idx = 0; local_idx < chunk->count; local_idx++) {
                if (chunk->visible[local_idx]) {  // Changed from active to visible
                    int entity_idx = chunk_idx * ENTITY_CHUNK_SIZE + local_idx;

                    // Add to batch
                    batch_entities[batch_count] = entity_idx;
                    batch_x[batch_count] = chunk->x[local_idx];
                    batch_y[batch_count] = chunk->y[local_idx];
                    batch_count++;

                    // Process batch if full
                    if (batch_count >= MAX_BATCH_SIZE) {
                        spatial_grid_add_batch(&engine->grid, batch_entities, batch_x, batch_y, batch_count, &engine->entities);
                        batch_count = 0;
                    }
                }
            }

            // Process remaining entities in batch
            if (batch_count > 0) {
                spatial_grid_add_batch(&engine->grid, batch_entities, batch_x, batch_y, batch_count, &engine->entities);
            }
        }

        // Cleanup
        SDL_aligned_free(batch_entities);
        SDL_aligned_free(batch_x);
        SDL_aligned_free(batch_y);
    }
}

void engine_render(Engine* engine) {
    PROFILE_FUNCTION();

    // Clear screen
    SDL_SetRenderDrawColor(engine->renderer, 0, 0, 0, 255);
    SDL_RenderClear(engine->renderer);

    // Get visible area for culling
    SDL_FRect visible_rect = get_visible_rect(&engine->camera);

    // Get reusable buffer for visible entities
    auto indices_handle = engine->entity_indices_pool.getBuffer();
    int* visible_indices = reinterpret_cast<int*>(indices_handle.data());
    int visible_count = 0;

    // Query visible entities from spatial grid
    {
        PROFILE_SCOPE("spatial_grid_query");
        spatial_grid_query(&engine->grid, visible_rect, visible_indices, &visible_count,
            engine->entities.total_count, &engine->entities);
    }

    // Sort visible entities by texture and z-order using 64-bit combined keys
    {
        PROFILE_SCOPE("sort_entities_by_key");

        // Get reusable buffer for sort keys
        auto keys_handle = engine->sort_keys_pool.getBuffer();
        uint64_t* sort_keys = reinterpret_cast<uint64_t*>(keys_handle.data());

        // Create combined texture/layer keys for sorting
        for (int i = 0; i < visible_count; i++) {
            int entity_idx = visible_indices[i];

            // Get chunk and local index
            int chunk_idx, local_idx;
            engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);

            EntityChunk* chunk = engine->entities.chunks[chunk_idx];
            int texture_id = chunk->texture_id[local_idx];
            int layer = chunk->layer[local_idx];

            // Create combined key
            sort_keys[i] = make_render_sort_key(texture_id, layer);
        }

        // Allocate temporary buffer for sorted results
        int* sorted_indices = static_cast<int*>(SDL_aligned_alloc(visible_count * sizeof(int), CACHE_LINE_SIZE));

        // Sort entities by keys
        radix_sort_by_keys(sort_keys, visible_indices, visible_count);

        // Copy sorted results back
        memcpy(visible_indices, sorted_indices, visible_count * sizeof(int));

        // Cleanup
        SDL_aligned_free(sorted_indices);
    }

    // Clear batches and reuse them
    {
        PROFILE_SCOPE("clear_batches");
        for (int i = 0; i < engine->batch_count; i++) {
            engine->batches[i].vertex_count = 0;
            engine->batches[i].index_count = 0;
        }
    }

    // Get reusable buffers for transformation
    auto coords_handle = engine->screen_coords_pool.getBuffer();
    float* screen_coords = reinterpret_cast<float*>(coords_handle.data());
    float* screen_x = &screen_coords[0];
    float* screen_y = &screen_coords[visible_count];
    float* screen_w = &screen_coords[visible_count * 2];
    float* screen_h = &screen_coords[visible_count * 3];

    // Extract entity data to contiguous arrays for processing
    std::vector<float> world_x(visible_count);
    std::vector<float> world_y(visible_count);
    std::vector<float> width_f(visible_count);
    std::vector<float> height_f(visible_count);

    // Copy entity data to contiguous arrays for processing
    {
        PROFILE_SCOPE("collect_entity_data");
        for (int i = 0; i < visible_count; i++) {
            int entity_idx = visible_indices[i];

            // Get chunk and local index
            int chunk_idx, local_idx;
            engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);

            EntityChunk* chunk = engine->entities.chunks[chunk_idx];
            world_x[i] = chunk->x[local_idx];
            world_y[i] = chunk->y[local_idx];
            width_f[i] = static_cast<float>(chunk->width[local_idx]);
            height_f[i] = static_cast<float>(chunk->height[local_idx]);
        }
    }

    // Calculate screen coordinates with SIMD optimization
    {
        PROFILE_SCOPE("transform_entity_batch_simd");
        transform_entity_batch_simd(
            world_x.data(), world_y.data(), width_f.data(), height_f.data(),
            screen_x, screen_y, screen_w, screen_h,
            engine->camera.zoom, engine->camera.zoom,
            -visible_rect.x * engine->camera.zoom, -visible_rect.y * engine->camera.zoom,
            visible_count
        );
    }

    // Track last texture/layer to batch together
    int last_texture_id = -1;
    int last_layer = -1;
    int current_batch_idx = -1;

    // Add visible entities to batches
    {
        PROFILE_SCOPE("add_to_batches");
        for (int i = 0; i < visible_count; i++) {
            int entity_idx = visible_indices[i];

            // Get chunk and local index
            int chunk_idx, local_idx;
            engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);

            EntityChunk* chunk = engine->entities.chunks[chunk_idx];
            int texture_id = chunk->texture_id[local_idx];
            int layer = chunk->layer[local_idx];

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
    }

    // Render batches
    {
        PROFILE_SCOPE("render_batches");
        for (int i = 0; i < engine->batch_count; i++) {
            if (engine->batches[i].vertex_count > 0) {
                // Single draw call per batch!
                SDL_RenderGeometry(engine->renderer, engine->atlas.texture,
                    engine->batches[i].vertices, engine->batches[i].vertex_count,
                    engine->batches[i].indices, engine->batches[i].index_count);
            }
        }
    }
}

int engine_add_entity(Engine* engine, float x, float y, int width, int height,
    int texture_id, int layer) {
    PROFILE_FUNCTION();

    // Add entity to manager
    int entity_idx = engine->entities.addEntity();

    // Get chunk and local index
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);

    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    // Set entity properties
    chunk->x[local_idx] = x;
    chunk->y[local_idx] = y;
    chunk->width[local_idx] = width;
    chunk->height[local_idx] = height;

    // Precompute right and bottom for faster culling
    chunk->right[local_idx] = x + width;
    chunk->bottom[local_idx] = y + height;

    chunk->texture_id[local_idx] = texture_id;
    chunk->layer[local_idx] = layer;

    // Initialize hierarchy properties
    chunk->parent_id[local_idx] = -1;
    chunk->first_child_id[local_idx] = -1;
    chunk->next_sibling_id[local_idx] = -1;
    chunk->local_x[local_idx] = x;
    chunk->local_y[local_idx] = y;

    // Set active by default and add to active indices list
    chunk->active[local_idx] = true;
    chunk->active_indices[chunk->active_count++] = local_idx;

    // Determine if entity should be visible based on grid loading
    int grid_x = (int)(x / engine->grid_cell_size);
    int grid_y = (int)(y / engine->grid_cell_size);

    if (grid_x >= 0 && grid_x < engine->grid_width &&
        grid_y >= 0 && grid_y < engine->grid_height) {
        chunk->visible[local_idx] = engine->grid_loaded[grid_x][grid_y];
    }
    else {
        chunk->visible[local_idx] = false;
    }

    // Add to spatial grid if visible
    if (chunk->visible[local_idx]) {
        spatial_grid_add(&engine->grid, entity_idx, x, y, &engine->entities);
    }

    return entity_idx;
}

int engine_add_texture(Engine* engine, SDL_Surface* surface, int x, int y) {
    PROFILE_FUNCTION();

    int texture_id = engine->atlas.region_count;

    // Ensure capacity
    if (texture_id >= engine->atlas.region_capacity) {
        int new_capacity = engine->atlas.region_capacity * 2;
        SDL_FRect* new_regions = static_cast<SDL_FRect*>(SDL_aligned_alloc(
            new_capacity * sizeof(SDL_FRect), CACHE_LINE_SIZE));

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
    PROFILE_FUNCTION();

    engine->camera.x = x;
    engine->camera.y = y;
}

void engine_set_camera_zoom(Engine* engine, float zoom) {
    PROFILE_FUNCTION();

    engine->camera.zoom = zoom;
}

void engine_set_entity_active(Engine* engine, int entity_idx, bool active) {
    PROFILE_FUNCTION();

    if (!engine->entities.isValidEntity(entity_idx)) {
        return;
    }

    // Get chunk and local index
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);

    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    // Only update if state changed
    if (chunk->active[local_idx] != active) {
        chunk->active[local_idx] = active;

        if (active) {
            // Add to active list if becoming active
            if (chunk->active_count < chunk->capacity) {
                chunk->active_indices[chunk->active_count++] = local_idx;
            }
        }
        else {
            // Remove from active list if becoming inactive
            for (int i = 0; i < chunk->active_count; i++) {
                if (chunk->active_indices[i] == local_idx) {
                    // Remove by swapping with last element and decreasing count
                    chunk->active_indices[i] = chunk->active_indices[--chunk->active_count];
                    break;
                }
            }

            // If not active, also not visible
            chunk->visible[local_idx] = false;
        }
    }
}

SDL_FRect engine_get_visible_rect(Engine* engine) {
    PROFILE_FUNCTION();

    return get_visible_rect(&engine->camera);
}

int engine_get_visible_entities_count(Engine* engine) {
    PROFILE_FUNCTION();

    SDL_FRect visible_rect = get_visible_rect(&engine->camera);

    auto handle = engine->entity_indices_pool.getBuffer();
    int* visible_indices = reinterpret_cast<int*>(handle.data());
    int visible_count = 0;

    spatial_grid_query(&engine->grid, visible_rect, visible_indices, &visible_count,
        engine->entities.total_count, &engine->entities);

    return visible_count;
}

// Set entity's parent
void engine_set_parent(Engine* engine, int entity_id, int parent_id) {
    PROFILE_FUNCTION();

    if (!engine->entities.isValidEntity(entity_id)) {
        return;
    }

    // Don't allow circular references
    int p = parent_id;
    while (p != -1) {
        if (p == entity_id) {
            return; // Would create a cycle
        }
        p = engine_get_parent(engine, p);
    }

    // Get chunk and local indices for entity
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    // Store current world position as local position if changing parent
    int old_parent = chunk->parent_id[local_idx];
    if (old_parent != parent_id) {
        // Store current world position in local coordinates
        chunk->local_x[local_idx] = chunk->x[local_idx];
        chunk->local_y[local_idx] = chunk->y[local_idx];

        // If had previous parent, detach from old parent's child list
        if (old_parent != -1) {
            engine_remove_parent(engine, entity_id);
        }

        // Set new parent
        chunk->parent_id[local_idx] = parent_id;

        // If has new parent, add to new parent's child list
        if (parent_id != -1 && engine->entities.isValidEntity(parent_id)) {
            int parent_chunk_idx, parent_local_idx;
            engine->entities.getChunkIndices(parent_id, &parent_chunk_idx, &parent_local_idx);
            EntityChunk* parent_chunk = engine->entities.chunks[parent_chunk_idx];

            // Get parent's first child
            int first_child = parent_chunk->first_child_id[parent_local_idx];

            // Make this entity the first child and link to old first child
            parent_chunk->first_child_id[parent_local_idx] = entity_id;
            chunk->next_sibling_id[local_idx] = first_child;

            // Update entity's local position relative to parent
            if (first_child != -1) {
                chunk->local_x[local_idx] = chunk->x[local_idx] - parent_chunk->x[parent_local_idx];
                chunk->local_y[local_idx] = chunk->y[local_idx] - parent_chunk->y[parent_local_idx];
            }

            // Mark parent as dirty
            mark_entity_transform_dirty(parent_chunk, parent_local_idx);
        }

        // Mark entity as dirty
        mark_entity_transform_dirty(chunk, local_idx);

        // Update transforms for this hierarchy branch only (selective update)
        update_dirty_transforms(engine);
    }
}

// Remove entity from its parent
void engine_remove_parent(Engine* engine, int entity_id) {
    PROFILE_FUNCTION();

    if (!engine->entities.isValidEntity(entity_id)) {
        return;
    }

    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    int parent_id = chunk->parent_id[local_idx];
    if (parent_id == -1) {
        return; // No parent to remove
    }

    // Find parent
    int parent_chunk_idx, parent_local_idx;
    engine->entities.getChunkIndices(parent_id, &parent_chunk_idx, &parent_local_idx);
    EntityChunk* parent_chunk = engine->entities.chunks[parent_chunk_idx];

    // Find and remove this entity from parent's child list
    int* child_id_ptr = &parent_chunk->first_child_id[parent_local_idx];
    while (*child_id_ptr != -1) {
        if (*child_id_ptr == entity_id) {
            // Get the sibling that will replace this entity in the list
            int next_sibling = chunk->next_sibling_id[local_idx];
            *child_id_ptr = next_sibling;
            break;
        }

        // Move to next sibling
        int sibling_chunk_idx, sibling_local_idx;
        engine->entities.getChunkIndices(*child_id_ptr, &sibling_chunk_idx, &sibling_local_idx);
        EntityChunk* sibling_chunk = engine->entities.chunks[sibling_chunk_idx];
        child_id_ptr = &sibling_chunk->next_sibling_id[sibling_local_idx];
    }

    // Reset parent relationship
    chunk->parent_id[local_idx] = -1;
    chunk->next_sibling_id[local_idx] = -1;

    // Mark entity as dirty
    mark_entity_transform_dirty(chunk, local_idx);
}

// Get entity's parent
int engine_get_parent(Engine* engine, int entity_id) {
    PROFILE_FUNCTION();

    if (!engine->entities.isValidEntity(entity_id)) {
        return -1;
    }

    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    return chunk->parent_id[local_idx];
}

// Get entity's first child
int engine_get_first_child(Engine* engine, int entity_id) {
    PROFILE_FUNCTION();

    if (!engine->entities.isValidEntity(entity_id)) {
        return -1;
    }

    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    return chunk->first_child_id[local_idx];
}

// Get entity's next sibling
int engine_get_next_sibling(Engine* engine, int entity_id) {
    PROFILE_FUNCTION();

    if (!engine->entities.isValidEntity(entity_id)) {
        return -1;
    }

    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    return chunk->next_sibling_id[local_idx];
}

// Set entity's local position (relative to parent)
void engine_set_entity_local_position(Engine* engine, int entity_id, float x, float y) {
    PROFILE_FUNCTION();

    if (!engine->entities.isValidEntity(entity_id)) {
        return;
    }

    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    chunk->local_x[local_idx] = x;
    chunk->local_y[local_idx] = y;

    // Mark entity as dirty
    mark_entity_transform_dirty(chunk, local_idx);

    // Update transforms selectively (only for this hierarchy branch)
    update_dirty_transforms(engine);
}

// Update entity transforms (rewritten to use dirty flags)
void engine_update_entity_transforms(Engine* engine) {
    PROFILE_FUNCTION();

    // Use optimized dirty transform update
    update_dirty_transforms(engine);
}

void engine_set_entity_position(Engine* engine, int entity_idx, float x, float y) {
    PROFILE_FUNCTION();

    if (!engine->entities.isValidEntity(entity_idx)) {
        return;
    }

    // Get chunk and local index
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    // Check if entity has a parent
    if (chunk->parent_id[local_idx] != -1) {
        // For entities with parents, we're setting world position, so we need
        // to calculate local position based on parent's world position
        int parent_id = chunk->parent_id[local_idx];
        int parent_chunk_idx, parent_local_idx;
        engine->entities.getChunkIndices(parent_id, &parent_chunk_idx, &parent_local_idx);
        EntityChunk* parent_chunk = engine->entities.chunks[parent_chunk_idx];

        // Set local position relative to parent
        chunk->local_x[local_idx] = x - parent_chunk->x[parent_local_idx];
        chunk->local_y[local_idx] = y - parent_chunk->y[parent_local_idx];
    }
    else {
        // For root entities, local position is the same as world position
        chunk->local_x[local_idx] = x;
        chunk->local_y[local_idx] = y;
    }

    chunk->x[local_idx] = x;
    chunk->y[local_idx] = y;

    // Update precomputed values
    chunk->right[local_idx] = x + chunk->width[local_idx];
    chunk->bottom[local_idx] = y + chunk->height[local_idx];

    // Mark entity as dirty
    mark_entity_transform_dirty(chunk, local_idx);

    // Update child entities selectively
    if (chunk->first_child_id[local_idx] != -1) {
        update_dirty_transforms(engine);
    }
}

int engine_add_child_entity(Engine* engine, int parent_id, float local_x, float local_y,
    int width, int height, int texture_id, int layer) {
    PROFILE_FUNCTION();

    if (!engine->entities.isValidEntity(parent_id)) {
        return -1;
    }

    // Get parent's world position
    int parent_chunk_idx, parent_local_idx;
    engine->entities.getChunkIndices(parent_id, &parent_chunk_idx, &parent_local_idx);
    EntityChunk* parent_chunk = engine->entities.chunks[parent_chunk_idx];

    float parent_x = parent_chunk->x[parent_local_idx];
    float parent_y = parent_chunk->y[parent_local_idx];

    // Create entity at parent's position + local offset
    int entity_id = engine_add_entity(engine, parent_x + local_x, parent_y + local_y,
        width, height, texture_id, layer);

    // Set parent relationship
    engine_set_parent(engine, entity_id, parent_id);

    // Make sure local position is set correctly
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];
    chunk->local_x[local_idx] = local_x;
    chunk->local_y[local_idx] = local_y;

    // Mark as dirty
    mark_entity_transform_dirty(chunk, local_idx);

    return entity_id;
}

// Register an entity type with the engine
void engine_register_entity_type(Engine* engine, int type_id, EntityTypeUpdateFunc update_func, size_t extra_data_size) {
    PROFILE_FUNCTION();

    // Ensure capacity for new type
    if (engine->entity_type_count >= engine->entity_type_capacity) {
        int new_capacity = engine->entity_type_capacity * 2;
        engine->entity_types = static_cast<EntityTypeConfig*>(realloc(
            engine->entity_types,
            new_capacity * sizeof(EntityTypeConfig)
        ));
        engine->entity_type_capacity = new_capacity;
    }

    // Add new type
    int type_index = engine->entity_type_count++;
    engine->type_id_to_index[type_id] = type_index;  // Store mapping for O(1) lookup

    EntityTypeConfig* config = &engine->entity_types[type_index];
    config->type_id = type_id;
    config->update_func = update_func;
    config->extra_data_size = extra_data_size;
    config->instance_count = 0;
    config->last_update_time = 0.0f;
    config->first_chunk_idx = -1;  // No chunks of this type yet
}

// Find or create a chunk for a specific entity type
int find_or_create_chunk_for_type(Engine* engine, int type_id) {
    PROFILE_FUNCTION();

    // Get the type index from the mapping
    auto it = engine->type_id_to_index.find(type_id);
    if (it == engine->type_id_to_index.end()) {
        // Type not registered
        return -1;
    }

    int type_index = it->second;
    EntityTypeConfig* config = &engine->entity_types[type_index];

    // First, look for an existing chunk of this type with available space
    int chunk_idx = config->first_chunk_idx;
    int prev_chunk_idx = -1;

    while (chunk_idx != -1) {
        EntityChunk* chunk = engine->entities.chunks[chunk_idx];
        if (chunk->count < chunk->capacity) {
            return chunk_idx;
        }
        prev_chunk_idx = chunk_idx;
        chunk_idx = chunk->next_chunk_of_type;
    }

    // No chunk found with space, create a new one
    if (engine->entities.chunk_count >= engine->entities.chunks_capacity) {
        engine->entities.chunks_capacity *= 2;
        engine->entities.chunks = static_cast<EntityChunk**>(realloc(
            engine->entities.chunks,
            engine->entities.chunks_capacity * sizeof(EntityChunk*)
        ));
    }

    // Create new chunk
    int new_chunk_idx = engine->entities.chunk_count++;
    engine->entities.chunks[new_chunk_idx] = new EntityChunk(
        type_id,
        ENTITY_CHUNK_SIZE,
        config->extra_data_size
    );

    // Add to the linked list of chunks for this type
    if (prev_chunk_idx != -1) {
        engine->entities.chunks[prev_chunk_idx]->next_chunk_of_type = new_chunk_idx;
    }
    else {
        config->first_chunk_idx = new_chunk_idx;
    }

    config->instance_count += ENTITY_CHUNK_SIZE; // Account for max possible entities

    return new_chunk_idx;
}

// Add entity with a specific type
int engine_add_entity_with_type(Engine* engine, int type_id, float x, float y,
    int width, int height, int texture_id, int layer) {
    PROFILE_FUNCTION();

    // Find or create appropriate chunk
    int chunk_idx = find_or_create_chunk_for_type(engine, type_id);
    if (chunk_idx == -1) return -1; // Type not registered

    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    // Local index in the chunk
    int local_idx = chunk->count++;
    int entity_idx = chunk_idx * ENTITY_CHUNK_SIZE + local_idx;

    // Ensure total_count is updated
    if (entity_idx >= engine->entities.total_count) {
        engine->entities.total_count = entity_idx + 1;
    }

    // Initialize entity data
    chunk->x[local_idx] = x;
    chunk->y[local_idx] = y;
    chunk->width[local_idx] = width;
    chunk->height[local_idx] = height;
    chunk->texture_id[local_idx] = texture_id;
    chunk->layer[local_idx] = layer;

    // Precompute right and bottom for faster culling
    chunk->right[local_idx] = x + width;
    chunk->bottom[local_idx] = y + height;

    // Initialize hierarchy properties to default values
    chunk->parent_id[local_idx] = -1;
    chunk->first_child_id[local_idx] = -1;
    chunk->next_sibling_id[local_idx] = -1;
    chunk->local_x[local_idx] = x;
    chunk->local_y[local_idx] = y;

    // Set active by default and add to active indices
    chunk->active[local_idx] = true;
    chunk->active_indices[chunk->active_count++] = local_idx;

    // Set visible status based on grid loading
    int grid_x = (int)(x / engine->grid_cell_size);
    int grid_y = (int)(y / engine->grid_cell_size);

    if (grid_x >= 0 && grid_x < engine->grid_width &&
        grid_y >= 0 && grid_y < engine->grid_height) {
        chunk->visible[local_idx] = engine->grid_loaded[grid_x][grid_y];
    }
    else {
        chunk->visible[local_idx] = false;
    }

    // Add to spatial grid if visible
    if (chunk->visible[local_idx]) {
        spatial_grid_add(&engine->grid, entity_idx, x, y, &engine->entities);
    }

    return entity_idx;
}

// Get type-specific data for an entity
void* engine_get_entity_type_data(Engine* engine, int entity_idx) {
    PROFILE_FUNCTION();

    if (!engine->entities.isValidEntity(entity_idx)) {
        return nullptr;
    }

    // Get chunk and local index
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);
    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    // If chunk has no type-specific data, return nullptr
    if (!chunk->type_data) {
        return nullptr;
    }

    // Find the type config (O(1) lookup with type_id_to_index map)
    auto it = engine->type_id_to_index.find(chunk->type_id);
    if (it == engine->type_id_to_index.end()) {
        return nullptr; // Type not registered
    }

    int type_index = it->second;
    size_t extra_data_size = engine->entity_types[type_index].extra_data_size;

    if (extra_data_size == 0) {
        return nullptr;
    }

    // Calculate the pointer to this entity's extra data
    uint8_t* typed_data = static_cast<uint8_t*>(chunk->type_data);
    return typed_data + (local_idx * extra_data_size);
}

void engine_update_entity_types(Engine* engine, float delta_time) {
    PROFILE_FUNCTION();

    Uint64 start_time = SDL_GetTicks();
    int total_active_entities = 0;

    // Using the type-based chunk linking for much faster updates
    for (int type_idx = 0; type_idx < engine->entity_type_count; type_idx++) {
        EntityTypeConfig* config = &engine->entity_types[type_idx];

        // Skip if no update function or no chunks
        if (!config->update_func || config->first_chunk_idx == -1) {
            continue;
        }

        Uint64 type_start_time = SDL_GetTicks();
        int type_active_count = 0;

        // Process all chunks of this type without searching
        int chunk_idx = config->first_chunk_idx;
        while (chunk_idx != -1) {
            EntityChunk* chunk = engine->entities.chunks[chunk_idx];

            // Update entity type (can use the optimized active_indices array)
            if (chunk->active_count > 0) {
                // Store original positions for entities that might move
                const int BATCH_SIZE = 128;
                float original_x[BATCH_SIZE];
                float original_y[BATCH_SIZE];
                int active_to_chunk_map[BATCH_SIZE];

                // Process in batches for better cache usage
                for (int batch_start = 0; batch_start < chunk->active_count; batch_start += BATCH_SIZE) {
                    int batch_size = std::min(BATCH_SIZE, chunk->active_count - batch_start);

                    // Collect original positions
                    for (int i = 0; i < batch_size; i++) {
                        int local_idx = chunk->active_indices[batch_start + i];
                        original_x[i] = chunk->x[local_idx];
                        original_y[i] = chunk->y[local_idx];
                        active_to_chunk_map[i] = local_idx;
                    }

                    // Call the update function
                    config->update_func(chunk, chunk->count, delta_time);

                    // Update dependent values for entities that moved
                    for (int i = 0; i < batch_size; i++) {
                        int local_idx = active_to_chunk_map[i];
                        float new_x = chunk->x[local_idx];
                        float new_y = chunk->y[local_idx];

                        // Check if position changed
                        if (new_x != original_x[i] || new_y != original_y[i]) {
                            // Update precomputed bounds
                            chunk->right[local_idx] = new_x + chunk->width[local_idx];
                            chunk->bottom[local_idx] = new_y + chunk->height[local_idx];

                            // Mark entity transform as dirty
                            mark_entity_transform_dirty(chunk, local_idx);

                            // Update hierarchy relationships if needed
                            if (chunk->parent_id[local_idx] != -1) {
                                int parent_id = chunk->parent_id[local_idx];
                                int parent_chunk_idx, parent_local_idx;
                                engine->entities.getChunkIndices(parent_id, &parent_chunk_idx, &parent_local_idx);
                                EntityChunk* parent_chunk = engine->entities.chunks[parent_chunk_idx];

                                chunk->local_x[local_idx] = new_x - parent_chunk->x[parent_local_idx];
                                chunk->local_y[local_idx] = new_y - parent_chunk->y[parent_local_idx];
                            }
                            else {
                                chunk->local_x[local_idx] = new_x;
                                chunk->local_y[local_idx] = new_y;
                            }
                        }
                    }
                }

                type_active_count += chunk->active_count;
            }

            // Move to next chunk of the same type
            chunk_idx = chunk->next_chunk_of_type;
        }

        // Update stats
        config->last_update_time = (SDL_GetTicks() - type_start_time);
        total_active_entities += type_active_count;
    }

    // Update engine stats
    engine->active_entity_count = total_active_entities;
    engine->update_time = (SDL_GetTicks() - start_time);
}

void engine_set_entity_visible(Engine* engine, int entity_idx, bool visible) {
    PROFILE_FUNCTION();

    if (!engine->entities.isValidEntity(entity_idx)) {
        return;
    }

    // Get chunk and local index
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(entity_idx, &chunk_idx, &local_idx);

    EntityChunk* chunk = engine->entities.chunks[chunk_idx];

    // Only set visible if entity is active
    if (visible && !chunk->active[local_idx]) {
        return; // Can't be visible if not active
    }

    chunk->visible[local_idx] = visible;
}