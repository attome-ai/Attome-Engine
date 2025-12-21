#ifndef HIGH_PERFORMANCE_ENGINE_H
#define HIGH_PERFORMANCE_ENGINE_H

// SIMD width detection
#if defined(__AVX512F__)
#define SIMD_WIDTH 16
#define USE_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define SIMD_WIDTH 8
#define USE_AVX2 1
#include <immintrin.h>
#elif defined(__AVX__)
#define SIMD_WIDTH 8
#define USE_AVX 1
#include <immintrin.h>
#elif defined(__SSE__)
#define SIMD_WIDTH 4
#define USE_SSE 1
#include <xmmintrin.h>
#else
#define SIMD_WIDTH 1
#define USE_SCALAR 1
#endif

// Memory and cache alignment
#define CACHE_LINE_SIZE 64
#define PAGE_SIZE (4096) 
#define HUGE_PAGE_SIZE (2 * 1024 * 1024) // 2MB huge pages
#define ALIGNED_SIZE(size) (((size) + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1))
#define ALIGN_TO_CACHE alignas(CACHE_LINE_SIZE)
#define ALIGN_TO_PAGE alignas(PAGE_SIZE)
#define ALIGN_TO_HUGE_PAGE alignas(HUGE_PAGE_SIZE)

// Performance-optimized chunk sizes
#define ENTITY_CHUNK_SIZE 16384         // Increased for better memory locality
#define ENTITY_BATCH_SIZE 512           // Process in larger cache-friendly batches
#define SPATIAL_CELL_CAPACITY 256       // More entities per spatial cell
#define VERTEX_BATCH_SIZE 16384         // Larger vertex batches
#define INSTANCE_BATCH_SIZE 2048        // Larger instance batches
#define MAX_HIERARCHY_DEPTH 32          // Maximum hierarchy depth
#define MAX_ENTITY_TYPES 131072         // 128K entity types
#define MAX_PHYSICS_LAYERS 32           // Physics layers
#define VERTEX_CACHE_SIZE 1024          // Increased vertex cache
#define MAX_COMPONENTS 256              // Maximum component types
#define MORTON_GRID_SIZE 256            // Larger spatial grid
#define MAX_RENDER_BATCHES 65536        // Maximum render batches

// Include common libraries
#include <SDL3/SDL.h>
#include <SDL3/SDL_gpu.h>
#include <SDL3_shadercross/SDL_shadercross.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <thread>
#include <atomic>

#include <mutex>
#include <functional>
#include <array>
#include <bitset>
#include <unordered_map>
#include <type_traits>
#include <span>
#include <deque>
#include <condition_variable>
#include <math.h>
#include <fstream>
#include <filesystem>
#include <string>

// Forward declarations
struct Engine;
struct EntityType;
struct TransformData;
struct RenderBatch;
struct HierarchyData;
struct PhysicsData;
struct RenderData;
class TaskSystem;
struct MemoryArena;
struct FrameArenaAllocator;
struct MortonGrid;
class ShaderCompiler;
struct GPUMemoryManager;
struct GPUBufferPools;
struct AABB;
struct Camera;
struct EntityUpdateJob;
struct EntityUpdateContext;
struct GPUResources;
struct GPURenderer;
struct RenderCommand;

// -----------------------------------------------
// Memory Management System - Optimized for performance
// -----------------------------------------------
// Push constant data for fast uniform updates
struct ALIGN_TO_CACHE PushConstantData {
    static constexpr uint32_t MAX_SIZE = 128; // Most GPUs support 128-256 bytes of push constants
    uint8_t data[MAX_SIZE];
    uint32_t size;

    PushConstantData() : size(0) {
        memset(data, 0, MAX_SIZE);
    }

    template<typename T>
    void setData(const T& value) {
        static_assert(sizeof(T) <= MAX_SIZE, "Push constant data exceeds maximum size");
        size = sizeof(T);
        memcpy(data, &value, sizeof(T));
    }

    template<typename T>
    T* getData() {
        return reinterpret_cast<T*>(data);
    }
};

// Fixed-size memory arena with cache alignment
struct ALIGN_TO_CACHE MemoryArena {
    uint8_t* memory;
    size_t capacity;
    size_t used;
    size_t peak_usage;
    bool owns_memory;

    MemoryArena(size_t size = HUGE_PAGE_SIZE);
    MemoryArena(void* external_memory, size_t size);
    ~MemoryArena();

    void* allocate(size_t size, size_t alignment = CACHE_LINE_SIZE);
    void reset();
    size_t getUsed() const { return used; }
    size_t getPeakUsage() const { return peak_usage; }
    size_t getAvailable() const { return capacity - used; }
};

// Frame arena for temporary allocations
struct ALIGN_TO_CACHE FrameArenaAllocator {
    MemoryArena** arenas;
    uint32_t arena_count;
    uint32_t current_arena;
    size_t total_capacity;
    std::atomic<size_t> total_used;

    FrameArenaAllocator(uint32_t arena_count = 2, size_t arena_size = 16 * 1024 * 1024);
    ~FrameArenaAllocator();

    void* allocate(size_t size, size_t alignment = CACHE_LINE_SIZE);
    void advanceFrame();
    void resetAll();
};

// Specialized pool allocator for fixed-size blocks
template<typename T, size_t BlockSize = 1024>
struct ALIGN_TO_CACHE TypedMemoryPool {
    struct ALIGN_TO_CACHE Block {
        uint8_t data[BlockSize * sizeof(T)];
        uint32_t free_list[BlockSize];
        uint32_t free_count;
        Block* next;
    };

    Block* blocks;
    uint32_t total_capacity;
    uint32_t total_allocated;

    TypedMemoryPool();
    ~TypedMemoryPool();

    T* allocate();
    void deallocate(T* ptr);
    void reset();
};

// -----------------------------------------------
// Low-Level SIMD Operations
// -----------------------------------------------

// SIMD Vector3 for SoA operations
struct ALIGN_TO_CACHE SimdVec3 {
#if defined(USE_AVX512)
    __m512 x, y, z;
#elif defined(USE_AVX) || defined(USE_AVX2)
    __m256 x, y, z;
#elif defined(USE_SSE)
    __m128 x, y, z;
#else
    float x[SIMD_WIDTH], y[SIMD_WIDTH], z[SIMD_WIDTH];
#endif

    // Load and store operations
    static SimdVec3 load(const float* x_ptr, const float* y_ptr, const float* z_ptr);
    void store(float* x_ptr, float* y_ptr, float* z_ptr) const;

    // Vector operations
    SimdVec3 add(const SimdVec3& other) const;
    SimdVec3 sub(const SimdVec3& other) const;
    SimdVec3 mul(const SimdVec3& other) const;
    SimdVec3 mul(float scalar) const;
    SimdVec3 cross(const SimdVec3& other) const;
};

// SIMD Quaternion for SoA operations
struct ALIGN_TO_CACHE SimdQuat {
#if defined(USE_AVX512)
    __m512 x, y, z, w;
#elif defined(USE_AVX) || defined(USE_AVX2)
    __m256 x, y, z, w;
#elif defined(USE_SSE)
    __m128 x, y, z, w;
#else
    float x[SIMD_WIDTH], y[SIMD_WIDTH], z[SIMD_WIDTH], w[SIMD_WIDTH];
#endif

    // Load and store operations
    static SimdQuat load(const float* x_ptr, const float* y_ptr, const float* z_ptr, const float* w_ptr);
    void store(float* x_ptr, float* y_ptr, float* z_ptr, float* w_ptr) const;

    // Quaternion operations
    SimdQuat mul(const SimdQuat& other) const;
    SimdVec3 rotate(const SimdVec3& v) const;
};

// SIMD Matrix4x4 for transform operations
struct ALIGN_TO_CACHE SimdMat4 {
#if defined(USE_AVX512)
    __m512 rows[4];
#elif defined(USE_AVX) || defined(USE_AVX2)
    __m256 rows[4][2]; // Two registers per row for 8 float elements
#elif defined(USE_SSE)
    __m128 rows[4][4]; // Four registers per row for 4 float elements
#else
    float rows[4][4][SIMD_WIDTH]; // Array per SIMD lane
#endif

    // Matrix operations
    static SimdMat4 load(const glm::mat4* matrices, uint32_t count);
    void store(glm::mat4* matrices, uint32_t count) const;
    SimdMat4 mul(const SimdMat4& other) const;

    // Transform operations
    static SimdMat4 createTransform(const SimdVec3& position, const SimdQuat& rotation, const SimdVec3& scale);
    SimdVec3 transformPoint(const SimdVec3& point) const;
};

// SIMD AABB for collision detection
struct ALIGN_TO_CACHE SimdAABB {
#if defined(USE_AVX512)
    __m512 min_x, min_y, min_z;
    __m512 max_x, max_y, max_z;
#elif defined(USE_AVX) || defined(USE_AVX2)
    __m256 min_x, min_y, min_z;
    __m256 max_x, max_y, max_z;
#elif defined(USE_SSE)
    __m128 min_x, min_y, min_z;
    __m128 max_x, max_y, max_z;
#else
    float min_x[SIMD_WIDTH], min_y[SIMD_WIDTH], min_z[SIMD_WIDTH];
    float max_x[SIMD_WIDTH], max_y[SIMD_WIDTH], max_z[SIMD_WIDTH];
#endif

    // AABB operations
    static SimdAABB load(const AABB* boxes, uint32_t count);
    void store(AABB* boxes, uint32_t count) const;
    bool intersect(const SimdAABB& other) const;
    void transform(const SimdMat4& matrix);
};

// Standard AABB
struct ALIGN_TO_CACHE AABB {
    glm::vec3 min;
    glm::vec3 max;

    AABB() : min(0), max(0) {}
    AABB(const glm::vec3& min, const glm::vec3& max) : min(min), max(max) {}

    bool intersect(const AABB& other) const;
    AABB transform(const glm::mat4& matrix) const;
};

// -----------------------------------------------
// Entity Component System - Pure SoA Design
// -----------------------------------------------

// Component type identifiers
using ComponentTypeId = uint32_t;

// Component registry - maps component types to unique IDs
struct ComponentRegistry {
    static ComponentTypeId nextId;

    template<typename T>
    static ComponentTypeId getId() {
        static ComponentTypeId id = nextId++;
        return id;
    }
};

// Component specification
struct ComponentSpec {
    ComponentTypeId typeId;
    uint32_t size;
    uint32_t alignment;
    uint32_t offset;
    bool hot_data; // Frequently accessed

    template<typename T>
    static ComponentSpec create(bool hot = false) {
        ComponentSpec spec;
        spec.typeId = ComponentRegistry::getId<T>();
        spec.size = sizeof(T);
        spec.alignment = alignof(T);
        spec.offset = 0; // Will be set during layout calculation
        spec.hot_data = hot;
        return spec;
    }
};

// Component mask for archetypes
struct ALIGN_TO_CACHE ComponentMask {
    static constexpr uint32_t BITS_PER_BLOCK = 64;
    static constexpr uint32_t NUM_BLOCKS = (MAX_COMPONENTS + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;

    uint64_t blocks[NUM_BLOCKS] = {};

    void set(ComponentTypeId id);
    void clear(ComponentTypeId id);
    bool test(ComponentTypeId id) const;
    bool contains(const ComponentMask& other) const;
    uint64_t hash() const;
};

// Pre-computed entity type info
struct ALIGN_TO_CACHE EntityType {
    uint32_t id;                             // Entity type ID
    ComponentMask component_mask;            // Component mask
    std::vector<ComponentSpec> components;   // Component specifications

    // Memory layout information
    uint32_t instance_size;                  // Total size per instance
    uint32_t hot_data_size;                  // Size of hot data portion
    uint32_t component_count;                // Number of components

    // Function pointers for processing
    void (*update_func)(EntityUpdateContext* ctx, uint32_t start_idx, uint32_t count);

    // Precomputed hash for fast comparison
    uint64_t type_hash;

    EntityType();
    ~EntityType();

    void addComponent(const ComponentSpec& component);
    void calculateLayout();
    void setUpdateFunction(void (*func)(EntityUpdateContext*, uint32_t, uint32_t));
};

// Entity type registry - stores all entity types
struct ALIGN_TO_CACHE EntityTypeRegistry {
    EntityType* types;             // Array of entity types
    uint32_t type_count;           // Number of registered types
    uint32_t capacity;             // Maximum number of types

    EntityTypeRegistry(uint32_t max_types = MAX_ENTITY_TYPES);
    ~EntityTypeRegistry();

    uint32_t registerType(EntityType&& type);
    EntityType* getType(uint32_t id) const;
};

// Entity identifier
struct EntityId {
    uint32_t index;                // Global entity index
    uint32_t generation;           // Generation counter for validation

    bool operator==(const EntityId& other) const {
        return index == other.index && generation == other.generation;
    }

    bool operator!=(const EntityId& other) const {
        return !(*this == other);
    }

    bool isValid() const { return index != 0xFFFFFFFF; }

    static EntityId invalid() { return { 0xFFFFFFFF, 0 }; }
};

// Hierarchy data - SoA layout
struct ALIGN_TO_CACHE HierarchyData {
    uint32_t max_entities;

    // Core hierarchy relationships
    int32_t* parent_indices;               // Parent entity indices
    int32_t* first_child_indices;          // First child indices
    int32_t* next_sibling_indices;         // Next sibling indices
    int32_t* depths;                       // Hierarchy depths

    // Indices for depth-first traversal
    uint32_t* depth_first_indices;         // Depth-first sorted indices
    uint32_t* depth_type_sorted_indices;   // Depth and type sorted indices

    HierarchyData(uint32_t max_entities);
    ~HierarchyData();

    void setParent(uint32_t entity_idx, uint32_t parent_idx);
    int32_t getParent(uint32_t entity_idx) const;
    void updateDepths();
    void sortByDepthAndType(const uint32_t* entity_type_ids, uint32_t entity_count);
};

// Transform data - Pure SoA layout for SIMD processing
struct ALIGN_TO_CACHE TransformData {
    uint32_t max_entities;

    // Local transform components
    float* local_pos_x;
    float* local_pos_y;
    float* local_pos_z;
    float* local_rot_x;
    float* local_rot_y;
    float* local_rot_z;
    float* local_rot_w;
    float* local_scale_x;
    float* local_scale_y;
    float* local_scale_z;

    // World transform components
    float* world_pos_x;
    float* world_pos_y;
    float* world_pos_z;
    float* world_rot_x;
    float* world_rot_y;
    float* world_rot_z;
    float* world_rot_w;
    float* world_scale_x;
    float* world_scale_y;
    float* world_scale_z;

    // Cached matrices
    glm::mat4* local_matrices;
    glm::mat4* world_matrices;

    // Dirty flags for minimal updates
    bool* dirty_flags;

    TransformData(uint32_t max_entities);
    ~TransformData();

    void setLocalPosition(uint32_t entity_idx, const glm::vec3& position);
    void setLocalRotation(uint32_t entity_idx, const glm::quat& rotation);
    void setLocalScale(uint32_t entity_idx, const glm::vec3& scale);

    glm::vec3 getWorldPosition(uint32_t entity_idx) const;
    glm::quat getWorldRotation(uint32_t entity_idx) const;
    glm::vec3 getWorldScale(uint32_t entity_idx) const;

    // SIMD operations
    void updateLocalMatrices(uint32_t start_idx, uint32_t count);
    void updateWorldMatrices(const int32_t* parent_indices,
        const int32_t* dirty_indices,
        uint32_t dirty_count);
    void updateWorldMatricesHierarchical(const int32_t* parent_indices,
        const uint32_t* hierarchy_indices,
        uint32_t count,
        uint32_t start_depth,
        uint32_t max_depth);
};

// Entity storage - Holds entity data with SoA layout
struct ALIGN_TO_CACHE EntityStorage {
    uint32_t max_entities;
    uint32_t entity_count;

    // Entity identification
    EntityId* entity_ids;                  // Entity IDs
    uint32_t* type_ids;                    // Entity type IDs
    bool* active_flags;                     // Active flags

    // SoA component data - one pool per component type
    struct ComponentPool {
        uint8_t* data;                     // Component data
        uint32_t elem_size;                // Element size
        uint32_t stride;                   // Stride between elements
        uint32_t count;                    // Number of elements
        ComponentTypeId component_id;      // Component type ID
    };

    ComponentPool* component_pools;        // Component data pools
    uint32_t pool_count;                   // Number of component pools

    // Storage for hot data components
    uint8_t* hot_data;                     // Hot data storage
    uint32_t hot_data_size;                // Size of hot data per entity

    // Type information for fast lookups
    uint32_t* entities_by_type;            // Entities sorted by type
    uint32_t* type_start_indices;          // Start indices for each type
    uint32_t* type_counts;                 // Count of entities per type
    uint32_t type_count;                   // Number of different types

    // Memory for future expansion
    MemoryArena* memory_arena;             // Memory for all entity data

    EntityStorage(uint32_t max_entities, uint32_t max_component_types);
    ~EntityStorage();

    uint32_t createEntity(uint32_t type_id);
    void destroyEntity(uint32_t entity_idx);

    // Component data access
    template<typename T>
    T* getComponent(uint32_t entity_idx) const {
        ComponentTypeId comp_id = ComponentRegistry::getId<T>();
        for (uint32_t i = 0; i < pool_count; ++i) {
            if (component_pools[i].component_id == comp_id) {
                return reinterpret_cast<T*>(component_pools[i].data + entity_idx * component_pools[i].stride);
            }
        }
        return nullptr;
    }

    void* getComponentByTypeId(uint32_t entity_idx, ComponentTypeId comp_id) const;
    void sortEntitiesByType();
};

// Physics data - SoA layout for SIMD processing
struct ALIGN_TO_CACHE PhysicsData {
    uint32_t max_entities;

    // Motion data
    float* velocity_x;
    float* velocity_y;
    float* velocity_z;

    // Collision data
    AABB* bounds;
    uint32_t* collision_layers;
    uint32_t* collision_masks;

    // Collision results
    struct CollisionPair {
        uint32_t entity_a;
        uint32_t entity_b;
    };

    CollisionPair* collision_pairs;
    uint32_t collision_pair_count;
    uint32_t max_collision_pairs;

    // Physics settings
    uint32_t collision_layer_matrix[MAX_PHYSICS_LAYERS];

    PhysicsData(uint32_t max_entities, uint32_t max_collisions = 1024 * 1024);
    ~PhysicsData();

    void setVelocity(uint32_t entity_idx, const glm::vec3& velocity);
    void setCollisionLayer(uint32_t entity_idx, uint32_t layer);
    void setCollisionMask(uint32_t entity_idx, uint32_t mask);
    void setBounds(uint32_t entity_idx, const AABB& bounds);

    void updatePositions(TransformData* transforms, float delta_time,
        uint32_t start_idx, uint32_t count);
    void detectCollisions(const EntityId* entity_indices, uint32_t count);

    void detectCollisions(const uint32_t* entity_indices, uint32_t count);
    void clearCollisions();
};

// Render data - SoA layout for efficient rendering
struct ALIGN_TO_CACHE RenderData {
    uint32_t max_entities;

    // Rendering properties
    int32_t* mesh_ids;
    int32_t* material_ids;
    int32_t* shader_ids;
    uint8_t* visibility_flags;
    uint8_t* lod_levels;

    // Render batch assignments
    int32_t* batch_indices;

    RenderData(uint32_t max_entities);
    ~RenderData();

    void setMesh(uint32_t entity_idx, int32_t mesh_id);
    void setMaterial(uint32_t entity_idx, int32_t material_id);
    void setShader(uint32_t entity_idx, int32_t shader_id);
    void setVisibility(uint32_t entity_idx, bool visible);

    int32_t getMesh(uint32_t entity_idx) const;
    int32_t getMaterial(uint32_t entity_idx) const;
    int32_t getShader(uint32_t entity_idx) const;
    bool isVisible(uint32_t entity_idx) const;
};

// -----------------------------------------------
// Spatial Partition System - Morton Code Grid
// -----------------------------------------------

// Morton code grid for spatial queries
struct ALIGN_TO_CACHE MortonGrid {
    struct Cell {
        uint32_t* entity_indices;
        uint32_t count;
        uint32_t capacity;
    };

    uint64_t* morton_codes;            // Morton codes for each entity
    Cell* cells;                       // Grid cells
    uint32_t cell_count;               // Number of cells
    uint32_t max_entities;             // Maximum entities

    // Cell lookup optimization
    uint32_t* cell_start_indices;      // Starting index for each cell
    uint32_t* sorted_entity_indices;   // Entity indices sorted by morton code

    // World bounds
    glm::vec3 world_min;
    glm::vec3 world_max;
    float cell_size;

    MortonGrid(uint32_t max_entities, const glm::vec3& world_min,
        const glm::vec3& world_max, float cell_size);
    ~MortonGrid();

    void insertEntity(uint32_t entity_idx, const glm::vec3& position);
    void updateEntity(uint32_t entity_idx, const glm::vec3& position);
    void removeEntity(uint32_t entity_idx);
    uint64_t encodeMorton(const glm::vec3& position) const;
    uint32_t queryRadius(const glm::vec3& center, float radius,
    uint32_t* result_indices, uint32_t max_results);
    uint32_t queryBox(const AABB& box, uint32_t* result_indices, uint32_t max_results);

    uint64_t calculateMortonCode(const glm::vec3& position) const;
    void rebuild(const TransformData* transforms, uint32_t entity_count);
};


// -----------------------------------------------
// Rendering System
// -----------------------------------------------

// Pre-sorted render command
struct ALIGN_TO_CACHE RenderCommand {
    uint64_t sort_key;         // Sorting key (shader|mesh|material|transparent)
    uint32_t first_instance;   // First instance in batch
    uint32_t instance_count;   // Number of instances
    int32_t pipelineID;
    int32_t typeID; // entity ID
    int32_t shader_id;         // Shader ID

    RenderCommand() : sort_key(0), first_instance(0), instance_count(0),
        pipelineID(-1), typeID(-1), shader_id(-1) {}

    static uint64_t createSortKey(int32_t shader_id, int32_t mesh_id,
        int32_t material_id, bool transparent);
};

// Vertex layout
struct ALIGN_TO_CACHE Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texcoord;
    glm::vec3 tangent;
    glm::vec3 bitangent;
};

// Instance data layout for GPU
struct ALIGN_TO_CACHE InstanceData {
    glm::mat4 transform;
    glm::vec4 color;
};

// Render batch - optimized for SIMD updates and minimal state changes
struct ALIGN_TO_CACHE RenderBatch {
    // Render state
    int32_t shader_id;
    int32_t mesh_id;
    int32_t material_id;

    // Instance data
    InstanceData* instance_data;
    uint32_t instance_count;
    uint32_t instance_capacity;

    // Entity mapping
    uint32_t* entity_indices;

    // GPU Resources
    int32_t instance_buffer_id;
    bool buffer_dirty;

    // Sort key for efficient batching
    uint64_t sort_key;

    RenderBatch(uint32_t capacity = INSTANCE_BATCH_SIZE);
    ~RenderBatch();

    void addEntity(uint32_t entity_idx);
    void removeEntity(uint32_t entity_idx);
    void clear();

    void updateInstanceData(const TransformData* transforms,
        uint32_t start_idx, uint32_t count);

    void uploadToGPU(GPUResources* resources);
};

// Material data
struct ALIGN_TO_CACHE Material {
    int32_t diffuse_texture;
    int32_t normal_texture;
    int32_t specular_texture;
    int32_t sampler_id;
    glm::vec4 diffuse_color;
    glm::vec4 specular_color;
    float shininess;

    // Pre-cached GPU resources for faster binding
    SDL_GPUTexture* diffuse_texture_ptr;
    SDL_GPUTexture* normal_texture_ptr;
    SDL_GPUTexture* specular_texture_ptr;
    SDL_GPUSampler* sampler_ptr;

    Material();
};

// GPU Mesh data
struct ALIGN_TO_CACHE Mesh {
    int32_t vertex_buffer_id;
    int32_t index_buffer_id;
    uint32_t vertex_count;
    uint32_t index_count;
    AABB bounds;

    // Pre-cached GPU resources
    SDL_GPUBuffer* vertex_buffer_ptr;
    SDL_GPUBuffer* index_buffer_ptr;

    // LOD variants
    struct LODMesh {
        int32_t vertex_buffer_id;
        int32_t index_buffer_id;
        uint32_t vertex_count;
        uint32_t index_count;

        SDL_GPUBuffer* vertex_buffer_ptr;
        SDL_GPUBuffer* index_buffer_ptr;
    };

    LODMesh* lod_meshes;
    uint32_t lod_count;

    Mesh();
    ~Mesh();
};

// GPU resources manager
struct ALIGN_TO_CACHE GPUResources {
    SDL_GPUDevice* device;

    // Resource arrays for fast indexing
    SDL_GPUTexture** textures;
    SDL_GPUBuffer** vertex_buffers;
    SDL_GPUBuffer** index_buffers;
    SDL_GPUBuffer** storage_buffers;
    SDL_GPUBuffer** instance_buffers;
    SDL_GPUSampler** samplers;


    // Resource capacities
    uint32_t max_textures;
    uint32_t max_vertex_buffers;
    uint32_t max_index_buffers;
    uint32_t max_storage_buffers;
    uint32_t max_instance_buffers;
    uint32_t max_samplers;

    // Resource counts (actually used)
    uint32_t texture_count;
    uint32_t vertex_buffer_count;
    uint32_t index_buffer_count;
    uint32_t storage_buffer_count;
    uint32_t instance_buffer_count;
    uint32_t sampler_count;

    SDL_GPUGraphicsPipeline** pipelines;
    uint32_t max_pipeline;
    uint32_t pipeline_count;

    // Asset data
    Mesh* meshes;
    uint32_t max_meshes;
    uint32_t mesh_count;

    Material* materials;
    uint32_t max_materials;
    uint32_t material_count;



    GPUResources(SDL_GPUDevice* device, uint32_t max_textures, uint32_t max_buffers,
        uint32_t max_shaders, uint32_t max_meshes, uint32_t max_materials,
        uint32_t max_samplers);
    ~GPUResources();

    int32_t createTexture(SDL_Surface* surface);
    int32_t createVertexBuffer(const void* data, size_t size);
    int32_t createIndexBuffer(const void* data, size_t size);
    int32_t createStorageBuffer(const void* data, size_t size);
    int32_t createInstanceBuffer(const void* data, size_t size);
    int32_t createSampler(SDL_GPUFilter min_filter, SDL_GPUFilter mag_filter,
        SDL_GPUSamplerAddressMode address_mode);
    int32_t createMesh(const Vertex* vertices, uint32_t vertex_count,
        const uint32_t* indices, uint32_t index_count);
    int32_t createMaterial(int32_t diffuse_texture, int32_t normal_texture,
        int32_t specular_texture, int32_t sampler_id,
        const glm::vec4& diffuse_color, const glm::vec4& specular_color,
        float shininess);


    int32_t addGraphicsPipeline(SDL_GPUGraphicsPipeline* pipeline);


    void updateBuffer(int32_t buffer_id, const void* data, size_t size, size_t offset = 0);
    void destroyResources();

    // Pre-cache pointers for direct access
    void updatePointerCaches();

};
// GPU memory manager for buffer pooling
struct ALIGN_TO_CACHE GPUBufferPools {
    struct BufferPool {
        SDL_GPUBuffer** buffers;
        size_t* buffer_sizes;
        bool* in_use_flags;
        uint32_t count;
        uint32_t capacity;
        SDL_GPUBufferUsageFlags usage_flags;
    };

    BufferPool vertex_pool;
    BufferPool index_pool;
    BufferPool storage_pool;
    BufferPool instance_pool;

    SDL_GPUDevice* device;

    GPUBufferPools(SDL_GPUDevice* device, uint32_t max_buffers_per_pool);
    ~GPUBufferPools();

    int32_t getInstanceBuffer(size_t size, GPUResources* resources);

    void returnBuffer(SDL_GPUBufferUsageFlags type, int32_t buffer_id);
    void destroyPools();
};

// High-performance GPU renderer
struct ALIGN_TO_CACHE GPURenderer {

    int32_t global_uniform_buffer;

    // Global uniforms structure (matches shader Globals block)
    struct Uniforms {
        glm::mat4 view_matrix;
        glm::mat4 projection_matrix;
        glm::vec4 camera_position;
        float time;
        float delta_time;
        float padding[2]; // For alignment
    } uniforms;

    SDL_GPUDevice* device;
    SDL_Window* window;

    // Command buffer management
    SDL_GPUCommandBuffer** command_buffers;
    uint32_t command_buffer_count;
    uint32_t current_command_buffer;

    // Render pass management
    SDL_GPURenderPass* main_render_pass;
    SDL_GPUTexture* swapchain_texture;

    // Frame synchronization
    SDL_GPUFence** frame_fences;
    uint32_t frame_fence_count;
    uint32_t current_frame_fence;

    // State caching for minimal state changes
    int32_t current_shader_id;
    int32_t current_mesh_id;
    int32_t current_material_id;
    int32_t current_vertex_buffer_id;
    int32_t current_index_buffer_id;
    std::array<int32_t, 8> current_texture_ids;
    std::array<int32_t, 8> current_sampler_ids;

    // Render commands
    RenderCommand* render_commands;
    uint32_t render_command_count;
    uint32_t max_render_commands;


    GPURenderer(SDL_GPUDevice* device, SDL_Window* window, uint32_t command_buffer_count = 3);
    ~GPURenderer();

    bool initialize();
    void setupRenderPass();

    void beginFrame();
    void endFrame();

    void beginRenderPass();
    void endRenderPass();

    // Optimized rendering methods
    void addRenderCommand(const RenderCommand& command);
    void clearRenderCommands();
    void sortRenderCommands();
    void executeRenderCommands(const GPUResources* resources);

    // Minimal state change rendering
    void bindShaderIfChanged(int32_t shader_id, const GPUResources* resources);
    void bindMeshIfChanged(int32_t mesh_id, const GPUResources* resources);
    void bindMaterialIfChanged(int32_t material_id, const GPUResources* resources);


    // Release resources
    void shutdown();
};

// Camera for view/projection
struct ALIGN_TO_CACHE Camera {
    glm::vec3 position;
    glm::vec3 target;
    glm::vec3 up;
    float fov;
    float aspect_ratio;
    float near_plane;
    float far_plane;

    // Cached matrices
    glm::mat4 view_matrix;
    glm::mat4 projection_matrix;
    glm::mat4 view_projection_matrix;
    bool matrices_dirty;

    // Frustum data for culling
    glm::vec4 frustum_planes[6];
    bool frustum_dirty;

    Camera();

    void setPosition(const glm::vec3& pos);
    void setTarget(const glm::vec3& target);
    void setUpVector(const glm::vec3& up);
    void setProjection(float fov, float aspect, float near_z, float far_z);

     glm::mat4& getViewMatrix();
     glm::mat4& getProjectionMatrix();
     glm::mat4& getViewProjectionMatrix();

    void updateMatrices();
    void updateFrustumPlanes();

    bool isPointVisible(const glm::vec3& point) const;
    bool isSphereVisible(const glm::vec3& center, float radius) const;
    bool isAABBVisible(const AABB& aabb) const;

    // SIMD-optimized batch frustum culling
    void cullPointsBatch(const float* x, const float* y, const float* z,
        bool* results, uint32_t count) const;
    void cullAABBsBatch(const AABB* aabbs, bool* results, uint32_t count) const;
};

// -----------------------------------------------
// Task System for Parallel Processing
// -----------------------------------------------

// Work batch for parallel processing
struct ALIGN_TO_CACHE WorkBatch {
    uint32_t start_idx;
    uint32_t count;
    uint32_t thread_id;
};

// Task system for parallel processing
struct ALIGN_TO_CACHE TaskSystem {
    std::vector<std::thread> threads;
    std::atomic<bool> running;

    // Work queue
    struct TaskQueue {
        std::mutex mutex;
        std::condition_variable cv;
        std::vector<std::function<void(uint32_t)>> tasks;
        std::atomic<uint32_t> completed_count;
        std::atomic<uint32_t> total_count;
    };

    TaskQueue queue;
    std::atomic<uint32_t> idle_thread_count;

    // Thread-local data
    struct ThreadData {
        uint32_t thread_id;
        FrameArenaAllocator* frame_allocator;
    };

    static thread_local ThreadData thread_data;

    TaskSystem(uint32_t thread_count = 0);
    ~TaskSystem();

    void initialize(uint32_t thread_count);
    void shutdown();

    // Submit a task to be run on multiple threads
    void submitTask(std::function<void(uint32_t)> task, uint32_t count = 1);

    // Wait for all tasks to complete
    void waitForAll();

    // Worker thread function
    void workerThreadFunc(uint32_t thread_id);

    // Get current thread ID
    static uint32_t getCurrentThreadId();

    // Get thread-local frame allocator
    static FrameArenaAllocator* getThreadFrameAllocator();

    // Reset all frame allocators
    void resetThreadFrameAllocators();
};

// Entity update context for processing
struct ALIGN_TO_CACHE EntityUpdateContext {
    // Entity data references
    const EntityType* entity_type;
    EntityStorage* entities;
    TransformData* transforms;
    PhysicsData* physics;
    RenderData* render_data;

    // Time data
    float delta_time;
    float total_time;

    // Thread-specific data
    uint32_t thread_id;
    void* thread_data;

    EntityUpdateContext();

    // Accessor helpers
    template<typename T>
    T* getComponent(uint32_t entity_idx) const {
        return entities->getComponent<T>(entity_idx);
    }

    void setTransform(uint32_t entity_idx, const glm::vec3& position,
        const glm::quat& rotation, const glm::vec3& scale);

    glm::mat4 getWorldMatrix(uint32_t entity_idx) const;
};

// Entity update job for batch processing
struct ALIGN_TO_CACHE EntityUpdateJob {
    // Processor function type
    using ProcessorFunc = void (*)(EntityUpdateContext* ctx, uint32_t start_idx, uint32_t count);

    ProcessorFunc processor;   // Function to process entities
    EntityUpdateContext* ctx;  // Context data
    uint32_t start_idx;        // Start index
    uint32_t count;            // Number of entities

    EntityUpdateJob() : processor(nullptr), ctx(nullptr), start_idx(0), count(0) {}
    EntityUpdateJob(ProcessorFunc proc, EntityUpdateContext* context,
        uint32_t start, uint32_t cnt)
        : processor(proc), ctx(context), start_idx(start), count(cnt) {}

    void execute() {
        if (processor && ctx && count > 0) {
            processor(ctx, start_idx, count);
        }
    }
};

// -----------------------------------------------
// Main Engine
// -----------------------------------------------

// Main engine class - optimized for performance
struct ALIGN_TO_CACHE Engine {
    // Core SDL resources
    SDL_Window* window;
    SDL_GPUDevice* device;

    // Core engine data
    EntityTypeRegistry* entity_types;
    EntityStorage* entities;
    TransformData* transforms;
    PhysicsData* physics;
    RenderData* render_data;
    HierarchyData* hierarchy;

    // Spatial partitioning
    MortonGrid* spatial_grid;

    // Rendering systems
    Camera* camera;
    GPUResources* gpu_resources;
    GPURenderer* gpu_renderer;
    GPUBufferPools* buffer_pools;

    // Memory management
    FrameArenaAllocator* frame_allocator;

    // Parallel processing
    TaskSystem* task_system;

    // Render batches
    RenderBatch** render_batches;
    uint32_t render_batch_count;
    uint32_t max_render_batches;

    // Timing data
    float delta_time;
    float total_time;
    uint64_t last_frame_time;

    // Update context for entity processing
    EntityUpdateContext update_context;

    // Engine statistics
    struct EngineStats {
        uint32_t active_entities;
        uint32_t render_batches;
        uint32_t draw_calls;
        uint32_t triangles_rendered;
        float frame_ms;
        float update_ms;
        float render_ms;
        float physics_ms;
    } stats;

    // World bounds
    glm::vec3 world_min;
    glm::vec3 world_max;
    float world_cell_size;

    Engine(int window_width, int window_height,
        float world_size_x, float world_size_y, float world_size_z,
        float cell_size, uint32_t max_entities);
    ~Engine();

    // Core engine methods
    bool initialize();
    void update();
    void render();
    void shutdown();

    // Entity type management
    uint32_t createEntityType();
    void finalizeEntityTypes();

    // Entity instance management (initialization time only)
    uint32_t createEntityInstance(uint32_t type_id,
        const glm::vec3& position = glm::vec3(0.0f),
        const glm::quat& rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        const glm::vec3& scale = glm::vec3(1.0f));

    // Entity hierarchy (initialization time only)
    void setParent(uint32_t entity_idx, uint32_t parent_idx);
    void finalizeHierarchy();

    // Transform operations
    void setPosition(uint32_t entity_idx, const glm::vec3& position);
    void setRotation(uint32_t entity_idx, const glm::quat& rotation);
    void setScale(uint32_t entity_idx, const glm::vec3& scale);

    glm::vec3 getPosition(uint32_t entity_idx) const;
    glm::quat getRotation(uint32_t entity_idx) const;
    glm::vec3 getScale(uint32_t entity_idx) const;

    // Component access
    template<typename T>
    T* getComponent(uint32_t entity_idx) {
        return entities->getComponent<T>(entity_idx);
    }

    template<typename T>
    const T* getComponent(uint32_t entity_idx) const {
        return entities->getComponent<T>(entity_idx);
    }

    // Rendering operations
    void setEntityMesh(uint32_t entity_idx, int32_t mesh_id);
    void setEntityMaterial(uint32_t entity_idx, int32_t material_id);
    void setEntityShader(uint32_t entity_idx, int32_t shader_id);
    void setEntityVisibility(uint32_t entity_idx, bool visible);
    void setUniform(uint32_t entity_idx, void* data);
    // Physics operations
    void setEntityVelocity(uint32_t entity_idx, const glm::vec3& velocity);
    void setEntityCollisionLayer(uint32_t entity_idx, uint32_t layer);
    void setEntityCollisionMask(uint32_t entity_idx, uint32_t mask);
    void setEntityBounds(uint32_t entity_idx, const AABB& bounds);

    // Camera control
    void setCameraPosition(const glm::vec3& position);
    void setCameraTarget(const glm::vec3& target);
    void setCameraFov(float fov_degrees);

    // Resource management
    int32_t addTexture(SDL_Surface* surface);
    int32_t addMesh(const Vertex* vertices, uint32_t vertex_count,
        const uint32_t* indices, uint32_t index_count);
    int32_t addMaterial(int32_t diffuse_texture, int32_t normal_texture,
        int32_t specular_texture, int32_t sampler_id,
        const glm::vec4& diffuse_color, const glm::vec4& specular_color,
        float shininess);

    int32_t addGraphicsPipeline(SDL_GPUGraphicsPipeline* pipeline);




    // Statistics and profiling
    float getFrameTime() const { return delta_time; }
    float getFPS() const { return 1.0f / delta_time; }
    const EngineStats& getStats() const { return stats; }

private:
    // Update phases
    void updateTransforms();
    void updatePhysics();
    void updateEntityLogic();
    void updateVisibility();
    void updateBatches();


    void prepareRenderCommands();

    // Create optimized batches
    void createRenderBatches();
    void optimizeBatches();

    // GPU resource management
    void updateGPUResources();

    // Internal memory management
    void* frameAlloc(size_t size, size_t alignment = CACHE_LINE_SIZE);
    void resetFrameAllocator();
};

#endif // HIGH_PERFORMANCE_ENGINE_H