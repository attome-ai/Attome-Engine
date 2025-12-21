#ifndef ENGINE_H
#define ENGINE_H

#include <SDL3/SDL.h>

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
// Memory and cache alignment
#define CACHE_LINE_SIZE 64
#define PAGE_SIZE (4096) // 2MB huge pages
#define ALIGNED_SIZE(size) (((size) + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1))
#define ALIGN_TO_CACHE alignas(CACHE_LINE_SIZE)
#define ALIGN_TO_PAGE alignas(PAGE_SIZE)

// Chunk sizes optimized for cache lines and SIMD widths
#define ENTITY_CHUNK_SIZE 4096     // Increased for better batching 
#define ENTITY_BATCH_SIZE 256      // Process in cache-friendly batches
#define SIMD_BATCH_SIZE 16         // Match SIMD width for processing
#define SPATIAL_CELL_CAPACITY 64
#define VERTEX_BATCH_SIZE 4096
#define INSTANCE_BATCH_SIZE 256
#define MAX_HIERARCHY_DEPTH 32
#define MAX_ENTITY_TYPES 131072    // Support for 100k+ entity types
#define MAX_PHYSICS_LAYERS 32
#define VERTEX_CACHE_SIZE 256
#define MAX_COMPONENTS 256
#define MORTON_GRID_SIZE 64      // Size for Morton-ordered spatial grid

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

// Prefetch hints
#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_AVX) || defined(USE_SSE)
#define PREFETCH_READ _MM_HINT_T0
#define PREFETCH_WRITE _MM_HINT_ET0
#define PREFETCH(addr) _mm_prefetch((const char*)(addr), PREFETCH_READ)
#define PREFETCH_WRITE_HINT(addr) _mm_prefetch((const char*)(addr), PREFETCH_WRITE)
#else
#define PREFETCH(addr)
#define PREFETCH_WRITE_HINT(addr)
#endif

// Forward declarations for core components
struct Entity;
struct EntityStorage;
struct HierarchyStream;
struct EntityChunk;
struct Engine;
struct RenderBatch;
struct Camera;
struct Transform2D;
struct Transform3D;
struct SpatialGrid2D;
struct SpatialGrid3D;
struct MortonOrderedGrid;
struct MortonSystem;
struct OctreeNode;
struct LazyOctreeNode;
struct TextureAtlas;
struct MipmappedTextureAtlas;
struct GPUResources;
struct CollisionSystem;
struct EnhancedCollisionSystem;
struct LODSystem;
struct PhysicsLayer;
class TaskSystem;
class WorkStealingTaskSystem;
class FiberTaskSystem;
class FrameArenaAllocator;
class LockFreeTaskSystem;
class AssetManager;
class EntityLayoutOptimizer;

// Forward declare SIMD types
struct SimdFloat;
struct SimdInt;
struct SimdMask;
struct SimdAABB;
struct Mat4SIMD;

struct AABB;
// Memory Management Tools
//------------------------

// Memory block aligned to huge page size for better TLB efficiency
struct ALIGN_TO_PAGE HugePageMemoryBlock {
    uint8_t* data;
    size_t size;

    HugePageMemoryBlock(size_t size_bytes);
    ~HugePageMemoryBlock();
};

// Fixed size memory allocator for zero runtime allocations
template<typename T, size_t BlockSize = 64 * 1024> // 64KB blocks
class FixedAllocator {
private:
    struct ALIGN_TO_CACHE Block {
        uint8_t data[BlockSize];
        Block* next;
    };

    Block* current_block;
    size_t current_offset;
    std::vector<Block*> blocks;

public:
    FixedAllocator();
    ~FixedAllocator();

    T* allocate();
    void reset();
    size_t get_allocated_blocks() const;
};

// Scalable memory allocator for entities with huge page allocations
template<typename T>
class EntityAllocator {
    std::vector<T*> blocks;
    T* currentBlock;
    size_t currentIndex;
    static constexpr size_t ENTITIES_PER_BLOCK = 8192; // Increased for huge page alignment

public:
    EntityAllocator();
    ~EntityAllocator();

    T* Allocate();
    void Reset();
};

// Frame arena allocator for temporary allocations
class FrameArenaAllocator {
private:
    uint8_t* memory_block;
    size_t capacity;
    size_t current_offset;

public:
    FrameArenaAllocator(size_t size = 64 * 1024); // 64KB default
    ~FrameArenaAllocator();

    void* allocate(size_t size, size_t alignment = 16);
    void reset();
    size_t getUsedMemory() const;
};

// Optimized, cache-friendly buffer pool for temporary allocations
class ALIGN_TO_CACHE BufferPool {
public:
    class BufferHandle {
    private:
        uint8_t* buffer_;
        size_t size_;
        BufferPool* pool_;

    public:
        BufferHandle() noexcept;
        BufferHandle(uint8_t* buffer, size_t size, BufferPool* pool) noexcept;
        BufferHandle(BufferHandle&& other) noexcept;
        BufferHandle& operator=(BufferHandle&& other) noexcept;
        ~BufferHandle();

        void release();
        template<typename T> T* as() const noexcept;
        uint8_t* data() const noexcept;
        size_t size() const noexcept;
        bool valid() const noexcept;
        operator bool() const noexcept;
        void clear();
    };

private:
    struct BufferBucket;
    std::vector<BufferBucket> buckets;
    std::mutex mutex;
    static thread_local FrameArenaAllocator frame_allocator;

public:
    BufferPool();
    ~BufferPool();

    BufferHandle getBuffer(size_t requested_size);
    static FrameArenaAllocator* getFrameArena();
    static void resetFrameArenas();

private:
    void returnBuffer(uint8_t* buffer, size_t size);
    friend class BufferHandle;
};

// Memory-Optimized Data Structures
//---------------------------------

// Cache-friendly component mask
struct ComponentMask {
    static constexpr size_t BITS_PER_BLOCK = 64;
    static constexpr size_t NUM_BLOCKS = (MAX_COMPONENTS + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;

    std::array<uint64_t, NUM_BLOCKS> blocks;

    ComponentMask();
    void set(size_t index);
    void clear(size_t index);
    bool test(size_t index) const;
    bool containsAll(const ComponentMask& other) const;
    bool containsNone(const ComponentMask& other) const;
};

// Static entity storage - SOA layout with explicit cache alignment
struct ALIGN_TO_CACHE EntityStorage {
     struct EntityData {
        // Hot Data (64-byte cache line aligned)
        union {
            struct {
                glm::vec3 position[SIMD_WIDTH];
                glm::quat rotation[SIMD_WIDTH];
                glm::vec3 scale[SIMD_WIDTH];
                glm::mat4 world_matrix[SIMD_WIDTH];
            };
            struct EntityDataBlock {
                glm::vec3 positions[SIMD_WIDTH];
                glm::quat rotations[SIMD_WIDTH];
                glm::vec3 scales[SIMD_WIDTH];
                glm::mat4 matrices[SIMD_WIDTH];
            } blocks;
        };

        // Cold Data (separate cache lines)
        ALIGN_TO_CACHE struct ColdData {
            char name[64][SIMD_WIDTH];
            uint64_t creation_time[SIMD_WIDTH];
        }*cold;

        // Hierarchy Metadata (separate structure)
        ALIGN_TO_CACHE struct HierarchyData {
            int32_t parent_idx[SIMD_WIDTH];
            int32_t first_child[SIMD_WIDTH];
            int32_t next_sibling[SIMD_WIDTH];
            int32_t depth[SIMD_WIDTH];
        }*hierarchy;
    };

    // Pre-allocated memory blocks for 1M entities
    EntityData** type_data;
    size_t type_data_capacity;
    std::vector<std::bitset<SIMD_WIDTH>> active;

    EntityStorage(size_t max_entity_types, size_t max_entities);
    ~EntityStorage();

    size_t allocateEntityType(size_t count);
    void deallocateEntityType(size_t type_id);
};

// Flattened hierarchy representation
struct ALIGN_TO_CACHE HierarchyStream {
    ALIGN_TO_CACHE int32_t depth_batches[MAX_HIERARCHY_DEPTH];
    ALIGN_TO_CACHE int32_t depth_ranges[MAX_HIERARCHY_DEPTH][2]; // start/end indices
    ALIGN_TO_CACHE glm::mat4* parent_matrices; // Precomputed parent transforms
    ALIGN_TO_CACHE int32_t* update_groups; // Update groups for parallel processing
    int max_depth;

    HierarchyStream(size_t max_entities);
    ~HierarchyStream();

    void sortByDepth(const int32_t* depths, size_t entity_count);
    void generateUpdateGroups(size_t target_batch_size);
};

// SoA Entity component data optimized for SIMD
struct ALIGN_TO_CACHE EntityTypeHandler {
    uint32_t type_id;
    uint64_t component_mask;

    // Function pointers for type-specific behavior
    void (*update_func)(void* data, size_t count, float delta_time);
    void (*render_func)(void* data, size_t count, void* render_context);
    void (*destroy_func)(void* data, size_t count);

    // SIMD-optimized transform update
    void (*update_transforms)(void* data, const glm::mat4* parent_transforms, size_t count);

    EntityTypeHandler();
    virtual ~EntityTypeHandler();

    template<typename... Components>
    static EntityTypeHandler* Create();
};

// Memory-efficient component storage 
template<typename T, int N = SIMD_WIDTH>
struct ALIGN_TO_CACHE ComponentBlock {
    T data[N];
};

// AoSoA layout for transform components
template<int N = SIMD_WIDTH>
struct ALIGN_TO_CACHE TransformBlock {
    glm::vec3 position[N];
    glm::quat rotation[N];
    glm::vec3 scale[N];
    glm::mat4 local_matrix[N];
    glm::mat4 world_matrix[N];
};

// Morton Code Spatial Hasher for efficient spatial queries
struct ALIGN_TO_CACHE MortonSystem {
    static constexpr size_t MORTON_CAPACITY = 1 << 24; // 16M entries

    ALIGN_TO_CACHE uint64_t* codes;
    ALIGN_TO_CACHE int32_t* entity_indices;
    ALIGN_TO_CACHE int32_t* cell_ranges;

    MortonSystem(size_t max_entities);
    ~MortonSystem();

    void insertEntity(uint32_t entity_id, const glm::vec3& position, float radius);
    void updateEntity(uint32_t entity_id, const glm::vec3& position, float radius);
    void removeEntity(uint32_t entity_id);

    // SIMD-accelerated queries
    size_t queryRange(const glm::vec3& min, const glm::vec3& max, uint32_t* results, size_t max_results);

    static uint64_t encodeMorton(const glm::vec3& position, float grid_size);
};

// Task scheduling and parallelism
//---------------------------------

// Hierarchy-aware task partitioning system for depth-based processing
class ALIGN_TO_CACHE HierarchyAwareTaskPartitioner {
public:
    struct DepthBatch {
        int depth;
        int start_idx;
        int count;
    };

    std::vector<DepthBatch> depth_batches;

    HierarchyAwareTaskPartitioner();
    ~HierarchyAwareTaskPartitioner();

    void Initialize(int max_depth);
    void PartitionByDepth(const int* depths, int entity_count);
    void OptimizePartitioning(int target_batch_size);
    int GetOptimalTaskCount(int depth, int total_entities_at_depth, int target_per_task) const;
};

// Lock-free task system for massively parallel workloads
class ALIGN_TO_CACHE LockFreeTaskSystem {
public:
    // Define Task structure
    struct Task {
        std::function<void()> func;
        std::atomic<int> dependencies;
        std::vector<Task*> dependents;
        Task() : dependencies(0) {}  // Default constructor

        Task(std::function<void()> f) : func(f), dependencies(0) {}
    };


    struct ALIGN_TO_CACHE TaskQueue {
        std::atomic<uint32_t> head;
        std::atomic<uint32_t> tail;
        Task** buffer;

        TaskQueue();
        ~TaskQueue();

        // Delete copy constructor and copy assignment
        TaskQueue(const TaskQueue&) = delete;
        TaskQueue& operator=(const TaskQueue&) = delete;

        // Add move constructor and move assignment
        TaskQueue(TaskQueue&& other) noexcept
            : buffer(other.buffer) {
            head.store(other.head.load(std::memory_order_relaxed));
            tail.store(other.tail.load(std::memory_order_relaxed));
            other.buffer = nullptr;
        }

        TaskQueue& operator=(TaskQueue&& other) noexcept {
            if (this != &other) {
                // Clean up existing resources if needed
                delete[] buffer;  // Assuming buffer is allocated with new[]

                // Move resources
                buffer = other.buffer;
                head.store(other.head.load(std::memory_order_relaxed));
                tail.store(other.tail.load(std::memory_order_relaxed));

                // Reset other
                other.buffer = nullptr;
            }
            return *this;
        }

        Task* Pop();
        bool Push(Task* task);
    };

    TaskQueue globalQueue;
    std::vector<TaskQueue> threadQueues;
    std::atomic<bool> running;
    std::vector<std::thread> workers;
    std::atomic<int> activeThreadCount;
    FixedAllocator<Task> taskAllocator;

    LockFreeTaskSystem(size_t threadCount = 0);
    ~LockFreeTaskSystem();

    Task* CreateTask(std::function<void()> func);
    void AddDependency(Task* dependent, Task* dependency);
    void ScheduleTask(Task* task);
    void ScheduleTaskForThread(Task* task, int threadId);
    Task* StealTask(int thiefThreadId);
    void WaitAll();

    // Hierarchy-aware batch scheduling
    void ScheduleHierarchicalBatches(
        const HierarchyAwareTaskPartitioner& partitioner,
        std::function<void(int depth, int start, int count)> processFunc);

protected:
    void WorkerMain(int threadId);
};

// Work-stealing scheduler
// Work-stealing scheduler
class ALIGN_TO_CACHE WorkStealingTaskSystem {
public:
    struct Task {
        std::function<void()> func;
        std::atomic<int> dependencies;
        std::vector<Task*> dependents;

        Task(); // Default constructor
        Task(std::function<void()> f);
    };

private:
    struct ALIGN_TO_CACHE ThreadLocalQueue {
        std::vector<Task*> local_tasks;
        std::mutex queue_mutex;

        // Add these lines to explicitly delete copy operations
        ThreadLocalQueue() = default;
        ThreadLocalQueue(const ThreadLocalQueue&) = delete;
        ThreadLocalQueue& operator=(const ThreadLocalQueue&) = delete;
        // Allow move operations
        ThreadLocalQueue(ThreadLocalQueue&&) noexcept = default;
        ThreadLocalQueue& operator=(ThreadLocalQueue&&) noexcept = default;
    };

    std::vector<std::unique_ptr<ThreadLocalQueue>> thread_queues;
    std::atomic<int> active_thread_count;
    std::atomic<bool> running;
    std::vector<std::thread> workers;
    std::atomic<int> active_tasks;
    FixedAllocator<Task> task_allocator;
    std::mutex task_mutex;
    std::condition_variable task_cv;

public:
    WorkStealingTaskSystem(size_t thread_count = 0);
    ~WorkStealingTaskSystem();

    Task* createTask(std::function<void()> func);
    void addDependency(Task* dependent, Task* dependency);
    void scheduleTask(Task* task);
    void scheduleTaskForThread(Task* task, int thread_id);
    Task* stealTask(int thief_thread_id);
    void waitAll();

    // Hierarchy-aware task scheduling
    void scheduleHierarchyAwareTasks(
        const HierarchyAwareTaskPartitioner& partitioner,
        std::function<void(int depth, int start, int count)> processFunc);

protected:
    void worker_main(int thread_id);
};
// Fiber-based tasks for complex dependencies
class ALIGN_TO_CACHE FiberTaskSystem {
public:
    struct FiberTask {
        void* fiber;
        std::function<void()> func;
        bool completed;
        FiberTask* parent;
    };

private:
    void* main_fiber;
    std::vector<void*> fiber_pool;
    std::vector<FiberTask*> ready_tasks;
    std::mutex task_mutex;
    std::atomic<bool> running;

public:
    FiberTaskSystem(size_t fiber_count = 16);
    ~FiberTaskSystem();

    FiberTask* createFiberTask(std::function<void()> func);
    void scheduleFiberTask(FiberTask* task);
    void yieldToScheduler();
    void waitForTask(FiberTask* task);
    void runUntilComplete();
};

// SIMD optimization helpers
//--------------------------

// SIMD-optimized float operations for vectorized processing
struct ALIGN_TO_CACHE SimdFloat {
#if defined(USE_AVX512)
    __m512 v;
#elif defined(USE_AVX) || defined(USE_AVX2)
    __m256 v;
#elif defined(USE_SSE)
    __m128 v;
#else
    float v[SIMD_WIDTH];
#endif

    SimdFloat();
    explicit SimdFloat(float val);

    static SimdFloat load(const float* ptr);
    void store(float* ptr) const;

    SimdFloat operator+(const SimdFloat& rhs) const;
    SimdFloat operator-(const SimdFloat& rhs) const;
    SimdFloat operator*(const SimdFloat& rhs) const;
    SimdFloat operator/(const SimdFloat& rhs) const;

    // Additional operations
    SimdFloat sqrt() const;
    SimdFloat rsqrt() const;

    static SimdFloat zero();
};

// SIMD-optimized integer operations
struct ALIGN_TO_CACHE SimdInt {
#if defined(USE_AVX512)
    __m512i v;
#elif defined(USE_AVX) || defined(USE_AVX2)
    __m256i v;
#elif defined(USE_SSE)
    __m128i v;
#else
    int v[SIMD_WIDTH];
#endif

    SimdInt();
    explicit SimdInt(int val);

    static SimdInt load(const int* ptr);
    void store(int* ptr) const;

    SimdInt operator&(const SimdInt& rhs) const;
    SimdInt operator|(const SimdInt& rhs) const;
    SimdInt operator^(const SimdInt& rhs) const;
};

// Bitfield mask for SIMD operations
struct ALIGN_TO_CACHE SimdMask {
#if defined(USE_AVX512)
    __mmask16 mask;
#elif defined(USE_AVX2) || defined(USE_AVX)
    __m256i mask;
#elif defined(USE_SSE)
    __m128i mask;
#else
    uint32_t mask;
#endif

    SimdMask();
    SimdMask(uint32_t bit_mask);

    bool get(int index) const;
    void set(int index, bool value);
    SimdMask operator&(const SimdMask& other) const;
    SimdMask operator|(const SimdMask& other) const;
    SimdMask operator^(const SimdMask& other) const;
    SimdMask operator~() const;
    bool none() const;
    bool any() const;
    bool all() const;
};

// SIMD-optimized AABB for spatial queries
struct ALIGN_TO_CACHE SimdAABB {
    SimdFloat min_x, min_y, min_z;
    SimdFloat max_x, max_y, max_z;

    SimdAABB();

    static SimdAABB load(const AABB* boxes);
    void store(AABB* boxes) const;

    SimdMask overlaps(const SimdAABB& other) const;
    SimdMask contains(const SimdAABB& other) const;
};

// SIMD-optimized 4x4 matrix operations
struct ALIGN_TO_CACHE Mat4SIMD {
    SimdFloat m[16];

    Mat4SIMD();
    explicit Mat4SIMD(const glm::mat4& mat);

    static Mat4SIMD load(const glm::mat4* matrices);
    void store(glm::mat4* matrices) const;

    Mat4SIMD operator*(const Mat4SIMD& rhs) const;

    static Mat4SIMD identity();
    static Mat4SIMD translation(const SimdFloat& x, const SimdFloat& y, const SimdFloat& z);
    static Mat4SIMD rotation(const SimdFloat& qx, const SimdFloat& qy, const SimdFloat& qz, const SimdFloat& qw);
    static Mat4SIMD scale(const SimdFloat& x, const SimdFloat& y, const SimdFloat& z);
};

// SIMD-optimized matrix operations namespace
namespace SimdMatrixOps {
    // Matrix multiplication with SIMD
    void multiplyBatch(const glm::mat4* matrices_a, const glm::mat4* matrices_b,
        glm::mat4* results, int count);

    // Transpose matrix batch
    void transposeBatch(const glm::mat4* matrices, glm::mat4* results, int count);

    // Create transformation matrices
    void createTranslation(const glm::vec3* translations, glm::mat4* matrices, int count);
    void createRotation(const glm::quat* rotations, glm::mat4* matrices, int count);
    void createScale(const glm::vec3* scales, glm::mat4* matrices, int count);

    // Transform vectors in batches
    void transformPoints(const glm::mat4* matrices, const glm::vec3* points,
        glm::vec3* results, int count);

    // Batch transform updates by hierarchy depth
    void updateTransformsByDepth(const glm::mat4* local, const glm::mat4* parent,
        glm::mat4* world, const int* parent_indices,
        int count, int depth);

    // SIMD-optimized AABB calculations
    void calculateAABB(const glm::vec3* points, int point_count, AABB* result);

    // SIMD-optimized hierarchical transform update (processes by depth)
    void updateHierarchicalTransforms(glm::mat4* local_transforms, glm::mat4* world_transforms,
        const int* parent_indices, const int* depth_indices, int entity_count);

    // Batch quaternion operations
    void multiplyQuaternions(const glm::quat* q1, const glm::quat* q2,
        glm::quat* result, int count);

    // SIMD-optimized quaternion to matrix conversion
    void convertQuaternionsToMatrices(const glm::quat* quaternions, glm::mat4* matrices, int count);
}

// Bitmasking and branch reduction
namespace BitMaskOps {
    // Find first set bit
    int findFirstSetBit(uint64_t mask);

    // Count set bits
    int countSetBits(uint64_t mask);

    // Process entities using bitmasks to avoid branches
    void processEntityBitMasked(void* entity_data, uint64_t mask,
        void (*process_func)(void* data, int entity_idx), int base_idx);
}

// Core Entity System Components
//------------------------------

// Cache-aligned bounding box
struct ALIGN_TO_CACHE AABB {
    glm::vec3 min;
    glm::vec3 max;

    AABB();
    AABB(const glm::vec3& min, const glm::vec3& max);

    bool contains(const AABB& other) const;
    bool overlaps(const AABB& other) const;
    glm::vec3 center() const;
    glm::vec3 extents() const;
    float volume() const;
};

// Advanced entity chunk with hot/cold data splitting
struct ALIGN_TO_CACHE EntityChunk {
    // Core identification 
    alignas(CACHE_LINE_SIZE) int type_id;
    int count;
    int capacity;

    // Hot data (frequently accessed)
    ALIGN_TO_CACHE struct HotData {
        // Transform data
        glm::vec3* position;
        glm::vec3* velocity;
        glm::quat* rotation;
        glm::vec3* scale;
        glm::mat4* local_transform;
        glm::mat4* world_transform;

        // State flags
        bool* active;
        uint32_t* collision_mask;
        uint32_t* physics_layer;
        uint8_t* lod_level;
    } hot;

    // Cold data (infrequently accessed)
    ALIGN_TO_CACHE struct ColdData {
        // Entity metadata
        int* debug_id;
        char** names;
        float* creation_time;
        float* last_accessed_time;
        int* flags;
    } cold;

    // Hierarchy data
    ALIGN_TO_CACHE struct HierarchyData {
        int* parent_id;
        int* first_child_id;
        int* next_sibling_id;
        int* depth;
        glm::mat4* parent_transform_cache;
    } hierarchy;

    // Type-specific data
    void* type_data;
    size_t type_data_stride;

    // Entity bookkeeping
    int* entity_id;       // Global entity ID
    int* local_index;     // Index in this chunk

    // Activity mask for branch reduction
    uint64_t* activity_masks;

    // Cached transform blocks for SIMD processing
    TransformBlock<SIMD_WIDTH>* transform_blocks;
    int transform_block_count;

    // Constructor/destructor
    EntityChunk(int type_id, int capacity, size_t type_data_size);
    ~EntityChunk();

    // Core methods
    void updateTransforms();
    void updateTransformsHierarchical(const int* depths_sorted, const int* entity_indices, int count);
    void processBatch(int start_idx, int batch_size, float delta_time);
    void updateActiveEntitiesMasked(float delta_time);
    uint64_t generateActiveMask(int start_idx, int count) const;

    // SIMD-optimized methods
    void simdUpdatePositions(float delta_time, int start_idx, int count);
    void simdUpdateTransforms();

    // Hierarchy-aware processing methods
    void sortEntitiesByDepth(int* depths_out, int* indices_out) const;
    void prefetchEntityData(int entity_idx) const;

    // Cache optimization
    void optimizeDataLayout();
};

// Depth-sorted entity container for hierarchy processing
struct ALIGN_TO_CACHE HierarchyDepthContainer {
    int** entities_by_depth;
    int* counts_by_depth;
    int max_depth;

    HierarchyDepthContainer();
    ~HierarchyDepthContainer();

    void allocate(int total_entities, int max_depth_level);
    void sortEntitiesByDepth(const int* entity_ids, const int* depths, int count);

    template<typename Func>
    void processInDepthOrder(Func process_func);
};

// Morton-ordered spatial grid for cache-friendly spatial queries
struct ALIGN_TO_CACHE MortonOrderedGrid {
    struct Cell {
        uint64_t morton_code;
        int* entity_indices;
        int count;
        int capacity;
    };

    Cell* cells;
    int cell_count;
    float cell_size;

    // SoA for better cache performance in queries
    uint64_t* morton_codes;      // Morton codes for all cells
    int* cell_start_indices;     // Starting indices into entity array
    int* cell_entity_counts;     // Number of entities in each cell
    int* all_entities;           // All entities in morton order

    // Query results buffer
    int* query_buffer;
    int query_buffer_size;

    MortonOrderedGrid(int grid_size, float cell_size);
    ~MortonOrderedGrid();

    void insertEntity(int entity_idx, const glm::vec3& position, float radius);
    void removeEntity(int entity_idx, uint64_t morton_code);

    // Query entities in region, returning entities in cache-friendly order
    int queryRegion(const glm::vec3& min, const glm::vec3& max, int* result, int max_results);

    // Perform coherent spatial walk using morton ordering
    void walkOrdered(void (*process_func)(int entity_idx, void* user_data), void* user_data);

    // Optimize cell size based on entity distribution
    void optimizeCellSize();
};

// Cache-aware spatial partitioning for 3D
struct ALIGN_TO_CACHE SpatialGrid3D {
    struct Cell {
        int* entity_indices;  // Indices of entities in this cell
        int count;            // Number of entities in cell
        int capacity;         // Capacity of entity_indices array
    };

    Cell* cells;             // Grid cells
    int width, height, depth; // Grid dimensions
    int cell_count;          // Total number of cells
    float cell_size;         // Size of each grid cell

    // Query buffer for spatial queries (preallocated)
    int* query_buffer;
    int query_buffer_size;

    // Optimized data layout for improved cache coherence
    int* cell_active_counts;    // Number of active entities per cell
    int* entity_to_cell_map;    // Maps entity index to cell index for quick removal
    uint32_t* cell_occupancy;   // Bitmask to quickly check if cell has entities
    uint64_t* cell_morton_codes; // Morton codes for cache-friendly traversal

    SpatialGrid3D(int width, int height, int depth, float cell_size);
    ~SpatialGrid3D();

    void insertEntity(int entity_idx, const AABB& bounds);
    void removeEntity(int entity_idx, int cell_idx);
    int queryBox(const AABB& box, int* result, int max_results);
    void updateEntity(int entity_idx, int old_cell_idx, const AABB& bounds);
    int posToCell(const glm::vec3& pos) const;
    float calculateOptimalCellSize() const;
    void reorderCellsForCacheCoherence();
};

// Level-of-Detail system
struct ALIGN_TO_CACHE LODSystem {
    struct LODLevel {
        float distance_threshold;  // Distance at which this LOD activates
        int mesh_id;               // Mesh to use at this LOD level
    };

    struct EntityLODData {
        LODLevel* levels;         // LOD levels for this entity type
        int level_count;          // Number of LOD levels
    };

    EntityLODData* entity_lod_data;  // LOD data per entity type
    int entity_type_count;            // Number of entity types

    // Optimized LOD data for SIMD processing
    struct LODDistanceBlock {
        float distances[SIMD_WIDTH];
    };
    LODDistanceBlock* distance_blocks;

    LODSystem(int max_entity_types);
    ~LODSystem();

    void registerEntityTypeLOD(int type_id, const LODLevel* levels, int level_count);
    void updateLOD(EntityChunk** chunks, int chunk_count, const glm::vec3& camera_pos);
    void updateLODParallel(LockFreeTaskSystem& tasks, EntityChunk** chunks, int chunk_count, const glm::vec3& camera_pos);
    void updateLODWithSimd(EntityChunk* chunk, const glm::vec3& camera_pos);
    void updateLODAndVisibility(EntityChunk** chunks, int chunk_count, const glm::vec3& camera_pos, const void* view_frustum);
};

// Enhanced collision system with precomputed layer matrix
struct ALIGN_TO_CACHE EnhancedCollisionSystem {
    // Physics layers
    struct PhysicsLayer {
        uint32_t layer_mask;     // Bitmask for this layer (power of 2)
        uint32_t collides_with;  // Bitmask of layers this layer collides with
    };

    PhysicsLayer layers[MAX_PHYSICS_LAYERS];
    uint32_t collisionMatrix[MAX_PHYSICS_LAYERS];
    int layer_count;

    // Collision storage
    struct CollisionPair {
        int entity_a;
        int entity_b;
    };

    CollisionPair* collision_pairs;
    int collision_pair_count;
    int collision_pairs_capacity;

    // Parallel collision detection jobs
    struct ALIGN_TO_CACHE CollisionJob {
        int start_cell;
        int end_cell;
        CollisionPair* local_collision_buffer;
        int collision_count;
    };

    // SIMD-optimized collision checking
    struct ALIGN_TO_CACHE CollisionSIMDData {
        AABB bounds[SIMD_WIDTH];
        int entity_ids[SIMD_WIDTH];
        int physics_layers[SIMD_WIDTH];
    };
    CollisionSIMDData* collision_simd_data;

    EnhancedCollisionSystem();
    ~EnhancedCollisionSystem();

    void registerLayer(int layer_id, uint32_t collides_with);
    void precomputeCollisionMatrix();
    bool canCollide(int layerA, int layerB) const;

    void broadphase3DWithLayers(SpatialGrid3D* grid, EntityChunk** chunks, int chunk_count);
    bool simdCheckAABBOverlap(const AABB* boxes_a, const AABB* boxes_b, int count_a, int count_b, bool* results);
    void detectCollisionsParallel(SpatialGrid3D* grid, EntityChunk** chunks, int chunk_count, LockFreeTaskSystem& tasks);
    void mergeCollisionResults(CollisionJob* jobs, int job_count);
    void broadphaseMortonOrderedWithLayers(MortonOrderedGrid* grid, EntityChunk** chunks, int chunk_count);
    void detectCollisionsSIMD(EntityChunk** chunks, int chunk_count);
};

// Camera systems
struct Camera3D {
    glm::vec3 position;         // Camera position
    glm::vec3 target;           // Look target
    glm::vec3 up;               // Up vector
    float fov;                  // Field of view in radians
    float aspect_ratio;         // Aspect ratio
    float near_plane;           // Near clipping plane
    float far_plane;            // Far clipping plane

    Camera3D();

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::mat4 getViewProjectionMatrix() const;

    // Frustum planes for culling
    void extractFrustumPlanes(glm::vec4* planes) const;

    // SIMD-optimized frustum culling
    int cullAABBsSimd(const AABB* bounds, int count, bool* results) const;
};

// Rendering systems
//------------------

// Material system for 3D rendering
struct Material {
    int diffuse_texture;    // Diffuse texture ID
    int normal_texture;     // Normal map texture ID
    int specular_texture;   // Specular map texture ID
    glm::vec4 diffuse_color;  // Diffuse color
    glm::vec4 specular_color; // Specular color
    float shininess;        // Specular power

    Material();
};

// Mesh data for 3D rendering with LOD support
struct Mesh {
    uint32_t vao;            // Vertex Array Object
    uint32_t vbo;            // Vertex Buffer Object
    uint32_t ibo;            // Index Buffer Object
    int vertex_count;        // Number of vertices
    int index_count;         // Number of indices
    AABB bounds;             // Bounding box

    // LOD variants
    struct LODMesh {
        uint32_t vao;
        uint32_t vbo;
        uint32_t ibo;
        int vertex_count;
        int index_count;
    };

    LODMesh* lod_meshes;     // LOD mesh variants
    int lod_count;           // Number of LOD levels

    Mesh();
    ~Mesh();
};

// Cache-friendly vertex cache
struct ALIGN_TO_CACHE VertexCache {
    SDL_Vertex vertices[VERTEX_CACHE_SIZE];
    int count;

    VertexCache();

    void addVertex(const SDL_Vertex& vertex);
    void flush(SDL_Vertex* dest, int offset);
};

// Instance data for 3D instanced rendering
struct ALIGN_TO_CACHE InstanceData {
    glm::mat4 transforms[INSTANCE_BATCH_SIZE];
    glm::vec4 colors[INSTANCE_BATCH_SIZE];
    int instance_count;

    // Cache-friendly layout for GPU upload
    struct GPUInstanceBlock {
        glm::mat4 transforms[16]; // Block size aligned to typical GPU vertex attribute fetch
    };
    GPUInstanceBlock* gpu_blocks;

    InstanceData();
    ~InstanceData();

    void clear();
    void resize(int capacity);
    void upload(uint32_t buffer_id);
};

// Render batch for 3D models with instancing
struct ALIGN_TO_CACHE RenderBatch3D {
    int material_id;           // Material ID
    int shader_id;             // Shader ID
    int* entity_indices;       // Entity indices in this batch
    int count;                 // Number of entities
    int capacity;              // Capacity of entity_indices
    InstanceData instance_data; // Instance data for instanced rendering
    uint32_t instance_buffer;   // GPU buffer for instance data

    // Typed entity indices for faster access during update
    struct TypedEntityBlock {
        int entity_indices[SIMD_WIDTH];
        int chunk_indices[SIMD_WIDTH]; // Which chunk each entity belongs to
    };
    TypedEntityBlock* typed_blocks;
    int typed_block_count;

    RenderBatch3D();
    ~RenderBatch3D();

    void renderInstanced();
    void updateInstanceData(EntityChunk* chunk, int start_idx, int count);
    void updateInstanceDataSIMD(EntityChunk** chunks, const int* chunk_indices, const int* entity_indices, int count);
};

// GPU-Driven rendering for massive entity counts
struct ALIGN_TO_CACHE GPUDrivenRenderer {
    struct VisibleObjectsBuffer {
        uint32_t visibleCount;
        uint32_t* visibleIndices;
        uint32_t capacity;
    };

    uint32_t entitiesBuffer;      // SSBO for entity transforms
    uint32_t visibleSSBO;         // SSBO for visible instance indices
    uint32_t frustumCullShader;   // Compute shader for frustum culling
    uint32_t indirectDrawBuffer;  // Buffer for indirect draw commands

    // For hierarchical depth ordering on GPU
    uint32_t hierarchyDepthBuffer; // SSBO for entity hierarchy depths
    uint32_t depthSortedIndices;   // SSBO for indices sorted by depth

    GPUDrivenRenderer(int maxEntities);
    ~GPUDrivenRenderer();

    void UpdateEntityTransforms(EntityChunk** chunks, int chunkCount);
    void CullWithCompute(const Camera3D& camera);
    void RenderVisible(RenderBatch3D* batches, int batchCount);
    void UpdateHierarchyDepths(EntityChunk** chunks, int chunkCount);
    void RenderVisibleHierarchical(RenderBatch3D* batches, int batchCount);
};

// Asset management system
class AssetManager {
public:
    struct LoadTask {
        std::function<void()> Load;
        std::function<void()> UploadToGPU;
        std::string name;
        int assetId;
    };

private:
    struct ConcurrentQueue {
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<LoadTask> tasks;

        void Enqueue(LoadTask task);
        bool TryDequeue(LoadTask& result);
        bool Empty();
    };

    ConcurrentQueue loadQueue;
    std::atomic<bool> running;
    std::vector<std::thread> ioThreads;
    std::unordered_map<std::string, int> loadedAssets;
    std::mutex assetsMutex;

public:
    AssetManager(int numThreads = 2);
    ~AssetManager();

    int QueueTextureLoad(const std::string& filename);
    int QueueMeshLoad(const std::string& filename);
    int QueueShaderLoad(const std::string& vsFilename, const std::string& fsFilename);

    bool IsAssetLoaded(const std::string& name, int* assetId = nullptr);
    bool AreAllAssetsLoaded();
    void WaitForAll();

private:
    void IOThreadMain();
};

// Entity Layout Optimizer
class EntityLayoutOptimizer {
    struct AccessPattern {
        int entity_id;
        std::vector<int> accessed_with; // Other entities frequently accessed together
        int access_count;
    };

    std::vector<AccessPattern> access_patterns;
    std::vector<int> optimized_order;

public:
    EntityLayoutOptimizer();
    ~EntityLayoutOptimizer();

    void recordAccess(const std::vector<int>& entities_accessed_together);
    void analyzePatterns();
    void optimizeLayout(EntityChunk** chunks, int chunk_count);
    const std::vector<int>& getOptimizedOrder() const;

    void applyToSpatialGrid(SpatialGrid3D* grid);
    void applyToMortonGrid(MortonOrderedGrid* grid);
};

// Entity Manager
//-------------

// Main entity manager with static memory layout
struct ALIGN_TO_CACHE EntityManager {
    // Entity storage
    EntityStorage* storage;

    // Entity chunks 
    EntityChunk** chunks;       // Array of entity chunks
    int chunk_count;            // Number of chunks
    int chunk_capacity;         // Capacity of chunks array
    int total_entity_count;     // Total number of entities

    // Type system
    std::vector<EntityTypeHandler*> entity_type_handlers;

    // Hierarchy system
    HierarchyStream* hierarchy_stream;

    // Quick lookup tables
    int* entity_to_chunk;      // Maps entity index to chunk index
    int* entity_to_local;      // Maps entity index to local index in chunk

    // Depth-first entity ordering for cache coherence
    HierarchyDepthContainer depth_container;

    // Entity layout optimization
    EntityLayoutOptimizer layout_optimizer;

    EntityManager(int total_entity_count);
    ~EntityManager();

    int registerEntityType(EntityTypeHandler* type_handler);
    int createEntity(int type_id);
    void destroyEntity(int entity_idx);
    void getChunkIndices(int entity_idx, int* chunk_idx, int* local_idx) const;
    bool isValidEntity(int entity_idx) const;

    void setParent(int entity_idx, int parent_idx);
    void sortByHierarchyDepth();
    void updateEntityTransforms();
    void updateEntityTransformsHierarchical();

    template<typename Func>
    void processEntitiesInOptimalOrder(Func process_func);

    void* getEntityTypeData(int entity_idx);
    void setEntityActive(int entity_idx, bool active);

    glm::vec3 getEntityPosition(int entity_idx);
    void setEntityPosition(int entity_idx, const glm::vec3& position);
    void setEntityLocalPosition(int entity_idx, const glm::vec3& local_position);

    void prefetchEntityData(int entity_idx);
    void prefetchChunkData(EntityChunk* chunk, int start_idx, int count);
    void prefetchEntityTransform(int entity_idx);

    void optimizeMemoryLayout();
    void reorderEntitiesByType();
    void reorderEntitiesByHierarchy();

    template<typename Func>
    void processByType(int type_id, Func process_func, int batch_size = ENTITY_BATCH_SIZE);
};

// GPU Resource Management
//----------------------

// Cross-platform GPU resources
struct GPUResources {
    // Texture management
    SDL_Texture** textures;
    int texture_count;
    int texture_capacity;

    // Shader management
    struct Shader {
        uint32_t program;          // Shader program ID
        std::unordered_map<std::string, int> uniforms;  // Uniform locations
    };

    Shader* shaders;
    int shader_count;
    int shader_capacity;

    // Mesh management
    Mesh* meshes;
    int mesh_count;
    int mesh_capacity;

    // Material management
    Material* materials;
    int material_count;
    int material_capacity;

    // Render targets
    SDL_Texture** render_targets;
    int render_target_count;
    int render_target_capacity;

    GPUResources(int max_textures, int max_shaders, int max_meshes, int max_materials, int max_render_targets);
    ~GPUResources();

    int createTexture(SDL_Renderer * renderer, SDL_Surface* surface);
    int createTextureAtlas(SDL_Surface** surfaces, int count, int* width, int* height);
    int createMesh(const float* vertices, int vertex_count, const int* indices, int index_count);
    int createMaterial(int diffuse_texture, int normal_texture, int specular_texture, const glm::vec4& diffuse_color, const glm::vec4& specular_color, float shininess);
    int createShader(const char* vertex_source, const char* fragment_source);
    int createRenderTarget(int width, int height);
};

// Main Engine
//-----------

// Main engine struct
struct ALIGN_TO_CACHE Engine {
    // Core SDL resources
    SDL_Window* window;
    SDL_Renderer* renderer;

    // Entity system
    EntityManager entities;

    // Spatial partitioning
    SpatialGrid3D grid3d;
    MortonOrderedGrid* morton_grid;  // Cache-optimized spatial grid
    MortonSystem* morton_system;    // Morton-encoded spatial hasher

    // Camera system
    Camera3D camera3d;

    // Rendering systems
    RenderBatch3D* batches3d;
    int batch_count3d;
    GPUDrivenRenderer* gpuRenderer;

    // GPU resources
    GPUResources gpu_resources;

    // Time tracking
    uint64_t last_frame_time;
    float delta_time;
    float fps;

    // Task scheduling
    LockFreeTaskSystem task_system;
    HierarchyAwareTaskPartitioner hierarchy_task_partitioner;

    // Systems
    LODSystem lod_system;
    EnhancedCollisionSystem collision_system;
    AssetManager* asset_manager;

    // Memory management
    BufferPool buffer_pool;

    // Temp work buffers
    int* entity_buffer;
    int entity_buffer_size;

    // Engine settings
    glm::vec3 world_min;
    glm::vec3 world_max;
    float grid_cell_size;
    int thread_count;

    // Debug settings
    bool debug_mode;
    bool profile_memory;

    // Transform update batches for SIMD optimization
    struct ALIGN_TO_CACHE TransformBatch {
        glm::mat4 local[SIMD_WIDTH];
        glm::mat4 parent[SIMD_WIDTH];
        glm::mat4 world[SIMD_WIDTH];
        int count;
    };
    TransformBatch* transform_batches;
    int transform_batch_count;

    Engine(int window_width, int window_height, float world_size_x, float world_size_y, float world_size_z, float cell_size, int total_entities);
    ~Engine();

    void update();
    void render();

    // Optimized update methods
    void updateByHierarchyDepth();
    void updateEntityChunksBatched(EntityChunk** chunks, int chunk_count);
    void updateTransformsHierarchical();
    void updateCollisionsBatched();

    // Entity creation
    int addEntity(int type_id, const glm::vec3& position, const glm::vec3& scale, int mesh_id, int material_id);

    // Entity manipulation
    void setEntityPosition(int entity_id, const glm::vec3& position);
    void setEntityRotation(int entity_id, const glm::quat& rotation);
    void setEntityScale(int entity_id, const glm::vec3& scale);
    void setEntityActive(int entity_id, bool active);
    void setEntityMesh(int entity_id, int mesh_id);
    void setEntityMaterial(int entity_id, int material_id);

    // Hierarchy
    void setParent(int entity_id, int parent_id);
    int getParent(int entity_id);
    void setEntityLocalPosition(int entity_id, const glm::vec3& position);

    // Spatial queries
    int queryBox(const AABB& box, int* result_buffer, int max_results);
    int queryFrustum(const Camera3D& camera, int* result_buffer, int max_results);

    // Cache-optimized spatial queries
    int queryMortonRegion(const glm::vec3& min, const glm::vec3& max, int* result_buffer, int max_results);

    // Camera control
    void setCameraPosition(const glm::vec3& position);
    void setCameraTarget(const glm::vec3& target);
    void setCameraUp(const glm::vec3& up);
    void setCameraFov(float fov_degrees);

    // Resource management
    int addTexture(SDL_Renderer* renderer, SDL_Surface* surface);
    int addMesh(const float* vertices, int vertex_count, const int* indices, int index_count);
    int addMaterial(int diffuse_texture, int normal_texture, int specular_texture, const glm::vec4& diffuse_color, const glm::vec4& specular_color, float shininess);
    int addShader(const char* vertex_source, const char* fragment_source);

    // Physics and collision
    void registerPhysicsLayer(int layer_id, uint32_t collides_with);
    void setEntityPhysicsLayer(int entity_id, int layer_id);
    void setEntityVelocity(int entity_id, const glm::vec3& velocity);
    int queryCollisions(int entity_id, int* result_buffer, int max_results);

    // Level of Detail
    void registerEntityLOD(int type_id, const LODSystem::LODLevel* levels, int level_count);

    // Asynchronous resource loading
    int queueTextureLoad(const char* filename);
    int queueMeshLoad(const char* filename);
    int queueShaderLoad(const char* vs_filename, const char* fs_filename);
    bool areAllAssetsLoaded();
    void waitForAssetLoading();

    // Debug/profiling
    void setDebugMode(bool enabled);
    float getFrameTime() const;
    float getFps() const;
    void getMemoryStats(size_t* total_allocated, size_t* temp_allocated) const;

    // Memory optimization
    void optimizeMemoryLayout();
    void collectPerformanceStats();
};

#endif // ENGINE_H