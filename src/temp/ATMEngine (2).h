#ifndef ENGINE_H
#define ENGINE_H

#include <SDL3/SDL.h>
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

// Memory and cache alignment
#define CACHE_LINE_SIZE 64
#define ALIGNED_SIZE(size) (((size) + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1))
#define ALIGN_TO_CACHE alignas(CACHE_LINE_SIZE)

// Chunk sizes optimized for cache lines
#define ENTITY_CHUNK_SIZE 1024
#define SPATIAL_CELL_CAPACITY 64
#define VERTEX_BATCH_SIZE 4096
#define INSTANCE_BATCH_SIZE 256
#define MAX_HIERARCHY_DEPTH 32
#define MAX_ENTITY_TYPES 16384
#define MAX_PHYSICS_LAYERS 32
#define VERTEX_CACHE_SIZE 256

// SIMD width detection
#if defined(__AVX512F__)
#define SIMD_WIDTH 16
#define USE_AVX512 1
#elif defined(__AVX2__)
#define SIMD_WIDTH 8
#define USE_AVX2 1
#elif defined(__AVX__)
#define SIMD_WIDTH 8
#define USE_AVX 1
#elif defined(__SSE__)
#define SIMD_WIDTH 4
#define USE_SSE 1
#else
#define SIMD_WIDTH 1
#define USE_SCALAR 1
#endif

// Prefetch hints
#define PREFETCH_READ _MM_HINT_T0
#define PREFETCH_WRITE _MM_HINT_ET0

// Forward declarations for core components
struct Entity;
struct EntityChunk;
struct Engine;
struct RenderBatch;
struct Camera;
struct Transform2D;
struct Transform3D;
struct SpatialGrid2D;
struct SpatialGrid3D;
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

// Math types
union Vec2 {
    struct { float x, y; };
    float data[2];
};

union Vec3 {
    struct { float x, y, z; };
    float data[3];
};

union Vec4 {
    struct { float x, y, z, w; };
    float data[4];
};

struct Rect {
    float x, y, width, height;
};

struct AABB {
    Vec3 min;
    Vec3 max;
};

struct Quaternion {
    float x, y, z, w;
};

// 3D Matrix for hierarchies
struct ALIGN_TO_CACHE Mat4 {
    union {
        float m[16];
        float rows[4][4];
        struct {
            float m00, m01, m02, m03;
            float m10, m11, m12, m13;
            float m20, m21, m22, m23;
            float m30, m31, m32, m33;
        };
    };
};

// Bitfield mask for SIMD operations
struct SimdMask {
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
};

// Memory Management Tools
//------------------------

// Fixed size memory allocator for zero runtime allocations
template<typename T, size_t BlockSize = 64 * 1024> // Increased from 4KB to 64KB
class FixedAllocator {
private:
    struct Block {
        alignas(CACHE_LINE_SIZE) uint8_t data[BlockSize];
        Block* next;
    };

    Block* current_block;
    size_t current_offset;
    std::vector<Block*> blocks;

public:
    FixedAllocator() : current_block(nullptr), current_offset(BlockSize) {}

    ~FixedAllocator() {
        for (auto block : blocks) {
            delete block;
        }
    }

    T* allocate() {
        if (current_offset + sizeof(T) > BlockSize) {
            // Allocate new block
            Block* new_block = new Block();
            new_block->next = nullptr;
            if (current_block) {
                current_block->next = new_block;
            }
            blocks.push_back(new_block);
            current_block = new_block;
            current_offset = 0;
        }

        // Ensure proper alignment
        size_t alignment = alignof(T);
        current_offset = (current_offset + alignment - 1) & ~(alignment - 1);

        T* result = reinterpret_cast<T*>(current_block->data + current_offset);
        current_offset += sizeof(T);
        return result;
    }

    void reset() {
        current_block = blocks.empty() ? nullptr : blocks[0];
        current_offset = 0;
    }

    size_t get_allocated_blocks() const {
        return blocks.size();
    }
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
        BufferHandle() noexcept : buffer_(nullptr), size_(0), pool_(nullptr) {}

        BufferHandle(uint8_t* buffer, size_t size, BufferPool* pool) noexcept
            : buffer_(buffer), size_(size), pool_(pool) {}

        BufferHandle(BufferHandle&& other) noexcept
            : buffer_(other.buffer_), size_(other.size_), pool_(other.pool_) {
            other.buffer_ = nullptr;
            other.size_ = 0;
            other.pool_ = nullptr;
        }

        BufferHandle& operator=(BufferHandle&& other) noexcept {
            if (this != &other) {
                release();
                buffer_ = other.buffer_;
                size_ = other.size_;
                pool_ = other.pool_;
                other.buffer_ = nullptr;
                other.size_ = 0;
                other.pool_ = nullptr;
            }
            return *this;
        }

        BufferHandle(const BufferHandle&) = delete;
        BufferHandle& operator=(const BufferHandle&) = delete;

        ~BufferHandle() {
            release();
        }

        void release() {
            if (buffer_ && pool_) {
                pool_->returnBuffer(buffer_, size_);
                buffer_ = nullptr;
                size_ = 0;
                pool_ = nullptr;
            }
        }

        template<typename T>
        T* as() const noexcept {
            return reinterpret_cast<T*>(buffer_);
        }

        uint8_t* data() const noexcept {
            return buffer_;
        }

        size_t size() const noexcept {
            return size_;
        }

        bool valid() const noexcept {
            return buffer_ != nullptr;
        }

        operator bool() const noexcept {
            return valid();
        }

        void clear() {
            if (buffer_) {
                std::memset(buffer_, 0, size_);
            }
        }
    };

private:
    struct BufferBucket {
        size_t buffer_size;
        std::vector<uint8_t*> free_buffers;
        std::vector<std::unique_ptr<uint8_t[]>> memory_blocks;
        size_t buffers_per_block;

        BufferBucket(size_t size, size_t init_count = 8)
            : buffer_size(ALIGNED_SIZE(size)), buffers_per_block(64) {
            free_buffers.reserve(init_count);
            allocateBlock();
        }

        void allocateBlock() {
            size_t block_size = buffer_size * buffers_per_block;
            auto memory = std::make_unique<uint8_t[]>(block_size);
            uint8_t* base = memory.get();

            for (size_t i = 0; i < buffers_per_block; i++) {
                free_buffers.push_back(base + i * buffer_size);
            }

            memory_blocks.push_back(std::move(memory));
        }
    };

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

// Task scheduling and parallelism
//---------------------------------

// Base Task system for parallel execution
class TaskSystem {
protected:
    struct Task {
        std::function<void()> func;
        std::atomic<int> dependencies;
        std::vector<Task*> dependents;

        Task(std::function<void()> f) : func(std::move(f)), dependencies(0) {}
    };

    std::vector<std::thread> workers;
    std::vector<Task*> ready_tasks;
    std::mutex task_mutex;
    std::condition_variable task_cv;
    std::atomic<bool> running;
    std::atomic<int> active_tasks;
    FixedAllocator<Task> task_allocator;

public:
    TaskSystem(size_t thread_count = 0) : running(true), active_tasks(0) {
        if (thread_count == 0) {
            thread_count = std::max(1u, std::thread::hardware_concurrency() - 1);
        }

        for (size_t i = 0; i < thread_count; i++) {
            workers.emplace_back([this]() {
                worker_main();
                });
        }
    }

    virtual ~TaskSystem() {
        {
            std::lock_guard<std::mutex> lock(task_mutex);
            running = false;
        }
        task_cv.notify_all();

        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // Create a task with no dependencies
    Task* createTask(std::function<void()> func) {
        Task* task = task_allocator.allocate();
        new (task) Task(std::move(func));
        return task;
    }

    // Add dependency between tasks
    void addDependency(Task* dependent, Task* dependency) {
        dependent->dependencies++;
        dependency->dependents.push_back(dependent);
    }

    // Schedule a task for execution
    virtual void scheduleTask(Task* task) {
        if (task->dependencies.load() == 0) {
            std::lock_guard<std::mutex> lock(task_mutex);
            ready_tasks.push_back(task);
            active_tasks++;
            task_cv.notify_one();
        }
    }

    // Wait for all tasks to complete
    virtual void waitAll() {
        std::unique_lock<std::mutex> lock(task_mutex);
        task_cv.wait(lock, [this]() {
            return active_tasks.load() == 0 && ready_tasks.empty();
            });

        // Reset task allocator for reuse in next frame
        task_allocator.reset();
    }

protected:
    virtual void worker_main() {
        while (running) {
            Task* task = nullptr;

            {
                std::unique_lock<std::mutex> lock(task_mutex);
                task_cv.wait(lock, [this]() {
                    return !running || !ready_tasks.empty();
                    });

                if (!running && ready_tasks.empty()) {
                    return;
                }

                if (!ready_tasks.empty()) {
                    task = ready_tasks.back();
                    ready_tasks.pop_back();
                }
            }

            if (task) {
                task->func();

                // Complete dependent tasks
                for (Task* dependent : task->dependents) {
                    int prev_deps = dependent->dependencies.fetch_sub(1);
                    if (prev_deps == 1) {
                        std::lock_guard<std::mutex> lock(task_mutex);
                        ready_tasks.push_back(dependent);
                        task_cv.notify_one();
                    }
                }

                active_tasks--;

                // Notify waiters if all tasks are done
                if (active_tasks.load() == 0) {
                    task_cv.notify_all();
                }
            }
        }
    }
};

// Work-stealing scheduler
class WorkStealingTaskSystem : public TaskSystem {
private:
    struct ThreadLocalQueue {
        ALIGN_TO_CACHE std::vector<Task*> local_tasks;
        std::mutex queue_mutex;
    };

    std::vector<ThreadLocalQueue> thread_queues;
    std::atomic<int> active_thread_count;

public:
    WorkStealingTaskSystem(size_t thread_count = 0);
    ~WorkStealingTaskSystem() override;

    void scheduleTaskForThread(Task* task, int thread_id);
    Task* stealTask(int thief_thread_id);

    // Override base methods
    void scheduleTask(Task* task) override;
    void waitAll() override;

protected:
    void worker_main() override;
};

// Fiber-based tasks
struct FiberTask {
    void* fiber;
    std::function<void()> func;
    bool completed;
    FiberTask* parent;
};

class FiberTaskSystem {
private:
    void* main_fiber;
    std::vector<void*> fiber_pool;
    std::vector<FiberTask*> ready_tasks;
    std::mutex task_mutex;

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

#ifdef USE_AVX512
#include <immintrin.h>

struct SimdFloat {
    __m512 v;

    SimdFloat() : v(_mm512_setzero_ps()) {}
    explicit SimdFloat(__m512 val) : v(val) {}
    explicit SimdFloat(float val) : v(_mm512_set1_ps(val)) {}

    static SimdFloat load(const float* ptr) {
        return SimdFloat(_mm512_loadu_ps(ptr));
    }

    void store(float* ptr) const {
        _mm512_storeu_ps(ptr, v);
    }

    SimdFloat operator+(const SimdFloat& rhs) const {
        return SimdFloat(_mm512_add_ps(v, rhs.v));
    }

    SimdFloat operator-(const SimdFloat& rhs) const {
        return SimdFloat(_mm512_sub_ps(v, rhs.v));
    }

    SimdFloat operator*(const SimdFloat& rhs) const {
        return SimdFloat(_mm512_mul_ps(v, rhs.v));
    }

    SimdFloat operator/(const SimdFloat& rhs) const {
        return SimdFloat(_mm512_div_ps(v, rhs.v));
    }
};

struct SimdInt {
    __m512i v;

    SimdInt() : v(_mm512_setzero_si512()) {}
    explicit SimdInt(__m512i val) : v(val) {}
    explicit SimdInt(int val) : v(_mm512_set1_epi32(val)) {}

    static SimdInt load(const int* ptr) {
        return SimdInt(_mm512_loadu_si512((__m512i*)ptr));
    }

    void store(int* ptr) const {
        _mm512_storeu_si512((__m512i*)ptr, v);
    }

    SimdInt operator&(const SimdInt& rhs) const {
        return SimdInt(_mm512_and_si512(v, rhs.v));
    }

    SimdInt operator|(const SimdInt& rhs) const {
        return SimdInt(_mm512_or_si512(v, rhs.v));
    }

    SimdInt operator^(const SimdInt& rhs) const {
        return SimdInt(_mm512_xor_si512(v, rhs.v));
    }
};
#elif defined(USE_AVX) || defined(USE_AVX2)
#include <immintrin.h>

struct SimdFloat {
    __m256 v;

    SimdFloat() : v(_mm256_setzero_ps()) {}
    explicit SimdFloat(__m256 val) : v(val) {}
    explicit SimdFloat(float val) : v(_mm256_set1_ps(val)) {}

    static SimdFloat load(const float* ptr) {
        return SimdFloat(_mm256_loadu_ps(ptr));
    }

    void store(float* ptr) const {
        _mm256_storeu_ps(ptr, v);
    }

    SimdFloat operator+(const SimdFloat& rhs) const {
        return SimdFloat(_mm256_add_ps(v, rhs.v));
    }

    SimdFloat operator-(const SimdFloat& rhs) const {
        return SimdFloat(_mm256_sub_ps(v, rhs.v));
    }

    SimdFloat operator*(const SimdFloat& rhs) const {
        return SimdFloat(_mm256_mul_ps(v, rhs.v));
    }

    SimdFloat operator/(const SimdFloat& rhs) const {
        return SimdFloat(_mm256_div_ps(v, rhs.v));
    }
};

struct SimdInt {
    __m256i v;

    SimdInt() : v(_mm256_setzero_si256()) {}
    explicit SimdInt(__m256i val) : v(val) {}
    explicit SimdInt(int val) : v(_mm256_set1_epi32(val)) {}

    static SimdInt load(const int* ptr) {
        return SimdInt(_mm256_loadu_si256((__m256i*)ptr));
    }

    void store(int* ptr) const {
        _mm256_storeu_si256((__m256i*)ptr, v);
    }

    SimdInt operator&(const SimdInt& rhs) const {
        return SimdInt(_mm256_and_si256(v, rhs.v));
    }

    SimdInt operator|(const SimdInt& rhs) const {
        return SimdInt(_mm256_or_si256(v, rhs.v));
    }

    SimdInt operator^(const SimdInt& rhs) const {
        return SimdInt(_mm256_xor_si256(v, rhs.v));
    }
};
#elif defined(USE_SSE)
#include <xmmintrin.h>

struct SimdFloat {
    __m128 v;

    SimdFloat() : v(_mm_setzero_ps()) {}
    explicit SimdFloat(__m128 val) : v(val) {}
    explicit SimdFloat(float val) : v(_mm_set1_ps(val)) {}

    static SimdFloat load(const float* ptr) {
        return SimdFloat(_mm_loadu_ps(ptr));
    }

    void store(float* ptr) const {
        _mm_storeu_ps(ptr, v);
    }

    SimdFloat operator+(const SimdFloat& rhs) const {
        return SimdFloat(_mm_add_ps(v, rhs.v));
    }

    SimdFloat operator-(const SimdFloat& rhs) const {
        return SimdFloat(_mm_sub_ps(v, rhs.v));
    }

    SimdFloat operator*(const SimdFloat& rhs) const {
        return SimdFloat(_mm_mul_ps(v, rhs.v));
    }

    SimdFloat operator/(const SimdFloat& rhs) const {
        return SimdFloat(_mm_div_ps(v, rhs.v));
    }
};

struct SimdInt {
    __m128i v;

    SimdInt() : v(_mm_setzero_si128()) {}
    explicit SimdInt(__m128i val) : v(val) {}
    explicit SimdInt(int val) : v(_mm_set1_epi32(val)) {}

    static SimdInt load(const int* ptr) {
        return SimdInt(_mm_loadu_si128((__m128i*)ptr));
    }

    void store(int* ptr) const {
        _mm_storeu_si128((__m128i*)ptr, v);
    }

    SimdInt operator&(const SimdInt& rhs) const {
        return SimdInt(_mm_and_si128(v, rhs.v));
    }

    SimdInt operator|(const SimdInt& rhs) const {
        return SimdInt(_mm_or_si128(v, rhs.v));
    }

    SimdInt operator^(const SimdInt& rhs) const {
        return SimdInt(_mm_xor_si128(v, rhs.v));
    }
};
#else
    // Scalar fallback 
struct SimdFloat {
    float v[4];

    SimdFloat() {
        v[0] = v[1] = v[2] = v[3] = 0.0f;
    }

    explicit SimdFloat(float val) {
        v[0] = v[1] = v[2] = v[3] = val;
    }

    static SimdFloat load(const float* ptr) {
        SimdFloat result;
        for (int i = 0; i < 4; i++) {
            result.v[i] = ptr[i];
        }
        return result;
    }

    void store(float* ptr) const {
        for (int i = 0; i < 4; i++) {
            ptr[i] = v[i];
        }
    }

    SimdFloat operator+(const SimdFloat& rhs) const {
        SimdFloat result;
        for (int i = 0; i < 4; i++) {
            result.v[i] = v[i] + rhs.v[i];
        }
        return result;
    }

    SimdFloat operator-(const SimdFloat& rhs) const {
        SimdFloat result;
        for (int i = 0; i < 4; i++) {
            result.v[i] = v[i] - rhs.v[i];
        }
        return result;
    }

    SimdFloat operator*(const SimdFloat& rhs) const {
        SimdFloat result;
        for (int i = 0; i < 4; i++) {
            result.v[i] = v[i] * rhs.v[i];
        }
        return result;
    }

    SimdFloat operator/(const SimdFloat& rhs) const {
        SimdFloat result;
        for (int i = 0; i < 4; i++) {
            result.v[i] = v[i] / rhs.v[i];
        }
        return result;
    }
};

struct SimdInt {
    int v[4];

    SimdInt() {
        v[0] = v[1] = v[2] = v[3] = 0;
    }

    explicit SimdInt(int val) {
        v[0] = v[1] = v[2] = v[3] = val;
    }

    static SimdInt load(const int* ptr) {
        SimdInt result;
        for (int i = 0; i < 4; i++) {
            result.v[i] = ptr[i];
        }
        return result;
    }

    void store(int* ptr) const {
        for (int i = 0; i < 4; i++) {
            ptr[i] = v[i];
        }
    }

    SimdInt operator&(const SimdInt& rhs) const {
        SimdInt result;
        for (int i = 0; i < 4; i++) {
            result.v[i] = v[i] & rhs.v[i];
        }
        return result;
    }

    SimdInt operator|(const SimdInt& rhs) const {
        SimdInt result;
        for (int i = 0; i < 4; i++) {
            result.v[i] = v[i] | rhs.v[i];
        }
        return result;
    }

    SimdInt operator^(const SimdInt& rhs) const {
        SimdInt result;
        for (int i = 0; i < 4; i++) {
            result.v[i] = v[i] ^ rhs.v[i];
        }
        return result;
    }
};
#endif

// SIMD Matrix Operations
namespace SimdMatrixOps {
    // Matrix multiplication with SIMD
    void multiplyAVX(const Mat4& a, const Mat4& b, Mat4& result);

    // Batch matrix multiplication
    void batchMultiply(const Mat4* matrices_a, const Mat4* matrices_b, Mat4* results, int count);

    // Transpose matrix
    void transposeAVX(const Mat4& m, Mat4& result);

    // Create transformation matrices
    void createTranslationAVX(float x, float y, float z, Mat4& result);
    void createRotationXYZAVX(float x, float y, float z, Mat4& result);
    void createScaleAVX(float x, float y, float z, Mat4& result);

    // Transform vectors
    void transformVectorAVX(const Mat4& m, const Vec3& v, Vec3& result);
    void batchTransformPoints(const Mat4* matrices, const Vec3* points, Vec3* results, int matrix_count, int point_count);

    // Batch transform updates
    void batchUpdateTransformsAVX(Mat4* local, Mat4* parent, Mat4* world, int count);

    // AABB calculations
    void batchCalculateAABB(const Vec3* points, int point_count, AABB* result);
}

// Core Entity System
//------------------

// Hot/cold data splitting
struct EntityHotData {
    // Frequently accessed data
    float* x;
    float* y;
    float* z;
    float* vel_x;
    float* vel_y;
    float* vel_z;
    bool* active;
    uint32_t* collision_mask;
    uint32_t* physics_layer;
    uint8_t* lod_level;
};

struct EntityColdData {
    // Rarely accessed data
    int* debug_id;
    char** names;
    float* creation_time;
    float* last_accessed_time;
    int* flags;
};

// Function typedefs
typedef void (*EntityUpdateFunc)(EntityChunk* chunk, int start_idx, int count, float delta_time);
typedef void (*EntityRenderFunc)(EntityChunk* chunk, int start_idx, int count, void* render_context);

// Base configuration for entity types
struct EntityTypeConfig {
    int type_id;
    EntityUpdateFunc update_func;
    EntityRenderFunc render_func;
    size_t type_data_size;  // Size of type-specific data per entity
    bool parallel_update;   // Whether this type can be updated in parallel
    uint32_t collision_mask; // Bitmask defining which layers this type collides with
    uint32_t physics_layer;  // Physics layer this type belongs to (0-31)
    bool is_3d;             // Whether this type uses 3D or 2D transforms
    uint32_t lod_levels;    // Number of LOD levels (0 = no LOD)
};

// Entity chunk with SoA layout and cache optimization
struct ALIGN_TO_CACHE EntityChunk {
    // Core identification and state
    alignas(CACHE_LINE_SIZE) int type_id;
    int count;
    int capacity;

    // Split hot/cold data
    alignas(CACHE_LINE_SIZE) EntityHotData hot;
    EntityColdData cold;

    // Entity state flags (hot data - frequently accessed)
    alignas(CACHE_LINE_SIZE) bool* active;
    int* entity_id;       // Global entity ID
    int* group_id;        // Group ID for query filtering

    // Spatial data (hot for both 2D and 3D)
    alignas(CACHE_LINE_SIZE) float* x;
    float* y;
    float* z;             // Z position (used in 3D mode)

    // Hierarchy data (accessed during hierarchy updates)
    alignas(CACHE_LINE_SIZE) int* parent_id;      // Parent entity ID (-1 for root)
    int* first_child_id;  // First child entity ID (-1 if none)
    int* next_sibling_id; // Next sibling entity ID (-1 if none)
    int* depth;           // Hierarchy depth (0 for root)

    // Transform data for 2D (hot data)
    alignas(CACHE_LINE_SIZE) float* local_x;
    float* local_y;
    float* scale_x;
    float* scale_y;
    float* rotation;      // Rotation in radians

    // Transform data for 3D (hot data)
    alignas(CACHE_LINE_SIZE) Mat4* local_transform;  // Local space transform
    Mat4* world_transform;   // World space transform
    Quaternion* rotation_3d; // 3D rotation
    float* scale_z;          // Z scale for 3D

    // Physics data (accessed during physics updates)
    alignas(CACHE_LINE_SIZE) float* vel_x;
    float* vel_y;
    float* vel_z;
    float* mass;
    uint32_t* collision_mask;
    uint32_t* physics_layer;

    // Rendering data (accessed during render prep)
    alignas(CACHE_LINE_SIZE) int* texture_id;
    int* render_layer;
    int* material_id;
    int* mesh_id;
    uint8_t* lod_level;   // Current LOD level

    // Spatial partitioning data (accessed during spatial updates)
    alignas(CACHE_LINE_SIZE) int* grid_cell_index_2d;  // 2D grid cell index
    int* grid_cell_index_3d;  // 3D grid cell index
    AABB* bounds;             // 3D bounding box

    // Type-specific data
    alignas(CACHE_LINE_SIZE) void* type_data;

    // Activity mask for branch reduction
    alignas(CACHE_LINE_SIZE) uint64_t* activity_masks;

    // Constructor
    EntityChunk(int type_id, int capacity, size_t type_data_size, bool is_3d);

    // Destructor
    ~EntityChunk();

    // Method to update entity world transforms
    void updateTransforms(bool is_3d);

    // SIMD optimized methods
    void simdUpdatePositions2D(float delta_time, int start_idx, int count);
    void simdUpdatePositions3D(float delta_time, int start_idx, int count);
    void simdUpdateTransforms2D();
    void simdUpdateTransforms3D();

    // Batch processing methods
    void processBatch(int start_idx, int batch_size, float delta_time);

    // Masked processing for entity updates
    void updateActiveEntitiesMasked(float delta_time);
    uint64_t generateActiveMask(int start_idx, int count) const;
    void processEntityBitMasked(uint64_t mask, int base_idx, float delta_time);
};

// Spatial partitioning for 2D
struct ALIGN_TO_CACHE SpatialGrid2D {
    struct Cell {
        int* entity_indices;  // Indices of entities in this cell
        int count;            // Number of entities in cell
        int capacity;         // Capacity of entity_indices array
    };

    Cell* cells;             // Grid cells
    int width, height;       // Grid dimensions
    int cell_count;          // Total number of cells
    float cell_size;         // Size of each grid cell

    // Query buffer for spatial queries (preallocated)
    int* query_buffer;
    int query_buffer_size;

    // Constructor
    SpatialGrid2D(int width, int height, float cell_size);

    // Destructor
    ~SpatialGrid2D();

    // Insert entity into grid
    void insertEntity(int entity_idx, float x, float y, float width, float height);

    // Remove entity from grid
    void removeEntity(int entity_idx, int cell_idx);

    // Query entities in rectangle
    int queryRect(float x, float y, float width, float height, int* result, int max_results);

    // Update entity position in grid
    void updateEntity(int entity_idx, int old_cell_idx, float x, float y, float width, float height);

    // Convert position to cell index
    int posToCell(float x, float y) const;

    // Dynamic grid optimizations
    void optimizeCellSize();
    float calculateOptimalCellSize() const;
    void redistributeEntities(float new_cell_size);
};

// Spatial partitioning for 3D
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

    // Constructor
    SpatialGrid3D(int width, int height, int depth, float cell_size);

    // Destructor
    ~SpatialGrid3D();

    // Insert entity into grid
    void insertEntity(int entity_idx, const AABB& bounds);

    // Remove entity from grid
    void removeEntity(int entity_idx, int cell_idx);

    // Query entities in box
    int queryBox(const AABB& box, int* result, int max_results);

    // Update entity position in grid
    void updateEntity(int entity_idx, int old_cell_idx, const AABB& bounds);

    // Convert position to cell index
    int posToCell(float x, float y, float z) const;

    // Dynamic grid optimizations
    void optimizeCellSize();
    float calculateOptimalCellSize() const;
    void redistributeEntities(float new_cell_size);
};

// Octree node for hierarchical spatial partitioning in 3D
struct ALIGN_TO_CACHE OctreeNode {
    AABB bounds;               // Bounds of this node
    OctreeNode* children[8];   // Child nodes (null if leaf)
    int* entity_indices;       // Entities in this node (if leaf)
    int entity_count;          // Number of entities
    int entity_capacity;       // Capacity of entity_indices
    bool is_leaf;              // Whether this is a leaf node
    int depth;                 // Depth in octree

    // Constructor
    OctreeNode(const AABB& bounds, int depth);

    // Destructor
    ~OctreeNode();

    // Insert entity
    void insertEntity(int entity_idx, const AABB& entity_bounds);

    // Remove entity
    bool removeEntity(int entity_idx);

    // Query entities in frustum
    int queryFrustum(const void* frustum, int* result, int* result_count, int max_results);
};

// Octree with lazy updates
struct ALIGN_TO_CACHE LazyOctreeNode : public OctreeNode {
    bool needs_update;
    float update_threshold;
    Vec3* last_entity_positions;

    LazyOctreeNode(const AABB& bounds, int depth);
    ~LazyOctreeNode();

    void insertEntityLazy(int entity_idx, const AABB& entity_bounds);
    void updateIfNeeded(const Vec3* entity_positions);
    bool shouldUpdate(const Vec3* entity_positions) const;
};

// Level-of-Detail system
struct LODSystem {
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

    // Constructor
    LODSystem(int max_entity_types);

    // Destructor
    ~LODSystem();

    // Register LOD levels for entity type
    void registerEntityTypeLOD(int type_id, const LODLevel* levels, int level_count);

    // Update LOD levels based on camera
    void updateLOD(EntityChunk** chunks, int chunk_count, const Vec3& camera_pos);

    // Parallel LOD updates
    void updateLODParallel(TaskSystem& tasks, EntityChunk** chunks, int chunk_count, const Vec3& camera_pos);
};

// Physics layer system for collision filtering
struct PhysicsLayer {
    uint32_t layer_mask;     // Bitmask for this layer (power of 2)
    uint32_t collides_with;  // Bitmask of layers this layer collides with

    PhysicsLayer() : layer_mask(0), collides_with(0) {}
};

// Enhanced collision system with precomputed layer matrix
struct ALIGN_TO_CACHE EnhancedCollisionSystem : public CollisionSystem {
    uint32_t collisionMatrix[MAX_PHYSICS_LAYERS];

    // Parallel collision detection jobs
    struct ALIGN_TO_CACHE CollisionJob {
        int start_cell;
        int end_cell;
        CollisionPair* local_collision_buffer;
        int collision_count;
    };

    EnhancedCollisionSystem();
    ~EnhancedCollisionSystem();

    void precomputeCollisionMatrix();

    // Fast layer collision check
    bool inline canCollide(int layerA, int layerB) const {
        return collisionMatrix[layerA] & (1 << layerB);
    }

    void broadphase2DWithLayers(SpatialGrid2D* grid, EntityChunk** chunks, int chunk_count);
    void broadphase3DWithLayers(SpatialGrid3D* grid, EntityChunk** chunks, int chunk_count);

    // Multithreaded collision detection
    void detectCollisionsParallel(SpatialGrid3D* grid, EntityChunk** chunks,
        int chunk_count, TaskSystem& tasks);
    void mergeCollisionResults(CollisionJob* jobs, int job_count);
};

// Camera systems
struct Camera2D {
    float x, y;            // Position
    float width, height;   // Viewport dimensions
    float rotation;        // Rotation in radians
    float zoom;            // Zoom level

    // View rectangle in world space
    Rect getViewRect() const;

    // World to screen conversion
    Vec2 worldToScreen(float world_x, float world_y) const;

    // Screen to world conversion
    Vec2 screenToWorld(float screen_x, float screen_y) const;
};

struct Camera3D {
    Vec3 position;         // Camera position
    Vec3 target;           // Look target
    Vec3 up;               // Up vector
    float fov;             // Field of view in radians
    float aspect_ratio;    // Aspect ratio
    float near_plane;      // Near clipping plane
    float far_plane;       // Far clipping plane

    // View-projection matrix
    Mat4 getViewMatrix() const;
    Mat4 getProjectionMatrix() const;
    Mat4 getViewProjectionMatrix() const;

    // Frustum planes for culling
    void extractFrustumPlanes(Vec4* planes) const;
};

// Rendering Systems
//-----------------

// Texture atlas for sprite batching
struct TextureAtlas {
    SDL_Texture* texture;    // Texture handle
    SDL_FRect* regions;      // UV regions for subtextures
    int region_count;        // Number of regions
    int region_capacity;     // Capacity of regions array
};

// Texture atlas with mipmapping
struct MipmappedTextureAtlas : public TextureAtlas {
    int mipmap_levels;
    SDL_Texture** mipmap_textures;

    MipmappedTextureAtlas(int max_regions, int mip_levels);
    ~MipmappedTextureAtlas();

    void generateMipmaps();
    SDL_Texture* getMipmapForDistance(float distance) const;
};

// Material system for 3D rendering
struct Material {
    int diffuse_texture;    // Diffuse texture ID
    int normal_texture;     // Normal map texture ID
    int specular_texture;   // Specular map texture ID
    Vec4 diffuse_color;     // Diffuse color
    Vec4 specular_color;    // Specular color
    float shininess;        // Specular power
};

// Mesh data for 3D rendering
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
};

// Cache-friendly vertex cache
struct ALIGN_TO_CACHE VertexCache {
    SDL_Vertex vertices[VERTEX_CACHE_SIZE];
    int count;

    VertexCache() : count(0) {}

    void addVertex(const SDL_Vertex& vertex) {
        if (count < VERTEX_CACHE_SIZE) {
            vertices[count++] = vertex;
        }
    }

    void flush(SDL_Vertex* dest, int offset) {
        if (count > 0) {
            memcpy(dest + offset, vertices, count * sizeof(SDL_Vertex));
            count = 0;
        }
    }
};

// Render batch for 2D sprites (optimized for batching)
struct ALIGN_TO_CACHE RenderBatch {
    int texture_id;           // Texture ID
    int layer;                // Z-ordering layer
    SDL_Vertex* vertices;     // Vertex data 
    int* indices;             // Index data
    int vertex_count;         // Current vertex count
    int index_count;          // Current index count
    int vertex_capacity;      // Vertex capacity
    int index_capacity;       // Index capacity
    VertexCache vertex_cache; // Cache for vertex data
};

// Instance data for 3D instanced rendering
struct ALIGN_TO_CACHE InstanceData {
    Mat4 transforms[INSTANCE_BATCH_SIZE];
    Vec4 colors[INSTANCE_BATCH_SIZE];
    int instance_count;
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

    void renderInstanced();
    void updateInstanceData(EntityChunk* chunk, int start_idx, int count);
};

// Shader program
struct Shader {
    uint32_t program;          // Shader program ID
    std::unordered_map<std::string, int> uniforms;  // Uniform locations
};

// Entity Manager 
//-------------

// Main entity manager with chunked SoA
struct ALIGN_TO_CACHE EntityManager {
    // Entity chunks
    EntityChunk** chunks;       // Array of entity chunks
    int chunk_count;            // Number of chunks
    int chunk_capacity;         // Capacity of chunks array
    int total_entity_count;     // Total number of entities
    int preallocated_count;     // Preallocated entity count

    // Hierarchy data
    int* roots;                 // Root entity indices
    int root_count;             // Number of root entities
    int* level_starts;          // Start indices per hierarchy level
    int level_count;            // Number of hierarchy levels
    int* sorted_entities;       // Entities sorted by hierarchy depth
    bool hierarchy_dirty;       // Whether hierarchy needs resorting

    // Type system
    EntityTypeConfig* entity_types;  // Entity type definitions
    int type_count;                  // Number of registered types
    int type_capacity;               // Capacity of entity_types array

    // Quick lookup tables
    int* entity_to_chunk;      // Maps entity index to chunk index
    int* entity_to_local;      // Maps entity index to local index in chunk

    // Constructor
    EntityManager(int total_entity_count);

    // Destructor
    ~EntityManager();

    // Register entity type
    int registerEntityType(const EntityTypeConfig& config);

    // Create entity of specified type
    int createEntity(int type_id);

    // Get chunk and local index from entity index
    void getChunkIndices(int entity_idx, int* chunk_idx, int* local_idx) const;

    // Check if entity is valid
    bool isValidEntity(int entity_idx) const;

    // Set parent-child relationship
    void setParent(int entity_idx, int parent_idx);

    // Sort entities by hierarchy depth
    void sortByHierarchyDepth();

    // Update all entity transformations
    void updateEntityTransforms();

    // Get type-specific data for entity
    void* getEntityTypeData(int entity_idx);

    // Set entity active state
    void setEntityActive(int entity_idx, bool active);

    // Get entity position (2D or 3D)
    Vec3 getEntityPosition(int entity_idx);

    // Set entity position (2D or 3D)
    void setEntityPosition(int entity_idx, float x, float y, float z = 0.0f);

    // Set entity local position (2D or 3D)
    void setEntityLocalPosition(int entity_idx, float x, float y, float z = 0.0f);

    // Explicit cache control
    void prefetchEntityData(int entity_idx);
    void prefetchChunkData(EntityChunk* chunk, int start_idx, int count);
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

    // Constructor
    GPUResources(int max_textures, int max_shaders, int max_meshes, int max_materials, int max_render_targets);

    // Destructor
    ~GPUResources();

    // Create and load a texture
    int createTexture(SDL_Surface* surface);

    // Create a texture atlas
    int createTextureAtlas(SDL_Surface** surfaces, int count, int* width, int* height);

    // Create a mipmapped texture atlas
    int createMipmappedTextureAtlas(SDL_Surface** surfaces, int count, int* width, int* height, int mipmap_levels);

    // Create a shader program
    int createShader(const char* vertex_source, const char* fragment_source);

    // Create a mesh from vertex data
    int createMesh(const float* vertices, int vertex_count, const int* indices, int index_count);

    // Create a material
    int createMaterial(int diffuse_texture, int normal_texture, int specular_texture, const Vec4& diffuse_color, const Vec4& specular_color, float shininess);

    // Create a render target
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
    SpatialGrid2D grid2d;
    SpatialGrid3D grid3d;
    LazyOctreeNode* octree_root;

    // Camera systems
    Camera2D camera2d;
    Camera3D camera3d;

    // Rendering systems
    RenderBatch* batches2d;
    int batch_count2d;
    RenderBatch3D* batches3d;
    int batch_count3d;

    // GPU resources
    GPUResources gpu_resources;

    // Time tracking
    uint64_t last_frame_time;
    float delta_time;
    float fps;

    // Task scheduling
    WorkStealingTaskSystem task_system;
    FiberTaskSystem fiber_system;

    // Systems
    LODSystem lod_system;
    EnhancedCollisionSystem collision_system;

    // Memory management
    BufferPool buffer_pool;

    // Temp work buffers
    int* entity_buffer;
    int entity_buffer_size;

    // Engine settings
    SDL_FRect world_bounds;
    float grid_cell_size;
    int thread_count;
    bool use_3d_mode;

    // Debug settings
    bool debug_mode;
    bool profile_memory;

    // Constructor
    Engine(int window_width, int window_height, int world_width, int world_height, int cell_size, int total_entities);

    // Destructor
    ~Engine();

    // Engine lifecycle
    void update();
    void render();

    // Entity creation during initialization phase
    int addEntity(int type_id, float x, float y, float z, int width, int height, int texture_id, int layer);
    int addEntity2D(int type_id, float x, float y, int width, int height, int texture_id, int layer);
    int addEntity3D(int type_id, float x, float y, float z, int mesh_id, int material_id, int layer);

    // Entity manipulation
    void setEntityPosition(int entity_id, float x, float y, float z = 0.0f);
    void setEntityRotation(int entity_id, float rotation);
    void setEntityRotation3D(int entity_id, float x, float y, float z);
    void setEntityScale(int entity_id, float x, float y, float z = 1.0f);
    void setEntityActive(int entity_id, bool active);
    void setEntityLayer(int entity_id, int layer);
    void setEntityTexture(int entity_id, int texture_id);
    void setEntityMesh(int entity_id, int mesh_id);
    void setEntityMaterial(int entity_id, int material_id);

    // Hierarchy
    void setParent(int entity_id, int parent_id);
    int getParent(int entity_id);
    int getFirstChild(int entity_id);
    int getNextSibling(int entity_id);
    void setEntityLocalPosition(int entity_id, float x, float y, float z = 0.0f);

    // Spatial queries
    int queryRect(float x, float y, float width, float height, int* result_buffer, int max_results);
    int queryBox(const AABB& box, int* result_buffer, int max_results);
    int queryFrustum(const Camera3D& camera, int* result_buffer, int max_results);
    int queryEntitiesByGroup(int group_id, int* result_buffer, int max_results);

    // Camera control
    void setCameraPosition2D(float x, float y);
    void setCameraRotation2D(float rotation);
    void setCameraZoom2D(float zoom);
    void setCameraPosition3D(float x, float y, float z);
    void setCameraTarget3D(float x, float y, float z);
    void setCameraUp3D(float x, float y, float z);
    void setCameraFov(float fov_degrees);

    // Resource management
    int addTexture(SDL_Surface* surface);
    int addTextureAtlas(SDL_Surface** surfaces, int count, int* width, int* height);
    int addMipmappedTextureAtlas(SDL_Surface** surfaces, int count, int* width, int* height, int mipmap_levels);
    int addShader(const char* vertex_source, const char* fragment_source);
    int addMesh(const float* vertices, int vertex_count, const int* indices, int index_count);
    int addMaterial(int diffuse_texture, int normal_texture, int specular_texture, const Vec4& diffuse_color, const Vec4& specular_color, float shininess);

    // Physics and collision
    void registerPhysicsLayer(int layer_id, uint32_t collides_with);
    void setEntityPhysicsLayer(int entity_id, int layer_id);
    void setEntityVelocity(int entity_id, float vx, float vy, float vz = 0.0f);
    void setEntityMass(int entity_id, float mass);
    int queryCollisions(int entity_id, int* result_buffer, int max_results);

    // Level of Detail
    void registerEntityLOD(int type_id, const LODSystem::LODLevel* levels, int level_count);

    // Debug/profiling
    void setDebugMode(bool enabled);
    float getFrameTime();
    float getFps();
    void getMemoryStats(size_t* total_allocated, size_t* temp_allocated);
};

// Inline implementation of frequently used methods
// These are defined here to encourage inlining for performance

inline void EntityManager::getChunkIndices(int entity_idx, int* chunk_idx, int* local_idx) const {
    assert(entity_idx >= 0 && entity_idx < total_entity_count);
    *chunk_idx = entity_to_chunk[entity_idx];
    *local_idx = entity_to_local[entity_idx];
}

inline bool EntityManager::isValidEntity(int entity_idx) const {
    if (entity_idx < 0 || entity_idx >= total_entity_count) {
        return false;
    }

    int chunk_idx, local_idx;
    getChunkIndices(entity_idx, &chunk_idx, &local_idx);

    if (chunk_idx < 0 || chunk_idx >= chunk_count) {
        return false;
    }

    EntityChunk* chunk = chunks[chunk_idx];
    return (local_idx >= 0 && local_idx < chunk->count && chunk->active[local_idx]);
}

inline void* EntityManager::getEntityTypeData(int entity_idx) {
    int chunk_idx, local_idx;
    getChunkIndices(entity_idx, &chunk_idx, &local_idx);

    EntityChunk* chunk = chunks[chunk_idx];
    if (!chunk->type_data) {
        return nullptr;
    }

    // The size of each entity's type-specific data is stored in the entity type config
    size_t type_data_size = entity_types[chunk->type_id].type_data_size;
    return (uint8_t*)chunk->type_data + (local_idx * type_data_size);
}

inline bool EnhancedCollisionSystem::canCollide(int layerA, int layerB) const {
    return collisionMatrix[layerA] & (1 << layerB);
}

// Explicit prefetching functions
inline void prefetchEntityData(EntityChunk* chunk, int start_idx, int count) {
    for (int i = start_idx; i < start_idx + count; i += CACHE_LINE_SIZE / sizeof(float)) {
#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_AVX) || defined(USE_SSE)
        _mm_prefetch((const char*)&chunk->x[i + 8], PREFETCH_READ);
        _mm_prefetch((const char*)&chunk->y[i + 8], PREFETCH_READ);
        if (chunk->z) _mm_prefetch((const char*)&chunk->z[i + 8], PREFETCH_READ);
#endif
    }
}

inline void prefetchTransformData(Mat4* transforms, int count) {
    for (int i = 0; i < count; i++) {
#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_AVX) || defined(USE_SSE)
        _mm_prefetch((const char*)&transforms[i].m[0], PREFETCH_READ);
        _mm_prefetch((const char*)&transforms[i].m[4], PREFETCH_READ);
        _mm_prefetch((const char*)&transforms[i].m[8], PREFETCH_READ);
        _mm_prefetch((const char*)&transforms[i].m[12], PREFETCH_READ);
#endif
    }
}

inline void prefetchVertexData(SDL_Vertex* vertices, int count) {
    for (int i = 0; i < count; i += CACHE_LINE_SIZE / sizeof(SDL_Vertex)) {
#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_AVX) || defined(USE_SSE)
        _mm_prefetch((const char*)&vertices[i + 4], PREFETCH_READ);
#endif
    }
}

// Engine API Function Declarations
//-------------------------------

// Engine lifecycle
Engine* engine_create(int window_width, int window_height, int world_width, int world_height, int cell_size, int total_entities);
void engine_destroy(Engine* engine);
void engine_update(Engine* engine);
void engine_render(Engine* engine);

// Entity creation (initialization only)
int engine_add_entity(Engine* engine, int type_id, float x, float y, float z, int width, int height, int texture_id, int layer);
int engine_add_entity_2d(Engine* engine, int type_id, float x, float y, int width, int height, int texture_id, int layer);
int engine_add_entity_3d(Engine* engine, int type_id, float x, float y, float z, int mesh_id, int material_id, int layer);

// Entity manipulation
void engine_set_entity_position(Engine* engine, int entity_id, float x, float y, float z);
void engine_set_entity_rotation(Engine* engine, int entity_id, float rotation);
void engine_set_entity_rotation_3d(Engine* engine, int entity_id, float x, float y, float z);
void engine_set_entity_scale(Engine* engine, int entity_id, float x, float y, float z);
void engine_set_entity_active(Engine* engine, int entity_id, bool active);
void engine_set_entity_layer(Engine* engine, int entity_id, int layer);
void engine_set_entity_texture(Engine* engine, int entity_id, int texture_id);
void engine_set_entity_mesh(Engine* engine, int entity_id, int mesh_id);
void engine_set_entity_material(Engine* engine, int entity_id, int material_id);

// Hierarchy management
void engine_set_parent(Engine* engine, int entity_id, int parent_id);
int engine_get_parent(Engine* engine, int entity_id);
int engine_get_first_child(Engine* engine, int entity_id);
int engine_get_next_sibling(Engine* engine, int entity_id);
void engine_set_entity_local_position(Engine* engine, int entity_id, float x, float y, float z);

// Spatial queries
int engine_query_rect(Engine* engine, float x, float y, float width, float height, int* result_buffer, int max_results);
int engine_query_box(Engine* engine, float min_x, float min_y, float min_z, float max_x, float max_y, float max_z, int* result_buffer, int max_results);
int engine_query_frustum(Engine* engine, int* result_buffer, int max_results);
int engine_query_entities_by_group(Engine* engine, int group_id, int* result_buffer, int max_results);

// Camera control
void engine_set_camera_position_2d(Engine* engine, float x, float y);
void engine_set_camera_rotation_2d(Engine* engine, float rotation);
void engine_set_camera_zoom_2d(Engine* engine, float zoom);
void engine_set_camera_position_3d(Engine* engine, float x, float y, float z);
void engine_set_camera_target_3d(Engine* engine, float x, float y, float z);
void engine_set_camera_up_3d(Engine* engine, float x, float y, float z);
void engine_set_camera_fov(Engine* engine, float fov_degrees);

// Resource management
int engine_add_texture(Engine* engine, SDL_Surface* surface);
int engine_add_texture_atlas(Engine* engine, SDL_Surface** surfaces, int count, int* width, int* height);
int engine_add_mipmapped_texture_atlas(Engine* engine, SDL_Surface** surfaces, int count, int* width, int* height, int mipmap_levels);
int engine_add_shader(Engine* engine, const char* vertex_source, const char* fragment_source);
int engine_add_mesh(Engine* engine, const float* vertices, int vertex_count, const int* indices, int index_count);
int engine_add_material(Engine* engine, int diffuse_texture, int normal_texture, int specular_texture,
    float diffuse_r, float diffuse_g, float diffuse_b, float diffuse_a,
    float specular_r, float specular_g, float specular_b, float specular_a,
    float shininess);

// Physics and collision
void engine_register_physics_layer(Engine* engine, int layer_id, uint32_t collides_with);
void engine_set_entity_physics_layer(Engine* engine, int entity_id, int layer_id);
void engine_set_entity_velocity(Engine* engine, int entity_id, float vx, float vy, float vz);
void engine_set_entity_mass(Engine* engine, int entity_id, float mass);
int engine_query_collisions(Engine* engine, int entity_id, int* result_buffer, int max_results);

// Level of Detail
void engine_register_entity_lod(Engine* engine, int type_id, const float* distance_thresholds, const int* mesh_ids, int level_count);

// Entity type system
void engine_register_entity_type(Engine* engine, int type_id, EntityUpdateFunc update_func, EntityRenderFunc render_func,
    size_t extra_data_size, bool parallel_update, uint32_t collision_mask, uint32_t physics_layer,
    bool is_3d);

// Debug/profiling
void engine_set_debug_mode(Engine* engine, bool enabled);
float engine_get_frame_time(Engine* engine);
float engine_get_fps(Engine* engine);
void engine_get_memory_stats(Engine* engine, size_t* total_allocated, size_t* temp_allocated);

#endif // ENGINE_H