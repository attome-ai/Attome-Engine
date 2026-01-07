#include "ATMCommon.h"
#include "ATMBufferPool.h"






// ATMBufferPool implementations
ATMBufferPool::BufferHandle::BufferHandle()
    : buffer(nullptr), pool(nullptr) {}

ATMBufferPool::BufferHandle::BufferHandle(std::vector<uint8_t>* buffer, ATMBufferPool* pool)
    : buffer(buffer), pool(pool) {}

ATMBufferPool::BufferHandle::BufferHandle(BufferHandle&& handle) noexcept
    : buffer(handle.buffer), pool(handle.pool) {
    handle.buffer = nullptr;
    handle.pool = nullptr;
}

ATMBufferPool::BufferHandle& ATMBufferPool::BufferHandle::operator=(BufferHandle&& handle) noexcept {
    if (this != &handle) {
        if (buffer && pool) pool->releaseBuffer(buffer);
        buffer = handle.buffer;
        pool = handle.pool;
        handle.buffer = nullptr;
        handle.pool = nullptr;
    }
    return *this;
}

ATMBufferPool::BufferHandle::~BufferHandle() {
    if (buffer && pool) pool->releaseBuffer(buffer);
}

ATMBufferPool::ATMBufferPool(const int& size)
    : default_buffer_size(size) {
    buffers.reserve(16); // Pre-reserve space for some buffers
}

ATMBufferPool::BufferHandle ATMBufferPool::getBuffer() {
    std::vector<uint8_t>* buffer;

    if (buffers.empty()) {
        buffer = new std::vector<uint8_t>();
        buffer->reserve(default_buffer_size);
    }
    else {
        buffer = buffers.back();
        buffers.pop_back();
    }

    buffer->clear();
    return BufferHandle(buffer, this);
}

void ATMBufferPool::releaseBuffer(std::vector<uint8_t>* buff) {
    buffers.push_back(buff);
}

size_t ATMBufferPool::size() const {
    return buffers.size();
}

///////////////////////////////Test////////////////////////
#include <chrono>
#include <functional>
#include <assert.h>
#include <random>
#include <iomanip>
#include <sstream>
#include <chrono>


#define ATMLOG(...) SDL_Log(__VA_ARGS__)

// FixedBufferPool Tests
void TestFixedBufferPoolBasic() {

    // Create a buffer pool with 1KB buffers, 10 initial buffers
    FixedBufferPool pool(1024, 10);

    // Verify initial state
    assert(pool.getTotalBuffers() == 64);
    assert(pool.getAvailableBuffers() == 64);
    assert(pool.getBufferSize() == 1024);

    // Get a buffer
    auto handle = pool.getBuffer();
    assert(handle.valid());
    assert(handle.data() != nullptr);
    assert(pool.getAvailableBuffers() == 63);

    // Release the buffer
    handle.release();
    assert(!handle.valid());
    assert(pool.getAvailableBuffers() == 64);

    ATMLOG("TestFixedBufferPoolBasic PASSED");
}

void TestFixedBufferPoolMoveSemantics() {

    FixedBufferPool pool(1024, 5);

    // Test move constructor
    {
        auto handle1 = pool.getBuffer();
        uint8_t* data = handle1.data();

        auto handle2 = std::move(handle1);
        assert(handle2.data() == data);
        assert(!handle1.valid());
        assert(handle2.valid());
    }

    // Test move assignment
    {
        auto handle1 = pool.getBuffer();
        auto handle2 = pool.getBuffer();
        uint8_t* data1 = handle1.data();

        handle2 = std::move(handle1);
        assert(handle2.data() == data1);
        assert(!handle1.valid());
        assert(handle2.valid());
    }

    ATMLOG("TestFixedBufferPoolMoveSemantics PASSED");
}

void TestFixedBufferPoolExpansion() {

    // Start with 5 buffers
    FixedBufferPool pool(1024, 5);
    assert(pool.getTotalBuffers() == 64);

    // Get all 5 buffers
    std::vector<FixedBufferPool::BufferHandle> handles;
    for (int i = 0; i < 5; i++) {
        handles.push_back(pool.getBuffer());
    }

    assert(pool.getAvailableBuffers() == 59);

    // Get one more - should cause expansion
    auto extraHandle = pool.getBuffer();

    // Should have doubled capacity to 10
    assert(pool.getTotalBuffers() == 64);
    assert(pool.getAvailableBuffers() == 58);  // 5 new buffers - 1 we just took

    ATMLOG("TestFixedBufferPoolExpansion PASSED");
}

// ATMBufferPool Tests
void TestATMBufferPoolBasic() {

    ATMBufferPool pool(1024);

    // Initial size should be 0
    assert(pool.size() == 0);

    // Get a buffer
    auto handle = pool.getBuffer();

    // Size should still be 0 as the buffer is in use
    assert(pool.size() == 0);

    // The buffer should be empty but usable
    assert(handle.buffer->empty());
    handle.buffer->push_back(42);
    assert(handle.buffer->size() == 1);
    assert((*handle.buffer)[0] == 42);

    // Releasing the buffer (via destructor)
    {
        auto temp = std::move(handle);
        // temp goes out of scope here and releases the buffer
    }

    // Buffer should now be back in the pool
    assert(pool.size() == 1);

    ATMLOG("TestATMBufferPoolBasic PASSED");
}

void TestATMBufferPoolMoveSemantics() {

    ATMBufferPool pool(1024);

    // Get a buffer and store data in it
    auto handle1 = pool.getBuffer();
    handle1.buffer->push_back(123);
    std::vector<uint8_t>* bufferAddr = handle1.buffer;

    // Move the handle
    auto handle2 = std::move(handle1);

    // Check that handle1 is now invalid
    assert(handle1.buffer == nullptr);
    assert(handle1.pool == nullptr);

    // Check that handle2 now has the buffer
    assert(handle2.buffer == bufferAddr);
    assert(handle2.buffer->size() == 1);
    assert((*handle2.buffer)[0] == 123);

    ATMLOG("TestATMBufferPoolMoveSemantics PASSED");
}

void TestATMBufferPoolMultipleBuffers() {

    ATMBufferPool pool(1024);

    // Get 10 buffers
    std::vector<ATMBufferPool::BufferHandle> handles;
    for (int i = 0; i < 10; i++) {
        auto handle = pool.getBuffer();
        // Put some data in each buffer to make sure they're distinct
        for (int j = 0; j <= i; j++) {
            handle.buffer->push_back(static_cast<uint8_t>(i));
        }
        handles.push_back(std::move(handle));
    }

    // Verify the data
    for (int i = 0; i < 10; i++) {
        assert(handles[i].buffer->size() == i + 1);
        for (int j = 0; j <= i; j++) {
            assert((*handles[i].buffer)[j] == i);
        }
    }

    // Release all buffers by clearing the vector
    handles.clear();

    // All buffers should be back in the pool
    assert(pool.size() == 10);

    ATMLOG("TestATMBufferPoolMultipleBuffers PASSED");
}

// Benchmarks

// Utility to measure execution time
template<typename Func>
double measureExecutionTime(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void BenchmarkFixedBufferPoolAllocation() {
    ATMLOG("Running BenchmarkFixedBufferPoolAllocation");

    const int numBuffers = 10000;
    const int bufferSize = 1024;

    double createTime = measureExecutionTime([&]() {
        FixedBufferPool pool(bufferSize, numBuffers);
        });

    FixedBufferPool pool(bufferSize, numBuffers);

    double allocTime = measureExecutionTime([&]() {
        std::vector<FixedBufferPool::BufferHandle> handles;
        handles.reserve(numBuffers);
        for (int i = 0; i < numBuffers; i++) {
            handles.push_back(pool.getBuffer());
        }
        });

    ATMLOG("FixedBufferPool: Creating pool with %d buffers of size %d: %.2f ms",
        numBuffers, bufferSize, createTime);
    ATMLOG("FixedBufferPool: Allocating %d buffers: %.2f ms",
        numBuffers, allocTime);
}

void BenchmarkFixedBufferPoolReuse() {
    ATMLOG("Running BenchmarkFixedBufferPoolReuse");

    const int numBuffers = 1000;
    const int iterations = 100;
    const int bufferSize = 1024;

    FixedBufferPool pool(bufferSize, numBuffers);

    double reuseTime = measureExecutionTime([&]() {
        for (int iter = 0; iter < iterations; iter++) {
            std::vector<FixedBufferPool::BufferHandle> handles;
            handles.reserve(numBuffers);

            // Allocate all buffers
            for (int i = 0; i < numBuffers; i++) {
                handles.push_back(pool.getBuffer());
            }

            // Write some data to each buffer
            for (auto& handle : handles) {
                handle.data()[0] = 42;
            }

            // Release all buffers
            handles.clear();
        }
        });

    ATMLOG("FixedBufferPool: Allocating and releasing %d buffers %d times: %.2f ms",
        numBuffers, iterations, reuseTime);
    ATMLOG("FixedBufferPool: Average time per cycle: %.2f ms",
        reuseTime / iterations);
}

void BenchmarkATMBufferPoolAllocation() {
    ATMLOG("Running BenchmarkATMBufferPoolAllocation");

    const int numBuffers = 10000;
    const int bufferSize = 1024;

    double createTime = measureExecutionTime([&]() {
        ATMBufferPool pool(bufferSize);
        });

    ATMBufferPool pool(bufferSize);

    double allocTime = measureExecutionTime([&]() {
        std::vector<ATMBufferPool::BufferHandle> handles;
        handles.reserve(numBuffers);
        for (int i = 0; i < numBuffers; i++) {
            handles.push_back(pool.getBuffer());
        }
        });

    ATMLOG("ATMBufferPool: Creating pool with buffer size %d: %.2f ms",
        bufferSize, createTime);
    ATMLOG("ATMBufferPool: Allocating %d buffers: %.2f ms",
        numBuffers, allocTime);
}

void BenchmarkATMBufferPoolReuse() {
    ATMLOG("Running BenchmarkATMBufferPoolReuse");

    const int numBuffers = 1000;
    const int iterations = 100;
    const int bufferSize = 1024;

    ATMBufferPool pool(bufferSize);

    double reuseTime = measureExecutionTime([&]() {
        for (int iter = 0; iter < iterations; iter++) {
            std::vector<ATMBufferPool::BufferHandle> handles;
            handles.reserve(numBuffers);

            // Allocate all buffers
            for (int i = 0; i < numBuffers; i++) {
                handles.push_back(pool.getBuffer());
            }

            // Write some data to each buffer
            for (auto& handle : handles) {
                handle.buffer->push_back(42);
            }

            // Release all buffers
            handles.clear();
        }
        });

    ATMLOG("ATMBufferPool: Allocating and releasing %d buffers %d times: %.2f ms",
        numBuffers, iterations, reuseTime);
    ATMLOG("ATMBufferPool: Average time per cycle: %.2f ms",
        reuseTime / iterations);
}

void BenchmarkCompareRealWorldUsage() {
    ATMLOG("Running BenchmarkCompareRealWorldUsage");

    const int bufferSize = 1024;
    const int numOperations = 1000000;

    // Random number generator for buffer sizes
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> bufferSizeDist(10, bufferSize);
    std::uniform_int_distribution<> holdTimeDist(1, 100);

    // Create pools
    FixedBufferPool fixedPool(bufferSize, 100000);
    ATMBufferPool atmPool(bufferSize);

    // Simulate real-world usage of FixedBufferPool
    double fixedPoolTime = measureExecutionTime([&]() {
        std::vector<FixedBufferPool::BufferHandle> handles;
        handles.reserve(100000);  // Reserve space to avoid reallocations

        for (int i = 0; i < numOperations; i++) {
            // Occasionally release a random buffer
            if (!handles.empty() && holdTimeDist(gen) < 20) {
                int index = gen() % handles.size();
                handles[index] = std::move(handles.back());
                handles.pop_back();
            }

            // Get a new buffer and use it
            auto handle = fixedPool.getBuffer();
            int dataSize = bufferSizeDist(gen);
            for (int j = 0; j < dataSize; j++) {
                handle.data()[j] = static_cast<uint8_t>(j & 0xFF);
            }

            handles.push_back(std::move(handle));
        }
        });

    // Simulate real-world usage of ATMBufferPool
    double atmPoolTime = measureExecutionTime([&]() {
        std::vector<ATMBufferPool::BufferHandle> handles;
        handles.reserve(100000);  // Reserve space to avoid reallocations

        for (int i = 0; i < numOperations; i++) {
            // Occasionally release a random buffer
            if (!handles.empty() && holdTimeDist(gen) < 20) {
                int index = gen() % handles.size();
                handles[index] = std::move(handles.back());
                handles.pop_back();
            }

            // Get a new buffer and use it
            auto handle = atmPool.getBuffer();
            int dataSize = bufferSizeDist(gen);
            handle.buffer->resize(dataSize);
            for (int j = 0; j < dataSize; j++) {
                (*handle.buffer)[j] = static_cast<uint8_t>(j & 0xFF);
            }

            handles.push_back(std::move(handle));
        }
        });

    ATMLOG("Real-world simulation with %d operations:", numOperations);
    ATMLOG("FixedBufferPool: %.2f ms", fixedPoolTime);
    ATMLOG("ATMBufferPool: %.2f ms", atmPoolTime);
    ATMLOG("Performance ratio: FixedBufferPool is %.2f times %s than ATMBufferPool",
        std::abs(fixedPoolTime / atmPoolTime),
        fixedPoolTime < atmPoolTime ? "faster" : "slower");
}

// Main test and benchmark functions
void ATMBufferPoolRunAllTests() {
    ATMLOG("=== Running all buffer pool tests ===");

    // FixedBufferPool tests
    TestFixedBufferPoolBasic();
    TestFixedBufferPoolMoveSemantics();
    TestFixedBufferPoolExpansion();

    // ATMBufferPool tests
    TestATMBufferPoolBasic();
    TestATMBufferPoolMoveSemantics();
    TestATMBufferPoolMultipleBuffers();

    ATMLOG("=== All tests PASSED ===");
}

void ATMBufferPoolRunAllBenchmarks() {
    ATMLOG("=== Running all buffer pool benchmarks ===");

    // FixedBufferPool benchmarks
    BenchmarkFixedBufferPoolAllocation();
    BenchmarkFixedBufferPoolReuse();

    // ATMBufferPool benchmarks
    BenchmarkATMBufferPoolAllocation();
    BenchmarkATMBufferPoolReuse();

    // Comparison benchmark
    BenchmarkCompareRealWorldUsage();

    ATMLOG("=== All benchmarks completed ===");
}