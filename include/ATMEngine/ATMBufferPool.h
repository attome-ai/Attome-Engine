#pragma once
#include <memory>
#include <vector>
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

class ATMBufferPool {
public:
    struct BufferHandle {
        std::vector<uint8_t>* buffer;
        ATMBufferPool* pool;
        BufferHandle();
        BufferHandle(std::vector<uint8_t>* buffer, ATMBufferPool* pool);
        BufferHandle(BufferHandle&& handle)noexcept;
        BufferHandle& operator=(BufferHandle&& handle)noexcept;

        ~BufferHandle();
    };

    std::vector<std::vector<uint8_t>*> buffers;
    const size_t default_buffer_size;
    ATMBufferPool(const int& size);
    BufferHandle getBuffer();
    void releaseBuffer(std::vector<uint8_t>* buff);
    inline size_t size() const;
};



void ATMBufferPoolRunAllBenchmarks();

void ATMBufferPoolRunAllTests();
