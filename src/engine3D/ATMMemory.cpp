#include <engine3D/ATMEngine.h>

// Constructor - initializes all buffer pools
GPUBufferPools::GPUBufferPools(SDL_GPUDevice* device, uint32_t max_buffers_per_pool)
    : device(device)
{
    // Initialize vertex buffer pool
    vertex_pool.buffers = static_cast<SDL_GPUBuffer**>(SDL_malloc(sizeof(SDL_GPUBuffer*) * max_buffers_per_pool));
    vertex_pool.buffer_sizes = static_cast<size_t*>(SDL_malloc(sizeof(size_t) * max_buffers_per_pool));
    vertex_pool.in_use_flags = static_cast<bool*>(SDL_malloc(sizeof(bool) * max_buffers_per_pool));
    vertex_pool.count = 0;
    vertex_pool.capacity = max_buffers_per_pool;
    vertex_pool.usage_flags = SDL_GPU_BUFFERUSAGE_VERTEX;

    // Initialize index buffer pool
    index_pool.buffers = static_cast<SDL_GPUBuffer**>(SDL_malloc(sizeof(SDL_GPUBuffer*) * max_buffers_per_pool));
    index_pool.buffer_sizes = static_cast<size_t*>(SDL_malloc(sizeof(size_t) * max_buffers_per_pool));
    index_pool.in_use_flags = static_cast<bool*>(SDL_malloc(sizeof(bool) * max_buffers_per_pool));
    index_pool.count = 0;
    index_pool.capacity = max_buffers_per_pool;
    index_pool.usage_flags = SDL_GPU_BUFFERUSAGE_INDEX;

   

    // Initialize storage buffer pool
    storage_pool.buffers = static_cast<SDL_GPUBuffer**>(SDL_malloc(sizeof(SDL_GPUBuffer*) * max_buffers_per_pool));
    storage_pool.buffer_sizes = static_cast<size_t*>(SDL_malloc(sizeof(size_t) * max_buffers_per_pool));
    storage_pool.in_use_flags = static_cast<bool*>(SDL_malloc(sizeof(bool) * max_buffers_per_pool));
    storage_pool.count = 0;
    storage_pool.capacity = max_buffers_per_pool;
    storage_pool.usage_flags = SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ;

    // Initialize instance buffer pool
    instance_pool.buffers = static_cast<SDL_GPUBuffer**>(SDL_malloc(sizeof(SDL_GPUBuffer*) * max_buffers_per_pool));
    instance_pool.buffer_sizes = static_cast<size_t*>(SDL_malloc(sizeof(size_t) * max_buffers_per_pool));
    instance_pool.in_use_flags = static_cast<bool*>(SDL_malloc(sizeof(bool) * max_buffers_per_pool));
    instance_pool.count = 0;
    instance_pool.capacity = max_buffers_per_pool;
    instance_pool.usage_flags = SDL_GPU_BUFFERUSAGE_VERTEX; // Used for instanced rendering

    // Initialize all in_use flags to false
    for (uint32_t i = 0; i < max_buffers_per_pool; ++i) {
        vertex_pool.in_use_flags[i] = false;
        index_pool.in_use_flags[i] = false;
        storage_pool.in_use_flags[i] = false;
        instance_pool.in_use_flags[i] = false;
    }
}

// Destructor
GPUBufferPools::~GPUBufferPools() {
    destroyPools();

    // Free allocated memory for pool tracking arrays
    SDL_free(vertex_pool.buffers);
    SDL_free(vertex_pool.buffer_sizes);
    SDL_free(vertex_pool.in_use_flags);

    SDL_free(index_pool.buffers);
    SDL_free(index_pool.buffer_sizes);
    SDL_free(index_pool.in_use_flags);


    SDL_free(storage_pool.buffers);
    SDL_free(storage_pool.buffer_sizes);
    SDL_free(storage_pool.in_use_flags);

    SDL_free(instance_pool.buffers);
    SDL_free(instance_pool.buffer_sizes);
    SDL_free(instance_pool.in_use_flags);
}

// Get an instance buffer of the requested size
int32_t GPUBufferPools::getInstanceBuffer(size_t size, GPUResources* resources) {
    // First try to find an existing buffer of sufficient size
    for (uint32_t i = 0; i < instance_pool.count; ++i) {
        if (!instance_pool.in_use_flags[i] && instance_pool.buffer_sizes[i] >= size) {
            instance_pool.in_use_flags[i] = true;
            // Calculate the buffer ID based on its position in the resources array
            for (uint32_t j = 0; j < resources->instance_buffer_count; ++j) {
                if (resources->instance_buffers[j] == instance_pool.buffers[i]) {
                    return j;
                }
            }
        }
    }

    // If no suitable buffer found, create a new one if there's space in the pool
    if (instance_pool.count < instance_pool.capacity) {
        // Create buffer with 20% extra size for future growth without reallocations
        size_t buffer_size = size * 1.2;
        SDL_GPUBufferCreateInfo desc = {};
        desc.size = buffer_size;
        desc.usage = instance_pool.usage_flags;
        desc.props = 0; // No special properties

        SDL_GPUBuffer* buffer = SDL_CreateGPUBuffer(device, &desc);
        if (buffer) {
            instance_pool.buffers[instance_pool.count] = buffer;
            instance_pool.buffer_sizes[instance_pool.count] = buffer_size;
            instance_pool.in_use_flags[instance_pool.count] = true;

            // Register buffer with resources
            if (resources->instance_buffer_count < resources->max_instance_buffers) {
                resources->instance_buffers[resources->instance_buffer_count] = buffer;
                int32_t buffer_id = resources->instance_buffer_count;
                resources->instance_buffer_count++;

                // Store the buffer in the pool
                uint32_t pool_index = instance_pool.count;
                instance_pool.count++;

                return buffer_id;
            }
        }
    }

    // If we got here, either the pool is full or there was an error creating the buffer
    return -1;
}


// Return a buffer to the pool (mark as not in use)
void GPUBufferPools::returnBuffer(SDL_GPUBufferUsageFlags type, int32_t buffer_id) {
    // Determine which pool to use based on buffer type
    BufferPool* pool = nullptr;
    SDL_GPUBuffer** resource_buffers = nullptr;
    uint32_t resource_count = 0;

    if (type == SDL_GPU_BUFFERUSAGE_VERTEX) {
        pool = &vertex_pool;
    }
    else if (type == SDL_GPU_BUFFERUSAGE_INDEX) {
        pool = &index_pool;
    }
    else if (type == (SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ)) {
        pool = &storage_pool;
    }
    else if (type == SDL_GPU_BUFFERUSAGE_VERTEX) { // For instance buffers
        pool = &instance_pool;
    }
    else {
        // Unknown buffer type
        return;
    }

    // Find the buffer in the pool and mark it as not in use
    for (uint32_t i = 0; i < pool->count; ++i) {
        if (i < pool->count && pool->buffers[i] == pool->buffers[i]) {
            pool->in_use_flags[i] = false;
            break;
        }
    }
}

// Destroy all buffer pools
void GPUBufferPools::destroyPools() {
    // Destroy vertex buffers
    for (uint32_t i = 0; i < vertex_pool.count; ++i) {
        if (vertex_pool.buffers[i]) {
            SDL_ReleaseGPUBuffer(device, vertex_pool.buffers[i]);
            vertex_pool.buffers[i] = nullptr;
        }
    }
    vertex_pool.count = 0;

    // Destroy index buffers
    for (uint32_t i = 0; i < index_pool.count; ++i) {
        if (index_pool.buffers[i]) {
            SDL_ReleaseGPUBuffer(device, index_pool.buffers[i]);
            index_pool.buffers[i] = nullptr;
        }
    }
    index_pool.count = 0;


    // Destroy storage buffers
    for (uint32_t i = 0; i < storage_pool.count; ++i) {
        if (storage_pool.buffers[i]) {
            SDL_ReleaseGPUBuffer(device, storage_pool.buffers[i]);
            storage_pool.buffers[i] = nullptr;
        }
    }
    storage_pool.count = 0;

    // Destroy instance buffers
    for (uint32_t i = 0; i < instance_pool.count; ++i) {
        if (instance_pool.buffers[i]) {
            SDL_ReleaseGPUBuffer(device, instance_pool.buffers[i]);
            instance_pool.buffers[i] = nullptr;
        }
    }
    instance_pool.count = 0;
}