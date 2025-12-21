#include <engine3D/ATMEngine.h>

#include <iostream>
// Constructor
GPUResources::GPUResources(SDL_GPUDevice* device, uint32_t max_textures, uint32_t max_buffers,
    uint32_t max_pipeline, uint32_t max_meshes, uint32_t max_materials,
    uint32_t max_samplers) : device(device) {

    // Set maximum capacities for all resources
    this->max_textures = max_textures;
    this->max_vertex_buffers = max_buffers;
    this->max_index_buffers = max_buffers;
    this->max_storage_buffers = max_buffers;
    this->max_instance_buffers = max_buffers;
    this->max_samplers = max_samplers;
    this->max_pipeline = max_pipeline;
    this->max_meshes = max_meshes;
    this->max_materials = max_materials;

    // Initialize resource counts
    texture_count = 0;
    vertex_buffer_count = 0;
    index_buffer_count = 0;
    storage_buffer_count = 0;
    instance_buffer_count = 0;
    sampler_count = 0;
    pipeline_count = 0;
    mesh_count = 0;
    material_count = 0;

    // Allocate arrays for fast indexing using SDL memory allocation
    textures = (SDL_GPUTexture**)SDL_calloc(max_textures, sizeof(SDL_GPUTexture*));
    vertex_buffers = (SDL_GPUBuffer**)SDL_calloc(max_vertex_buffers, sizeof(SDL_GPUBuffer*));
    index_buffers = (SDL_GPUBuffer**)SDL_calloc(max_index_buffers, sizeof(SDL_GPUBuffer*));
    storage_buffers = (SDL_GPUBuffer**)SDL_calloc(max_storage_buffers, sizeof(SDL_GPUBuffer*));
    instance_buffers = (SDL_GPUBuffer**)SDL_calloc(max_instance_buffers, sizeof(SDL_GPUBuffer*));
    samplers = (SDL_GPUSampler**)SDL_calloc(max_samplers, sizeof(SDL_GPUSampler*));

    // Allocate shader, mesh, and material arrays using SDL memory allocation
    pipelines = (SDL_GPUGraphicsPipeline**)SDL_calloc(max_pipeline, sizeof(pipelines));
    meshes = (Mesh*)SDL_calloc(max_meshes, sizeof(Mesh));
    materials = (Material*)SDL_calloc(max_materials, sizeof(Material));

}

// Destructor
GPUResources::~GPUResources() {
    // Clean up all resources
    destroyResources();



    // Free all arrays using SDL_free
    SDL_free(textures);
    SDL_free(vertex_buffers);
    SDL_free(index_buffers);
    SDL_free(storage_buffers);
    SDL_free(instance_buffers);
    SDL_free(samplers);
    SDL_free(pipelines);
    SDL_free(meshes);
    SDL_free(materials);
}

// Cleanup all allocated resources
void GPUResources::destroyResources() {
    // Release all textures
    for (uint32_t i = 0; i < texture_count; ++i) {
        if (textures[i]) {
            SDL_ReleaseGPUTexture(device, textures[i]);
            textures[i] = nullptr;
        }
    }

    // Release all vertex buffers
    for (uint32_t i = 0; i < vertex_buffer_count; ++i) {
        if (vertex_buffers[i]) {
            SDL_ReleaseGPUBuffer(device, vertex_buffers[i]);
            vertex_buffers[i] = nullptr;
        }
    }

    // Release all index buffers
    for (uint32_t i = 0; i < index_buffer_count; ++i) {
        if (index_buffers[i]) {
            SDL_ReleaseGPUBuffer(device, index_buffers[i]);
            index_buffers[i] = nullptr;
        }
    }



    // Release all storage buffers
    for (uint32_t i = 0; i < storage_buffer_count; ++i) {
        if (storage_buffers[i]) {
            SDL_ReleaseGPUBuffer(device, storage_buffers[i]);
            storage_buffers[i] = nullptr;
        }
    }

    // Release all instance buffers
    for (uint32_t i = 0; i < instance_buffer_count; ++i) {
        if (instance_buffers[i]) {
            SDL_ReleaseGPUBuffer(device, instance_buffers[i]);
            instance_buffers[i] = nullptr;
        }
    }

    // Release all samplers
    for (uint32_t i = 0; i < sampler_count; ++i) {
        if (samplers[i]) {
            SDL_ReleaseGPUSampler(device, samplers[i]);
            samplers[i] = nullptr;
        }
    }

    // Release all shader resources
    for (uint32_t i = 0; i < pipeline_count; ++i) 
    {
        SDL_ReleaseGPUGraphicsPipeline(device, pipelines[i]);
        pipelines[i] = nullptr;

    }

    // Reset all counters
    texture_count = 0;
    vertex_buffer_count = 0;
    index_buffer_count = 0;
    storage_buffer_count = 0;
    instance_buffer_count = 0;
    sampler_count = 0;
    pipeline_count = 0;
    mesh_count = 0;
    material_count = 0;
}

// Create a texture from an SDL_Surface
int32_t GPUResources::createTexture(SDL_Surface* surface) {
    if (!surface || texture_count >= max_textures) {
        return -1;
    }

    // Get the surface format
    SDL_GPUTextureFormat format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;

    auto detail = SDL_GetPixelFormatDetails(surface->format);
    // Determine format based on surface details
    if (detail->bytes_per_pixel == 4) {
        if (detail->Rmask == 0xff000000 && detail->Amask == 0x000000ff) {
            format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM; // ABGR
        }
        else if (detail->Rmask == 0x000000ff && detail->Amask == 0xff000000) {
            format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM; // RGBA
        }
        else if (detail->Rmask == 0x00ff0000 && detail->Amask == 0x000000ff) {
            format = SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM; // BGRA
        }
    }
    else if (detail->bytes_per_pixel == 3) {
        // Note: SDL3 doesn't have exact RGB/BGR formats, closest approximation
        format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    }

    // Create texture description
    SDL_GPUTextureCreateInfo createInfo;
    SDL_memset(&createInfo, 0, sizeof(createInfo));
    createInfo.type = SDL_GPU_TEXTURETYPE_2D;
    createInfo.width = surface->w;
    createInfo.height = surface->h;
    createInfo.layer_count_or_depth = 1;
    createInfo.format = format;
    createInfo.num_levels = 0; // Generate full mipmap chain
    createInfo.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
    createInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
    createInfo.props = 0;

    // Create the texture
    SDL_GPUTexture* texture = SDL_CreateGPUTexture(device, &createInfo);
    if (!texture) {
        return -1;
    }

    // Upload texture data - prepare transfer buffer
    SDL_GPUTransferBufferCreateInfo transferInfo;
    SDL_memset(&transferInfo, 0, sizeof(transferInfo));
    transferInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    transferInfo.size = surface->pitch * surface->h;
    transferInfo.props = 0;

    SDL_GPUTransferBuffer* transfer = SDL_CreateGPUTransferBuffer(device, &transferInfo);
    if (!transfer) {
        SDL_ReleaseGPUTexture(device, texture);
        return -1;
    }

    // Copy data to transfer buffer
    void* mapped_data = SDL_MapGPUTransferBuffer(device, transfer, false);
    if (mapped_data) {
        SDL_memcpy(mapped_data, surface->pixels, surface->pitch * surface->h);
        SDL_UnmapGPUTransferBuffer(device, transfer);

        // Get command buffer for transfer
        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);
        if (cmd) {
            // Begin copy pass
            SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
            if (copy) {
                // Setup transfer info
                SDL_GPUTextureTransferInfo transferInfo;
                SDL_memset(&transferInfo, 0, sizeof(transferInfo));
                transferInfo.transfer_buffer = transfer;
                transferInfo.offset = 0;
                transferInfo.pixels_per_row = surface->w;
                transferInfo.rows_per_layer = surface->h;

                // Setup destination region
                SDL_GPUTextureRegion region;
                SDL_memset(&region, 0, sizeof(region));
                region.texture = texture;
                region.mip_level = 0;
                region.layer = 0;
                region.x = 0;
                region.y = 0;
                region.z = 0;
                region.w = surface->w;
                region.h = surface->h;
                region.d = 1;

                // Upload data
                SDL_UploadToGPUTexture(copy, &transferInfo, &region, false);
                SDL_EndGPUCopyPass(copy);

                // Generate mipmaps if requested
                SDL_GenerateMipmapsForGPUTexture(cmd, texture);

                // Submit command buffer
                SDL_SubmitGPUCommandBuffer(cmd);
            }
        }

        // Release transfer buffer
        SDL_ReleaseGPUTransferBuffer(device, transfer);
    }
    else {
        SDL_ReleaseGPUTexture(device, texture);
        SDL_ReleaseGPUTransferBuffer(device, transfer);
        return -1;
    }

    // Store the texture
    uint32_t texture_id = texture_count++;
    textures[texture_id] = texture;

    return static_cast<int32_t>(texture_id);
}

// Create a vertex buffer
int32_t GPUResources::createVertexBuffer(const void* data, size_t size) {
    if (!data || size == 0 || vertex_buffer_count >= max_vertex_buffers) {
        return -1;
    }

    // Create buffer description
    SDL_GPUBufferCreateInfo createInfo;
    SDL_memset(&createInfo, 0, sizeof(createInfo));
    createInfo.size = size;
    createInfo.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
    createInfo.props = 0;

    // Create the buffer
    SDL_GPUBuffer* buffer = SDL_CreateGPUBuffer(device, &createInfo);
    if (!buffer) {
        return -1;
    }

    // Create a transfer buffer for the upload
    SDL_GPUTransferBufferCreateInfo transferInfo;
    SDL_memset(&transferInfo, 0, sizeof(transferInfo));
    transferInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    transferInfo.size = size;
    transferInfo.props = 0;

    SDL_GPUTransferBuffer* transfer = SDL_CreateGPUTransferBuffer(device, &transferInfo);
    if (!transfer) {
        SDL_ReleaseGPUBuffer(device, buffer);
        return -1;
    }

    // Map transfer buffer and copy data
    void* mapped_data = SDL_MapGPUTransferBuffer(device, transfer, false);
    if (mapped_data) {
        SDL_memcpy(mapped_data, data, size);
        SDL_UnmapGPUTransferBuffer(device, transfer);

        // Get command buffer for transfer
        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);
        if (cmd) {
            // Begin copy pass
            SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
            if (copy) {
                // Setup source location
                SDL_GPUTransferBufferLocation src;
                src.transfer_buffer = transfer;
                src.offset = 0;

                // Setup destination region
                SDL_GPUBufferRegion dst;
                dst.buffer = buffer;
                dst.offset = 0;
                dst.size = size;

                // Upload data
                SDL_UploadToGPUBuffer(copy, &src, &dst, false);
                SDL_EndGPUCopyPass(copy);

                // Submit command buffer
                SDL_SubmitGPUCommandBuffer(cmd);
            }
        }

        // Release transfer buffer
        SDL_ReleaseGPUTransferBuffer(device, transfer);
    }
    else {
        SDL_ReleaseGPUBuffer(device, buffer);
        SDL_ReleaseGPUTransferBuffer(device, transfer);
        return -1;
    }

    // Store the buffer
    uint32_t buffer_id = vertex_buffer_count++;
    vertex_buffers[buffer_id] = buffer;

    return static_cast<int32_t>(buffer_id);
}

// Create an index buffer
int32_t GPUResources::createIndexBuffer(const void* data, size_t size) {
    if (!data || size == 0 || index_buffer_count >= max_index_buffers) {
        return -1;
    }

    // Create buffer description
    SDL_GPUBufferCreateInfo createInfo;
    SDL_memset(&createInfo, 0, sizeof(createInfo));
    createInfo.size = size;
    createInfo.usage = SDL_GPU_BUFFERUSAGE_INDEX;
    createInfo.props = 0;

    // Create the buffer
    SDL_GPUBuffer* buffer = SDL_CreateGPUBuffer(device, &createInfo);
    if (!buffer) {
        return -1;
    }

    // Create a transfer buffer for the upload
    SDL_GPUTransferBufferCreateInfo transferInfo;
    SDL_memset(&transferInfo, 0, sizeof(transferInfo));
    transferInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    transferInfo.size = size;
    transferInfo.props = 0;

    SDL_GPUTransferBuffer* transfer = SDL_CreateGPUTransferBuffer(device, &transferInfo);
    if (!transfer) {
        SDL_ReleaseGPUBuffer(device, buffer);
        return -1;
    }

    // Map transfer buffer and copy data
    void* mapped_data = SDL_MapGPUTransferBuffer(device, transfer, false);
    if (mapped_data) {
        SDL_memcpy(mapped_data, data, size);
        SDL_UnmapGPUTransferBuffer(device, transfer);

        // Get command buffer for transfer
        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);
        if (cmd) {
            // Begin copy pass
            SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
            if (copy) {
                // Setup source location
                SDL_GPUTransferBufferLocation src;
                src.transfer_buffer = transfer;
                src.offset = 0;

                // Setup destination region
                SDL_GPUBufferRegion dst;
                dst.buffer = buffer;
                dst.offset = 0;
                dst.size = size;

                // Upload data
                SDL_UploadToGPUBuffer(copy, &src, &dst, false);
                SDL_EndGPUCopyPass(copy);

                // Submit command buffer
                SDL_SubmitGPUCommandBuffer(cmd);
            }
        }

        // Release transfer buffer
        SDL_ReleaseGPUTransferBuffer(device, transfer);
    }
    else {
        SDL_ReleaseGPUBuffer(device, buffer);
        SDL_ReleaseGPUTransferBuffer(device, transfer);
        return -1;
    }

    // Store the buffer
    uint32_t buffer_id = index_buffer_count++;
    index_buffers[buffer_id] = buffer;

    return static_cast<int32_t>(buffer_id);
}


// Create a storage buffer
int32_t GPUResources::createStorageBuffer(const void* data, size_t size) {
    if (storage_buffer_count >= max_storage_buffers) {
        return -1;
    }

    // Create buffer description
    SDL_GPUBufferCreateInfo createInfo;
    SDL_memset(&createInfo, 0, sizeof(createInfo));
    createInfo.size = size;
    createInfo.usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE;
    createInfo.props = 0;

    // Create the buffer
    SDL_GPUBuffer* buffer = SDL_CreateGPUBuffer(device, &createInfo);
    if (!buffer) {
        return -1;
    }

    // Upload initial data if provided
    if (data && size > 0) {
        // Create a transfer buffer for the upload
        SDL_GPUTransferBufferCreateInfo transferInfo;
        SDL_memset(&transferInfo, 0, sizeof(transferInfo));
        transferInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        transferInfo.size = size;
        transferInfo.props = 0;

        SDL_GPUTransferBuffer* transfer = SDL_CreateGPUTransferBuffer(device, &transferInfo);
        if (transfer) {
            // Map transfer buffer and copy data
            void* mapped_data = SDL_MapGPUTransferBuffer(device, transfer, false);
            if (mapped_data) {
                SDL_memcpy(mapped_data, data, size);
                SDL_UnmapGPUTransferBuffer(device, transfer);

                // Get command buffer for transfer
                SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);
                if (cmd) {
                    // Begin copy pass
                    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
                    if (copy) {
                        // Setup source location
                        SDL_GPUTransferBufferLocation src;
                        src.transfer_buffer = transfer;
                        src.offset = 0;

                        // Setup destination region
                        SDL_GPUBufferRegion dst;
                        dst.buffer = buffer;
                        dst.offset = 0;
                        dst.size = size;

                        // Upload data
                        SDL_UploadToGPUBuffer(copy, &src, &dst, false);
                        SDL_EndGPUCopyPass(copy);

                        // Submit command buffer
                        SDL_SubmitGPUCommandBuffer(cmd);
                    }
                }
            }

            // Release transfer buffer
            SDL_ReleaseGPUTransferBuffer(device, transfer);
        }
    }

    // Store the buffer
    uint32_t buffer_id = storage_buffer_count++;
    storage_buffers[buffer_id] = buffer;

    return static_cast<int32_t>(buffer_id);
}

// Create an instance buffer
int32_t GPUResources::createInstanceBuffer(const void* data, size_t size) {
    if (instance_buffer_count >= max_instance_buffers) {
        return -1;
    }

    // Create buffer description
    SDL_GPUBufferCreateInfo createInfo;
    SDL_memset(&createInfo, 0, sizeof(createInfo));
    createInfo.size = size;
    createInfo.usage = SDL_GPU_BUFFERUSAGE_VERTEX; // Used as vertex buffer for instances
    createInfo.props = 0;

    // Create the buffer
    SDL_GPUBuffer* buffer = SDL_CreateGPUBuffer(device, &createInfo);
    if (!buffer) {
        return -1;
    }

    // Upload initial data if provided
    if (data && size > 0) {
        // Create a transfer buffer for the upload
        SDL_GPUTransferBufferCreateInfo transferInfo;
        SDL_memset(&transferInfo, 0, sizeof(transferInfo));
        transferInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        transferInfo.size = size;
        transferInfo.props = 0;

        SDL_GPUTransferBuffer* transfer = SDL_CreateGPUTransferBuffer(device, &transferInfo);
        if (transfer) {
            // Map transfer buffer and copy data
            void* mapped_data = SDL_MapGPUTransferBuffer(device, transfer, false);
            if (mapped_data) {
                SDL_memcpy(mapped_data, data, size);
                SDL_UnmapGPUTransferBuffer(device, transfer);

                // Get command buffer for transfer
                SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);
                if (cmd) {
                    // Begin copy pass
                    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
                    if (copy) {
                        // Setup source location
                        SDL_GPUTransferBufferLocation src;
                        src.transfer_buffer = transfer;
                        src.offset = 0;

                        // Setup destination region
                        SDL_GPUBufferRegion dst;
                        dst.buffer = buffer;
                        dst.offset = 0;
                        dst.size = size;

                        // Upload data
                        SDL_UploadToGPUBuffer(copy, &src, &dst, false);
                        SDL_EndGPUCopyPass(copy);

                        // Submit command buffer
                        SDL_SubmitGPUCommandBuffer(cmd);
                    }
                }
            }

            // Release transfer buffer
            SDL_ReleaseGPUTransferBuffer(device, transfer);
        }
    }

    // Store the buffer
    uint32_t buffer_id = instance_buffer_count++;
    instance_buffers[buffer_id] = buffer;

    return static_cast<int32_t>(buffer_id);
}

// Create a sampler
int32_t GPUResources::createSampler(SDL_GPUFilter min_filter, SDL_GPUFilter mag_filter,
    SDL_GPUSamplerAddressMode address_mode) {
    if (sampler_count >= max_samplers) {
        return -1;
    }

    // Create sampler description
    SDL_GPUSamplerCreateInfo createInfo;
    SDL_memset(&createInfo, 0, sizeof(createInfo));
    createInfo.min_filter = min_filter;
    createInfo.mag_filter = mag_filter;
    createInfo.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
    createInfo.address_mode_u = address_mode;
    createInfo.address_mode_v = address_mode;
    createInfo.address_mode_w = address_mode;
    createInfo.mip_lod_bias = 0.0f;
    createInfo.enable_anisotropy = true;
    createInfo.max_anisotropy = 16.0f;  // High quality filtering
    createInfo.enable_compare = false;
    createInfo.compare_op = SDL_GPU_COMPAREOP_NEVER;
    createInfo.min_lod = 0.0f;
    createInfo.max_lod = 1000.0f;
    createInfo.props = 0;

    // Create the sampler
    SDL_GPUSampler* sampler = SDL_CreateGPUSampler(device, &createInfo);
    if (!sampler) {
        return -1;
    }

    // Store the sampler
    uint32_t sampler_id = sampler_count++;
    samplers[sampler_id] = sampler;

    return static_cast<int32_t>(sampler_id);
}

// Create a mesh from vertices and indices
int32_t GPUResources::createMesh(const Vertex* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t index_count) {
    if (!vertices || !indices || vertex_count == 0 || index_count == 0 || mesh_count >= max_meshes) {
        return -1;
    }

    // Create vertex buffer
    int32_t vb_id = createVertexBuffer(vertices, vertex_count * sizeof(Vertex));
    if (vb_id < 0) {
        return -1;
    }

    // Create index buffer
    int32_t ib_id = createIndexBuffer(indices, index_count * sizeof(uint32_t));
    if (ib_id < 0) {
        return -1;
    }

    // Create mesh
    uint32_t mesh_id = mesh_count++;

    // Initialize mesh data
    meshes[mesh_id].vertex_buffer_id = vb_id;
    meshes[mesh_id].index_buffer_id = ib_id;
    meshes[mesh_id].vertex_count = vertex_count;
    meshes[mesh_id].index_count = index_count;

    // Pre-cache pointers for faster access
    meshes[mesh_id].vertex_buffer_ptr = vertex_buffers[vb_id];
    meshes[mesh_id].index_buffer_ptr = index_buffers[ib_id];

    // Calculate mesh bounds
    AABB bounds;
    if (vertex_count > 0) {
        bounds.min = vertices[0].position;
        bounds.max = vertices[0].position;

        for (uint32_t i = 1; i < vertex_count; ++i) {
            bounds.min = glm::min(bounds.min, vertices[i].position);
            bounds.max = glm::max(bounds.max, vertices[i].position);
        }
    }
    meshes[mesh_id].bounds = bounds;

    // Initialize LOD data
    meshes[mesh_id].lod_meshes = nullptr;
    meshes[mesh_id].lod_count = 0;

    return static_cast<int32_t>(mesh_id);
}

// Create a material
int32_t GPUResources::createMaterial(int32_t diffuse_texture, int32_t normal_texture,
    int32_t specular_texture, int32_t sampler_id,
    const glm::vec4& diffuse_color, const glm::vec4& specular_color,
    float shininess) {
    if (material_count >= max_materials) {
        return -1;
    }

    uint32_t material_id = material_count++;

    // Set texture IDs
    materials[material_id].diffuse_texture = diffuse_texture;
    materials[material_id].normal_texture = normal_texture;
    materials[material_id].specular_texture = specular_texture;
    materials[material_id].sampler_id = sampler_id;

    // Set material properties
    materials[material_id].diffuse_color = diffuse_color;
    materials[material_id].specular_color = specular_color;
    materials[material_id].shininess = shininess;

    // Pre-cache pointers for faster access
    if (diffuse_texture >= 0 && diffuse_texture < static_cast<int32_t>(texture_count)) {
        materials[material_id].diffuse_texture_ptr = textures[diffuse_texture];
    }

    if (normal_texture >= 0 && normal_texture < static_cast<int32_t>(texture_count)) {
        materials[material_id].normal_texture_ptr = textures[normal_texture];
    }

    if (specular_texture >= 0 && specular_texture < static_cast<int32_t>(texture_count)) {
        materials[material_id].specular_texture_ptr = textures[specular_texture];
    }

    if (sampler_id >= 0 && sampler_id < static_cast<int32_t>(sampler_count)) {
        materials[material_id].sampler_ptr = samplers[sampler_id];
    }

    return static_cast<int32_t>(material_id);
}

int32_t GPUResources::addGraphicsPipeline(SDL_GPUGraphicsPipeline* pipeline) {
    // Get the current index before incrementing
    int32_t pipeline_id = pipeline_count;

    // Increment the count
    pipeline_count++;

    // Store the pipeline at the current index
    pipelines[pipeline_id] = pipeline;

    // Return the ID (index) of the added pipeline
    return pipeline_id;
}



// Update buffer data
void GPUResources::updateBuffer(int32_t buffer_id, const void* data, size_t size, size_t offset) {
    if (!data || size == 0) {
        return;
    }

    // Determine buffer type and update appropriate buffer
    SDL_GPUBuffer* buffer = nullptr;

    if (buffer_id >= 0 && buffer_id < static_cast<int32_t>(vertex_buffer_count)) {
        buffer = vertex_buffers[buffer_id];
    }
    else if (buffer_id >= 0 && buffer_id < static_cast<int32_t>(index_buffer_count)) {
        buffer = index_buffers[buffer_id];
    }

    else if (buffer_id >= 0 && buffer_id < static_cast<int32_t>(storage_buffer_count)) {
        buffer = storage_buffers[buffer_id];
    }
    else if (buffer_id >= 0 && buffer_id < static_cast<int32_t>(instance_buffer_count)) {
        buffer = instance_buffers[buffer_id];
    }
    else {
        return;
    }

    if (buffer) {
        // Create a transfer buffer for the update
        SDL_GPUTransferBufferCreateInfo transferInfo;
        SDL_memset(&transferInfo, 0, sizeof(transferInfo));
        transferInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        transferInfo.size = size;
        transferInfo.props = 0;

        SDL_GPUTransferBuffer* transfer = SDL_CreateGPUTransferBuffer(device, &transferInfo);
        if (transfer) {
            // Map transfer buffer and copy data
            void* mapped_data = SDL_MapGPUTransferBuffer(device, transfer, false);
            if (mapped_data) {
                SDL_memcpy(mapped_data, data, size);
                SDL_UnmapGPUTransferBuffer(device, transfer);

                // Get command buffer for transfer
                SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);
                if (cmd) {
                    // Begin copy pass
                    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
                    if (copy) {
                        // Setup source location
                        SDL_GPUTransferBufferLocation src;
                        src.transfer_buffer = transfer;
                        src.offset = 0;

                        // Setup destination region
                        SDL_GPUBufferRegion dst;
                        dst.buffer = buffer;
                        dst.offset = offset;
                        dst.size = size;

                        // Upload data
                        SDL_UploadToGPUBuffer(copy, &src, &dst, false);
                        SDL_EndGPUCopyPass(copy);

                        // Submit command buffer
                        SDL_SubmitGPUCommandBuffer(cmd);
                    }
                }
            }

            // Release transfer buffer
            SDL_ReleaseGPUTransferBuffer(device, transfer);
        }
    }
}

// Update pointers in all resources for direct access
void GPUResources::updatePointerCaches() {
    // Update material texture pointers
    for (uint32_t i = 0; i < material_count; ++i) {
        Material& material = materials[i];

        if (material.diffuse_texture >= 0 && material.diffuse_texture < static_cast<int32_t>(texture_count)) {
            material.diffuse_texture_ptr = textures[material.diffuse_texture];
        }
        else {
            material.diffuse_texture_ptr = nullptr;
        }

        if (material.normal_texture >= 0 && material.normal_texture < static_cast<int32_t>(texture_count)) {
            material.normal_texture_ptr = textures[material.normal_texture];
        }
        else {
            material.normal_texture_ptr = nullptr;
        }

        if (material.specular_texture >= 0 && material.specular_texture < static_cast<int32_t>(texture_count)) {
            material.specular_texture_ptr = textures[material.specular_texture];
        }
        else {
            material.specular_texture_ptr = nullptr;
        }

        if (material.sampler_id >= 0 && material.sampler_id < static_cast<int32_t>(sampler_count)) {
            material.sampler_ptr = samplers[material.sampler_id];
        }
        else {
            material.sampler_ptr = nullptr;
        }
    }

    // Update mesh buffer pointers
    for (uint32_t i = 0; i < mesh_count; ++i) {
        Mesh& mesh = meshes[i];

        if (mesh.vertex_buffer_id >= 0 && mesh.vertex_buffer_id < static_cast<int32_t>(vertex_buffer_count)) {
            mesh.vertex_buffer_ptr = vertex_buffers[mesh.vertex_buffer_id];
        }
        else {
            mesh.vertex_buffer_ptr = nullptr;
        }

        if (mesh.index_buffer_id >= 0 && mesh.index_buffer_id < static_cast<int32_t>(index_buffer_count)) {
            mesh.index_buffer_ptr = index_buffers[mesh.index_buffer_id];
        }
        else {
            mesh.index_buffer_ptr = nullptr;
        }

        // Update LOD mesh pointers
        for (uint32_t j = 0; j < mesh.lod_count; ++j) {
            Mesh::LODMesh& lod = mesh.lod_meshes[j];

            if (lod.vertex_buffer_id >= 0 && lod.vertex_buffer_id < static_cast<int32_t>(vertex_buffer_count)) {
                lod.vertex_buffer_ptr = vertex_buffers[lod.vertex_buffer_id];
            }
            else {
                lod.vertex_buffer_ptr = nullptr;
            }

            if (lod.index_buffer_id >= 0 && lod.index_buffer_id < static_cast<int32_t>(index_buffer_count)) {
                lod.index_buffer_ptr = index_buffers[lod.index_buffer_id];
            }
            else {
                lod.index_buffer_ptr = nullptr;
            }
        }
    }
}

