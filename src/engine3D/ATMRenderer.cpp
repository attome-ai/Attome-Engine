#include <engine3D/ATMEngine.h>
// GPURenderer constructor implementation
GPURenderer::GPURenderer(SDL_GPUDevice* device, SDL_Window* window, uint32_t command_buffer_count)
    : device(device), window(window), command_buffer_count(command_buffer_count) {

    // Initialize command buffers
    command_buffers = nullptr;
    current_command_buffer = 0;

    // Initialize render pass
    main_render_pass = nullptr;
    swapchain_texture = nullptr;

    // Initialize frame fences
    frame_fences = nullptr;
    frame_fence_count = command_buffer_count;
    current_frame_fence = 0;

    // Initialize state cache values
    current_shader_id = -1;
    current_mesh_id = -1;
    current_material_id = -1;
    current_vertex_buffer_id = -1;
    current_index_buffer_id = -1;
    for (uint32_t i = 0; i < current_texture_ids.size(); ++i) {
        current_texture_ids[i] = -1;
        current_sampler_ids[i] = -1;
    }

    // Initialize render commands
    render_commands = nullptr;
    render_command_count = 0;
    max_render_commands = 0;

    // Initialize global uniform buffer
    global_uniform_buffer = -1;

    // Initialize global uniforms
    uniforms.view_matrix = glm::mat4(1.0f);
    uniforms.projection_matrix = glm::mat4(1.0f);
    uniforms.camera_position = glm::vec4(0.0f);
    uniforms.time = 0.0f;
    uniforms.delta_time = 0.0f;
}

// GPURenderer destructor implementation
GPURenderer::~GPURenderer() {
    shutdown();
}

// Initialize the renderer
bool GPURenderer::initialize() {


    // Create command buffers
    command_buffers = (SDL_GPUCommandBuffer**)SDL_calloc(command_buffer_count, sizeof(SDL_GPUCommandBuffer*));
    if (!command_buffers) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to allocate command buffer array");
        return false;
    }

    for (uint32_t i = 0; i < command_buffer_count; ++i) {
        command_buffers[i] = SDL_AcquireGPUCommandBuffer(device);
        if (!command_buffers[i]) {
            SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to acquire command buffer %u", i);
            return false;
        }
    }

    // Create frame fences
    frame_fences = (SDL_GPUFence**)SDL_calloc(frame_fence_count, sizeof(SDL_GPUFence*));
    if (!frame_fences) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to allocate frame fence array");
        return false;
    }

    // Allocate render commands
    max_render_commands = MAX_RENDER_BATCHES;
    render_commands = (RenderCommand*)SDL_calloc(max_render_commands, sizeof(RenderCommand));
    if (!render_commands) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to allocate render commands");
        return false;
    }

    // Set up the render pass
    setupRenderPass();

    return true;
}

// Set up the render pass
void GPURenderer::setupRenderPass() {
    // We don't create the render pass here as it will be created per-frame
    // with SDL_BeginGPURenderPass when we have the swapchain texture
}

// Begin the frame
void GPURenderer::beginFrame() {
    // Wait for the previous frame's fence if it exists
    if (frame_fences[current_frame_fence]) {
        SDL_WaitForGPUFences(device, true, &frame_fences[current_frame_fence], 1);
        SDL_ReleaseGPUFence(device, frame_fences[current_frame_fence]);
        frame_fences[current_frame_fence] = nullptr;
    }

    // Clear the render commands
    clearRenderCommands();

    // Reset the state cache
    current_shader_id = -1;
    current_mesh_id = -1;
    current_material_id = -1;
    current_vertex_buffer_id = -1;
    current_index_buffer_id = -1;
    for (uint32_t i = 0; i < current_texture_ids.size(); ++i) {
        current_texture_ids[i] = -1;
        current_sampler_ids[i] = -1;
    }

    uniforms.time += uniforms.delta_time;

}

// End the frame
void GPURenderer::endFrame() {
    // Submit the command buffer and get a fence
    frame_fences[current_frame_fence] = SDL_SubmitGPUCommandBufferAndAcquireFence(
        command_buffers[current_command_buffer]);

    // Advance to the next command buffer and fence
    current_command_buffer = (current_command_buffer + 1) % command_buffer_count;
    current_frame_fence = (current_frame_fence + 1) % frame_fence_count;
}

// Begin the render pass
void GPURenderer::beginRenderPass() {

    
    // Acquire the swapchain texture for rendering
    uint32_t width, height;
    if (!SDL_WaitAndAcquireGPUSwapchainTexture(
        command_buffers[current_command_buffer],
        window,
        &swapchain_texture,
        &width,
        &height)) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to acquire swapchain texture");
        return;
    }

    // Set up color target info
    SDL_GPUColorTargetInfo colorTargetInfo = {};
    colorTargetInfo.texture = swapchain_texture;
    colorTargetInfo.mip_level = 0;
    colorTargetInfo.layer_or_depth_plane = 0;
    colorTargetInfo.clear_color = { 0.1f, 0.1f, 0.1f, 1.0f }; // Dark gray background
    colorTargetInfo.load_op = SDL_GPU_LOADOP_CLEAR;
    colorTargetInfo.store_op = SDL_GPU_STOREOP_STORE;
    colorTargetInfo.resolve_texture = nullptr;
    colorTargetInfo.cycle = false;
    colorTargetInfo.cycle_resolve_texture = false;

    // Setup depth stencil target info if needed
    // SDL_GPUDepthStencilTargetInfo depthStencilTargetInfo = {};
    // This would be populated if we had a depth-stencil texture

    // Begin the render pass
    main_render_pass = SDL_BeginGPURenderPass(
        command_buffers[current_command_buffer],
        &colorTargetInfo,
        1,
        nullptr);  // No depth-stencil target for now

    if (!main_render_pass) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to begin render pass");
        return;
    }

    // Set viewport to match the swapchain size
    SDL_GPUViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.w = (float)width;
    viewport.h = (float)height;
    viewport.min_depth = 0.0f;
    viewport.max_depth = 1.0f;
    SDL_SetGPUViewport(main_render_pass, &viewport);

    // Set scissor to match the swapchain size
    SDL_Rect scissor = { 0, 0, (int)width, (int)height };
    SDL_SetGPUScissor(main_render_pass, &scissor);
}

// End the render pass
void GPURenderer::endRenderPass() {
    if (main_render_pass) {
        SDL_EndGPURenderPass(main_render_pass);
        main_render_pass = nullptr;
    }
}

// Add a render command
void GPURenderer::addRenderCommand(const RenderCommand& command) {
    if (render_command_count < max_render_commands) {
            [render_command_count++] = command;
    }
}

// Clear all render commands
void GPURenderer::clearRenderCommands() {
    render_command_count = 0;
}

// Sort render commands for minimal state changes
void GPURenderer::sortRenderCommands() {
    if (render_command_count > 0) {
        // Sort by sort key (shader|mesh|material|transparent)
        std::sort(render_commands, render_commands + render_command_count,
            [](const RenderCommand& a, const RenderCommand& b) {
                return a.sort_key < b.sort_key;
            });
    }
}

// Execute all render commands
void GPURenderer::executeRenderCommands(const GPUResources* resources) {
    if (!main_render_pass) {
        return;
    }
    SDL_BindGPUGraphicsPipeline(main_render_pass,resources->pipelines[0]);

    for (uint32_t i = 0; i < render_command_count; ++i) {
        const RenderCommand& cmd = render_commands[i];

        // Bind the shader if changed
        bindShaderIfChanged(cmd.shader_id, resources);

        // Push uniform data to the command buffer
        // This must happen after binding the pipeline but before drawing
        // We use slot 0 to match the shader's layout(set = 0, binding = 0)
        SDL_PushGPUVertexUniformData(
            command_buffers[current_command_buffer],
            0, // slot index - matches binding = 0 in the shader
            &uniforms,
            sizeof(uniforms)
        );

        // Bind the mesh if changed
        bindMeshIfChanged(cmd.mesh_id, resources);

        // Bind the material if changed
        bindMaterialIfChanged(cmd.material_id, resources);

        // Draw instanced
        if (resources->meshes[cmd.mesh_id].index_count > 0) {
            SDL_DrawGPUIndexedPrimitives(
                main_render_pass,
                resources->meshes[cmd.mesh_id].index_count,
                cmd.instance_count,
                0,  // first_index
                0,  // vertex_offset
                cmd.first_instance
            );
        }
        else {
            SDL_DrawGPUPrimitives(
                main_render_pass,
                resources->meshes[cmd.mesh_id].vertex_count,
                cmd.instance_count,
                0,  // first_vertex
                cmd.first_instance
            );
        }
    }
}
// Bind shader if different from currently bound shader
void GPURenderer::bindShaderIfChanged(int32_t shader_id, const GPUResources* resources) {
    if (shader_id != current_shader_id && shader_id >= 0 && main_render_pass) {
        SDL_BindGPUGraphicsPipeline(main_render_pass, resources->pipelines[shader_id]);
        current_shader_id = shader_id;
    }
}

// Bind mesh if different from currently bound mesh
void GPURenderer::bindMeshIfChanged(int32_t mesh_id, const GPUResources* resources) {
    if (mesh_id >= 0 && main_render_pass) {
        const Mesh& mesh = resources->meshes[mesh_id];

        // Bind vertex buffer if changed
        if (mesh.vertex_buffer_id != current_vertex_buffer_id) {
            SDL_GPUBufferBinding vertexBinding = {
                mesh.vertex_buffer_ptr,
                0  // offset
            };
            SDL_BindGPUVertexBuffers(main_render_pass, 0, &vertexBinding, 1);
            current_vertex_buffer_id = mesh.vertex_buffer_id;
        }

        // Bind index buffer if changed
        if (mesh.index_buffer_id != current_index_buffer_id && mesh.index_buffer_ptr) {
            SDL_GPUBufferBinding indexBinding = {
                mesh.index_buffer_ptr,
                0  // offset
            };
            SDL_BindGPUIndexBuffer(main_render_pass, &indexBinding, SDL_GPU_INDEXELEMENTSIZE_32BIT);
            current_index_buffer_id = mesh.index_buffer_id;
        }
    }
}

// Bind material if different from currently bound material
void GPURenderer::bindMaterialIfChanged(int32_t material_id, const GPUResources* resources) {
    if (material_id >= 0 && material_id != current_material_id && main_render_pass) {
        const Material& material = resources->materials[material_id];

        // Prepare texture-sampler bindings for the material
        SDL_GPUTextureSamplerBinding bindings[3];  // For diffuse, normal, specular
        uint32_t bindingCount = 0;

        // Add diffuse texture if available
        if (material.diffuse_texture >= 0 && material.diffuse_texture_ptr) {
            bindings[bindingCount].texture = material.diffuse_texture_ptr;
            bindings[bindingCount].sampler = material.sampler_ptr;
            bindingCount++;
            current_texture_ids[0] = material.diffuse_texture;
        }

        // Add normal texture if available
        if (material.normal_texture >= 0 && material.normal_texture_ptr) {
            bindings[bindingCount].texture = material.normal_texture_ptr;
            bindings[bindingCount].sampler = material.sampler_ptr;
            bindingCount++;
            current_texture_ids[1] = material.normal_texture;
        }

        // Add specular texture if available
        if (material.specular_texture >= 0 && material.specular_texture_ptr) {
            bindings[bindingCount].texture = material.specular_texture_ptr;
            bindings[bindingCount].sampler = material.sampler_ptr;
            bindingCount++;
            current_texture_ids[2] = material.specular_texture;
        }

        // Bind textures to fragment shader
        if (bindingCount > 0) {
            SDL_BindGPUFragmentSamplers(main_render_pass, 0, bindings, bindingCount);
        }

        current_material_id = material_id;
        current_sampler_ids[0] = material.sampler_id;
    }
}



// Shutdown the renderer
void GPURenderer::shutdown() {
    // Release frame fences
    if (frame_fences) {
        for (uint32_t i = 0; i < frame_fence_count; ++i) {
            if (frame_fences[i]) {
                SDL_ReleaseGPUFence(device, frame_fences[i]);
                frame_fences[i] = nullptr;
            }
        }
        SDL_free(frame_fences);
        frame_fences = nullptr;
    }

    // Command buffers are acquired and released per frame
    // We don't need to explicitly release them
    if (command_buffers) {
        SDL_free(command_buffers);
        command_buffers = nullptr;
    }

    // Free render commands
    if (render_commands) {
        SDL_free(render_commands);
        render_commands = nullptr;
    }

    // Wait for the device to be idle before shutting down
    SDL_WaitForGPUIdle(device);
}