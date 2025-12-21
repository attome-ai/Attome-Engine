#include <engine3D/ATMLog.h>
#include <engine3D/ATMPipeline.h>

    // Helper function to create a shader
    SDL_GPUShader* createShader(SDL_GPUDevice* device, const char* filePath, SDL_GPUShaderStage stage, const char* entrypoint, int uni = 1)
    {
        // Check if the file exists and get its info
        SDL_PathInfo pathInfo;
        if (SDL_GetPathInfo(filePath, &pathInfo) < 0) {
            ATMLOG("Failed to find shader file: %s - %s", filePath, SDL_GetError());
            return nullptr;
        }

        // Verify it's a regular file
        if (pathInfo.type != SDL_PATHTYPE_FILE) {
            ATMLOG("Path is not a file: %s", filePath);
            return nullptr;
        }

        // Get file size from the path info
        size_t fileSize = pathInfo.size;

        // Allocate memory for the shader code
        Uint8* shaderCode = new Uint8[fileSize + 1];
        if (!shaderCode) {
            ATMLOG("Failed to allocate memory for shader code");
            return nullptr;
        }

        // Open the file using SDL's RWops
        SDL_IOStream* file = SDL_IOFromFile(filePath, "rb");

        ATMLOGC(!file,"Failed to open shader file: %s - %s", filePath, SDL_GetError())


        // Read the shader code from the file
        size_t bytesRead = SDL_ReadIO(file, shaderCode, fileSize);
        SDL_CloseIO(file);

        ATMLOGC(bytesRead != fileSize, "Failed to read shader file: %s - %s", filePath, SDL_GetError());


        // Null-terminate if it's a text shader (like GLSL)
        // For binary formats like SPIR-V, this won't matter
        shaderCode[fileSize] = '\0';

        // Set up shader info
        SDL_GPUShaderCreateInfo shaderInfo = { 0 };
        shaderInfo.code = shaderCode;
        shaderInfo.code_size = fileSize;
        shaderInfo.entrypoint = entrypoint;
        shaderInfo.format = SDL_GPU_SHADERFORMAT_SPIRV;
        shaderInfo.stage = stage;
        shaderInfo.num_uniform_buffers = uni;
        


        // Create the shader
        SDL_GPUShader* shader = SDL_CreateGPUShader(device, &shaderInfo);

        // Free the shader code memory
        delete[] shaderCode;

        ATMLOGC(!shader, "Failed to create shader from file %s: %s", filePath, SDL_GetError());
        ATMLOGS(shader, "Successfully loaded shader from %s", filePath);

        return shader;
    }
    SDL_GPUGraphicsPipeline* createBasicPipeline(SDL_GPUDevice* device) {


        // Define a standard vertex with position, normal, texcoord
        struct Vertex {
            float position[3];
            float normal[3];
            float texCoord[2];
        };

        // Define vertex buffer description
        SDL_GPUVertexBufferDescription vertexBufferDesc = { 0 };
        vertexBufferDesc.slot = 0;
        vertexBufferDesc.pitch = sizeof(Vertex);
        vertexBufferDesc.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;

        // Define vertex attributes
        static SDL_GPUVertexAttribute vertexAttributes[3];
        
        // Position attribute
        vertexAttributes[0].location = 0;
        vertexAttributes[0].buffer_slot = 0;
        vertexAttributes[0].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3;
        vertexAttributes[0].offset = offsetof(Vertex, position);

        // Normal attribute
        vertexAttributes[1].location = 1;
        vertexAttributes[1].buffer_slot = 0;
        vertexAttributes[1].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3;
        vertexAttributes[1].offset = offsetof(Vertex, normal);

        // TexCoord attribute
        vertexAttributes[2].location = 2;
        vertexAttributes[2].buffer_slot = 0;
        vertexAttributes[2].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2;
        vertexAttributes[2].offset = offsetof(Vertex, texCoord);

        // Create vertex input state
        SDL_GPUVertexInputState inputState = { 0 };
        inputState.vertex_buffer_descriptions = &vertexBufferDesc;
        inputState.num_vertex_buffers = 1;
        inputState.vertex_attributes = vertexAttributes;
        inputState.num_vertex_attributes = 3;

        SDL_GPUGraphicsPipeline* dummyPipeline = {}; // This will be empty

        

        // Create shaders
        SDL_GPUShader* vertexShader = createShader(device, "C:/Users/Custom_PC_bh/Downloads/everything/Template/shaders/basic_vert.spv", SDL_GPU_SHADERSTAGE_VERTEX,  "main",0);
        SDL_GPUShader* fragmentShader = createShader(device, "C:/Users/Custom_PC_bh/Downloads/everything/Template/shaders/basic_frag.spv", SDL_GPU_SHADERSTAGE_FRAGMENT, "main",0);

        if (!vertexShader || !fragmentShader) {
            if (vertexShader) SDL_ReleaseGPUShader(device, vertexShader);
            if (fragmentShader) SDL_ReleaseGPUShader(device, fragmentShader);
            return dummyPipeline;
        }


        // Define color target description
        SDL_GPUColorTargetDescription colorTargetDesc = { };

        colorTargetDesc.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
        colorTargetDesc.blend_state.enable_blend = false;
        colorTargetDesc.blend_state.color_write_mask = SDL_GPU_COLORCOMPONENT_R |
            SDL_GPU_COLORCOMPONENT_G |
            SDL_GPU_COLORCOMPONENT_B |
            SDL_GPU_COLORCOMPONENT_A;

        // Define graphics pipeline target info
        SDL_GPUGraphicsPipelineTargetInfo targetInfo = { };
        targetInfo.color_target_descriptions = &colorTargetDesc;
        targetInfo.num_color_targets = 1;
        targetInfo.has_depth_stencil_target = true;
        targetInfo.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT;

        // Define rasterizer state
        SDL_GPURasterizerState rasterizerState = {  };
        rasterizerState.fill_mode = SDL_GPU_FILLMODE_FILL;
        rasterizerState.cull_mode = SDL_GPU_CULLMODE_BACK;
        rasterizerState.front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE;
        rasterizerState.enable_depth_clip = true;

        // Define multisample state
        SDL_GPUMultisampleState multisampleState = {  };
        multisampleState.sample_count = SDL_GPU_SAMPLECOUNT_1;
        multisampleState.sample_mask = 0;

        // Define depth stencil state
        SDL_GPUDepthStencilState depthStencilState = {  };
        depthStencilState.enable_depth_test = true;
        depthStencilState.enable_depth_write = true;
        depthStencilState.enable_stencil_test = false;
        depthStencilState.compare_op = SDL_GPU_COMPAREOP_LESS;

        // Create pipeline info
        SDL_GPUGraphicsPipelineCreateInfo pipelineInfo = { 0 };
        pipelineInfo.vertex_shader = vertexShader;
        pipelineInfo.fragment_shader = fragmentShader;
        pipelineInfo.vertex_input_state = inputState;
        pipelineInfo.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
        pipelineInfo.rasterizer_state = rasterizerState;
        pipelineInfo.multisample_state = multisampleState;
        pipelineInfo.depth_stencil_state = depthStencilState;
        pipelineInfo.target_info = targetInfo;

        
        // Create the pipeline
        SDL_GPUGraphicsPipeline* pipelinePtr = SDL_CreateGPUGraphicsPipeline(device, &pipelineInfo);

        // Release shaders as they're no longer needed after pipeline creation
        SDL_ReleaseGPUShader(device, vertexShader);
        SDL_ReleaseGPUShader(device, fragmentShader);

        ATMLOGC(!pipelinePtr, "Failed to create graphics pipeline: %s", SDL_GetError());



        return pipelinePtr;
    }

    // Create multiple pipelines with different configurations
    std::vector<SDL_GPUGraphicsPipeline*> createPipelines(SDL_GPUDevice* device)
    {
        std::vector<SDL_GPUGraphicsPipeline*> pipelines;
        ATMLOGC(!device, "Invalid device for pipeline creation");
        pipelines.push_back(createBasicPipeline(device));
        return pipelines;
    }

    // Utility to cleanup pipelines
    void releasePipelines(SDL_GPUDevice* device, std::vector<SDL_GPUGraphicsPipeline*>& pipelines) {
        for (auto pipeline : pipelines) {
            SDL_ReleaseGPUGraphicsPipeline(device, pipeline);
        }
        pipelines.clear();
    }

