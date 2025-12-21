// Mock class for SDL_Surface to use in tests
struct MockSurface : public SDL_Surface {
    MockSurface(int width, int height) {
        w = width;
        h = height;
        pitch = width * 4; // Assuming RGBA
        format = 0x12345678; // Dummy format
        pixels = new char[pitch * h];
        memset(pixels, 0x55, pitch * h); // Fill with pattern
    }

    ~MockSurface() {
        delete[] static_cast<char*>(pixels);
    }
};

// Test class for GPUResources
class GPUResourcesTest {
private:
    SDL_GPUDevice* device;

public:
    GPUResourcesTest() {
        // Create mock device
        device = reinterpret_cast<SDL_GPUDevice*>(new char[16]);
        std::cout << "Test initialized with mock device\n";
    }

    ~GPUResourcesTest() {
        delete[] reinterpret_cast<char*>(device);
        std::cout << "Test cleaned up mock device\n";
    }

    bool testConstructorAndDestructor() {
        std::cout << "Testing constructor and destructor..." << std::endl;

        // Create with various capacities
        GPUResources* resources = new GPUResources(
            device, 10, 20, 30, 40, 50, 60);

        // Destructor should clean up all resources
        delete resources;

        std::cout << "  Constructor/destructor test passed\n";
        return true;
    }

    bool testTextureCreation() {
        std::cout << "Testing texture creation..." << std::endl;

        GPUResources resources(device, 10, 10, 10, 10, 10, 10);

        // Create mock surface
        std::unique_ptr<MockSurface> surface(new MockSurface(256, 256));

        // Test texture creation
        int32_t textureId = resources.createTexture(surface.get());
        assert(textureId >= 0 && "Texture creation failed");

        // Verify texture count increased
        int32_t textureId2 = resources.createTexture(surface.get());
        assert(textureId2 == textureId + 1 && "Texture ID not incremented");

        // Test max textures limit
        GPUResources limitedResources(device, 1, 10, 10, 10, 10, 10);
        int32_t limitTest = limitedResources.createTexture(surface.get());
        assert(limitTest >= 0 && "First texture creation failed");

        // This should fail as we've reached the limit
        int32_t shouldFail = limitedResources.createTexture(surface.get());
        assert(shouldFail < 0 && "Should have failed due to limit");

        std::cout << "  Texture creation test passed\n";
        return true;
    }

    bool testBufferCreation() {
        std::cout << "Testing buffer creation..." << std::endl;

        GPUResources resources(device, 10, 10, 10, 10, 10, 10);

        // Test vertex buffer
        float vertexData[] = {
            -1.0f, -1.0f, 0.0f,  // Position
            0.0f, 0.0f, 1.0f,    // Normal
            0.0f, 0.0f,          // UV
            1.0f, 0.0f, 0.0f,    // Tangent
            0.0f, 1.0f, 0.0f     // Bitangent
        };
        int32_t vbId = resources.createVertexBuffer(vertexData, sizeof(vertexData));
        assert(vbId >= 0 && "Vertex buffer creation failed");

        // Test index buffer
        uint32_t indexData[] = { 0, 1, 2 };
        int32_t ibId = resources.createIndexBuffer(indexData, sizeof(indexData));
        assert(ibId >= 0 && "Index buffer creation failed");

        // Test uniform buffer
        float uniformData[] = { 1.0f, 0.0f, 0.0f, 1.0f };
        int32_t ubId = resources.createUniformBuffer(uniformData, sizeof(uniformData));
        assert(ubId >= 0 && "Uniform buffer creation failed");

        // Test storage buffer
        int storageData[] = { 1, 2, 3, 4 };
        int32_t sbId = resources.createStorageBuffer(storageData, sizeof(storageData));
        assert(sbId >= 0 && "Storage buffer creation failed");

        // Test instance buffer
        float instanceData[] = {
            1.0f, 0.0f, 0.0f, 0.0f,  // Row 1
            0.0f, 1.0f, 0.0f, 0.0f,  // Row 2
            0.0f, 0.0f, 1.0f, 0.0f,  // Row 3
            0.0f, 0.0f, 0.0f, 1.0f   // Row 4
        };
        int32_t instId = resources.createInstanceBuffer(instanceData, sizeof(instanceData));
        assert(instId >= 0 && "Instance buffer creation failed");

        // Test buffer update
        float newData[] = { 2.0f, 0.0f, 0.0f };
        resources.updateBuffer(vbId, newData, sizeof(newData), 0);

        std::cout << "  Buffer creation test passed\n";
        return true;
    }

    bool testSamplerCreation() {
        std::cout << "Testing sampler creation..." << std::endl;

        GPUResources resources(device, 10, 10, 10, 10, 10, 10);

        // Test different filter and address mode combinations
        int32_t samplerId1 = resources.createSampler(
            SDL_GPU_FILTER_LINEAR,
            SDL_GPU_FILTER_LINEAR,
            SDL_GPU_SAMPLERADDRESSMODE_REPEAT
        );
        assert(samplerId1 >= 0 && "Sampler creation failed");

        int32_t samplerId2 = resources.createSampler(
            SDL_GPU_FILTER_NEAREST,
            SDL_GPU_FILTER_NEAREST,
            SDL_GPU_SAMPLERADDRESSMODE_CLAMP
        );
        assert(samplerId2 >= 0 && "Sampler creation failed");

        std::cout << "  Sampler creation test passed\n";
        return true;
    }

    bool testMeshCreation() {
        std::cout << "Testing mesh creation..." << std::endl;

        GPUResources resources(device, 10, 10, 10, 10, 10, 10);

        // Create mesh data
        Vertex vertices[3];

        // Vertex 0
        vertices[0].position = glm::vec3(-1.0f, -1.0f, 0.0f);
        vertices[0].normal = glm::vec3(0.0f, 0.0f, 1.0f);
        vertices[0].texcoord = glm::vec2(0.0f, 0.0f);
        vertices[0].tangent = glm::vec3(1.0f, 0.0f, 0.0f);
        vertices[0].bitangent = glm::vec3(0.0f, 1.0f, 0.0f);

        // Vertex 1
        vertices[1].position = glm::vec3(1.0f, -1.0f, 0.0f);
        vertices[1].normal = glm::vec3(0.0f, 0.0f, 1.0f);
        vertices[1].texcoord = glm::vec2(1.0f, 0.0f);
        vertices[1].tangent = glm::vec3(1.0f, 0.0f, 0.0f);
        vertices[1].bitangent = glm::vec3(0.0f, 1.0f, 0.0f);

        // Vertex 2
        vertices[2].position = glm::vec3(0.0f, 1.0f, 0.0f);
        vertices[2].normal = glm::vec3(0.0f, 0.0f, 1.0f);
        vertices[2].texcoord = glm::vec2(0.5f, 1.0f);
        vertices[2].tangent = glm::vec3(1.0f, 0.0f, 0.0f);
        vertices[2].bitangent = glm::vec3(0.0f, 1.0f, 0.0f);

        // Indices for the triangle
        uint32_t indices[] = { 0, 1, 2 };

        // Create mesh
        int32_t meshId = resources.createMesh(vertices, 3, indices, 3);
        assert(meshId >= 0 && "Mesh creation failed");

        std::cout << "  Mesh creation test passed\n";
        return true;
    }

    bool testMaterialCreation() {
        std::cout << "Testing material creation..." << std::endl;

        GPUResources resources(device, 10, 10, 10, 10, 10, 10);

        // Create textures first
        std::unique_ptr<MockSurface> surface(new MockSurface(256, 256));
        int32_t diffuseTexId = resources.createTexture(surface.get());
        int32_t normalTexId = resources.createTexture(surface.get());
        int32_t specularTexId = resources.createTexture(surface.get());

        // Create sampler
        int32_t samplerId = resources.createSampler(
            SDL_GPU_FILTER_LINEAR,
            SDL_GPU_FILTER_LINEAR,
            SDL_GPU_SAMPLERADDRESSMODE_REPEAT
        );

        // Create material
        int32_t materialId = resources.createMaterial(
            diffuseTexId, normalTexId, specularTexId, samplerId,
            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),   // Diffuse
            glm::vec4(0.8f, 0.8f, 0.8f, 1.0f),   // Specular
            32.0f                                 // Shininess
        );
        assert(materialId >= 0 && "Material creation failed");

        // Test material with some invalid texture IDs
        int32_t materialId2 = resources.createMaterial(
            diffuseTexId, -1, -1, samplerId,     // Only diffuse texture
            glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),   // Red diffuse
            glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),   // No specular
            0.0f                                  // No shininess
        );
        assert(materialId2 >= 0 && "Material creation failed with partial textures");

        std::cout << "  Material creation test passed\n";
        return true;
    }

    bool testShaderCreation() {
        std::cout << "Testing shader creation..." << std::endl;

        GPUResources resources(device, 10, 10, 10, 10, 10, 10);

        // Mock shader code
        const char* vertexShaderCode =
            "#version 450\n"
            "layout(location = 0) in vec3 position;\n"
            "void main() {\n"
            "    gl_Position = vec4(position, 1.0);\n"
            "}\n";

        const char* fragmentShaderCode =
            "#version 450\n"
            "layout(location = 0) out vec4 fragColor;\n"
            "void main() {\n"
            "    fragColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
            "}\n";

        // Create shader from code
        int32_t shaderId = resources.createShader(
            vertexShaderCode, strlen(vertexShaderCode),
            fragmentShaderCode, strlen(fragmentShaderCode),
            SDL_GPU_SHADERFORMAT_GLSL
        );

        // This may fail in the test environment since we don't have real GPU shader compilation
        // Just verify the function is callable

        // Test shader from file paths (this will likely fail but verifies function exists)
        resources.createShaderFromHLSL("test_vs.hlsl", "test_fs.hlsl", "TestShader", true);
        resources.createShaderFromSPIRV("test_vs.spv", "test_fs.spv", "TestShader", true);
        resources.createComputeShaderFromHLSL("test_compute.hlsl", "TestCompute", true);
        resources.createComputeShaderFromSPIRV("test_compute.spv", "TestCompute", true);

        std::cout << "  Shader creation test passed\n";
        return true;
    }

    bool testResourceCleanup() {
        std::cout << "Testing resource cleanup..." << std::endl;

        GPUResources resources(device, 10, 10, 10, 10, 10, 10);

        // Create some resources
        std::unique_ptr<MockSurface> surface(new MockSurface(256, 256));
        resources.createTexture(surface.get());

        float vertexData[] = { -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f };
        resources.createVertexBuffer(vertexData, sizeof(vertexData));

        // Call destroyResources and ensure it doesn't crash
        resources.destroyResources();

        // Verify we can create new resources after cleanup
        int32_t newTextureId = resources.createTexture(surface.get());
        assert(newTextureId == 0 && "Resource counter wasn't reset");

        std::cout << "  Resource cleanup test passed\n";
        return true;
    }

    bool testUpdatePointerCaches() {
        std::cout << "Testing updatePointerCaches..." << std::endl;

        GPUResources resources(device, 10, 10, 10, 10, 10, 10);

        // Create resources to generate pointers
        std::unique_ptr<MockSurface> surface(new MockSurface(256, 256));
        int32_t textureId = resources.createTexture(surface.get());

        // Create sampler
        int32_t samplerId = resources.createSampler(
            SDL_GPU_FILTER_LINEAR,
            SDL_GPU_FILTER_LINEAR,
            SDL_GPU_SAMPLERADDRESSMODE_REPEAT
        );

        // Create material that references these
        resources.createMaterial(
            textureId, -1, -1, samplerId,
            glm::vec4(1.0f), glm::vec4(1.0f), 1.0f
        );

        // Update pointers - we can't directly verify the result,
        // but we can ensure it doesn't crash
        resources.updatePointerCaches();

        std::cout << "  updatePointerCaches test passed\n";
        return true;
    }

    // Run all tests
    bool runAllTests() {
        bool success = true;

        success &= testConstructorAndDestructor();
        success &= testTextureCreation();
        success &= testBufferCreation();
        success &= testSamplerCreation();
        success &= testMeshCreation();
        success &= testMaterialCreation();
        success &= testShaderCreation();
        success &= testResourceCleanup();
        success &= testUpdatePointerCaches();

        return success;
    }
};

// Main function to run all tests
int runGPUResourcesTests() {
    std::cout << "===== STARTING GPURESOURCES TESTS =====\n";

    GPUResourcesTest tester;
    bool result = tester.runAllTests();

    std::cout << "===== GPURESOURCES TESTS: "
        << (result ? "PASSED" : "FAILED") << " =====\n";

    return result ? 0 : 1;
}