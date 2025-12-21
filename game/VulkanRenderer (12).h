#ifndef VULKAN_RENDERER_H
#define VULKAN_RENDERER_H

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <optional>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <iostream>
#include <set>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <chrono>
#include <thread>
#include "ATMProfiler.h"

// Forward declarations
struct Engine;
class TextureAtlas;
class SpatialGrid;
class RenderBatchManager;
class RenderableEntityContainer;

// Vulkan configuration constants
namespace VulkanConfig {

    constexpr uint32_t MAX_QUADS = 10000;
    constexpr uint32_t MAX_VERTICES = MAX_QUADS * 4;  // 40,000 vertices
    constexpr uint32_t MAX_INDICES = MAX_QUADS * 6;   // 60,000 indice

    // Number of frames that can be processed concurrently
    constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;

    // Maximum number of descriptor sets per pool
    constexpr uint32_t MAX_DESCRIPTOR_SETS = 10;

    // Maximum texture dimensions
    constexpr uint32_t MAX_TEXTURE_WIDTH = 4096;
    constexpr uint32_t MAX_TEXTURE_HEIGHT = 4096;

    // Default texture atlas size
    constexpr uint32_t TEXTURE_ATLAS_WIDTH = 2048;
    constexpr uint32_t TEXTURE_ATLAS_HEIGHT = 2048;

    // Initial batch capacity
    constexpr uint32_t INITIAL_BATCH_VERTEX_CAPACITY = 10000;
    constexpr uint32_t INITIAL_BATCH_INDEX_CAPACITY = 15000;

    // Vulkan allocator settings
    constexpr uint32_t STAGING_BUFFER_SIZE = 8 * 1024 * 1024; // 8MB staging buffer

    // Validation layers
    const std::vector<const char*> validationLayers = {
                "VK_LAYER_KHRONOS_validation"

    };

    // Device extensions
    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    // Enable validation layers in debug builds
#ifdef _DEBUG
    constexpr bool ENABLE_VALIDATION_LAYERS = true;
#else
    constexpr bool ENABLE_VALIDATION_LAYERS = false;
#endif

    // Shader file paths
    const std::string VERTEX_SHADER_PATH = "shaders/test_vert.spv";
    const std::string FRAGMENT_SHADER_PATH = "shaders/test_frag.spv";
}

// Queue family indices
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// Swapchain support details
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

// Vertex data structure for Vulkan
struct VulkanVertex {
    // Pack position and texcoords into a single 64-bit value
    // Position: 16-bit per component (X, Y)
    // TexCoord: 16-bit per component (U, V)
    uint64_t positionAndTexCoord;

    // Keep color as RGBA8 (32-bit total)
    uint32_t color;       // Pack RGBA into a single uint32_t (8 bits per channel)

    // Helper methods to set/get values
    void setPosition(float x, float y) {
        // Convert float to half (16-bit) and pack into the lower 32 bits
        uint16_t halfX = floatToHalf(x);
        uint16_t halfY = floatToHalf(y);
        uint32_t packedPos = (static_cast<uint32_t>(halfY) << 16) | halfX;

        // Keep upper 32 bits (texCoords) and replace lower 32 bits
        positionAndTexCoord = (positionAndTexCoord & 0xFFFFFFFF00000000ULL) | packedPos;
    }

    void setTexCoord(float u, float v) {
        // Convert float to half (16-bit) and pack into the upper 32 bits
        uint16_t halfU = floatToHalf(u);
        uint16_t halfV = floatToHalf(v);
        uint64_t packedTex = (static_cast<uint64_t>(halfV) << 48) | (static_cast<uint64_t>(halfU) << 32);

        // Keep lower 32 bits (position) and replace upper 32 bits
        positionAndTexCoord = (positionAndTexCoord & 0x00000000FFFFFFFFULL) | packedTex;
    }

    // Float32 to Float16 conversion
    static uint16_t floatToHalf(float value) {
        uint32_t x = *((uint32_t*)&value);
        uint16_t h = ((x >> 16) & 0x8000) | // sign
            ((((x >> 23) & 0xff) - 127 + 15) << 10) | // exponent
            ((x >> 13) & 0x3ff); // mantissa
        return h;
    }

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(VulkanVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        // Position (16-bit float X,Y in lower 32 bits)
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R16G16_SFLOAT;
        attributeDescriptions[0].offset = 0; // Start of positionAndTexCoord

        // Texture coordinates (16-bit float U,V in upper 32 bits)
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R16G16_SFLOAT;
        attributeDescriptions[1].offset = 4; // 4 bytes into positionAndTexCoord

        // Color (packed RGBA8)
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R8G8B8A8_UNORM;
        attributeDescriptions[2].offset = offsetof(VulkanVertex, color);

        return attributeDescriptions;
    }
};

// Vulkan texture representation
struct VulkanTexture {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
    VkSampler sampler;
    std::vector<VkDescriptorSet> descriptorSets;
    uint32_t width;
    uint32_t height;
    SDL_FRect region;
};

struct VulkanBatch {
    int textureId;
    int zIndex;

    // Fixed-size arrays instead of vectors
    VulkanVertex vertices[VulkanConfig::MAX_VERTICES];
    uint32_t indices[VulkanConfig::MAX_INDICES];

    // Track current count
    uint32_t vertexCount;
    uint32_t indexCount;

    // Host-visible buffers (both CPU and GPU accessible)
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexMemory;
    bool needsUpdate;

    VulkanBatch(int textureId, int zIndex)
        : textureId(textureId), zIndex(zIndex), needsUpdate(true),
        vertexCount(0), indexCount(0),
        vertexBuffer(VK_NULL_HANDLE), vertexMemory(VK_NULL_HANDLE),
        indexBuffer(VK_NULL_HANDLE), indexMemory(VK_NULL_HANDLE) {
    }

    void addQuad(float x, float y, float w, float h, SDL_FRect texRegion);
    void clear();
};
// VulkanRenderer class
class VulkanRenderer {
public:
    VulkanRenderer(Engine* engine);
    ~VulkanRenderer();

    // Initialize Vulkan
    bool initialize();

    // Main rendering functions
    void renderScene();
    void present(uint32_t imageIndex);
    // Resource management
    int registerTexture(SDL_Surface* surface, int x, int y, int width, int height);
    SDL_FRect getTextureRegion(int textureId);
    // Handle window resize
    void handleResize(int width, int height);

    // Engine reference
    Engine* engine;

private:
    // Vulkan instance and device
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkPhysicalDeviceMemoryProperties memoryProperties;

    // Swapchain
    VkSwapchainKHR swapchain;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainExtent;

    // Render pass and pipeline
    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    // Framebuffers
    std::vector<VkFramebuffer> swapchainFramebuffers;

    // Command pools and buffers
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    // Synchronization
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imageFences;

    // Descriptor pool and sets
    VkDescriptorPool descriptorPool;

    // Textures
    std::vector<VulkanTexture> textures;

    // Uniform buffer for camera
    struct UniformBufferObject {
        float projectionMatrix[16];  // Column-major 4x4 matrix
    };
    VkBuffer uniformBuffer;
    VkDeviceMemory uniformBufferMemory;

    // Vulkan Batch Manager
    std::vector<VulkanBatch> batches;
    std::unordered_map<uint64_t, size_t> batchMap;

    // Current frame index
    uint32_t currentFrame = 0;

    // Window dimensions
    int windowWidth;
    int windowHeight;
    bool framebufferResized = false;

    // Initialization functions
    bool createInstance();
    bool setupDebugMessenger();
    bool createSurface();
    bool pickPhysicalDevice();
    bool createLogicalDevice();
    bool createSwapChain();
    bool createImageViews();
    bool createRenderPass();
    bool createDescriptorSetLayout();
    bool createGraphicsPipeline();
    bool createFramebuffers();
    bool createCommandPool();
    bool createUniformBuffer();
    bool createDescriptorPool();
    bool createDescriptorSets();
    bool createCommandBuffers();
    bool createSyncObjects();

    // Helper functions
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    bool isDeviceSuitable(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    VkShaderModule createShaderModule(const std::vector<char>& code);

    // Resource management
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    bool createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

    // Texture management
    bool createTextureImage(SDL_Surface* surface, VulkanTexture& texture);
    bool createTextureImageView(VulkanTexture& texture);
    bool createTextureSampler(VulkanTexture& texture);

    // Rendering helpers
    void updateUniformBuffer();
    VulkanBatch& getBatch(int textureId, int zIndex);
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void updateBatchBuffers();

    // Cleanup
    void cleanupSwapChain();
    bool recreateSwapChain();

    // Static debug callback
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);
};

// Helper function to read binary file (for shader loading)
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

#endif // VULKAN_RENDERER_H