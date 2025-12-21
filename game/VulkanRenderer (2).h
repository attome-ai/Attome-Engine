#ifndef VULKAN_RENDERER_H
#define VULKAN_RENDERER_H

#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <optional>
#include <unordered_map>
#include <atomic>
#include "ATMProfiler.h"

// Forward declarations
class Engine;
struct SDL_Window;
// Vertex struct that matches our shader input
struct Vertex {
    float pos[2];
    float texCoord[2];
    float color[4];

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        // Position
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        // Texture coordinates
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, texCoord);

        // Color
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

// RenderBatch for batched rendering
struct RenderBatch {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    int textureId;
    uint32_t indexOffset = 0;

    void addQuad(const Vertex& topLeft, const Vertex& topRight,
        const Vertex& bottomRight, const Vertex& bottomLeft) {
        vertices.push_back(topLeft);
        vertices.push_back(topRight);
        vertices.push_back(bottomRight);
        vertices.push_back(bottomLeft);

        // Add indices for two triangles forming a quad
        indices.push_back(indexOffset);
        indices.push_back(indexOffset + 1);
        indices.push_back(indexOffset + 2);
        indices.push_back(indexOffset);
        indices.push_back(indexOffset + 2);
        indices.push_back(indexOffset + 3);

        indexOffset += 4;
    }

    void clear() {
        vertices.clear();
        indices.clear();
        indexOffset = 0;
    }
};

// Optimized frame resource structure that tracks buffer capacities
struct FrameResource {
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;
    size_t vertexCapacity = 0;
    size_t indexCapacity = 0;
};

// Encapsulate Vulkan texture info
struct VulkanTexture {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
    VkSampler sampler;
    VkDescriptorSet descriptorSet;
    int width;
    int height;
    bool hasAlpha;
};


// This struct holds queue family indices
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// This struct holds swapchain support details
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class VulkanRenderer {
public:
    VulkanRenderer();
    ~VulkanRenderer();

    bool initialize(SDL_Window* window, int width, int height);
    void cleanup();

    // Texture management
    int createTextureAtlas(int width, int height);
    int registerTexture(SDL_Surface* surface, int atlasId, int x, int y, int width, int height);
    VkImageView getTextureView(int textureId) const;
    VkSampler getTextureSampler(int textureId) const;
    void updateDescriptorSets(VulkanTexture& texture);

    // Rendering methods
    void beginFrame();
    void endFrame();

    // New batched rendering methods
    void beginBatch(int textureId);
    void addQuadToBatch(const Vertex& topLeft, const Vertex& topRight,
        const Vertex& bottomRight, const Vertex& bottomLeft);
    void flushBatch();

    // Existing render method for backward compatibility
    void render(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, int textureId);

    void setClearColor(float r, float g, float b, float a);

    // Get texture coordinates for a registered texture region
    struct TextureRegion
    {
        uint16_t x;
        uint16_t y;
        uint16_t width;
        uint16_t height;
    };

    inline TextureRegion getTextureRegion(int textureId) const {
        if (textureId >= 0 && textureId < textureRegions.size()) {
            return textureRegions[textureId];
        }
        // Return an empty region if textureId is invalid
        return { 0, 0, 0, 0 };
    }

    // Utility methods
    VkDevice getDevice() const { return device; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }

    bool recreateSwapChain();
    void waitIdle();

private:
    // Initialization helpers
    bool createInstance();
    bool setupDebugMessenger();
    bool createSurface(SDL_Window* window);
    bool pickPhysicalDevice();
    bool createLogicalDevice();
    bool createSwapChain();
    bool createImageViews();
    bool createRenderPass();
    bool createDescriptorSetLayout();
    bool createGraphicsPipeline();
    bool createFramebuffers();
    bool createCommandPool();
    bool createCommandBuffers();
    bool createSyncObjects();
    bool createDescriptorPool();
    void createPersistentBuffers();
    void resizeBufferIfNeeded(size_t frameIndex, size_t vertexCount, size_t indexCount);

    // Texture helpers
    bool createTextureImage(SDL_Surface* surface, VulkanTexture& texture);
    bool createTextureImageView(VulkanTexture& texture);
    bool createTextureSampler(VulkanTexture& texture);
    bool createDescriptorSets(VulkanTexture& texture);

    // Utility functions
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    bool isDeviceSuitable(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
        VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
        VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
        VkImage& image, VkDeviceMemory& imageMemory);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    VkImageView createImageView(VkImage image, VkFormat format);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);

    // Debug validation functions
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();

    // Vulkan objects
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    VkDescriptorPool descriptorPool;
    VkClearValue clearColor;
    uint32_t imageIndex;

    // Frame synchronization
    size_t currentFrame = 0;
    const int MAX_FRAMES_IN_FLIGHT = 2;
    bool framebufferResized = false;

    // Persistent buffers for performance optimization
    std::vector<FrameResource> persistentBuffers;
    std::vector<FrameResource> frameResources;
    std::vector<FrameResource> previousFrameResources;

    // Current active render batch
    RenderBatch currentBatch;

    // Mapping of texture IDs to their corresponding batches
    std::unordered_map<int, RenderBatch> textureBatches;

    // Window dimensions
    int windowWidth;
    int windowHeight;

    // Texture management
    std::vector<VulkanTexture> textures;
    std::vector<TextureRegion> textureRegions;
    std::unordered_map<int, int> textureAtlases; // Maps atlas ID to texture ID

    // Debug settings
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
};

#endif // VULKAN_RENDERER_H