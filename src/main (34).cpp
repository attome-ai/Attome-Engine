
#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <array>
#include <optional>
#include <set>
#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <iomanip>

// Enable or disable profiling (1 = on, 0 = off)
#define ENABLE_PROFILING 1

// Window dimensions
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// Maximum number of frames that can be processed concurrently
// Increased for higher throughput
const int MAX_FRAMES_IN_FLIGHT = 3;

// Error codes
#define VK_APP_SUCCESS 0
#define VK_APP_ERROR_SDL_INIT -1
#define VK_APP_ERROR_SDL_WINDOW -2
#define VK_APP_ERROR_INSTANCE_CREATION -3
#define VK_APP_ERROR_DEBUG_MESSENGER -4
#define VK_APP_ERROR_SURFACE_CREATION -5
#define VK_APP_ERROR_NO_GPU -6
#define VK_APP_ERROR_NO_SUITABLE_GPU -7
#define VK_APP_ERROR_DEVICE_CREATION -8
#define VK_APP_ERROR_SWAPCHAIN_CREATION -9
#define VK_APP_ERROR_IMAGEVIEW_CREATION -10
#define VK_APP_ERROR_RENDERPASS_CREATION -11
#define VK_APP_ERROR_PIPELINE_LAYOUT_CREATION -12
#define VK_APP_ERROR_PIPELINE_CREATION -13
#define VK_APP_ERROR_FRAMEBUFFER_CREATION -14
#define VK_APP_ERROR_COMMAND_POOL_CREATION -15
#define VK_APP_ERROR_BUFFER_CREATION -16
#define VK_APP_ERROR_MEMORY_ALLOCATION -17
#define VK_APP_ERROR_COMMAND_BUFFER_ALLOCATION -18
#define VK_APP_ERROR_SYNC_OBJECTS_CREATION -19
#define VK_APP_ERROR_SHADER_MODULE_CREATION -20
#define VK_APP_ERROR_MEMORY_MAP -21
#define VK_APP_ERROR_COMMAND_BUFFER_RECORDING -22
#define VK_APP_ERROR_QUEUE_SUBMIT -23
#define VK_APP_ERROR_SWAPCHAIN_ACQUIRE -24
#define VK_APP_ERROR_SWAPCHAIN_PRESENT -25
#define VK_APP_ERROR_OFFSCREEN_RESOURCES -26

// Disable validation layers in all build configurations for performance
const bool enableValidationLayers = false;

// Validation layers (unused when disabled)
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// Device extensions
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// Structure to hold profiling data for a single function
struct ProfileData {
    uint64_t callCount = 0;
    double totalTimeMs = 0.0;
    double minTimeMs = std::numeric_limits<double>::max();
    double maxTimeMs = 0.0;
    std::chrono::steady_clock::time_point lastReportTime = std::chrono::steady_clock::now();
};

// Global profiling data
class Profiler {
private:
    std::unordered_map<std::string, ProfileData> stats;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> startTimes;
    std::mutex mutex;
    std::chrono::steady_clock::time_point lastReportTime = std::chrono::steady_clock::now();
    bool reportEnabled = ENABLE_PROFILING; // Determined by ENABLE_PROFILING flag
    double reportIntervalSeconds = 2.0;    // Longer interval for less overhead

public:
    static Profiler& getInstance() {
        static Profiler instance;
        return instance;
    }

    void begin(const std::string& functionName) {
        if (!reportEnabled) return;

        std::lock_guard<std::mutex> lock(mutex);
        startTimes[functionName] = std::chrono::steady_clock::now();
    }

    void end(const std::string& functionName) {
        if (!reportEnabled) return;

        std::lock_guard<std::mutex> lock(mutex);

        auto endTime = std::chrono::steady_clock::now();
        auto it = startTimes.find(functionName);

        if (it != startTimes.end()) {
            auto startTime = it->second;
            double elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

            auto& data = stats[functionName];
            data.callCount++;
            data.totalTimeMs += elapsedMs;
            data.minTimeMs = std::min(data.minTimeMs, elapsedMs);
            data.maxTimeMs = std::max(data.maxTimeMs, elapsedMs);

            startTimes.erase(it);

            // Check if we should generate a report
            if (reportEnabled) {
                auto now = std::chrono::steady_clock::now();
                double secondsSinceLastReport =
                    std::chrono::duration<double>(now - lastReportTime).count();

                if (secondsSinceLastReport >= reportIntervalSeconds) {
                    generateReport();
                    lastReportTime = now;
                }
            }
        }
    }

    void setReportInterval(double seconds) {
        std::lock_guard<std::mutex> lock(mutex);
        reportIntervalSeconds = seconds;
    }

    void enableReporting(bool enable) {
        std::lock_guard<std::mutex> lock(mutex);
        reportEnabled = enable;
    }

    void resetStats() {
        std::lock_guard<std::mutex> lock(mutex);
        stats.clear();
        startTimes.clear();
    }

    void generateReport() {
        if (!reportEnabled) return;

        std::cout << "\n=== Profiling Report ===\n";
        std::cout << std::left << std::setw(30) << "Function"
            << std::right << std::setw(10) << "Calls"
            << std::setw(15) << "Total (ms)"
            << std::setw(15) << "Avg (ms)"
            << std::setw(15) << "Min (ms)"
            << std::setw(15) << "Max (ms)" << std::endl;
        std::cout << std::string(100, '-') << std::endl;

        // Sort by total time (highest first) to focus on bottlenecks
        std::vector<std::pair<std::string, ProfileData>> sortedStats(stats.begin(), stats.end());
        std::sort(sortedStats.begin(), sortedStats.end(),
            [](const auto& a, const auto& b) { return a.second.totalTimeMs > b.second.totalTimeMs; });

        for (const auto& entry : sortedStats) {
            const auto& name = entry.first;
            const auto& data = entry.second;
            double avgTime = data.callCount > 0 ? data.totalTimeMs / data.callCount : 0;

            std::cout << std::left << std::setw(30) << name
                << std::right << std::setw(10) << data.callCount
                << std::setw(15) << std::fixed << std::setprecision(3) << data.totalTimeMs
                << std::setw(15) << std::fixed << std::setprecision(3) << avgTime
                << std::setw(15) << std::fixed << std::setprecision(3)
                << (data.callCount > 0 ? data.minTimeMs : 0)
                << std::setw(15) << std::fixed << std::setprecision(3) << data.maxTimeMs
                << std::endl;
        }

        std::cout << "========================\n";
    }
};

// Profiling macros - toggle with ENABLE_PROFILING define
#if ENABLE_PROFILING
#define PROFILE_BEGIN(name) Profiler::getInstance().begin(name)
#define PROFILE_END(name) Profiler::getInstance().end(name)
#define PROFILE_FUNCTION() auto CONCAT_IMPL(__profiler_, __LINE__) = ProfilerGuard(__FUNCTION__)
#define PROFILE_SCOPE(name) auto CONCAT_IMPL(__profiler_, __LINE__) = ProfilerGuard(name)
#define CONCAT_IMPL(a, b) a ## b

// Helper class for automatic begin/end in a scope
class ProfilerGuard {
public:
    ProfilerGuard(const std::string& name) : name(name) {
        Profiler::getInstance().begin(name);
    }

    ~ProfilerGuard() {
        Profiler::getInstance().end(name);
    }
private:
    std::string name;
};
#else
#define PROFILE_BEGIN(name) ((void)0)
#define PROFILE_END(name) ((void)0)
#define PROFILE_FUNCTION() ((void)0)
#define PROFILE_SCOPE(name) ((void)0)
#endif

// FPS counter class - optimized for high framerate
class FPSCounter {
private:
    uint32_t frameCount;
    std::chrono::steady_clock::time_point lastTime;
    double fps;
    // FPS reporting interval in seconds (higher for less overhead)
    const double reportInterval = 1.0;

public:
    FPSCounter() : frameCount(0), fps(0.0) {
        lastTime = std::chrono::steady_clock::now();
    }

    void update() {
        frameCount++;

        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(currentTime - lastTime).count();

        if (elapsed >= reportInterval) {
            fps = frameCount / elapsed;
            std::cout << "FPS: " << std::fixed << std::setprecision(1) << fps << std::endl;

            frameCount = 0;
            lastTime = currentTime;
        }
    }

    double getFPS() const {
        return fps;
    }
};

std::vector<char> readFile(const std::string& filename) {
    PROFILE_FUNCTION();
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

// Forward declarations
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);

// Vertex structure
struct Vertex {
    float pos[2];
    float color[3];

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

// Triangle vertices
const std::vector<Vertex> vertices = {
    {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
};

// Queue Family Indices
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// Swap Chain Support Details
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

// Application class
class vapp {
public:
    vapp() : window(nullptr), currentFrame(0) {}

    int init() {
        PROFILE_FUNCTION();
        int result = initWindow();
        if (result != VK_APP_SUCCESS) {
            return result;
        }

        // Initialize FPS counter
        fpsCounter = std::make_unique<FPSCounter>();

        return initVulkan();
    }

    void cleanup() {
        PROFILE_FUNCTION();
        cleanupVulkan();
        cleanupWindow();
    }

    int drawFrame() {
        PROFILE_FUNCTION();

        // Get the current frame resources
        const uint32_t frameIndex = currentFrame % MAX_FRAMES_IN_FLIGHT;

        // Wait for the previous frame to finish - MINIMAL WAIT TIME
        {
            PROFILE_SCOPE("WaitForFence");
            // Use VK_FALSE for non-blocking wait - if fence not ready, skip this frame
            // This helps achieve much higher FPS by not stalling the CPU
            VkResult waitResult = vkWaitForFences(device, 1, &inFlightFences[frameIndex], VK_FALSE, 0);
            if (waitResult != VK_SUCCESS) {
                // Skip this frame if the fence isn't ready - prevents stalling
                currentFrame = (currentFrame + 1) % UINT32_MAX;
                return VK_APP_SUCCESS;
            }

            // Reset the fence immediately
            vkResetFences(device, 1, &inFlightFences[frameIndex]);
        }

        // Acquire the next image from the swap chain - NO TIMEOUT
        uint32_t imageIndex;
        {
            PROFILE_SCOPE("AcquireNextImage");
            VkResult result = vkAcquireNextImageKHR(device, swapChain, 0, // No timeout
                imageAvailableSemaphores[frameIndex],
                VK_NULL_HANDLE, &imageIndex);

            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapChain();
                return VK_APP_SUCCESS;
            }
            else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
                // Don't fail on acquire error, just try again next frame
                return VK_APP_SUCCESS;
            }
        }

        // Update the command buffer for offscreen rendering and copying to swapchain
        {
            PROFILE_SCOPE("UpdateCommandBuffer");
            updateCommandBuffer(frameIndex, imageIndex);
        }

        // Submit the command buffer
        {
            PROFILE_SCOPE("QueueSubmit");
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

            VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[frameIndex] };
            VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffers[frameIndex];

            VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[frameIndex] };
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;

            // Submit to queue - optimize by checking result
            VkResult result = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[frameIndex]);
            if (result != VK_SUCCESS) {
                // Don't crash on submit error, just skip this frame
                return VK_APP_SUCCESS;
            }
        }

        // Present the result to the screen
        {
            PROFILE_SCOPE("QueuePresent");
            VkPresentInfoKHR presentInfo{};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = &renderFinishedSemaphores[frameIndex];

            VkSwapchainKHR swapChains[] = { swapChain };
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = swapChains;
            presentInfo.pImageIndices = &imageIndex;
            presentInfo.pResults = nullptr;

            VkResult result = vkQueuePresentKHR(presentQueue, &presentInfo);

            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
                framebufferResized = false;
                recreateSwapChain();
            }
        }

        // Move to the next frame (even if there was an error)
        currentFrame = (currentFrame + 1) % UINT32_MAX; // Use UINT32_MAX for maximum range

        // Update FPS counter
        if (fpsCounter) {
            fpsCounter->update();
        }

        return VK_APP_SUCCESS;
    }

    void setFramebufferResized() {
        framebufferResized = true;
    }

    SDL_Window* getWindow() const {
        return window;
    }

private:
    SDL_Window* window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    // Offscreen rendering resources
    VkImage offscreenImage;
    VkDeviceMemory offscreenImageMemory;
    VkImageView offscreenImageView;
    VkFramebuffer offscreenFramebuffer;
    VkRenderPass offscreenRenderPass;

    // Multiple frames in flight resources
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame;

    bool framebufferResized = false;

    // FPS counter
    std::unique_ptr<FPSCounter> fpsCounter;

    int initWindow() {
        PROFILE_FUNCTION();
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
            return VK_APP_ERROR_SDL_INIT;
        }

        window = SDL_CreateWindow("Vulkan Triangle", WIDTH, HEIGHT, SDL_WINDOW_VULKAN);
        if (!window) {
            std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
            return VK_APP_ERROR_SDL_WINDOW;
        }

        return VK_APP_SUCCESS;
    }

    void cleanupWindow() {
        PROFILE_FUNCTION();
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    int initVulkan() {
        PROFILE_FUNCTION();
        int result;

        result = createInstance();
        if (result != VK_APP_SUCCESS) return result;

        result = setupDebugMessenger();
        if (result != VK_APP_SUCCESS) return result;

        result = createSurface();
        if (result != VK_APP_SUCCESS) return result;

        result = pickPhysicalDevice();
        if (result != VK_APP_SUCCESS) return result;

        result = createLogicalDevice();
        if (result != VK_APP_SUCCESS) return result;

        result = createSwapChain();
        if (result != VK_APP_SUCCESS) return result;

        result = createImageViews();
        if (result != VK_APP_SUCCESS) return result;

        result = createRenderPass();
        if (result != VK_APP_SUCCESS) return result;

        result = createOffscreenRenderPass();
        if (result != VK_APP_SUCCESS) return result;

        result = createGraphicsPipeline();
        if (result != VK_APP_SUCCESS) return result;

        result = createFramebuffers();
        if (result != VK_APP_SUCCESS) return result;

        result = createOffscreenResources();
        if (result != VK_APP_SUCCESS) return result;

        result = createCommandPool();
        if (result != VK_APP_SUCCESS) return result;

        result = createVertexBuffer();
        if (result != VK_APP_SUCCESS) return result;

        result = createCommandBuffers();
        if (result != VK_APP_SUCCESS) return result;

        result = createSyncObjects();
        if (result != VK_APP_SUCCESS) return result;

        // Pre-record command buffers
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            preRecordCommandBuffer(i);
        }

        return VK_APP_SUCCESS;
    }

    void cleanupVulkan() {
        PROFILE_FUNCTION();
        vkDeviceWaitIdle(device);

        cleanupSwapChain();
        cleanupOffscreenResources();

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroyRenderPass(device, offscreenRenderPass, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
    }

    void cleanupSwapChain() {
        PROFILE_FUNCTION();
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void cleanupOffscreenResources() {
        PROFILE_FUNCTION();
        vkDestroyFramebuffer(device, offscreenFramebuffer, nullptr);
        vkDestroyImageView(device, offscreenImageView, nullptr);
        vkDestroyImage(device, offscreenImage, nullptr);
        vkFreeMemory(device, offscreenImageMemory, nullptr);
    }

    int recreateSwapChain() {
        PROFILE_FUNCTION();
        int width = 0, height = 0;
        SDL_GetWindowSizeInPixels(window, &width, &height);
        while (width == 0 || height == 0) {
            SDL_GetWindowSizeInPixels(window, &width, &height);
            SDL_WaitEvent(nullptr);
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();
        cleanupOffscreenResources();

        int result = createSwapChain();
        if (result != VK_APP_SUCCESS) return result;

        result = createImageViews();
        if (result != VK_APP_SUCCESS) return result;

        result = createFramebuffers();
        if (result != VK_APP_SUCCESS) return result;

        result = createOffscreenResources();
        if (result != VK_APP_SUCCESS) return result;

        // Need to re-record command buffers because framebuffers have changed
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            preRecordCommandBuffer(i);
        }

        return VK_APP_SUCCESS;
    }

    int createOffscreenResources() {
        PROFILE_FUNCTION();

        // Create offscreen image
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = swapChainExtent.width;
        imageInfo.extent.height = swapChainExtent.height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = swapChainImageFormat;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        // Use as color attachment and source for transfer
        imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

        if (vkCreateImage(device, &imageInfo, nullptr, &offscreenImage) != VK_SUCCESS) {
            std::cerr << "Failed to create offscreen image!" << std::endl;
            return VK_APP_ERROR_OFFSCREEN_RESOURCES;
        }

        // Allocate memory for the offscreen image
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, offscreenImage, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;

        uint32_t memoryTypeIndex;
        if (!findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memoryTypeIndex)) {
            vkDestroyImage(device, offscreenImage, nullptr);
            std::cerr << "Failed to find suitable memory type for offscreen image!" << std::endl;
            return VK_APP_ERROR_OFFSCREEN_RESOURCES;
        }
        allocInfo.memoryTypeIndex = memoryTypeIndex;

        if (vkAllocateMemory(device, &allocInfo, nullptr, &offscreenImageMemory) != VK_SUCCESS) {
            vkDestroyImage(device, offscreenImage, nullptr);
            std::cerr << "Failed to allocate offscreen image memory!" << std::endl;
            return VK_APP_ERROR_OFFSCREEN_RESOURCES;
        }

        vkBindImageMemory(device, offscreenImage, offscreenImageMemory, 0);

        // Create image view for the offscreen image
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = offscreenImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = swapChainImageFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &viewInfo, nullptr, &offscreenImageView) != VK_SUCCESS) {
            vkDestroyImage(device, offscreenImage, nullptr);
            vkFreeMemory(device, offscreenImageMemory, nullptr);
            std::cerr << "Failed to create offscreen image view!" << std::endl;
            return VK_APP_ERROR_OFFSCREEN_RESOURCES;
        }

        // Create offscreen framebuffer
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = offscreenRenderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &offscreenImageView;
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &offscreenFramebuffer) != VK_SUCCESS) {
            vkDestroyImageView(device, offscreenImageView, nullptr);
            vkDestroyImage(device, offscreenImage, nullptr);
            vkFreeMemory(device, offscreenImageMemory, nullptr);
            std::cerr << "Failed to create offscreen framebuffer!" << std::endl;
            return VK_APP_ERROR_OFFSCREEN_RESOURCES;
        }

        return VK_APP_SUCCESS;
    }

    int createOffscreenRenderPass() {
        PROFILE_FUNCTION();

        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        // Use TRANSFER_SRC_OPTIMAL since we'll be copying from this image
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &offscreenRenderPass) != VK_SUCCESS) {
            std::cerr << "Failed to create offscreen render pass!" << std::endl;
            return VK_APP_ERROR_RENDERPASS_CREATION;
        }

        return VK_APP_SUCCESS;
    }

    int createInstance() {
        PROFILE_FUNCTION();
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            std::cerr << "Validation layers requested, but not available!" << std::endl;
            return VK_APP_ERROR_INSTANCE_CREATION;
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            std::cerr << "Failed to create instance!" << std::endl;
            return VK_APP_ERROR_INSTANCE_CREATION;
        }

        return VK_APP_SUCCESS;
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    int setupDebugMessenger() {
        PROFILE_FUNCTION();
        if (!enableValidationLayers) return VK_APP_SUCCESS;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            std::cerr << "Failed to set up debug messenger!" << std::endl;
            return VK_APP_ERROR_DEBUG_MESSENGER;
        }

        return VK_APP_SUCCESS;
    }

    int createSurface() {
        PROFILE_FUNCTION();
        if (!SDL_Vulkan_CreateSurface(window, instance, 0, &surface)) {
            std::cerr << "Failed to create window surface!" << std::endl;
            return VK_APP_ERROR_SURFACE_CREATION;
        }

        return VK_APP_SUCCESS;
    }

    int pickPhysicalDevice() {
        PROFILE_FUNCTION();
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            std::cerr << "Failed to find GPUs with Vulkan support!" << std::endl;
            return VK_APP_ERROR_NO_GPU;
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            std::cerr << "Failed to find a suitable GPU!" << std::endl;
            return VK_APP_ERROR_NO_SUITABLE_GPU;
        }

        // Log GPU name for info
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
        std::cout << "Selected GPU: " << deviceProperties.deviceName << std::endl;

        return VK_APP_SUCCESS;
    }

    int createLogicalDevice() {
        PROFILE_FUNCTION();
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            std::cerr << "Failed to create logical device!" << std::endl;
            return VK_APP_ERROR_DEVICE_CREATION;
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

        return VK_APP_SUCCESS;
    }

    int createSwapChain() {
        PROFILE_FUNCTION();
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        // For maximum performance, use minImageCount + 1 which reduces wait times
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        // Make sure we don't exceed maxImageCount (if it's not 0, which means unlimited)
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT; // Allow transfer to swapchain image

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            std::cerr << "Failed to create swap chain!" << std::endl;
            return VK_APP_ERROR_SWAPCHAIN_CREATION;
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

        return VK_APP_SUCCESS;
    }

    int createImageViews() {
        PROFILE_FUNCTION();
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                std::cerr << "Failed to create image views!" << std::endl;
                return VK_APP_ERROR_IMAGEVIEW_CREATION;
            }
        }

        return VK_APP_SUCCESS;
    }

    int createRenderPass() {
        PROFILE_FUNCTION();
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            std::cerr << "Failed to create render pass!" << std::endl;
            return VK_APP_ERROR_RENDERPASS_CREATION;
        }

        return VK_APP_SUCCESS;
    }

    int createGraphicsPipeline() {
        PROFILE_FUNCTION();

        // Load shader files
        std::vector<char> vertShaderCode;
        std::vector<char> fragShaderCode;

        try {
            vertShaderCode = readFile("C:/Users/Custom_PC_bh/Downloads/everything/Template/shaders/simple_vert.spv");
            fragShaderCode = readFile("C:/Users/Custom_PC_bh/Downloads/everything/Template/shaders/simple_frag.spv");
        }
        catch (const std::exception& e) {
            std::cerr << "Failed to load shader files: " << e.what() << std::endl;
            return VK_APP_ERROR_SHADER_MODULE_CREATION;
        }

        // Vertex shader module
        VkShaderModuleCreateInfo vertShaderModuleCreateInfo{};
        vertShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        vertShaderModuleCreateInfo.codeSize = vertShaderCode.size();
        vertShaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(vertShaderCode.data());

        VkShaderModule vertShaderModule;
        if (vkCreateShaderModule(device, &vertShaderModuleCreateInfo, nullptr, &vertShaderModule) != VK_SUCCESS) {
            std::cerr << "Failed to create vertex shader module!" << std::endl;
            return VK_APP_ERROR_SHADER_MODULE_CREATION;
        }

        // Fragment shader module
        VkShaderModuleCreateInfo fragShaderModuleCreateInfo{};
        fragShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        fragShaderModuleCreateInfo.codeSize = fragShaderCode.size();
        fragShaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(fragShaderCode.data());

        VkShaderModule fragShaderModule;
        if (vkCreateShaderModule(device, &fragShaderModuleCreateInfo, nullptr, &fragShaderModule) != VK_SUCCESS) {
            vkDestroyShaderModule(device, vertShaderModule, nullptr);
            std::cerr << "Failed to create fragment shader module!" << std::endl;
            return VK_APP_ERROR_SHADER_MODULE_CREATION;
        }

        // Shader stages
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        // Vertex input
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        // Input assembly
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // Viewport
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        // Scissor
        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        // Viewport state
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        // Rasterizer
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        // Multisampling
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Color blending
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        // Pipeline layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            vkDestroyShaderModule(device, fragShaderModule, nullptr);
            vkDestroyShaderModule(device, vertShaderModule, nullptr);
            std::cerr << "Failed to create pipeline layout!" << std::endl;
            return VK_APP_ERROR_PIPELINE_LAYOUT_CREATION;
        }

        // Graphics pipeline
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = nullptr;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = offscreenRenderPass; // Use offscreen render pass to generate pipeline
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            vkDestroyShaderModule(device, fragShaderModule, nullptr);
            vkDestroyShaderModule(device, vertShaderModule, nullptr);
            std::cerr << "Failed to create graphics pipeline!" << std::endl;
            return VK_APP_ERROR_PIPELINE_CREATION;
        }

        // Cleanup shader modules
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);

        return VK_APP_SUCCESS;
    }

    int createFramebuffers() {
        PROFILE_FUNCTION();
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                std::cerr << "Failed to create framebuffer!" << std::endl;
                return VK_APP_ERROR_FRAMEBUFFER_CREATION;
            }
        }

        return VK_APP_SUCCESS;
    }

    int createCommandPool() {
        PROFILE_FUNCTION();
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            std::cerr << "Failed to create command pool!" << std::endl;
            return VK_APP_ERROR_COMMAND_POOL_CREATION;
        }

        return VK_APP_SUCCESS;
    }

    int createVertexBuffer() {
        PROFILE_FUNCTION();
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        int result = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        if (result != VK_APP_SUCCESS) {
            return result;
        }

        void* data;
        if (vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data) != VK_SUCCESS) {
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
            std::cerr << "Failed to map memory!" << std::endl;
            return VK_APP_ERROR_MEMORY_MAP;
        }

        memcpy(data, vertices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        result = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
        if (result != VK_APP_SUCCESS) {
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
            return result;
        }

        result = copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
        if (result != VK_APP_SUCCESS) {
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
            return result;
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        return VK_APP_SUCCESS;
    }

    int createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        PROFILE_FUNCTION();
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            std::cerr << "Failed to create buffer!" << std::endl;
            return VK_APP_ERROR_BUFFER_CREATION;
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;

        if (!findMemoryType(memRequirements.memoryTypeBits, properties, allocInfo.memoryTypeIndex)) {
            vkDestroyBuffer(device, buffer, nullptr);
            std::cerr << "Failed to find suitable memory type!" << std::endl;
            return VK_APP_ERROR_MEMORY_ALLOCATION;
        }

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            vkDestroyBuffer(device, buffer, nullptr);
            std::cerr << "Failed to allocate buffer memory!" << std::endl;
            return VK_APP_ERROR_MEMORY_ALLOCATION;
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);

        return VK_APP_SUCCESS;
    }

    int copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        PROFILE_FUNCTION();
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
            std::cerr << "Failed to allocate command buffer for buffer copy!" << std::endl;
            return VK_APP_ERROR_COMMAND_BUFFER_ALLOCATION;
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
            std::cerr << "Failed to begin command buffer for buffer copy!" << std::endl;
            return VK_APP_ERROR_COMMAND_BUFFER_RECORDING;
        }

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
            std::cerr << "Failed to end command buffer for buffer copy!" << std::endl;
            return VK_APP_ERROR_COMMAND_BUFFER_RECORDING;
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
            std::cerr << "Failed to submit command buffer for buffer copy!" << std::endl;
            return VK_APP_ERROR_QUEUE_SUBMIT;
        }

        vkQueueWaitIdle(graphicsQueue);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

        return VK_APP_SUCCESS;
    }

    bool findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, uint32_t& memoryTypeIndex) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                memoryTypeIndex = i;
                return true;
            }
        }

        return false;
    }

    int createCommandBuffers() {
        PROFILE_FUNCTION();
        // Allocate command buffers for each frame in flight
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            std::cerr << "Failed to allocate command buffers!" << std::endl;
            return VK_APP_ERROR_COMMAND_BUFFER_ALLOCATION;
        }

        return VK_APP_SUCCESS;
    }

    // Pre-record command buffer for a specific frame
    int preRecordCommandBuffer(size_t commandBufferIndex) {
        PROFILE_FUNCTION();
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        // Allow the command buffer to be resubmitted while it is also already pending execution
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

        if (vkBeginCommandBuffer(commandBuffers[commandBufferIndex], &beginInfo) != VK_SUCCESS) {
            std::cerr << "Failed to begin recording command buffer!" << std::endl;
            return VK_APP_ERROR_COMMAND_BUFFER_RECORDING;
        }

        // First render to offscreen framebuffer
        VkRenderPassBeginInfo offscreenRenderPassInfo{};
        offscreenRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        offscreenRenderPassInfo.renderPass = offscreenRenderPass;
        offscreenRenderPassInfo.framebuffer = offscreenFramebuffer;
        offscreenRenderPassInfo.renderArea.offset = { 0, 0 };
        offscreenRenderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
        offscreenRenderPassInfo.clearValueCount = 1;
        offscreenRenderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffers[commandBufferIndex], &offscreenRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffers[commandBufferIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkBuffer vertexBuffers[] = { vertexBuffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffers[commandBufferIndex], 0, 1, vertexBuffers, offsets);

        vkCmdDraw(commandBuffers[commandBufferIndex], static_cast<uint32_t>(vertices.size()), 1, 0, 0);
        vkCmdEndRenderPass(commandBuffers[commandBufferIndex]);

        // This will be filled in later when we know which swapchain image to use

        if (vkEndCommandBuffer(commandBuffers[commandBufferIndex]) != VK_SUCCESS) {
            std::cerr << "Failed to record command buffer!" << std::endl;
            return VK_APP_ERROR_COMMAND_BUFFER_RECORDING;
        }

        return VK_APP_SUCCESS;
    }

    // Update command buffer to use the correct framebuffer at drawing time
    void updateCommandBuffer(size_t commandBufferIndex, uint32_t imageIndex) {
        PROFILE_FUNCTION();
        // Reset and re-record the command buffer with the current framebuffer
        vkResetCommandBuffer(commandBuffers[commandBufferIndex], 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

        if (vkBeginCommandBuffer(commandBuffers[commandBufferIndex], &beginInfo) != VK_SUCCESS) {
            // Just log the error and continue - don't terminate
            std::cerr << "Failed to begin command buffer recording!" << std::endl;
            return;
        }

        // First render to offscreen framebuffer
        VkRenderPassBeginInfo offscreenRenderPassInfo{};
        offscreenRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        offscreenRenderPassInfo.renderPass = offscreenRenderPass;
        offscreenRenderPassInfo.framebuffer = offscreenFramebuffer;
        offscreenRenderPassInfo.renderArea.offset = { 0, 0 };
        offscreenRenderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
        offscreenRenderPassInfo.clearValueCount = 1;
        offscreenRenderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffers[commandBufferIndex], &offscreenRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffers[commandBufferIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkBuffer vertexBuffers[] = { vertexBuffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffers[commandBufferIndex], 0, 1, vertexBuffers, offsets);

        vkCmdDraw(commandBuffers[commandBufferIndex], static_cast<uint32_t>(vertices.size()), 1, 0, 0);
        vkCmdEndRenderPass(commandBuffers[commandBufferIndex]);

        // Transition swapchain image to transfer destination
        VkImageMemoryBarrier dstBarrier{};
        dstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        dstBarrier.srcAccessMask = 0;
        dstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        dstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        dstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        dstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        dstBarrier.image = swapChainImages[imageIndex];
        dstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        dstBarrier.subresourceRange.baseMipLevel = 0;
        dstBarrier.subresourceRange.levelCount = 1;
        dstBarrier.subresourceRange.baseArrayLayer = 0;
        dstBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(commandBuffers[commandBufferIndex],
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &dstBarrier);

        // Copy offscreen image to swapchain image
        VkImageCopy copyRegion{};
        copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.srcSubresource.mipLevel = 0;
        copyRegion.srcSubresource.baseArrayLayer = 0;
        copyRegion.srcSubresource.layerCount = 1;
        copyRegion.srcOffset = { 0, 0, 0 };
        copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.dstSubresource.mipLevel = 0;
        copyRegion.dstSubresource.baseArrayLayer = 0;
        copyRegion.dstSubresource.layerCount = 1;
        copyRegion.dstOffset = { 0, 0, 0 };
        copyRegion.extent = { swapChainExtent.width, swapChainExtent.height, 1 };

        vkCmdCopyImage(commandBuffers[commandBufferIndex],
            offscreenImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            swapChainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &copyRegion);

        // Transition swapchain image to present layout
        VkImageMemoryBarrier presentBarrier{};
        presentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        presentBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        presentBarrier.dstAccessMask = 0;
        presentBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        presentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        presentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        presentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        presentBarrier.image = swapChainImages[imageIndex];
        presentBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        presentBarrier.subresourceRange.baseMipLevel = 0;
        presentBarrier.subresourceRange.levelCount = 1;
        presentBarrier.subresourceRange.baseArrayLayer = 0;
        presentBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(commandBuffers[commandBufferIndex],
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &presentBarrier);

        if (vkEndCommandBuffer(commandBuffers[commandBufferIndex]) != VK_SUCCESS) {
            std::cerr << "Failed to end command buffer recording!" << std::endl;
        }
    }

    int createSyncObjects() {
        PROFILE_FUNCTION();
        // Create multiple sets of synchronization objects
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Start signaled so first frame doesn't wait

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                std::cerr << "Failed to create synchronization objects for frame " << i << "!" << std::endl;
                return VK_APP_ERROR_SYNC_OBJECTS_CREATION;
            }
        }

        return VK_APP_SUCCESS;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        PROFILE_FUNCTION();


        // Mailbox is the next best option if immediate is not available
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                std::cout << "Using VK_PRESENT_MODE_MAILBOX_KHR" << std::endl;
                return availablePresentMode;
            }
        }

        // FIFO is guaranteed to be available
        std::cout << "Using VK_PRESENT_MODE_FIFO_KHR (vsync on)" << std::endl;
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            SDL_GetWindowSizeInPixels(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        PROFILE_FUNCTION();
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        Uint32 count = 0;
        const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&count);
        if (!sdlExtensions) {
            std::cerr << "Failed to get required SDL Vulkan extensions" << std::endl;
            // Return empty vector, will cause instance creation to fail
            return {};
        }

        std::vector<const char*> extensions(sdlExtensions, sdlExtensions + count);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        PROFILE_FUNCTION();
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        PROFILE_FUNCTION();
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) {

        if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        }

        return VK_FALSE;
    }
};

// Vulkan Triangle App instance
vapp app;

// SDL callback functions for SDL_MAIN_USE_CALLBACKS
SDL_AppResult SDL_AppInit(void** appstate, int argc, char* argv[]) {
    std::cout << "Initializing Vulkan application..." << std::endl;

    // Initialize profiler settings if needed
    if (ENABLE_PROFILING) {
        std::cout << "Profiling is ENABLED - set to 0 to disable" << std::endl;
        Profiler::getInstance().setReportInterval(2.0); // Report every 2 seconds
        Profiler::getInstance().enableReporting(true);
    }
    else {
        std::cout << "Profiling is DISABLED - set to 1 to enable" << std::endl;
    }

    PROFILE_FUNCTION();
    int result = app.init();
    if (result != VK_APP_SUCCESS) {
        return SDL_APP_FAILURE;
    }

    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult result) {
    PROFILE_FUNCTION();
    std::cout << "Shutting down Vulkan application..." << std::endl;

    // Generate final profiling report if enabled
    if (ENABLE_PROFILING) {
        Profiler::getInstance().generateReport();
    }

    app.cleanup();
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* eve) {
    if (eve->type == SDL_EVENT_WINDOW_RESIZED) {
        app.setFramebufferResized();
    }
    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appstate) {
    int result = app.drawFrame();
    if (result != VK_APP_SUCCESS) {
        std::cerr << "Error during frame rendering: " << result << std::endl;
        // Don't terminate on render error, just log it
    }

    return SDL_APP_CONTINUE;
}

// Helper function implementations
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}