#include "VulkanRenderer.h"
#include "ATMEngine.h"


// VulkanRenderer implementation
VulkanRenderer::VulkanRenderer(Engine* engine)
    : engine(engine),
    instance(VK_NULL_HANDLE),
    debugMessenger(VK_NULL_HANDLE),
    surface(VK_NULL_HANDLE),
    physicalDevice(VK_NULL_HANDLE),
    device(VK_NULL_HANDLE),
    graphicsQueue(VK_NULL_HANDLE),
    presentQueue(VK_NULL_HANDLE),
    swapchain(VK_NULL_HANDLE),
    swapchainImageFormat(VK_FORMAT_UNDEFINED),
    renderPass(VK_NULL_HANDLE),
    descriptorSetLayout(VK_NULL_HANDLE),
    pipelineLayout(VK_NULL_HANDLE),
    graphicsPipeline(VK_NULL_HANDLE),
    commandPool(VK_NULL_HANDLE),
    descriptorPool(VK_NULL_HANDLE),
    uniformBuffer(VK_NULL_HANDLE),
    uniformBufferMemory(VK_NULL_HANDLE),
    currentFrame(0),
    framebufferResized(false) {

    // Get window dimensions
    SDL_GetWindowSize(engine->window, &windowWidth, &windowHeight);
    swapchainExtent = { static_cast<uint32_t>(windowWidth), static_cast<uint32_t>(windowHeight) };
}

VulkanRenderer::~VulkanRenderer() {
    // Wait for device to finish all operations before destroying anything
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
    }

    // Now clean up resources after ensuring all operations have completed
    cleanupSwapChain();

    // Clean up textures
    for (auto& texture : textures) {
        if (texture.sampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, texture.sampler, nullptr);
        }
        if (texture.view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, texture.view, nullptr);
        }
        if (texture.image != VK_NULL_HANDLE) {
            vkDestroyImage(device, texture.image, nullptr);
        }
        if (texture.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, texture.memory, nullptr);
        }
    }

    // Clean up batch resources
    for (auto& batch : batches) {
        if (batch.vertexBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, batch.vertexBuffer, nullptr);
        }
        if (batch.vertexMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, batch.vertexMemory, nullptr);
        }
        if (batch.indexBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, batch.indexBuffer, nullptr);
        }
        if (batch.indexMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, batch.indexMemory, nullptr);
        }

    }



    // Clean up uniform buffer
    if (uniformBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, uniformBuffer, nullptr);
    }
    if (uniformBufferMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, uniformBufferMemory, nullptr);
    }

    // Clean up descriptor resources
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }


    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

    // Clean up synchronization objects
    for (size_t i = 0; i < VulkanConfig::MAX_FRAMES_IN_FLIGHT; i++) {
        if (renderFinishedSemaphores[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        }
        if (imageAvailableSemaphores[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        }
        if (inFlightFences[i] != VK_NULL_HANDLE) {
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }
    }

    // Clean up command pool
    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
    }

    // Clean up pipeline
    if (graphicsPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
    }
    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    }
    if (renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, renderPass, nullptr);
    }

    // Clean up logical device
    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
    }

    // Clean up debug messenger
    if (debugMessenger != VK_NULL_HANDLE) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance, debugMessenger, nullptr);
        }
    }

    // Clean up surface
    if (surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
    }

    // Clean up instance
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }
}

bool VulkanRenderer::initialize() {

    // Initialize Vulkan resources
    if (!createInstance()) {
        std::cerr << "Failed to create Vulkan instance!" << std::endl;
        return false;
    }

    if (VulkanConfig::ENABLE_VALIDATION_LAYERS && !setupDebugMessenger()) {
        std::cerr << "Failed to set up debug messenger!" << std::endl;
        return false;
    }

    if (!createSurface()) {
        std::cerr << "Failed to create window surface!" << std::endl;
        return false;
    }

    if (!pickPhysicalDevice()) {
        std::cerr << "Failed to find a suitable GPU!" << std::endl;
        return false;
    }

    if (!createLogicalDevice()) {
        std::cerr << "Failed to create logical device!" << std::endl;
        return false;
    }

    if (!createSwapChain()) {
        std::cerr << "Failed to create swap chain!" << std::endl;
        return false;
    }

    if (!createImageViews()) {
        std::cerr << "Failed to create image views!" << std::endl;
        return false;
    }

    if (!createRenderPass()) {
        std::cerr << "Failed to create render pass!" << std::endl;
        return false;
    }

    if (!createDescriptorSetLayout()) {
        std::cerr << "Failed to create descriptor set layout!" << std::endl;
        return false;
    }

    if (!createGraphicsPipeline()) {
        std::cerr << "Failed to create graphics pipeline!" << std::endl;
        return false;
    }

    if (!createFramebuffers()) {
        std::cerr << "Failed to create framebuffers!" << std::endl;
        return false;
    }

    if (!createCommandPool()) {
        std::cerr << "Failed to create command pool!" << std::endl;
        return false;
    }

    if (!createUniformBuffer()) {
        std::cerr << "Failed to create uniform buffer!" << std::endl;
        return false;
    }

    if (!createDescriptorPool()) {
        std::cerr << "Failed to create descriptor pool!" << std::endl;
        return false;
    }

    if (!createDescriptorSets()) {
        std::cerr << "Failed to create descriptor sets!" << std::endl;
        return false;
    }

    if (!createCommandBuffers()) {
        std::cerr << "Failed to create command buffers!" << std::endl;
        return false;
    }

    if (!createSyncObjects()) {
        std::cerr << "Failed to create synchronization objects!" << std::endl;
        return false;
    }

    return true;
}

bool VulkanRenderer::createInstance() {

    if (VulkanConfig::ENABLE_VALIDATION_LAYERS && !checkValidationLayerSupport()) {
        std::cerr << "Validation layers requested, but not available!" << std::endl;
        return false;
    }

    // Application info
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "2D Game Engine";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "ATM Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    // Create info for instance
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // Get required extensions
    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    // Set up validation layers
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (VulkanConfig::ENABLE_VALIDATION_LAYERS) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(VulkanConfig::validationLayers.size());
        createInfo.ppEnabledLayerNames = VulkanConfig::validationLayers.data();

        // Set up debug messenger for instance creation and destruction
        debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCreateInfo.pfnUserCallback = debugCallback;
        createInfo.pNext = &debugCreateInfo;
    }
    else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    // Create the Vulkan instance
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool VulkanRenderer::setupDebugMessenger() {

    if (!VulkanConfig::ENABLE_VALIDATION_LAYERS) return true;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;

    // Get function pointer for creating debug messenger
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func == nullptr) {
        return false;
    }

    // Create debug messenger
    if (func(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool VulkanRenderer::createSurface() {

    // Use SDL to create the Vulkan surface
    if (SDL_Vulkan_CreateSurface(engine->window, instance, nullptr, &surface) != true) {
        std::cerr << "Failed to create Vulkan surface: " << SDL_GetError() << std::endl;
        return false;
    }

    return true;
}

bool VulkanRenderer::pickPhysicalDevice() {

    // Get number of available physical devices
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        std::cerr << "Failed to find GPUs with Vulkan support!" << std::endl;
        return false;
    }

    // Get all available physical devices
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Find a suitable device
    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            physicalDevice = device;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        std::cerr << "Failed to find a suitable GPU!" << std::endl;
        return false;
    }

    // Get memory properties for later use
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    return true;
}

bool VulkanRenderer::createLogicalDevice() {

    // Get queue family indices
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    // Create a set of unique queue families
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {
        indices.graphicsFamily.value(),
        indices.presentFamily.value()
    };

    // Queue priority
    float queuePriority = 1.0f;

    // Create queue create info for each unique family
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    // Device features
    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    // Device create info
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;

    // Enable device extensions (like swapchain)
    createInfo.enabledExtensionCount = static_cast<uint32_t>(VulkanConfig::deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = VulkanConfig::deviceExtensions.data();

    // Enable validation layers if needed
    if (VulkanConfig::ENABLE_VALIDATION_LAYERS) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(VulkanConfig::validationLayers.size());
        createInfo.ppEnabledLayerNames = VulkanConfig::validationLayers.data();
    }
    else {
        createInfo.enabledLayerCount = 0;
    }

    // Create the logical device
    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        return false;
    }

    // Get queue handles
    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

    return true;
}

bool VulkanRenderer::createSwapChain() {
    PROFILE_FUNCTION();

    // Query swapchain support
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

    // Choose the best settings for the swapchain
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    // Decide how many images we want in the swapchain
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    // Create swapchain info
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    // Set up queue family indices
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

    // If we're recreating the swapchain, we need to specify the old one
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    // Create swapchain
    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS) {
        std::cerr << "Failed to create swap chain!" << std::endl;
        return false;
    }

    // Get swapchain images
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

    swapchainImageFormat = surfaceFormat.format;
    swapchainExtent = extent;

    return true;
}
bool VulkanRenderer::createImageViews() {

    swapchainImageViews.resize(swapchainImages.size());

    for (size_t i = 0; i < swapchainImages.size(); i++) {
        swapchainImageViews[i] = createImageView(swapchainImages[i], swapchainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        if (swapchainImageViews[i] == VK_NULL_HANDLE) {
            return false;
        }
    }

    return true;
}

bool VulkanRenderer::createRenderPass() {

    // Color attachment description
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapchainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // Attachment reference
    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Subpass description
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    // Subpass dependency
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    // Render pass create info
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    // Create render pass
    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        return false;
    }

    return true;
}

// Fix 1: Create a single descriptor set layout with both bindings
bool VulkanRenderer::createDescriptorSetLayout() {
    // Create bindings array with both texture sampler and uniform buffer
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};

    // Binding 0: Texture sampler
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[0].pImmutableSamplers = nullptr;

    // Binding 1: Uniform buffer (matches shader's expectation)
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    bindings[1].pImmutableSamplers = nullptr;

    // Create descriptor set layout with both bindings
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        return false;
    }

    return true;
}
bool VulkanRenderer::createGraphicsPipeline() {

    try {
        // Load shader bytecode
        auto vertShaderCode = readFile(VulkanConfig::VERTEX_SHADER_PATH);
        auto fragShaderCode = readFile(VulkanConfig::FRAGMENT_SHADER_PATH);

        // Create shader modules
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        // Create shader stage info
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
        auto bindingDescription = VulkanVertex::getBindingDescription();
        auto attributeDescriptions = VulkanVertex::getAttributeDescriptions();

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

        // Viewport and scissor
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapchainExtent.width);
        viewport.height = static_cast<float>(swapchainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapchainExtent;

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
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        // Multisampling
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Color blending
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        // Fix 2: Use a single descriptor set layout for the pipeline
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
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
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create graphics pipeline!");
        }

        // Clean up shader modules
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);

        return true;
    }
    catch (std::exception& e) {
        std::cerr << "Error creating graphics pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool VulkanRenderer::createFramebuffers() {

    swapchainFramebuffers.resize(swapchainImageViews.size());

    for (size_t i = 0; i < swapchainImageViews.size(); i++) {
        VkImageView attachments[] = {
            swapchainImageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapchainExtent.width;
        framebufferInfo.height = swapchainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapchainFramebuffers[i]) != VK_SUCCESS) {
            return false;
        }
    }

    return true;
}

bool VulkanRenderer::createCommandPool() {

    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool VulkanRenderer::createUniformBuffer() {

    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    if (!createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        uniformBuffer, uniformBufferMemory)) {
        return false;
    }

    return true;
}

bool VulkanRenderer::createDescriptorPool() {

    // We need one descriptor set per texture plus one for the uniform buffer
    uint32_t maxSets = VulkanConfig::MAX_DESCRIPTOR_SETS;

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = maxSets;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = maxSets;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxSets;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool VulkanRenderer::createDescriptorSets() {

    return true;
}

bool VulkanRenderer::createCommandBuffers() {

    commandBuffers.resize(VulkanConfig::MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool VulkanRenderer::createSyncObjects() {
    imageAvailableSemaphores.resize(VulkanConfig::MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(VulkanConfig::MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(VulkanConfig::MAX_FRAMES_IN_FLIGHT);

    // Create a fence for each swapchain image (not just in-flight frames)
    // Initialize these fences to the signaled state
    imageFences.resize(swapchainImages.size(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Start signaled so we don't wait forever

    for (size_t i = 0; i < VulkanConfig::MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            return false;
        }
    }

    return true;
}



// Replace the updateUniformBuffer() function with this implementation:
void VulkanRenderer::updateUniformBuffer() {
    PROFILE_SCOPE("updateUniformBuffer");

    UniformBufferObject ubo{};

    // Set up orthographic projection matrix
    float left = 0.0f;
    float right = static_cast<float>(swapchainExtent.width);
    float bottom = static_cast<float>(swapchainExtent.height);
    float top = 0.0f;
    float near = -1.0f;
    float far = 1.0f;

    // Orthographic projection matrix (column-major)
    ubo.projectionMatrix[0] = 2.0f / (right - left);
    ubo.projectionMatrix[1] = 0.0f;
    ubo.projectionMatrix[2] = 0.0f;
    ubo.projectionMatrix[3] = 0.0f;

    ubo.projectionMatrix[4] = 0.0f;
    ubo.projectionMatrix[5] = 2.0f / (top - bottom);
    ubo.projectionMatrix[6] = 0.0f;
    ubo.projectionMatrix[7] = 0.0f;

    ubo.projectionMatrix[8] = 0.0f;
    ubo.projectionMatrix[9] = 0.0f;
    ubo.projectionMatrix[10] = 1.0f / (far - near);
    ubo.projectionMatrix[11] = 0.0f;

    ubo.projectionMatrix[12] = -(right + left) / (right - left);
    ubo.projectionMatrix[13] = -(top + bottom) / (top - bottom);
    ubo.projectionMatrix[14] = -near / (far - near);
    ubo.projectionMatrix[15] = 1.0f;

    // Update the uniform buffer
    void* data;
    vkMapMemory(device, uniformBufferMemory, 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device, uniformBufferMemory);

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// VulkanBatch method implementations
void VulkanBatch::addQuad(float x, float y, float w, float h, SDL_FRect texRegion) {
    // Safety check - skip if we would exceed limits (shouldn't happen with proper sizing)
    if (vertexCount + 4 > VulkanConfig::MAX_VERTICES ||
        indexCount + 6 > VulkanConfig::MAX_INDICES) {
        return;
    }

    // Get base index for this quad
    uint32_t baseIndex = vertexCount;

    // Pack color (white) - 0xAABBGGRR format (RGBA8) - predefined constant
    constexpr uint32_t packedColor = 0xFFFFFFFF;

    // Precompute values to avoid redundant calculations
    const float texRight = texRegion.x + texRegion.w;
    const float texBottom = texRegion.y + texRegion.h;
    const float xRight = x + w;
    const float yBottom = y + h;

    // Top-left vertex (0)
    vertices[baseIndex].setPosition(x, y);
    vertices[baseIndex].setTexCoord(texRegion.x, texRegion.y);
    vertices[baseIndex].color = packedColor;

    // Top-right vertex (1)
    vertices[baseIndex + 1].setPosition(xRight, y);
    vertices[baseIndex + 1].setTexCoord(texRight, texRegion.y);
    vertices[baseIndex + 1].color = packedColor;

    // Bottom-right vertex (2)
    vertices[baseIndex + 2].setPosition(xRight, yBottom);
    vertices[baseIndex + 2].setTexCoord(texRight, texBottom);
    vertices[baseIndex + 2].color = packedColor;

    // Bottom-left vertex (3)
    vertices[baseIndex + 3].setPosition(x, yBottom);
    vertices[baseIndex + 3].setTexCoord(texRegion.x, texBottom);
    vertices[baseIndex + 3].color = packedColor;

    // Add indices for two triangles (6 indices total)
    indices[indexCount] = baseIndex;        // Top-left
    indices[indexCount + 1] = baseIndex + 1;  // Top-right
    indices[indexCount + 2] = baseIndex + 2;  // Bottom-right

    indices[indexCount + 3] = baseIndex;      // Top-left
    indices[indexCount + 4] = baseIndex + 2;  // Bottom-right
    indices[indexCount + 5] = baseIndex + 3;  // Bottom-left

    // Update counts
    vertexCount += 4;
    indexCount += 6;

    // Mark batch as needing update
    needsUpdate = true;
}
void VulkanBatch::clear() {
    vertexCount = 0;
    indexCount = 0;
    needsUpdate = true;
}



void VulkanRenderer::renderScene() {
    // Check if window is minimized
    if (windowWidth == 0 || windowHeight == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return;
    }

    // Wait for the previous frame to finish
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // Attempt to acquire an image from the swapchain
    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
        imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    // Handle swapchain recreation
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        vkDeviceWaitIdle(device);

        if (recreateSwapChain()) {
            return;
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return;
        }
    }
    else if (result != VK_SUCCESS) {
        static int consecutiveFailures = 0;
        consecutiveFailures++;

        if (consecutiveFailures > 3) {
            consecutiveFailures = 0;
            vkDeviceWaitIdle(device);
            recreateSwapChain();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return;
    }

    // Reset consecutive failures counter on success
    static int consecutiveFailures = 0;
    consecutiveFailures = 0;

    // If this image has a fence associated with it, wait for it to be signaled
    if (imageFences[imageIndex] != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &imageFences[imageIndex], VK_TRUE, UINT64_MAX);
    }

    // Mark this image as now being in use by this frame's fence
    imageFences[imageIndex] = inFlightFences[currentFrame];

    // Reset the fence for this frame
    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    // Clear batches that need to be cleared
    for (auto& batch : batches) {
        if (batch.vertexCount > 0) { // Changed from !batch.vertices.empty()
            batch.clear();
        }
    }

    // Pre-calculate camera bounds once (optimization)
    const float camX = engine->camera.x;
    const float camY = engine->camera.y;
    const float halfWidth = engine->camera.width * 0.5f;
    const float halfHeight = engine->camera.height * 0.5f;

    // Query visible area with a small margin (reduced from 100 to 50 for optimization)
    const float x1 = camX - halfWidth - 50.0f;
    const float y1 = camY - halfHeight - 50.0f;
    const float x2 = camX + halfWidth + 50.0f;
    const float y2 = camY + halfHeight + 50.0f;

    // Query the spatial grid for entities in the visible area
    std::vector<EntityRef>& visible_entities = engine->grid.queryRect(x1, y1, x2, y2);

    // Sort entities by z-index and other attributes
    // Using lambda capture by reference for engine to avoid repetitive access
    auto& entityManager = engine->entityManager;
    std::sort(visible_entities.begin(), visible_entities.end(),
        [&entityManager](const EntityRef& a, const EntityRef& b) {
            auto containerA = entityManager.containers[a.type].get();
            auto containerB = entityManager.containers[b.type].get();

            bool renderableA = containerA->containerFlag & (uint8_t)ContainerFlag::RENDERABLE;
            bool renderableB = containerB->containerFlag & (uint8_t)ContainerFlag::RENDERABLE;

            if (!renderableA && renderableB) return false;
            if (renderableA && !renderableB) return true;
            if (!renderableA && !renderableB) return false;

            RenderableEntityContainer* rContainerA = reinterpret_cast<RenderableEntityContainer*>(containerA);
            RenderableEntityContainer* rContainerB = reinterpret_cast<RenderableEntityContainer*>(containerB);

            uint8_t zIndexA = rContainerA->z_indices[a.index];
            uint8_t zIndexB = rContainerB->z_indices[b.index];

            if (zIndexA != zIndexB) return zIndexA < zIndexB;
            if (a.type != b.type) return a.type < b.type;
            return a.index < b.index;
        });

    // Cache frequently accessed values
    const float screenCenterX = halfWidth;
    const float screenCenterY = halfHeight;

    // Process visible entities and build batch data
    uint64_t lastBatchKey = 0;
    VulkanBatch* currentBatch = nullptr;

    for (const EntityRef& entity : visible_entities) {
        auto container = entityManager.containers[entity.type].get();
        if (!(container->containerFlag & (uint8_t)ContainerFlag::RENDERABLE))
            continue;

        RenderableEntityContainer* renderableContainer = static_cast<RenderableEntityContainer*>(container);

        // Calculate screen coordinates (minimizing calculations)
        float entityX = renderableContainer->x_positions[entity.index];
        float entityY = renderableContainer->y_positions[entity.index];
        float width = renderableContainer->widths[entity.index];
        float height = renderableContainer->heights[entity.index];

        // Convert world to screen coordinates
        float screenX = entityX - camX + screenCenterX;
        float screenY = entityY - camY + screenCenterY;

        // Precise culling - skip if outside camera view
        if (screenX + width < 0 || screenX > engine->camera.width ||
            screenY + height < 0 || screenY > engine->camera.height)
            continue;

        // Batch key calculation (optimization to avoid getBatch calls)
        int textureId = renderableContainer->texture_ids[entity.index];
        int zIndex = renderableContainer->z_indices[entity.index];
        uint64_t batchKey = (static_cast<uint64_t>(textureId) << 32) | static_cast<uint64_t>(zIndex);

        // Only look up batch if it's different from the last one
        if (!currentBatch || batchKey != lastBatchKey) {
            auto it = batchMap.find(batchKey);
            if (it != batchMap.end()) {
                currentBatch = &batches[it->second];
            }
            else {
                // Create a new batch if needed
                size_t newIndex = batches.size();
                batches.emplace_back(textureId, zIndex);
                batchMap[batchKey] = newIndex;
                currentBatch = &batches[newIndex];
            }
            lastBatchKey = batchKey;
        }

        // Get texture region directly from textures array (avoiding function call)
        SDL_FRect texRegion;
        if (textureId >= 0 && textureId < textures.size()) {
            texRegion = textures[textureId].region;
        }
        else {
            texRegion = { 0, 0, 1, 1 }; // Default region
        }

        // Add quad to current batch
        currentBatch->addQuad(screenX, screenY, width, height, texRegion);
    }

    // Update batch buffers and wait for completion
    updateBatchBuffers();

    // Update uniform buffer for camera projection
    updateUniformBuffer();

    // Record command buffer for rendering
    vkResetCommandBuffer(commandBuffers[currentFrame], 0);
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

    // Submit command buffer to graphics queue
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

    VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
        std::cerr << "Failed to submit draw command buffer!" << std::endl;
        return;
    }

    // Present rendered image
    present(imageIndex);
}

void VulkanRenderer::present(uint32_t imageIndex) {
    PROFILE_SCOPE("present");

    // Present the rendered image
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapchains[] = { swapchain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;

    VkResult result = vkQueuePresentKHR(presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        // Mark for recreation on next frame, don't recreate here to avoid doing it twice
        framebufferResized = true;
    }
    else if (result != VK_SUCCESS) {
        std::cerr << "Failed to present swapchain image! Error code: " << result << std::endl;
    }

    // Update current frame index
    currentFrame = (currentFrame + 1) % VulkanConfig::MAX_FRAMES_IN_FLIGHT;
}


int VulkanRenderer::registerTexture(SDL_Surface* surface, int x, int y, int width, int height) {
    PROFILE_SCOPE("registerTexture");

    // Create a new texture
    VulkanTexture texture{};

    // Set texture region
    float atlas_width = VulkanConfig::TEXTURE_ATLAS_WIDTH;
    float atlas_height = VulkanConfig::TEXTURE_ATLAS_HEIGHT;

    int tex_width = (width > 0) ? width : surface->w;
    int tex_height = (height > 0) ? height : surface->h;

    texture.region = {
        static_cast<float>(x) / atlas_width,
        static_cast<float>(y) / atlas_height,
        static_cast<float>(tex_width) / atlas_width,
        static_cast<float>(tex_height) / atlas_height
    };

    // Create texture image
    if (!createTextureImage(surface, texture)) {
        std::cerr << "Failed to create texture image!" << std::endl;
        return -1;
    }

    // Create texture image view
    if (!createTextureImageView(texture)) {
        std::cerr << "Failed to create texture image view!" << std::endl;
        return -1;
    }

    // Create texture sampler
    if (!createTextureSampler(texture)) {
        std::cerr << "Failed to create texture sampler!" << std::endl;
        return -1;
    }

    // Create one descriptor set per frame in flight
    texture.descriptorSets.resize(VulkanConfig::MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < VulkanConfig::MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &texture.descriptorSets[i]) != VK_SUCCESS) {
            std::cerr << "Failed to allocate texture descriptor set!" << std::endl;
            return -1;
        }

        // Initialize descriptor sets with both the texture and uniform buffer
        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

        // Descriptor for texture (binding 0)
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = texture.view;
        imageInfo.sampler = texture.sampler;

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = texture.descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pImageInfo = &imageInfo;

        // Descriptor for UBO (binding 1)
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = texture.descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &bufferInfo;

        // Update descriptor set with both texture and UBO
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    // Add the texture to the list
    int textureId = textures.size();
    textures.push_back(texture);

    return textureId;
}


void VulkanRenderer::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    // Begin command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = nullptr;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        std::cerr << "Failed to begin recording command buffer!" << std::endl;
        return;
    }

    // Begin render pass
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapchainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = swapchainExtent;

    // Set clear color (background color - e.g., black)
    VkClearValue clearColor = { {{0.1f, 0.1f, 0.1f, 1.0f}} };  // Dark gray background
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Bind the graphics pipeline
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    // Draw each batch
    for (const auto& batch : batches) {
        // Skip empty batches or batches without buffers
        if (batch.vertexCount == 0 || batch.indexCount == 0 ||  // Changed from empty() checks
            batch.vertexBuffer == VK_NULL_HANDLE || batch.indexBuffer == VK_NULL_HANDLE) {
            continue;
        }

        // Get the texture for this batch
        if (batch.textureId >= textures.size()) {
            continue;
        }

        // Use the descriptor set for the current frame
        VkDescriptorSet textureDescriptorSet = textures[batch.textureId].descriptorSets[currentFrame];
        if (textureDescriptorSet == VK_NULL_HANDLE) {
            continue;
        }

        // Bind the vertex buffer
        VkBuffer vertexBuffers[] = { batch.vertexBuffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

        // Bind the index buffer
        vkCmdBindIndexBuffer(commandBuffer, batch.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // Only bind one descriptor set which contains both the texture sampler (binding 0) 
        // and the uniform buffer (binding 1)
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
            0, 1, &textureDescriptorSet, 0, nullptr);

        // Draw the vertices - use indexCount instead of indices.size()
        vkCmdDrawIndexed(commandBuffer, batch.indexCount, 1, 0, 0, 0);
    }

    // End render pass
    vkCmdEndRenderPass(commandBuffer);

    // End command buffer
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        std::cerr << "Failed to record command buffer!" << std::endl;
    }
}

SDL_FRect VulkanRenderer::getTextureRegion(int textureId) {
    PROFILE_FUNCTION();

    if (textureId >= 0 && textureId < textures.size()) {
        return textures[textureId].region;
    }

    // Return default texture region if invalid
    return { 0, 0, 1, 1 };
}

bool VulkanRenderer::createTextureImage(SDL_Surface* surface, VulkanTexture& texture) {
    PROFILE_FUNCTION();

    // Get texture dimensions
    texture.width = surface->w;
    texture.height = surface->h;

    // Calculate buffer size
    VkDeviceSize imageSize = texture.width * texture.height * 4;  // RGBA format

    // Create staging buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    if (!createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingBufferMemory)) {
        return false;
    }

    // Map memory and copy pixel data
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);

    // Convert SDL surface to RGBA format if needed
    SDL_Surface* rgbaSurface = nullptr;

    if (surface->format != SDL_PIXELFORMAT_RGBA32) {
        rgbaSurface = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_RGBA32);
        if (!rgbaSurface) {
            std::cerr << "Failed to convert surface to RGBA format: " << SDL_GetError() << std::endl;
            vkUnmapMemory(device, stagingBufferMemory);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
            return false;
        }

        memcpy(data, rgbaSurface->pixels, imageSize);
        SDL_DestroySurface(rgbaSurface);
    }
    else {
        memcpy(data, surface->pixels, imageSize);
    }

    vkUnmapMemory(device, stagingBufferMemory);

    // Create the actual texture image
    if (!createImage(texture.width, texture.height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texture.image, texture.memory)) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
        return false;
    }

    // Transition image layout and copy buffer to image
    transitionImageLayout(texture.image, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(stagingBuffer, texture.image, texture.width, texture.height);
    transitionImageLayout(texture.image, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Clean up staging buffer
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    return true;
}

bool VulkanRenderer::createTextureImageView(VulkanTexture& texture) {
    PROFILE_SCOPE("createTextureImageView");

    texture.view = createImageView(texture.image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    return texture.view != VK_NULL_HANDLE;
}

bool VulkanRenderer::createTextureSampler(VulkanTexture& texture) {
    PROFILE_SCOPE("createTextureSampler");

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;  // For pixel art, use NEAREST
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &texture.sampler) != VK_SUCCESS) {
        std::cerr << "Failed to create texture sampler!" << std::endl;
        return false;
    }

    return true;
}

void VulkanRenderer::handleResize(int width, int height) {
    PROFILE_FUNCTION();

    windowWidth = width;
    windowHeight = height;
    framebufferResized = true;
}

void VulkanRenderer::cleanupSwapChain() {
    PROFILE_FUNCTION();

    // Clean up framebuffers
    for (auto framebuffer : swapchainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    // Clean up image views
    for (auto imageView : swapchainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }

    // Clean up swapchain
    if (swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
}

bool VulkanRenderer::recreateSwapChain() {
    PROFILE_SCOPE("recreateSwapChain");

    // Handle minimized window
    int width, height;
    SDL_GetWindowSize(engine->window, &width, &height);
    while (width == 0 || height == 0) {
        // Window is minimized - wait until it's restored
        SDL_GetWindowSize(engine->window, &width, &height);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    windowWidth = width;
    windowHeight = height;

    // Wait for device to finish operations
    vkDeviceWaitIdle(device);

    // Reset all fences to be safe
    for (size_t i = 0; i < VulkanConfig::MAX_FRAMES_IN_FLIGHT; i++) {
        if (inFlightFences[i] != VK_NULL_HANDLE) {
            vkResetFences(device, 1, &inFlightFences[i]);
        }
    }

    // Clean up old swapchain
    cleanupSwapChain();

    // Create new swapchain
    if (!createSwapChain()) {
        std::cerr << "Failed to create swapchain" << std::endl;
        return false;
    }

    if (!createImageViews()) {
        std::cerr << "Failed to create image views" << std::endl;
        return false;
    }

    if (!createFramebuffers()) {
        std::cerr << "Failed to create framebuffers" << std::endl;
        return false;
    }

    // Reset and resize imageFences array to match the new swapchain
    imageFences.resize(swapchainImages.size(), VK_NULL_HANDLE);

    // Update the uniform buffer to match new swapchain dimensions
    updateUniformBuffer();

    return true;
}
// Helper methods
bool VulkanRenderer::checkValidationLayerSupport() {
    PROFILE_SCOPE("checkValidationLayerSupport");

    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : VulkanConfig::validationLayers) {
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

std::vector<const char*> VulkanRenderer::getRequiredExtensions() {
    PROFILE_SCOPE("findQueueFamilies");

    // Get required extensions from SDL
    unsigned int count;
    const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&count);

    std::vector<const char*> extensions(sdlExtensions, sdlExtensions + count);

    // Add debug extension if validation layers are enabled
    if (VulkanConfig::ENABLE_VALIDATION_LAYERS) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

QueueFamilyIndices VulkanRenderer::findQueueFamilies(VkPhysicalDevice device) {
    PROFILE_SCOPE("findQueueFamilies");

    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    // Find queue family with graphics support
    for (int i = 0; i < queueFamilies.size(); i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }

        // Check if this queue family supports presentation
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

        if (presentSupport) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }
    }

    return indices;
}

bool VulkanRenderer::isDeviceSuitable(VkPhysicalDevice device) {
    PROFILE_SCOPE("isDeviceSuitable");

    // Check if device supports required queue families
    QueueFamilyIndices indices = findQueueFamilies(device);

    // Check if device supports required extensions
    bool extensionsSupported = [device]() {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(VulkanConfig::deviceExtensions.begin(), VulkanConfig::deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
        }();

    // Check if device supports swapchain adequately
    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    // Check device features
    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

    return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}

SwapChainSupportDetails VulkanRenderer::querySwapChainSupport(VkPhysicalDevice device) {
    PROFILE_SCOPE("querySwapChainSupport");

    SwapChainSupportDetails details;

    // Get surface capabilities
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    // Get supported surface formats
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    // Get supported presentation modes
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR VulkanRenderer::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    PROFILE_SCOPE("chooseSwapSurfaceFormat");

    // Prefer SRGB format for better color accuracy
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    // If preferred format not available, just use the first one
    return availableFormats[0];
}

VkPresentModeKHR VulkanRenderer::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    PROFILE_SCOPE("chooseSwapPresentMode");

    // Prefer mailbox mode (triple buffering) for lower latency
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }

    // FIFO (V-Sync) is guaranteed to be available
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanRenderer::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    PROFILE_SCOPE("chooseSwapExtent");

    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }
    else {
        VkExtent2D actualExtent = { static_cast<uint32_t>(windowWidth), static_cast<uint32_t>(windowHeight) };

        actualExtent.width = std::clamp(actualExtent.width,
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height,
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

VkShaderModule VulkanRenderer::createShaderModule(const std::vector<char>& code) {
    PROFILE_SCOPE("createShaderModule");

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }

    return shaderModule;
}

uint32_t VulkanRenderer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    PROFILE_FUNCTION();

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

bool VulkanRenderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    PROFILE_SCOPE("createBuffer");

    // Create buffer
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        std::cerr << "Failed to create buffer!" << std::endl;
        return false;
    }

    // Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    // Allocate memory
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        std::cerr << "Failed to allocate buffer memory!" << std::endl;
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }

    // Bind memory to buffer
    if (vkBindBufferMemory(device, buffer, bufferMemory, 0) != VK_SUCCESS) {
        std::cerr << "Failed to bind buffer memory!" << std::endl;
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, bufferMemory, nullptr);
        buffer = VK_NULL_HANDLE;
        bufferMemory = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

bool VulkanRenderer::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
    PROFILE_SCOPE("createImage");

    // Create image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.flags = 0;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        std::cerr << "Failed to create image!" << std::endl;
        return false;
    }

    // Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    // Allocate memory
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        std::cerr << "Failed to allocate image memory!" << std::endl;
        vkDestroyImage(device, image, nullptr);
        return false;
    }

    // Bind memory to image
    if (vkBindImageMemory(device, image, imageMemory, 0) != VK_SUCCESS) {
        std::cerr << "Failed to bind image memory!" << std::endl;
        vkDestroyImage(device, image, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
        return false;
    }

    return true;
}

VkImageView VulkanRenderer::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
    PROFILE_SCOPE("createImageView");

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    return imageView;
}

void VulkanRenderer::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    PROFILE_SCOPE("copyBuffer");

    // Create command buffer for the copy operation
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    // Begin recording command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // Record copy command
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    // End recording
    vkEndCommandBuffer(commandBuffer);

    // Submit command buffer
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    // Free command buffer
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void VulkanRenderer::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    PROFILE_SCOPE("transitionImageLayout");

    // Create command buffer for the transition
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    // Begin recording command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // Create image barrier
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    // Set up barrier based on layouts
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else {
        throw std::invalid_argument("Unsupported layout transition!");
    }

    // Record barrier command
    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    // End recording
    vkEndCommandBuffer(commandBuffer);

    // Submit command buffer
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    // Free command buffer
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void VulkanRenderer::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    PROFILE_SCOPE("copyBufferToImage");

    // Create command buffer for the copy operation
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    // Begin recording command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // Define copy region
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { width, height, 1 };

    // Record copy command
    vkCmdCopyBufferToImage(
        commandBuffer,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );

    // End recording
    vkEndCommandBuffer(commandBuffer);

    // Submit command buffer
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    // Free command buffer
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanRenderer::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
    }

    return VK_FALSE;
}
VulkanBatch& VulkanRenderer::getBatch(int textureId, int zIndex) {
    PROFILE_SCOPE("getBatch");

    // Create a 64-bit key from textureId and zIndex
    uint64_t key = (static_cast<uint64_t>(textureId) << 32) | static_cast<uint64_t>(zIndex);

    // Check if a batch for this key already exists
    auto it = batchMap.find(key);
    auto m = batchMap;

    if (it != batchMap.end()) {
        return batches[it->second];
    }
    // Create a new batch
    size_t newIndex = batches.size();
    batches.emplace_back(textureId, zIndex);
    batchMap[key] = newIndex;

    return batches[newIndex];
}


void VulkanRenderer::updateBatchBuffers() {
    // First pass: Check if any batches need updates
    bool needsUpdate = false;
    for (auto& batch : batches) {
        if (batch.vertexCount > 0 && batch.needsUpdate) {  // Changed from !batch.vertices.empty()
            needsUpdate = true;
            break;
        }
    }
    if (!needsUpdate) return;

    // Process each batch that needs an update
    for (auto& batch : batches) {
        if (batch.vertexCount == 0 || !batch.needsUpdate) {  // Changed from batch.vertices.empty()
            continue;
        }

        // Calculate buffer sizes - use vertexCount and indexCount instead of size()
        VkDeviceSize vertexBufferSize = sizeof(VulkanVertex) * batch.vertexCount;
        VkDeviceSize indexBufferSize = sizeof(uint32_t) * batch.indexCount;

        // ---- VERTEX BUFFER ----
        // Only recreate if NULL or needs to grow - using MAX_VERTICES constant instead of capacity()
        if (batch.vertexBuffer == VK_NULL_HANDLE) {
            // Create host-visible buffer for all possible vertices
            if (!createBuffer(
                sizeof(VulkanVertex) * VulkanConfig::MAX_VERTICES,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                batch.vertexBuffer,
                batch.vertexMemory
            )) {
                std::cerr << "Failed to create vertex buffer!" << std::endl;
                continue; // Skip this batch if buffer creation failed
            }
        }

        // ---- INDEX BUFFER ----
        // Only recreate if NULL - using MAX_INDICES constant
        if (batch.indexBuffer == VK_NULL_HANDLE) {
            // Create host-visible buffer for all possible indices
            if (!createBuffer(
                sizeof(uint32_t) * VulkanConfig::MAX_INDICES,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                batch.indexBuffer,
                batch.indexMemory
            )) {
                std::cerr << "Failed to create index buffer!" << std::endl;
                continue; // Skip this batch if buffer creation failed
            }
        }

        // Copy vertex data directly to the GPU-accessible memory
        if (batch.vertexBuffer != VK_NULL_HANDLE) {
            void* data;
            vkMapMemory(device, batch.vertexMemory, 0, vertexBufferSize, 0, &data);
            memcpy(data, batch.vertices, vertexBufferSize);  // Using batch.vertices directly (array)
            vkUnmapMemory(device, batch.vertexMemory);
        }

        // Copy index data directly to the GPU-accessible memory
        if (batch.indexBuffer != VK_NULL_HANDLE) {
            void* data;
            vkMapMemory(device, batch.indexMemory, 0, indexBufferSize, 0, &data);
            memcpy(data, batch.indices, indexBufferSize);  // Using batch.indices directly (array)
            vkUnmapMemory(device, batch.indexMemory);
        }

        // Mark as updated
        batch.needsUpdate = false;
    }
}