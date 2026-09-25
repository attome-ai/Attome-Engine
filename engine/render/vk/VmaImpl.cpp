// The single VMA implementation TU. Static functions off, dynamic on:
// VkContext passes vkGetInstanceProcAddr / vkGetDeviceProcAddr in
// VmaVulkanFunctions so VMA fetches device-level entry points itself
// (no loader trampolines on the allocation paths).

#include <vulkan/vulkan.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
