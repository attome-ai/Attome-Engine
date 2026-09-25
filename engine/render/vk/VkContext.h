#pragma once

// Instance, device, queues, VMA allocator and device capabilities
// (docs/GPU_3D_PLAN.md §4.3, §9.3). Created with vk-bootstrap.

#include "VkCommon.h"

#include <VkBootstrap.h>

#include <string>

struct SDL_Window;

namespace atm::render::vk {

struct DeviceCaps {
  VkFormat depthFormat = VK_FORMAT_UNDEFINED;
  VkSampleCountFlagBits msaa = VK_SAMPLE_COUNT_1_BIT;
  float timestampPeriod = 0.0f;       // ns per tick; 0 = no timestamps
  uint64_t timestampMask = 0;         // valid bits of the graphics queue
  VkDeviceSize maxStorageBufferRange = 0;
  VkDeviceSize minUniformAlign = 256;
  VkDeviceSize nonCoherentAtomSize = 256;
  std::string deviceName;
};

class VkContext {
public:
  bool init(SDL_Window *window, bool validation, int requestedMsaa, std::string *error);
  void shutdown();

  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue graphicsQueue = VK_NULL_HANDLE;
  uint32_t graphicsFamily = 0;
  VmaAllocator allocator = VK_NULL_HANDLE;
  DeviceCaps caps;

  vkb::Instance vkbInstance;
  vkb::Device vkbDevice;
};

} // namespace atm::render::vk
