#include "VkContext.h"

#include <SDL3/SDL_error.h>
#include <SDL3/SDL_vulkan.h>

#include <vector>

namespace atm::render::vk {

const char *resultString(VkResult r) {
  switch (r) {
  case VK_SUCCESS: return "VK_SUCCESS";
  case VK_NOT_READY: return "VK_NOT_READY";
  case VK_TIMEOUT: return "VK_TIMEOUT";
  case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
  case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
  case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
  case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
  case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
  case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
  case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
  case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
  case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
  case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
  case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
  case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
  case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
  default: return "VkResult(other)";
  }
}

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT *data,
                                             void *) {
  const char *msg = data && data->pMessage ? data->pMessage : "(null)";
  if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "[vulkan] %s", msg);
  else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    SDL_LogWarn(SDL_LOG_CATEGORY_RENDER, "[vulkan] %s", msg);
  else
    SDL_LogDebug(SDL_LOG_CATEGORY_RENDER, "[vulkan] %s", msg);
  return VK_FALSE;
}

bool fail(std::string *error, const std::string &msg) {
  SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Renderer: %s", msg.c_str());
  if (error) *error = msg;
  return false;
}

} // namespace

bool VkContext::init(SDL_Window *window, bool validation, int requestedMsaa,
                     std::string *error) {
  // --- instance -------------------------------------------------------------
  vkb::InstanceBuilder ib(vkGetInstanceProcAddr);
  ib.set_app_name("Attome")
      .set_engine_name("Attome Engine")
      .require_api_version(1, 3, 0);
  Uint32 sdlExtCount = 0;
  const char *const *sdlExts = SDL_Vulkan_GetInstanceExtensions(&sdlExtCount);
  for (Uint32 i = 0; sdlExts && i < sdlExtCount; ++i) ib.enable_extension(sdlExts[i]);
  if (validation) {
    ib.request_validation_layers(true)
        .set_debug_callback(debugCallback)
        .set_debug_messenger_severity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
  }
  auto instRet = ib.build();
  if (!instRet) return fail(error, "Vulkan 1.3 instance: " + instRet.error().message());
  vkbInstance = instRet.value();
  instance = vkbInstance.instance;

  if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface))
    return fail(error, std::string("SDL_Vulkan_CreateSurface: ") + SDL_GetError());

  // --- physical device --------------------------------------------------------
  VkPhysicalDeviceFeatures f10{};
  f10.multiDrawIndirect = VK_TRUE;          // drawCount > 1
  f10.drawIndirectFirstInstance = VK_TRUE;  // firstInstance = chunk slot
  VkPhysicalDeviceVulkan12Features f12{};
  f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  f12.drawIndirectCount = VK_TRUE;
  VkPhysicalDeviceVulkan13Features f13{};
  f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  f13.dynamicRendering = VK_TRUE;
  f13.synchronization2 = VK_TRUE;

  vkb::PhysicalDeviceSelector sel(vkbInstance);
  sel.set_surface(surface)
      .set_minimum_version(1, 3)
      .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
      .set_required_features(f10)
      .set_required_features_12(f12)
      .set_required_features_13(f13);
  auto pdRet = sel.select();
  if (!pdRet)
    return fail(error, "No GPU with Vulkan 1.3 (dynamic rendering, synchronization2, "
                       "drawIndirectCount): " + pdRet.error().message());
  vkb::PhysicalDevice pd = pdRet.value();
  physicalDevice = pd.physical_device;

  // --- device + queues ------------------------------------------------------
  vkb::DeviceBuilder db(pd);
  auto devRet = db.build();
  if (!devRet) return fail(error, "vkCreateDevice: " + devRet.error().message());
  vkbDevice = devRet.value();
  device = vkbDevice.device;

  auto qRet = vkbDevice.get_queue(vkb::QueueType::graphics);
  auto qiRet = vkbDevice.get_queue_index(vkb::QueueType::graphics);
  if (!qRet || !qiRet) return fail(error, "no graphics queue");
  graphicsQueue = qRet.value();
  graphicsFamily = qiRet.value();

  // --- VMA --------------------------------------------------------------------
  VmaVulkanFunctions fns{};
  fns.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  fns.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
  VmaAllocatorCreateInfo aci{};
  aci.vulkanApiVersion = VK_API_VERSION_1_3;
  aci.instance = instance;
  aci.physicalDevice = physicalDevice;
  aci.device = device;
  aci.pVulkanFunctions = &fns;
  if (!ATM_VK_OK(vmaCreateAllocator(&aci, &allocator))) return fail(error, "vmaCreateAllocator");

  // --- capabilities (§9.3) -------------------------------------------------------
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physicalDevice, &props);
  caps.deviceName = props.deviceName;
  caps.maxStorageBufferRange = props.limits.maxStorageBufferRange;
  caps.minUniformAlign = props.limits.minUniformBufferOffsetAlignment;
  caps.nonCoherentAtomSize = props.limits.nonCoherentAtomSize;

  const VkFormat depthCandidates[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT,
                                      VK_FORMAT_D16_UNORM};
  for (VkFormat f : depthCandidates) {
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, f, &fp);
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
      caps.depthFormat = f;
      break;
    }
  }
  if (caps.depthFormat == VK_FORMAT_UNDEFINED) return fail(error, "no depth format");

  const VkSampleCountFlags both =
      props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
  int samples = requestedMsaa >= 8 ? 8 : requestedMsaa >= 4 ? 4 : requestedMsaa >= 2 ? 2 : 1;
  while (samples > 1 && !(both & VkSampleCountFlags(samples))) samples /= 2;
  caps.msaa = VkSampleCountFlagBits(samples);

  uint32_t qfCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, nullptr);
  std::vector<VkQueueFamilyProperties> qfs(qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, qfs.data());
  const uint32_t validBits = graphicsFamily < qfCount ? qfs[graphicsFamily].timestampValidBits : 0;
  if (validBits > 0 && props.limits.timestampPeriod > 0.0f) {
    caps.timestampPeriod = props.limits.timestampPeriod;
    caps.timestampMask = validBits >= 64 ? ~0ull : ((1ull << validBits) - 1ull);
  }

  SDL_Log("Renderer: %s, depth %d, MSAA %dx, timestamps %s", caps.deviceName.c_str(),
          int(caps.depthFormat), samples, caps.timestampPeriod > 0 ? "yes" : "n/a");
  return true;
}

void VkContext::shutdown() {
  if (allocator) {
    vmaDestroyAllocator(allocator);
    allocator = VK_NULL_HANDLE;
  }
  if (device) {
    vkb::destroy_device(vkbDevice);
    device = VK_NULL_HANDLE;
  }
  if (surface) {
    vkb::destroy_surface(vkbInstance, surface);
    surface = VK_NULL_HANDLE;
  }
  if (instance) {
    vkb::destroy_instance(vkbInstance);
    instance = VK_NULL_HANDLE;
  }
}

} // namespace atm::render::vk
