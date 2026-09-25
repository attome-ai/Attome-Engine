#pragma once

// Swapchain + per-image "render finished" semaphores (one per swapchain image,
// not per frame in flight, so a semaphore is never re-signalled while the
// presentation engine may still wait on it).

#include "VkCommon.h"

#include <VkBootstrap.h>

#include <vector>

namespace atm::render::vk {

class VkContext;

class Swapchain {
public:
  // Creates or recreates (old swapchain passed to the builder). Caller must
  // ensure the device is idle. Returns false on failure or 0x0 size.
  bool create(VkContext &ctx, uint32_t width, uint32_t height, bool vsync);
  void destroy(VkContext &ctx);

  VkSwapchainKHR handle() const { return swapchain_.swapchain; }
  VkFormat format() const { return swapchain_.image_format; }
  VkExtent2D extent() const { return swapchain_.extent; }
  uint32_t imageCount() const { return uint32_t(images.size()); }
  bool srgb() const { return srgb_; }

  std::vector<VkImage> images;
  std::vector<VkImageView> views;
  std::vector<VkSemaphore> renderFinished; // per image
  std::vector<VkImageLayout> layouts;      // last known layout per image

private:
  void destroyViews(VkContext &ctx);
  vkb::Swapchain swapchain_{};
  bool srgb_ = false;
};

} // namespace atm::render::vk
