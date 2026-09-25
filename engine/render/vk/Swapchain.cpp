#include "Swapchain.h"

#include "VkContext.h"

namespace atm::render::vk {

void Swapchain::destroyViews(VkContext &ctx) {
  for (VkImageView v : views) vkDestroyImageView(ctx.device, v, nullptr);
  views.clear();
  for (VkSemaphore s : renderFinished) vkDestroySemaphore(ctx.device, s, nullptr);
  renderFinished.clear();
  images.clear();
  layouts.clear();
}

bool Swapchain::create(VkContext &ctx, uint32_t width, uint32_t height, bool vsync) {
  if (width == 0 || height == 0) return false;

  vkb::SwapchainBuilder b(ctx.vkbDevice, ctx.surface);
  b.set_desired_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
      .add_fallback_format({VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
      .add_fallback_format({VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
      .set_desired_extent(width, height)
      .set_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
      .set_desired_min_image_count(3)
      .set_old_swapchain(swapchain_);
  if (vsync) {
    b.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR);
  } else {
    b.set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
        .add_fallback_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
        .add_fallback_present_mode(VK_PRESENT_MODE_FIFO_KHR);
  }
  auto ret = b.build();
  if (!ret) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "swapchain: %s", ret.error().message().c_str());
    return false;
  }

  // The old swapchain was retired by the builder; release it now (device idle).
  destroyViews(ctx);
  if (swapchain_.swapchain != VK_NULL_HANDLE) vkb::destroy_swapchain(swapchain_);
  swapchain_ = ret.value();

  auto imgs = swapchain_.get_images();
  auto vws = swapchain_.get_image_views();
  if (!imgs || !vws) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "swapchain images unavailable");
    return false;
  }
  images = imgs.value();
  views = vws.value();
  layouts.assign(images.size(), VK_IMAGE_LAYOUT_UNDEFINED);
  const VkFormat f = swapchain_.image_format;
  srgb_ = f == VK_FORMAT_B8G8R8A8_SRGB || f == VK_FORMAT_R8G8B8A8_SRGB;

  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  renderFinished.resize(images.size(), VK_NULL_HANDLE);
  for (VkSemaphore &s : renderFinished)
    if (!ATM_VK_OK(vkCreateSemaphore(ctx.device, &sci, nullptr, &s))) return false;
  return true;
}

void Swapchain::destroy(VkContext &ctx) {
  destroyViews(ctx);
  if (swapchain_.swapchain != VK_NULL_HANDLE) vkb::destroy_swapchain(swapchain_);
  swapchain_ = vkb::Swapchain{};
}

} // namespace atm::render::vk
