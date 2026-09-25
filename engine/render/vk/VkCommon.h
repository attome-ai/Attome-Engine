#pragma once

// Shared Vulkan includes for engine/render. Vulkan prototypes come from the
// loader (vulkan-1 / libvulkan, linked via Vulkan::Vulkan).
//
// volk is deliberately NOT used: the prebuilt vcpkg imgui_impl_vulkan is
// compiled with prototypes and calls vkXxx directly; volk.c defines global
// function-pointer *variables* with the same C names, so linking both makes
// imgui's calls land on data (crash) or yields duplicate symbols. One
// consistent model (prototypes + loader) for every TU avoids that.

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <SDL3/SDL_log.h>

#include <cstdint>

namespace atm::render::vk {

const char *resultString(VkResult r);

} // namespace atm::render::vk

// Logs failures; evaluates to true on success.
#define ATM_VK_OK(expr)                                                        \
  ([&]() -> bool {                                                             \
    const VkResult atm_vk_r_ = (expr);                                         \
    if (atm_vk_r_ != VK_SUCCESS) {                                             \
      SDL_LogError(SDL_LOG_CATEGORY_RENDER, "%s failed: %s (%s:%d)", #expr,    \
                   ::atm::render::vk::resultString(atm_vk_r_), __FILE__,       \
                   __LINE__);                                                  \
      return false;                                                            \
    }                                                                          \
    return true;                                                               \
  }())
