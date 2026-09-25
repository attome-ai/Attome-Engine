#pragma once

// Dear ImGui Vulkan backend (dynamic rendering). The application owns the
// ImGui context and the SDL3 platform backend; the renderer owns
// ImGui_ImplVulkan and draws ImGui::GetDrawData() at the end of the frame.

#include "VkCommon.h"

namespace atm::render::vk {

class VkContext;

class ImGuiLayer {
public:
  bool init(VkContext &ctx, VkFormat colorFormat, uint32_t imageCount);
  void shutdown(VkDevice device);
  void setImageCount(uint32_t imageCount);
  void newFrame();                       // before the app calls ImGui::NewFrame()
  // Calls ImGui::Render() and records the draw data into `cmd` (inside an
  // active dynamic rendering scope). No-op if the app did not start a frame.
  void render(VkCommandBuffer cmd);
  bool active() const { return initialised_; }

private:
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  bool initialised_ = false;
  bool ownsContext_ = false;
};

} // namespace atm::render::vk
