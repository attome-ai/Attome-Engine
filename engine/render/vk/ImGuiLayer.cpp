#include "ImGuiLayer.h"

#include "VkContext.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_vulkan.h>

namespace atm::render::vk {

namespace {

VkFormat g_colorFormat = VK_FORMAT_UNDEFINED; // must outlive Init (backend keeps the pointer)

void checkVk(VkResult r) {
  if (r != VK_SUCCESS)
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "imgui vulkan: %s", resultString(r));
}

// ImGui_ImplVulkan_InitInfo changed between releases; fill whichever fields
// this version has (the template makes the unused branches disappear).
template <class Info>
void fillPipelineInfo(Info &info, const VkPipelineRenderingCreateInfo &rendering) {
  if constexpr (requires { info.PipelineInfoMain; }) {          // 1.92.x
    info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.PipelineInfoMain.PipelineRenderingCreateInfo = rendering;
  } else if constexpr (requires { info.PipelineRenderingCreateInfo; }) { // 1.90.x-1.91.x
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.PipelineRenderingCreateInfo = rendering;
  } else {                                                       // older 1.90
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.ColorAttachmentFormat = rendering.pColorAttachmentFormats[0];
  }
  if constexpr (requires { info.ApiVersion; }) info.ApiVersion = VK_API_VERSION_1_3;
}

} // namespace

bool ImGuiLayer::init(VkContext &ctx, VkFormat colorFormat, uint32_t imageCount) {
  if (!ImGui::GetCurrentContext()) {
    // The client normally creates the context (before Renderer::init).
    ImGui::CreateContext();
    ownsContext_ = true;
  }

  const VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64}};
  VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pci.maxSets = 64;
  pci.poolSizeCount = 1;
  pci.pPoolSizes = sizes;
  if (!ATM_VK_OK(vkCreateDescriptorPool(ctx.device, &pci, nullptr, &pool_))) return false;

  g_colorFormat = colorFormat;
  VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering.colorAttachmentCount = 1;
  rendering.pColorAttachmentFormats = &g_colorFormat;

  ImGui_ImplVulkan_InitInfo info{};
  info.Instance = ctx.instance;
  info.PhysicalDevice = ctx.physicalDevice;
  info.Device = ctx.device;
  info.QueueFamily = ctx.graphicsFamily;
  info.Queue = ctx.graphicsQueue;
  info.DescriptorPool = pool_;
  info.MinImageCount = imageCount < 2 ? 2 : imageCount;
  info.ImageCount = imageCount < 2 ? 2 : imageCount;
  info.UseDynamicRendering = true;
  info.CheckVkResultFn = checkVk;
  fillPipelineInfo(info, rendering);

  if (!ImGui_ImplVulkan_Init(&info)) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "ImGui_ImplVulkan_Init failed");
    return false;
  }
  initialised_ = true;
  return true;
}

void ImGuiLayer::setImageCount(uint32_t imageCount) {
  if (initialised_) ImGui_ImplVulkan_SetMinImageCount(imageCount < 2 ? 2 : imageCount);
}

void ImGuiLayer::newFrame() {
  if (initialised_) ImGui_ImplVulkan_NewFrame();
}

void ImGuiLayer::render(VkCommandBuffer cmd) {
  if (!initialised_) return;
  ImGuiContext *g = ImGui::GetCurrentContext();
  if (!g || !g->WithinFrameScope) return; // app skipped ImGui::NewFrame()
  ImGui::Render();
  ImDrawData *dd = ImGui::GetDrawData();
  if (dd) ImGui_ImplVulkan_RenderDrawData(dd, cmd); // also processes texture updates (1.92+)
}

void ImGuiLayer::shutdown(VkDevice device) {
  if (initialised_) {
    // The backend lives in the ImGui context: if the app already destroyed
    // it (it must call Renderer::shutdown() first), calling Shutdown would
    // dereference a null context. Its Vulkan objects then leak until the
    // device is destroyed.
    if (ImGui::GetCurrentContext()) ImGui_ImplVulkan_Shutdown();
    else
      SDL_LogWarn(SDL_LOG_CATEGORY_RENDER,
                  "ImGui context destroyed before Renderer::shutdown(); backend leaked");
    initialised_ = false;
  }
  if (pool_) {
    vkDestroyDescriptorPool(device, pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
  }
  if (ownsContext_ && ImGui::GetCurrentContext()) {
    ImGui::DestroyContext();
    ownsContext_ = false;
  }
}

} // namespace atm::render::vk
