#pragma once

// Thin ownership helpers: VMA buffers/images, barriers, pipeline creation.

#include "VkCommon.h"

#include <cstdint>
#include <span>

namespace atm::render::vk {

enum class MemoryKind {
  DeviceLocal,   // GPU only
  Upload,        // host-visible, persistently mapped, sequential writes
  Readback,      // host-visible, persistently mapped, cached reads
};

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  void *mapped = nullptr;
  VkDeviceSize size = 0;
  bool coherent = true;
};

bool createBuffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
                  MemoryKind kind, Buffer &out, const char *debugName = nullptr);
void destroyBuffer(VmaAllocator allocator, Buffer &buffer);
// Flush host writes when the memory is not HOST_COHERENT.
void flushBuffer(VmaAllocator allocator, const Buffer &buffer, VkDeviceSize offset,
                 VkDeviceSize size);
void invalidateBuffer(VmaAllocator allocator, const Buffer &buffer);

struct Image {
  VkImage image = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;    // all mips
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkExtent2D extent{0, 0};
  uint32_t mipLevels = 1;
  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
};

bool createImage2D(VkDevice device, VmaAllocator allocator, VkFormat format, VkExtent2D extent,
                   uint32_t mipLevels, VkSampleCountFlagBits samples, VkImageUsageFlags usage,
                   VkImageAspectFlags aspect, Image &out);
void destroyImage(VkDevice device, VmaAllocator allocator, Image &image);
VkImageView createView(VkDevice device, VkImage image, VkFormat format,
                       VkImageAspectFlags aspect, uint32_t baseMip, uint32_t mipCount);

// synchronization2 image barrier on one command buffer.
void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                  VkImageLayout oldLayout, VkImageLayout newLayout,
                  VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                  VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                  uint32_t baseMip = 0, uint32_t mipCount = VK_REMAINING_MIP_LEVELS);
void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                   VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

// Graphics pipeline description for dynamic rendering (one colour target).
struct GraphicsPipelineDesc {
  VkShaderModule vert = VK_NULL_HANDLE, frag = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkFormat colorFormat = VK_FORMAT_UNDEFINED;
  VkFormat depthFormat = VK_FORMAT_UNDEFINED;   // UNDEFINED = no depth attachment
  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkCullModeFlags cull = VK_CULL_MODE_BACK_BIT;
  bool depthTest = true, depthWrite = true;
  VkCompareOp depthCompare = VK_COMPARE_OP_GREATER; // reverse-Z
  bool alphaBlend = false;
};
VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineCache cache,
                                  const GraphicsPipelineDesc &desc);
VkPipeline createComputePipeline(VkDevice device, VkPipelineCache cache, VkShaderModule module,
                                 VkPipelineLayout layout);
VkShaderModule createShaderModule(VkDevice device, const uint32_t *code, size_t sizeBytes);

} // namespace atm::render::vk
