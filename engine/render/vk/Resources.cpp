#include "Resources.h"

namespace atm::render::vk {

bool createBuffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
                  MemoryKind kind, Buffer &out, const char *debugName) {
  out = Buffer{};
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo aci{};
  switch (kind) {
  case MemoryKind::DeviceLocal:
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    break;
  case MemoryKind::Upload:
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    break;
  case MemoryKind::Readback:
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    break;
  }
  VmaAllocationInfo info{};
  if (!ATM_VK_OK(vmaCreateBuffer(allocator, &bci, &aci, &out.buffer, &out.allocation, &info)))
    return false;
  out.mapped = info.pMappedData;
  out.size = size;
  VkMemoryPropertyFlags flags = 0;
  vmaGetAllocationMemoryProperties(allocator, out.allocation, &flags);
  out.coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
  if (debugName) vmaSetAllocationName(allocator, out.allocation, debugName);
  return true;
}

void destroyBuffer(VmaAllocator allocator, Buffer &buffer) {
  if (buffer.buffer) vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
  buffer = Buffer{};
}

void flushBuffer(VmaAllocator allocator, const Buffer &buffer, VkDeviceSize offset,
                 VkDeviceSize size) {
  if (!buffer.coherent && size > 0) vmaFlushAllocation(allocator, buffer.allocation, offset, size);
}

void invalidateBuffer(VmaAllocator allocator, const Buffer &buffer) {
  if (!buffer.coherent) vmaInvalidateAllocation(allocator, buffer.allocation, 0, VK_WHOLE_SIZE);
}

VkImageView createView(VkDevice device, VkImage image, VkFormat format,
                       VkImageAspectFlags aspect, uint32_t baseMip, uint32_t mipCount) {
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = format;
  vci.subresourceRange.aspectMask = aspect;
  vci.subresourceRange.baseMipLevel = baseMip;
  vci.subresourceRange.levelCount = mipCount;
  vci.subresourceRange.baseArrayLayer = 0;
  vci.subresourceRange.layerCount = 1;
  VkImageView view = VK_NULL_HANDLE;
  if (!ATM_VK_OK(vkCreateImageView(device, &vci, nullptr, &view))) return VK_NULL_HANDLE;
  return view;
}

bool createImage2D(VkDevice device, VmaAllocator allocator, VkFormat format, VkExtent2D extent,
                   uint32_t mipLevels, VkSampleCountFlagBits samples, VkImageUsageFlags usage,
                   VkImageAspectFlags aspect, Image &out) {
  out = Image{};
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = format;
  ici.extent = {extent.width, extent.height, 1};
  ici.mipLevels = mipLevels;
  ici.arrayLayers = 1;
  ici.samples = samples;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = usage;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  aci.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT; // render targets
  if (!ATM_VK_OK(vmaCreateImage(allocator, &ici, &aci, &out.image, &out.allocation, nullptr)))
    return false;
  out.format = format;
  out.extent = extent;
  out.mipLevels = mipLevels;
  out.aspect = aspect;
  out.view = createView(device, out.image, format, aspect, 0, mipLevels);
  return out.view != VK_NULL_HANDLE;
}

void destroyImage(VkDevice device, VmaAllocator allocator, Image &image) {
  if (image.view) vkDestroyImageView(device, image.view, nullptr);
  if (image.image) vmaDestroyImage(allocator, image.image, image.allocation);
  image = Image{};
}

void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                  VkImageLayout oldLayout, VkImageLayout newLayout,
                  VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                  VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess, uint32_t baseMip,
                  uint32_t mipCount) {
  VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  b.srcStageMask = srcStage;
  b.srcAccessMask = srcAccess;
  b.dstStageMask = dstStage;
  b.dstAccessMask = dstAccess;
  b.oldLayout = oldLayout;
  b.newLayout = newLayout;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.subresourceRange.aspectMask = aspect;
  b.subresourceRange.baseMipLevel = baseMip;
  b.subresourceRange.levelCount = mipCount;
  b.subresourceRange.baseArrayLayer = 0;
  b.subresourceRange.layerCount = 1;
  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep.imageMemoryBarrierCount = 1;
  dep.pImageMemoryBarriers = &b;
  vkCmdPipelineBarrier2(cmd, &dep);
}

void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                   VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
  VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  b.srcStageMask = srcStage;
  b.srcAccessMask = srcAccess;
  b.dstStageMask = dstStage;
  b.dstAccessMask = dstAccess;
  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep.memoryBarrierCount = 1;
  dep.pMemoryBarriers = &b;
  vkCmdPipelineBarrier2(cmd, &dep);
}

VkShaderModule createShaderModule(VkDevice device, const uint32_t *code, size_t sizeBytes) {
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = sizeBytes;
  ci.pCode = code;
  VkShaderModule m = VK_NULL_HANDLE;
  if (!ATM_VK_OK(vkCreateShaderModule(device, &ci, nullptr, &m))) return VK_NULL_HANDLE;
  return m;
}

VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineCache cache,
                                  const GraphicsPipelineDesc &d) {
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = d.vert;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = d.frag;
  stages[1].pName = "main";

  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = d.topology;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1;
  vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = d.cull;
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = 1.0f;
  if (d.depthBias) {
    rs.depthBiasEnable = VK_TRUE;
    rs.depthBiasConstantFactor = d.depthBiasConstant;
    rs.depthBiasSlopeFactor = d.depthBiasSlope;
  }
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = d.samples;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  ds.depthTestEnable = d.depthTest ? VK_TRUE : VK_FALSE;
  ds.depthWriteEnable = d.depthWrite ? VK_TRUE : VK_FALSE;
  ds.depthCompareOp = d.depthCompare;
  VkPipelineColorBlendAttachmentState att{};
  att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                       VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  if (d.alphaBlend) {
    att.blendEnable = VK_TRUE;
    att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    att.colorBlendOp = VK_BLEND_OP_ADD;
    att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    att.alphaBlendOp = VK_BLEND_OP_ADD;
  }
  VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = d.colorFormat != VK_FORMAT_UNDEFINED ? 1u : 0u;
  cb.pAttachments = &att;
  const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2;
  dy.pDynamicStates = dyn;

  VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  ri.colorAttachmentCount = d.colorFormat != VK_FORMAT_UNDEFINED ? 1u : 0u;
  ri.pColorAttachmentFormats = &d.colorFormat;
  ri.depthAttachmentFormat = d.depthFormat;

  VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pci.pNext = &ri;
  pci.stageCount = d.frag ? 2u : 1u; // no fragment shader: depth-only pass
  pci.pStages = stages;
  pci.pVertexInputState = &vi;
  pci.pInputAssemblyState = &ia;
  pci.pViewportState = &vp;
  pci.pRasterizationState = &rs;
  pci.pMultisampleState = &ms;
  pci.pDepthStencilState = &ds;
  pci.pColorBlendState = &cb;
  pci.pDynamicState = &dy;
  pci.layout = d.layout;
  VkPipeline p = VK_NULL_HANDLE;
  if (!ATM_VK_OK(vkCreateGraphicsPipelines(device, cache, 1, &pci, nullptr, &p)))
    return VK_NULL_HANDLE;
  return p;
}

VkPipeline createComputePipeline(VkDevice device, VkPipelineCache cache, VkShaderModule module,
                                 VkPipelineLayout layout) {
  VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  ci.stage.module = module;
  ci.stage.pName = "main";
  ci.layout = layout;
  VkPipeline p = VK_NULL_HANDLE;
  if (!ATM_VK_OK(vkCreateComputePipelines(device, cache, 1, &ci, nullptr, &p)))
    return VK_NULL_HANDLE;
  return p;
}

} // namespace atm::render::vk
