// Renderer: public API forwarding, init/shutdown, GPU resource creation.
// Geometry management: RendererGeometry.cpp. Per-frame work: RendererFrame.cpp.

#include "RendererImpl.h"

#include "vk/ShaderLibrary.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <iterator>

namespace atm::render {

// =============================================================================
// Public API
// =============================================================================

Renderer::Renderer() : impl_(std::make_unique<Impl>()) {}
Renderer::~Renderer() { shutdown(); }

bool Renderer::init(SDL_Window *window, const RendererConfig &config, std::string *error) {
  if (impl_->initialised) return true;
  if (!impl_->init(window, config, error)) {
    impl_->shutdown();
    return false;
  }
  return true;
}

void Renderer::shutdown() {
  if (impl_) impl_->shutdown();
}

void Renderer::onWindowResized() { impl_->swapchainDirty = true; }

void Renderer::setMaterials(std::span<const Material> materials) {
  impl_->setMaterials(materials.first(std::min<size_t>(materials.size(), kModelMaterialBase)), 0);
}

uint16_t Renderer::setModelMaterials(std::span<const Material> materials) {
  Impl &I = *impl_;
  const uint32_t base = I.modelMaterialCursor;
  if (materials.empty()) return uint16_t(base < kMaxMaterials ? base : 0xFFFFu);
  if (base + materials.size() > kMaxMaterials) {
    SDL_LogWarn(SDL_LOG_CATEGORY_RENDER, "Renderer: material table full");
    return 0xFFFF;
  }
  I.setMaterials(materials, base);
  I.modelMaterialCursor = base + uint32_t(materials.size());
  return uint16_t(base);
}

void Renderer::setChunkMesh(voxel::ChunkCoord coord, const voxel::ChunkMeshData &mesh) {
  impl_->setChunkMesh(coord, mesh);
}
void Renderer::removeChunk(voxel::ChunkCoord coord) { impl_->removeChunk(coord); }
ModelMeshId Renderer::createModelMesh(const voxel::ChunkMeshData &mesh) {
  return impl_->createModelMesh(mesh);
}
void Renderer::destroyModelMesh(ModelMeshId id) { impl_->destroyModelMesh(id); }

bool Renderer::beginFrame(const Camera &camera, const Environment &env) {
  return impl_->beginFrame(camera, env);
}

void Renderer::drawModel(const ModelInstance &instance) {
  Impl &I = *impl_;
  if (!I.frameActive || instance.mesh >= I.models.size()) return;
  if (!I.models[instance.mesh].ready) return;
  if (I.instances.size() >= kMaxModelInstances) return;
  I.instances.push_back({instance.mesh, uint32_t(I.instances.size()), instance});
}

void Renderer::drawBlockHighlight(voxel::BlockPos block) {
  impl_->highlightValid = true;
  impl_->highlightBlock = block;
}

void Renderer::endFrame() { impl_->endFrame(); }

const FrameStats &Renderer::stats() const { return impl_->stats; }

void Renderer::requestScreenshot(const std::string &path) { impl_->screenshotRequest = path; }

// =============================================================================
// Init / shutdown
// =============================================================================

bool Renderer::Impl::init(SDL_Window *win, const RendererConfig &cfg, std::string *error) {
  window = win;
  config = cfg;
  framesInFlight = uint32_t(std::clamp(cfg.framesInFlight, 2, int(kMaxFramesInFlight)));
  config.uploadBudgetBytes = std::clamp(cfg.uploadBudgetBytes, 64u << 10, 256u << 20);

  if (!ctx.init(window, cfg.validation, cfg.msaaSamples, error)) return false;
  depthAspect = ctx.caps.depthFormat == VK_FORMAT_D24_UNORM_S8_UINT
                    ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
                    : VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT);

  VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  if (!ATM_VK_OK(vkCreatePipelineCache(ctx.device, &pcci, nullptr, &pipelineCache))) return false;

  int w = 0, h = 0;
  SDL_GetWindowSizeInPixels(window, &w, &h);
  if (!swapchain.create(ctx, uint32_t(std::max(w, 1)), uint32_t(std::max(h, 1)), cfg.vsync)) {
    if (error) *error = "swapchain creation failed";
    return false;
  }
  swapchainFormat = swapchain.format();

  auto failWith = [&](const char *what) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Renderer: %s failed", what);
    if (error) *error = std::string(what) + " failed";
    return false;
  };
  if (!createFrames()) return failWith("frame resources");
  if (!createStaticBuffers()) return failWith("GPU buffers");
  if (!createDescriptors()) return failWith("descriptors");
  if (!createPipelines()) return failWith("pipelines");
  const VkExtent2D ext = swapchain.extent();
  if (!createTargets(ext.width, ext.height)) return failWith("render targets");
  if (!imgui.init(ctx, swapchainFormat, swapchain.imageCount())) return failWith("ImGui Vulkan backend");

  // CPU-side tables.
  chunks.resize(kMaxChunkSlots);
  freeSlots.reserve(kMaxChunkSlots);
  for (uint32_t i = kMaxChunkSlots; i-- > 0;) freeSlots.push_back(i);
  slotOf.reserve(kMaxChunkSlots);
  cullChunks.reserve(kMaxChunkSlots);
  uploadQueue.reserve(4096);
  dirtySlots.reserve(4096);
  visibleSlots.reserve(kMaxChunkSlots);
  translucentOrder.reserve(kMaxChunkSlots);
  instances.reserve(kMaxModelInstances);
  modelDraws.reserve(1024);
  arenaCopies.reserve(1024);
  modelCopies.reserve(256);
  metaCopies.reserve(kMetaStagingBytes / sizeof(gpu::ChunkGpu));
  materialCopies.reserve(4);

  stats.chunkArenaCapacity = uint64_t(arenaAlloc.capacity()) * 8u;
  initialised = true;
  return true;
}

void Renderer::Impl::shutdown() {
  if (!ctx.device) {
    ctx.shutdown();
    initialised = false;
    return;
  }
  vkDeviceWaitIdle(ctx.device);
  saveScreenshotIfReady();

  imgui.shutdown(ctx.device);
  destroyTargets();

  VkDevice d = ctx.device;
  VmaAllocator a = ctx.allocator;
  for (FrameData &f : frames) {
    if (f.pool) vkDestroyCommandPool(d, f.pool, nullptr);
    if (f.fence) vkDestroyFence(d, f.fence, nullptr);
    if (f.imageAvailable) vkDestroySemaphore(d, f.imageAvailable, nullptr);
    if (f.queries) vkDestroyQueryPool(d, f.queries, nullptr);
    for (vk::Buffer *b : {&f.ubo, &f.visible, &f.draws, &f.drawCount, &f.translucentDraws,
                          &f.shadowDraws, &f.instances, &f.staging})
      vk::destroyBuffer(a, *b);
    f = FrameData{};
  }
  for (vk::Buffer *b : {&indexBuffer, &faceArena, &chunkMeta, &materials, &modelFaces,
                        &screenshotBuffer})
    vk::destroyBuffer(a, *b);

  for (VkPipeline *p : {&opaquePipeline, &translucentPipeline, &modelPipeline, &skyPipeline,
                        &highlightPipeline, &cullPipeline, &bloomPipeline, &tonemapPipeline,
                        &shadowPipeline, &shadowModelPipeline, &ssaoPipeline}) {
    if (*p) vkDestroyPipeline(d, *p, nullptr);
    *p = VK_NULL_HANDLE;
  }
  for (VkPipelineLayout *l : {&scenePipelineLayout, &bloomPipelineLayout, &tonemapPipelineLayout,
                              &ssaoPipelineLayout}) {
    if (*l) vkDestroyPipelineLayout(d, *l, nullptr);
    *l = VK_NULL_HANDLE;
  }
  for (VkDescriptorSetLayout *l : {&sceneSetLayout, &bloomSetLayout, &tonemapSetLayout, &ssaoSetLayout}) {
    if (*l) vkDestroyDescriptorSetLayout(d, *l, nullptr);
    *l = VK_NULL_HANDLE;
  }
  if (descriptorPool) vkDestroyDescriptorPool(d, descriptorPool, nullptr);
  descriptorPool = VK_NULL_HANDLE;
  if (linearSampler) vkDestroySampler(d, linearSampler, nullptr);
  linearSampler = VK_NULL_HANDLE;
  if (shadowSampler) vkDestroySampler(d, shadowSampler, nullptr);
  shadowSampler = VK_NULL_HANDLE;
  if (nearestSampler) vkDestroySampler(d, nearestSampler, nullptr);
  nearestSampler = VK_NULL_HANDLE;
  vk::destroyImage(d, a, shadowMap);
  if (pipelineCache) vkDestroyPipelineCache(d, pipelineCache, nullptr);
  pipelineCache = VK_NULL_HANDLE;

  swapchain.destroy(ctx);
  ctx.shutdown();

  chunks.clear();
  freeSlots.clear();
  slotOf.clear();
  cullChunks.clear();
  uploadQueue.clear();
  uploadHead = 0;
  models.clear();
  freeModelIds.clear();
  modelUploadQueue.clear();
  initialised = false;
  frameActive = false;
}

// =============================================================================
// Resource creation
// =============================================================================

bool Renderer::Impl::createFrames() {
  VkDevice d = ctx.device;
  VmaAllocator a = ctx.allocator;
  const VkDeviceSize stagingBytes = VkDeviceSize(config.uploadBudgetBytes) + kMetaStagingBytes;
  for (uint32_t i = 0; i < framesInFlight; ++i) {
    FrameData &f = frames[i];
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = ctx.graphicsFamily;
    if (!ATM_VK_OK(vkCreateCommandPool(d, &cpci, nullptr, &f.pool))) return false;
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = f.pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (!ATM_VK_OK(vkAllocateCommandBuffers(d, &cbai, &f.cmd))) return false;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (!ATM_VK_OK(vkCreateFence(d, &fci, nullptr, &f.fence))) return false;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    if (!ATM_VK_OK(vkCreateSemaphore(d, &sci, nullptr, &f.imageAvailable))) return false;
    if (ctx.caps.timestampPeriod > 0.0f) {
      VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
      qci.queryCount = 2;
      if (!ATM_VK_OK(vkCreateQueryPool(d, &qci, nullptr, &f.queries))) return false;
    }
    using vk::MemoryKind;
    bool ok = true;
    ok = ok && vk::createBuffer(a, sizeof(gpu::FrameUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                MemoryKind::Upload, f.ubo, "frame ubo");
    ok = ok && vk::createBuffer(a, sizeof(uint32_t) * kMaxChunkSlots,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryKind::Upload, f.visible,
                                "visible chunks");
    ok = ok && vk::createBuffer(a, sizeof(gpu::DrawIndexedIndirect) * kMaxDraws,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                MemoryKind::DeviceLocal, f.draws, "opaque draws");
    ok = ok && vk::createBuffer(a, 16,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                MemoryKind::DeviceLocal, f.drawCount, "draw count");
    ok = ok && vk::createBuffer(a, sizeof(gpu::DrawIndexedIndirect) * kMaxChunkSlots,
                                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, MemoryKind::Upload,
                                f.translucentDraws, "translucent draws");
    ok = ok && vk::createBuffer(a, sizeof(gpu::DrawIndexedIndirect) * kMaxChunkSlots,
                                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, MemoryKind::Upload,
                                f.shadowDraws, "shadow draws");
    ok = ok && vk::createBuffer(a, sizeof(gpu::ModelInstanceGpu) * kMaxModelInstances,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryKind::Upload, f.instances,
                                "model instances");
    ok = ok && vk::createBuffer(a, stagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                MemoryKind::Upload, f.staging, "upload ring slot");
    if (!ok) return false;
    if (!f.ubo.mapped || !f.visible.mapped || !f.translucentDraws.mapped || !f.shadowDraws.mapped ||
        !f.instances.mapped || !f.staging.mapped)
      return false;
    f.deferred.reserve(1024);
  }
  return true;
}

bool Renderer::Impl::createStaticBuffers() {
  VmaAllocator a = ctx.allocator;
  using vk::MemoryKind;

  // Face arena: capped so vertexOffset = face * 4 fits in int32 and the whole
  // buffer is addressable as one storage buffer.
  uint64_t arenaBytes = config.chunkArenaBytes;
  arenaBytes = std::min<uint64_t>(arenaBytes, uint64_t(1) << 31);
  arenaBytes = std::min<uint64_t>(arenaBytes, uint64_t(ctx.caps.maxStorageBufferRange));
  arenaBytes &= ~uint64_t(7);
  if (arenaBytes < (1u << 20)) arenaBytes = 1u << 20;
  const VkBufferUsageFlags storageDst =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (!vk::createBuffer(a, arenaBytes, storageDst, MemoryKind::DeviceLocal, faceArena, "face arena"))
    return false;
  arenaAlloc.reset(uint32_t(arenaBytes / 8));
  if (!vk::createBuffer(a, sizeof(gpu::ChunkGpu) * kMaxChunkSlots, storageDst,
                        MemoryKind::DeviceLocal, chunkMeta, "chunk metadata"))
    return false;
  if (!vk::createBuffer(a, sizeof(gpu::MaterialGpu) * kMaxMaterials, storageDst,
                        MemoryKind::DeviceLocal, materials, "materials"))
    return false;
  if (!vk::createBuffer(a, kModelFaceBytes, storageDst, MemoryKind::DeviceLocal, modelFaces,
                        "model faces"))
    return false;
  modelAlloc.reset(kModelFaceBytes / 8);

  // Sun shadow map (size independent of the window).
  if (!vk::createImage2D(ctx.device, a, VK_FORMAT_D32_SFLOAT, {kShadowMapSize, kShadowMapSize}, 1,
                         VK_SAMPLE_COUNT_1_BIT,
                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         VK_IMAGE_ASPECT_DEPTH_BIT, shadowMap))
    return false;

  const VkDeviceSize indexBytes = VkDeviceSize(kMaxFacesPerDraw) * 6 * sizeof(uint32_t);
  if (!vk::createBuffer(a, indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        MemoryKind::DeviceLocal, indexBuffer, "quad indices"))
    return false;

  // Default materials: white, opaque.
  materialsCpu.assign(kMaxMaterials, gpu::MaterialGpu{0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                                                      0.0f, 1.0f, 0u, 0u, 0u});
  const VkDeviceSize materialBytes = sizeof(gpu::MaterialGpu) * kMaxMaterials;

  vk::Buffer staging;
  if (!vk::createBuffer(a, indexBytes + materialBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        MemoryKind::Upload, staging, "init staging") ||
      !staging.mapped) {
    vk::destroyBuffer(a, staging);
    return false;
  }
  auto *idx = static_cast<uint32_t *>(staging.mapped);
  for (uint32_t q = 0; q < kMaxFacesPerDraw; ++q) {
    const uint32_t v = q * 4u;
    idx[q * 6 + 0] = v + 0;
    idx[q * 6 + 1] = v + 1;
    idx[q * 6 + 2] = v + 2;
    idx[q * 6 + 3] = v + 0;
    idx[q * 6 + 4] = v + 2;
    idx[q * 6 + 5] = v + 3;
  }
  std::memcpy(static_cast<uint8_t *>(staging.mapped) + indexBytes, materialsCpu.data(),
              size_t(materialBytes));
  vk::flushBuffer(a, staging, 0, VK_WHOLE_SIZE);

  struct StaticInit {
    vk::Buffer *staging;
    VkDeviceSize indexBytes, materialBytes;
  };
  StaticInit si{&staging, indexBytes, materialBytes};
  const bool ok = immediateSubmit(
      [](Impl &I, VkCommandBuffer cmd, void *user) {
        const StaticInit &st = *static_cast<const StaticInit *>(user);
        VkBufferCopy c{0, 0, st.indexBytes};
        vkCmdCopyBuffer(cmd, st.staging->buffer, I.indexBuffer.buffer, 1, &c);
        VkBufferCopy m{st.indexBytes, 0, st.materialBytes};
        vkCmdCopyBuffer(cmd, st.staging->buffer, I.materials.buffer, 1, &m);
        vkCmdFillBuffer(cmd, I.chunkMeta.buffer, 0, VK_WHOLE_SIZE, 0u);
        // Every frame renders the shadow map before sampling it; start in the
        // layout the scene descriptor expects.
        vk::imageBarrier(cmd, I.shadowMap.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                         VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        vk::memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT);
      },
      &si);
  vk::destroyBuffer(a, staging);
  return ok;
}

bool Renderer::Impl::immediateSubmit(void (*record)(Impl &, VkCommandBuffer, void *), void *user) {
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = frames[0].pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (!ATM_VK_OK(vkAllocateCommandBuffers(ctx.device, &cbai, &cmd))) return false;
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);
  record(*this, cmd, user);
  vkEndCommandBuffer(cmd);
  VkCommandBufferSubmitInfo cbsi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cbsi.commandBuffer = cmd;
  VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  si.commandBufferInfoCount = 1;
  si.pCommandBufferInfos = &cbsi;
  const bool ok = ATM_VK_OK(vkQueueSubmit2(ctx.graphicsQueue, 1, &si, VK_NULL_HANDLE)) &&
                  ATM_VK_OK(vkQueueWaitIdle(ctx.graphicsQueue));
  vkFreeCommandBuffers(ctx.device, frames[0].pool, 1, &cmd);
  return ok;
}

bool Renderer::Impl::createDescriptors() {
  VkDevice d = ctx.device;

  // Scene set: every binding visible to every stage (compute + graphics).
  VkDescriptorSetLayoutBinding sb[gpu::kSceneBindingCount]{};
  for (uint32_t i = 0; i < gpu::kSceneBindingCount; ++i) {
    sb[i].binding = i;
    sb[i].descriptorType = i == gpu::kBindFrame       ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                           : i == gpu::kBindShadowMap ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                      : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sb[i].descriptorCount = 1;
    sb[i].stageFlags = VK_SHADER_STAGE_ALL;
  }
  VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  lci.bindingCount = gpu::kSceneBindingCount;
  lci.pBindings = sb;
  if (!ATM_VK_OK(vkCreateDescriptorSetLayout(d, &lci, nullptr, &sceneSetLayout))) return false;

  VkDescriptorSetLayoutBinding bb[2]{};
  bb[0].binding = 0;
  bb[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bb[0].descriptorCount = 1;
  bb[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bb[1].binding = 1;
  bb[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  bb[1].descriptorCount = 1;
  bb[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  lci.bindingCount = 2;
  lci.pBindings = bb;
  if (!ATM_VK_OK(vkCreateDescriptorSetLayout(d, &lci, nullptr, &bloomSetLayout))) return false;

  // Tonemap: hdr, bloom, scene depth (sun shafts), SSAO.
  VkDescriptorSetLayoutBinding tb[4]{};
  for (uint32_t i = 0; i < 4; ++i) {
    tb[i].binding = i;
    tb[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    tb[i].descriptorCount = 1;
    tb[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  }
  lci.bindingCount = 4;
  lci.pBindings = tb;
  if (!ATM_VK_OK(vkCreateDescriptorSetLayout(d, &lci, nullptr, &tonemapSetLayout))) return false;

  // SSAO: scene depth in, AO storage image out (same shape as bloom's set).
  lci.bindingCount = 2;
  lci.pBindings = bb;
  if (!ATM_VK_OK(vkCreateDescriptorSetLayout(d, &lci, nullptr, &ssaoSetLayout))) return false;

  const VkDescriptorPoolSize sizes[] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kMaxFramesInFlight},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxFramesInFlight * (gpu::kSceneBindingCount - 2)},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * kMaxBloomMips + 4 + 1 + kMaxFramesInFlight},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kMaxBloomMips + 1},
  };
  VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pci.maxSets = kMaxFramesInFlight + 2 * kMaxBloomMips + 2;
  pci.poolSizeCount = uint32_t(std::size(sizes));
  pci.pPoolSizes = sizes;
  if (!ATM_VK_OK(vkCreateDescriptorPool(d, &pci, nullptr, &descriptorPool))) return false;

  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = descriptorPool;
  ai.descriptorSetCount = 1;
  for (uint32_t i = 0; i < framesInFlight; ++i) {
    ai.pSetLayouts = &sceneSetLayout;
    if (!ATM_VK_OK(vkAllocateDescriptorSets(d, &ai, &frames[i].sceneSet))) return false;
  }
  for (VkDescriptorSet &s : bloomSets) {
    ai.pSetLayouts = &bloomSetLayout;
    if (!ATM_VK_OK(vkAllocateDescriptorSets(d, &ai, &s))) return false;
  }
  ai.pSetLayouts = &tonemapSetLayout;
  if (!ATM_VK_OK(vkAllocateDescriptorSets(d, &ai, &tonemapSet))) return false;
  ai.pSetLayouts = &ssaoSetLayout;
  if (!ATM_VK_OK(vkAllocateDescriptorSets(d, &ai, &ssaoSet))) return false;

  VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sci.magFilter = VK_FILTER_LINEAR;
  sci.minFilter = VK_FILTER_LINEAR;
  sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.maxLod = 0.0f;
  if (!ATM_VK_OK(vkCreateSampler(d, &sci, nullptr, &linearSampler))) return false;
  VkSamplerCreateInfo nsci = sci; // depth reads: no filtering across edges
  nsci.magFilter = VK_FILTER_NEAREST;
  nsci.minFilter = VK_FILTER_NEAREST;
  if (!ATM_VK_OK(vkCreateSampler(d, &nsci, nullptr, &nearestSampler))) return false;

  // Shadow compare sampler: bilinear PCF in hardware; outside the map = lit.
  VkSamplerCreateInfo ssci = sci;
  ssci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  ssci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  ssci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  ssci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
  ssci.compareEnable = VK_TRUE;
  ssci.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  if (!ATM_VK_OK(vkCreateSampler(d, &ssci, nullptr, &shadowSampler))) return false;

  // Scene sets never change: all buffers are fixed-size, the shadow map too.
  for (uint32_t i = 0; i < framesInFlight; ++i) {
    FrameData &f = frames[i];
    const VkBuffer bufs[gpu::kSceneBindingCount] = {
        f.ubo.buffer,     faceArena.buffer,  chunkMeta.buffer,
        materials.buffer, f.visible.buffer,  f.draws.buffer,
        f.drawCount.buffer, modelFaces.buffer, f.instances.buffer, VK_NULL_HANDLE};
    VkDescriptorBufferInfo infos[gpu::kSceneBindingCount]{};
    VkWriteDescriptorSet writes[gpu::kSceneBindingCount]{};
    const VkDescriptorImageInfo shadowInfo{shadowSampler, shadowMap.view,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    for (uint32_t b = 0; b < gpu::kSceneBindingCount; ++b) {
      writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[b].dstSet = f.sceneSet;
      writes[b].dstBinding = b;
      writes[b].descriptorCount = 1;
      writes[b].descriptorType = sb[b].descriptorType;
      if (b == gpu::kBindShadowMap) {
        writes[b].pImageInfo = &shadowInfo;
      } else {
        infos[b] = {bufs[b], 0, VK_WHOLE_SIZE};
        writes[b].pBufferInfo = &infos[b];
      }
    }
    vkUpdateDescriptorSets(d, gpu::kSceneBindingCount, writes, 0, nullptr);
  }
  return true;
}

bool Renderer::Impl::createPipelines() {
  VkDevice d = ctx.device;
  VkPushConstantRange scenePush{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                sizeof(gpu::HighlightPush)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &sceneSetLayout;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &scenePush;
  if (!ATM_VK_OK(vkCreatePipelineLayout(d, &plci, nullptr, &scenePipelineLayout))) return false;

  VkPushConstantRange bloomPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gpu::BloomPush)};
  plci.pSetLayouts = &bloomSetLayout;
  plci.pPushConstantRanges = &bloomPush;
  if (!ATM_VK_OK(vkCreatePipelineLayout(d, &plci, nullptr, &bloomPipelineLayout))) return false;

  VkPushConstantRange ssaoPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gpu::SsaoPush)};
  plci.pSetLayouts = &ssaoSetLayout;
  plci.pPushConstantRanges = &ssaoPush;
  if (!ATM_VK_OK(vkCreatePipelineLayout(d, &plci, nullptr, &ssaoPipelineLayout))) return false;

  VkPushConstantRange tonePush{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(gpu::TonemapPush)};
  plci.pSetLayouts = &tonemapSetLayout;
  plci.pPushConstantRanges = &tonePush;
  if (!ATM_VK_OK(vkCreatePipelineLayout(d, &plci, nullptr, &tonemapPipelineLayout))) return false;

  VkShaderModule mods[size_t(vk::ShaderId::Count)]{};
  bool ok = true;
  for (size_t i = 0; i < size_t(vk::ShaderId::Count); ++i) {
    const vk::SpirvBlob blob = vk::shaderSpirv(vk::ShaderId(i));
    mods[i] = blob.code ? vk::createShaderModule(d, blob.code, blob.sizeBytes) : VK_NULL_HANDLE;
    ok = ok && mods[i] != VK_NULL_HANDLE;
  }
  auto mod = [&](vk::ShaderId id) { return mods[size_t(id)]; };

  if (ok) {
    vk::GraphicsPipelineDesc g{};
    g.layout = scenePipelineLayout;
    g.colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    g.depthFormat = ctx.caps.depthFormat;
    g.samples = ctx.caps.msaa;

    g.vert = mod(vk::ShaderId::VoxelVert);
    g.frag = mod(vk::ShaderId::VoxelFrag);
    opaquePipeline = vk::createGraphicsPipeline(d, pipelineCache, g);

    vk::GraphicsPipelineDesc t = g;
    t.alphaBlend = true;
    t.depthWrite = false;
    t.cull = VK_CULL_MODE_NONE; // water surface seen from below
    translucentPipeline = vk::createGraphicsPipeline(d, pipelineCache, t);

    vk::GraphicsPipelineDesc m = g;
    m.vert = mod(vk::ShaderId::ModelVert);
    modelPipeline = vk::createGraphicsPipeline(d, pipelineCache, m);

    vk::GraphicsPipelineDesc s = g;
    s.vert = mod(vk::ShaderId::FullscreenVert);
    s.frag = mod(vk::ShaderId::SkyFrag);
    s.cull = VK_CULL_MODE_NONE;
    s.depthWrite = false;
    s.depthCompare = VK_COMPARE_OP_EQUAL; // only where depth is still the cleared 0
    skyPipeline = vk::createGraphicsPipeline(d, pipelineCache, s);

    vk::GraphicsPipelineDesc h = g;
    h.vert = mod(vk::ShaderId::HighlightVert);
    h.frag = mod(vk::ShaderId::HighlightFrag);
    h.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    h.cull = VK_CULL_MODE_NONE;
    h.depthWrite = false;
    h.depthCompare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    highlightPipeline = vk::createGraphicsPipeline(d, pipelineCache, h);

    // Shadow map: depth only, standard depth (0 = nearest the sun), both
    // face sides (voxel meshes are closed), slope-scaled bias against acne.
    vk::GraphicsPipelineDesc sh{};
    sh.layout = scenePipelineLayout;
    sh.vert = mod(vk::ShaderId::ShadowVert);
    sh.colorFormat = VK_FORMAT_UNDEFINED;
    sh.depthFormat = VK_FORMAT_D32_SFLOAT;
    sh.cull = VK_CULL_MODE_NONE;
    sh.depthCompare = VK_COMPARE_OP_LESS_OR_EQUAL;
    sh.depthBias = true;
    sh.depthBiasConstant = 1.25f;
    sh.depthBiasSlope = 1.75f;
    shadowPipeline = vk::createGraphicsPipeline(d, pipelineCache, sh);
    sh.vert = mod(vk::ShaderId::ShadowModelVert);
    shadowModelPipeline = vk::createGraphicsPipeline(d, pipelineCache, sh);

    cullPipeline = vk::createComputePipeline(d, pipelineCache, mod(vk::ShaderId::CullComp),
                                             scenePipelineLayout);
    bloomPipeline = vk::createComputePipeline(d, pipelineCache, mod(vk::ShaderId::BloomComp),
                                              bloomPipelineLayout);
    ssaoPipeline = vk::createComputePipeline(d, pipelineCache, mod(vk::ShaderId::SsaoComp),
                                             ssaoPipelineLayout);
  }
  for (VkShaderModule m : mods)
    if (m) vkDestroyShaderModule(d, m, nullptr);
  if (!ok || !opaquePipeline || !translucentPipeline || !modelPipeline || !skyPipeline ||
      !highlightPipeline || !cullPipeline || !bloomPipeline || !shadowPipeline ||
      !shadowModelPipeline || !ssaoPipeline)
    return false;
  return createTonemapPipeline();
}

bool Renderer::Impl::createTonemapPipeline() {
  VkDevice d = ctx.device;
  if (tonemapPipeline) vkDestroyPipeline(d, tonemapPipeline, nullptr);
  tonemapPipeline = VK_NULL_HANDLE;
  const vk::SpirvBlob vb = vk::shaderSpirv(vk::ShaderId::FullscreenVert);
  const vk::SpirvBlob fb = vk::shaderSpirv(vk::ShaderId::TonemapFrag);
  VkShaderModule v = vk::createShaderModule(d, vb.code, vb.sizeBytes);
  VkShaderModule f = vk::createShaderModule(d, fb.code, fb.sizeBytes);
  if (v && f) {
    vk::GraphicsPipelineDesc g{};
    g.vert = v;
    g.frag = f;
    g.layout = tonemapPipelineLayout;
    g.colorFormat = swapchainFormat;
    g.depthFormat = VK_FORMAT_UNDEFINED;
    g.samples = VK_SAMPLE_COUNT_1_BIT;
    g.cull = VK_CULL_MODE_NONE;
    g.depthTest = false;
    g.depthWrite = false;
    tonemapPipeline = vk::createGraphicsPipeline(d, pipelineCache, g);
  }
  if (v) vkDestroyShaderModule(d, v, nullptr);
  if (f) vkDestroyShaderModule(d, f, nullptr);
  return tonemapPipeline != VK_NULL_HANDLE;
}

// =============================================================================
// Size-dependent targets
// =============================================================================

bool Renderer::Impl::createTargets(uint32_t width, uint32_t height) {
  VkDevice d = ctx.device;
  VmaAllocator a = ctx.allocator;
  extent = {width, height};
  const VkSampleCountFlagBits samples = ctx.caps.msaa;
  const VkFormat hdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

  if (samples != VK_SAMPLE_COUNT_1_BIT) {
    if (!vk::createImage2D(d, a, hdrFormat, extent, 1, samples,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                           VK_IMAGE_ASPECT_COLOR_BIT, hdrMsaa))
      return false;
  }
  if (!vk::createImage2D(d, a, hdrFormat, extent, 1, VK_SAMPLE_COUNT_1_BIT,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, hdr))
    return false;
  // Scene depth is sampled after the scene pass (SSAO, sun shafts): with MSAA
  // it is resolved (sample 0) into a single-sample image, else read directly.
  const bool msaa = samples != VK_SAMPLE_COUNT_1_BIT;
  if (!vk::createImage2D(d, a, ctx.caps.depthFormat, extent, 1, samples,
                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                             (msaa ? VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT : VK_IMAGE_USAGE_SAMPLED_BIT),
                         depthAspect, depth))
    return false;
  if (msaa && !vk::createImage2D(d, a, ctx.caps.depthFormat, extent, 1, VK_SAMPLE_COUNT_1_BIT,
                                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                 depthAspect, depthResolve))
    return false;
  sceneDepthImage = msaa ? depthResolve.image : depth.image;
  sceneDepthView = msaa ? depthResolve.view : depth.view;
  if (!vk::createImage2D(d, a, VK_FORMAT_R32_SFLOAT, {std::max(1u, width / 2), std::max(1u, height / 2)}, 1,
                         VK_SAMPLE_COUNT_1_BIT,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, ao))
    return false;

  // Bloom chain starts at half resolution.
  const VkExtent2D b0{std::max(1u, width / 2), std::max(1u, height / 2)};
  uint32_t mips = 1;
  for (uint32_t m = std::min(b0.width, b0.height); m > 8 && mips < kMaxBloomMips; m /= 2) ++mips;
  bloomMips = mips;
  if (!vk::createImage2D(d, a, hdrFormat, b0, mips, VK_SAMPLE_COUNT_1_BIT,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, bloom))
    return false;
  for (uint32_t m = 0; m < mips; ++m) {
    bloomMipViews[m] = vk::createView(d, bloom.image, hdrFormat, VK_IMAGE_ASPECT_COLOR_BIT, m, 1);
    if (!bloomMipViews[m]) return false;
  }
  writeTargetDescriptors();
  return immediateSubmit(
      [](Impl &I, VkCommandBuffer cmd, void *) {
        vk::imageBarrier(cmd, I.bloom.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                         VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        // Clear so an unused chain (bloom off) samples black.
        VkClearColorValue black{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1};
        vkCmdClearColorImage(cmd, I.bloom.image, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &range);
        // SSAO target lives in GENERAL; start fully unoccluded.
        vk::imageBarrier(cmd, I.ao.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                         VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkClearColorValue white{};
        white.float32[0] = 1.0f;
        VkImageSubresourceRange aoRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, I.ao.image, VK_IMAGE_LAYOUT_GENERAL, &white, 1, &aoRange);
        vk::memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
      },
      nullptr);
}

void Renderer::Impl::writeTargetDescriptors() {
  // Bloom: set i (i < mips) = downsample into mip i; set mips + j = upsample into mip j.
  VkDescriptorImageInfo img[2 * kMaxBloomMips][2]{};
  VkWriteDescriptorSet w[2 * kMaxBloomMips * 2 + 6]{};
  uint32_t n = 0;
  auto add = [&](VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                 const VkDescriptorImageInfo *info) {
    VkWriteDescriptorSet &x = w[n++];
    x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    x.dstSet = set;
    x.dstBinding = binding;
    x.descriptorCount = 1;
    x.descriptorType = type;
    x.pImageInfo = info;
  };
  for (uint32_t i = 0; i < bloomMips; ++i) {
    img[i][0] = {linearSampler, i == 0 ? hdr.view : bloomMipViews[i - 1],
                 i == 0 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL};
    img[i][1] = {VK_NULL_HANDLE, bloomMipViews[i], VK_IMAGE_LAYOUT_GENERAL};
    add(bloomSets[i], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &img[i][0]);
    add(bloomSets[i], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img[i][1]);
  }
  for (uint32_t j = 0; j + 1 < bloomMips; ++j) {
    const uint32_t s = kMaxBloomMips + j;
    img[s][0] = {linearSampler, bloomMipViews[j + 1], VK_IMAGE_LAYOUT_GENERAL};
    img[s][1] = {VK_NULL_HANDLE, bloomMipViews[j], VK_IMAGE_LAYOUT_GENERAL};
    add(bloomSets[s], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &img[s][0]);
    add(bloomSets[s], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img[s][1]);
  }
  VkDescriptorImageInfo tone[4] = {
      {linearSampler, hdr.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
      {linearSampler, bloomMipViews[0], VK_IMAGE_LAYOUT_GENERAL},
      {nearestSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
      {linearSampler, ao.view, VK_IMAGE_LAYOUT_GENERAL}};
  for (uint32_t i = 0; i < 4; ++i)
    add(tonemapSet, i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &tone[i]);
  VkDescriptorImageInfo ssaoImg[2] = {
      {nearestSampler, sceneDepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
      {VK_NULL_HANDLE, ao.view, VK_IMAGE_LAYOUT_GENERAL}};
  add(ssaoSet, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ssaoImg[0]);
  add(ssaoSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ssaoImg[1]);
  vkUpdateDescriptorSets(ctx.device, n, w, 0, nullptr);
}

void Renderer::Impl::destroyTargets() {
  VkDevice d = ctx.device;
  VmaAllocator a = ctx.allocator;
  if (!d) return;
  for (VkImageView &v : bloomMipViews) {
    if (v) vkDestroyImageView(d, v, nullptr);
    v = VK_NULL_HANDLE;
  }
  vk::destroyImage(d, a, bloom);
  vk::destroyImage(d, a, ao);
  vk::destroyImage(d, a, depthResolve);
  vk::destroyImage(d, a, depth);
  sceneDepthImage = VK_NULL_HANDLE;
  sceneDepthView = VK_NULL_HANDLE;
  vk::destroyImage(d, a, hdr);
  vk::destroyImage(d, a, hdrMsaa);
  bloomMips = 0;
}

bool Renderer::Impl::recreateSwapchain() {
  int w = 0, h = 0;
  SDL_GetWindowSizeInPixels(window, &w, &h);
  if (w <= 0 || h <= 0) return false; // minimised: keep the old one, try later
  vkDeviceWaitIdle(ctx.device);
  saveScreenshotIfReady();
  if (!swapchain.create(ctx, uint32_t(w), uint32_t(h), config.vsync)) return false;
  if (swapchain.format() != swapchainFormat) {
    swapchainFormat = swapchain.format();
    createTonemapPipeline();
    // The ImGui pipeline bakes the colour format (dynamic rendering): re-init.
    if (imgui.active()) {
      imgui.shutdown(ctx.device);
      imgui.init(ctx, swapchainFormat, swapchain.imageCount());
    }
  }
  destroyTargets();
  const VkExtent2D ext = swapchain.extent();
  if (!createTargets(ext.width, ext.height)) return false;
  imgui.setImageCount(swapchain.imageCount());
  swapchainDirty = false;
  return true;
}

// =============================================================================
// Screenshot (BMP via SDL_SaveBMP)
// =============================================================================

void Renderer::Impl::saveScreenshotIfReady() {
  if (screenshotPath.empty() || !screenshotBuffer.buffer) return;
  // Only called after the recording frame's fence (or a device wait).
  vk::invalidateBuffer(ctx.allocator, screenshotBuffer);
  SDL_PixelFormat fmt = SDL_PIXELFORMAT_UNKNOWN;
  if (screenshotFormat == VK_FORMAT_B8G8R8A8_UNORM || screenshotFormat == VK_FORMAT_B8G8R8A8_SRGB)
    fmt = SDL_PIXELFORMAT_BGRA32;
  else if (screenshotFormat == VK_FORMAT_R8G8B8A8_UNORM ||
           screenshotFormat == VK_FORMAT_R8G8B8A8_SRGB)
    fmt = SDL_PIXELFORMAT_RGBA32;
  else if (screenshotFormat == VK_FORMAT_A2B10G10R10_UNORM_PACK32)
    fmt = SDL_PIXELFORMAT_ABGR2101010;
  if (fmt != SDL_PIXELFORMAT_UNKNOWN) {
    SDL_Surface *s = SDL_CreateSurfaceFrom(int(screenshotExtent.width), int(screenshotExtent.height),
                                           fmt, screenshotBuffer.mapped,
                                           int(screenshotExtent.width * 4));
    if (s) {
      SDL_Surface *out = SDL_ConvertSurface(s, SDL_PIXELFORMAT_XRGB8888); // opaque BMP
      if (out && SDL_SaveBMP(out, screenshotPath.c_str()))
        SDL_Log("Renderer: screenshot saved to %s", screenshotPath.c_str());
      else
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "screenshot: %s", SDL_GetError());
      if (out) SDL_DestroySurface(out);
      SDL_DestroySurface(s);
    }
  } else {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "screenshot: unsupported swapchain format");
  }
  screenshotPath.clear();
  for (FrameData &f : frames) f.screenshotPending = false;
}

} // namespace atm::render
