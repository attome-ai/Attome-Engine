// Per-frame work: camera, culling, instance lists, command recording, submit
// and present.

#include "RendererImpl.h"

#include <SDL3/SDL.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace atm::render {

namespace {

glm::vec3 srgbToLinear(const glm::vec3 &c) {
  return glm::pow(glm::max(c, glm::vec3(0.0f)), glm::vec3(2.2f));
}

// Reverse-Z, infinite far plane, Vulkan clip space (y down => flip).
glm::mat4 reverseInfinitePerspective(float fovYRadians, float aspect, float zNear) {
  const float f = 1.0f / std::tan(fovYRadians * 0.5f);
  glm::mat4 p(0.0f);
  p[0][0] = f / aspect;
  p[1][1] = -f;
  p[2][3] = -1.0f;  // w = -z_view
  p[3][2] = zNear;  // z = near  => depth = near / -z_view (1 at near, 0 at infinity)
  return p;
}

} // namespace

// =============================================================================
// beginFrame
// =============================================================================

bool Renderer::Impl::beginFrame(const Camera &cam, const Environment &environment) {
  if (!initialised) return false;
  frameStartTicks = SDL_GetPerformanceCounter();
  frameActive = false;

  if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) return false;
  if (swapchainDirty && !recreateSwapchain()) return false;

  VkDevice d = ctx.device;
  FrameData &f = frames[frameIndex];
  vkWaitForFences(d, 1, &f.fence, VK_TRUE, UINT64_MAX);

  if (f.queriesWritten && f.queries) {
    uint64_t ts[2] = {0, 0};
    if (vkGetQueryPoolResults(d, f.queries, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
      const uint64_t ticks = (ts[1] - ts[0]) & ctx.caps.timestampMask;
      stats.gpuMs = float(double(ticks) * double(ctx.caps.timestampPeriod) * 1e-6);
    }
  }
  if (f.screenshotPending) saveScreenshotIfReady();
  processDeferred(f);

  const VkResult acq = vkAcquireNextImageKHR(d, swapchain.handle(), UINT64_MAX, f.imageAvailable,
                                             VK_NULL_HANDLE, &imageIndex);
  if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
    swapchainDirty = true;
    return false;
  }
  if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "vkAcquireNextImageKHR: %s", vk::resultString(acq));
    return false;
  }
  if (acq == VK_SUBOPTIMAL_KHR) swapchainDirty = true; // recreate after this frame
  vkResetFences(d, 1, &f.fence);

  camera = cam;
  env = environment;
  instances.clear();
  highlightValid = false;
  imgui.newFrame();
  frameActive = true;
  return true;
}

// =============================================================================
// Frame uniforms + culling
// =============================================================================

void Renderer::Impl::buildFrameUniforms(FrameData &f) {
  // Camera-relative: integer block + small float fraction.
  const glm::dvec3 camFloor = glm::floor(camera.position);
  camBlock = glm::ivec3(int32_t(camFloor.x), int32_t(camFloor.y), int32_t(camFloor.z));
  camFrac = glm::vec3(camera.position - camFloor);

  const float pitch = std::clamp(camera.pitch, -1.5605f, 1.5605f);
  const glm::vec3 dir(-std::sin(camera.yaw) * std::cos(pitch), std::sin(pitch),
                      -std::cos(camera.yaw) * std::cos(pitch));
  const glm::mat4 view = glm::lookAt(glm::vec3(0.0f), dir, glm::vec3(0.0f, 1.0f, 0.0f));
  const float fovDeg = camera.fovYDegrees > 1.0f ? camera.fovYDegrees : config.fovYDegrees;
  const float aspect = extent.height > 0 ? float(extent.width) / float(extent.height) : 1.0f;
  const float zNear = camera.nearPlane > 0.0f ? camera.nearPlane : 0.05f;
  viewProj = reverseInfinitePerspective(glm::radians(fovDeg), aspect, zNear) * view;

  // Environment.
  glm::vec3 sun = env.sunDirection;
  const float sunLen = glm::length(sun);
  sun = sunLen > 1e-5f ? sun / sunLen : glm::vec3(0.0f, 1.0f, 0.0f);
  const float t = env.timeOfDay;
  const float day = glm::smoothstep(-0.1f, 0.25f, std::sin(6.2831853f * (t - 0.25f)));
  const float dayTint = glm::mix(0.06f, 1.0f, day);
  const float sunI = glm::smoothstep(-0.05f, 0.2f, sun.y) * 1.1f;
  const glm::vec3 sunColor = glm::mix(glm::vec3(1.0f, 0.5f, 0.25f), glm::vec3(1.0f, 0.95f, 0.86f),
                                      glm::smoothstep(0.0f, 0.35f, sun.y));
  const float fogEnd = std::max(config.viewDistanceBlocks * 0.95f, 16.0f);
  const float fogStart = fogEnd * 0.6f;

  // Sun shadow map: orthographic, centred a little ahead of the camera, its
  // texel grid snapped in world space so shadows do not shimmer when moving.
  // Maps camera-relative positions (like every other matrix here).
  const double shadowTexel = 2.0 * double(kShadowRadius) / double(kShadowMapSize);
  {
    const glm::dvec3 L(sun);
    const glm::dvec3 up = std::abs(L.y) > 0.99 ? glm::dvec3(0.0, 0.0, 1.0) : glm::dvec3(0.0, 1.0, 0.0);
    const glm::dmat3 R(glm::lookAt(glm::dvec3(0.0), -L, up));
    glm::dvec3 fwd(dir.x, 0.0, dir.z);
    const double fl = glm::length(fwd);
    fwd = fl > 1e-4 ? fwd / fl : glm::dvec3(0.0);
    const glm::dvec3 centre = camera.position + fwd * (double(kShadowRadius) * 0.45);
    const glm::dvec3 lc = R * centre;
    const glm::dvec2 snapped = glm::floor(glm::dvec2(lc) / shadowTexel) * shadowTexel;
    // Light space of camera-relative p: R * p + off.
    const glm::dvec3 off = R * camera.position - glm::dvec3(snapped, lc.z);
    const double r = kShadowRadius, zr = kShadowDepthRange;
    glm::dmat4 P(1.0);
    P[0][0] = 1.0 / r;
    P[1][1] = 1.0 / r;
    P[2][2] = -0.5 / zr; // depth 0 = towards the sun
    P[3][0] = off.x / r;
    P[3][1] = off.y / r;
    P[3][2] = 0.5 - 0.5 * off.z / zr;
    lightViewProj = glm::mat4(P * glm::dmat4(R));
  }

  gpu::FrameUniforms u{};
  u.viewProj = viewProj;
  u.invViewProj = glm::inverse(viewProj);
  u.camBlock = glm::ivec4(camBlock, 0);
  u.camFrac = glm::vec4(camFrac, float(double(SDL_GetTicks()) * 0.001));
  u.sunDir = glm::vec4(sun, env.ambient * glm::mix(0.25f, 1.0f, day));
  u.skyColor = glm::vec4(srgbToLinear(env.skyColor) * dayTint, t);
  u.fogColor = glm::vec4(srgbToLinear(env.fogColor) * dayTint, fogStart);
  u.fogParams = glm::vec4(fogEnd, 1.0f / (fogEnd - fogStart), sunI, 0.0f);
  u.sunColor = glm::vec4(sunColor, 0.0f);
  u.counts = glm::uvec4(visibleCount, kMaxDraws, 0u, 0u);
  u.lightViewProj = lightViewProj;
  u.shadowParams = glm::vec4(float(shadowTexel), 1.0f, 0.0f, 0.0f);
  std::memcpy(f.ubo.mapped, &u, sizeof(u));
  vk::flushBuffer(ctx.allocator, f.ubo, 0, sizeof(u));
  fogColorLinear = glm::vec3(u.fogColor);
}

void Renderer::Impl::cullAndBuildLists(FrameData &f) {
  // camBlock / camFrac / viewProj were set by buildFrameUniforms().
  const Frustum frustum = Frustum::fromViewProj(viewProj);
  ChunkCuller::Params p;
  p.camBlock = camBlock;
  p.camFrac = camFrac;
  p.viewDistance = config.viewDistanceBlocks;
  p.caveCulling = true;
  culler.run(cullChunks, p, frustum, visibleSlots);
  visibleCount = uint32_t(std::min<size_t>(visibleSlots.size(), kMaxChunkSlots));
  if (visibleCount > 0) {
    std::memcpy(f.visible.mapped, visibleSlots.data(), size_t(visibleCount) * sizeof(uint32_t));
    vk::flushBuffer(ctx.allocator, f.visible, 0, VkDeviceSize(visibleCount) * sizeof(uint32_t));
  }

  // CPU mirror of cull.comp for statistics + translucent ordering.
  opaqueDrawEstimate = 0;
  facesEstimate = 0;
  translucentOrder.clear();
  for (uint32_t i = 0; i < visibleCount; ++i) {
    const uint32_t slot = visibleSlots[i];
    const ChunkRecord &r = chunks[slot];
    const voxel::BlockPos o = voxel::chunkOrigin(r.coord);
    const glm::vec3 cam = glm::vec3(camBlock - glm::ivec3(o.x, o.y, o.z)) + camFrac;
    const ChunkMeshState &s = r.live;
    const bool facing[6] = {cam.x > s.aabbMin[0], cam.x < s.aabbMax[0], cam.y > s.aabbMin[1],
                            cam.y < s.aabbMax[1], cam.z > s.aabbMin[2], cam.z < s.aabbMax[2]};
    for (size_t d = 0; d < 6; ++d) {
      const uint32_t n = s.dirOffset[d + 1] - s.dirOffset[d];
      if (facing[d] && n > 0) {
        ++opaqueDrawEstimate;
        facesEstimate += n;
      }
    }
    if (s.translucentCount > 0) {
      const glm::vec3 c = glm::vec3(16.0f) - cam;
      translucentOrder.emplace_back(glm::dot(c, c), slot);
      facesEstimate += s.translucentCount;
    }
  }
  // Back to front.
  std::sort(translucentOrder.begin(), translucentOrder.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
  auto *cmds = static_cast<gpu::DrawIndexedIndirect *>(f.translucentDraws.mapped);
  translucentCount = 0;
  for (const auto &e : translucentOrder) {
    const ChunkMeshState &s = chunks[e.second].live;
    gpu::DrawIndexedIndirect &c = cmds[translucentCount++];
    c.indexCount = std::min(s.translucentCount, kMaxFacesPerDraw) * 6u;
    c.instanceCount = 1;
    c.firstIndex = 0;
    c.vertexOffset = int32_t((s.base + s.dirOffset[6]) * 4u);
    c.firstInstance = e.second;
  }
  if (translucentCount > 0)
    vk::flushBuffer(ctx.allocator, f.translucentDraws, 0,
                    VkDeviceSize(translucentCount) * sizeof(gpu::DrawIndexedIndirect));
}

void Renderer::Impl::buildShadowList(FrameData &f) {
  // Every resident chunk inside the light's box casts, including ones outside
  // the camera frustum (a tree behind the camera still shades the ground in
  // front of it). One draw per chunk covers all its opaque direction groups.
  auto *cmds = static_cast<gpu::DrawIndexedIndirect *>(f.shadowDraws.mapped);
  shadowDrawCount = 0;
  const float margin = 28.0f / kShadowRadius;           // chunk half-diagonal in NDC
  const float zMargin = 28.0f / (2.0f * kShadowDepthRange);
  for (const auto &entry : slotOf) {
    const uint32_t slot = entry.second;
    const ChunkRecord &r = chunks[slot];
    const ChunkMeshState &s = r.live;
    if (!r.used || s.units == 0 || s.base == FaceAllocator::kInvalid) continue;
    const uint32_t n = s.dirOffset[6] - s.dirOffset[0];
    if (n == 0) continue;
    const voxel::BlockPos o = voxel::chunkOrigin(r.coord);
    const glm::vec3 c = glm::vec3(glm::ivec3(o.x, o.y, o.z) - camBlock) - camFrac + glm::vec3(16.0f);
    const glm::vec4 lc = lightViewProj * glm::vec4(c, 1.0f);
    if (std::abs(lc.x) > 1.0f + margin || std::abs(lc.y) > 1.0f + margin || lc.z < -zMargin ||
        lc.z > 1.0f + zMargin)
      continue;
    gpu::DrawIndexedIndirect &d = cmds[shadowDrawCount++];
    d.indexCount = std::min(n, kMaxFacesPerDraw) * 6u;
    d.instanceCount = 1;
    d.firstIndex = 0;
    d.vertexOffset = int32_t((s.base + s.dirOffset[0]) * 4u);
    d.firstInstance = slot;
  }
  if (shadowDrawCount > 0)
    vk::flushBuffer(ctx.allocator, f.shadowDraws, 0,
                    VkDeviceSize(shadowDrawCount) * sizeof(gpu::DrawIndexedIndirect));
}

uint32_t Renderer::Impl::writeModelInstances(FrameData &f) {
  modelDraws.clear();
  if (instances.empty()) return 0;
  std::sort(instances.begin(), instances.end(), [](const PendingInstance &a, const PendingInstance &b) {
    return a.mesh != b.mesh ? a.mesh < b.mesh : a.order < b.order;
  });
  auto *dst = static_cast<gpu::ModelInstanceGpu *>(f.instances.mapped);
  uint32_t n = 0;
  ModelMeshId lastMesh = kInvalidModelMesh;
  for (const PendingInstance &pi : instances) {
    const ModelRecord &m = models[pi.mesh];
    if (!m.used || !m.ready || m.units == 0) continue;
    const ModelInstance &in = pi.instance;
    // Built in double, camera-relative, then converted to float.
    const glm::dvec3 rel = in.origin - camera.position;
    glm::dmat4 M = glm::translate(glm::dmat4(1.0), rel);
    M = M * glm::mat4_cast(glm::dquat(in.rotation));
    M = glm::scale(M, glm::dvec3(double(in.voxelScale)));
    M = glm::translate(M, -glm::dvec3(in.pivot));
    gpu::ModelInstanceGpu &g = dst[n];
    g.model = glm::mat4(M);
    g.tint = in.tint;
    g.paletteOffset = in.paletteOffset;
    g.pad0 = g.pad1 = 0;
    if (pi.mesh != lastMesh) {
      modelDraws.push_back({m.base, m.units, n, 0});
      lastMesh = pi.mesh;
    }
    ++modelDraws.back().instanceCount;
    ++n;
  }
  if (n > 0)
    vk::flushBuffer(ctx.allocator, f.instances, 0, VkDeviceSize(n) * sizeof(gpu::ModelInstanceGpu));
  return n;
}

// =============================================================================
// Command recording
// =============================================================================

void Renderer::Impl::recordBloom(VkCommandBuffer cmd) {
  // The previous frame's tonemap read the chain: finish before overwriting.
  vk::memoryBarrier(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_NONE);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bloomPipeline);
  auto mipSize = [&](uint32_t m) {
    return VkExtent2D{std::max(1u, bloom.extent.width >> m), std::max(1u, bloom.extent.height >> m)};
  };
  auto step = [&](VkDescriptorSet set, VkExtent2D src, VkExtent2D dst, uint32_t mode) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bloomPipelineLayout, 0, 1, &set,
                            0, nullptr);
    gpu::BloomPush p{};
    p.srcTexel = glm::vec2(1.0f / float(src.width), 1.0f / float(src.height));
    p.threshold = 1.0f;
    p.intensity = 1.0f;
    p.mode = mode;
    vkCmdPushConstants(cmd, bloomPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
    vkCmdDispatch(cmd, (dst.width + 7) / 8, (dst.height + 7) / 8, 1);
    // Next step samples this mip, and the upsample pass later read-modify-writes
    // mips written here (write-after-write): make writes visible to both.
    vk::memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  };
  for (uint32_t i = 0; i < bloomMips; ++i)
    step(bloomSets[i], i == 0 ? extent : mipSize(i - 1), mipSize(i), i == 0 ? 0u : 1u);
  for (uint32_t j = bloomMips - 1; j-- > 0;)
    step(bloomSets[kMaxBloomMips + j], mipSize(j + 1), mipSize(j), 2u);
}

void Renderer::Impl::recordShadowPass(FrameData &f, uint32_t instanceCount) {
  VkCommandBuffer cmd = f.cmd;
  const VkPipelineStageFlags2 depthStages =
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
  // The previous frame's scene pass sampled it: wait, then discard.
  vk::imageBarrier(cmd, shadowMap.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_NONE, depthStages,
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

  VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  da.imageView = shadowMap.view;
  da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  da.clearValue.depthStencil = {1.0f, 0}; // standard depth: 1 = far from the sun
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {kShadowMapSize, kShadowMapSize}};
  ri.layerCount = 1;
  ri.pDepthAttachment = &da;
  vkCmdBeginRendering(cmd, &ri);

  const VkViewport vp{0.0f, 0.0f, float(kShadowMapSize), float(kShadowMapSize), 0.0f, 1.0f};
  const VkRect2D sc{{0, 0}, {kShadowMapSize, kShadowMapSize}};
  vkCmdSetViewport(cmd, 0, 1, &vp);
  vkCmdSetScissor(cmd, 0, 1, &sc);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout, 0, 1,
                          &f.sceneSet, 0, nullptr);
  vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
  if (shadowDrawCount > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
    vkCmdDrawIndexedIndirect(cmd, f.shadowDraws.buffer, 0, shadowDrawCount,
                             sizeof(gpu::DrawIndexedIndirect));
  }
  if (instanceCount > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowModelPipeline);
    for (const ModelDraw &md : modelDraws)
      vkCmdDrawIndexed(cmd, md.faceCount * 6u, md.instanceCount, 0, int32_t(md.faceBase * 4u),
                       md.firstInstance);
  }
  vkCmdEndRendering(cmd);

  vk::imageBarrier(cmd, shadowMap.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, depthStages,
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void Renderer::Impl::recordFrame(FrameData &f, uint32_t translucentDraws, uint32_t instanceCount) {
  VkCommandBuffer cmd = f.cmd;
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);

  if (f.queries) {
    vkCmdResetQueryPool(cmd, f.queries, 0, 2);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, f.queries, 0);
  }

  // --- uploads + GPU culling ------------------------------------------------------
  recordUploads(f, cmd);
  vkCmdFillBuffer(cmd, f.drawCount.buffer, 0, sizeof(uint32_t), 0u);
  vk::memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scenePipelineLayout, 0, 1,
                          &f.sceneSet, 0, nullptr);
  if (visibleCount > 0) vkCmdDispatch(cmd, (visibleCount + 63) / 64, 1, 1);
  vk::memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                    VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

  // --- sun shadow map ----------------------------------------------------------------
  recordShadowPass(f, instanceCount);

  // --- scene pass (HDR, MSAA, reverse-Z) ----------------------------------------
  const bool msaa = ctx.caps.msaa != VK_SAMPLE_COUNT_1_BIT;
  const VkPipelineStageFlags2 prevReaders =
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  vk::imageBarrier(cmd, hdr.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   prevReaders | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_NONE,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
  if (msaa)
    vk::imageBarrier(cmd, hdrMsaa.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_NONE,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
  const VkPipelineStageFlags2 depthStages =
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
  vk::imageBarrier(cmd, depth.image, depthAspect, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, depthStages,
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, depthStages,
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

  VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  color.imageView = msaa ? hdrMsaa.view : hdr.view;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = msaa ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
  color.clearValue.color = {{fogColorLinear.r, fogColorLinear.g, fogColorLinear.b, 1.0f}};
  if (msaa) {
    color.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
    color.resolveImageView = hdr.view;
    color.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  }
  VkRenderingAttachmentInfo depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  depthAtt.imageView = depth.view;
  depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAtt.clearValue.depthStencil = {0.0f, 0}; // reverse-Z: 0 = infinitely far

  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, extent};
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &color;
  ri.pDepthAttachment = &depthAtt;
  vkCmdBeginRendering(cmd, &ri);

  const VkViewport viewport{0.0f, 0.0f, float(extent.width), float(extent.height), 0.0f, 1.0f};
  const VkRect2D scissor{{0, 0}, extent};
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout, 0, 1,
                          &f.sceneSet, 0, nullptr);
  vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

  // Opaque terrain: GPU-generated draws, count from cull.comp.
  if (visibleCount > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, opaquePipeline);
    vkCmdDrawIndexedIndirectCount(cmd, f.draws.buffer, 0, f.drawCount.buffer, 0,
                                  std::min(visibleCount * 6u, kMaxDraws),
                                  sizeof(gpu::DrawIndexedIndirect));
  }
  // Models, instanced per mesh.
  if (instanceCount > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, modelPipeline);
    for (const ModelDraw &md : modelDraws)
      vkCmdDrawIndexed(cmd, md.faceCount * 6u, md.instanceCount, 0, int32_t(md.faceBase * 4u),
                       md.firstInstance);
  }
  // Sky where nothing was drawn (depth still 0).
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  // Translucent, back to front.
  if (translucentDraws > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, translucentPipeline);
    vkCmdDrawIndexedIndirect(cmd, f.translucentDraws.buffer, 0, translucentDraws,
                             sizeof(gpu::DrawIndexedIndirect));
  }
  // Block highlight.
  if (highlightValid) {
    const glm::vec3 rel = glm::vec3(glm::ivec3(highlightBlock.x, highlightBlock.y, highlightBlock.z) -
                                    camBlock) - camFrac;
    gpu::HighlightPush hp{};
    hp.minCorner = glm::vec4(rel - glm::vec3(0.004f), 0.0f);
    hp.maxCorner = glm::vec4(rel + glm::vec3(1.004f), 0.0f);
    hp.color = glm::vec4(0.02f, 0.02f, 0.02f, 1.0f);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, highlightPipeline);
    vkCmdPushConstants(cmd, scenePipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(hp), &hp);
    vkCmdDraw(cmd, 24, 1, 0, 0);
  }
  vkCmdEndRendering(cmd);

  vk::imageBarrier(cmd, hdr.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

  // --- bloom -----------------------------------------------------------------------
  if (config.bloom && bloomMips > 0) recordBloom(cmd);

  // --- tonemap + UI into the swapchain image ------------------------------------------
  const VkImage swapImage = swapchain.images[imageIndex];
  vk::imageBarrier(cmd, swapImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_NONE,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
  VkRenderingAttachmentInfo out{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  out.imageView = swapchain.views[imageIndex];
  out.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  out.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  out.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkRenderingInfo ui{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ui.renderArea = {{0, 0}, swapchain.extent()};
  ui.layerCount = 1;
  ui.colorAttachmentCount = 1;
  ui.pColorAttachments = &out;
  vkCmdBeginRendering(cmd, &ui);
  const VkViewport vp2{0.0f, 0.0f, float(swapchain.extent().width), float(swapchain.extent().height),
                       0.0f, 1.0f};
  const VkRect2D sc2{{0, 0}, swapchain.extent()};
  vkCmdSetViewport(cmd, 0, 1, &vp2);
  vkCmdSetScissor(cmd, 0, 1, &sc2);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemapPipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemapPipelineLayout, 0, 1,
                          &tonemapSet, 0, nullptr);
  gpu::TonemapPush tp{};
  tp.exposure = 1.0f;
  tp.bloomStrength = 0.5f;
  tp.bloomEnabled = (config.bloom && bloomMips > 0) ? 1u : 0u;
  tp.srgbOutput = swapchain.srgb() ? 1u : 0u;
  vkCmdPushConstants(cmd, tonemapPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(tp), &tp);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  imgui.render(cmd);
  vkCmdEndRendering(cmd);

  // --- screenshot readback + present transition -----------------------------------------
  bool screenshotRecorded = false;
  if (!screenshotRequest.empty() && screenshotPath.empty()) {
    const VkExtent2D se = swapchain.extent();
    const VkDeviceSize bytes = VkDeviceSize(se.width) * se.height * 4;
    if (screenshotBuffer.size < bytes) {
      vk::destroyBuffer(ctx.allocator, screenshotBuffer); // no frame uses it (path empty)
      vk::createBuffer(ctx.allocator, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       vk::MemoryKind::Readback, screenshotBuffer, "screenshot");
    }
    if (screenshotBuffer.buffer && screenshotBuffer.mapped) {
      vk::imageBarrier(cmd, swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT);
      VkBufferImageCopy region{};
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.imageExtent = {se.width, se.height, 1};
      vkCmdCopyImageToBuffer(cmd, swapImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             screenshotBuffer.buffer, 1, &region);
      vk::memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
      vk::imageBarrier(cmd, swapImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
      screenshotPath = screenshotRequest;
      screenshotExtent = se;
      screenshotFormat = swapchain.format();
      f.screenshotPending = true;
      screenshotRecorded = true;
    }
    screenshotRequest.clear();
  }
  if (!screenshotRecorded) {
    vk::imageBarrier(cmd, swapImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE,
                     VK_ACCESS_2_NONE);
  }

  if (f.queries) {
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 1);
    f.queriesWritten = true;
  }
  vkEndCommandBuffer(cmd);
}

// =============================================================================
// endFrame: build lists, record, submit, present
// =============================================================================

void Renderer::Impl::endFrame() {
  if (!frameActive) return;
  frameActive = false;
  FrameData &f = frames[frameIndex];

  // Camera matrices first (culling needs them), then the UBO with counts.
  visibleCount = 0;
  buildFrameUniforms(f);
  cullAndBuildLists(f);
  buildFrameUniforms(f); // rewrite with the final visible count
  buildShadowList(f);
  const uint32_t instanceCount = writeModelInstances(f);
  recordFrame(f, translucentCount, instanceCount);

  VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  wait.semaphore = f.imageAvailable;
  wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  signal.semaphore = swapchain.renderFinished[imageIndex];
  signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cbi.commandBuffer = f.cmd;
  VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  si.waitSemaphoreInfoCount = 1;
  si.pWaitSemaphoreInfos = &wait;
  si.commandBufferInfoCount = 1;
  si.pCommandBufferInfos = &cbi;
  si.signalSemaphoreInfoCount = 1;
  si.pSignalSemaphoreInfos = &signal;
  if (!ATM_VK_OK(vkQueueSubmit2(ctx.graphicsQueue, 1, &si, f.fence))) {
    // The fence will never signal: recreate it signalled so the next wait works.
    vkDestroyFence(ctx.device, f.fence, nullptr);
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(ctx.device, &fci, nullptr, &f.fence);
    f.queriesWritten = false;
    f.screenshotPending = false;
    screenshotPath.clear();
    swapchainDirty = true;
  } else {
    const VkSwapchainKHR sc = swapchain.handle();
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &swapchain.renderFinished[imageIndex];
    pi.swapchainCount = 1;
    pi.pSwapchains = &sc;
    pi.pImageIndices = &imageIndex;
    const VkResult pr = vkQueuePresentKHR(ctx.graphicsQueue, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) swapchainDirty = true;
    else if (pr != VK_SUCCESS)
      SDL_LogError(SDL_LOG_CATEGORY_RENDER, "vkQueuePresentKHR: %s", vk::resultString(pr));
  }

  // --- stats ------------------------------------------------------------------------
  stats.drawCalls = opaqueDrawEstimate + translucentCount + uint32_t(modelDraws.size());
  stats.chunksVisible = visibleCount;
  stats.chunksResident = uint32_t(cullChunks.size());
  stats.facesDrawn = facesEstimate;
  for (const ModelDraw &md : modelDraws) stats.facesDrawn += uint64_t(md.faceCount) * md.instanceCount;
  stats.uploadBytes = uploadBytesThisFrame;
  stats.chunkArenaUsed = uint64_t(arenaAlloc.used()) * 8u;
  stats.chunkArenaCapacity = uint64_t(arenaAlloc.capacity()) * 8u;
  stats.chunkArenaFragmentation = arenaAlloc.fragmentation();
  stats.chunkUploadsPending = uint32_t(uploadQueue.size() - uploadHead);
  stats.modelInstances = instanceCount;
  stats.cpuMs = float(double(SDL_GetPerformanceCounter() - frameStartTicks) * 1000.0 /
                      double(SDL_GetPerformanceFrequency()));

  frameIndex = (frameIndex + 1) % framesInFlight;
  ++frameCounter;
}

} // namespace atm::render
