// Chunk + model geometry management: face arena allocation, budgeted uploads
// through the per-frame staging ring, chunk metadata and materials.

#include "RendererImpl.h"

#include <algorithm>
#include <cstring>

namespace atm::render {

using voxel::PackedFace;

namespace {

void buildMeshState(const voxel::ChunkMeshData &mesh, ChunkMeshState &s) {
  const uint32_t opaqueCount = uint32_t(mesh.opaque.size());
  uint32_t prev = 0;
  for (int d = 0; d < 6; ++d) {
    // Clamp to the real face count and keep offsets monotonic (defensive).
    uint32_t off = std::min(mesh.opaqueDirOffset[size_t(d)], opaqueCount);
    off = std::max(off, prev);
    s.dirOffset[size_t(d)] = off;
    prev = off;
  }
  s.dirOffset[6] = opaqueCount;
  s.translucentCount = uint32_t(mesh.translucent.size());
  s.units = opaqueCount + s.translucentCount;
  s.base = FaceAllocator::kInvalid;
  s.connectivity = mesh.faceConnectivity;
  const bool validBox = mesh.minX <= mesh.maxX && mesh.minY <= mesh.maxY && mesh.minZ <= mesh.maxZ;
  if (validBox) {
    // max may be inclusive (block index) or exclusive: +1 is conservative.
    s.aabbMin[0] = std::min<uint8_t>(mesh.minX, 32);
    s.aabbMin[1] = std::min<uint8_t>(mesh.minY, 32);
    s.aabbMin[2] = std::min<uint8_t>(mesh.minZ, 32);
    s.aabbMax[0] = uint8_t(std::min(int(mesh.maxX) + 1, 32));
    s.aabbMax[1] = uint8_t(std::min(int(mesh.maxY) + 1, 32));
    s.aabbMax[2] = uint8_t(std::min(int(mesh.maxZ) + 1, 32));
  } else {
    for (int i = 0; i < 3; ++i) {
      s.aabbMin[i] = 0;
      s.aabbMax[i] = 32;
    }
  }
}

void releaseVector(std::vector<PackedFace> &v) {
  v.clear();
  v.shrink_to_fit();
}

} // namespace

// =============================================================================
// Materials
// =============================================================================

void Renderer::Impl::setMaterials(std::span<const Material> m, uint32_t base) {
  if (materialsCpu.empty() || m.empty()) return;
  const uint32_t n = uint32_t(std::min<size_t>(m.size(), kMaxMaterials - base));
  for (uint32_t i = 0; i < n; ++i) {
    const Material &src = m[i];
    gpu::MaterialGpu &dst = materialsCpu[base + i];
    dst.top = src.top;
    dst.side = src.side;
    dst.bottom = src.bottom;
    dst.emissive = src.emissive;
    dst.alpha = src.alpha;
    dst.flags = src.flags;
  }
  if (materialsDirtyBegin == materialsDirtyEnd) {
    materialsDirtyBegin = base;
    materialsDirtyEnd = base + n;
  } else {
    materialsDirtyBegin = std::min(materialsDirtyBegin, base);
    materialsDirtyEnd = std::max(materialsDirtyEnd, base + n);
  }
}

// =============================================================================
// Chunks
// =============================================================================

void Renderer::Impl::markMetaDirty(uint32_t slot) {
  ChunkRecord &r = chunks[slot];
  if (!r.metaDirty) {
    r.metaDirty = true;
    dirtySlots.push_back(slot);
  }
}

void Renderer::Impl::deferFree(FaceAllocator &alloc, uint32_t offset, uint32_t units) {
  if (offset == FaceAllocator::kInvalid || units == 0) return;
  // The range may be referenced by any frame submitted so far, but never by
  // the frame being built (it no longer references the range). The most
  // recently submitted slot's fence covers all submitted work; its list is
  // processed when that slot is reused, right after waiting on that fence.
  const uint32_t lastSubmitted = (frameIndex + framesInFlight - 1) % framesInFlight;
  frames[lastSubmitted].deferred.push_back({&alloc, offset, units});
}

void Renderer::Impl::processDeferred(FrameData &f) {
  for (const DeferredFree &d : f.deferred) d.allocator->free(d.offset, d.units);
  f.deferred.clear();
}

void Renderer::Impl::setChunkMesh(voxel::ChunkCoord coord, const voxel::ChunkMeshData &mesh) {
  if (!initialised) return;
  // Air chunks (no faces, fully open) are simply not resident: the culler
  // treats missing chunks as open space.
  if (mesh.empty() && mesh.faceConnectivity == ~0ull) {
    removeChunk(coord);
    return;
  }

  uint32_t slot;
  auto it = slotOf.find(coord.key());
  if (it != slotOf.end()) {
    slot = it->second;
  } else {
    if (freeSlots.empty()) {
      SDL_LogWarn(SDL_LOG_CATEGORY_RENDER, "Renderer: out of chunk slots (%u)", kMaxChunkSlots);
      return;
    }
    slot = freeSlots.back();
    freeSlots.pop_back();
    slotOf.emplace(coord.key(), slot);
    ChunkRecord &r = chunks[slot];
    r.used = true;
    r.coord = coord;
    r.live = ChunkMeshState{};
    r.pending = false;
    r.uploadedUnits = 0;
    r.cullIndex = uint32_t(cullChunks.size());
    CullChunk cc;
    cc.coord = coord;
    cc.slot = slot;
    cc.connectivity = mesh.faceConnectivity;
    cc.drawable = false;
    cullChunks.push_back(cc);
  }

  ChunkRecord &r = chunks[slot];
  // A pending (never drawn) range: copies on the same queue are ordered, so
  // immediate reuse would be safe; deferring anyway keeps this correct if
  // uploads move to a dedicated transfer queue later.
  if (r.pending && r.next.base != FaceAllocator::kInvalid)
    deferFree(arenaAlloc, r.next.base, r.next.units);
  buildMeshState(mesh, r.next);
  if (r.next.units > kMaxFacesPerDraw * 2) {
    SDL_LogWarn(SDL_LOG_CATEGORY_RENDER, "Renderer: chunk mesh too large (%u faces)", r.next.units);
  }

  if (r.next.units == 0) {
    // Solid/hidden chunk: nothing to draw but keeps its connectivity.
    deferFree(arenaAlloc, r.live.base, r.live.units);
    r.live = r.next;
    r.pending = false;
    releaseVector(r.nextFaces);
    CullChunk &cc = cullChunks[r.cullIndex];
    cc.connectivity = r.live.connectivity;
    cc.drawable = false;
    markMetaDirty(slot);
    return;
  }

  r.nextFaces.clear();
  r.nextFaces.reserve(r.next.units);
  r.nextFaces.insert(r.nextFaces.end(), mesh.opaque.begin(), mesh.opaque.end());
  r.nextFaces.insert(r.nextFaces.end(), mesh.translucent.begin(), mesh.translucent.end());
  r.uploadedUnits = 0;
  r.pending = true;
  if (!r.queued) {
    r.queued = true;
    uploadQueue.push_back(slot);
  }
}

void Renderer::Impl::removeChunk(voxel::ChunkCoord coord) {
  if (!initialised) return;
  auto it = slotOf.find(coord.key());
  if (it == slotOf.end()) return;
  const uint32_t slot = it->second;
  slotOf.erase(it);
  ChunkRecord &r = chunks[slot];
  deferFree(arenaAlloc, r.live.base, r.live.units);
  // The pending range may be the target of an in-flight upload copy, so it
  // is released through the deferred path like the live range.
  if (r.pending && r.next.base != FaceAllocator::kInvalid)
    deferFree(arenaAlloc, r.next.base, r.next.units);
  r.pending = false;
  r.live = ChunkMeshState{};
  r.next = ChunkMeshState{};
  releaseVector(r.nextFaces);
  r.used = false;
  // `queued` stays: the stale queue entry is skipped (not pending) and clears it.

  const uint32_t idx = r.cullIndex;
  const CullChunk last = cullChunks.back();
  cullChunks[idx] = last;
  chunks[last.slot].cullIndex = idx;
  cullChunks.pop_back();

  freeSlots.push_back(slot);
  markMetaDirty(slot); // writes flags = 0
}

gpu::ChunkGpu Renderer::Impl::chunkGpu(uint32_t slot) const {
  gpu::ChunkGpu g{};
  const ChunkRecord &r = chunks[slot];
  if (!r.used || r.live.units == 0 || r.live.base == FaceAllocator::kInvalid) return g;
  const voxel::BlockPos o = voxel::chunkOrigin(r.coord);
  g.originX = o.x;
  g.originY = o.y;
  g.originZ = o.z;
  g.flags = 1u;
  g.faceBase = r.live.base;
  g.translucentCount = r.live.translucentCount;
  for (size_t d = 0; d < 7; ++d) g.dirOffset[d] = r.live.dirOffset[d];
  g.aabbMin = uint32_t(r.live.aabbMin[0]) | uint32_t(r.live.aabbMin[1]) << 8 |
              uint32_t(r.live.aabbMin[2]) << 16;
  g.aabbMax = uint32_t(r.live.aabbMax[0]) | uint32_t(r.live.aabbMax[1]) << 8 |
              uint32_t(r.live.aabbMax[2]) << 16;
  return g;
}

// =============================================================================
// Models
// =============================================================================

ModelMeshId Renderer::Impl::createModelMesh(const voxel::ChunkMeshData &mesh) {
  if (!initialised) return kInvalidModelMesh;
  ModelMeshId id;
  if (!freeModelIds.empty()) {
    id = freeModelIds.back();
    freeModelIds.pop_back();
  } else {
    id = ModelMeshId(models.size());
    models.emplace_back();
  }
  ModelRecord &m = models[id];
  m.used = true;
  m.ready = false;
  m.base = FaceAllocator::kInvalid;
  m.uploadedUnits = 0;
  m.faces.clear();
  m.faces.reserve(mesh.opaque.size() + mesh.translucent.size());
  m.faces.insert(m.faces.end(), mesh.opaque.begin(), mesh.opaque.end());
  m.faces.insert(m.faces.end(), mesh.translucent.begin(), mesh.translucent.end());
  if (m.faces.size() > kMaxFacesPerDraw) m.faces.resize(kMaxFacesPerDraw);
  m.units = uint32_t(m.faces.size());
  if (m.units == 0) {
    m.ready = true; // valid, draws nothing
    return id;
  }
  if (!m.queued) {
    m.queued = true;
    modelUploadQueue.push_back(id);
  }
  return id;
}

void Renderer::Impl::destroyModelMesh(ModelMeshId id) {
  if (!initialised || id >= models.size() || !models[id].used) return;
  ModelRecord &m = models[id];
  if (m.base != FaceAllocator::kInvalid) {
    if (m.ready) deferFree(modelAlloc, m.base, m.units);   // may be drawn in flight
    else modelAlloc.free(m.base, m.units);                 // never drawn
  }
  m.base = FaceAllocator::kInvalid;
  m.used = false;
  m.ready = false;
  m.units = 0;
  releaseVector(m.faces);
  freeModelIds.push_back(id);
}

// =============================================================================
// Uploads (recorded at the start of the frame's command buffer)
// =============================================================================

void Renderer::Impl::recordUploads(FrameData &f, VkCommandBuffer cmd) {
  arenaCopies.clear();
  modelCopies.clear();
  metaCopies.clear();
  materialCopies.clear();
  auto *stage = static_cast<uint8_t *>(f.staging.mapped);
  VkDeviceSize cursor = 0;
  const VkDeviceSize faceBudget =
      std::min<VkDeviceSize>(config.uploadBudgetBytes, f.staging.size - kMetaStagingBytes);

  // Copies `units` faces (or as many as the budget allows) into staging.
  auto stageFaces = [&](const PackedFace *src, uint32_t units, uint32_t dstUnit,
                        std::vector<VkBufferCopy> &copies) -> uint32_t {
    const VkDeviceSize room = (faceBudget - std::min(cursor, faceBudget)) & ~VkDeviceSize(7);
    const uint32_t take = uint32_t(std::min<VkDeviceSize>(VkDeviceSize(units) * 8, room) / 8);
    if (take == 0) return 0;
    std::memcpy(stage + cursor, src, size_t(take) * 8);
    copies.push_back({cursor, VkDeviceSize(dstUnit) * 8, VkDeviceSize(take) * 8});
    cursor += VkDeviceSize(take) * 8;
    return take;
  };

  // 1. Model meshes first (small, needed for characters to appear).
  size_t mi = 0;
  for (; mi < modelUploadQueue.size(); ++mi) {
    ModelRecord &m = models[modelUploadQueue[mi]];
    if (!m.used || m.ready) {
      m.queued = false;
      continue;
    }
    if (m.base == FaceAllocator::kInvalid) {
      m.base = modelAlloc.allocate(m.units);
      if (m.base == FaceAllocator::kInvalid) {
        SDL_LogWarn(SDL_LOG_CATEGORY_RENDER, "Renderer: model face buffer full");
        break;
      }
    }
    const uint32_t remaining = m.units - m.uploadedUnits;
    const uint32_t took = stageFaces(m.faces.data() + m.uploadedUnits, remaining,
                                     m.base + m.uploadedUnits, modelCopies);
    m.uploadedUnits += took;
    if (m.uploadedUnits < m.units) break; // budget used up
    m.ready = true;
    m.queued = false;
    releaseVector(m.faces);
  }
  modelUploadQueue.erase(modelUploadQueue.begin(), modelUploadQueue.begin() + ptrdiff_t(mi));

  // 2. Chunk meshes (FIFO), switched atomically once fully uploaded.
  while (uploadHead < uploadQueue.size()) {
    const uint32_t slot = uploadQueue[uploadHead];
    ChunkRecord &r = chunks[slot];
    if (!r.used || !r.pending) {
      r.queued = false;
      ++uploadHead;
      continue;
    }
    if (r.next.base == FaceAllocator::kInvalid) {
      r.next.base = arenaAlloc.allocate(r.next.units);
      if (r.next.base == FaceAllocator::kInvalid) {
        if (!arenaFullLogged) {
          SDL_LogWarn(SDL_LOG_CATEGORY_RENDER,
                      "Renderer: chunk face arena full (%u of %u faces used, fragmentation %.2f)",
                      arenaAlloc.used(), arenaAlloc.capacity(), double(arenaAlloc.fragmentation()));
          arenaFullLogged = true;
        }
        break; // retry next frame (removals free space)
      }
      arenaFullLogged = false;
      r.uploadedUnits = 0;
    }
    const uint32_t remaining = r.next.units - r.uploadedUnits;
    const uint32_t took = stageFaces(r.nextFaces.data() + r.uploadedUnits, remaining,
                                     r.next.base + r.uploadedUnits, arenaCopies);
    r.uploadedUnits += took;
    if (r.uploadedUnits < r.next.units) break; // continues next frame

    deferFree(arenaAlloc, r.live.base, r.live.units);
    r.live = r.next;
    r.next = ChunkMeshState{};
    r.pending = false;
    r.queued = false;
    releaseVector(r.nextFaces);
    CullChunk &cc = cullChunks[r.cullIndex];
    cc.connectivity = r.live.connectivity;
    for (int i = 0; i < 3; ++i) {
      cc.aabbMin[i] = r.live.aabbMin[i];
      cc.aabbMax[i] = r.live.aabbMax[i];
    }
    cc.drawable = true;
    markMetaDirty(slot);
    ++uploadHead;
  }
  if (uploadHead >= uploadQueue.size()) {
    uploadQueue.clear();
    uploadHead = 0;
  } else if (uploadHead > 4096 && uploadHead * 2 > uploadQueue.size()) {
    uploadQueue.erase(uploadQueue.begin(), uploadQueue.begin() + ptrdiff_t(uploadHead));
    uploadHead = 0;
  }

  // 3. Materials (dirty range, may span frames).
  cursor = (cursor + 15) & ~VkDeviceSize(15);
  if (materialsDirtyBegin < materialsDirtyEnd) {
    const VkDeviceSize room = f.staging.size - cursor;
    const uint32_t count = uint32_t(std::min<VkDeviceSize>(
        materialsDirtyEnd - materialsDirtyBegin, room / sizeof(gpu::MaterialGpu)));
    if (count > 0) {
      const VkDeviceSize bytes = VkDeviceSize(count) * sizeof(gpu::MaterialGpu);
      std::memcpy(stage + cursor, materialsCpu.data() + materialsDirtyBegin, size_t(bytes));
      materialCopies.push_back(
          {cursor, VkDeviceSize(materialsDirtyBegin) * sizeof(gpu::MaterialGpu), bytes});
      cursor += bytes;
      materialsDirtyBegin += count;
      if (materialsDirtyBegin >= materialsDirtyEnd) materialsDirtyBegin = materialsDirtyEnd = 0;
    }
  }

  // 4. Chunk metadata (dirty slots).
  while (!dirtySlots.empty() && cursor + sizeof(gpu::ChunkGpu) <= f.staging.size) {
    const uint32_t slot = dirtySlots.back();
    dirtySlots.pop_back();
    chunks[slot].metaDirty = false;
    const gpu::ChunkGpu g = chunkGpu(slot);
    std::memcpy(stage + cursor, &g, sizeof(g));
    metaCopies.push_back({cursor, VkDeviceSize(slot) * sizeof(gpu::ChunkGpu), sizeof(g)});
    cursor += sizeof(g);
  }

  uploadBytesThisFrame = cursor;
  if (cursor == 0) return;
  vk::flushBuffer(ctx.allocator, f.staging, 0, cursor);

  // Earlier frames' shader reads of these buffers must finish before the
  // copies overwrite them (write-after-read), and earlier copies into a range
  // that was freed without ever being drawn (destroyModelMesh of a partially
  // uploaded mesh) must finish before a new copy reuses it (write-after-write).
  vk::memoryBarrier(cmd,
                    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT);
  if (!modelCopies.empty())
    vkCmdCopyBuffer(cmd, f.staging.buffer, modelFaces.buffer, uint32_t(modelCopies.size()),
                    modelCopies.data());
  if (!arenaCopies.empty())
    vkCmdCopyBuffer(cmd, f.staging.buffer, faceArena.buffer, uint32_t(arenaCopies.size()),
                    arenaCopies.data());
  if (!materialCopies.empty())
    vkCmdCopyBuffer(cmd, f.staging.buffer, materials.buffer, uint32_t(materialCopies.size()),
                    materialCopies.data());
  if (!metaCopies.empty())
    vkCmdCopyBuffer(cmd, f.staging.buffer, chunkMeta.buffer, uint32_t(metaCopies.size()),
                    metaCopies.data());
}

} // namespace atm::render
