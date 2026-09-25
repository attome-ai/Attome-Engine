#pragma once

// Private implementation of atm::render::Renderer (split over Renderer.cpp,
// RendererGeometry.cpp and RendererFrame.cpp). Never included by game code.

#include "Renderer.h"

#include "Culling.h"
#include "FaceAllocator.h"
#include "GpuLayout.h"
#include "vk/ImGuiLayer.h"
#include "vk/Resources.h"
#include "vk/Swapchain.h"
#include "vk/VkContext.h"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace atm::render {

inline constexpr uint32_t kMaxFramesInFlight = 3;
inline constexpr uint32_t kMaxChunkSlots = 32768;
inline constexpr uint32_t kMaxDraws = kMaxChunkSlots * 6;       // opaque indirect draws
// Worst case faces in one draw: translucent group of a checkerboard chunk,
// 6 directions x 33 planes x 32 x 32. Model parts are <= 32^3 so also fit.
inline constexpr uint32_t kMaxFacesPerDraw = 6u * 33u * 32u * 32u;
inline constexpr uint32_t kMaxModelInstances = 16384;
inline constexpr uint32_t kModelFaceBytes = 16u << 20;          // 2M model faces
inline constexpr uint32_t kMetaStagingBytes = 512u << 10;       // metadata + materials / frame
inline constexpr uint32_t kMaxBloomMips = 6;

// Deferred release of a face range (after the frame's fence signals).
struct DeferredFree {
  FaceAllocator *allocator;
  uint32_t offset, units;
};

struct FrameData {
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkSemaphore imageAvailable = VK_NULL_HANDLE;
  VkQueryPool queries = VK_NULL_HANDLE;
  bool queriesWritten = false;
  vk::Buffer ubo;               // FrameUniforms
  vk::Buffer visible;           // uint[kMaxChunkSlots]
  vk::Buffer draws;             // DrawIndexedIndirect[kMaxDraws] (compute output)
  vk::Buffer drawCount;         // uint
  vk::Buffer translucentDraws;  // DrawIndexedIndirect[kMaxChunkSlots] (CPU written)
  vk::Buffer instances;         // ModelInstanceGpu[kMaxModelInstances]
  vk::Buffer staging;           // upload ring slot
  VkDescriptorSet sceneSet = VK_NULL_HANDLE;
  std::vector<DeferredFree> deferred;
  bool screenshotPending = false;
};

struct ChunkMeshState {
  uint32_t base = FaceAllocator::kInvalid, units = 0;
  std::array<uint32_t, 7> dirOffset{};  // opaque groups; [6] = opaque total
  uint32_t translucentCount = 0;
  uint8_t aabbMin[3]{0, 0, 0}, aabbMax[3]{0, 0, 0};
  uint64_t connectivity = ~0ull;
};

struct ChunkRecord {
  bool used = false;
  voxel::ChunkCoord coord;
  uint32_t cullIndex = 0;
  ChunkMeshState live;                  // what is drawn
  bool pending = false;                 // a newer mesh is uploading
  bool queued = false;                  // in the upload queue
  ChunkMeshState next;
  std::vector<voxel::PackedFace> nextFaces; // opaque then translucent
  uint32_t uploadedUnits = 0;
  bool metaDirty = false;
};

struct ModelRecord {
  bool used = false;
  bool ready = false;                   // fully uploaded
  bool queued = false;
  uint32_t base = FaceAllocator::kInvalid, units = 0, uploadedUnits = 0;
  std::vector<voxel::PackedFace> faces; // cleared after upload
};

struct PendingInstance {
  ModelMeshId mesh;
  uint32_t order;
  ModelInstance instance;
};

struct ModelDraw {
  uint32_t faceBase, faceCount, firstInstance, instanceCount;
};

struct Renderer::Impl {
  // --- config / window ---------------------------------------------------------
  SDL_Window *window = nullptr;
  RendererConfig config;
  bool initialised = false;
  uint32_t framesInFlight = 2;

  // --- Vulkan core --------------------------------------------------------------
  vk::VkContext ctx;
  vk::Swapchain swapchain;
  bool swapchainDirty = false;
  VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;
  VkPipelineCache pipelineCache = VK_NULL_HANDLE;
  std::array<FrameData, kMaxFramesInFlight> frames{};
  uint32_t frameIndex = 0;
  uint32_t imageIndex = 0;
  bool frameActive = false;
  vk::ImGuiLayer imgui;

  // --- size-dependent targets ---------------------------------------------------
  VkExtent2D extent{0, 0};
  vk::Image hdrMsaa;            // only when msaa > 1
  vk::Image hdr;                // resolved HDR colour (sampled)
  vk::Image depth;
  vk::Image bloom;              // mip chain, GENERAL layout
  std::array<VkImageView, kMaxBloomMips> bloomMipViews{};
  uint32_t bloomMips = 0;
  VkImageAspectFlags depthAspect = VK_IMAGE_ASPECT_DEPTH_BIT;

  // --- descriptors / pipelines --------------------------------------------------
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout bloomSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout tonemapSetLayout = VK_NULL_HANDLE;
  std::array<VkDescriptorSet, 2 * kMaxBloomMips> bloomSets{};
  VkDescriptorSet tonemapSet = VK_NULL_HANDLE;
  VkPipelineLayout scenePipelineLayout = VK_NULL_HANDLE;
  VkPipelineLayout bloomPipelineLayout = VK_NULL_HANDLE;
  VkPipelineLayout tonemapPipelineLayout = VK_NULL_HANDLE;
  VkPipeline opaquePipeline = VK_NULL_HANDLE;
  VkPipeline translucentPipeline = VK_NULL_HANDLE;
  VkPipeline modelPipeline = VK_NULL_HANDLE;
  VkPipeline skyPipeline = VK_NULL_HANDLE;
  VkPipeline highlightPipeline = VK_NULL_HANDLE;
  VkPipeline cullPipeline = VK_NULL_HANDLE;
  VkPipeline bloomPipeline = VK_NULL_HANDLE;
  VkPipeline tonemapPipeline = VK_NULL_HANDLE;
  VkSampler linearSampler = VK_NULL_HANDLE;

  // --- persistent GPU buffers ---------------------------------------------------
  vk::Buffer indexBuffer;       // shared quad indices
  vk::Buffer faceArena;         // chunk faces
  vk::Buffer chunkMeta;         // ChunkGpu[kMaxChunkSlots]
  vk::Buffer materials;         // MaterialGpu[kMaxMaterials]
  vk::Buffer modelFaces;        // model part faces
  FaceAllocator arenaAlloc;
  FaceAllocator modelAlloc;

  // --- world state ---------------------------------------------------------------
  std::vector<ChunkRecord> chunks;              // indexed by slot
  std::vector<uint32_t> freeSlots;
  std::unordered_map<uint64_t, uint32_t> slotOf; // ChunkCoord::key() -> slot
  std::vector<CullChunk> cullChunks;
  std::vector<uint32_t> uploadQueue;            // chunk slots, FIFO
  size_t uploadHead = 0;
  std::vector<uint32_t> dirtySlots;
  std::vector<gpu::MaterialGpu> materialsCpu;
  uint32_t materialsDirtyBegin = 0, materialsDirtyEnd = 0;
  uint32_t modelMaterialCursor = kModelMaterialBase;
  std::vector<ModelRecord> models;
  std::vector<ModelMeshId> freeModelIds;
  std::vector<ModelMeshId> modelUploadQueue;
  bool arenaFullLogged = false;

  // --- per-frame CPU scratch (reused, no steady-state allocations) -------------
  ChunkCuller culler;
  std::vector<uint32_t> visibleSlots;
  std::vector<std::pair<float, uint32_t>> translucentOrder;
  std::vector<PendingInstance> instances;
  std::vector<ModelDraw> modelDraws;
  std::vector<VkBufferCopy> arenaCopies, modelCopies, metaCopies, materialCopies;
  bool highlightValid = false;
  voxel::BlockPos highlightBlock;
  Camera camera;
  Environment env;
  glm::mat4 viewProj{1.0f};
  glm::ivec3 camBlock{0};
  glm::vec3 camFrac{0.0f};
  uint64_t frameStartTicks = 0;
  uint64_t frameCounter = 0;
  FrameStats stats;
  uint64_t uploadBytesThisFrame = 0;
  uint32_t visibleCount = 0;
  uint32_t translucentCount = 0;
  uint32_t opaqueDrawEstimate = 0;
  uint64_t facesEstimate = 0;
  glm::vec3 fogColorLinear{0.0f};

  // --- screenshot ------------------------------------------------------------------
  std::string screenshotRequest;     // requested, not yet recorded
  std::string screenshotPath;        // recorded into a frame, waiting for readback
  vk::Buffer screenshotBuffer;
  VkExtent2D screenshotExtent{0, 0};
  VkFormat screenshotFormat = VK_FORMAT_UNDEFINED;

  // Renderer.cpp
  bool init(SDL_Window *window, const RendererConfig &config, std::string *error);
  void shutdown();
  bool createFrames();
  bool createDescriptors();
  bool createPipelines();
  bool createTonemapPipeline();
  bool createStaticBuffers();
  bool createTargets(uint32_t width, uint32_t height);
  void destroyTargets();
  void writeTargetDescriptors();
  bool recreateSwapchain();
  bool immediateSubmit(void (*record)(Impl &, VkCommandBuffer, void *), void *user);
  void saveScreenshotIfReady();

  // RendererGeometry.cpp
  void setMaterials(std::span<const Material> m, uint32_t base);
  void setChunkMesh(voxel::ChunkCoord coord, const voxel::ChunkMeshData &mesh);
  void removeChunk(voxel::ChunkCoord coord);
  ModelMeshId createModelMesh(const voxel::ChunkMeshData &mesh);
  void destroyModelMesh(ModelMeshId id);
  void markMetaDirty(uint32_t slot);
  void deferFree(FaceAllocator &alloc, uint32_t offset, uint32_t units);
  void processDeferred(FrameData &f);
  // Records all copies of this frame into `cmd` (staging -> GPU buffers).
  void recordUploads(FrameData &f, VkCommandBuffer cmd);
  gpu::ChunkGpu chunkGpu(uint32_t slot) const;

  // RendererFrame.cpp
  bool beginFrame(const Camera &camera, const Environment &env);
  void endFrame();
  void buildFrameUniforms(FrameData &f);
  void cullAndBuildLists(FrameData &f);
  uint32_t writeModelInstances(FrameData &f);
  void recordFrame(FrameData &f, uint32_t translucentCount, uint32_t instanceCount);
  void recordBloom(VkCommandBuffer cmd);
};

} // namespace atm::render
