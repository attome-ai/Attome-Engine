# AttomeRender — Vulkan 1.3 voxel renderer

Public API: `Renderer.h` only. Everything Vulkan lives in this directory
(`vk/` = thin ownership wrappers). Design background: `docs/GPU_3D_PLAN.md`
§4, §5, §9, §9A.

## Architecture

| File | Role |
|---|---|
| `vk/VkContext` | Vulkan loader (Vulkan::Vulkan; no volk, see VkCommon.h), vk-bootstrap instance/device (Vulkan 1.3: dynamic rendering, sync2; 1.2 drawIndirectCount; multiDrawIndirect, drawIndirectFirstInstance), SDL3 surface, VMA, capability fallback (depth D32→D24S8→D16, MSAA halved until colour+depth support it, timestamps) |
| `vk/Swapchain` | vk-bootstrap swapchain (FIFO for vsync, else MAILBOX→IMMEDIATE→FIFO), one render-finished semaphore per swapchain image |
| `vk/Resources` | VMA buffers/images, sync2 barriers, pipeline creation (dynamic rendering) |
| `vk/ShaderLibrary` | SPIR-V embedded at build time (`glslangValidator` + `EmbedSpirv.cmake`) |
| `vk/ImGuiLayer` | `ImGui_ImplVulkan` with dynamic rendering |
| `FaceAllocator` | best-fit range allocator (8-byte units) with coalescing, for the face arenas |
| `Culling` | CPU cave-culling BFS + frustum test |
| `GpuLayout.h` | GPU structs (std140/std430) with static_asserts; mirrors `shaders/voxel_common.glsl` |
| `Renderer*.cpp` | init/resources, geometry + uploads, per-frame work |

## Frame

1. `beginFrame`: wait the frame slot's fence, read its timestamps, save a
   pending screenshot, release deferred face ranges, acquire the image.
2. `endFrame`:
   - CPU: cave-culling BFS + frustum → visible chunk slots (front to back);
     translucent chunks sorted back to front → CPU-written indirect draws;
     model instances sorted by mesh → instance buffer (camera-relative
     matrices built in double).
   - GPU: staged copies (model faces, chunk faces within `uploadBudgetBytes`,
     materials, chunk metadata) → `cull.comp` expands visible chunks × 6
     directions into `VkDrawIndexedIndirectCommand`s (drops directions whose
     faces all point away from the camera) → HDR MSAA pass (reverse-Z,
     infinite far): opaque `vkCmdDrawIndexedIndirectCount`, models
     (`vkCmdDrawIndexed`, instanced per mesh), sky (fullscreen, depth EQUAL 0,
     only uncovered pixels), translucent (blend, no depth write), block
     outline → resolve → bloom (compute mip chain, 13-tap down / tent up) →
     tonemap (ACES fit + gamma) into the swapchain → ImGui → present.

## Buffers

| Buffer | Content |
|---|---|
| Face arena (device, `chunkArenaBytes`, ≤ 2 GiB) | `PackedFace` (uint64) of all chunks: opaque groups by direction, then translucent |
| Chunk metadata (device, 32768 × 64 B) | origin, face base, per-direction offsets, translucent count, AABB, flags |
| Materials (device, 65536 × 32 B) | block ids `[0, 16384)`, model palettes appended by `setModelMaterials` |
| Model faces (device, 16 MiB) | model part faces, same format |
| Index buffer (device) | shared quad pattern `4j + (0,1,2, 0,2,3)` for 202752 faces |
| Per frame (host-visible) | uniforms, visible list, translucent draws, instances, staging ring slot; device: opaque draws + count |

Vertex pulling: `vertexOffset = firstFace * 4`, so `gl_VertexIndex >> 2` is
the absolute face index and `& 3` the corner; `firstInstance` carries the
chunk slot (or first model instance). The arena is capped at 2 GiB so
`firstFace * 4` fits in int32.

## Culling

- **Cave culling** (Checchi): BFS over a dense grid around the camera chunk;
  a chunk entered through face *e* may be left through *g* only if the
  mesher's `faceConnectivity` connects *e* and *g*, and never in the
  direction opposite to one already taken. Missing chunks (air, not loaded)
  are open. Chunks with an empty mesh but restricted connectivity stay
  resident (not drawn) so they still block the search.
- **Frustum + view distance** on every visited chunk; the search does not
  expand out of chunks outside the frustum.
- **Direction culling** on the GPU (`cull.comp`), conservative against the
  chunk's content AABB.

## Conventions

- Colours are memory bytes R,G,B,A (uint32 `0xAABBGGRR`, `atm::voxel::packRGBA`, GLSL `unpackUnorm4x8`), authored in sRGB, converted to linear in shaders.
  Environment colours are sRGB too.
- `PackedFace` x/y/z is the quad's (u0,v0) corner in plane coordinates
  (0..32; +X faces of block x sit on plane x+1). AO corner k is bits 2k..2k+1.
- Camera: `dir = (-sin(yaw)cos(pitch), sin(pitch), -cos(yaw)cos(pitch))`.
- The application creates the ImGui context before `Renderer::init` and runs
  `ImGui_ImplSDL3_*` + `ImGui::NewFrame()` between `beginFrame` and `endFrame`.
- Screenshots are written as **BMP** (`SDL_SaveBMP`) one frame later.

## Known limitations / TODO(demo)

- No transfer-queue uploads (graphics queue only); no pipeline-cache file on disk.
- No arena defragmentation (`FrameStats::chunkArenaFragmentation` reports it).
- Translucent faces are sorted per chunk only, not within a chunk.
- Model parts get full sky light (no world light sampling yet).
- Hi-Z occlusion culling and LOD (plan §9A, M7) are not implemented.
- ImGui pipeline is not rebuilt if the swapchain format changes on resize.
