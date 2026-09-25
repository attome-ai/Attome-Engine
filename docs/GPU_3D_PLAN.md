# Attome Engine: Vulkan renderer + voxel 3D — design plan

Status: proposal for review (revision 4). Nothing here is implemented yet.

- Revision 3: **raw Vulkan** instead of SDL3 GPU, **web dropped**, **Steam**
  is the ship target.
- Revision 4: 3D sections rewritten for a **Trove-style voxel world** (blocks,
  building, action movement) instead of OSRS-style terrain. The game itself is
  described in `docs/GAME_DESIGN.md`.

## 1. Goals

1. Replace the SDL_Renderer drawing path with a **Vulkan renderer**, with no
   performance regression on existing workloads and a measurable gain on
   large scenes.
2. Ship on **Steam**: Windows first, Steam Deck/Linux native, macOS optional
   through MoltenVK.
3. Add 3D rendering for a **Trove-style voxel MMO**: an editable block world
   with large draw distances, voxel characters and props, bloom/glow effects,
   particles, and a third-person action camera.

Non-goals for this plan: PBR materials, shadows, texture-heavy art, animation
blend trees, VR. Items that may be needed later are listed in §15 with the
measurement that would trigger them.

## 2. Decisions

| Decision | Choice | Why |
|---|---|---|
| GPU API | **Raw Vulkan 1.3** (1.2 + `dynamic_rendering`, `synchronization2`, `descriptor_indexing` accepted) | In-engine GPU timing (timestamp queries), bindless textures, multi-threaded command recording, async compute, and a path to mesh shaders — all useful for a large voxel MMO and unavailable in SDL3 GPU. |
| Vulkan helpers | **vk-bootstrap** (instance/device/swapchain setup), **VMA** (memory), standard Vulkan loader | Removes most of the boilerplate while keeping full control. volk was dropped: ImGui's Vulkan backend (vcpkg) calls Vulkan functions directly, which conflicts with volk's same-named function pointers. |
| Platform layer | **SDL3 stays** for window, input, audio, events, and `SDL_Vulkan_CreateSurface` | Only drawing moves to Vulkan. |
| Platforms | Windows (primary), Linux / Steam Deck (native), macOS via MoltenVK (optional), Android later | Matches Steam's audience. |
| Web | **Dropped** | Browsers have no Vulkan, and a voxel MMO isn't a browser target. The SDL_Renderer path is removed once all games run on Vulkan (M5). TowerSwarm and Meteor Dodge lose their web builds. |
| Abstraction depth | **No RHI layer**; all Vulkan code is confined to one module (§4.3) | Game and world code never see Vulkan types. |
| Order | Measure → 2D port → 3D | The 2D port carries the performance risk and proves the renderer before 3D is built on it. |

## 3. Where we are today

- `engine_render_scene()` builds every visible sprite as 4 `SDL_Vertex` + 6
  indices on the CPU (104 bytes per sprite, every frame) and submits one
  `SDL_RenderGeometry` per (texture, z) batch.
- Static chunks are cached, but their vertices are **copied and
  camera-transformed on the CPU every frame** (`append_transformed_batch`).
- `TextureAtlas` is not an atlas: every registered image is its own GPU
  texture, so N textures means at least N draw calls.
- Games draw their HUDs directly with SDL_Renderer:

  | Game | Direct SDL_Render* calls |
  |---|---|
  | tower_swarm | ~270 (114 debug text, 67 fill rect, 53 rect, 8 line) |
  | snake | ~20 |
  | hello-world, meteor_dodge, _template | a handful each |
  | planet_optimized | ImGui via `imgui_impl_sdlrenderer3` |

  These calls must move to an engine 2D API, otherwise they won't draw on the
  GPU backend.

## 4. Architecture

### 4.1 Layers

```
                         Game code
                             │
          ┌──────────────────┼───────────────────┐
          │                  │                   │
     Simulation            World              UI / HUD
  (entities, rules,   (blocks, chunks,       (atm::draw2d)
   tick, pathing)      streaming, scenery)
          │                  │                   │
          └────────► World state (read-only for rendering)
                             │
                         Renderer
            ┌────────────────┼──────────────────┐
       Renderer3D       Renderer2D          UIRenderer
            └────────────────┼──────────────────┘
                     Pass list (frame)
                             │
                     GPU module (the only code that includes Vulkan headers)
                             │
                     Vulkan (MoltenVK on macOS)
```

Rules:

- **Simulation never depends on the renderer or on networking.** It reads
  input/commands and produces world state. A local game and a networked game
  feed the same simulation; the renderer only reads the resulting state. This
  keeps simulation testable headless.
- During the migration (M2–M5) the existing SDL_Renderer code is kept behind
  the Renderer2D/UIRenderer interfaces so games keep running; it is deleted
  at M5.
- Game code keeps using the `engine_*` entity API and `atm::draw2d`; it never
  sees GPU handles.

### 4.2 Frame and passes

The frame is an ordered **list of pass objects**, each declaring the targets it
reads and writes. The first version runs them in list order:

1. **Upload** — one copy pass: sprite instances, draw2d geometry, 3D instance
   transforms, uniforms, and this frame's share of the chunk upload queue
   (§10.2), written into this frame's slot of the upload ring (§4.5).
2. **3D opaque** — terrain, static scenery, dynamic models (colour + depth,
   4× MSAA).
3. **3D transparent** — back-to-front, depth test on, depth write off.
4. **MSAA resolve.**
5. **2D world** — sprites (static chunks + dynamic).
6. **UI** — draw2d shapes, text, ImGui.
7. Submit + present.

Because passes are data, render-to-texture features (minimap, water,
post-processing, outlines) are added as new pass objects. Ordering by
declared dependencies (a small render graph) is introduced only once there is
a pass whose order isn't fixed (§15).

### 4.3 The GPU module

One module, `engine/render/vk/`, owns every Vulkan object:

| Engine type | Wraps | Notes |
|---|---|---|
| `VkContext` | instance, physical/logical device, queues (graphics, transfer, async compute when present) | Created with vk-bootstrap; validation layers on in debug builds |
| `Swapchain` | swapchain, images, present mode | FIFO (vsync) or MAILBOX/IMMEDIATE (uncapped) from config; recreated on resize / surface loss |
| `FrameContext` ×N | command pools, per-frame fences and semaphores, timestamp query pool | N = frames in flight (§4.5) |
| `GpuBuffer` | `VkBuffer` + VMA allocation | Vertex / index / storage / indirect usage |
| `UploadRing` | host-visible VMA buffer split into N slots | §4.5 |
| `GpuTexture`, `Sampler` | image, view, VMA allocation | Atlas pages, depth, MSAA targets |
| `BindlessTable` | one large descriptor set of sampled images | Textures referenced by index from any shader |
| `ShaderLibrary` | `VkShaderModule` | SPIR-V embedded in the engine (§7) |
| `PipelineCache` | `VkPipeline` + `VkPipelineCache` saved to disk | Keyed by (shaders, vertex layout, blend, depth, MSAA); disk cache avoids hitches on second launch |

These are thin ownership wrappers, not a second abstraction. Rendering uses
**dynamic rendering** (no render-pass/framebuffer objects) and
**synchronization2** barriers.

**Surface loss** (minimise, resize, Alt+Tab from fullscreen, Deck
suspend/resume): `VK_ERROR_OUT_OF_DATE_KHR` / `SUBOPTIMAL` triggers swapchain
recreation plus the size-dependent targets (depth, MSAA, post-process).
Buffers and textures survive.

### 4.4 Render extraction

The renderer never walks gameplay data directly during drawing:

```
World / containers ──extract──▶ RenderWorld snapshot ──▶ Renderer
```

- Extraction runs once per rendered frame on the main thread and copies what
  the renderer needs (transforms, model/anim ids, lights, particles) into a
  flat `RenderWorld`.
- With a fixed simulation tick, the snapshot keeps **previous and current**
  state so the renderer interpolates with `fixed_alpha`.
- For 2D sprites, filling the instance buffer **is** the extraction — no
  intermediate copy on the hot path.
- This boundary is what later allows rendering on its own thread.

### 4.5 Frames in flight, uploads and resource lifetime

- **Frames in flight: 2** (configurable 2–3). Each `FrameContext` has a fence;
  before reusing a frame slot, the CPU waits on that slot's fence (normally
  already signalled).
- **Upload ring**: one persistently mapped host-visible buffer with N slots.
  Frame *k* writes only slot *k mod N*, which is reused only after frame
  *k − N*'s fence has signalled. Oversized uploads (chunk meshes) go through
  the budgeted `GpuUploadQueue` (§10.2), never block the frame, and fall back
  to the next frame when the budget is used up.
- **Deferred destruction**: releasing a GPU resource puts it in the current
  frame's deletion list; the list is freed when that frame's fence signals.
  So a chunk evicted while a frame is still in flight is destroyed safely
  N frames later. Rule for systems: remove the resource from the visible set
  first, then release it.
- The **transfer queue** is used for large chunk uploads when the device has
  a dedicated one, with queue ownership transfer barriers; otherwise uploads
  run on the graphics queue.

## 5. Coordinate contract

All spaces, one conversion function each, defined in `engine/world/Coords.h`:

| Space | Type | Definition |
|---|---|---|
| Block | `int32 x, y, z` | One voxel, ≈ 1 m. `y` is height, 0..511. Unlimited X/Z range. |
| Chunk | `int32 cx, cy, cz` | 32×32×32 blocks: `cx = floor(x / 32)`. The meshing, networking and storage unit. |
| Local block | `uint8 lx, ly, lz` | `lx = x - cx * 32` |
| Sector | `int32 sx, sz` | 256×256 blocks, full height (8×8 chunks wide, 16 high). The ownership, auction and zone-assignment unit. |
| World (sim) | `int64` fixed-point, 1 unit = 1/1024 block | `world_x = x * 1024 + sub_x`. Exact, no drift, safe for networking and prediction. |
| Render | `float32`, **camera-relative** | `render_x = float(world_x - camera_world_x) / 1024` computed on the CPU. |
| View / clip | `float32` | Camera matrix applied on the GPU. |
| Screen | pixels, origin top-left | For picking and draw2d. |

- Right-handed, **Y up**, ground is the XZ plane. The existing spatial grid
  indexes X and Z (entities), so culling, hybrid updates, interest
  management and proximity queries keep working.
- **Why camera-relative:** a world that grows every day will reach
  coordinates where 32-bit floats can't resolve small movements, which shows
  up as jittering models far from the origin. Simulation positions stay exact
  integers; only the renderer converts to floats, relative to the camera.
  Chunk meshes are stored relative to the chunk origin, and the chunk's offset
  from the camera goes in a per-draw value.

## 6. Phase 1 — measure first

Performance claims need a baseline taken on the real renderer, not the
software rasteriser `atm_bench` uses today.

`atm_frame_bench` opens a real window with the default SDL_Renderer (D3D11 on
Windows), vsync off, and runs fixed scenes for N frames:

- 20k / 100k dynamic sprites, 1 texture and 16 textures
- 200k static sprites (chunk cache path), camera panning
- HUD-heavy scene (tower_swarm-like: 300 rects + 100 text lines)

Metrics per scene:

| Metric | Why |
|---|---|
| CPU frame time: mean, p50, p99 | Headline cost |
| **1% low and 0.1% low FPS** | Hitches, which averages hide |
| Time blocked acquiring the swapchain / presenting | Separates GPU-bound or vsync-bound from CPU-bound |
| Draw calls, pipeline switches | Batching quality |
| Bytes uploaded per frame (buffers, textures) | Upload-bound detection |
| Visible sprites / triangles | Normalises the numbers |
| RAM and VRAM estimate | Resource growth |

| **GPU time per pass** (Vulkan timestamp queries) | Measured in-engine for the Vulkan renderer; shows which pass costs what |

For the SDL_Renderer baseline, GPU time isn't available in-engine and is taken
with an external profiler (Nsight, Radeon GPU Profiler, RenderDoc) instead.

Results go in `docs/perf/<milestone>-<machine>.txt`. Two kinds of gate:

- **Regression gate** — existing workloads (the 2D scenes above) must never
  get slower than the recorded baseline (mean and 1% low), same machine,
  same build type.
- **Feature targets** — new workloads (3D terrain, streaming, animation,
  crowds) get their own frame-time and 1%-low targets when introduced; once
  met, those numbers become the regression baseline for later milestones.

Reference machines: the development PC, and a **Steam Deck** (1280×800,
target 60 fps) as the minimum-spec device.

## 7. Shaders

- **Source language: GLSL** (or HLSL via DXC — both produce SPIR-V; pick one
  and use it everywhere). One file per shader in `engine/shaders/`.
- **Build step:** `glslangValidator` (or DXC) compiles to **SPIR-V** at build
  time; the bytes are embedded in the engine library. Vulkan and MoltenVK
  both consume SPIR-V, so no other formats are needed.
- **Validation:** `spirv-val` in debug builds; the Vulkan validation layers
  in development runs.
- **Bindings:** a fixed convention (set 0 = per-frame, set 1 = bindless
  textures, set 2 = per-pass, push constants = per-draw), stated in a shared
  header included by every shader.

## 8. Phase 2 — 2D on the GPU

### 8.1 GPU data layout (ABI)

GPU-visible structs live in `engine/render/gpu/GpuLayout.h`, shared as the
source of truth with the shader headers:

- Storage buffers use **std430** rules; uniform buffers use **std140**.
- Only 4-byte scalars and explicitly padded vectors; no `bool`, no implicit
  padding, no `vec3` in storage buffers.
- Each struct has `static_assert(sizeof(...) == N)` and
  `static_assert(offsetof(...) == ...)` per field, and a matching comment in
  the HLSL.
- Little-endian (all supported targets).
- A `kGpuLayoutVersion` constant is bumped on any change and checked against
  the value compiled into the shaders at startup.

### 8.2 Sprites: instancing instead of CPU vertices

```
struct SpriteInstance {        // 32 bytes, std430, version 1
  float    x, y;               //  0: world position
  float    w, h;               //  8: size
  uint32_t uv_rect;            // 16: index into the atlas region table
  float    rotation;           // 20: radians
  uint32_t color;              // 24: RGBA8 tint
  uint32_t layer_flags;        // 28: atlas page (bits 0-7), flags
};
```

The vertex shader builds the quad from the vertex index (6 vertices per
instance, no index buffer), reads the instance from a storage buffer, and
applies the camera from a uniform.

- 3.25× less data per sprite than today (32 B vs 104 B), no CPU transform.
- Rotation costs the CPU nothing, so the `ROTATABLE` opt-in stops mattering on
  this backend.

### 8.3 Static chunks live on the GPU

Each static chunk keeps its instances in a GPU buffer, rebuilt only when the
chunk is dirty. Drawing a visible chunk is one draw call with the camera in a
uniform — no per-frame CPU copying. This is the biggest expected win for large
maps.

### 8.4 A real texture atlas

Pack registered images into a few large pages (e.g. 4096², shelf packer like
`ATMText`), so a scene with many small images uses 1–2 textures and far fewer
draw calls. `engine_register_texture()` keeps its signature; ids map to
(page, rect).

### 8.5 Sorting and batching

Same visible order as today: z-index, then atlas page. Within a (z, page)
group everything is one instanced draw. Transparency keeps painter's order.

### 8.6 `atm::draw2d` — immediate-mode 2D for HUDs

```cpp
atm::draw2d::fillRect(rect, color);
atm::draw2d::rect(rect, color, thickness);
atm::draw2d::line(x0, y0, x1, y1, color);
atm::draw2d::text(font, "Score", x, y, color);
atm::draw2d::debugText(x, y, fmt, ...);   // replaces SDL_RenderDebugTextFormat
```

Calls append to one per-frame vertex buffer, drawn in the UI pass with a single
pipeline (shapes use a white pixel in the atlas, so shapes and text batch
together). During migration it is also implemented on the SDL_Renderer path
so games can be moved over before the Vulkan renderer exists.

Migration: a mechanical rename in tower_swarm (~270 call sites), snake,
hello-world, meteor_dodge and the template, with screenshot checks. Blend
mode calls become a `draw2d` state or disappear (alpha blending is the
default).

ImGui: switch planet_optimized to `imgui_impl_vulkan` (already installed with
the vcpkg imgui port).

### 8.7 Acceptance

- Every `atm_frame_bench` scene ≥ baseline (mean and 1% low). Expected: large
  gains on static maps and 100k sprites; parity on small scenes.
- All existing games render identically (screenshot comparison at fixed
  camera positions; the bench can dump frames).
- SDL_Renderer path and the web build scripts deleted.

## 9. Phase 3 — 3D core

### 9.1 Camera

- **Third-person action camera** (Trove-style): behind and above the player,
  mouse controls yaw and pitch, scroll zooms; the aim point is a ray from the
  screen centre.
- Collision: the camera pulls in when a block is between it and the player.
- Optional lock-on for controller play.
- Perspective projection, reverse-Z depth for precision at long view
  distances.
- Camera-relative rendering (§5): the view matrix has no translation.
- Culling: chunks are tested against the frustum by AABB (a chunk list
  per sector, so the test is hierarchical: sector → chunk).

### 9.2 Pipelines

| Pipeline | Vertex format | Notes |
|---|---|---|
| Voxel chunk (opaque) | **8 bytes/vertex** packed: position in chunk (3×6 bits), face normal (3 bits), AO (2 bits), block light + sky light (2×4 bits), block type (16 bits) | Colours/textures looked up by block type in a bindless table |
| Voxel chunk (transparent) | same | Water, glass, leaves; sorted per chunk back-to-front |
| Voxel model (characters, props) | position, normal, colour | Meshed from `.vox` at build time; rigid parts |
| Model instances | per-instance transform (camera-relative) | Trees, props, items on the ground: instanced |
| Particles | GPU buffer, updated by compute | Spell effects, hits, block breaking |
| Post-process | full-screen | Bloom (emissive blocks and effects), fog, colour grading, FXAA/MSAA resolve |

- **Lighting**: sunlight + block light levels baked into vertices (flood
  fill on workers), per-vertex ambient occlusion, plus a limited number of
  dynamic point lights (spells, torches held by players) in a light list.
- **Glow**: emissive block types and effects write to a bright pass for
  bloom — the main source of Trove's look.
- **Fog** to the sky colour; far plane at the draw distance.
- **4× MSAA** on the 3D pass by default, configurable (1/2/4/8).

### 9.3 Device capability fallback

Checked once at startup, logged, and exposed in `engine->gpu_caps`:

| Setting | Preference | Fallback |
|---|---|---|
| Depth format | `D32_SFLOAT` | `D24_UNORM_S8_UINT` → `D16_UNORM` |
| MSAA samples | configured value | halve until supported for colour **and** depth: 8 → 4 → 2 → 1 |
| Present mode | config (`vsync` → FIFO, uncapped → MAILBOX → IMMEDIATE) | FIFO (always available) |
| Bindless | `descriptor_indexing` with enough sampled-image descriptors | Fixed-size texture array (e.g. 16 atlas pages) |
| Async compute | separate compute queue | graphics queue |
| Timestamp queries | `timestampPeriod > 0` on graphics queue | metrics show "n/a" |

A device below Vulkan 1.2 or without dynamic rendering fails with a clear
message box (for Steam, listed as the minimum requirement).

## 9A. Voxel rendering techniques (research, 2026)

Survey of the fastest open-source voxel renderers and meshers, and what we
take from each. Sources are linked; figures are as published.

| Technique | Best-known user | Numbers | Our design |
|---|---|---|---|
| Compact vertex format | Sodium: 20 B/vertex = **80 B per quad** ([source](https://github.com/CaffeineMC/sodium)) | — | **8 B per quad** (`PackedFace`, incl. AO + light) — 10× less |
| Vertex pulling from a storage buffer | Vercidium, vkguide "Ascendant" ([voxel.wiki](https://voxel.wiki/wiki/vertex-pulling/)) | Same speed as vertex buffers, far less memory | Faces read by `gl_VertexIndex`, one shared quad index buffer |
| Binary greedy meshing | cgerikj/binary-greedy-meshing ([repo](https://github.com/cgerikj/binary-greedy-meshing)) | **74 µs/chunk** (62³, Ryzen 3800X), 8 B/quad without AO/light | Same algorithm on 32³ (+border) with 64-bit column masks; merges only equal AO/light |
| Per-direction face buckets | Sodium, Nick McDonald ([blog](https://nickmcd.me/2021/04/04/high-performance-voxel-engine/)) | ~2× vs one mesh per chunk | Faces grouped by direction; back-facing groups never drawn |
| Arena + multi-draw | Sodium regions | Few draws for many chunks | One GPU face arena; **compute cull → `vkCmdDrawIndexedIndirectCount`** |
| Cave / visibility-graph culling | Minecraft (Checchi) ([part 1](https://tomcc.github.io/2014/08/31/visibility-1.html)) | Culls 50–99% underground; 0.1–0.2 ms per edit | Face-connectivity bits per chunk (built by the mesher), BFS on the CPU as a pre-pass |
| Two-phase Hi-Z occlusion | niagara (zeux) ([shader](https://github.com/zeux/niagara/blob/master/src/shaders/drawcull.comp.glsl)) | No one-frame popping | M7+: early pass (last frame's visible) → depth pyramid → late pass |
| Hierarchical LOD | Voxy (32³ sections, 5 levels, GPU traversal) | Very long view distance | M7: 2×/4×/8× downsampled chunks meshed by the same mesher; edits propagate to LOD lazily with a budget |
| Palette + uniform sub-chunks | Minecraft, Veloren | Veloren network chunks: LZ4 ≈ 25% of raw | `Chunk`: 0-bit uniform chunks, 1–16-bit palette indices, RLE on the wire |
| Flood-fill light (add/remove queues) | Minecraft, 0fps ([article](https://0fps.net/2018/02/21/voxel-lighting/)) | — | 1 byte per voxel (sky + block), budgeted per frame |
| Vertex AO + diagonal flip | 0fps ([article](https://0fps.net/2013/07/03/ambient-occlusion-for-minecraft-like-worlds/)) | — | 2 bits per corner in `PackedFace`; flip when a00+a11 > a01+a10 |
| Mesh shaders | Nvidium (NVIDIA-only GL) | No published measurements | Optional backend later; MDI-count path is primary (works everywhere incl. Steam Deck) |
| Ray-marched SVO/DAG | ESVO, HashDAG, Aokana (2025) | Great far-field memory; complex edits | Not for primary visibility; possible far-field layer later |

**Benchmark to claim "fastest"** (none of these projects publishes one):

- Scenes (fixed seeds): flat plains, dense forest, cave-heavy mountains,
  ocean (translucency), dense player-built city.
- Fixed camera flythroughs at 16 / 32 / 64 / 128-chunk distances, plus LOD
  to 1–4 km.
- Metrics: frame time p50/p99/p99.9, CPU and GPU ms per pass (timestamps),
  terrain VRAM (bytes per face, MB per km²), faces drawn vs resident, mesh
  time per chunk, **edit-to-photon latency**, streaming chunks/s.
- Hardware: NVIDIA (Turing+), AMD desktop, Intel Arc/iGPU, Steam Deck.
- Run Sodium / Nvidium / Voxy on the same machines with equivalent content
  and distance for the comparison.
- First reproduce the published mesher number (~74 µs per chunk) to validate
  the harness.

## 10. Phase 4 — voxel world

### 10.1 Data

- **Chunks** (32³ blocks) are the unit for meshing, networking and storage.
- **Block storage**: palette-compressed per chunk (a list of block types used
  in the chunk + indices of 1–16 bits), so typical chunks use a few KB.
- **Block types**: data-driven (JSON): hardness, transparency, light
  emission, collision shape, sounds, drops, colour/texture.
- **Base terrain** is procedural (seeded generator per biome); only **edits**
  are stored, as deltas against the generated chunk. Untouched terrain
  costs no storage.
- **Meshing** (worker threads): greedy meshing merges coplanar faces of the
  same block type into larger quads; faces between opaque blocks are culled;
  AO and light are computed per vertex.
- **Level of detail**: far chunks use a downsampled chunk (2×, 4×, 8× block
  size) meshed the same way; LOD chosen by distance with hysteresis.
- **Edits**: a block change re-meshes only the affected chunk (and
  neighbours when it's on a border), on a worker, and swaps the mesh
  atomically.

### 10.2 Streaming subsystem

`WorldStreamer` keeps the chunks within the draw distance resident, without
blocking the frame:

```
Unloaded → Loading → Decoded → Meshed → Uploading → Resident → Evicting → Unloaded
            (I/O)    (worker)  (worker)  (main, budgeted)        (main)
```

| Component | Thread | Job |
|---|---|---|
| `ChunkLocator` | main | Decides which chunks should be resident and at which LOD (radius + hysteresis so chunks at the edge don't thrash); priority by distance and view direction |
| `ChunkSource` | worker | Client: receives chunks from the server (compressed). Offline/tools: generates from the seed + applies stored edits |
| `ChunkDecoder` | worker | Validates version, decompresses the palette data, builds collision data |
| `ChunkMesher` | worker | Greedy meshing, AO, lighting, LOD meshes |
| `GpuUploadQueue` | main | Uploads at most **N bytes per frame** (config, e.g. 4 MB) in the Upload pass |
| `ChunkCache` | main | LRU of decoded chunks kept in RAM beyond the render radius, so walking back is instant |

- A chunk becomes visible **atomically** once all of its buffers are uploaded
  — never half-drawn.
- Collision data is available at the **Decoded** stage, before the mesh
  exists, so movement and physics never wait on the renderer.
- Workers communicate with the main thread through the existing
  `engine/net/SpscQueue.h`-style queues; no locks on the frame path.

### 10.3 Interaction and physics

- **Block picking**: ray march through the voxel grid (DDA) from the camera
  to find the targeted block and face for building/breaking.
- **Character physics**: swept AABB against block collision data, step-up of
  1 block, jump, double jump, glide, swim. Deterministic fixed-step code
  **shared by client and server** (client prediction, server verification).
- **Monsters**: navigation on the block grid (walkable surfaces extracted per
  chunk), A* over a local area, steering for crowds.
- **Movement**: simulated on the fixed tick, rendered with interpolation
  between ticks (uses the fixed timestep and `fixed_alpha`).

### 10.4 Timing

Three independent rates, all configurable:

| Rate | Typical | Notes |
|---|---|---|
| Simulation tick | 20–30 Hz for action movement; a game may choose 600 ms (OSRS) as a rule | Fixed step, deterministic |
| Render | uncapped or vsync (60–240 Hz) | Interpolates between the last two simulation states |
| Network send | 10–30 Hz | Independent of both; snapshots/deltas |

Nothing in the engine assumes a particular tick length.

## 11. Phase 5 — models and animation

- **Authoring**: MagicaVoxel `.vox` files; one file per model, with separate
  parts (head, body, arms, legs, weapon) as named objects.
- **Build step**: the asset compiler meshes each part (greedy meshing, same
  code as chunks) into `.atmmodel`: positions, normals, colours, indices,
  part hierarchy with pivots, and **two bone indices + two weights per
  vertex**. Voxel models use one bone per part (weight 1.0); blended skinning
  later needs no format change.
- **Animation** (`.atmanim`): keyframed rigid part transforms (Trove-style:
  limbs move as whole pieces), sampled and interpolated on the render frame;
  part matrices in a storage buffer.
- **Attachments**: weapons, hats, gliders, mounts attach to named sockets.
- **Modular characters** (see `GAME_DESIGN.md` §10.1): all players share one
  rig and one animation set; equipment pieces replace or attach to rig parts.
  Rendering groups pieces by mesh id and draws each group **instanced** with
  per-instance part transforms and a palette index, so gear swaps are an id
  change and crowds of players cost a handful of draws.
- **Entities**: a `ModelContainer` (SoA like today) with exact world
  position, yaw, `model_id`, `anim_id`, `anim_time`. It registers as Dynamic
  or Hybrid like any container, so monsters out of view stop animating for
  free.

## 12. Asset pipeline

```
assets/                      (source, in git)
  models/*.vox
  animations/*.json
  textures/*.png
  blocks/*.json
  worlds/<name>/generator.json, prefabs/*.vox
        │
   atm_assetc  (asset compiler, runs as a build step)
        │
build/assets/                (generated)
  manifest.json
  models/*.atmmodel
  worlds/<name>/prefabs/*.atmprefab
  textures/*.atmtex
```

- Every compiled file starts with a **magic number, format version and
  source hash**. The runtime rejects mismatched versions with a clear error.
- `manifest.json` maps **stable asset IDs** (hash of the source path) to files
  and lists dependencies (a model's parts, a prefab's block types).
- The compiler only rebuilds assets whose source or dependencies changed.
- Validation at compile time: part/vertex limits, missing sockets, unknown
  block types in prefabs.
- Hot reload in development: the runtime watches the manifest (same mechanism
  as `Tunables::reloadIfChanged`) and reloads changed assets.
- Compression (e.g. LZ4 per file) is added when prefab/model sizes justify it.

## 13. Phase 6 — voxel prototype

A playable offline prototype that proves the engine for the game in
`GAME_DESIGN.md`: a generated voxel world, Trove movement, building and
breaking blocks, one weapon, one monster type. Then the same simulation
driven by AttomeNet with a zone server and several clients. The renderer is
identical in both modes. The full game roadmap continues in
`GAME_DESIGN.md` §32.

## 14. Milestones

| # | Deliverable | Acceptance |
|---|---|---|
| M1 | `atm_frame_bench` + baseline numbers | Numbers recorded for this machine |
| M2 | Renderer2D/UIRenderer interfaces; existing code moved into the SDL_Renderer fallback | Games and bench unchanged |
| M3 | `atm::draw2d` on the fallback + game HUD migration | Screenshots identical |
| M4 | Vulkan module (context, swapchain, frames in flight, upload ring, deferred destruction, timestamps, SPIR-V build step, `GpuLayout.h`); sprite instancing | Sprite scenes ≥ baseline; validation layers clean |
| M5 | Static chunks on GPU, packed atlas + bindless, draw2d/text, ImGui; SDL_Renderer path and web scripts removed | All bench scenes ≥ baseline; all games run on Vulkan |
| M6 | 3D core: coordinates (§5), camera, depth, MSAA, voxel chunk pipeline, post-process (bloom, fog) | Static voxel scene at target frame rate on dev PC and Steam Deck, no jitter far from origin |
| M7 | Voxel world: generator, greedy meshing, AO/lighting, LOD, streaming subsystem, block picking and editing, character physics | Fly across 10+ sectors with no frame over budget; edits re-mesh within one frame budget |
| M8 | Voxel models + rigid animation (`.vox` → `.atmmodel`), particles, asset manifest + hot reload | Animated characters and effects in the world |
| M9 | Voxel prototype: offline, then one zone server + several clients | Clients move, fight and build together smoothly |

M1–M5 is the Vulkan migration; M6–M9 is 3D. Steam work (§14.1) runs in
parallel from M5. Each milestone ends with the
benchmark gate and a working build.

### 14.1 Steam

| Item | When |
|---|---|
| Steamworks SDK integration (init, overlay, Steam ID for login, friends/invites) | With the first playable build |
| Steam Input (controller support) + UI readable at 1280×800 | Before Steam Deck Verified review |
| Achievements, Cloud saves | Before release |
| Depot builds: Windows x64, Linux x64 (native, Steam Runtime), macOS (MoltenVK) optional | From M5 |
| Minimum spec: Vulkan 1.2-capable GPU with current drivers; reference device Steam Deck | Store page |
| Pipeline cache on disk, shader warm-up at load | M5 (avoids first-run hitches) |

## 15. Future work and the trigger for each

These are designed for, not built now. Each is started when a benchmark shows
the listed symptom.

| Feature | What the plan already does to allow it | Trigger |
|---|---|---|
| Render graph (dependency-ordered passes) | Passes are data with declared inputs/outputs (§4.2) | First pass whose order depends on others (render-to-texture, post-processing) |
| GPU-driven rendering: compute culling, indirect draws | Instance data already lives in GPU buffers; `GpuBuffer` supports indirect usage | CPU culling/submission > ~2 ms per frame |
| Model LOD, far-terrain impostors | Chunk LOD already exists (§10.1); models can add LOD levels the same way | Crowds or draw distance make the scene geometry-bound |
| Multi-threaded command recording | Per-frame command pools per thread in `FrameContext` | Command recording > ~2 ms per frame |
| Mesh shaders for voxel chunks | Chunk meshes already GPU-resident | Chunk geometry becomes the bottleneck on supported GPUs |
| Blended skinning | Format already stores 2 bones/vertex | Art needs smooth joints |

## 16. Open questions for review

Answered in revision 4 (see `GAME_DESIGN.md`): movement/combat (full Trove
action), land bids (in-game gold), scale (one map split across zone servers,
up to 10,000 players each), building (owned land + wilderness editable by
all).

Still open:

1. **Shader language**: GLSL or HLSL (both compile to SPIR-V)?
2. **Draw distance target**: how many chunks on the Steam Deck vs a desktop
   GPU? (Proposed: 12 chunks full detail + LOD to 32 on desktop; 8 + LOD to 16
   on Deck.)

Gameplay questions still open are listed in `GAME_DESIGN.md` §35.

## 17. Risks

- **Vulkan complexity**: synchronization and layout bugs can appear on one
  GPU vendor and not another. Mitigation: validation layers + synchronization
  validation in every development run, testing on NVIDIA, AMD (incl. Steam
  Deck) and Intel before each milestone closes.
- **Small scenes** may show no gain: both backends are GPU-idle there. The gate
  is "not slower", not "faster everywhere".
- **Old or buggy drivers** on Windows: fail early with a clear message and
  the minimum requirement rather than crashing.
- **Losing web builds** for TowerSwarm and Meteor Dodge (accepted).
- **HUD migration** touches ~270 lines in tower_swarm; mechanical, but needs
  screenshot checks.
- **Streaming threads** introduce concurrency into an engine that is
  single-threaded today; confined to the streamer with message queues and
  covered by tests that stream a synthetic world.
