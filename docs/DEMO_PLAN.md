# Attome Online — playable demo plan

Status: in progress. First playable demo of the game in `GAME_DESIGN.md`,
built on the engine in `GPU_3D_PLAN.md` and the network library in
`NETWORK_PLAN.md`. Written without a local build environment; built and
tested on a machine with Vulkan SDK 1.3 and vcpkg.

## 1. What the demo contains

| Area | Demo scope |
|---|---|
| Renderer | Vulkan 1.3: packed-face vertex pulling, per-direction buckets, GPU arena + compute cull + indirect-count draws, AO + light, fog, sky, bloom, MSAA, voxel models, block highlight, ImGui HUD |
| World | Procedural terrain (hills, plains, sand, snow, water, trees, caves, ores, crystals), 32³ palette chunks, binary greedy meshing on worker threads, sky + block light, cave culling, streaming by distance |
| Player | Modular humanoid rig, shared animations (idle, walk, run, jump, fall, glide, swim, dash, mine, place, sword, bow, hit, death), gear swapping (helmet, armour, weapon, back) |
| Movement | Trove-style: run, sprint, jump + double jump, glide, dash, swim, step-up; shared client/server code |
| Gameplay | Mining and building (break/place blocks, drops into a 28-slot inventory), sword + bow combat, 3 monster types with AI, damage-share loot, 21 skills with the RS-to-99-then-linear curve (XP from Mining, Attack, Strength, Ranged, Hitpoints, Construction in the demo), level-up effect |
| Audio | SDL3 audio with procedurally synthesised effects (footsteps, jump, land, glide wind, dash, swing, bow, hit, break, place, pickup, level-up, UI) and a generated ambient loop |
| Network | AttomeNet 2 transport (UDP, challenge cookie, reliable/unreliable/bulk channels, fragmentation, timeouts), X-macro protocol with schema hash, snapshots with interest + priority + field deltas, client prediction + reconciliation, interpolation |
| Server | Headless authoritative zone server: world edits, players, monsters, loot, XP; snapshot builder with spatial-grid interest |
| Client | Connects to a server, or `--local` starts an in-process server (single player) |
| Bots | Headless bot clients (walk, jump, mine, fight) for load testing: `ao_bots --count 500` |

## 2. Layout and build targets

```
engine/
  voxel/    AttomeVoxel   Chunk, BlockRegistry, ChunkMesher, WorldGenerator, light, ChunkStreamer
  model/    AttomeModel   Rig, AnimLibrary, Animator, ModelLibrary (demo content), meshPart
  render/   AttomeRender  Renderer (Vulkan: vk/*, shaders/*.glsl compiled to SPIR-V at build time)
  net2/     AttomeNet2    Host (UDP transport), BitStream, Schema (X-macro messages)
attome_online/
  shared/   ao_shared     GameTypes, Skills, Items, Monsters, Movement, Protocol, Snapshot, Sfx synth data
  server/   ao_server     ZoneServer, ServerWorld, Entities, Combat, Loot (exe + library for --local)
  client/   ao_client     App, ClientWorld, Prediction, Interpolation, CharacterRenderer, Hud, Audio
  bots/     ao_bots       Headless bot clients
  tests/    ao_tests      Unit tests (atm_test.h runner): chunk, mesher, bitstream, schema, transport, movement, skills
```

Dependencies (vcpkg.json): sdl3, glm, imgui (sdl3 + vulkan bindings), glslang (tools),
vulkan-memory-allocator, vk-bootstrap. Shaders: `glslangValidator` from the
Vulkan SDK, or vcpkg glslang tools as a fallback.

## 3. Contracts (written first; all modules build against these)

| Header | Owner of implementation |
|---|---|
| `engine/voxel/VoxelTypes.h`, `MeshTypes.h`, `BlockRegistry.h`, `Chunk.h` | World module |
| `engine/render/Renderer.h` | Renderer module |
| `engine/model/Character.h` | Characters module |
| `engine/net2/Net.h`, `BitStream.h`, `Schema.h` | Network module |
| `attome_online/shared/GameTypes.h`, `Movement.h` | Characters/gameplay module |
| `attome_online/shared/Protocol.h`, `Snapshot.h` | Network module |

Changing a contract header requires updating every user of it.

## 4. Threading

| Thread | Work |
|---|---|
| Main (client) | Input, prediction, network pump, animation, audio mixing control, renderer submission |
| Mesh workers (client, N = cores − 2) | World generation of missing chunks, light, meshing |
| Server main | Fixed 30 Hz simulation, network pump, snapshot building |
| Server workers | World generation + chunk serialisation for clients |

## 5. Definition of done (per module)

- Implements its contract headers fully; no stubs left without a `TODO(demo)`
  comment explaining what is missing.
- Unit tests for the module's logic in `attome_online/tests`.
- No undefined behaviour on malformed network input (decoders bounds-checked).
- Code reviewed against the contracts by a second pass before integration.

## 6. Status

| Module | Status |
|---|---|
| Contracts | Done |
| World (engine/voxel) | Written |
| Renderer (engine/render) | Written |
| Characters + gameplay + audio | Written |
| Network + server + bots | Written |
| Client integration | Written |
| Review + fixes (compiler-style review of every module) | Done (4 review passes, ~35 fixes) |
| First build + run on a Vulkan machine | Not started (no build environment here) |
