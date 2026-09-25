# Attome Online — voxel MMO demo

A playable demo of the game described in `docs/GAME_DESIGN.md`: Trove-style
voxel world and movement, RuneScape-style skills and loot, authoritative
multiplayer server. Plans: `docs/DEMO_PLAN.md`, `docs/GPU_3D_PLAN.md`,
`docs/NETWORK_PLAN.md`.

> **Status:** written without a local build environment and **not yet
> compiled**. Expect a first round of compile fixes on the build machine.
> See "First build" below.

## Requirements

- Vulkan SDK 1.3+ (for `glslangValidator` and validation layers)
- CMake 3.24+, a C++20 compiler (MSVC 2022, Clang 16+, GCC 13+)
- vcpkg (manifest mode; `vcpkg.json` at the repo root lists every library:
  SDL3, glm, imgui with SDL3 + Vulkan bindings, VMA, vk-bootstrap, glslang,
  asio, libsodium, …)
- A GPU with Vulkan 1.3 (or 1.2 with dynamic rendering + synchronization2)

## Build

```sh
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release --target ao_client ao_server ao_bots ao_tests
```

Useful options: `-DATTOME_RUNTIME_GRID=ON` (runtime-sized 2D grid),
`-DATTOME_BUILD_NET=OFF` (skip the old asio network library),
`-DATTOME_BUILD_TESTS=ON`.

## Run

| Command | What |
|---|---|
| `ao_client --local` | Single player: starts a server inside the client (default in `config/client.json`) |
| `ao_server --port 27015` | Dedicated zone server (settings: `config/server.json`) |
| `ao_client --host 1.2.3.4 --port 27015 --name Alice` | Join a server |
| `ao_bots --host 127.0.0.1 --count 500 --behaviour wander` | Load test with headless bots (`wander`, `mine`, `fight`) |
| `ao_tests` | Unit tests (voxel, network, gameplay) |
| `ao_client --validation` | Enable Vulkan validation layers |

## Controls

| Key | Action |
|---|---|
| W A S D | Move |
| Space | Jump · double jump · hold in the air to glide |
| Left Shift | Dash |
| Left Ctrl | Sprint |
| Mouse | Look · wheel = zoom (Trove style) |
| Left mouse | Attack (weapon) · mine (pickaxe / hands) |
| Right mouse | Place the selected block |
| 1–9 / Shift + wheel | Select hotbar slot |
| Tab / I | Inventory and equipment (click to equip — gear swaps instantly) |
| K | Skills (21 RuneScape-style skills, total and combat level) |
| Enter | Chat |
| F3 / F11 | Debug overlay (FPS, GPU time, chunks, network) |
| F2 | Screenshot (`screenshot.bmp`) |
| Esc | Release the mouse |

Everything is rebindable in `config/client.json` → `"input"`. Movement
tuning lives in `config/game.json` → `"movement"` and reloads live.

## Layout

```
engine/voxel    chunks (palette), binary greedy mesher, lighting, world gen, streamed VoxelWorld
engine/render   Vulkan renderer: packed faces, GPU culling + indirect draws, bloom, models, ImGui
engine/model    modular characters: shared rig, animations, equipment, monsters
engine/net2     AttomeNet 2 transport, bit streams, X-macro message schema
attome_online/shared   game rules shared by client/server: skills, items, monsters, movement, protocol
attome_online/server   authoritative zone server
attome_online/client   game client
attome_online/bots     headless load-test clients
attome_online/tests    unit tests
```

## First build

Because the code was written without compiling, the first build will likely
need small fixes (typos, a missed include, a library API detail). Fix order
that keeps each step small:

1. `ao_tests` (voxel, net, gameplay — no GPU needed) → run it.
2. `ao_server` → run it, then `ao_bots --count 20` against it.
3. `ao_client --local --validation` → fix any Vulkan validation errors.
4. Multiplayer: `ao_server` + two `ao_client --host 127.0.0.1`.

## Known limitations (demo)

- No encryption / connect tokens yet (NETWORK_PLAN milestone N2).
- Single network thread; batched I/O and multi-threading are milestone N1.
- No LOD or Hi-Z occlusion culling yet (cave culling + frustum + per-direction culling are in).
- Light is recomputed per chunk mesh job (no incremental light updates).
- Server doesn't yet validate block-break time; no lag compensation for hits.
- Dye colours in appearances aren't applied yet.
- The client keeps every edited chunk it has received in memory (no eviction yet).
- The dedicated server links SDL3 (through the engine's JSON/config helpers)
  even though it opens no window.
- In `--local` mode, editing `config/game.json` while playing reloads tunables
  on the client thread while the in-process server reads them (harmless in
  practice, not strictly thread-safe).
- The server binds all interfaces; on multi-homed hosts, bind a specific
  address (TODO) so replies come from the address clients connect to.
