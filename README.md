# Attome Game Engine

This folder contains the lightweight 2D game engine and sample games used for
engine iteration.

The engine core lives in `engine/` and is built once as the `AttomeCore`
static library (see `engine/CMakeLists.txt`); games link it via
`attome_add_game()`:

- `engine/ATMEngine.h`
- `engine/ATMEngine.cpp`
- `engine/ATMDynamicArray.h`
- `engine/ATMProfiler.h`

The key architectural change in this branch is the explicit split between
`Dynamic`, `Hybrid`, and `Static` runtime kinds.

## Runtime Kinds

### Dynamic

Dynamic objects are the full simulation path.

- Updated globally every frame
- Indexed in the moving spatial grid
- Rendered only when visible
- Best for players, bullets, AI agents, moving enemies, and anything that must
  keep simulating even when off-screen

### Hybrid

Hybrid objects are optimized dynamic objects.

- Indexed in the moving spatial grid
- Rendered only when visible
- Updated only when they are inside or near the camera rectangle
- Best for enemies, crowds, or ambient actors that do not need full-scene
  simulation every frame

### Static

Static objects are the cheapest runtime path.

- Not updated every frame
- Not inserted into the moving spatial grid
- Stored in chunked static caches for rendering
- Only visible chunks are rebuilt and drawn
- Best for terrain, background planets, decorations, buildings, and other
  scene geometry that rarely changes

Static objects may still be:

- created
- removed
- moved
- hidden or shown

When that happens, the static cache is marked dirty and rebuilt on demand.

## Recommended Registration API

Use the explicit engine registration functions when you want the runtime split
to be active:

```cpp
int dynamic_type = engine_register_dynamic_type(engine, dynamic_container);
int hybrid_type = engine_register_hybrid_type(engine, hybrid_container);
int static_type = engine_register_static_type(engine, static_container);
```

Create and manage entities through the runtime-aware helper API:

```cpp
EntityHandle id = engine_create_entity(engine, hybrid_type);
engine_set_entity_position(engine, id, hybrid_type, x, y);
engine_set_entity_visible(engine, id, hybrid_type, true);
engine_set_entity_z_index(engine, id, hybrid_type, 10);
```

Destroy entities with:

```cpp
engine_destroy_entity(engine, id, hybrid_type);
```

If you mutate static entities directly inside the container, call:

```cpp
engine_mark_static_dirty(engine);
```

## Compatibility Rules

The old path is still supported:

```cpp
engine->entityManager.registerEntityType(container);
engine->entityManager.createEntity(type_id);
```

Important notes:

- `registerEntityType(container)` defaults to `Dynamic`
- direct `createEntity(type_id)` does not automatically choose the optimized
  runtime path for you
- if you use the old direct path for dynamic or hybrid objects, you are still
  responsible for correct grid management
- the explicit `engine_register_*` and `engine_*entity*` helpers are now the
  intended path for scenes that use all 3 runtime kinds

## Frame Pipeline

### Update

`engine_update(engine)` now owns the runtime split:

1. process pending removals
2. update all dynamic types globally
3. update hybrid types only inside a camera-expanded query rectangle
4. process pending removals again before render

`engine_update_entity_types(engine, dt)` is kept only for compatibility. The
runtime-aware update path is in `engine_update(engine)`.

### Render

`engine_render_scene(engine)` now does this:

1. rebuild static chunk references if the static scene changed
2. rebuild only dirty visible static chunks
3. append visible static cached geometry into the batch manager
4. query the moving grid for visible dynamic and hybrid entities
5. batch all visible geometry by texture and z-index
6. render the final merged batches

### Spatial Grid

`SpatialGrid::rebuild_grid(engine)` only rebuilds entries for:

- Dynamic types
- Hybrid types

Static types are skipped completely.

## Scene Construction Pattern

A typical optimized scene should look like this:

- Register background or map geometry as `Static`
- Register semi-active enemies or ambient actors as `Hybrid`
- Register always-simulated gameplay entities as `Dynamic`

That gives the intended Option B behavior:

- static objects are cheap to keep in large counts
- hybrid objects avoid full-scene CPU cost
- dynamic objects keep full gameplay correctness

## Current Sample Folders

This tree contains sample or experimental game folders such as:

- `_template`
- `hello-world`
- `snake`
- `meteor_dodge`
- `ashlands-dominion`

The engine sources live only in `engine/` and are shared by every game rather
than duplicated per game.

## Engine Modules

| File | What it does |
|---|---|
| `engine/ATMEngine.h/.cpp` | Entities, runtime kinds, spatial grid, batching, static chunk cache |
| `engine/ATMConfig.h/.cpp` | `EngineConfig` (window, world, grid, timestep, audio) and `ATM_TUNABLE` game values, both loaded from JSON |
| `engine/ATMJson.h/.cpp` | Dependency-free JSON reader/writer (allows `//` comments and trailing commas) |
| `engine/ATMInput.h/.cpp` | Action map: keys, mouse and gamepad bound to named actions and axes, rebindable in JSON |
| `engine/ATMAudio.h/.cpp` | Sound playback on SDL3 audio streams (voices, looping, volume) |
| `engine/ATMText.h/.cpp` | Glyph-cached TTF text (needs SDL3_ttf; `ATM_HAS_TEXT` is defined when available) |
| `engine/ATMAssets.h/.cpp` | Reference-counted textures, sounds and fonts |

`_template/` is a working starting point that uses all of them.

## Configuration (JSON)

Keep one JSON file per game in `<game>/config/`. `attome_add_game()` copies
`config/` and `assets/` next to the executable on every build, so the shipped
file can be edited without recompiling. See `_template/config/game.json`.

### Engine settings

```cpp
EngineConfig cfg;                           // defaults
engine_config_load("config/game.json", cfg); // reads the "engine" section
Engine *engine = engine_create_with_config(cfg);
```

| Key | Default | Notes |
|---|---|---|
| `window.title/width/height/resizable/vsync` | "Attome Engine", 1280, 720, false, false | |
| `world.width/height` | 50000 | Size of the spatial grid ¹ |
| `grid.cell_size` | 64 | Grid cell size in world units ¹ |
| `grid.node_reserve` | 3200000 | Grid nodes preallocated (≈20 bytes each). Lower it for small games or web |
| `grid.query_pad_cells` | 4 | Extra cells searched up/left so large sprites aren't culled |
| `grid.static_chunk_size` | 512 | Static cache chunk size |
| `grid.hybrid_activation_margin` | 150 | How far outside the camera hybrid entities keep updating |
| `time.max_frame_dt` | 0.1 | Clamp for long frames |
| `time.fixed_timestep_hz` | 0 | 0 = one update per frame (old behaviour). >0 = fixed-rate simulation |
| `time.max_fixed_steps_per_frame` | 4 | Caps catch-up steps after a stall |
| `audio.enabled/max_voices/master_volume` | true, 32, 1.0 | |

¹ **Grid size is compile-time by default.** Cell math runs on every entity
move, and reading the size at runtime measured ~5% slower on
`engine_set_entity_position`. So by default the grid uses the
`WORLD_WIDTH`/`WORLD_HEIGHT`/`GRID_CELL_SIZE` constants (identical to the
original engine), and these three keys are ignored with a log message.
Configure with `-DATTOME_RUNTIME_GRID=ON` to size the grid from JSON instead.

The old `engine_create(w, h, world_w, world_h, cell)` still works and keeps
the previous fixed 50000×50000 / 64px grid, so existing games are unaffected.
Code that needs grid math should use `engine->grid.cellSize()`,
`invCellSize()`, `cellsWide()` and `cellsHigh()` rather than the
`GRID_CELL_*` constants.

**Fixed timestep.** With `fixed_timestep_hz > 0`, `engine_update()` runs
0–N entity updates per frame, each with the same `dt`, and exposes
`engine->sim_steps`, `engine->sim_dt` and `engine->fixed_alpha` (0..1, for
interpolating rendering). `engine->frame_dt` is always the real frame time.

### Game tunables

Declare game values in a header instead of `constexpr`:

```cpp
#include "ATMConfig.h"
namespace my_game {
ATM_TUNABLE_SECTION("player");
ATM_TUNABLE(float, kSpeed, 540.0f);
ATM_TUNABLE(int, kLives, 3);
}
```

```cpp
atm::Tunables::instance().loadFile(atm::resolve_path("config/game.json"));
// once per frame — re-reads the file only when its timestamp changes:
atm::Tunables::instance().reloadIfChanged();
```

```json
{ "player": { "kSpeed": 600, "kLives": 5 } }
```

- Missing keys keep their compiled-in default; wrong types, out-of-range
  values and unknown keys (typos) are logged and ignored.
- A file that fails to parse during a live reload is ignored until it's fixed.
- `Tunables::instance().writeFile(path)` dumps every tunable with its current
  value — a quick way to generate the initial file.
- Values that are only read at startup (pool sizes, window size) need a
  restart. Use `addReloadListener()` to recompute anything derived from
  tunables.
- Custom types work by specialising `atm::TunableTraits<T>` (see
  `tower_swarm/src/Constants.h` for a colour type).

**Performance note.** A tunable is a normal global, so the compiler can't
assume it stays constant inside a loop that writes through pointers. In a
tight per-entity loop, copy it to a local first:

```cpp
const float speed = my_game::kSpeed;   // hoisted: same speed as constexpr
for (int i = 0; i < count; ++i) x[i] += vx[i] * speed * dt;
```

`atm_bench` measures both forms (see below).

### Input

```cpp
atm::InputMap input;
input.bind("fire", "Space");                 // defaults in code...
input.loadBindings(*config.find("input"));   // ...overridden by JSON
const atm::ActionId fire = input.actionId("fire");

input.beginFrame();
while (SDL_PollEvent(&e)) input.handleEvent(e);
if (input.pressed(fire)) { ... }
float move = input.axis("move_x");           // -1..1, keys or stick
```

Bindings use SDL scancode names (`"A"`, `"Left Shift"`), `"mouse:left"` and
`"pad:a"`. Gamepads are picked up automatically when the game initialises
`SDL_INIT_GAMEPAD`.

### Audio

```cpp
atm::Audio audio;
audio.init(engine->config);
atm::SoundId boom = audio.loadSound("sfx/boom.wav");
audio.play(boom);                              // one-shot
atm::VoiceId music = audio.play(song, 0.5f, true); // looping
```

Sounds are converted to the device format when loaded. SDL mixes voices on
its audio thread, and looping voices are refilled there too, so audio adds no
per-frame work. When all voices are busy, the oldest one-shot is replaced;
looping voices are replaced last.

### Text

```cpp
atm::Font font;
font.load(engine->renderer, "assets/fonts/AtomicMd.ttf", 18.0f);
font.draw(engine->renderer, "Score: 42", 12, 12, {1, 1, 1, 1});
```

Glyphs are rasterised once into an atlas; each `draw()` is a single
`SDL_RenderGeometry` call.

### Assets

```cpp
atm::Assets assets(engine, &audio);
int tex = assets.acquireTexture("resource/ship1.png"); // PNG/JPG/BMP/TGA
assets.releaseTexture("resource/ship1.png");           // freed at 0 refs
```

### Rotation

Sprites draw axis-aligned unless their container opts in:

```cpp
container->enableRotation();                     // ContainerFlag::ROTATABLE
engine_set_entity_rotation(engine, id, type, radians);
```

Containers that don't opt in take exactly the same render path as before.

## Building, Tests and Benchmarks

```sh
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
build/engine/bench/Release/atm_bench          # micro-benchmarks
```

Options: `ATTOME_RUNTIME_GRID` (see above), `ATTOME_BUILD_NET` (needs
asio/libsodium; Snake depends on it),
`ATTOME_BUILD_TESTS`, `ATTOME_BUILD_BENCH`, `ATTOME_WITH_TEXT`,
`ATTOME_BUILD_<GAME>`.

- `engine/tests/` — unit tests with a tiny built-in runner (`atm_test.h`).
  They run headless (SDL dummy video/audio drivers), so they need no display.
  Filter by name: `atm_tests grid`.
- `engine/bench/atm_bench` — entity churn, grid move/query, render batching,
  rotation cost, and tunable vs constexpr loops.
- `engine/bench/atm_compare` — uses only the original API, so the same source
  can be built against an older engine revision for a before/after check.

### Measured cost of this round of changes

Release, MSVC 2022, `atm_compare` built against the previous engine and the
current one (interleaved runs):

| | old | new |
|---|---|---|
| `engine_render_scene`, 20k sprites (median) | 6196 µs | 6207 µs |
| `engine_set_entity_position` ×100k (p25) | 1261 µs | 1230 µs |

Both are within run-to-run noise. `atm_bench` also shows: runtime vs legacy
grid identical, rotation opt-in at angle 0 free, and an `ATM_TUNABLE` read
in a tight loop 3× slower than `constexpr` unless hoisted into a local
(then identical).
