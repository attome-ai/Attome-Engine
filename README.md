# Attome Game Engine

Attome Game Engine is the `games/` workspace for building 2D games on top of the Attome runtime. It is a separate CMake workspace from the backend and is intended to be opened directly in Visual Studio or configured from the `games/` folder.

The engine is built around:

- C++20
- SDL3 / SDL3_image / SDL3_ttf
- ImGui
- GLM
- `nlohmann_json`
- CMake + vcpkg

## Workspace Layout

- `game/` - public engine headers
- `gameSrc/` - engine implementation
- `include/` - shared support headers
- `hello-world/` - minimal starter sample
- `snake/` - simple gameplay sample
- `snake_optionb_test/` - object-runtime split validation sample
- `planet_optionb_benchmark/` - planet-style benchmark using dynamic, hybrid, and static objects
- `platforms/android/` - Android packaging support
- `_template/` - new game template

## Runtime Object Model

The engine now supports 3 runtime kinds for entity registration:

1. `Dynamic`
   Processed globally every frame.
   Use this for gameplay-critical moving objects that must always update.

2. `Hybrid`
   Stored in the moving grid, but only processed when inside the camera activation region.
   Use this for moving objects that only need simulation when near or visible.

3. `Static`
   Not processed every frame.
   Rendered through chunked cached geometry and rebuilt only when static content changes.
   Use this for level geometry, props, walls, decorations, and other rarely changing objects.

This split is the current recommended high-performance model for scene construction.

## Registration API

Register each entity container explicitly by runtime behavior:

```cpp
int dynamic_type = engine_register_dynamic_type(engine, dynamic_container);
int static_type = engine_register_static_type(engine, static_container);
int hybrid_type = engine_register_hybrid_type(engine, hybrid_container);
```

Create and manage entities through stable handles:

```cpp
EntityHandle h = engine_create_entity(engine, dynamic_type);
engine_set_entity_position(engine, h, 100.0f, 200.0f);
engine_set_entity_visible(engine, h, true);
engine_destroy_entity(engine, h);
```

Important notes:

- `Dynamic` and `Hybrid` entities are inserted into the moving spatial grid.
- `Hybrid` containers should override `updateVisible(...)` for best performance.
- `Static` containers should not rely on per-frame update logic.
- Static rendering is cached in chunks and invalidated automatically on static add/remove/move/visibility/z-order changes.

## Current Example Games

- `HelloWorld`
  Basic engine startup and rendering check.

- `Snake`
  Small gameplay example using the current workspace structure.

- `SnakeOptionBTest`
  Focused validation scene for the 3 runtime kinds:
  dynamic snake entities, static walls/gates, and hybrid visible-only objects.

- `PlanetOptionBBenchmark`
  Planet-style benchmark scene with:
  dynamic enemy planets,
  hybrid enemy planets,
  static planets,
  player movement and shooting,
  FPS logging in the console.

## Build

Open the `games/` folder directly in Visual Studio, or configure from the command line.

### Visual Studio

Open:

`c:\Users\Computia.me\Downloads\everything\testing\attome-root\games`

Visual Studio should detect the CMake workspace and generate the targets from `games/CMakeLists.txt`.

### Command Line

```powershell
cd games
cmake -S . -B out\build\vs2022_fresh -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/Users/Computia.me/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build out\build\vs2022_fresh --config Debug
```

If `VCPKG_ROOT` is configured, you can point `CMAKE_TOOLCHAIN_FILE` to:

`$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake`

## Current Targets

The active game targets in this workspace are:

- `HelloWorld`
- `Snake`
- `SnakeOptionBTest`
- `PlanetOptionBBenchmark`

## Performance Guidance

For maximum performance:

- Keep always-active gameplay objects in `Dynamic`.
- Move off-screen-only simulation to `Hybrid`.
- Put non-simulated world content in `Static`.
- Prefer stable handle APIs over raw slot assumptions for gameplay code.
- Treat static content as chunked render data, not as normal update-driven entities.

The intended scene-building workflow is:

1. register the cheapest runtime kind for each object class
2. create entities through handles
3. let the engine route update and render work through the matching runtime path

## Main Engine Files

If you want to inspect the current runtime split implementation, start here:

- `game/ATMEngine.h`
- `gameSrc/ATMEngine.cpp`
