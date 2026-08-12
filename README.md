

# Attome Game Engine

This folder contains the lightweight 2D game engine and sample games used for
engine iteration.

The engine core currently lives in:

- `ATMEngine.h`
- `ATMEngine.cpp`
- `ATMDynamicArray.h`
- `ATMProfiler.h`

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
- `tower_swarm`

The engine sources are shared from the `games/` root rather than duplicated per
game.

## Validation Status

The current engine implementation was syntax-checked with MSVC for:

- `games/ATMEngine.cpp`
- `games/_template/src/main.cpp` together with `games/ATMEngine.cpp`

That confirms the runtime-split API and engine implementation compile at the
translation-unit level in this workspace.
