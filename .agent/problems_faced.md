# Problems Faced and Fixes

A log of issues encountered while developing this game engine, and their solutions.

---

## 1. Grid Deadlock from Parallel Updates

**Date:** 2025-12-22

**Symptom:** Program deadlocks in `SpatialGrid::queryRect()` or `queryCircle()` - infinite loop in `while (nodeIdx != -1)`.

**Root Cause:** `ObstacleEntityContainer::update()` called `engine->grid.move()` inside a parallel `std::for_each`. Multiple threads modifying the grid's linked list (`cell_heads`, `nodes[].next`, `nodes[].prev`) corrupted pointers, creating cycles or dangling references.

**Fix:** Deferred grid updates:
1. Parallel phase: Only update positions, collect entity indices that changed cells into a buffer using `std::atomic<uint32_t>` counter
2. Serial phase: Apply `grid.move()` calls after parallel loop completes

```cpp
// Buffer for deferred grid updates (thread-safe collection)
static std::vector<uint32_t> pending_moves;
static std::atomic<uint32_t> pending_count{0};
pending_count.store(0, std::memory_order_relaxed);

// Parallel phase - collect, don't modify grid
std::for_each(std::execution::par, indices.begin(), indices.end(), [&](uint32_t i) {
    // ... position updates ...
    if (cell changed) {
        uint32_t slot = pending_count.fetch_add(1, std::memory_order_relaxed);
        pending_moves[slot] = i;
    }
});

// Serial phase - safe grid modification
for (uint32_t j = 0; j < pending_count; ++j) {
    engine->grid.move(grid_node_indices[pending_moves[j]], ...);
}
```

**Lesson:** Never modify shared data structures (linked lists, hash maps) from parallel threads without proper synchronization. Prefer collect-then-apply patterns for performance-critical parallel code.

---

## 2. Player Grid Position Not Updating

**Date:** 2025-12-22

**Symptom:** Spatial queries (collision detection) returned incorrect results for player after moving.

**Root Cause:** `handle_input()` updated `x_positions`/`y_positions` but never called `engine->grid.move()` to update the spatial grid.

**Fix:** Added grid update after player position change:
```cpp
if (oldCellX != newCellX || oldCellY != newCellY) {
    engine->grid.move(pCont->grid_node_indices[player_idx], next_x, next_y);
    pCont->cell_x[player_idx] = newCellX;
    pCont->cell_y[player_idx] = newCellY;
}
```

**Lesson:** Any entity that moves must update both its position arrays AND its grid position for spatial queries to work correctly.

---

## 3. Obstacles Moving in Same Direction

**Date:** 2025-12-22

**Symptom:** All obstacles moved in the same direction (right), not randomly.

**Root Cause:** 
- `createEntity()` used fixed direction `(15, 0.3)` instead of random
- `setup_game()` passed fixed values `(15, 15.0f, 0.0f)` to obstacle creation

**Fix:** Generate random angle, convert to unit vector:
```cpp
float angle = static_cast<float>(rand()) / RAND_MAX * 2.0f * 3.14159265f;
dir_x[index] = cosf(angle);
dir_y[index] = sinf(angle);
```

**Lesson:** Always verify that "random" values are actually being randomized at the call site, not just in one code path.

---

## 4. Entity Removal Corrupts Grid Indices (Swap-and-Pop)

**Date:** 2025-12-22

**Symptom:** Eating food has no effect - score doesn't increase, snake doesn't grow.

**Root Cause:** Engine constraint says "removing not allowed", and `removeEntity()` uses swap-and-pop pattern:
1. Entity at index X is removed
2. Entity at index LAST is moved to index X
3. BUT the grid still stores the OLD index for the moved entity
4. Grid queries return stale/wrong entity indices

**Fix:** Instead of removing entities, **relocate** them:
```cpp
// RELOCATE food instead of removing (engine constraint: no removal)
float new_x = static_cast<float>(rand() % (WORLD_WIDTH - GRID_SIZE));
float new_y = static_cast<float>(rand() % (WORLD_HEIGHT - GRID_SIZE));
fCont->x_positions[entity.index] = new_x;
fCont->y_positions[entity.index] = new_y;
fCont->values[entity.index] = 10 + (rand() % 20);

// Update grid position
int32_t nodeIdx = fCont->grid_node_indices[entity.index];
engine->grid.move(nodeIdx, new_x, new_y);
fCont->cell_x[entity.index] = static_cast<uint16_t>(new_x * INV_GRID_CELL_SIZE);
fCont->cell_y[entity.index] = static_cast<uint16_t>(new_y * INV_GRID_CELL_SIZE);
```

**Lesson:** Respect engine constraints ("removing not allowed"). When entities need to "disappear", teleport/reuse them instead of removing. This keeps grid indices valid and avoids swap-and-pop corruption.

---

## 5. Query Range Too Small for Grid Cell Size

**Date:** 2025-12-22

**Symptom:** Collision detection never finds food/enemies even when snake is touching them.

**Root Cause:** `queryCircle` checks distance from query center to **cell corners**, not to entity positions. The grid cell size is 1024px, but query radius was only 64px (`GRID_SIZE * 2`). The query could only find entities if the cell corner happened to be within 64px of the search center - which almost never happens.

**Fix:** Query radius must be at least `GRID_CELL_SIZE` to reliably find entities in adjacent cells:
```cpp
// Query nearby entities - use GRID_CELL_SIZE to ensure we cover adjacent cells
// The queryCircle checks distance to cell corners, so we need radius >= cell size
float query_range = static_cast<float>(GRID_CELL_SIZE) * 1.5f;
const auto &nearby = engine->grid.queryCircle(head_x, head_y, query_range);
```

**Lesson:** Understand the spatial grid's query semantics. When `queryCircle` checks cell corners (not entity positions), the radius must be comparable to `GRID_CELL_SIZE`, not entity size.

---

## 6. All Textures Same Color (Atlas Overlap)

**Date:** 2025-12-22

**Symptom:** All entities appear the same color (yellow). Food should be red, snake should be green, enemies purple, but everything is yellow.

**Root Cause:** All textures were registered at the same atlas position `(0, 0)`:
```cpp
engine_register_texture(engine, head_surface, 0, 0, GRID_SIZE, GRID_SIZE);
engine_register_texture(engine, body_surface, 0, 0, GRID_SIZE, GRID_SIZE);
engine_register_texture(engine, food_surface, 0, 0, GRID_SIZE, GRID_SIZE);
// ... all at (0, 0)!
```
Each texture overwrites the previous one. The last one (yellow power-up) is the only visible color.

**Fix:** Place each texture at a different position in the atlas:
```cpp
engine_register_texture(engine, head_surface, 0, 0, GRID_SIZE, GRID_SIZE);              // y=0
engine_register_texture(engine, body_surface, 0, GRID_SIZE, GRID_SIZE, GRID_SIZE);      // y=32
engine_register_texture(engine, food_surface, 0, GRID_SIZE * 2, GRID_SIZE, GRID_SIZE);  // y=64
engine_register_texture(engine, enemy_surface, 0, GRID_SIZE * 3, GRID_SIZE, GRID_SIZE); // y=96
engine_register_texture(engine, powerup_surface, 0, GRID_SIZE * 4, GRID_SIZE, GRID_SIZE); // y=128
```

**Lesson:** When using a texture atlas, each texture must be placed at a unique position. The (x, y) parameter specifies WHERE in the atlas the texture is copied - they must not overlap.

---

## 7. SDL3 API Changes (bool* vs Uint8*)

**Date:** 2025-12-22

**Symptom:** Compilation error: `cannot convert from 'const bool *' to 'const Uint8 *'`

**Root Cause:** SDL3 changed `SDL_GetKeyboardState()` to return `const bool *` instead of `const Uint8 *` (as it was in SDL2).

**Fix:** Update function signatures and variable types:
```cpp
// Old (SDL2):
const Uint8 *keyboard_state = SDL_GetKeyboardState(NULL);

// New (SDL3):
const bool *keyboard_state = SDL_GetKeyboardState(NULL);
```

**Lesson:** When using SDL3, be aware of API changes from SDL2. Check return types in documentation.

---

## 8. Keyboard Input Not Responding (Stale State)

**Date:** 2025-12-22

**Symptom:** Player snake moves in one direction but doesn't respond to keyboard input (WASD/arrows).

**Root Cause:** `SDL_GetKeyboardState()` was called BEFORE `SDL_PollEvent()`. The keyboard state is only updated when `SDL_PumpEvents()` is called (which `SDL_PollEvent` calls internally). Getting the state before polling returns stale data.

**Fix:** Call `SDL_GetKeyboardState()` AFTER the event polling loop:
```cpp
// WRONG - stale keyboard state:
const bool *keyboard_state = SDL_GetKeyboardState(NULL);
while (SDL_PollEvent(&event)) { ... }
handle_input(keyboard_state);  // Input doesn't work!

// CORRECT - fresh keyboard state:
while (SDL_PollEvent(&event)) { ... }
const bool *keyboard_state = SDL_GetKeyboardState(NULL);  // After events pumped
handle_input(keyboard_state);  // Input works!
```

**Lesson:** `SDL_GetKeyboardState()` reflects the state at the time of the last `SDL_PumpEvents()` call. Always get keyboard state after event processing.

---

## 9. Smooth Movement Breaks Chain Following

**Date:** 2025-12-22

**Symptom:** When implementing smooth pixel-by-pixel movement, snake body segments either don't follow or all collapse to the same position.

**Root Cause:** Classic snake body following works by "each segment moves to where the previous segment WAS". With smooth movement, each frame only moves a few pixels, so the old position is nearly identical to the new position - causing segments to bunch up.

**Fix:** Keep discrete LOGIC positions (grid-aligned) and smooth VISUAL positions (interpolated) separate:
```cpp
struct SnakeSegment {
  float x, y;               // Logic position (grid-aligned)
  float visual_x, visual_y; // Visual position (smooth)
  uint32_t entity_index;
};

// Discrete logic: body follows in chain (every N frames)
seg.x = prev_x;  // Logic jumps by GRID_SIZE

// Smooth visual: lerp toward logic each frame
seg.visual_x += (seg.x - seg.visual_x) * lerp_factor;
```

**Lesson:** For snake games with smooth rendering, separate logic (for correct mechanics) from visuals (for smooth appearance). Logic uses discrete grid steps; visuals interpolate between them.

---

## 10. Z-Index Layering with Many Entities

**Date:** 2025-12-22

**Symptom:** Snake becomes invisible or appears to be "inside" food when there are 1 million+ food entities.

**Root Cause:** With massive entity counts, render order becomes critical. If snake has z=10 and food has z=5, but rendering doesn't properly sort, the sheer volume of food can obscure the snake.

**Fix:** Use significantly different z-indices:
```cpp
z_indices[index] = 5;    // Food
z_indices[index] = 99;   // Snake body
z_indices[index] = 100;  // Snake head
```

**Lesson:** When dealing with millions of entities, use large gaps between z-indices for different entity types. Don't rely on subtle differences like 9 vs 10.

---

## 11. Struct Initializer Mismatch After Adding Fields

**Date:** 2025-12-22

**Symptom:** `no instance of overloaded function "push_back" matches the argument list` or `narrowing conversion` errors.

**Root Cause:** When adding new fields to a struct (like `visual_x, visual_y`), existing brace-initializer lists `{x, y, idx}` no longer match the struct's field count `{x, y, visual_x, visual_y, idx}`.

**Fix:** Update ALL places that create struct instances:
```cpp
// Old (3 fields):
snake_body.push_back({x, y, idx});

// New (5 fields):
snake_body.push_back({x, y, x, y, idx});  // x,y used for both logic and initial visual
```

**Lesson:** When modifying struct definitions, grep the entire codebase for all instantiation sites. Brace initializers are sensitive to field order and count.

---
