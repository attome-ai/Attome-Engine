# Engine Problems & Patterns

Quick reference for common engine pitfalls. Written for AI understanding.

---

## 1. Parallel Grid Modification = Deadlock

**Problem:** Calling `grid.move()` from multiple threads corrupts linked list → deadlock.

**Pattern:** Collect indices in parallel, apply grid changes serially.
```cpp
// Parallel: collect only
pending_moves[atomic_counter++] = i;
// Serial: modify grid
for (j) grid.move(pending_moves[j], ...);
```

---

## 2. Must Update Grid After Position Change

**Problem:** Changed entity position but collision detection uses old location.

**Pattern:** Always call `grid.move()` after modifying `x_positions`/`y_positions`.
```cpp
x_positions[idx] = new_x;
grid.move(grid_node_indices[idx], new_x, new_y);
cell_x[idx] = new_x * INV_GRID_CELL_SIZE;
```

---

## 3. No Entity Removal → Relocate Instead

**Problem:** Engine constraint forbids removal. Removing corrupts grid indices.

**Pattern:** Teleport entity to new position instead of removing.
```cpp
// "Remove" food = relocate it
x_positions[idx] = rand() % WORLD_WIDTH;
grid.move(node_idx, x_positions[idx], y_positions[idx]);
```

---

## 4. Query Radius Must Match Cell Size

**Problem:** `queryCircle` checks cell corners, not entity positions. Small radius misses entities.

**Pattern:** Use `GRID_CELL_SIZE * 1.5f` as minimum query radius.
```cpp
float query_range = GRID_CELL_SIZE * 1.5f;
grid.queryCircle(x, y, query_range);
```

---

## 5. Texture Atlas Positions Must Be Unique

**Problem:** All textures at `(0,0)` → they overwrite each other.

**Pattern:** Offset each texture by its size.
```cpp
register_texture(surface1, 0, 0, SIZE, SIZE);        // y=0
register_texture(surface2, 0, SIZE, SIZE, SIZE);     // y=32
register_texture(surface3, 0, SIZE*2, SIZE, SIZE);   // y=64
```

---

## 6. SDL3: bool* Not Uint8*

**Problem:** SDL3 changed `SDL_GetKeyboardState()` return type.

**Pattern:** Use `const bool*` for SDL3.
```cpp
const bool *keys = SDL_GetKeyboardState(NULL);  // SDL3
```

---

## 7. Get Keyboard State AFTER PollEvent

**Problem:** Keyboard state is stale if read before event pump.

**Pattern:** Call `SDL_GetKeyboardState` after event loop.
```cpp
while (SDL_PollEvent(&e)) { ... }  // Pumps events
const bool *keys = SDL_GetKeyboardState(NULL);  // Now fresh
```

---

## 8. Separate Logic vs Visual Positions

**Problem:** Smooth movement breaks discrete game mechanics.

**Pattern:** Store both, lerp visual toward logic each frame.
```cpp
// Logic: discrete grid steps
logic_x += GRID_SIZE;
// Visual: smooth interpolation
visual_x += (logic_x - visual_x) * lerp_factor;
// Render uses visual, collision uses logic
```

---

## 9. Large Z-Index Gaps for Many Entities

**Problem:** With millions of entities, subtle z-index differences get lost.

**Pattern:** Use large gaps (e.g., 5, 99, 100) not small (9, 10, 11).

---

## 10. Update All Struct Initializers After Adding Fields

**Problem:** Adding struct fields breaks existing `{...}` initializers.

**Pattern:** Grep codebase for struct name, update all instantiation sites.

---
