// DynamicArray, entity containers, spatial grid, rendering and timestep.

#include "atm_test.h"

#include "ATMEngine.h"

#include <cmath>
#include <set>

namespace {

class TestContainer : public RenderableEntityContainer {
public:
  int updates = 0;
  float last_dt = 0.0f;

  explicit TestContainer(int capacity = 4)
      : RenderableEntityContainer(0, 0, capacity) {}

  void update(float delta_time) override {
    ++updates;
    last_dt = delta_time;
  }
};

Engine *make_engine(EngineConfig config = {}) {
  config.window_width = 320;
  config.window_height = 240;
  if (config.grid_node_reserve == EngineConfig{}.grid_node_reserve)
    config.grid_node_reserve = 1024;
  return engine_create_with_config(config);
}

} // namespace

// SECTION: dynamic_array

ATM_TEST(dynamic_array_default_and_resize) {
  DynamicArray<int> empty;
  ATM_CHECK_EQ(empty.capacity(), 0);
  ATM_CHECK(empty.data() == nullptr);

  DynamicArray<int> a(4, 7);
  ATM_CHECK_EQ(a.capacity(), 4);
  a[0] = 1;
  a[3] = 4;
  a.resize(8, 4, -1);
  ATM_CHECK_EQ(a.capacity(), 8);
  ATM_CHECK_EQ(a[0], 1);
  ATM_CHECK_EQ(a[3], 4);
  ATM_CHECK_EQ(a[4], -1);

  a.resize(2); // shrink keeps what fits
  ATM_CHECK_EQ(a.capacity(), 2);
  ATM_CHECK_EQ(a[0], 1);

  DynamicArray<int> moved(std::move(a));
  ATM_CHECK_EQ(moved.capacity(), 2);
  ATM_CHECK_EQ(a.capacity(), 0);
}

ATM_TEST(aligned_array_is_aligned) {
  AlignedDynamicArray<uint16_t, CACHE_LINE_SIZE> a(100);
  ATM_CHECK_EQ(reinterpret_cast<uintptr_t>(a.data()) % CACHE_LINE_SIZE, 0u);
  a[99] = 5;
  a.resize(300, 100);
  ATM_CHECK_EQ(reinterpret_cast<uintptr_t>(a.data()) % CACHE_LINE_SIZE, 0u);
  ATM_CHECK_EQ(a[99], 5);
  ATM_CHECK_EQ(a.capacity(), 300);
}

// SECTION: entity_container

ATM_TEST(container_create_remove_keeps_ids_stable) {
  TestContainer c(2);
  const EntityHandle a = c.createEntity();
  const EntityHandle b = c.createEntity();
  const EntityHandle d = c.createEntity(); // forces a resize
  ATM_CHECK_EQ(c.count, 3);
  ATM_CHECK(c.capacity >= 3);

  c.x_positions[c.getSlot(a)] = 10.0f;
  c.x_positions[c.getSlot(b)] = 20.0f;
  c.x_positions[c.getSlot(d)] = 30.0f;

  c.removeEntity(a); // swap-remove moves `d` into a's slot
  ATM_CHECK(!c.isAlive(a));
  ATM_CHECK(c.isAlive(b));
  ATM_CHECK(c.isAlive(d));
  ATM_CHECK_EQ(c.count, 2);
  ATM_CHECK_EQ(c.x_positions[c.getSlot(b)], 20.0f);
  ATM_CHECK_EQ(c.x_positions[c.getSlot(d)], 30.0f);
  ATM_CHECK_EQ(c.getStableId(c.getSlot(d)), d);

  c.removeEntity(a); // double remove is a no-op
  ATM_CHECK_EQ(c.count, 2);
}

ATM_TEST(container_renderable_fields_follow_swaps) {
  TestContainer c;
  const EntityHandle a = c.createEntity();
  const EntityHandle b = c.createEntity();
  c.widths[c.getSlot(b)] = 33;
  c.rotations[c.getSlot(b)] = 1.5f;
  c.removeEntity(a);
  ATM_CHECK_EQ(c.widths[c.getSlot(b)], 33);
  ATM_CHECK_EQ(c.rotations[c.getSlot(b)], 1.5f);
}

// SECTION: spatial_grid

ATM_TEST(grid_runtime_dimensions) {
  SpatialGrid grid(1000, 500, 100, 16, 2);
#if ATM_RUNTIME_GRID
  ATM_CHECK_EQ(grid.cellsWide(), 10);
  ATM_CHECK_EQ(grid.cellsHigh(), 5);
  ATM_CHECK_EQ(grid.cellSize(), 100.0f);
#else
  ATM_CHECK_EQ(grid.cellsWide(), static_cast<int32_t>(GRID_CELL_WIDTH));
#endif
  ATM_CHECK_EQ(grid.queryPadCells(), 2);

  SpatialGrid legacy;
  ATM_CHECK_EQ(legacy.cellsWide(), static_cast<int32_t>(GRID_CELL_WIDTH));
  ATM_CHECK_EQ(legacy.cellSize(), static_cast<float>(GRID_CELL_SIZE));
}

// Grid tests are written in cell units so they hold for both the default
// compile-time grid and ATM_RUNTIME_GRID.

ATM_TEST(grid_add_move_remove_query) {
  SpatialGrid grid(1000, 1000, 100, 16, 0);
  const float c = grid.cellSize();
  const float last = (grid.cellsWide() - 1) * c; // origin of the last column
  const int32_t n1 = grid.add({0, 1}, 0.5f * c, 0.5f * c);
  const int32_t n2 = grid.add({0, 2}, last + 0.5f * c, last + 0.5f * c);
  grid.add({0, 3}, last * 10.0f, last * 10.0f); // clamped into the last cell

  ATM_CHECK_EQ(grid.queryRect(0, 0, c - 1, c - 1).size(), 1u);

  ATM_CHECK(grid.move(n1, last + 0.6f * c, last + 0.6f * c));
  ATM_CHECK(!grid.move(n1, last + 0.7f * c, last + 0.7f * c)); // same cell
  ATM_CHECK_EQ(grid.queryRect(last, last, last + c - 1, last + c - 1).size(),
               3u);

  grid.remove(n2);
  ATM_CHECK_EQ(grid.queryRect(last, last, last + c - 1, last + c - 1).size(),
               2u);

  const auto &circle =
      grid.queryCircle(last + 0.6f * c, last + 0.6f * c, 0.1f * c);
  ATM_CHECK_EQ(circle.size(), 2u);
}

ATM_TEST(grid_negative_positions_match_legacy_behaviour) {
  // The original engine clamped negative coordinates into the last cell;
  // the grid keeps that so existing games see identical results.
  SpatialGrid grid(1000, 1000, 100, 16, 0);
  const float c = grid.cellSize();
  const float last = (grid.cellsWide() - 1) * c;
  grid.add({0, 7}, -3.0f * c, -3.0f * c);
  ATM_CHECK_EQ(grid.queryRect(0, 0, c - 1, c - 1).size(), 0u);
  ATM_CHECK_EQ(grid.queryRect(last, last, last + c - 1, last + c - 1).size(),
               1u);
}

ATM_TEST(grid_node_reuse_after_remove) {
  SpatialGrid grid(1000, 1000, 100, 16, 0);
  const float c = grid.cellSize();
  const int32_t a = grid.add({0, 1}, 0.1f * c, 0.1f * c);
  grid.remove(a);
  const int32_t b = grid.add({0, 2}, 0.1f * c, 0.1f * c);
  ATM_CHECK_EQ(a, b);
  ATM_CHECK_EQ(grid.queryRect(0, 0, c - 1, c - 1).size(), 1u);
}

// SECTION: engine

ATM_TEST(engine_config_sizes_grid) {
  EngineConfig cfg;
  cfg.world_width = 2048;
  cfg.world_height = 1024;
  cfg.grid_cell_size = 128;
  Engine *engine = make_engine(cfg);
  ATM_REQUIRE(engine);
#if ATM_RUNTIME_GRID
  ATM_CHECK_EQ(engine->grid.cellsWide(), 16);
  ATM_CHECK_EQ(engine->grid.cellsHigh(), 8);
#else
  ATM_CHECK_EQ(engine->grid.cellsWide(), static_cast<int32_t>(GRID_CELL_WIDTH));
#endif
  ATM_CHECK_EQ(engine->config.grid_cell_size, 128);
  engine_destroy(engine);
}

ATM_TEST(engine_legacy_create_keeps_old_grid) {
  Engine *engine = engine_create(320, 240, 1000, 1000, 16);
  ATM_REQUIRE(engine);
  ATM_CHECK_EQ(engine->grid.cellsWide(), static_cast<int32_t>(GRID_CELL_WIDTH));
  ATM_CHECK_EQ(engine->config.fixed_timestep_hz, 0.0f);
  engine_destroy(engine);
}

ATM_TEST(engine_entity_lifecycle) {
  Engine *engine = make_engine();
  ATM_REQUIRE(engine);
  auto *c = new TestContainer();
  const int type = engine_register_dynamic_type(engine, c);
  ATM_REQUIRE(type >= 0);

  const EntityHandle e = engine_create_entity(engine, type);
  ATM_CHECK(engine_is_handle_valid(engine, e, type));
  ATM_CHECK(engine_set_entity_position(engine, e, type, 100, 100));
  ATM_CHECK(engine_set_entity_visible(engine, e, type, true));
  ATM_CHECK(engine_set_entity_rotation(engine, e, type, 0.5f));
  ATM_CHECK_EQ(c->rotations[c->getSlot(e)], 0.5f);
  ATM_CHECK_EQ(engine->grid.queryRect(90, 90, 110, 110).size(), 1u);

  engine_destroy_entity(engine, e, type);
  ATM_CHECK(!engine_is_handle_valid(engine, e, type));
  ATM_CHECK_EQ(engine->grid.queryRect(90, 90, 110, 110).size(), 0u);
  engine_destroy(engine);
}

ATM_TEST(engine_variable_timestep_updates_once) {
  Engine *engine = make_engine();
  ATM_REQUIRE(engine);
  auto *c = new TestContainer();
  engine_register_dynamic_type(engine, c);

  engine->last_frame_time =
      SDL_GetPerformanceCounter() - SDL_GetPerformanceFrequency() / 50; // 20ms
  engine_update(engine);
  ATM_CHECK_EQ(c->updates, 1);
  ATM_CHECK_EQ(engine->sim_steps, 1);
  ATM_CHECK(c->last_dt > 0.015f && c->last_dt < 0.1f);
  engine_destroy(engine);
}

ATM_TEST(engine_fixed_timestep_runs_whole_steps) {
  EngineConfig cfg;
  cfg.fixed_timestep_hz = 100.0f; // 10ms steps
  cfg.max_fixed_steps_per_frame = 4;
  Engine *engine = make_engine(cfg);
  ATM_REQUIRE(engine);
  auto *c = new TestContainer();
  engine_register_dynamic_type(engine, c);

  const Uint64 freq = SDL_GetPerformanceFrequency();
  engine->last_frame_time = SDL_GetPerformanceCounter() - freq * 35 / 1000;
  engine_update(engine); // ~35ms -> 3 steps, ~5ms carried over
  ATM_CHECK(engine->sim_steps == 3 || engine->sim_steps == 4);
  ATM_CHECK_EQ(c->updates, engine->sim_steps);
  ATM_CHECK_NEAR(c->last_dt, 0.01f, 1e-6);
  ATM_CHECK(engine->fixed_alpha >= 0.0f && engine->fixed_alpha < 1.0f);

  // A long stall is capped and the backlog dropped (no spiral of death).
  c->updates = 0;
  engine->last_frame_time = SDL_GetPerformanceCounter() - freq / 10; // 100ms
  engine_update(engine);
  ATM_CHECK_EQ(c->updates, 4);
  ATM_CHECK(engine->fixed_accumulator < 0.01f);
  engine_destroy(engine);
}

ATM_TEST(render_batches_rotated_quads_only_when_opted_in) {
  RenderBatch batch(0, 0, 16);
  batch.addQuadRotated(0, 0, 10, 20, 1.57079632679f, {0, 0, 1, 1});
  ATM_REQUIRE(batch.vertices.size() == 4);
  // 90 degrees around the centre (5, 10): top-left (-5,-10) -> (10, -5) + c.
  ATM_CHECK_NEAR(batch.vertices[0].position.x, 15.0f, 1e-4);
  ATM_CHECK_NEAR(batch.vertices[0].position.y, 5.0f, 1e-4);
  ATM_CHECK_EQ(batch.indices.size(), 6u);

  Engine *engine = make_engine();
  ATM_REQUIRE(engine);
  SDL_Surface *surface = SDL_CreateSurface(4, 4, SDL_PIXELFORMAT_RGBA32);
  const int tex = engine_register_texture(engine, surface, 0, 0, 0, 0);
  SDL_DestroySurface(surface);
  ATM_REQUIRE(tex >= 0);

  auto *c = new TestContainer();
  const int type = engine_register_dynamic_type(engine, c);
  const EntityHandle e = engine_create_entity(engine, type);
  const uint32_t slot = c->getSlot(e);
  c->widths[slot] = 10;
  c->heights[slot] = 10;
  c->texture_ids[slot] = static_cast<int16_t>(tex);
  engine_set_entity_position(engine, e, type, 100, 100);
  engine_set_entity_visible(engine, e, type, true);
  engine_set_entity_rotation(engine, e, type, 0.3f);

  engine->camera.x = 100;
  engine->camera.y = 100;
  engine_render_scene(engine);
  const auto &plain = engine->renderBatchManager.getBatches();
  ATM_REQUIRE(!plain.empty());
  // Without ROTATABLE the quad stays axis-aligned: v0 and v1 share y.
  ATM_CHECK_EQ(plain[0].vertices[0].position.y, plain[0].vertices[1].position.y);

  c->enableRotation();
  engine_render_scene(engine);
  const auto &rotated = engine->renderBatchManager.getBatches();
  ATM_CHECK(rotated[0].vertices[0].position.y !=
            rotated[0].vertices[1].position.y);
  engine_destroy(engine);
}

ATM_TEST(atlas_reuses_unregistered_ids) {
  Engine *engine = make_engine();
  ATM_REQUIRE(engine);
  SDL_Surface *surface = SDL_CreateSurface(2, 2, SDL_PIXELFORMAT_RGBA32);
  const int a = engine_register_texture(engine, surface, 0, 0, 0, 0);
  const int b = engine_register_texture(engine, surface, 0, 0, 0, 0);
  engine->atlas.unregisterTexture(a);
  ATM_CHECK(engine->atlas.getTexture(a) == nullptr);
  const int c = engine_register_texture(engine, surface, 0, 0, 0, 0);
  ATM_CHECK_EQ(c, a);
  ATM_CHECK(engine->atlas.getTexture(b) != nullptr);
  SDL_DestroySurface(surface);
  engine_destroy(engine);
}

ATM_TEST(static_entities_render_from_chunk_cache) {
  EngineConfig cfg;
  cfg.static_chunk_size = 128;
  Engine *engine = make_engine(cfg);
  ATM_REQUIRE(engine);
  auto *c = new TestContainer();
  const int type = engine_register_static_type(engine, c);
  for (int i = 0; i < 3; ++i) {
    const EntityHandle e = engine_create_entity(engine, type);
    const uint32_t slot = c->getSlot(e);
    c->widths[slot] = 8;
    c->heights[slot] = 8;
    engine_set_entity_position(engine, e, type, 100.0f + i * 20.0f, 100.0f);
    engine_set_entity_visible(engine, e, type, true);
  }
  engine->camera.x = 100;
  engine->camera.y = 100;
  engine_render_scene(engine);
  size_t quads = 0;
  for (const auto &batch : engine->renderBatchManager.getBatches())
    quads += batch.vertices.size() / 4;
  ATM_CHECK_EQ(quads, 3u);
  engine_destroy(engine);
}
