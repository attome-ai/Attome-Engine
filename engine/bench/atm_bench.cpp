// Engine micro-benchmarks. Build in Release and run:
//   atm_bench            (all)
//   atm_bench grid       (only benchmarks whose name contains "grid")
//
// Each benchmark runs several repetitions and reports the fastest, which is
// the least noisy number on a desktop machine.

#include "ATMConfig.h"
#include "ATMEngine.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace {

// Same shape as a game tunable, to compare against a compile-time constant.
namespace bench_tunables {
ATM_TUNABLE_SECTION("bench");
ATM_TUNABLE(float, kSpeed, 120.0f);
} // namespace bench_tunables
constexpr float kConstexprSpeed = 120.0f;

volatile float g_sink = 0.0f; // keeps results observable

struct Result {
  const char *name;
  double ns_per_op;
};

std::vector<Result> g_results;
const char *g_filter = nullptr;

void bench(const char *name, int ops, const std::function<void()> &setup,
           const std::function<void()> &body, int reps = 7) {
  if (g_filter && !std::strstr(name, g_filter))
    return;
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    if (setup)
      setup();
    const auto t0 = std::chrono::steady_clock::now();
    body();
    const auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
  }
  const double per_op = best / ops;
  g_results.push_back({name, per_op});
  std::printf("%-44s %10.2f ns/op   (%d ops, best of %d)\n", name, per_op, ops,
              reps);
}

class BenchContainer : public RenderableEntityContainer {
public:
  explicit BenchContainer(int capacity)
      : RenderableEntityContainer(0, 0, capacity) {}
  void update(float) override {}
};

Engine *make_engine() {
  EngineConfig cfg;
  cfg.window_width = 1280;
  cfg.window_height = 720;
  return engine_create_with_config(cfg);
}

} // namespace

int main(int argc, char **argv) {
  g_filter = argc > 1 ? argv[1] : nullptr;
  SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "dummy");
  SDL_SetHint(SDL_HINT_RENDER_DRIVER, "software");
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    std::printf("SDL_Init failed: %s\n", SDL_GetError());
    return 1;
  }

#if !defined(NDEBUG)
  std::printf("WARNING: debug build - numbers are not representative.\n\n");
#endif

  constexpr int kEntities = 100000;
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> pos(0.0f, 20000.0f);

  // --- entity churn --------------------------------------------------------
  {
    BenchContainer c(kEntities);
    std::vector<EntityHandle> ids(kEntities);
    bench("entity create+remove", kEntities * 2, [&] {},
          [&] {
            for (int i = 0; i < kEntities; ++i)
              ids[i] = c.createEntity();
            for (int i = kEntities - 1; i >= 0; i -= 2)
              c.removeEntity(ids[i]);
            for (int i = kEntities - 2; i >= 0; i -= 2)
              c.removeEntity(ids[i]);
          });
  }

  // --- spatial grid -----------------------------------------------------------
  {
    std::vector<float> xs(kEntities), ys(kEntities);
    for (int i = 0; i < kEntities; ++i) {
      xs[i] = pos(rng);
      ys[i] = pos(rng);
    }

    auto run_grid = [&](const char *move_name, const char *query_name,
                        SpatialGrid &grid) {
      std::vector<int32_t> nodes(kEntities);
      for (int i = 0; i < kEntities; ++i)
        nodes[i] = grid.add({0, static_cast<EntityHandle>(i)}, xs[i], ys[i]);
      bench(move_name, kEntities, nullptr, [&] {
        for (int i = 0; i < kEntities; ++i) {
          xs[i] += 7.0f;
          if (xs[i] > 20000.0f)
            xs[i] -= 20000.0f;
          grid.move(nodes[i], xs[i], ys[i]);
        }
      });
      bench(query_name, 1000, nullptr, [&] {
        size_t total = 0;
        for (int q = 0; q < 1000; ++q) {
          const float x = static_cast<float>((q * 97) % 18000);
          const float y = static_cast<float>((q * 61) % 18000);
          total += grid.queryRect(x, y, x + 1280.0f, y + 720.0f).size();
        }
        g_sink = static_cast<float>(total);
      });
    };

    // Legacy constructor (compile-time defaults) vs runtime-configured grid
    // with identical dimensions: shows the runtime config costs nothing.
    SpatialGrid legacy;
    run_grid("grid move 100k (legacy defaults)", "grid queryRect viewport (legacy)",
             legacy);
    SpatialGrid configured(static_cast<int>(WORLD_WIDTH),
                           static_cast<int>(WORLD_HEIGHT),
                           static_cast<int>(GRID_CELL_SIZE), 3200000, 4);
    run_grid("grid move 100k (runtime config)", "grid queryRect viewport (runtime)",
             configured);
  }

  // --- tunable vs constexpr in a hot loop -------------------------------------
  {
    std::vector<float> x(kEntities, 0.0f), vx(kEntities, 1.0f);
    const float dt = 1.0f / 60.0f;
    bench("move loop, constexpr speed", kEntities, nullptr, [&] {
      for (int i = 0; i < kEntities; ++i)
        x[i] += vx[i] * kConstexprSpeed * dt;
      g_sink = x[kEntities / 2];
    });
    bench("move loop, ATM_TUNABLE speed", kEntities, nullptr, [&] {
      for (int i = 0; i < kEntities; ++i)
        x[i] += vx[i] * bench_tunables::kSpeed * dt;
      g_sink = x[kEntities / 2];
    });
    // The fix for hot loops: copy the tunable into a local first, so the
    // compiler knows stores to x[] can't change it and vectorises again.
    bench("move loop, ATM_TUNABLE hoisted to local", kEntities, nullptr, [&] {
      const float speed = bench_tunables::kSpeed;
      for (int i = 0; i < kEntities; ++i)
        x[i] += vx[i] * speed * dt;
      g_sink = x[kEntities / 2];
    });
  }

  // --- render batching ----------------------------------------------------------
  {
    Engine *engine = make_engine();
    if (!engine) {
      std::printf("cannot create engine: %s\n", SDL_GetError());
      return 1;
    }
    SDL_Surface *surface = SDL_CreateSurface(8, 8, SDL_PIXELFORMAT_RGBA32);
    const int tex = engine_register_texture(engine, surface, 0, 0, 0, 0);
    SDL_DestroySurface(surface);

    constexpr int kVisible = 20000;
    auto *c = new BenchContainer(kVisible);
    const int type = engine_register_dynamic_type(engine, c);
    std::uniform_real_distribution<float> screen_x(0.0f, 1270.0f);
    std::uniform_real_distribution<float> screen_y(0.0f, 710.0f);
    for (int i = 0; i < kVisible; ++i) {
      const EntityHandle e = engine_create_entity(engine, type);
      const uint32_t slot = c->getSlot(e);
      c->widths[slot] = 8;
      c->heights[slot] = 8;
      c->texture_ids[slot] = static_cast<int16_t>(tex);
      engine_set_entity_position(engine, e, type, screen_x(rng), screen_y(rng));
      engine_set_entity_visible(engine, e, type, true);
    }
    engine->camera.x = 640;
    engine->camera.y = 360;

    RenderBatch batch(0, 0, kVisible * 4);
    bench("RenderBatch::addQuad", kVisible, [&] { batch.clear(); }, [&] {
      for (int i = 0; i < kVisible; ++i)
        batch.addQuad(static_cast<float>(i % 1280), 10.0f, 8.0f, 8.0f,
                      {0, 0, 1, 1});
    });
    bench("RenderBatch::addQuadRotated", kVisible, [&] { batch.clear(); }, [&] {
      for (int i = 0; i < kVisible; ++i)
        batch.addQuadRotated(static_cast<float>(i % 1280), 10.0f, 8.0f, 8.0f,
                             0.3f, {0, 0, 1, 1});
    });

    // Whole frame (grid query + batching + software rasterisation). The last
    // two show that opting in to rotation costs nothing for unrotated sprites.
    bench("render_scene 20k sprites (no rotation)", kVisible, nullptr,
          [&] { engine_render_scene(engine); }, 5);
    c->enableRotation();
    bench("render_scene 20k (ROTATABLE, angle 0)", kVisible, nullptr,
          [&] { engine_render_scene(engine); }, 5);
    for (int i = 0; i < c->count; ++i)
      c->rotations[i] = 0.3f;
    bench("render_scene 20k (ROTATABLE, rotated)", kVisible, nullptr,
          [&] { engine_render_scene(engine); }, 5);

    engine_destroy(engine);
  }

  SDL_Quit();
  return 0;
}
