// Before/after check: uses only the original public API, so the same file
// compiles against the previous engine (git history) and the current one.

#include "ATMEngine.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

namespace {

class C : public RenderableEntityContainer {
public:
  explicit C(int n) : RenderableEntityContainer(0, 0, n) {}
  void update(float) override {}
};

template <typename F> double best_ns(int reps, F &&f) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
  }
  return best;
}

} // namespace

int main(int, char **) {
  SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "dummy");
  SDL_SetHint(SDL_HINT_RENDER_DRIVER, "software");
  SDL_Init(SDL_INIT_VIDEO);
  Engine *engine = engine_create(1280, 720, 50000, 50000, 64);

  SDL_Surface *surface = SDL_CreateSurface(8, 8, SDL_PIXELFORMAT_RGBA32);
  const int tex = engine_register_texture(engine, surface, 0, 0, 0, 0);
  SDL_DestroySurface(surface);

  constexpr int kVisible = 20000;
  constexpr int kMoving = 100000;
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> sx(0.0f, 1270.0f), sy(0.0f, 710.0f);
  std::uniform_real_distribution<float> wx(0.0f, 20000.0f);

  auto *visible = new C(kVisible);
  const int vtype = engine_register_dynamic_type(engine, visible);
  for (int i = 0; i < kVisible; ++i) {
    const EntityHandle e = engine_create_entity(engine, vtype);
    const uint32_t slot = visible->getSlot(e);
    visible->widths[slot] = 8;
    visible->heights[slot] = 8;
    visible->texture_ids[slot] = static_cast<int16_t>(tex);
    engine_set_entity_position(engine, e, vtype, sx(rng), sy(rng));
    engine_set_entity_visible(engine, e, vtype, true);
  }

  auto *moving = new C(kMoving);
  const int mtype = engine_register_dynamic_type(engine, moving);
  std::vector<EntityHandle> ids(kMoving);
  std::vector<float> xs(kMoving), ys(kMoving);
  for (int i = 0; i < kMoving; ++i) {
    ids[i] = engine_create_entity(engine, mtype);
    xs[i] = 2000.0f + wx(rng);
    ys[i] = 2000.0f + wx(rng);
    engine_set_entity_position(engine, ids[i], mtype, xs[i], ys[i]);
  }

  engine->camera.x = 640;
  engine->camera.y = 360;

  const double render = best_ns(15, [&] { engine_render_scene(engine); });
  const double move = best_ns(15, [&] {
    for (int i = 0; i < kMoving; ++i) {
      xs[i] += 7.0f;
      engine_set_entity_position(engine, ids[i], mtype, xs[i], ys[i]);
    }
  });
  const double update = best_ns(15, [&] { engine_update(engine); });

  std::printf("render_scene 20k sprites     %10.1f us\n", render / 1000.0);
  std::printf("set_position 100k entities   %10.1f us\n", move / 1000.0);
  std::printf("engine_update (120k ents)    %10.1f us\n", update / 1000.0);

  engine_destroy(engine);
  SDL_Quit();
  return 0;
}
