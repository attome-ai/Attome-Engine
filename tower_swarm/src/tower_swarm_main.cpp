#include "Constants.h"
#include "InputManager.h"
#include "TowerSwarmGame.h"

#include "ATMEngine.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

#include <algorithm>

namespace {

struct TowerSwarmRuntime final {
  Engine *engine = nullptr;
  tower_swarm::TowerSwarmGame *game = nullptr;
  tower_swarm::InputManager input;
  bool running = true;
  Uint64 last_counter = 0;
  Uint64 counter_freq = 0;
  float fps_accum_sec = 0.0f;
  std::uint32_t fps_frames = 0;
};

float compute_dt_sec(Uint64 now_counter, Uint64 freq, Uint64 &last_counter) {
  if (last_counter == 0 || freq == 0) {
    last_counter = now_counter;
    return tower_swarm::kDefaultFrameDtSec;
  }
  const float dt =
      static_cast<float>(now_counter - last_counter) / static_cast<float>(freq);
  last_counter = now_counter;
  return std::clamp(dt, 0.0f, tower_swarm::kMaxFrameDtSec);
}

void shutdown_runtime(TowerSwarmRuntime &rt) {
  if (rt.game) {
    delete rt.game;
    rt.game = nullptr;
  }
  if (rt.engine) {
    engine_destroy(rt.engine);
    rt.engine = nullptr;
  }
}

void run_frame(TowerSwarmRuntime &rt) {
  const Uint64 frame_start = SDL_GetPerformanceCounter();
  const Uint64 now = frame_start;
  const float dt = compute_dt_sec(now, rt.counter_freq, rt.last_counter);

  rt.fps_accum_sec += std::max(0.0f, dt);
  rt.fps_frames += 1;
  if (rt.fps_accum_sec >= 1.0f) {
    const float fps =
        (rt.fps_accum_sec > 0.0f)
            ? (static_cast<float>(rt.fps_frames) / rt.fps_accum_sec)
            : 0.0f;
    SDL_Log("TowerSwarm FPS: %.1f (%.2f ms)", fps,
            (fps > 0.0f) ? (1000.0f / fps) : 0.0f);
    rt.fps_accum_sec = 0.0f;
    rt.fps_frames = 0;
  }

  rt.input.beginFrame();

  SDL_Event ev;
  while (SDL_PollEvent(&ev)) {
    rt.input.handleEvent(ev);

    if (ev.type == SDL_EVENT_QUIT) {
      rt.running = false;
    } else if (ev.type == SDL_EVENT_KEY_DOWN &&
               ev.key.scancode == SDL_SCANCODE_ESCAPE) {
      rt.running = false;
    }
  }

  if (rt.game) {
    rt.game->tick(dt, rt.input);
  }

  engine_update(rt.engine);

  SDL_SetRenderDrawColor(rt.engine->renderer, tower_swarm::kClearColor.r,
                         tower_swarm::kClearColor.g,
                         tower_swarm::kClearColor.b,
                         tower_swarm::kClearColor.a);
  SDL_RenderClear(rt.engine->renderer);

  engine_render_scene(rt.engine);

  if (rt.game) {
    rt.game->renderHUD(rt.input);
  }

  engine_present(rt.engine);

#if !defined(__EMSCRIPTEN__)
  // Cap at 120 FPS to reduce CPU usage and keep frame pacing stable.
  if (rt.counter_freq > 0) {
    constexpr double kTargetFrameSec = 1.0 / 120.0;
    const Uint64 frame_end = SDL_GetPerformanceCounter();
    const double elapsed_sec =
        static_cast<double>(frame_end - frame_start) /
        static_cast<double>(rt.counter_freq);
    const double remaining_sec = kTargetFrameSec - elapsed_sec;
    if (remaining_sec > 0.0) {
      const Uint64 remaining_ns =
          static_cast<Uint64>(remaining_sec * 1000.0 * 1000.0 * 1000.0);
      SDL_DelayPrecise(remaining_ns);
    }
  }
#endif
}

#if defined(__EMSCRIPTEN__)
void web_main_loop(void *arg) {
  auto *rt = static_cast<TowerSwarmRuntime *>(arg);
  if (!rt || !rt->running) {
    emscripten_cancel_main_loop();
    if (rt) {
      shutdown_runtime(*rt);
    }
    SDL_Quit();
    return;
  }
  run_frame(*rt);
}
#endif

} // namespace

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  TowerSwarmRuntime rt;
  rt.counter_freq = SDL_GetPerformanceFrequency();
  rt.engine = engine_create(tower_swarm::kWindowWidthPx,
                            tower_swarm::kWindowHeightPx,
                            tower_swarm::kWorldWidthPx,
                            tower_swarm::kWorldHeightPx, tower_swarm::kTileSizePx);
  if (!rt.engine) {
    SDL_Log("TowerSwarm: failed to create engine.");
    return 1;
  }

  rt.game = new tower_swarm::TowerSwarmGame(rt.engine);
  if (!rt.game->initialize()) {
    SDL_Log("TowerSwarm: failed to initialize game.");
    shutdown_runtime(rt);
    SDL_Quit();
    return 1;
  }

#if defined(__EMSCRIPTEN__)
  emscripten_set_main_loop_arg(web_main_loop, &rt, 0, true);
#else
  while (rt.running) {
    run_frame(rt);
  }
  shutdown_runtime(rt);
  SDL_Quit();
#endif

  return 0;
}
