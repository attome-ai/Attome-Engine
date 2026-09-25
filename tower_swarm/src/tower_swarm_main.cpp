#include "Constants.h"
#include "InputManager.h"
#include "TowerSwarmGame.h"

#include "ATMConfig.h"
#include "ATMEngine.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

#include <algorithm>
#include <cstring>
#include <string>

namespace {

// Relative paths are resolved against the working directory first, then the
// executable directory (atm::resolve_path). On web the file is preloaded into
// the virtual FS at the same path (see build_web.ps1).
constexpr const char *kDefaultConfigPath = "config/tower_swarm.json";

struct CommandLine final {
  std::string config_path = kDefaultConfigPath;
  std::string write_default_config_path; // empty = run the game
};

bool parse_command_line(int argc, char *argv[], CommandLine &out) {
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    const bool has_value = (i + 1) < argc;
    if (std::strcmp(arg, "--config") == 0 && has_value) {
      out.config_path = argv[++i];
    } else if (std::strcmp(arg, "--write-default-config") == 0 && has_value) {
      out.write_default_config_path = argv[++i];
    } else if (std::strcmp(arg, "--config") == 0 ||
               std::strcmp(arg, "--write-default-config") == 0) {
      SDL_Log("TowerSwarm: %s needs a file path.", arg);
      return false;
    }
    // Anything else is ignored (platform launchers may add their own args).
  }
  return true;
}

// Engine settings for this game. Window and world size come from the game
// tunables (general.kWindowWidthPx, ...), so they have a single source of
// truth; the optional "engine" section of the config file can override the
// rest (grid, timestep, vsync, audio). Everything else keeps the EngineConfig
// defaults, which match the legacy engine_create() setup (64px grid cells,
// variable timestep).
EngineConfig make_engine_config(const atm::Json *engine_section) {
  EngineConfig cfg;
  if (engine_section) {
    std::string error;
    if (!engine_config_from_json(*engine_section, cfg, &error)) {
      SDL_Log("TowerSwarm: config \"engine\" section: %s (keeping defaults "
              "for the bad values).",
              error.c_str());
    }
    if (engine_section->findPath("window.width") ||
        engine_section->findPath("window.height") ||
        engine_section->find("world")) {
      SDL_Log("TowerSwarm: engine.window.width/height and engine.world.* are "
              "ignored; set general.kWindowWidthPx/kWindowHeightPx/"
              "kWorldWidthPx/kWorldHeightPx instead.");
    }
  }
  cfg.window_width = tower_swarm::kWindowWidthPx;
  cfg.window_height = tower_swarm::kWindowHeightPx;
  cfg.world_width = tower_swarm::kWorldWidthPx;
  cfg.world_height = tower_swarm::kWorldHeightPx;
  return cfg;
}

// Writes every tunable with its compiled-in default plus the engine section.
bool write_default_config(const std::string &path) {
  atm::Json root = atm::Tunables::instance().toJson();
  atm::Json engine = engine_config_to_json(make_engine_config(nullptr));
  // Window/world size live in "general"; the title is set by the game.
  atm::Json::Object &sections = engine.asObject();
  sections.erase("world");
  if (auto window = sections.find("window"); window != sections.end()) {
    window->second.asObject().erase("width");
    window->second.asObject().erase("height");
    window->second.asObject().erase("title");
  }
  root["engine"] = std::move(engine);
  if (!atm::write_text_file(path, root.dump(2))) {
    SDL_Log("TowerSwarm: could not write '%s': %s", path.c_str(),
            SDL_GetError());
    return false;
  }
  SDL_Log("TowerSwarm: wrote %zu tunables + engine settings to '%s'.",
          atm::Tunables::instance().count(), path.c_str());
  return true;
}

// Loads the game config. Missing or broken files are not fatal: the game runs
// with the compiled-in defaults. Returns the parsed document (for the
// "engine" section), or a null Json when nothing usable was read.
atm::Json load_config(const std::string &requested_path) {
  const std::string path = atm::resolve_path(requested_path);
  if (!SDL_GetPathInfo(path.c_str(), nullptr)) {
    SDL_Log("TowerSwarm: no config at '%s'; using compiled defaults.",
            path.c_str());
    return atm::Json();
  }

  atm::Tunables &tunables = atm::Tunables::instance();
  std::string error;
  if (!tunables.loadFile(path, &error)) {
    if (tunables.loadedPath().empty()) {
      // Could not read/parse the file at all: nothing was applied.
      SDL_Log("TowerSwarm: failed to load config '%s': %s. Using compiled "
              "defaults; fix the file and restart.",
              path.c_str(), error.c_str());
      return atm::Json();
    }
    SDL_Log("TowerSwarm: config '%s' has invalid values (%s); those keep "
            "their defaults.",
            path.c_str(), error.c_str());
  }
  SDL_Log("TowerSwarm: loaded config '%s' (%zu tunables, live reload on).",
          path.c_str(), tunables.count());

  atm::Json root;
  if (!atm::Json::parseFile(path, root, &error)) {
    return atm::Json();
  }
  return root;
}

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
  // Picks up edits to the config file (checks its timestamp twice a second;
  // the engine logs "[tunables] reloaded ..." when it applies a change).
  atm::Tunables::instance().reloadIfChanged();

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
  CommandLine cli;
  if (!parse_command_line(argc, argv, cli)) {
    return 2;
  }

  // Tool mode: dump the defaults and exit without opening a window.
  if (!cli.write_default_config_path.empty()) {
    return write_default_config(cli.write_default_config_path) ? 0 : 1;
  }

  const atm::Json config = load_config(cli.config_path);
  const EngineConfig engine_config =
      make_engine_config(config.isObject() ? config.find("engine") : nullptr);

  TowerSwarmRuntime rt;
  rt.counter_freq = SDL_GetPerformanceFrequency();
  rt.engine = engine_create_with_config(engine_config);
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
