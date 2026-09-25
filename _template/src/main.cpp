#include "ATMAssets.h"
#include "ATMAudio.h"
#include "ATMConfig.h"
#include "ATMEngine.h"
#include "ATMInput.h"
#if defined(ATM_HAS_TEXT)
#include "ATMText.h"
#endif

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <string>

// ============================================================
// Attome Engine — Game Template
// Replace this file with your actual game logic.
//
// Everything tweakable lives in config/game.json (copied next to the exe):
//   "engine" — window, world/grid, timestep, audio (read once at startup)
//   "input"  — action bindings (rebind keys without recompiling)
//   other sections — your ATM_TUNABLEs below (edited live while running)
//
// ATMEngine API reference:
//   engine_create_with_config(cfg)   — window + renderer + grid from config
//   engine_update(engine*)           — runs container updates (fixed or
//                                      variable timestep, see config)
//   engine_render_scene(engine*)     — batch-render all visible entities
//   engine_present(engine*)          — flip the renderer buffer
//   engine_destroy(engine*)          — cleanup
// ============================================================

namespace game {
ATM_TUNABLE_SECTION("game");
ATM_TUNABLE(float, kPlayerSpeed, 400.0f);
ATM_TUNABLE(int, kClearR, 10);
ATM_TUNABLE(int, kClearG, 10);
ATM_TUNABLE(int, kClearB, 15);
} // namespace game

int main(int argc, char *argv[]) {
  std::string config_path = atm::resolve_path("config/game.json");
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--config")
      config_path = argv[i + 1];
  }

  // ── configuration ─────────────────────────────────────────
  atm::Json config;
  std::string error;
  if (!atm::Json::parseFile(config_path, config, &error)) {
    SDL_Log("Using built-in defaults (%s)", error.c_str());
    config = atm::Json::object();
  }

  EngineConfig engine_config;
  engine_config.window_title = "My Attome Game";
  if (const atm::Json *section = config.find("engine"))
    engine_config_from_json(*section, engine_config);
  if (!config.asObject().empty())
    atm::Tunables::instance().loadFile(config_path); // applies + live reload

  Engine *engine = engine_create_with_config(engine_config);
  if (!engine) {
    SDL_Log("Failed to create Attome Engine.");
    return 1;
  }

  // ── input ─────────────────────────────────────────────────
  atm::InputMap input;
  input.bind("quit", "Escape");
  input.bind("left", "A");
  input.bind("left", "Left");
  input.bind("right", "D");
  input.bind("right", "Right");
  input.defineAxis("move_x", "left", "right");
  if (const atm::Json *section = config.find("input"))
    input.loadBindings(*section); // JSON overrides the defaults above
  const atm::ActionId quit = input.actionId("quit");

  // ── audio / assets ────────────────────────────────────────
  atm::Audio audio;
  audio.init(engine->config); // silently skipped when audio.enabled = false
  atm::Assets assets(engine, &audio);
  // int ship_tex = assets.acquireTexture("resource/ship.png");
  // atm::SoundId blip = assets.acquireSound("sfx/blip.wav");

#if defined(ATM_HAS_TEXT)
  atm::Font *font = assets.acquireFont("assets/fonts/AtomicMd.ttf", 18.0f);
#endif

  float player_x = engine_config.window_width * 0.5f;
  bool running = true;
  while (running) {
    // ── events ──────────────────────────────────────────────
    input.beginFrame();
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      input.handleEvent(event);
      if (event.type == SDL_EVENT_QUIT)
        running = false;
    }
    if (input.pressed(quit))
      running = false;

    // Picks up edits to config/game.json while the game runs.
    atm::Tunables::instance().reloadIfChanged();

    // ── update ──────────────────────────────────────────────
    engine_update(engine);
    player_x += input.axis("move_x") * game::kPlayerSpeed * engine->frame_dt;
    // TODO: add your game logic here

    // ── render ──────────────────────────────────────────────
    SDL_SetRenderDrawColor(engine->renderer, static_cast<Uint8>(game::kClearR),
                           static_cast<Uint8>(game::kClearG),
                           static_cast<Uint8>(game::kClearB), 255);
    SDL_RenderClear(engine->renderer);

    engine_render_scene(engine);

    SDL_SetRenderDrawColor(engine->renderer, 240, 240, 255, 255);
    const SDL_FRect player{player_x - 16.0f,
                           engine_config.window_height * 0.5f - 16.0f, 32.0f,
                           32.0f};
    SDL_RenderFillRect(engine->renderer, &player);

#if defined(ATM_HAS_TEXT)
    if (font)
      font->draw(engine->renderer, "A/D to move - Esc to quit", 12, 12);
#endif

    engine_present(engine);
  }

  assets.releaseAll();
  audio.shutdown();
  engine_destroy(engine);
  SDL_Quit();
  return 0;
}
