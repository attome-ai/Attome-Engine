#include "ATMEngine.h"
#include <SDL3/SDL.h>

// ============================================================
// Attome Engine — Game Template
// Replace this file with your actual game logic.
//
// ATMEngine API reference:
//   engine_create(w, h, world_w, world_h, cell_size)
//   engine_update(engine*)           — process input / delta time
//   engine_update_entity_types(e,dt) — update your entity containers
//   engine_render_scene(engine*)     — batch-render all visible entities
//   engine_present(engine*)          — flip the renderer buffer
//   engine_destroy(engine*)          — cleanup
// ============================================================

int main(int argc, char *argv[]) {
  // Create a 1280×720 window with a 50000×50000 world, 64px grid cells
  Engine *engine = engine_create(1280, 720, 50000, 50000, 64);
  if (!engine) {
    SDL_Log("Failed to create Attome Engine.");
    return 1;
  }

  SDL_SetWindowTitle(engine->window, "My Attome Game");

  bool running = true;
  float deltaTime = 0.0f;

  while (running) {
    Uint64 frameStart = SDL_GetTicks();

    // ── event handling ────────────────────────────────────
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_QUIT)
        running = false;
      if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE)
        running = false;
    }

    // ── update ────────────────────────────────────────────
    engine_update(engine);
    engine_update_entity_types(engine, deltaTime);
    // TODO: add your game logic here

    // ── render ────────────────────────────────────────────
    SDL_SetRenderDrawColor(engine->renderer, 10, 10, 15, 255);
    SDL_RenderClear(engine->renderer);

    engine_render_scene(engine);
    // TODO: add debug/UI drawing here

    engine_present(engine);

    // ── timing ────────────────────────────────────────────
    Uint64 frameEnd = SDL_GetTicks();
    deltaTime = (frameEnd - frameStart) / 1000.0f;
    if (deltaTime < 0.001f)
      deltaTime = 0.001f;
  }

  engine_destroy(engine);
  return 0;
}
