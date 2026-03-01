#include "ATMEngine.h"
#include <SDL3/SDL.h>
#include <cmath>

// ============================================================
// Hello World — First Attome Engine Game
// Draws a colored rectangle that moves in a circle.
// Press ESC or close the window to quit.
// ============================================================

int main(int argc, char *argv[]) {
  Engine *engine = engine_create(1280, 720, 50000, 50000, 64);
  if (!engine) {
    SDL_Log("Failed to create Attome Engine.");
    return 1;
  }

  SDL_SetWindowTitle(engine->window, "Hello World — Attome Engine");
  SDL_Log("Hello from Attome Engine!");

  bool running = true;
  float deltaTime = 0.0f;
  float time = 0.0f;

  while (running) {
    Uint64 frameStart = SDL_GetTicks();

    // ── events ────────────────────────────────────────────
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_QUIT)
        running = false;
      if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE)
        running = false;
    }

    // ── update ────────────────────────────────────────────
    time += deltaTime;
    engine_update(engine);

    // ── render ────────────────────────────────────────────
    SDL_SetRenderDrawColor(engine->renderer, 10, 10, 20, 255);
    SDL_RenderClear(engine->renderer);

    // Draw a pink box bouncing around the screen
    float cx = 640.0f + std::cos(time) * 250.0f;
    float cy = 360.0f + std::sin(time * 1.3f) * 150.0f;
    SDL_FRect box = {cx - 50, cy - 50, 100, 100};
    SDL_SetRenderDrawColor(engine->renderer, 236, 72, 153, 255); // pink
    SDL_RenderFillRect(engine->renderer, &box);

    // Draw a teal border
    SDL_SetRenderDrawColor(engine->renderer, 20, 184, 166, 255); // teal
    SDL_RenderRect(engine->renderer, &box);

    engine_render_scene(engine);
    engine_present(engine);

    // ── timing ────────────────────────────────────────────
    Uint64 frameEnd = SDL_GetTicks();
    deltaTime = (frameEnd - frameStart) / 1000.0f;
    if (deltaTime < 0.001f)
      deltaTime = 0.001f;
  }

  SDL_Log("Goodbye from Attome Engine!");
  engine_destroy(engine);
  return 0;
}
