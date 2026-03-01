#include "../engine/ATMEngine.h"
#include "AshlandsGame.h"
#include <SDL3/SDL_main.h>
#include <iostream>

int main(int argc, char *argv[]) {
  // Initialize the Engine with enough bounds for the 19000 hex map
  Engine *engine = engine_create(1280, 720, 20000, 20000, 256);
  if (!engine) {
    std::cerr << "Failed to initialize engine.\n";
    return -1;
  }

  // Register simple renderable entity type (Type 0)
  TileContainer *renderContainer = new TileContainer(0, 0, 30000);
  engine->entityManager.registerEntityType(renderContainer);

  // Register SettlementContainer (Type 1)
  SettlementContainer *settlementContainer =
      new SettlementContainer(1, 0, 10000);
  engine->entityManager.registerEntityType(settlementContainer);

  // Register UnitContainer (Type 2)
  UnitContainer *unitContainer = new UnitContainer(2, 1, 1000);
  engine->entityManager.registerEntityType(unitContainer);

  AshlandsGame game(engine);
  game.initialize();

  bool running = true;
  SDL_Event event;

  // Basic game loop
  while (running) {
    Uint64 start = SDL_GetPerformanceCounter();

    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_QUIT) {
        running = false;
      }
    }

    // Calculate delta time
    Uint64 end = SDL_GetPerformanceCounter();
    float dt = (end - start) / (float)SDL_GetPerformanceFrequency();
    engine->last_frame_time = end;
    if (dt == 0.0f)
      dt = 1.0f / 60.0f;

    game.update(dt);

    // engine_update_entity_types and core systems
    engine_update(engine);

    // Render
    engine_render_scene(engine);
    engine_present(engine);
  }

  engine_destroy(engine);
  return 0;
}
