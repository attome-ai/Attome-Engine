#include "MeteorDodgeConstants.h"

#include "../../engine/ATMEngine.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <utility>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

namespace {

enum GameEntityType {
  ENTITY_TYPE_PLAYER = 0,
  ENTITY_TYPE_METEOR,
  ENTITY_TYPE_COUNT
};

float randomFloat(float minValue, float maxValue) {
  const float t = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  return minValue + t * (maxValue - minValue);
}

SDL_Surface *create_colored_surface(int width, int height, Uint8 r, Uint8 g, Uint8 b) {
  SDL_Surface *surface = SDL_CreateSurface(width, height, SDL_PIXELFORMAT_RGBA8888);
  if (!surface) {
    return nullptr;
  }

  const Uint32 color = SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format), nullptr, r, g, b, 255);
  SDL_FillSurfaceRect(surface, nullptr, color);
  return surface;
}

SDL_Surface *load_texture(const char *filename) {
  (void)filename;
  return nullptr;
}

class PlayerContainer final : public RenderableEntityContainer {
public:
  PlayerContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity) {}

  EntityHandle createPlayer(float x, float y, int textureId) {
    EntityHandle id = RenderableEntityContainer::createEntity();
    if (id == INVALID_ID) {
      return INVALID_ID;
    }

    const uint32_t slot = getSlot(id);
    x_positions[slot] = x;
    y_positions[slot] = y;
    widths[slot] = meteor_dodge::PLAYER_WIDTH;
    heights[slot] = meteor_dodge::PLAYER_HEIGHT;
    texture_ids[slot] = textureId;
    z_indices[slot] = 100;
    flags[slot] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    return id;
  }

  void update(float deltaTime) override {
    (void)deltaTime;
    for (int i = 0; i < count; i++) {
      flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    }
  }
};

class MeteorContainer final : public RenderableEntityContainer {
public:
  DynamicArray<float> speeds;
  DynamicArray<float> drifts;

  MeteorContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity),
        speeds(initialCapacity, meteor_dodge::METEOR_MIN_SPEED),
        drifts(initialCapacity, 0.0f) {}

  EntityHandle createMeteor(float x, float y, float speed, float drift, int textureId) {
    EntityHandle id = RenderableEntityContainer::createEntity();
    if (id == INVALID_ID) {
      return INVALID_ID;
    }

    const uint32_t slot = getSlot(id);
    x_positions[slot] = x;
    y_positions[slot] = y;
    widths[slot] = meteor_dodge::METEOR_WIDTH;
    heights[slot] = meteor_dodge::METEOR_HEIGHT;
    texture_ids[slot] = textureId;
    z_indices[slot] = 50;
    speeds[slot] = speed;
    drifts[slot] = drift;
    flags[slot] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    return id;
  }

  void update(float deltaTime) override {
    (void)deltaTime;
    for (int i = 0; i < count; i++) {
      flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    }
  }

protected:
  void swapSlots(uint32_t a, uint32_t b) override {
    if (a == b) {
      return;
    }

    std::swap(speeds[a], speeds[b]);
    std::swap(drifts[a], drifts[b]);
    RenderableEntityContainer::swapSlots(a, b);
  }

  void resizeArrays(int newCapacity) override {
    if (newCapacity <= capacity) {
      return;
    }

    speeds.resize(newCapacity, count, meteor_dodge::METEOR_MIN_SPEED);
    drifts.resize(newCapacity, count, 0.0f);
    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

struct MeteorDodgeRuntime {
  Engine *engine = nullptr;
  PlayerContainer *players = nullptr;
  MeteorContainer *meteors = nullptr;

  EntityHandle playerId = INVALID_ID;

  int playerTextureId = -1;
  int meteorTextureId = -1;

  bool running = true;
  bool gameOver = false;

  int score = 0;
  int bestScore = 0;

  Uint64 lastFrameTicks = 0;
  Uint64 lastTitleTicks = 0;
};

void moveEntityWithGrid(Engine *engine, RenderableEntityContainer *container, uint32_t slot, float newX, float newY) {
  container->x_positions[slot] = newX;
  container->y_positions[slot] = newY;

  const int32_t nodeIndex = container->grid_node_indices[slot];
  if (nodeIndex != -1) {
    engine->grid.move(nodeIndex, newX, newY);
  }

  container->cell_x[slot] = static_cast<uint16_t>(newX * INV_GRID_CELL_SIZE);
  container->cell_y[slot] = static_cast<uint16_t>(newY * INV_GRID_CELL_SIZE);
}

void respawnMeteor(MeteorDodgeRuntime &runtime, uint32_t slot, bool addScore) {
  float x = randomFloat(0.0f, static_cast<float>(meteor_dodge::WINDOW_WIDTH - meteor_dodge::METEOR_WIDTH));
  float y = randomFloat(-static_cast<float>(meteor_dodge::WINDOW_HEIGHT), -20.0f);
  float speed = randomFloat(meteor_dodge::METEOR_MIN_SPEED, meteor_dodge::METEOR_MAX_SPEED);
  float drift = randomFloat(-meteor_dodge::METEOR_DRIFT_RANGE, meteor_dodge::METEOR_DRIFT_RANGE);

  runtime.meteors->speeds[slot] = speed;
  runtime.meteors->drifts[slot] = drift;
  moveEntityWithGrid(runtime.engine, runtime.meteors, slot, x, y);

  if (addScore) {
    runtime.score += 1;
  }
}

void resetRound(MeteorDodgeRuntime &runtime) {
  runtime.gameOver = false;
  runtime.score = 0;

  const uint32_t playerSlot = runtime.players->getSlot(runtime.playerId);
  if (playerSlot != INVALID_ID) {
    const float playerX = (meteor_dodge::WINDOW_WIDTH - meteor_dodge::PLAYER_WIDTH) * 0.5f;
    const float playerY = static_cast<float>(meteor_dodge::WINDOW_HEIGHT - meteor_dodge::PLAYER_HEIGHT - 24);
    moveEntityWithGrid(runtime.engine, runtime.players, playerSlot, playerX, playerY);
  }

  for (uint32_t slot = 0; slot < static_cast<uint32_t>(runtime.meteors->count); slot++) {
    respawnMeteor(runtime, slot, false);
  }
}

void updateTitle(MeteorDodgeRuntime &runtime, Uint64 ticksNow) {
  if (ticksNow - runtime.lastTitleTicks < meteor_dodge::TITLE_UPDATE_MS) {
    return;
  }

  runtime.lastTitleTicks = ticksNow;

  char titleBuffer[128] = {};
  if (runtime.gameOver) {
    std::snprintf(
      titleBuffer,
      sizeof(titleBuffer),
      "Meteor Dodge | Score: %d | Best: %d | Press Space/Enter to restart",
      runtime.score,
      runtime.bestScore
    );
  } else {
    std::snprintf(titleBuffer, sizeof(titleBuffer), "Meteor Dodge | Score: %d | Best: %d", runtime.score, runtime.bestScore);
  }

  SDL_SetWindowTitle(runtime.engine->window, titleBuffer);
}

bool hasPlayerCollision(const MeteorDodgeRuntime &runtime, uint32_t meteorSlot, uint32_t playerSlot) {
  const float meteorX = runtime.meteors->x_positions[meteorSlot];
  const float meteorY = runtime.meteors->y_positions[meteorSlot];
  const float meteorW = static_cast<float>(runtime.meteors->widths[meteorSlot]);
  const float meteorH = static_cast<float>(runtime.meteors->heights[meteorSlot]);

  const float playerX = runtime.players->x_positions[playerSlot];
  const float playerY = runtime.players->y_positions[playerSlot];
  const float playerW = static_cast<float>(runtime.players->widths[playerSlot]);
  const float playerH = static_cast<float>(runtime.players->heights[playerSlot]);

  const bool overlapX = meteorX < (playerX + playerW) && (meteorX + meteorW) > playerX;
  const bool overlapY = meteorY < (playerY + playerH) && (meteorY + meteorH) > playerY;
  return overlapX && overlapY;
}

void updateGameplay(MeteorDodgeRuntime &runtime, float deltaTime) {
  const bool *keys = SDL_GetKeyboardState(nullptr);

  if (runtime.gameOver) {
    if (keys[SDL_SCANCODE_SPACE] || keys[SDL_SCANCODE_RETURN]) {
      resetRound(runtime);
    }
    return;
  }

  const uint32_t playerSlot = runtime.players->getSlot(runtime.playerId);
  if (playerSlot == INVALID_ID) {
    return;
  }

  float playerX = runtime.players->x_positions[playerSlot];
  const float playerY = runtime.players->y_positions[playerSlot];

  if (keys[SDL_SCANCODE_LEFT] || keys[SDL_SCANCODE_A]) {
    playerX -= meteor_dodge::PLAYER_SPEED * deltaTime;
  }

  if (keys[SDL_SCANCODE_RIGHT] || keys[SDL_SCANCODE_D]) {
    playerX += meteor_dodge::PLAYER_SPEED * deltaTime;
  }

  const float maxPlayerX = static_cast<float>(meteor_dodge::WINDOW_WIDTH - meteor_dodge::PLAYER_WIDTH);
  playerX = std::max(0.0f, std::min(playerX, maxPlayerX));
  moveEntityWithGrid(runtime.engine, runtime.players, playerSlot, playerX, playerY);

  for (uint32_t meteorSlot = 0; meteorSlot < static_cast<uint32_t>(runtime.meteors->count); meteorSlot++) {
    float meteorX = runtime.meteors->x_positions[meteorSlot];
    float meteorY = runtime.meteors->y_positions[meteorSlot];

    meteorX += runtime.meteors->drifts[meteorSlot] * deltaTime;
    meteorY += runtime.meteors->speeds[meteorSlot] * deltaTime;

    if (meteorX < -meteor_dodge::METEOR_WIDTH) {
      meteorX = static_cast<float>(meteor_dodge::WINDOW_WIDTH);
    } else if (meteorX > meteor_dodge::WINDOW_WIDTH) {
      meteorX = -static_cast<float>(meteor_dodge::METEOR_WIDTH);
    }

    moveEntityWithGrid(runtime.engine, runtime.meteors, meteorSlot, meteorX, meteorY);

    if (meteorY > meteor_dodge::WINDOW_HEIGHT + meteor_dodge::METEOR_HEIGHT) {
      respawnMeteor(runtime, meteorSlot, true);
      continue;
    }

    if (hasPlayerCollision(runtime, meteorSlot, playerSlot)) {
      runtime.gameOver = true;
      runtime.bestScore = std::max(runtime.bestScore, runtime.score);
      break;
    }
  }
}

bool initializeGame(MeteorDodgeRuntime &runtime) {
  runtime.engine = engine_create(
    meteor_dodge::WINDOW_WIDTH,
    meteor_dodge::WINDOW_HEIGHT,
    meteor_dodge::WORLD_WIDTH,
    meteor_dodge::WORLD_HEIGHT,
    meteor_dodge::GRID_CELL_SIZE
  );

  if (!runtime.engine) {
    SDL_Log("Failed to initialize engine.");
    return false;
  }

  runtime.players = new PlayerContainer(ENTITY_TYPE_PLAYER, 0, 4);
  runtime.meteors = new MeteorContainer(ENTITY_TYPE_METEOR, 0, meteor_dodge::MAX_METEOR_COUNT);

  runtime.engine->entityManager.registerEntityType(runtime.players);
  runtime.engine->entityManager.registerEntityType(runtime.meteors);

  SDL_Surface *playerSurface = create_colored_surface(meteor_dodge::PLAYER_WIDTH, meteor_dodge::PLAYER_HEIGHT, 120, 236, 252);
  SDL_Surface *meteorSurface = create_colored_surface(meteor_dodge::METEOR_WIDTH, meteor_dodge::METEOR_HEIGHT, 255, 153, 88);

  if (!playerSurface || !meteorSurface) {
    SDL_Log("Failed to create runtime surfaces.");
    if (playerSurface) {
      SDL_DestroySurface(playerSurface);
    }
    if (meteorSurface) {
      SDL_DestroySurface(meteorSurface);
    }
    return false;
  }

  runtime.playerTextureId = engine_register_texture(runtime.engine, playerSurface, 0, 0, meteor_dodge::PLAYER_WIDTH, meteor_dodge::PLAYER_HEIGHT);
  runtime.meteorTextureId = engine_register_texture(runtime.engine, meteorSurface, 0, meteor_dodge::PLAYER_HEIGHT + 8, meteor_dodge::METEOR_WIDTH, meteor_dodge::METEOR_HEIGHT);

  SDL_DestroySurface(playerSurface);
  SDL_DestroySurface(meteorSurface);

  const float playerX = (meteor_dodge::WINDOW_WIDTH - meteor_dodge::PLAYER_WIDTH) * 0.5f;
  const float playerY = static_cast<float>(meteor_dodge::WINDOW_HEIGHT - meteor_dodge::PLAYER_HEIGHT - 24);
  runtime.playerId = runtime.players->createPlayer(playerX, playerY, runtime.playerTextureId);

  if (runtime.playerId == INVALID_ID) {
    SDL_Log("Failed to create player entity.");
    return false;
  }

  for (int i = 0; i < meteor_dodge::INITIAL_METEOR_COUNT; i++) {
    const float startX = randomFloat(0.0f, static_cast<float>(meteor_dodge::WINDOW_WIDTH - meteor_dodge::METEOR_WIDTH));
    const float startY = randomFloat(-static_cast<float>(meteor_dodge::WINDOW_HEIGHT), -20.0f);
    const float speed = randomFloat(meteor_dodge::METEOR_MIN_SPEED, meteor_dodge::METEOR_MAX_SPEED);
    const float drift = randomFloat(-meteor_dodge::METEOR_DRIFT_RANGE, meteor_dodge::METEOR_DRIFT_RANGE);
    runtime.meteors->createMeteor(startX, startY, speed, drift, runtime.meteorTextureId);
  }

  runtime.engine->grid.rebuild_grid(runtime.engine);
  runtime.engine->camera.x = meteor_dodge::WINDOW_WIDTH * 0.5f;
  runtime.engine->camera.y = meteor_dodge::WINDOW_HEIGHT * 0.5f;

  runtime.lastFrameTicks = SDL_GetTicks();
  runtime.lastTitleTicks = runtime.lastFrameTicks;
  runtime.running = true;
  runtime.gameOver = false;
  runtime.score = 0;

  SDL_SetWindowTitle(runtime.engine->window, "Meteor Dodge | Score: 0 | Best: 0");

  return true;
}

void shutdownGame(MeteorDodgeRuntime &runtime) {
  if (runtime.engine) {
    engine_destroy(runtime.engine);
  }
  runtime.engine = nullptr;
  runtime.players = nullptr;
  runtime.meteors = nullptr;
}

void runSingleFrame(MeteorDodgeRuntime &runtime) {
  Uint64 now = SDL_GetTicks();
  float deltaTime = std::min((now - runtime.lastFrameTicks) / 1000.0f, 0.05f);
  runtime.lastFrameTicks = now;

  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_EVENT_QUIT) {
      runtime.running = false;
    } else if (event.type == SDL_EVENT_KEY_DOWN && event.key.scancode == SDL_SCANCODE_ESCAPE) {
      runtime.running = false;
    }
  }

  updateGameplay(runtime, deltaTime);
  updateTitle(runtime, now);

  engine_update(runtime.engine);

  SDL_SetRenderDrawColor(runtime.engine->renderer, 6, 10, 24, 255);
  SDL_RenderClear(runtime.engine->renderer);
  engine_render_scene(runtime.engine);

  if (runtime.gameOver) {
    SDL_FRect overlay = {0.0f, 0.0f, static_cast<float>(meteor_dodge::WINDOW_WIDTH), static_cast<float>(meteor_dodge::WINDOW_HEIGHT)};
    SDL_SetRenderDrawBlendMode(runtime.engine->renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(runtime.engine->renderer, 5, 8, 16, 120);
    SDL_RenderFillRect(runtime.engine->renderer, &overlay);
    SDL_SetRenderDrawBlendMode(runtime.engine->renderer, SDL_BLENDMODE_NONE);
  }

  engine_present(runtime.engine);
}

#if defined(__EMSCRIPTEN__)
void webMainLoop(void *arg) {
  MeteorDodgeRuntime *runtime = static_cast<MeteorDodgeRuntime *>(arg);
  if (!runtime->running) {
    emscripten_cancel_main_loop();
    shutdownGame(*runtime);
    return;
  }

  runSingleFrame(*runtime);
}
#endif

} // namespace

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    SDL_Log("SDL_Init failed: %s", SDL_GetError());
    return 1;
  }

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  MeteorDodgeRuntime runtime;
  if (!initializeGame(runtime)) {
    shutdownGame(runtime);
    SDL_Quit();
    return 1;
  }

#if defined(__EMSCRIPTEN__)
  emscripten_set_main_loop_arg(webMainLoop, &runtime, 0, true);
#else
  while (runtime.running) {
    runSingleFrame(runtime);
  }
  shutdownGame(runtime);
#endif

  SDL_Quit();
  return 0;
}
