#include "ATMEngine.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kWindowW = 1280;
constexpr int kWindowH = 720;
constexpr int kCellSize = 28;
constexpr int kGridCols = 34;
constexpr int kGridRows = 20;
constexpr float kSnakeStepSec = 0.11f;
constexpr int kWorldOriginX = 1200;
constexpr int kWorldOriginY = 1100;

struct Cell {
  int x{0};
  int y{0};

  bool operator==(const Cell &other) const {
    return x == other.x && y == other.y;
  }
};

enum class Direction : uint8_t {
  Up = 0,
  Down = 1,
  Left = 2,
  Right = 3,
};

class DynamicSnakeContainer : public RenderableEntityContainer {
public:
  explicit DynamicSnakeContainer(int initial_capacity)
      : RenderableEntityContainer(-1, 2, initial_capacity),
        spin_speed(initial_capacity, 0.0f) {}

  uint32_t createEntity() override {
    const uint32_t slot = RenderableEntityContainer::createEntity();
    if (slot != INVALID_ID) {
      spin_speed[slot] = 0.0f;
    }
    return slot;
  }

  void removeEntity(size_t index) override {
    if (index < static_cast<size_t>(count)) {
      const size_t last = static_cast<size_t>(count - 1);
      if (index < last) {
        spin_speed[index] = spin_speed[last];
      }
    }
    RenderableEntityContainer::removeEntity(index);
  }

  void update(float delta_time) override {
    for (int i = 0; i < count; ++i) {
      rotations[i] += spin_speed[i] * delta_time;
      if (rotations[i] > 6.2831853f) {
        rotations[i] -= 6.2831853f;
      }
    }
  }

protected:
  void resizeArrays(int new_capacity) override {
    RenderableEntityContainer::resizeArrays(new_capacity);
    spin_speed.resize(new_capacity, count, 0.0f);
  }

public:
  DynamicArray<float> spin_speed;
};

class StaticBlockContainer : public RenderableEntityContainer {
public:
  explicit StaticBlockContainer(int initial_capacity)
      : RenderableEntityContainer(-1, 0, initial_capacity) {}

  void update(float delta_time) override { (void)delta_time; }
};

class HybridOrbContainer : public RenderableEntityContainer {
public:
  explicit HybridOrbContainer(int initial_capacity)
      : RenderableEntityContainer(-1, 3, initial_capacity),
        pulse_speed(initial_capacity, 0.0f), phase(initial_capacity, 0.0f) {}

  uint32_t createEntity() override {
    const uint32_t slot = RenderableEntityContainer::createEntity();
    if (slot != INVALID_ID) {
      pulse_speed[slot] = 1.0f + static_cast<float>((slot % 9) * 0.15f);
      phase[slot] = static_cast<float>(slot % 13) * 0.17f;
      rotations[slot] = phase[slot];
    }
    return slot;
  }

  void removeEntity(size_t index) override {
    if (index < static_cast<size_t>(count)) {
      const size_t last = static_cast<size_t>(count - 1);
      if (index < last) {
        pulse_speed[index] = pulse_speed[last];
        phase[index] = phase[last];
      }
    }
    RenderableEntityContainer::removeEntity(index);
  }

  void update(float delta_time) override { (void)delta_time; }

  void updateVisible(const std::vector<uint32_t> &active_indices,
                     float delta_time) override {
    last_active_count = static_cast<int>(active_indices.size());

    for (uint32_t idx : active_indices) {
      if (idx >= static_cast<uint32_t>(count)) {
        continue;
      }
      phase[idx] += pulse_speed[idx] * delta_time;
      rotations[idx] = phase[idx];
    }

    log_timer += delta_time;
    if (log_timer > 1.0f) {
      log_timer = 0.0f;
      SDL_Log("[OptionBTest] Hybrid active this frame: %d", last_active_count);
    }
  }

protected:
  void resizeArrays(int new_capacity) override {
    RenderableEntityContainer::resizeArrays(new_capacity);
    pulse_speed.resize(new_capacity, count, 0.0f);
    phase.resize(new_capacity, count, 0.0f);
  }

public:
  DynamicArray<float> pulse_speed;
  DynamicArray<float> phase;
  int last_active_count{0};
  float log_timer{0.0f};
};

static SDL_Surface *makeColorSurface(int width, int height, uint8_t r, uint8_t g,
                                     uint8_t b, uint8_t a = 255) {
  SDL_Surface *surface = SDL_CreateSurface(width, height, SDL_PIXELFORMAT_ABGR8888);
  if (!surface) {
    return nullptr;
  }

  SDL_FillSurfaceRect(surface, nullptr,
                      SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format), nullptr,
                                  r, g, b, a));
  return surface;
}

static bool resolveSlot(Engine *engine, const EntityHandle &handle, uint32_t &slot) {
  if (!engine || !handle.isValid()) {
    return false;
  }
  return engine->entityManager.resolveEntitySlot(handle, slot);
}

static bool configureRenderable(Engine *engine, RenderableEntityContainer *container,
                                const EntityHandle &handle, int16_t texture_id,
                                int16_t width, int16_t height, uint8_t z_index,
                                float spin_speed = 0.0f) {
  if (!engine || !container || !handle.isValid()) {
    return false;
  }

  uint32_t slot = INVALID_SLOT;
  if (!resolveSlot(engine, handle, slot)) {
    return false;
  }

  if (slot >= static_cast<uint32_t>(container->count)) {
    return false;
  }

  container->texture_ids[slot] = texture_id;
  container->widths[slot] = width;
  container->heights[slot] = height;
  container->rotations[slot] = 0.0f;
  container->z_indices[slot] = z_index;
  engine_set_entity_visible(engine, handle, true);

  auto *dynamic_container = dynamic_cast<DynamicSnakeContainer *>(container);
  if (dynamic_container) {
    dynamic_container->spin_speed[slot] = spin_speed;
  }

  return true;
}

static float cellWorldX(int grid_x) {
  return static_cast<float>(kWorldOriginX + grid_x * kCellSize);
}

static float cellWorldY(int grid_y) {
  return static_cast<float>(kWorldOriginY + grid_y * kCellSize);
}

static Cell randomFreeCell(const std::deque<Cell> &snake,
                           const std::vector<Cell> &blocked_cells,
                           std::mt19937 &rng) {
  std::uniform_int_distribution<int> xdist(1, kGridCols - 2);
  std::uniform_int_distribution<int> ydist(1, kGridRows - 2);

  for (int attempt = 0; attempt < 1024; ++attempt) {
    const Cell candidate{xdist(rng), ydist(rng)};

    bool occupied = false;
    for (const Cell &segment : snake) {
      if (segment == candidate) {
        occupied = true;
        break;
      }
    }
    if (occupied) {
      continue;
    }

    for (const Cell &blocked : blocked_cells) {
      if (blocked == candidate) {
        occupied = true;
        break;
      }
    }
    if (!occupied) {
      return candidate;
    }
  }

  return Cell{2, 2};
}

static bool isOpposite(Direction a, Direction b) {
  return (a == Direction::Up && b == Direction::Down) ||
         (a == Direction::Down && b == Direction::Up) ||
         (a == Direction::Left && b == Direction::Right) ||
         (a == Direction::Right && b == Direction::Left);
}

} // namespace

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  Engine *engine = engine_create(kWindowW, kWindowH, 50000, 50000, 64);
  if (!engine) {
    SDL_Log("Failed to create engine");
    return 1;
  }

  SDL_SetWindowTitle(engine->window, "Snake OptionB Test (Dynamic + Static + Hybrid)");
  std::mt19937 rng(static_cast<uint32_t>(SDL_GetTicks()));

  SDL_Surface *head_surface = makeColorSurface(kCellSize, kCellSize, 239, 68, 153);
  SDL_Surface *body_surface = makeColorSurface(kCellSize, kCellSize, 190, 24, 93);
  SDL_Surface *food_surface = makeColorSurface(kCellSize, kCellSize, 20, 184, 166);
  SDL_Surface *wall_surface = makeColorSurface(kCellSize, kCellSize, 90, 90, 120);
  SDL_Surface *gate_surface = makeColorSurface(kCellSize, kCellSize, 225, 179, 22);
  SDL_Surface *hybrid_surface = makeColorSurface(kCellSize - 10, kCellSize - 10, 255, 255, 255);

  if (!head_surface || !body_surface || !food_surface || !wall_surface ||
      !gate_surface || !hybrid_surface) {
    SDL_Log("Failed to create one or more surfaces");
    SDL_DestroySurface(head_surface);
    SDL_DestroySurface(body_surface);
    SDL_DestroySurface(food_surface);
    SDL_DestroySurface(wall_surface);
    SDL_DestroySurface(gate_surface);
    SDL_DestroySurface(hybrid_surface);
    engine_destroy(engine);
    return 1;
  }

  const int16_t tex_head =
      static_cast<int16_t>(engine_register_texture(engine, head_surface, 0, 0, 0, 0));
  const int16_t tex_body =
      static_cast<int16_t>(engine_register_texture(engine, body_surface, 40, 0, 0, 0));
  const int16_t tex_food =
      static_cast<int16_t>(engine_register_texture(engine, food_surface, 80, 0, 0, 0));
  const int16_t tex_wall =
      static_cast<int16_t>(engine_register_texture(engine, wall_surface, 120, 0, 0, 0));
  const int16_t tex_gate =
      static_cast<int16_t>(engine_register_texture(engine, gate_surface, 160, 0, 0, 0));
  const int16_t tex_hybrid =
      static_cast<int16_t>(engine_register_texture(engine, hybrid_surface, 200, 0, 0, 0));

  SDL_DestroySurface(head_surface);
  SDL_DestroySurface(body_surface);
  SDL_DestroySurface(food_surface);
  SDL_DestroySurface(wall_surface);
  SDL_DestroySurface(gate_surface);
  SDL_DestroySurface(hybrid_surface);

  auto *dynamic_container = new DynamicSnakeContainer(1024);
  auto *static_container = new StaticBlockContainer(2048);
  auto *hybrid_container = new HybridOrbContainer(1024);

  const int dynamic_type = engine_register_dynamic_type(engine, dynamic_container);
  const int static_type = engine_register_static_type(engine, static_container);
  const int hybrid_type = engine_register_hybrid_type(engine, hybrid_container);

  if (dynamic_type < 0 || static_type < 0 || hybrid_type < 0) {
    SDL_Log("Failed to register one or more entity types");
    engine_destroy(engine);
    return 1;
  }

  std::vector<EntityHandle> static_border_handles;
  std::vector<EntityHandle> static_gate_handles;
  std::vector<Cell> blocked_cells;

  auto spawnStaticBlock = [&](const Cell &cell, int16_t texture_id,
                              std::vector<EntityHandle> *store) {
    EntityHandle h = engine_create_entity(engine, static_type);
    if (!h.isValid()) {
      return;
    }
    configureRenderable(engine, static_container, h, texture_id, kCellSize, kCellSize,
                        2);
    engine_set_entity_position(engine, h, cellWorldX(cell.x), cellWorldY(cell.y));
    if (store) {
      store->push_back(h);
    }
    blocked_cells.push_back(cell);
  };

  for (int x = 0; x < kGridCols; ++x) {
    spawnStaticBlock(Cell{x, 0}, tex_wall, &static_border_handles);
    spawnStaticBlock(Cell{x, kGridRows - 1}, tex_wall, &static_border_handles);
  }
  for (int y = 1; y < kGridRows - 1; ++y) {
    spawnStaticBlock(Cell{0, y}, tex_wall, &static_border_handles);
    spawnStaticBlock(Cell{kGridCols - 1, y}, tex_wall, &static_border_handles);
  }

  const int gate_y = kGridRows / 2;
  for (int x = 10; x < 15; ++x) {
    spawnStaticBlock(Cell{x, gate_y}, tex_gate, &static_gate_handles);
  }

  // Hybrid entities spread over a larger region; only visible ones should run
  // updateVisible.
  std::uniform_int_distribution<int> hx(kWorldOriginX - 2000,
                                        kWorldOriginX + kGridCols * kCellSize + 2000);
  std::uniform_int_distribution<int> hy(kWorldOriginY - 1800,
                                        kWorldOriginY + kGridRows * kCellSize + 1800);
  for (int i = 0; i < 220; ++i) {
    EntityHandle h = engine_create_entity(engine, hybrid_type);
    if (!h.isValid()) {
      continue;
    }
    configureRenderable(engine, hybrid_container, h, tex_hybrid, kCellSize - 10,
                        kCellSize - 10, 20);
    engine_set_entity_position(engine, h, static_cast<float>(hx(rng)),
                               static_cast<float>(hy(rng)));
  }

  std::deque<Cell> snake_cells;
  std::deque<EntityHandle> snake_handles;
  EntityHandle food_handle{};
  Cell food_cell{};
  Direction direction = Direction::Right;
  Direction next_direction = Direction::Right;
  int score = 0;
  bool gate_visible = true;
  std::vector<EntityHandle> bonus_static_handles;

  auto setSnakeHeadVisual = [&](const EntityHandle &h) {
    uint32_t slot = INVALID_SLOT;
    if (!resolveSlot(engine, h, slot)) {
      return;
    }
    if (slot >= static_cast<uint32_t>(dynamic_container->count)) {
      return;
    }
    dynamic_container->texture_ids[slot] = tex_head;
    dynamic_container->spin_speed[slot] = 0.8f;
  };

  auto setSnakeBodyVisual = [&](const EntityHandle &h) {
    uint32_t slot = INVALID_SLOT;
    if (!resolveSlot(engine, h, slot)) {
      return;
    }
    if (slot >= static_cast<uint32_t>(dynamic_container->count)) {
      return;
    }
    dynamic_container->texture_ids[slot] = tex_body;
    dynamic_container->spin_speed[slot] = 0.0f;
    dynamic_container->rotations[slot] = 0.0f;
  };

  auto spawnFood = [&]() {
    food_cell = randomFreeCell(snake_cells, blocked_cells, rng);
    if (!food_handle.isValid()) {
      food_handle = engine_create_entity(engine, dynamic_type);
      if (!food_handle.isValid()) {
        return;
      }
      configureRenderable(engine, dynamic_container, food_handle, tex_food, kCellSize,
                          kCellSize, 8, 1.7f);
    }
    engine_set_entity_position(engine, food_handle, cellWorldX(food_cell.x),
                               cellWorldY(food_cell.y));
  };

  auto resetGame = [&]() {
    for (const EntityHandle &h : snake_handles) {
      engine_destroy_entity(engine, h);
    }
    snake_handles.clear();
    snake_cells.clear();

    direction = Direction::Right;
    next_direction = Direction::Right;
    score = 0;
    gate_visible = true;

    for (const EntityHandle &h : static_gate_handles) {
      engine_set_entity_visible(engine, h, true);
    }

    const int start_x = kGridCols / 2;
    const int start_y = kGridRows / 2;
    for (int i = 0; i < 5; ++i) {
      EntityHandle segment = engine_create_entity(engine, dynamic_type);
      if (!segment.isValid()) {
        continue;
      }
      configureRenderable(engine, dynamic_container, segment, tex_body, kCellSize,
                          kCellSize, 10);
      const Cell cell{start_x - i, start_y};
      snake_cells.push_back(cell);
      snake_handles.push_back(segment);
      engine_set_entity_position(engine, segment, cellWorldX(cell.x),
                                 cellWorldY(cell.y));
    }

    if (!snake_handles.empty()) {
      setSnakeHeadVisual(snake_handles.front());
    }

    spawnFood();
  };

  resetGame();

  bool running = true;
  float dt = 0.016f;
  float step_timer = 0.0f;
  bool game_over = false;

  while (running) {
    const uint64_t frame_start = SDL_GetTicks();

    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
      if (ev.type == SDL_EVENT_QUIT) {
        running = false;
      } else if (ev.type == SDL_EVENT_KEY_DOWN) {
        const SDL_Keycode key = ev.key.key;
        if (key == SDLK_ESCAPE) {
          running = false;
        }

        Direction desired = next_direction;
        if (key == SDLK_UP || key == SDLK_W) {
          desired = Direction::Up;
        } else if (key == SDLK_DOWN || key == SDLK_S) {
          desired = Direction::Down;
        } else if (key == SDLK_LEFT || key == SDLK_A) {
          desired = Direction::Left;
        } else if (key == SDLK_RIGHT || key == SDLK_D) {
          desired = Direction::Right;
        } else if (key == SDLK_R) {
          resetGame();
          game_over = false;
          step_timer = 0.0f;
        }

        if (!isOpposite(direction, desired)) {
          next_direction = desired;
        }
      }
    }

    if (!game_over) {
      step_timer += dt;
      while (step_timer >= kSnakeStepSec) {
        step_timer -= kSnakeStepSec;
        direction = next_direction;

        Cell next = snake_cells.front();
        if (direction == Direction::Up) {
          next.y -= 1;
        } else if (direction == Direction::Down) {
          next.y += 1;
        } else if (direction == Direction::Left) {
          next.x -= 1;
        } else if (direction == Direction::Right) {
          next.x += 1;
        }

        bool hit_obstacle = false;
        for (const Cell &blocked : blocked_cells) {
          if (blocked == next) {
            hit_obstacle = true;
            break;
          }
        }
        if (!hit_obstacle) {
          for (const Cell &segment : snake_cells) {
            if (segment == next) {
              hit_obstacle = true;
              break;
            }
          }
        }

        if (hit_obstacle) {
          game_over = true;
          SDL_Log("[OptionBTest] Game over at score=%d. Press R to reset.", score);
          break;
        }

        if (!snake_handles.empty()) {
          setSnakeBodyVisual(snake_handles.front());
        }

        const bool ate_food = (next == food_cell);
        snake_cells.push_front(next);

        if (ate_food) {
          score += 1;
          EntityHandle new_head = engine_create_entity(engine, dynamic_type);
          if (new_head.isValid()) {
            configureRenderable(engine, dynamic_container, new_head, tex_head, kCellSize,
                                kCellSize, 10, 0.8f);
            engine_set_entity_position(engine, new_head, cellWorldX(next.x),
                                       cellWorldY(next.y));
            snake_handles.push_front(new_head);
          }

          if (score % 2 == 0) {
            gate_visible = !gate_visible;
            for (const EntityHandle &h : static_gate_handles) {
              engine_set_entity_visible(engine, h, gate_visible);
            }
          }

          if (score % 4 == 0 && !static_border_handles.empty()) {
            engine_destroy_entity(engine, static_border_handles.back());
            static_border_handles.pop_back();
          }

          if (score % 3 == 0) {
            Cell bonus = randomFreeCell(snake_cells, blocked_cells, rng);
            EntityHandle extra = engine_create_entity(engine, static_type);
            if (extra.isValid()) {
              configureRenderable(engine, static_container, extra, tex_gate, kCellSize,
                                  kCellSize, 4);
              engine_set_entity_position(engine, extra, cellWorldX(bonus.x),
                                         cellWorldY(bonus.y));
              bonus_static_handles.push_back(extra);
              blocked_cells.push_back(bonus);
            }
          }

          spawnFood();
        } else {
          if (!snake_handles.empty()) {
            EntityHandle tail = snake_handles.back();
            snake_handles.pop_back();
            snake_handles.push_front(tail);
            snake_cells.pop_back();
            engine_set_entity_position(engine, tail, cellWorldX(next.x),
                                       cellWorldY(next.y));
            setSnakeHeadVisual(tail);
          }
        }
      }
    }

    if (!snake_cells.empty()) {
      const float cam_x = cellWorldX(snake_cells.front().x) + (kCellSize * 0.5f);
      const float cam_y = cellWorldY(snake_cells.front().y) + (kCellSize * 0.5f);
      engine->camera.x = cam_x;
      engine->camera.y = cam_y;
    }

    engine_update(engine);
    engine_render_scene(engine);

    // Minimal HUD accents to make camera movement boundaries easy to see.
    SDL_SetRenderDrawColor(engine->renderer, 12, 12, 22, 160);
    SDL_FRect hud = {10.0f, 10.0f, 360.0f, 38.0f};
    SDL_RenderFillRect(engine->renderer, &hud);
    SDL_SetRenderDrawColor(engine->renderer, 220, 220, 240, 255);
    SDL_RenderRect(engine->renderer, &hud);

    engine_present(engine);

    const uint64_t frame_end = SDL_GetTicks();
    dt = static_cast<float>(frame_end - frame_start) / 1000.0f;
    if (dt < 0.001f) {
      dt = 0.001f;
    }
  }

  for (const EntityHandle &h : snake_handles) {
    engine_destroy_entity(engine, h);
  }
  if (food_handle.isValid()) {
    engine_destroy_entity(engine, food_handle);
  }
  for (const EntityHandle &h : static_border_handles) {
    engine_destroy_entity(engine, h);
  }
  for (const EntityHandle &h : static_gate_handles) {
    engine_destroy_entity(engine, h);
  }
  for (const EntityHandle &h : bonus_static_handles) {
    engine_destroy_entity(engine, h);
  }

  engine_destroy(engine);
  return 0;
}
