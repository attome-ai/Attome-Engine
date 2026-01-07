#include "ATMEngine.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <execution>
#include <iostream>
#include <numeric>
#include <vector>

// --- Constants ---
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1020
#define GRID_SIZE 32      // Size of each snake segment / food / enemy
#define SNAKE_SPEED 15.0f // Segments per second (faster = responsive)
#define NUM_FOODS 1000000 // Increased food count
#define NUM_ENEMIES 5000  // Increased enemy count
#define NUM_POWER_UPS 1000
#define INITIAL_SNAKE_LENGTH 5
#define NUM_FOOD_TYPES 10 // 10 different food types

// --- Game-specific entity types ---
enum GameEntityTypes {
  ENTITY_TYPE_SNAKE_HEAD = 0,
  ENTITY_TYPE_SNAKE_BODY,
  ENTITY_TYPE_FOOD,
  ENTITY_TYPE_ENEMY,
  ENTITY_TYPE_POWER_UP,
  ENTITY_TYPE_COUNT
};

// --- Snake Segment ---
struct SnakeSegment {
  float x, y;               // Logic position (grid-aligned)
  float visual_x, visual_y; // Visual position (smooth)
  uint32_t entity_index;
};

// --- Direction enum ---
enum Direction { UP = 0, DOWN, LEFT, RIGHT };

// --- Snake Head Container ---
class SnakeHeadContainer : public RenderableEntityContainer {
public:
  Direction *directions;
  float *speeds;

  SnakeHeadContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity) {
    directions = new Direction[capacity];
    speeds = new float[capacity];
    std::fill(directions, directions + capacity, RIGHT);
    std::fill(speeds, speeds + capacity, SNAKE_SPEED);
  }

  ~SnakeHeadContainer() override {
    delete[] directions;
    delete[] speeds;
  }

  uint32_t createEntity() override {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    directions[index] = RIGHT;
    speeds[index] = SNAKE_SPEED;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 100; // High z-index to render above all food
    return index;
  }

  uint32_t createEntity(float x, float y, int texture_id) {
    uint32_t index = createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = GRID_SIZE;
    heights[index] = GRID_SIZE;
    texture_ids[index] = texture_id;
    return index;
  }

  void removeEntity(size_t index) override {
    if (index >= count)
      return;
    size_t last = count - 1;
    if (index < last) {
      directions[index] = directions[last];
      speeds[index] = speeds[last];
    }
    RenderableEntityContainer::removeEntity(index);
  }

  void update(float delta_time) override {
    for (int i = 0; i < count; ++i) {
      flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    }
  }

protected:
  void resizeArrays(int newCapacity) override {
    if (newCapacity <= capacity)
      return;

    Direction *newDirections = new Direction[newCapacity];
    float *newSpeeds = new float[newCapacity];

    if (count > 0) {
      std::copy(directions, directions + count, newDirections);
      std::copy(speeds, speeds + count, newSpeeds);
    }
    std::fill(newDirections + count, newDirections + newCapacity, RIGHT);
    std::fill(newSpeeds + count, newSpeeds + newCapacity, SNAKE_SPEED);

    delete[] directions;
    delete[] speeds;

    directions = newDirections;
    speeds = newSpeeds;

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Snake Body Container ---
class SnakeBodyContainer : public RenderableEntityContainer {
public:
  SnakeBodyContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity) {}

  uint32_t createEntity(float x, float y, int texture_id) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = GRID_SIZE;
    heights[index] = GRID_SIZE;
    texture_ids[index] = texture_id;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 99; // High z-index to render above food
    return index;
  }

  void update(float delta_time) override {
    for (int i = 0; i < count; ++i) {
      flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    }
  }
};

// --- Food Container with 10 types ---
class FoodContainer : public RenderableEntityContainer {
public:
  int *values;     // Points for eating
  int *growth;     // Segments to add when eaten (1-10)
  int *food_types; // Food type (0-9)

  FoodContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity) {
    values = new int[capacity];
    growth = new int[capacity];
    food_types = new int[capacity];
    std::fill(values, values + capacity, 10);
    std::fill(growth, growth + capacity, 1);
    std::fill(food_types, food_types + capacity, 0);
  }

  ~FoodContainer() override {
    delete[] values;
    delete[] growth;
    delete[] food_types;
  }

  uint32_t createEntity(float x, float y, int texture_id, int value, int grow,
                        int type) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = GRID_SIZE;
    heights[index] = GRID_SIZE;
    texture_ids[index] = texture_id;
    values[index] = value;
    growth[index] = grow;
    food_types[index] = type;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 5;
    return index;
  }

  void removeEntity(size_t index) override {
    if (index >= count)
      return;
    size_t last = count - 1;
    if (index < last) {
      values[index] = values[last];
      growth[index] = growth[last];
      food_types[index] = food_types[last];
    }
    RenderableEntityContainer::removeEntity(index);
  }

  void update(float delta_time) override {
    for (int i = 0; i < count; ++i) {
      flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    }
  }

protected:
  void resizeArrays(int newCapacity) override {
    if (newCapacity <= capacity)
      return;

    int *newValues = new int[newCapacity];
    int *newGrowth = new int[newCapacity];
    int *newFoodTypes = new int[newCapacity];
    if (count > 0) {
      std::copy(values, values + count, newValues);
      std::copy(growth, growth + count, newGrowth);
      std::copy(food_types, food_types + count, newFoodTypes);
    }
    std::fill(newValues + count, newValues + newCapacity, 10);
    std::fill(newGrowth + count, newGrowth + newCapacity, 1);
    std::fill(newFoodTypes + count, newFoodTypes + newCapacity, 0);

    delete[] values;
    delete[] growth;
    delete[] food_types;
    values = newValues;
    growth = newGrowth;
    food_types = newFoodTypes;

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Enemy Container ---
class EnemyContainer : public RenderableEntityContainer {
public:
  float *speeds;
  float *dir_x;
  float *dir_y;
  Engine *engine;

  EnemyContainer(Engine *engine, int typeId, uint8_t defaultLayer,
                 int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity),
        engine(engine) {
    speeds = new float[capacity];
    dir_x = new float[capacity];
    dir_y = new float[capacity];
    std::fill(speeds, speeds + capacity, 50.0f);
    std::fill(dir_x, dir_x + capacity, 1.0f);
    std::fill(dir_y, dir_y + capacity, 0.0f);
  }

  ~EnemyContainer() override {
    delete[] speeds;
    delete[] dir_x;
    delete[] dir_y;
  }

  uint32_t createEntity(float x, float y, int texture_id, float speed = 50.0f) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = GRID_SIZE;
    heights[index] = GRID_SIZE;
    texture_ids[index] = texture_id;
    speeds[index] = speed;

    // Random direction
    float angle = static_cast<float>(rand()) / RAND_MAX * 2.0f * 3.14159265f;
    dir_x[index] = cosf(angle);
    dir_y[index] = sinf(angle);

    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 6;
    return index;
  }

  void removeEntity(size_t index) override {
    if (index >= count)
      return;
    size_t last = count - 1;
    if (index < last) {
      speeds[index] = speeds[last];
      dir_x[index] = dir_x[last];
      dir_y[index] = dir_y[last];
    }
    RenderableEntityContainer::removeEntity(index);
  }

  void update(float delta_time) override {
    PROFILE_FUNCTION();
    if (count == 0)
      return;
    delta_time = std::min(delta_time, 0.1f);

    // Buffer for deferred grid updates
    static std::vector<uint32_t> indices;
    static std::vector<uint32_t> pending_moves;
    static std::atomic<uint32_t> pending_count{0};

    if (indices.size() < (size_t)count) {
      indices.resize(count);
      std::iota(indices.begin(), indices.end(), 0);
    }
    if (pending_moves.size() < (size_t)count) {
      pending_moves.resize(count);
    }
    pending_count.store(0, std::memory_order_relaxed);

    // Parallel position update
    std::for_each(std::execution::par, indices.begin(), indices.begin() + count,
                  [&](uint32_t i) {
                    float &px = x_positions[i];
                    float &py = y_positions[i];
                    float &dx = dir_x[i];
                    float &dy = dir_y[i];
                    const float speed = speeds[i] * delta_time;

                    uint16_t oldCellX = cell_x[i];
                    uint16_t oldCellY = cell_y[i];

                    px += dx * speed;
                    py += dy * speed;

                    // Boundary bounce
                    if (px < 0) {
                      px = 0;
                      dx = -dx;
                    } else if (px > WORLD_WIDTH - GRID_SIZE) {
                      px = WORLD_WIDTH - GRID_SIZE;
                      dx = -dx;
                    }

                    if (py < 0) {
                      py = 0;
                      dy = -dy;
                    } else if (py > WORLD_HEIGHT - GRID_SIZE) {
                      py = WORLD_HEIGHT - GRID_SIZE;
                      dy = -dy;
                    }

                    uint16_t newCellX =
                        static_cast<uint16_t>(px * INV_GRID_CELL_SIZE);
                    uint16_t newCellY =
                        static_cast<uint16_t>(py * INV_GRID_CELL_SIZE);

                    if (oldCellX != newCellX || oldCellY != newCellY) {
                      uint32_t slot =
                          pending_count.fetch_add(1, std::memory_order_relaxed);
                      pending_moves[slot] = i;
                      cell_x[i] = newCellX;
                      cell_y[i] = newCellY;
                    }

                    flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
                  });

    // Serial grid update
    uint32_t num_moves = pending_count.load(std::memory_order_relaxed);
    for (uint32_t j = 0; j < num_moves; ++j) {
      uint32_t i = pending_moves[j];
      int32_t nodeIdx = grid_node_indices[i];
      engine->grid.move(nodeIdx, x_positions[i], y_positions[i]);
    }
  }

protected:
  void resizeArrays(int newCapacity) override {
    if (newCapacity <= capacity)
      return;

    float *newSpeeds = new float[newCapacity];
    float *newDirX = new float[newCapacity];
    float *newDirY = new float[newCapacity];

    if (count > 0) {
      std::copy(speeds, speeds + count, newSpeeds);
      std::copy(dir_x, dir_x + count, newDirX);
      std::copy(dir_y, dir_y + count, newDirY);
    }
    std::fill(newSpeeds + count, newSpeeds + newCapacity, 50.0f);
    std::fill(newDirX + count, newDirX + newCapacity, 1.0f);
    std::fill(newDirY + count, newDirY + newCapacity, 0.0f);

    delete[] speeds;
    delete[] dir_x;
    delete[] dir_y;

    speeds = newSpeeds;
    dir_x = newDirX;
    dir_y = newDirY;

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Power-Up Container ---
class PowerUpContainer : public RenderableEntityContainer {
public:
  int *types; // 0=speed boost, 1=invincibility, 2=score multiplier

  PowerUpContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity) {
    types = new int[capacity];
    std::fill(types, types + capacity, 0);
  }

  ~PowerUpContainer() override { delete[] types; }

  uint32_t createEntity(float x, float y, int texture_id, int type = 0) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = GRID_SIZE;
    heights[index] = GRID_SIZE;
    texture_ids[index] = texture_id;
    types[index] = type;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 4;
    return index;
  }

  void removeEntity(size_t index) override {
    if (index >= count)
      return;
    size_t last = count - 1;
    if (index < last) {
      types[index] = types[last];
    }
    RenderableEntityContainer::removeEntity(index);
  }

  void update(float delta_time) override {
    for (int i = 0; i < count; ++i) {
      flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    }
  }

protected:
  void resizeArrays(int newCapacity) override {
    if (newCapacity <= capacity)
      return;

    int *newTypes = new int[newCapacity];
    if (count > 0) {
      std::copy(types, types + count, newTypes);
    }
    std::fill(newTypes + count, newTypes + newCapacity, 0);

    delete[] types;
    types = newTypes;

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Game State ---
struct GameState {
  // Snake data
  std::deque<SnakeSegment> snake_body;
  uint32_t head_entity_index;
  Direction current_direction;
  Direction queued_direction;
  float move_timer;
  float speed_boost_timer;
  float invincibility_timer;
  bool is_alive;

  // Score
  int score;
  int score_multiplier;
  float multiplier_timer;

  // Textures
  int head_texture_id;
  int body_texture_id;
  int food_texture_ids[NUM_FOOD_TYPES]; // 10 different colors for food types
  int enemy_texture_id;
  int powerup_texture_id;

  // Containers
  SnakeHeadContainer *head_container;
  SnakeBodyContainer *body_container;
  FoodContainer *food_container;
  EnemyContainer *enemy_container;
  PowerUpContainer *powerup_container;

  // FPS tracking
  Uint64 last_fps_time;
  int frame_count;
  float current_fps;

  // Visual interpolation for smooth movement
  float head_visual_x, head_visual_y;
  float head_logic_x, head_logic_y; // Grid-aligned logic position
};

// --- Function Declarations ---
void setup_game(Engine *engine, GameState *game_state);
void handle_input(const bool *keyboard_state, GameState *game_state);
void update_snake(Engine *engine, GameState *game_state, float delta_time);
void check_collisions(Engine *engine, GameState *game_state);
void spawn_food(Engine *engine, GameState *game_state);
SDL_Surface *create_colored_surface(int width, int height, Uint8 r, Uint8 g,
                                    Uint8 b);

// --- Main Function ---
int main(int argc, char *argv[]) {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
    return 1;
  }
  srand(static_cast<unsigned int>(time(nullptr)));

  SDL_Window *window = SDL_CreateWindow("Snake Game - Big World", WINDOW_WIDTH,
                                        WINDOW_HEIGHT, 0);
  if (!window) {
    std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
    SDL_Quit();
    return 1;
  }

  SDL_Renderer *renderer = SDL_CreateRenderer(window, NULL);
  if (!renderer) {
    std::cerr << "Failed to create renderer: " << SDL_GetError() << std::endl;
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 1;
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);

  Engine *engine = engine_create(WINDOW_WIDTH, WINDOW_HEIGHT, WORLD_WIDTH,
                                 WORLD_HEIGHT, GRID_CELL_SIZE);
  if (!engine) {
    std::cerr << "Failed to create engine" << std::endl;
    SDL_Quit();
    return 1;
  }

  // Register entity containers
  SnakeHeadContainer *head_container =
      new SnakeHeadContainer(ENTITY_TYPE_SNAKE_HEAD, 0, 10);
  SnakeBodyContainer *body_container =
      new SnakeBodyContainer(ENTITY_TYPE_SNAKE_BODY, 0, 10000);
  FoodContainer *food_container =
      new FoodContainer(ENTITY_TYPE_FOOD, 0, NUM_FOODS + 100);
  EnemyContainer *enemy_container =
      new EnemyContainer(engine, ENTITY_TYPE_ENEMY, 0, NUM_ENEMIES + 100);
  PowerUpContainer *powerup_container =
      new PowerUpContainer(ENTITY_TYPE_POWER_UP, 0, NUM_POWER_UPS + 100);

  engine->entityManager.registerEntityType(head_container);
  engine->entityManager.registerEntityType(body_container);
  engine->entityManager.registerEntityType(food_container);
  engine->entityManager.registerEntityType(enemy_container);
  engine->entityManager.registerEntityType(powerup_container);

  // Initialize game state
  GameState game_state = {};
  game_state.head_container = head_container;
  game_state.body_container = body_container;
  game_state.food_container = food_container;
  game_state.enemy_container = enemy_container;
  game_state.powerup_container = powerup_container;
  game_state.head_entity_index = INVALID_ID;
  game_state.is_alive = true;
  game_state.score_multiplier = 1;
  game_state.last_fps_time = SDL_GetTicks();

  setup_game(engine, &game_state);

  // Initial grid population
  engine->grid.rebuild_grid(engine);

  // Initialize cell tracking
  for (uint32_t cIdx = 0; cIdx < engine->entityManager.containers.size();
       ++cIdx) {
    auto &container = engine->entityManager.containers[cIdx];
    for (int i = 0; i < container->count; ++i) {
      container->cell_x[i] =
          static_cast<uint16_t>(container->x_positions[i] * INV_GRID_CELL_SIZE);
      container->cell_y[i] =
          static_cast<uint16_t>(container->y_positions[i] * INV_GRID_CELL_SIZE);
    }
  }

  // Set camera to snake head
  if (game_state.head_entity_index != INVALID_ID) {
    float head_x = head_container->x_positions[game_state.head_entity_index];
    float head_y = head_container->y_positions[game_state.head_entity_index];
    engine->camera.x = head_x + GRID_SIZE / 2.0f;
    engine->camera.y = head_y + GRID_SIZE / 2.0f;
  }

  // Game loop
  bool quit = false;
  SDL_Event event;
  Uint64 last_time = SDL_GetTicks();

  while (!quit) {
    Uint64 current_time = SDL_GetTicks();
    float delta_time = std::min((current_time - last_time) / 1000.0f, 0.1f);
    last_time = current_time;

    // FPS calculation
    game_state.frame_count++;
    Uint64 time_since_last_fps = current_time - game_state.last_fps_time;
    if (time_since_last_fps >= 1000) {
      game_state.current_fps =
          static_cast<float>(game_state.frame_count * 1000.0f) /
          static_cast<float>(time_since_last_fps);
      game_state.last_fps_time = current_time;
      game_state.frame_count = 0;
      std::cout << "FPS: " << game_state.current_fps
                << " | Score: " << game_state.score
                << " | Length: " << game_state.snake_body.size() + 1
                << std::endl;
    }

    // Process events - this internally calls SDL_PumpEvents
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_QUIT)
        quit = true;
      else if (event.type == SDL_EVENT_KEY_DOWN) {
        if (event.key.scancode == SDL_SCANCODE_ESCAPE)
          quit = true;
        else if (event.key.scancode == SDL_SCANCODE_R && !game_state.is_alive) {
          // Reset game
          game_state.snake_body.clear();
          game_state.score = 0;
          game_state.is_alive = true;
          setup_game(engine, &game_state);
          engine->grid.rebuild_grid(engine);
        }
      }
    }

    // Get keyboard state AFTER events are pumped (so it's fresh)
    // SDL3 returns bool* instead of Uint8*
    const bool *keyboard_state = SDL_GetKeyboardState(NULL);

    if (game_state.is_alive) {
      handle_input(keyboard_state, &game_state);
      update_snake(engine, &game_state, delta_time);
      check_collisions(engine, &game_state);

      // Update power-up timers
      if (game_state.speed_boost_timer > 0)
        game_state.speed_boost_timer -= delta_time;
      if (game_state.invincibility_timer > 0)
        game_state.invincibility_timer -= delta_time;
      if (game_state.multiplier_timer > 0) {
        game_state.multiplier_timer -= delta_time;
        if (game_state.multiplier_timer <= 0)
          game_state.score_multiplier = 1;
      }
    }

    // Camera zoom controls
    if (keyboard_state[SDL_SCANCODE_EQUALS] ||
        keyboard_state[SDL_SCANCODE_KP_PLUS]) {
      engine->camera.width *= 0.98f;
      engine->camera.height *= 0.98f;
    }
    if (keyboard_state[SDL_SCANCODE_MINUS] ||
        keyboard_state[SDL_SCANCODE_KP_MINUS]) {
      engine->camera.width *= 1.02f;
      engine->camera.height *= 1.02f;
    }

    engine_update(engine);

    SDL_SetRenderDrawColor(engine->renderer, 20, 20, 30, 255);
    SDL_RenderClear(engine->renderer);
    engine_render_scene(engine);
    engine_present(engine);
  }

  engine_destroy(engine);
  SDL_Quit();
  return 0;
}

SDL_Surface *create_colored_surface(int width, int height, Uint8 r, Uint8 g,
                                    Uint8 b) {
  SDL_Surface *surface =
      SDL_CreateSurface(width, height, SDL_PIXELFORMAT_RGBA8888);
  if (!surface)
    return nullptr;
  SDL_FillSurfaceRect(
      surface, NULL,
      SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format), 0, r, g, b, 255));
  return surface;
}

void setup_game(Engine *engine, GameState *game_state) {
  game_state->current_direction = RIGHT;
  game_state->queued_direction = RIGHT;
  game_state->move_timer = 0.0f;
  game_state->speed_boost_timer = 0.0f;
  game_state->invincibility_timer = 0.0f;

  // Create textures
  // Create textures - IMPORTANT: place at different atlas positions to avoid
  // overlap! Each texture goes at a different Y offset: 0, 32, 64, 96, 128
  SDL_Surface *head_surface =
      create_colored_surface(GRID_SIZE, GRID_SIZE, 0, 200, 0); // Green head
  game_state->head_texture_id =
      engine_register_texture(engine, head_surface, 0, 0, GRID_SIZE, GRID_SIZE);
  SDL_DestroySurface(head_surface);

  SDL_Surface *body_surface = create_colored_surface(
      GRID_SIZE, GRID_SIZE, 0, 150, 0); // Darker green body
  game_state->body_texture_id = engine_register_texture(
      engine, body_surface, 0, GRID_SIZE, GRID_SIZE, GRID_SIZE);
  SDL_DestroySurface(body_surface);

  // Create 10 food textures with different colors (gradient from red to violet)
  // Colors progress: red -> orange -> yellow -> lime -> green -> cyan -> blue
  // -> purple -> magenta -> white
  struct {
    uint8_t r, g, b;
  } food_colors[NUM_FOOD_TYPES] = {
      {255, 50, 50},   // Type 0: Red (small, +1)
      {255, 128, 0},   // Type 1: Orange (+2)
      {255, 200, 0},   // Type 2: Yellow (+3)
      {180, 255, 0},   // Type 3: Lime (+4)
      {0, 200, 80},    // Type 4: Green (+5)
      {0, 220, 220},   // Type 5: Cyan (+6)
      {50, 100, 255},  // Type 6: Blue (+7)
      {150, 50, 255},  // Type 7: Purple (+8)
      {255, 100, 200}, // Type 8: Pink (+9)
      {255, 255, 255}  // Type 9: White (legendary, +10)
  };

  for (int t = 0; t < NUM_FOOD_TYPES; ++t) {
    SDL_Surface *food_surface =
        create_colored_surface(GRID_SIZE, GRID_SIZE, food_colors[t].r,
                               food_colors[t].g, food_colors[t].b);
    // Place each food texture at a different Y position: starting at y=64, each
    // 32 apart
    game_state->food_texture_ids[t] = engine_register_texture(
        engine, food_surface, 0, GRID_SIZE * (2 + t), GRID_SIZE, GRID_SIZE);
    SDL_DestroySurface(food_surface);
  }

  // Enemy at y = GRID_SIZE * 12 (after 10 food textures)
  SDL_Surface *enemy_surface = create_colored_surface(
      GRID_SIZE, GRID_SIZE, 200, 50, 200); // Purple enemies
  game_state->enemy_texture_id = engine_register_texture(
      engine, enemy_surface, 0, GRID_SIZE * 12, GRID_SIZE, GRID_SIZE);
  SDL_DestroySurface(enemy_surface);

  // Power-up at y = GRID_SIZE * 13
  SDL_Surface *powerup_surface = create_colored_surface(
      GRID_SIZE, GRID_SIZE, 255, 255, 0); // Yellow power-ups
  game_state->powerup_texture_id = engine_register_texture(
      engine, powerup_surface, 0, GRID_SIZE * 13, GRID_SIZE, GRID_SIZE);
  SDL_DestroySurface(powerup_surface);

  // Create snake head at center
  float start_x = WORLD_WIDTH / 2.0f;
  float start_y = WORLD_HEIGHT / 2.0f;
  game_state->head_entity_index = game_state->head_container->createEntity(
      start_x, start_y, game_state->head_texture_id);

  // Initialize logic and visual positions
  game_state->head_logic_x = start_x;
  game_state->head_logic_y = start_y;
  game_state->head_visual_x = start_x;
  game_state->head_visual_y = start_y;

  std::cout << "SNAKE HEAD CREATED: index=" << game_state->head_entity_index
            << " at (" << start_x << ", " << start_y << ")"
            << " head_container count=" << game_state->head_container->count
            << std::endl;

  // Create initial body segments
  for (int i = 1; i <= INITIAL_SNAKE_LENGTH; ++i) {
    float seg_x = start_x - i * GRID_SIZE;
    float seg_y = start_y;
    uint32_t seg_idx = game_state->body_container->createEntity(
        seg_x, seg_y, game_state->body_texture_id);
    // Initialize with visual_x/y same as logic position
    game_state->snake_body.push_back({seg_x, seg_y, seg_x, seg_y, seg_idx});
    std::cout << "BODY SEG " << i << ": index=" << seg_idx << " at (" << seg_x
              << ", " << seg_y << ")" << std::endl;
  }
  std::cout << "Total body segments: " << game_state->snake_body.size()
            << " body_container count=" << game_state->body_container->count
            << std::endl;

  // Create foods distributed across the world - 10 different types
  // Type 0-9: growth = type+1 (1-10 segments), points = (type+1)*10 (10-100
  // points)
  for (int i = 0; i < NUM_FOODS; ++i) {
    float x = static_cast<float>(rand() % (WORLD_WIDTH - GRID_SIZE));
    float y = static_cast<float>(rand() % (WORLD_HEIGHT - GRID_SIZE));
    int food_type = rand() % NUM_FOOD_TYPES; // 0-9
    int growth = food_type + 1;              // 1-10 segments
    int value = growth * 10;                 // 10-100 points
    game_state->food_container->createEntity(
        x, y, game_state->food_texture_ids[food_type], value, growth,
        food_type);
  }

  // Create enemies distributed across the world
  for (int i = 0; i < NUM_ENEMIES; ++i) {
    float x = static_cast<float>(rand() % (WORLD_WIDTH - GRID_SIZE));
    float y = static_cast<float>(rand() % (WORLD_HEIGHT - GRID_SIZE));
    float speed = 30.0f + static_cast<float>(rand() % 70); // 30-100 speed
    game_state->enemy_container->createEntity(
        x, y, game_state->enemy_texture_id, speed);
  }

  // Create power-ups
  for (int i = 0; i < NUM_POWER_UPS; ++i) {
    float x = static_cast<float>(rand() % (WORLD_WIDTH - GRID_SIZE));
    float y = static_cast<float>(rand() % (WORLD_HEIGHT - GRID_SIZE));
    int type = rand() % 3; // 0=speed, 1=invincibility, 2=multiplier
    game_state->powerup_container->createEntity(
        x, y, game_state->powerup_texture_id, type);
  }

  // Log entity counts
  std::cout << "=== Game Setup Complete ===" << std::endl;
  std::cout << "  Food:     " << game_state->food_container->count << std::endl;
  std::cout << "  Enemies:  " << game_state->enemy_container->count
            << std::endl;
  std::cout << "  Power-ups:" << game_state->powerup_container->count
            << std::endl;
  std::cout << "  Snake:    " << (game_state->snake_body.size() + 1)
            << " segments" << std::endl;
  std::cout << "  World:    " << WORLD_WIDTH << "x" << WORLD_HEIGHT
            << std::endl;
}

void handle_input(const bool *keyboard_state, GameState *game_state) {
  Direction new_dir = game_state->current_direction;

  if (keyboard_state[SDL_SCANCODE_W] || keyboard_state[SDL_SCANCODE_UP]) {
    if (game_state->current_direction != DOWN) {
      new_dir = UP;
      std::cout << "INPUT: UP" << std::endl;
    }
  }
  if (keyboard_state[SDL_SCANCODE_S] || keyboard_state[SDL_SCANCODE_DOWN]) {
    if (game_state->current_direction != UP) {
      new_dir = DOWN;
      std::cout << "INPUT: DOWN" << std::endl;
    }
  }
  if (keyboard_state[SDL_SCANCODE_A] || keyboard_state[SDL_SCANCODE_LEFT]) {
    if (game_state->current_direction != RIGHT) {
      new_dir = LEFT;
      std::cout << "INPUT: LEFT" << std::endl;
    }
  }
  if (keyboard_state[SDL_SCANCODE_D] || keyboard_state[SDL_SCANCODE_RIGHT]) {
    if (game_state->current_direction != LEFT) {
      new_dir = RIGHT;
      std::cout << "INPUT: RIGHT" << std::endl;
    }
  }

  game_state->queued_direction = new_dir;
}

void update_snake(Engine *engine, GameState *game_state, float delta_time) {
  if (game_state->head_entity_index == INVALID_ID)
    return;

  SnakeHeadContainer *hCont = game_state->head_container;
  SnakeBodyContainer *bCont = game_state->body_container;
  uint32_t head_idx = game_state->head_entity_index;

  // Speed calculation - discrete steps per second
  float base_speed = SNAKE_SPEED;
  if (game_state->speed_boost_timer > 0)
    base_speed *= 1.5f;

  game_state->move_timer += delta_time * base_speed;

  // Only move when timer reaches 1.0 (one step)
  if (game_state->move_timer >= 1.0f) {
    game_state->move_timer -= 1.0f;

    // Apply queued direction
    game_state->current_direction = game_state->queued_direction;

    // Get current head LOGIC position (from GameState, not container)
    float head_x = game_state->head_logic_x;
    float head_y = game_state->head_logic_y;

    // Store old position for body
    float old_head_x = head_x;
    float old_head_y = head_y;

    // Move head one grid step
    switch (game_state->current_direction) {
    case UP:
      head_y -= GRID_SIZE;
      break;
    case DOWN:
      head_y += GRID_SIZE;
      break;
    case LEFT:
      head_x -= GRID_SIZE;
      break;
    case RIGHT:
      head_x += GRID_SIZE;
      break;
    }

    // Wrap around world boundaries
    if (head_x < 0)
      head_x += WORLD_WIDTH;
    else if (head_x >= WORLD_WIDTH)
      head_x -= WORLD_WIDTH;
    if (head_y < 0)
      head_y += WORLD_HEIGHT;
    else if (head_y >= WORLD_HEIGHT)
      head_y -= WORLD_HEIGHT;

    // Update head LOGIC position in GameState
    game_state->head_logic_x = head_x;
    game_state->head_logic_y = head_y;

    // Update head grid position for collision detection
    uint16_t newCellX = static_cast<uint16_t>(head_x * INV_GRID_CELL_SIZE);
    uint16_t newCellY = static_cast<uint16_t>(head_y * INV_GRID_CELL_SIZE);
    if (hCont->cell_x[head_idx] != newCellX ||
        hCont->cell_y[head_idx] != newCellY) {
      engine->grid.move(hCont->grid_node_indices[head_idx], head_x, head_y);
      hCont->cell_x[head_idx] = newCellX;
      hCont->cell_y[head_idx] = newCellY;
    }

    // Move body segments - each follows where the previous one WAS
    if (!game_state->snake_body.empty()) {
      float prev_x = old_head_x;
      float prev_y = old_head_y;

      for (auto &seg : game_state->snake_body) {
        float old_seg_x = seg.x;
        float old_seg_y = seg.y;
        seg.x = prev_x;
        seg.y = prev_y;

        if (seg.entity_index < bCont->count) {
          bCont->x_positions[seg.entity_index] = seg.x;
          bCont->y_positions[seg.entity_index] = seg.y;

          uint16_t segCellX = static_cast<uint16_t>(seg.x * INV_GRID_CELL_SIZE);
          uint16_t segCellY = static_cast<uint16_t>(seg.y * INV_GRID_CELL_SIZE);
          if (bCont->cell_x[seg.entity_index] != segCellX ||
              bCont->cell_y[seg.entity_index] != segCellY) {
            engine->grid.move(bCont->grid_node_indices[seg.entity_index], seg.x,
                              seg.y);
            bCont->cell_x[seg.entity_index] = segCellX;
            bCont->cell_y[seg.entity_index] = segCellY;
          }
        }

        prev_x = old_seg_x;
        prev_y = old_seg_y;
      }
    }
  }

  // Visual interpolation - smooth lerp toward logic positions every frame
  float lerp_speed = 20.0f; // How fast visuals catch up (higher = faster)
  float lerp_factor = std::min(1.0f, lerp_speed * delta_time);

  // Lerp head visual toward logic position (from GameState)
  game_state->head_visual_x +=
      (game_state->head_logic_x - game_state->head_visual_x) * lerp_factor;
  game_state->head_visual_y +=
      (game_state->head_logic_y - game_state->head_visual_y) * lerp_factor;

  // Update head render position (using visual pos)
  hCont->x_positions[head_idx] = game_state->head_visual_x;
  hCont->y_positions[head_idx] = game_state->head_visual_y;

  // Lerp body segment visuals toward their logic positions
  for (auto &seg : game_state->snake_body) {
    seg.visual_x += (seg.x - seg.visual_x) * lerp_factor;
    seg.visual_y += (seg.y - seg.visual_y) * lerp_factor;

    if (seg.entity_index < bCont->count) {
      bCont->x_positions[seg.entity_index] = seg.visual_x;
      bCont->y_positions[seg.entity_index] = seg.visual_y;
    }
  }

  // Update camera to follow visual head position (smooth)
  engine->camera.x = game_state->head_visual_x + GRID_SIZE / 2.0f;
  engine->camera.y = game_state->head_visual_y + GRID_SIZE / 2.0f;
}

void check_collisions(Engine *engine, GameState *game_state) {
  if (game_state->head_entity_index == INVALID_ID)
    return;

  SnakeHeadContainer *hCont = game_state->head_container;
  FoodContainer *fCont = game_state->food_container;
  EnemyContainer *eCont = game_state->enemy_container;
  PowerUpContainer *pCont = game_state->powerup_container;
  SnakeBodyContainer *bCont = game_state->body_container;

  uint32_t head_idx = game_state->head_entity_index;
  float head_x = hCont->x_positions[head_idx];
  float head_y = hCont->y_positions[head_idx];

  // Query nearby entities - use GRID_CELL_SIZE to ensure we cover adjacent
  // cells The queryCircle checks distance to cell corners, so we need radius >=
  // cell size
  float query_range = static_cast<float>(GRID_CELL_SIZE) * 1.5f;
  const auto &nearby = engine->grid.queryCircle(
      head_x + GRID_SIZE / 2, head_y + GRID_SIZE / 2, query_range);

  std::vector<EntityRef> to_remove;

  for (const auto &entity : nearby) {
    if (entity.type == ENTITY_TYPE_FOOD) {
      if (entity.index >= fCont->count)
        continue;

      float fx = fCont->x_positions[entity.index];
      float fy = fCont->y_positions[entity.index];

      // Simple collision (same grid cell)
      if (fabsf(head_x - fx) < GRID_SIZE && fabsf(head_y - fy) < GRID_SIZE) {
        // Eat food - add score
        game_state->score +=
            fCont->values[entity.index] * game_state->score_multiplier;

        // Grow snake by food's growth value (1-10 segments based on type)
        int segments_to_add = fCont->growth[entity.index];
        for (int seg = 0; seg < segments_to_add; ++seg) {
          float tail_x, tail_y;
          if (game_state->snake_body.empty()) {
            tail_x = head_x;
            tail_y = head_y;
          } else {
            tail_x = game_state->snake_body.back().x;
            tail_y = game_state->snake_body.back().y;
          }
          uint32_t new_seg =
              bCont->createEntity(tail_x, tail_y, game_state->body_texture_id);
          // Register new body segment with grid
          EntityRef seg_ref = {ENTITY_TYPE_SNAKE_BODY, new_seg};
          bCont->grid_node_indices[new_seg] =
              engine->grid.add(seg_ref, tail_x, tail_y);
          bCont->cell_x[new_seg] =
              static_cast<uint16_t>(tail_x * INV_GRID_CELL_SIZE);
          bCont->cell_y[new_seg] =
              static_cast<uint16_t>(tail_y * INV_GRID_CELL_SIZE);
          game_state->snake_body.push_back(
              {tail_x, tail_y, tail_x, tail_y, new_seg});
        }

        // RELOCATE food with new random type (engine constraint: no removal)
        float new_x = static_cast<float>(rand() % (WORLD_WIDTH - GRID_SIZE));
        float new_y = static_cast<float>(rand() % (WORLD_HEIGHT - GRID_SIZE));
        int new_food_type = rand() % NUM_FOOD_TYPES;
        int new_growth = new_food_type + 1;
        int new_value = new_growth * 10;

        fCont->x_positions[entity.index] = new_x;
        fCont->y_positions[entity.index] = new_y;
        fCont->values[entity.index] = new_value;
        fCont->growth[entity.index] = new_growth;
        fCont->food_types[entity.index] = new_food_type;
        fCont->texture_ids[entity.index] =
            game_state->food_texture_ids[new_food_type];

        // Update grid position
        int32_t nodeIdx = fCont->grid_node_indices[entity.index];
        engine->grid.move(nodeIdx, new_x, new_y);
        fCont->cell_x[entity.index] =
            static_cast<uint16_t>(new_x * INV_GRID_CELL_SIZE);
        fCont->cell_y[entity.index] =
            static_cast<uint16_t>(new_y * INV_GRID_CELL_SIZE);
      }
    } else if (entity.type == ENTITY_TYPE_ENEMY) {
      if (game_state->invincibility_timer > 0)
        continue; // Invincible
      if (entity.index >= eCont->count)
        continue;

      float ex = eCont->x_positions[entity.index];
      float ey = eCont->y_positions[entity.index];

      if (fabsf(head_x - ex) < GRID_SIZE * 0.8f &&
          fabsf(head_y - ey) < GRID_SIZE * 0.8f) {
        game_state->is_alive = false;
        std::cout << "Game Over! Final Score: " << game_state->score
                  << std::endl;
      }
    } else if (entity.type == ENTITY_TYPE_POWER_UP) {
      if (entity.index >= pCont->count)
        continue;

      float px = pCont->x_positions[entity.index];
      float py = pCont->y_positions[entity.index];

      if (fabsf(head_x - px) < GRID_SIZE && fabsf(head_y - py) < GRID_SIZE) {
        int type = pCont->types[entity.index];
        switch (type) {
        case 0: // Speed boost
          game_state->speed_boost_timer = 5.0f;
          break;
        case 1: // Invincibility
          game_state->invincibility_timer = 5.0f;
          break;
        case 2: // Score multiplier
          game_state->score_multiplier = 3;
          game_state->multiplier_timer = 10.0f;
          break;
        }

        // RELOCATE power-up instead of removing
        float new_x = static_cast<float>(rand() % (WORLD_WIDTH - GRID_SIZE));
        float new_y = static_cast<float>(rand() % (WORLD_HEIGHT - GRID_SIZE));
        pCont->x_positions[entity.index] = new_x;
        pCont->y_positions[entity.index] = new_y;
        pCont->types[entity.index] = rand() % 3; // New random type

        // Update grid position
        int32_t nodeIdx = pCont->grid_node_indices[entity.index];
        engine->grid.move(nodeIdx, new_x, new_y);
        pCont->cell_x[entity.index] =
            static_cast<uint16_t>(new_x * INV_GRID_CELL_SIZE);
        pCont->cell_y[entity.index] =
            static_cast<uint16_t>(new_y * INV_GRID_CELL_SIZE);
      }
    } else if (entity.type == ENTITY_TYPE_SNAKE_BODY) {
      // Self-collision (skip first few segments)
      if (entity.index >= bCont->count)
        continue;

      // Find segment in deque
      bool is_near_head = false;
      int seg_count = 0;
      for (const auto &seg : game_state->snake_body) {
        if (seg.entity_index == entity.index) {
          is_near_head = (seg_count < 3); // Skip first 3 segments
          break;
        }
        seg_count++;
      }

      if (!is_near_head && game_state->invincibility_timer <= 0) {
        float bx = bCont->x_positions[entity.index];
        float by = bCont->y_positions[entity.index];

        if (fabsf(head_x - bx) < GRID_SIZE * 0.5f &&
            fabsf(head_y - by) < GRID_SIZE * 0.5f) {
          game_state->is_alive = false;
          std::cout << "Game Over! You hit yourself! Final Score: "
                    << game_state->score << std::endl;
        }
      }
    }
  }

  // No removals needed - food and power-ups are relocated
}

void spawn_food(Engine *engine, GameState *game_state) {
  float x = static_cast<float>(rand() % (WORLD_WIDTH - GRID_SIZE));
  float y = static_cast<float>(rand() % (WORLD_HEIGHT - GRID_SIZE));
  int food_type = rand() % NUM_FOOD_TYPES;
  int growth = food_type + 1;
  int value = growth * 10;
  uint32_t food_idx = game_state->food_container->createEntity(
      x, y, game_state->food_texture_ids[food_type], value, growth, food_type);

  // Register new food with grid
  FoodContainer *fCont = game_state->food_container;
  EntityRef food_ref = {ENTITY_TYPE_FOOD, food_idx};
  fCont->grid_node_indices[food_idx] = engine->grid.add(food_ref, x, y);
  fCont->cell_x[food_idx] = static_cast<uint16_t>(x * INV_GRID_CELL_SIZE);
  fCont->cell_y[food_idx] = static_cast<uint16_t>(y * INV_GRID_CELL_SIZE);
}
