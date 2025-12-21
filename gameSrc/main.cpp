#include "../game/ATMEngine.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <execution>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

// --- Constants ---
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1020
#define NUM_OBSTACLES 1000000
#define NUM_ITEMS 100
#define PLAYER_SIZE 32
#define OBSTACLE_SIZE 32
#define ITEM_SIZE 24
#define PLAYER_SPEED 350.0f
#define PLAYER_MAX_HEALTH 100.0f
#define OBSTACLE_DAMAGE 10.0f

// --- Game-specific entity types ---
enum GameEntityTypes {
  ENTITY_TYPE_PLAYER = 0,
  ENTITY_TYPE_OBSTACLE,
  ENTITY_TYPE_ITEM,
  ENTITY_TYPE_COUNT
};

// --- Player Entity Container ---
class PlayerEntityContainer : public RenderableEntityContainer {
public:
  float *speeds;
  float *health;
  bool *isMoving;

  PlayerEntityContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity) {
    speeds = new float[capacity];
    health = new float[capacity];
    isMoving = new bool[capacity];
    std::fill(speeds, speeds + capacity, 0.0f);
    std::fill(health, health + capacity, 0.0f);
    std::fill(isMoving, isMoving + capacity, false);
  }

  ~PlayerEntityContainer() override {
    delete[] speeds;
    delete[] health;
    delete[] isMoving;
  }

  uint32_t createEntity() override {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    speeds[index] = PLAYER_SPEED;
    health[index] = PLAYER_MAX_HEALTH;
    isMoving[index] = false;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 10;
    return index;
  }

  uint32_t createEntity(float x, float y, int width, int height,
                        int texture_id) {
    uint32_t index = createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = width;
    heights[index] = height;
    texture_ids[index] = texture_id;

    return index;
  }

  void removeEntity(size_t index) override {
    if (index >= count)
      return;
    size_t last = count - 1;
    if (index < last) {
      speeds[index] = speeds[last];
      health[index] = health[last];
      isMoving[index] = isMoving[last];
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

    float *newSpeeds = new float[newCapacity];
    float *newHealth = new float[newCapacity];
    bool *newIsMoving = new bool[newCapacity];

    if (count > 0) {
      std::copy(speeds, speeds + count, newSpeeds);
      std::copy(health, health + count, newHealth);
      std::copy(isMoving, isMoving + count, newIsMoving);
    }
    std::fill(newSpeeds + count, newSpeeds + newCapacity, 0.0f);
    std::fill(newHealth + count, newHealth + newCapacity, 0.0f);
    std::fill(newIsMoving + count, newIsMoving + newCapacity, false);

    delete[] speeds;
    delete[] health;
    delete[] isMoving;

    speeds = newSpeeds;
    health = newHealth;
    isMoving = newIsMoving;

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Item Entity Container ---
class ItemEntityContainer : public RenderableEntityContainer {
public:
  int *item_types;
  int *values;

  ItemEntityContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity) {
    item_types = new int[capacity];
    values = new int[capacity];
    std::fill(item_types, item_types + capacity, 0);
    std::fill(values, values + capacity, 0);
  }

  ~ItemEntityContainer() override {
    delete[] item_types;
    delete[] values;
  }

  uint32_t createEntity() override {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;
    item_types[index] = 0;
    values[index] = 10;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 1;
    return index;
  }

  uint32_t createEntity(float x, float y, int width, int height, int texture_id,
                        int value, int type = 0) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = width;
    heights[index] = height;
    texture_ids[index] = texture_id;
    values[index] = value;
    item_types[index] = type;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 1;

    return index;
  }

  void removeEntity(size_t index) override {
    if (index >= count)
      return;
    size_t last = count - 1;
    if (index < last) {
      item_types[index] = item_types[last];
      values[index] = values[last];
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
    int *newItemTypes = new int[newCapacity];
    int *newValues = new int[newCapacity];
    if (count > 0) {
      std::copy(item_types, item_types + count, newItemTypes);
      std::copy(values, values + count, newValues);
    }
    std::fill(newItemTypes + count, newItemTypes + newCapacity, 0);
    std::fill(newValues + count, newValues + newCapacity, 0);
    delete[] item_types;
    delete[] values;
    item_types = newItemTypes;
    values = newValues;
    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Obstacle Entity Container ---
class ObstacleEntityContainer : public RenderableEntityContainer {
public:
  int *obstacle_types;
  bool *damaging;
  float *speeds;
  float *dir_x;
  float *dir_y;
  Engine *engine;

  ObstacleEntityContainer(Engine *engine, int typeId, uint8_t defaultLayer,
                          int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity),
        engine(engine) {
    obstacle_types = new int[capacity];
    damaging = new bool[capacity];
    speeds = new float[capacity];
    dir_x = new float[capacity];
    dir_y = new float[capacity];
    std::fill(obstacle_types, obstacle_types + capacity, 0);
    std::fill(damaging, damaging + capacity, false);
    std::fill(speeds, speeds + capacity, 0.0f);
    std::fill(dir_x, dir_x + capacity, 0.0f);
    std::fill(dir_y, dir_y + capacity, 0.0f);
  }

  ~ObstacleEntityContainer() override {
    delete[] obstacle_types;
    delete[] damaging;
    delete[] speeds;
    delete[] dir_x;
    delete[] dir_y;
  }

  uint32_t createEntity() override {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    obstacle_types[index] = 0;
    damaging[index] = (rand() % 4 == 0); // 25% chance
    speeds[index] =
        50.0f + static_cast<float>(rand() % 101); // 50-150 pixels/sec

    // Random initial direction vector (uniform random angle)
    float angle = static_cast<float>(rand()) / RAND_MAX * 2.0f * 3.14159265f;
    dir_x[index] = cosf(angle);
    dir_y[index] = sinf(angle);

    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 5;

    return index;
  }

  uint32_t createEntity(float x, float y, int width, int height, int texture_id,
                        bool is_damaging = false, float speed = 50.0f,
                        float direction_x = 150.0f, float direction_y = 0.0f) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = width;
    heights[index] = height;
    texture_ids[index] = texture_id;
    obstacle_types[index] = 0;
    damaging[index] = is_damaging;
    speeds[index] = speed;

    // Normalize direction
    float len = sqrt(direction_x * direction_x + direction_y * direction_y);
    dir_x[index] = (len > 0.001f) ? direction_x / len : 1.0f;
    dir_y[index] = (len > 0.001f) ? direction_y / len : 0.0f;

    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 5;
    return index;
  }

  void removeEntity(size_t index) override {
    if (index >= count)
      return;
    size_t last = count - 1;
    if (index < last) {
      obstacle_types[index] = obstacle_types[last];
      damaging[index] = damaging[last];
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

    // Define a vector of indices for parallel processing
    static std::vector<uint32_t> indices;
    if (indices.size() < (size_t)count) {
      indices.resize(count);
      std::iota(indices.begin(), indices.end(), 0);
    }

    // Buffer for deferred grid updates (thread-safe collection)
    static std::vector<uint32_t> pending_moves;
    static std::atomic<uint32_t> pending_count{0};
    if (pending_moves.size() < (size_t)count) {
      pending_moves.resize(count);
    }
    pending_count.store(0, std::memory_order_relaxed);

    // Parallel phase: update positions, collect entities that changed cells
    std::for_each(std::execution::par, indices.begin(), indices.begin() + count,
                  [&](uint32_t i) {
                    float &px = x_positions[i];
                    float &py = y_positions[i];
                    float &dx = dir_x[i];
                    float &dy = dir_y[i];
                    const float speed = speeds[i] * delta_time;

                    // Store old cell
                    uint16_t oldCellX = cell_x[i];
                    uint16_t oldCellY = cell_y[i];

                    // Move entity
                    px += dx * speed;
                    py += dy * speed;

                    // Fast boundary checks
                    if (px < 0) {
                      px = 0;
                      dx = -dx;
                    } else if (px > WORLD_WIDTH - OBSTACLE_SIZE) {
                      px = WORLD_WIDTH - OBSTACLE_SIZE;
                      dx = -dx;
                    }

                    if (py < 0) {
                      py = 0;
                      dy = -dy;
                    } else if (py > WORLD_HEIGHT - OBSTACLE_SIZE) {
                      py = WORLD_HEIGHT - OBSTACLE_SIZE;
                      dy = -dy;
                    }

                    // Compute new cell
                    uint16_t newCellX =
                        static_cast<uint16_t>(px * INV_GRID_CELL_SIZE);
                    uint16_t newCellY =
                        static_cast<uint16_t>(py * INV_GRID_CELL_SIZE);

                    // If cell changed, add to pending buffer (atomic)
                    if (oldCellX != newCellX || oldCellY != newCellY) {
                      uint32_t slot =
                          pending_count.fetch_add(1, std::memory_order_relaxed);
                      pending_moves[slot] = i;
                      cell_x[i] = newCellX;
                      cell_y[i] = newCellY;
                    }
                  });

    // Serial phase: apply grid moves (safe - no concurrent access)
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

    int *newObstacleTypes = new int[newCapacity];
    bool *newDamaging = new bool[newCapacity];
    float *newSpeeds = new float[newCapacity];
    float *newDirX = new float[newCapacity];
    float *newDirY = new float[newCapacity];

    if (count > 0) {
      std::copy(obstacle_types, obstacle_types + count, newObstacleTypes);
      std::copy(damaging, damaging + count, newDamaging);
      std::copy(speeds, speeds + count, newSpeeds);
      std::copy(dir_x, dir_x + count, newDirX);
      std::copy(dir_y, dir_y + count, newDirY);
    }
    std::fill(newObstacleTypes + count, newObstacleTypes + newCapacity, 0);
    std::fill(newDamaging + count, newDamaging + newCapacity, false);
    std::fill(newSpeeds + count, newSpeeds + newCapacity, 0.0f);
    std::fill(newDirX + count, newDirX + newCapacity, 0.0f);
    std::fill(newDirY + count, newDirY + newCapacity, 0.0f);

    delete[] obstacle_types;
    delete[] damaging;
    delete[] speeds;
    delete[] dir_x;
    delete[] dir_y;

    obstacle_types = newObstacleTypes;
    damaging = newDamaging;
    speeds = newSpeeds;
    dir_x = newDirX;
    dir_y = newDirY;

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Game State ---
struct GameState {
  uint32_t player_entity_index;
  int player_texture_id;
  int obstacle_texture_id;
  int item_texture_id;
  int score;
  bool game_over;
  bool game_won;
  int items_collected;
  int total_items;
  PlayerEntityContainer *player_container;
  ObstacleEntityContainer *obstacle_container;
  ItemEntityContainer *item_container;
  Uint64 last_fps_time;
  int frame_count;
  float current_fps;
};

// --- Function Declarations ---
void setup_game(Engine *engine, GameState *game_state);
void handle_input(const Uint8 *keyboard_state, Engine *engine,
                  GameState *game_state, float delta_time);
void check_collisions(Engine *engine, GameState *game_state);
void reset_game(Engine *engine, GameState *game_state);
SDL_Surface *create_colored_surface(int width, int height, Uint8 r, Uint8 g,
                                    Uint8 b);

// --- Main Function ---
int main(int argc, char *argv[]) {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
    return 1;
  }
  srand(static_cast<unsigned int>(time(nullptr)));

  SDL_Window *window =
      SDL_CreateWindow("2D Game Engine - SDL3", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
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

  // --- Register Entity Types ---
  PlayerEntityContainer *player_container =
      new PlayerEntityContainer(ENTITY_TYPE_PLAYER, 0, 10);
  ObstacleEntityContainer *obstacle_container = new ObstacleEntityContainer(
      engine, ENTITY_TYPE_OBSTACLE, 0, NUM_OBSTACLES + 100);
  ItemEntityContainer *item_container =
      new ItemEntityContainer(ENTITY_TYPE_ITEM, 0, NUM_ITEMS + 100);

  engine->entityManager.registerEntityType(player_container);
  engine->entityManager.registerEntityType(obstacle_container);
  engine->entityManager.registerEntityType(item_container);

  // --- Initialize Game State ---
  GameState game_state = {};
  game_state.player_container = player_container;
  game_state.obstacle_container = obstacle_container;
  game_state.item_container = item_container;
  game_state.player_entity_index = INVALID_ID;
  game_state.last_fps_time = SDL_GetTicks();
  game_state.frame_count = 0;
  game_state.current_fps = 0.0f;

  setup_game(engine, &game_state);

  // --- Initial Grid Population ---
  // Populate grid and set initial cell tracking for all entities
  engine->grid.rebuild_grid(engine);

  // Initialize cell tracking for all entities
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

  if (game_state.player_entity_index != INVALID_ID) {
    // Set camera position to player
    float player_x =
        player_container->x_positions[game_state.player_entity_index];
    float player_y =
        player_container->y_positions[game_state.player_entity_index];
    engine->camera.x = player_x + PLAYER_SIZE / 2.0f;
    engine->camera.y = player_y + PLAYER_SIZE / 2.0f;
    player_container->flags[game_state.player_entity_index] |=
        static_cast<uint8_t>(EntityFlag::VISIBLE);
  } else {
    engine->camera.x = WORLD_WIDTH / 2.0f;
    engine->camera.y = WORLD_HEIGHT / 2.0f;
  }

  // --- Game Loop ---
  bool quit = false;
  SDL_Event event;
  Uint64 last_time = SDL_GetTicks();
  int frameCounter = 0;

  while (!quit) {
    Uint64 current_time = SDL_GetTicks();
    float delta_time = std::min((current_time - last_time) / 1000.0f, 0.1f);
    last_time = current_time;

    // Calculate FPS
    game_state.frame_count++;
    Uint64 time_since_last_fps = current_time - game_state.last_fps_time;
    if (time_since_last_fps >= 1000) {
      game_state.current_fps =
          static_cast<float>(game_state.frame_count * 1000.0f) /
          static_cast<float>(time_since_last_fps);
      game_state.last_fps_time = current_time;
      game_state.frame_count = 0;
      std::cout << "FPS: " << game_state.current_fps << std::endl;
    }
    const Uint8 *keyboard_state = (uint8_t *)SDL_GetKeyboardState(NULL);

    // Process Events
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_QUIT)
        quit = true;
      else if (event.type == SDL_EVENT_KEY_DOWN) {
        if (event.key.scancode == SDL_SCANCODE_ESCAPE)
          quit = true;
        else if (event.key.scancode == SDL_SCANCODE_R &&
                 (game_state.game_over || game_state.game_won)) {
          reset_game(engine, &game_state);
        }
      }
    }

    // Handle Input and Game Logic
    if (!game_state.game_over && !game_state.game_won) {
      handle_input(keyboard_state, engine, &game_state, delta_time);

      if (frameCounter > 20) { // Start collisions after initial frames
        check_collisions(engine, &game_state);

        if (game_state.player_entity_index != INVALID_ID &&
            game_state.player_container
                    ->health[game_state.player_entity_index] <= 0) {
          game_state.game_over = true;
        }

        if (game_state.items_collected >= game_state.total_items) {
          game_state.game_won = true;
        }
      }
    }

    engine_update(engine);

    // Rendering
    SDL_SetRenderDrawColor(engine->renderer, 0, 0, 0, 255);
    SDL_RenderClear(engine->renderer);
    engine_render_scene(engine);
    engine_present(engine);

    frameCounter++;
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
  game_state->score = 0;
  game_state->game_over = false;
  game_state->game_won = false;
  game_state->items_collected = 0;
  game_state->total_items = NUM_ITEMS;
  game_state->player_entity_index = INVALID_ID;

  // Register Textures
  SDL_Surface *player_surface =
      create_colored_surface(PLAYER_SIZE, PLAYER_SIZE, 0, 0, 255);
  game_state->player_texture_id = engine_register_texture(
      engine, player_surface, 0, 0, PLAYER_SIZE, PLAYER_SIZE);
  SDL_DestroySurface(player_surface);

  SDL_Surface *obstacle_surface =
      create_colored_surface(OBSTACLE_SIZE, OBSTACLE_SIZE, 255, 0, 0);
  game_state->obstacle_texture_id = engine_register_texture(
      engine, obstacle_surface, 0, 0, OBSTACLE_SIZE, OBSTACLE_SIZE);
  SDL_DestroySurface(obstacle_surface);

  SDL_Surface *item_surface =
      create_colored_surface(ITEM_SIZE, ITEM_SIZE, 255, 255, 0);
  game_state->item_texture_id =
      engine_register_texture(engine, item_surface, 0, 0, ITEM_SIZE, ITEM_SIZE);
  SDL_DestroySurface(item_surface);

  // Create Player
  float start_x = WORLD_WIDTH / 2 - PLAYER_SIZE / 2;
  float start_y = WORLD_HEIGHT / 2 - PLAYER_SIZE / 2;
  game_state->player_entity_index = game_state->player_container->createEntity(
      start_x, start_y, PLAYER_SIZE, PLAYER_SIZE,
      game_state->player_texture_id);

  // Create Obstacles
  ObstacleEntityContainer *oCont = game_state->obstacle_container;

  // Divide world into grid for distribution
  const int dist_grid_width = 50;
  const int dist_grid_height = 50;
  const float cell_width = WORLD_WIDTH / (float)dist_grid_width;
  const float cell_height = WORLD_HEIGHT / (float)dist_grid_height;
  const int obstacles_per_cell =
      NUM_OBSTACLES / (dist_grid_width * dist_grid_height);
  const int obstacles_remainder =
      NUM_OBSTACLES % (dist_grid_width * dist_grid_height);

  float player_center_x = WORLD_WIDTH / 2.0f;
  float player_center_y = WORLD_HEIGHT / 2.0f;
  float safe_radius_sq = pow(PLAYER_SIZE * 10.0f, 2);

  for (int grid_y = 0; grid_y < dist_grid_height; ++grid_y) {
    for (int grid_x = 0; grid_x < dist_grid_width; ++grid_x) {
      int cell_obstacles = obstacles_per_cell;
      if ((grid_y * dist_grid_width + grid_x) < obstacles_remainder) {
        cell_obstacles++;
      }

      float cell_min_x = grid_x * cell_width;
      float cell_min_y = grid_y * cell_height;

      for (int i = 0; i < cell_obstacles; ++i) {
        float margin = OBSTACLE_SIZE * 0.5f;
        float x_pos = cell_min_x + margin +
                      (static_cast<float>(rand()) / RAND_MAX) *
                          (cell_width - OBSTACLE_SIZE - margin * 2);
        float y_pos = cell_min_y + margin +
                      (static_cast<float>(rand()) / RAND_MAX) *
                          (cell_height - OBSTACLE_SIZE - margin * 2);

        // Keep obstacles away from player
        float start_dist_sq =
            pow(x_pos - player_center_x, 2) + pow(y_pos - player_center_y, 2);
        if (start_dist_sq < safe_radius_sq) {
          float angle =
              atan2f(y_pos - player_center_y, x_pos - player_center_x);
          float safe_dist = PLAYER_SIZE * 10.0f + (rand() % 50);
          x_pos = player_center_x + cosf(angle) * safe_dist;
          y_pos = player_center_y + sinf(angle) * safe_dist;

          // If still problematic, try alternative position
          if ((pow(x_pos - player_center_x, 2) +
                   pow(y_pos - player_center_y, 2) <
               safe_radius_sq) ||
              x_pos < 0 || x_pos > WORLD_WIDTH - OBSTACLE_SIZE || y_pos < 0 ||
              y_pos > WORLD_HEIGHT - OBSTACLE_SIZE) {
            int quadrant = rand() % 4;
            switch (quadrant) {
            case 0: // Top-left
              x_pos = (float)(rand() % (WORLD_WIDTH / 4));
              y_pos = (float)(rand() % (WORLD_HEIGHT / 4));
              break;
            case 1: // Top-right
              x_pos = (float)(WORLD_WIDTH * 3 / 4 + rand() % (WORLD_WIDTH / 4));
              y_pos = (float)(rand() % (WORLD_HEIGHT / 4));
              break;
            case 2: // Bottom-left
              x_pos = (float)(rand() % (WORLD_WIDTH / 4));
              y_pos =
                  (float)(WORLD_HEIGHT * 3 / 4 + rand() % (WORLD_HEIGHT / 4));
              break;
            case 3: // Bottom-right
              x_pos = (float)(WORLD_WIDTH * 3 / 4 + rand() % (WORLD_WIDTH / 4));
              y_pos =
                  (float)(WORLD_HEIGHT * 3 / 4 + rand() % (WORLD_HEIGHT / 4));
              break;
            }
          }
        }

        // Clamp to world bounds
        x_pos =
            std::max(0.0f, std::min((float)WORLD_WIDTH - OBSTACLE_SIZE, x_pos));
        y_pos = std::max(0.0f,
                         std::min((float)WORLD_HEIGHT - OBSTACLE_SIZE, y_pos));

        bool is_damaging = (rand() % 4 == 0);
        float speed = 50.0f + static_cast<float>(rand() % 101);
        // Random direction for each obstacle
        float angle =
            static_cast<float>(rand()) / RAND_MAX * 2.0f * 3.14159265f;
        float dir_x = cosf(angle);
        float dir_y = sinf(angle);
        oCont->createEntity(x_pos, y_pos, OBSTACLE_SIZE, OBSTACLE_SIZE,
                            game_state->obstacle_texture_id, is_damaging, speed,
                            dir_x, dir_y);
      }
    }
  }

  // Create Items
  ItemEntityContainer *iCont = game_state->item_container;
  for (int i = 0; i < NUM_ITEMS; ++i) {
    float x_pos = rand() % (WORLD_WIDTH - ITEM_SIZE);
    float y_pos = rand() % (WORLD_HEIGHT - ITEM_SIZE);
    int value = 10 + (rand() % 11);
    iCont->createEntity(x_pos, y_pos, ITEM_SIZE, ITEM_SIZE,
                        game_state->item_texture_id, value);
  }
}

void handle_input(const Uint8 *keyboard_state, Engine *engine,
                  GameState *game_state, float delta_time) {
  if (game_state->player_entity_index == INVALID_ID)
    return;

  PlayerEntityContainer *pCont = game_state->player_container;
  uint32_t player_idx = game_state->player_entity_index;

  float speed_pixels_per_sec = pCont->speeds[player_idx];
  float move_norm_x = 0.0f;
  float move_norm_y = 0.0f;

  if (keyboard_state[SDL_SCANCODE_W] || keyboard_state[SDL_SCANCODE_UP])
    move_norm_y -= 1.0f;
  if (keyboard_state[SDL_SCANCODE_S] || keyboard_state[SDL_SCANCODE_DOWN])
    move_norm_y += 1.0f;
  if (keyboard_state[SDL_SCANCODE_A] || keyboard_state[SDL_SCANCODE_LEFT])
    move_norm_x -= 1.0f;
  if (keyboard_state[SDL_SCANCODE_D] || keyboard_state[SDL_SCANCODE_RIGHT])
    move_norm_x += 1.0f;

  // Camera zoom controls
  if (keyboard_state[SDL_SCANCODE_EQUALS] ||
      keyboard_state[SDL_SCANCODE_KP_PLUS]) {
    engine->camera.width *= 1.1;
    engine->camera.height *= 1.1;
  }
  if (keyboard_state[SDL_SCANCODE_MINUS] ||
      keyboard_state[SDL_SCANCODE_KP_MINUS]) {
    engine->camera.width *= 0.9;
    engine->camera.height *= 0.9;
  }

  pCont->isMoving[player_idx] = (move_norm_x != 0.0f || move_norm_y != 0.0f);
  pCont->flags[player_idx] |= static_cast<uint8_t>(EntityFlag::VISIBLE);

  if (pCont->isMoving[player_idx]) {
    // Normalize direction vector
    float length = sqrt(move_norm_x * move_norm_x + move_norm_y * move_norm_y);
    if (length > 0.001f) {
      move_norm_x /= length;
      move_norm_y /= length;
    } else {
      pCont->isMoving[player_idx] = false;
      return;
    }

    // Move player
    float move_x = move_norm_x * speed_pixels_per_sec * delta_time;
    float move_y = move_norm_y * speed_pixels_per_sec * delta_time;
    float current_x = pCont->x_positions[player_idx];
    float current_y = pCont->y_positions[player_idx];
    float next_x = current_x + move_x;
    float next_y = current_y + move_y;

    // Clamp to world bounds
    next_x = std::max(
        0.0f,
        std::min(static_cast<float>(WORLD_WIDTH - pCont->widths[player_idx]),
                 next_x));
    next_y = std::max(
        0.0f,
        std::min(static_cast<float>(WORLD_HEIGHT - pCont->heights[player_idx]),
                 next_y));
    pCont->x_positions[player_idx] = next_x;
    pCont->y_positions[player_idx] = next_y;

    // Update grid position for correct spatial queries
    uint16_t oldCellX = pCont->cell_x[player_idx];
    uint16_t oldCellY = pCont->cell_y[player_idx];
    uint16_t newCellX = static_cast<uint16_t>(next_x * INV_GRID_CELL_SIZE);
    uint16_t newCellY = static_cast<uint16_t>(next_y * INV_GRID_CELL_SIZE);
    if (oldCellX != newCellX || oldCellY != newCellY) {
      int32_t nodeIdx = pCont->grid_node_indices[player_idx];
      engine->grid.move(nodeIdx, next_x, next_y);
      pCont->cell_x[player_idx] = newCellX;
      pCont->cell_y[player_idx] = newCellY;
    }

    // Update camera
    engine->camera.x = next_x + pCont->widths[player_idx] / 2.0f;
    engine->camera.y = next_y + pCont->heights[player_idx] / 2.0f;
  }
}

void check_collisions(Engine *engine, GameState *game_state) {
  if (game_state->player_entity_index == INVALID_ID)
    return;

  PlayerEntityContainer *pCont = game_state->player_container;
  ObstacleEntityContainer *oCont = game_state->obstacle_container;
  ItemEntityContainer *iCont = game_state->item_container;
  uint32_t player_idx = game_state->player_entity_index;

  if (player_idx >= pCont->count)
    return;

  // Get player position
  float player_x = pCont->x_positions[player_idx];
  float player_y = pCont->y_positions[player_idx];
  int player_width = pCont->widths[player_idx];
  int player_height = pCont->heights[player_idx];

  // Query grid for nearby entities
  float player_pos_x = player_x + player_width / 2.0f;
  float player_pos_y = player_y + player_height / 2.0f;
  float query_range = std::max(player_width, player_height) * 2.0f;

  const auto &nearby_entities =
      engine->grid.queryCircle(player_pos_x, player_pos_y, query_range);
  std::vector<EntityRef> items_to_remove;

  // Check collision with each nearby entity
  for (const auto &entity : nearby_entities) {
    // Skip player itself
    if (entity.type == ENTITY_TYPE_PLAYER &&
        entity.index == static_cast<int32_t>(player_idx))
      continue;
    if (entity.type < 0 || entity.index < 0)
      continue;

    if (entity.type == ENTITY_TYPE_OBSTACLE) {
      if (entity.index >= oCont->count)
        continue;

      float obstacle_x = oCont->x_positions[entity.index];
      float obstacle_y = oCont->y_positions[entity.index];
      int obstacle_width = oCont->widths[entity.index];
      int obstacle_height = oCont->heights[entity.index];

      // AABB collision
      if (player_x < obstacle_x + obstacle_width &&
          player_x + player_width > obstacle_x &&
          player_y < obstacle_y + obstacle_height &&
          player_y + player_height > obstacle_y) {

        if (oCont->damaging[entity.index]) {
          pCont->health[player_idx] -= OBSTACLE_DAMAGE * 0.016f;
        }
      }
    } else if (entity.type == ENTITY_TYPE_ITEM) {
      if (entity.index >= iCont->count)
        continue;

      float item_x = iCont->x_positions[entity.index];
      float item_y = iCont->y_positions[entity.index];
      int item_width = iCont->widths[entity.index];
      int item_height = iCont->heights[entity.index];

      // AABB collision
      if (player_x < item_x + item_width && player_x + player_width > item_x &&
          player_y < item_y + item_height &&
          player_y + player_height > item_y) {

        game_state->score += iCont->values[entity.index];
        game_state->items_collected++;
        EntityRef item_ref = {ENTITY_TYPE_ITEM, entity.index};
        items_to_remove.push_back(item_ref);
        iCont->flags[entity.index] &=
            ~static_cast<uint8_t>(EntityFlag::VISIBLE);
      }
    }
  }

  // Remove collected items
  for (const auto &item_ref : items_to_remove) {
    engine->pending_removals.push_back(item_ref);
  }
}

void reset_game(Engine *engine, GameState *game_state) {
  // Clear existing entities
  for (int i = 0; i < game_state->item_container->count; i++) {
    engine->entityManager.removeEntity(i, ENTITY_TYPE_ITEM);
  }
  for (int i = 0; i < game_state->obstacle_container->count; i++) {
    engine->entityManager.removeEntity(i, ENTITY_TYPE_OBSTACLE);
  }
  if (game_state->player_entity_index != INVALID_ID) {
    engine->entityManager.removeEntity(game_state->player_entity_index,
                                       ENTITY_TYPE_PLAYER);
  }

  // Setup game again
  setup_game(engine, game_state);

  if (game_state->player_entity_index != INVALID_ID) {
    float player_x = game_state->player_container
                         ->x_positions[game_state->player_entity_index];
    float player_y = game_state->player_container
                         ->y_positions[game_state->player_entity_index];
    engine->camera.x = player_x + PLAYER_SIZE / 2.0f;
    engine->camera.y = player_y + PLAYER_SIZE / 2.0f;
    game_state->player_container->flags[game_state->player_entity_index] |=
        static_cast<uint8_t>(EntityFlag::VISIBLE);
  } else {
    engine->camera.x = WORLD_WIDTH / 2.0f;
    engine->camera.y = WORLD_HEIGHT / 2.0;
  }
}