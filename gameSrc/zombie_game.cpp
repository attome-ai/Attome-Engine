#include "../game/ATMEngine.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <execution>
#include <iostream>
#include <numeric>
#include <vector>

// --- Constants ---
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1020
#define PLAYER_SIZE 48
#define ZOMBIE_SIZE 40
#define BULLET_SIZE 8
#define DAMAGE_TEXT_SIZE 20
#define PLAYER_SPEED 300.0f
#define BULLET_SPEED 1200.0f
#define FIRE_RATE 0.04f // Seconds between shots (~25 bursts/sec)
#define FIXED_DAMAGE                                                           \
  1100 // Constant damage per bullet - one-shots most zombies!
#define NUM_ZOMBIES 1000000
#define MAX_BULLETS 8000
#define MAX_DAMAGE_TEXTS 500
#define DAMAGE_TEXT_LIFETIME 0.4f
#define DAMAGE_TEXT_FLOAT_SPEED 100.0f
#define BULLETS_PER_SHOT 11 // Shotgun spread!
#define SPREAD_ANGLE 0.4f   // Radians of spread (~23 degrees)

// --- Zombie Type Stats ---
struct ZombieTypeStats {
  float speed;
  float health;
  uint8_t r, g, b;
};

// Type 0=Green(slow,tanky), 1=Yellow, 2=Orange, 3=Red(fast), 4=Purple(boss)
static const ZombieTypeStats ZOMBIE_STATS[5] = {
    {40.0f, 100.0f, 50, 200, 50},   // Green - slow, high health
    {60.0f, 75.0f, 220, 220, 50},   // Yellow - medium
    {80.0f, 50.0f, 255, 150, 50},   // Orange - faster, lower health
    {100.0f, 150.0f, 200, 50, 50},  // Red - fast, tanky
    {120.0f, 200.0f, 180, 50, 200}, // Purple - boss zombie
};

// --- Game-specific entity types ---
enum GameEntityTypes {
  ENTITY_TYPE_PLAYER = 0,
  ENTITY_TYPE_ZOMBIE,
  ENTITY_TYPE_BULLET,
  ENTITY_TYPE_DAMAGE_TEXT,
  ENTITY_TYPE_COUNT
};

// --- Player Container ---
class PlayerContainer : public RenderableEntityContainer {
public:
  float *health;
  float *facing_angles;

  PlayerContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity) {
    health = new float[capacity];
    facing_angles = new float[capacity];
    std::fill(health, health + capacity, 100.0f);
    std::fill(facing_angles, facing_angles + capacity, 0.0f);
  }

  ~PlayerContainer() override {
    delete[] health;
    delete[] facing_angles;
  }

  uint32_t createEntity(float x, float y, int texture_id) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = PLAYER_SIZE;
    heights[index] = PLAYER_SIZE;
    texture_ids[index] = texture_id;
    health[index] = 100.0f;
    facing_angles[index] = 0.0f;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 100;
    return index;
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

    float *newHealth = new float[newCapacity];
    float *newAngles = new float[newCapacity];

    if (count > 0) {
      std::copy(health, health + count, newHealth);
      std::copy(facing_angles, facing_angles + count, newAngles);
    }
    std::fill(newHealth + count, newHealth + newCapacity, 100.0f);
    std::fill(newAngles + count, newAngles + newCapacity, 0.0f);

    delete[] health;
    delete[] facing_angles;

    health = newHealth;
    facing_angles = newAngles;

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Zombie Container (OPTIMIZED) ---
class ZombieContainer : public RenderableEntityContainer {
public:
  float *speeds;
  float *health;
  float *max_health;
  uint8_t *zombie_types;
  Engine *engine;

  // SINGLE global target instead of per-zombie arrays!
  float global_target_x = 0.0f;
  float global_target_y = 0.0f;

  ZombieContainer(Engine *engine, int typeId, uint8_t defaultLayer,
                  int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity),
        engine(engine) {
    speeds = new float[capacity];
    health = new float[capacity];
    max_health = new float[capacity];
    zombie_types = new uint8_t[capacity];

    std::fill(speeds, speeds + capacity, 50.0f);
    std::fill(health, health + capacity, 100.0f);
    std::fill(max_health, max_health + capacity, 100.0f);
    std::fill(zombie_types, zombie_types + capacity, 0);
  }

  ~ZombieContainer() override {
    delete[] speeds;
    delete[] health;
    delete[] max_health;
    delete[] zombie_types;
  }

  uint32_t createEntity(float x, float y, int texture_id, uint8_t type) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = ZOMBIE_SIZE;
    heights[index] = ZOMBIE_SIZE;
    texture_ids[index] = texture_id;
    zombie_types[index] = type;
    speeds[index] = ZOMBIE_STATS[type].speed;
    health[index] = ZOMBIE_STATS[type].health;
    max_health[index] = ZOMBIE_STATS[type].health;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 50;
    return index;
  }

  // O(1) set target - just 2 assignments!
  void setTarget(float tx, float ty) {
    global_target_x = tx;
    global_target_y = ty;
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

    // Cache target locally for lambda capture
    const float tx = global_target_x;
    const float ty = global_target_y;

    // Parallel position update - greedy pathfinding toward target
    std::for_each(std::execution::par, indices.begin(), indices.begin() + count,
                  [&, tx, ty](uint32_t i) {
                    // Skip dead zombies (health <= 0)
                    if (health[i] <= 0)
                      return;

                    float &px = x_positions[i];
                    float &py = y_positions[i];
                    const float speed = speeds[i] * delta_time;

                    uint16_t oldCellX = cell_x[i];
                    uint16_t oldCellY = cell_y[i];

                    // Greedy pathfinding: move directly toward target
                    float dx = tx - px;
                    float dy = ty - py;
                    float dist = sqrtf(dx * dx + dy * dy);

                    if (dist > 1.0f) {
                      float nx = dx / dist;
                      float ny = dy / dist;
                      px += nx * speed;
                      py += ny * speed;
                    }

                    // Clamp to world bounds
                    if (px < 0)
                      px = 0;
                    else if (px > WORLD_WIDTH - ZOMBIE_SIZE)
                      px = WORLD_WIDTH - ZOMBIE_SIZE;

                    if (py < 0)
                      py = 0;
                    else if (py > WORLD_HEIGHT - ZOMBIE_SIZE)
                      py = WORLD_HEIGHT - ZOMBIE_SIZE;

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
    float *newHealth = new float[newCapacity];
    float *newMaxHealth = new float[newCapacity];
    uint8_t *newTypes = new uint8_t[newCapacity];

    if (count > 0) {
      std::copy(speeds, speeds + count, newSpeeds);
      std::copy(health, health + count, newHealth);
      std::copy(max_health, max_health + count, newMaxHealth);
      std::copy(zombie_types, zombie_types + count, newTypes);
    }
    std::fill(newSpeeds + count, newSpeeds + newCapacity, 50.0f);
    std::fill(newHealth + count, newHealth + newCapacity, 100.0f);
    std::fill(newMaxHealth + count, newMaxHealth + newCapacity, 100.0f);
    std::fill(newTypes + count, newTypes + newCapacity, 0);

    delete[] speeds;
    delete[] health;
    delete[] max_health;
    delete[] zombie_types;

    speeds = newSpeeds;
    health = newHealth;
    max_health = newMaxHealth;
    zombie_types = newTypes;

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Bullet Container with O(1) Free List ---
class BulletContainer : public RenderableEntityContainer {
public:
  float *vel_x;
  float *vel_y;
  uint8_t *active; // 1 = active, 0 = pooled/inactive
  Engine *engine;

  // FREE LIST for O(1) allocation!
  std::vector<int> free_list;

  // ACTIVE LIST for O(active_count) iteration instead of O(total)!
  std::vector<int> active_list;
  int next_scan_idx = 0; // Fallback scan position

  BulletContainer(Engine *engine, int typeId, uint8_t defaultLayer,
                  int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity),
        engine(engine) {
    vel_x = new float[capacity];
    vel_y = new float[capacity];
    active = new uint8_t[capacity];

    std::fill(vel_x, vel_x + capacity, 0.0f);
    std::fill(vel_y, vel_y + capacity, 0.0f);
    std::fill(active, active + capacity, 0);

    free_list.reserve(capacity);
    active_list.reserve(1024); // Reasonable max active bullets
  }

  ~BulletContainer() override {
    delete[] vel_x;
    delete[] vel_y;
    delete[] active;
  }

  uint32_t createEntity(float x, float y, float vx, float vy, int texture_id) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = BULLET_SIZE;
    heights[index] = BULLET_SIZE;
    texture_ids[index] = texture_id;
    vel_x[index] = vx;
    vel_y[index] = vy;
    active[index] = 1;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 80;
    return index;
  }

  // O(1) find inactive bullet using free list!
  int findInactive() {
    if (!free_list.empty()) {
      int idx = free_list.back();
      free_list.pop_back();
      return idx;
    }
    return -1; // No free bullets
  }

  // Activate pooled bullet
  void activateBullet(int index, float x, float y, float vx, float vy) {
    x_positions[index] = x;
    y_positions[index] = y;
    vel_x[index] = vx;
    vel_y[index] = vy;
    active[index] = 1;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);

    // Add to active list for fast iteration!
    active_list.push_back(index);

    // Update grid
    engine->grid.move(grid_node_indices[index], x, y);
    cell_x[index] = static_cast<uint16_t>(x * INV_GRID_CELL_SIZE);
    cell_y[index] = static_cast<uint16_t>(y * INV_GRID_CELL_SIZE);
  }

  // Deactivate bullet and add to free list
  void deactivateBullet(int index) {
    if (active[index] == 0)
      return; // Already inactive
    active[index] = 0;
    flags[index] &= ~static_cast<uint8_t>(EntityFlag::VISIBLE);
    // Move far off-screen
    x_positions[index] = -10000.0f;
    y_positions[index] = -10000.0f;
    // Add to free list for O(1) reuse!
    free_list.push_back(index);
  }

  void update(float delta_time) override {
    PROFILE_FUNCTION();
    if (active_list.empty())
      return;
    delta_time = std::min(delta_time, 0.1f);

    // Rebuild active list while updating (removes deactivated bullets)
    std::vector<int> new_active_list;
    new_active_list.reserve(active_list.size());

    // Only iterate through ACTIVE bullets!
    for (int idx : active_list) {
      int i = idx;
      if (active[i] == 0)
        continue; // Was deactivated, skip

      float &px = x_positions[i];
      float &py = y_positions[i];

      px += vel_x[i] * delta_time;
      py += vel_y[i] * delta_time;

      // Deactivate if out of bounds
      if (px < -100 || px > WORLD_WIDTH + 100 || py < -100 ||
          py > WORLD_HEIGHT + 100) {
        deactivateBullet(i);
        continue; // Don't add to new active list
      }

      // Update grid position
      uint16_t newCellX = static_cast<uint16_t>(px * INV_GRID_CELL_SIZE);
      uint16_t newCellY = static_cast<uint16_t>(py * INV_GRID_CELL_SIZE);
      if (cell_x[i] != newCellX || cell_y[i] != newCellY) {
        engine->grid.move(grid_node_indices[i], px, py);
        cell_x[i] = newCellX;
        cell_y[i] = newCellY;
      }

      flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
      new_active_list.push_back(i); // Still active
    }

    // Swap to new active list
    active_list = std::move(new_active_list);
  }

protected:
  void resizeArrays(int newCapacity) override {
    if (newCapacity <= capacity)
      return;

    float *newVelX = new float[newCapacity];
    float *newVelY = new float[newCapacity];
    uint8_t *newActive = new uint8_t[newCapacity];

    if (count > 0) {
      std::copy(vel_x, vel_x + count, newVelX);
      std::copy(vel_y, vel_y + count, newVelY);
      std::copy(active, active + count, newActive);
    }
    std::fill(newVelX + count, newVelX + newCapacity, 0.0f);
    std::fill(newVelY + count, newVelY + newCapacity, 0.0f);
    std::fill(newActive + count, newActive + newCapacity, 0);

    delete[] vel_x;
    delete[] vel_y;
    delete[] active;

    vel_x = newVelX;
    vel_y = newVelY;
    active = newActive;

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Damage Text Container with O(1) Free List ---
class DamageTextContainer : public RenderableEntityContainer {
public:
  float *lifetimes;
  uint8_t *active;

  // FREE LIST for O(1) allocation!
  std::vector<int> free_list;

  DamageTextContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity) {
    lifetimes = new float[capacity];
    active = new uint8_t[capacity];

    std::fill(lifetimes, lifetimes + capacity, 0.0f);
    std::fill(active, active + capacity, 0);

    free_list.reserve(capacity);
  }

  ~DamageTextContainer() override {
    delete[] lifetimes;
    delete[] active;
  }

  uint32_t createEntity(float x, float y, int texture_id) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = DAMAGE_TEXT_SIZE;
    heights[index] = DAMAGE_TEXT_SIZE;
    texture_ids[index] = texture_id;
    lifetimes[index] = DAMAGE_TEXT_LIFETIME;
    active[index] = 1;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 200; // Above everything
    return index;
  }

  // O(1) find inactive using free list!
  int findInactive() {
    if (!free_list.empty()) {
      int idx = free_list.back();
      free_list.pop_back();
      return idx;
    }
    return -1;
  }

  void activateText(int index, float x, float y) {
    x_positions[index] = x;
    y_positions[index] = y;
    lifetimes[index] = DAMAGE_TEXT_LIFETIME;
    active[index] = 1;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
  }

  void deactivateText(int index) {
    if (active[index] == 0)
      return; // Already inactive
    active[index] = 0;
    flags[index] &= ~static_cast<uint8_t>(EntityFlag::VISIBLE);
    x_positions[index] = -10000.0f;
    y_positions[index] = -10000.0f;
    // Add to free list for O(1) reuse!
    free_list.push_back(index);
  }

  void update(float delta_time) override {
    for (int i = 0; i < count; ++i) {
      if (active[i] == 0)
        continue;

      // Float upward
      y_positions[i] -= DAMAGE_TEXT_FLOAT_SPEED * delta_time;

      // Decrease lifetime
      lifetimes[i] -= delta_time;
      if (lifetimes[i] <= 0) {
        deactivateText(i);
        continue;
      }

      flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    }
  }

protected:
  void resizeArrays(int newCapacity) override {
    if (newCapacity <= capacity)
      return;

    float *newLifetimes = new float[newCapacity];
    uint8_t *newActive = new uint8_t[newCapacity];

    if (count > 0) {
      std::copy(lifetimes, lifetimes + count, newLifetimes);
      std::copy(active, active + count, newActive);
    }
    std::fill(newLifetimes + count, newLifetimes + newCapacity, 0.0f);
    std::fill(newActive + count, newActive + newCapacity, 0);

    delete[] lifetimes;
    delete[] active;

    lifetimes = newLifetimes;
    active = newActive;

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Game State ---
struct GameState {
  uint32_t player_index;
  bool is_alive;
  float fire_timer;
  bool is_shooting;

  // Textures
  int player_texture_id;
  int zombie_texture_ids[5];
  int bullet_texture_id;
  int damage_text_texture_id;

  // Containers
  PlayerContainer *player_container;
  ZombieContainer *zombie_container;
  BulletContainer *bullet_container;
  DamageTextContainer *damage_text_container;

  // FPS tracking
  Uint64 last_fps_time;
  int frame_count;
  float current_fps;

  // Stats
  int zombies_killed;
  int total_shots;
};

// --- Function Declarations ---
void setup_game(Engine *engine, GameState *game_state);
void handle_input(Engine *engine, const bool *keyboard_state,
                  GameState *game_state, float delta_time);
void update_game(Engine *engine, GameState *game_state, float delta_time);
void check_collisions(Engine *engine, GameState *game_state);
SDL_Surface *create_colored_surface(int width, int height, Uint8 r, Uint8 g,
                                    Uint8 b);

// --- Main Function ---
int main(int argc, char *argv[]) {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
    return 1;
  }
  srand(static_cast<unsigned int>(time(nullptr)));

  SDL_Window *window = SDL_CreateWindow("Zombie Shooter - Engine Demo",
                                        WINDOW_WIDTH, WINDOW_HEIGHT, 0);
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
  PlayerContainer *player_container =
      new PlayerContainer(ENTITY_TYPE_PLAYER, 0, 10);
  ZombieContainer *zombie_container =
      new ZombieContainer(engine, ENTITY_TYPE_ZOMBIE, 0, NUM_ZOMBIES + 500);
  BulletContainer *bullet_container =
      new BulletContainer(engine, ENTITY_TYPE_BULLET, 0, MAX_BULLETS + 100);
  DamageTextContainer *damage_text_container = new DamageTextContainer(
      ENTITY_TYPE_DAMAGE_TEXT, 0, MAX_DAMAGE_TEXTS + 50);

  engine->entityManager.registerEntityType(player_container);
  engine->entityManager.registerEntityType(zombie_container);
  engine->entityManager.registerEntityType(bullet_container);
  engine->entityManager.registerEntityType(damage_text_container);

  // Initialize game state
  GameState game_state = {};
  game_state.player_container = player_container;
  game_state.zombie_container = zombie_container;
  game_state.bullet_container = bullet_container;
  game_state.damage_text_container = damage_text_container;
  game_state.player_index = INVALID_ID;
  game_state.is_alive = true;
  game_state.fire_timer = 0.0f;
  game_state.is_shooting = false;
  game_state.last_fps_time = SDL_GetTicks();
  game_state.zombies_killed = 0;
  game_state.total_shots = 0;

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

  // Set camera to player
  if (game_state.player_index != INVALID_ID) {
    float player_x = player_container->x_positions[game_state.player_index];
    float player_y = player_container->y_positions[game_state.player_index];
    engine->camera.x = player_x + PLAYER_SIZE / 2.0f;
    engine->camera.y = player_y + PLAYER_SIZE / 2.0f;
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
                << " | Zombies: " << zombie_container->count
                << " | Killed: " << game_state.zombies_killed
                << " | Shots: " << game_state.total_shots << std::endl;
    }

    // Process events
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_QUIT)
        quit = true;
      else if (event.type == SDL_EVENT_KEY_DOWN) {
        if (event.key.scancode == SDL_SCANCODE_ESCAPE)
          quit = true;
        else if (event.key.scancode == SDL_SCANCODE_R && !game_state.is_alive) {
          // Reset game
          game_state.is_alive = true;
          game_state.zombies_killed = 0;
          game_state.total_shots = 0;
          setup_game(engine, &game_state);
          engine->grid.rebuild_grid(engine);
        }
      } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
        if (event.button.button == SDL_BUTTON_LEFT) {
          game_state.is_shooting = true;
        }
      } else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
        if (event.button.button == SDL_BUTTON_LEFT) {
          game_state.is_shooting = false;
        }
      }
    }

    const bool *keyboard_state = SDL_GetKeyboardState(NULL);

    if (game_state.is_alive) {
      handle_input(engine, keyboard_state, &game_state, delta_time);
      update_game(engine, &game_state, delta_time);
      check_collisions(engine, &game_state);
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

    SDL_SetRenderDrawColor(engine->renderer, 30, 30, 40, 255);
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
  // Create textures
  // Player texture (blue)
  SDL_Surface *player_surface =
      create_colored_surface(PLAYER_SIZE, PLAYER_SIZE, 50, 100, 255);
  game_state->player_texture_id = engine_register_texture(
      engine, player_surface, 0, 0, PLAYER_SIZE, PLAYER_SIZE);
  SDL_DestroySurface(player_surface);

  // Zombie textures (5 types with different colors)
  for (int t = 0; t < 5; ++t) {
    SDL_Surface *zombie_surface =
        create_colored_surface(ZOMBIE_SIZE, ZOMBIE_SIZE, ZOMBIE_STATS[t].r,
                               ZOMBIE_STATS[t].g, ZOMBIE_STATS[t].b);
    game_state->zombie_texture_ids[t] = engine_register_texture(
        engine, zombie_surface, 0, PLAYER_SIZE + t * ZOMBIE_SIZE, ZOMBIE_SIZE,
        ZOMBIE_SIZE);
    SDL_DestroySurface(zombie_surface);
  }

  // Bullet texture (yellow)
  SDL_Surface *bullet_surface =
      create_colored_surface(BULLET_SIZE, BULLET_SIZE, 255, 255, 100);
  game_state->bullet_texture_id = engine_register_texture(
      engine, bullet_surface, 0, PLAYER_SIZE + 5 * ZOMBIE_SIZE, BULLET_SIZE,
      BULLET_SIZE);
  SDL_DestroySurface(bullet_surface);

  // Damage text texture (red - "25")
  SDL_Surface *damage_surface =
      create_colored_surface(DAMAGE_TEXT_SIZE, DAMAGE_TEXT_SIZE, 255, 80, 80);
  game_state->damage_text_texture_id = engine_register_texture(
      engine, damage_surface, 0, PLAYER_SIZE + 5 * ZOMBIE_SIZE + BULLET_SIZE,
      DAMAGE_TEXT_SIZE, DAMAGE_TEXT_SIZE);
  SDL_DestroySurface(damage_surface);

  // Create player at center
  float start_x = WORLD_WIDTH / 2.0f;
  float start_y = WORLD_HEIGHT / 2.0f;
  game_state->player_index = game_state->player_container->createEntity(
      start_x, start_y, game_state->player_texture_id);

  std::cout << "PLAYER CREATED: index=" << game_state->player_index << " at ("
            << start_x << ", " << start_y << ")" << std::endl;

  // Create zombies distributed across the world
  for (int i = 0; i < NUM_ZOMBIES; ++i) {
    // Spawn zombies away from player initially
    float x, y;
    do {
      x = static_cast<float>(rand() % (WORLD_WIDTH - ZOMBIE_SIZE));
      y = static_cast<float>(rand() % (WORLD_HEIGHT - ZOMBIE_SIZE));
    } while (fabsf(x - start_x) < 500 && fabsf(y - start_y) < 500);

    uint8_t type = rand() % 5;
    game_state->zombie_container->createEntity(
        x, y, game_state->zombie_texture_ids[type], type);
  }

  // Pre-allocate bullet pool and add to free list
  for (int i = 0; i < MAX_BULLETS; ++i) {
    uint32_t idx = game_state->bullet_container->createEntity(
        -10000.0f, -10000.0f, 0, 0, game_state->bullet_texture_id);
    game_state->bullet_container->active[idx] = 0;
    game_state->bullet_container->flags[idx] &=
        ~static_cast<uint8_t>(EntityFlag::VISIBLE);
    // Add to free list for O(1) allocation!
    game_state->bullet_container->free_list.push_back(idx);
  }

  // Pre-allocate damage text pool and add to free list
  for (int i = 0; i < MAX_DAMAGE_TEXTS; ++i) {
    uint32_t idx = game_state->damage_text_container->createEntity(
        -10000.0f, -10000.0f, game_state->damage_text_texture_id);
    game_state->damage_text_container->active[idx] = 0;
    game_state->damage_text_container->flags[idx] &=
        ~static_cast<uint8_t>(EntityFlag::VISIBLE);
    // Add to free list for O(1) allocation!
    game_state->damage_text_container->free_list.push_back(idx);
  }

  std::cout << "=== Zombie Shooter Setup Complete ===" << std::endl;
  std::cout << "  Zombies: " << game_state->zombie_container->count
            << std::endl;
  std::cout << "  Bullet pool: " << game_state->bullet_container->count
            << std::endl;
  std::cout << "  Damage text pool: "
            << game_state->damage_text_container->count << std::endl;
  std::cout << "Controls: WASD to move, Mouse to aim, Left-click to shoot"
            << std::endl;
  std::cout << "  +/- to zoom, ESC to quit, R to restart" << std::endl;
}

void handle_input(Engine *engine, const bool *keyboard_state,
                  GameState *game_state, float delta_time) {
  PROFILE_FUNCTION();
  if (game_state->player_index == INVALID_ID)
    return;

  PlayerContainer *pCont = game_state->player_container;
  uint32_t player_idx = game_state->player_index;

  float &px = pCont->x_positions[player_idx];
  float &py = pCont->y_positions[player_idx];

  float dx = 0, dy = 0;

  if (keyboard_state[SDL_SCANCODE_W] || keyboard_state[SDL_SCANCODE_UP])
    dy -= 1.0f;
  if (keyboard_state[SDL_SCANCODE_S] || keyboard_state[SDL_SCANCODE_DOWN])
    dy += 1.0f;
  if (keyboard_state[SDL_SCANCODE_A] || keyboard_state[SDL_SCANCODE_LEFT])
    dx -= 1.0f;
  if (keyboard_state[SDL_SCANCODE_D] || keyboard_state[SDL_SCANCODE_RIGHT])
    dx += 1.0f;

  // Normalize diagonal movement
  if (dx != 0 && dy != 0) {
    float len = sqrtf(dx * dx + dy * dy);
    dx /= len;
    dy /= len;
  }

  // Move player
  px += dx * PLAYER_SPEED * delta_time;
  py += dy * PLAYER_SPEED * delta_time;

  // Clamp to world bounds
  if (px < 0)
    px = 0;
  else if (px > WORLD_WIDTH - PLAYER_SIZE)
    px = WORLD_WIDTH - PLAYER_SIZE;

  if (py < 0)
    py = 0;
  else if (py > WORLD_HEIGHT - PLAYER_SIZE)
    py = WORLD_HEIGHT - PLAYER_SIZE;

  // Update grid position
  uint16_t newCellX = static_cast<uint16_t>(px * INV_GRID_CELL_SIZE);
  uint16_t newCellY = static_cast<uint16_t>(py * INV_GRID_CELL_SIZE);
  if (pCont->cell_x[player_idx] != newCellX ||
      pCont->cell_y[player_idx] != newCellY) {
    engine->grid.move(pCont->grid_node_indices[player_idx], px, py);
    pCont->cell_x[player_idx] = newCellX;
    pCont->cell_y[player_idx] = newCellY;
  }

  // Get mouse position and calculate aim direction
  float mouse_x, mouse_y;
  SDL_GetMouseState(&mouse_x, &mouse_y);

  // Convert screen coordinates to world coordinates
  float world_mouse_x = engine->camera.x - engine->camera.width / 2 + mouse_x;
  float world_mouse_y = engine->camera.y - engine->camera.height / 2 + mouse_y;

  float player_center_x = px + PLAYER_SIZE / 2.0f;
  float player_center_y = py + PLAYER_SIZE / 2.0f;

  float aim_dx = world_mouse_x - player_center_x;
  float aim_dy = world_mouse_y - player_center_y;
  float aim_angle = atan2f(aim_dy, aim_dx);
  pCont->facing_angles[player_idx] = aim_angle;

  // Shooting - SHOTGUN SPREAD!
  game_state->fire_timer -= delta_time;
  if (game_state->is_shooting && game_state->fire_timer <= 0) {
    game_state->fire_timer = FIRE_RATE;
    game_state->total_shots += BULLETS_PER_SHOT;

    // Normalize aim direction
    float aim_len = sqrtf(aim_dx * aim_dx + aim_dy * aim_dy);
    if (aim_len > 0) {
      float base_angle = aim_angle;
      float half_spread = SPREAD_ANGLE / 2.0f;
      float angle_step = SPREAD_ANGLE / (BULLETS_PER_SHOT - 1);

      // Fire multiple bullets in a spread pattern
      for (int i = 0; i < BULLETS_PER_SHOT; ++i) {
        float bullet_angle = base_angle - half_spread + (angle_step * i);
        float vx = cosf(bullet_angle) * BULLET_SPEED;
        float vy = sinf(bullet_angle) * BULLET_SPEED;

        // Find pooled bullet or create new
        int bullet_idx = game_state->bullet_container->findInactive();
        if (bullet_idx >= 0) {
          game_state->bullet_container->activateBullet(
              bullet_idx, player_center_x, player_center_y, vx, vy);
        }
      }
    }
  }

  // Update camera to follow player
  engine->camera.x = player_center_x;
  engine->camera.y = player_center_y;

  // Set zombie target to player position
  game_state->zombie_container->setTarget(player_center_x, player_center_y);
}

void update_game(Engine *engine, GameState *game_state, float delta_time) {
  // Entity updates are handled by engine_update through container update
  // methods
}

void check_collisions(Engine *engine, GameState *game_state) {
  PROFILE_FUNCTION();
  if (game_state->player_index == INVALID_ID)
    return;

  PlayerContainer *pCont = game_state->player_container;
  ZombieContainer *zCont = game_state->zombie_container;
  BulletContainer *bCont = game_state->bullet_container;

  uint32_t player_idx = game_state->player_index;
  float player_x = pCont->x_positions[player_idx];
  float player_y = pCont->y_positions[player_idx];
  float player_cx = player_x + PLAYER_SIZE / 2.0f;
  float player_cy = player_y + PLAYER_SIZE / 2.0f;

  // Pre-compute collision threshold
  const float collision_dist = (BULLET_SIZE + ZOMBIE_SIZE) / 2.0f;
  const float collision_dist_sq = collision_dist * collision_dist;

  // Query radius must be at least GRID_CELL_SIZE for spatial queries to work!
  const float bullet_query_range = static_cast<float>(GRID_CELL_SIZE);

  // OPTIMIZED: Only iterate ACTIVE bullets using active_list!
  for (int b : bCont->active_list) {
    if (bCont->active[b] == 0)
      continue; // Was deactivated during this frame

    float bx = bCont->x_positions[b];
    float by = bCont->y_positions[b];

    // Query tiny area around THIS bullet
    const auto &nearby_zombies =
        engine->grid.queryCircle(bx, by, bullet_query_range);

    for (const auto &entity : nearby_zombies) {
      if (entity.type != ENTITY_TYPE_ZOMBIE)
        continue;
      if (entity.index >= zCont->count)
        continue;
      if (zCont->health[entity.index] <= 0)
        continue;

      float zx = zCont->x_positions[entity.index];
      float zy = zCont->y_positions[entity.index];

      // Fast squared distance check
      float dx = bx - zx;
      float dy = by - zy;
      if (dx * dx + dy * dy < collision_dist_sq) {
        // Hit! Apply fixed damage
        zCont->health[entity.index] -= FIXED_DAMAGE;

        // Deactivate bullet
        bCont->deactivateBullet(b);

        // Check if zombie died
        if (zCont->health[entity.index] <= 0) {
          game_state->zombies_killed++;

          // Quick respawn
          float new_x =
              static_cast<float>(rand() % (WORLD_WIDTH - ZOMBIE_SIZE));
          float new_y =
              static_cast<float>(rand() % (WORLD_HEIGHT - ZOMBIE_SIZE));

          uint8_t new_type = rand() % 5;
          zCont->x_positions[entity.index] = new_x;
          zCont->y_positions[entity.index] = new_y;
          zCont->zombie_types[entity.index] = new_type;
          zCont->speeds[entity.index] = ZOMBIE_STATS[new_type].speed;
          zCont->health[entity.index] = ZOMBIE_STATS[new_type].health;
          zCont->max_health[entity.index] = ZOMBIE_STATS[new_type].health;
          zCont->texture_ids[entity.index] =
              game_state->zombie_texture_ids[new_type];

          // Update grid position
          engine->grid.move(zCont->grid_node_indices[entity.index], new_x,
                            new_y);
          zCont->cell_x[entity.index] =
              static_cast<uint16_t>(new_x * INV_GRID_CELL_SIZE);
          zCont->cell_y[entity.index] =
              static_cast<uint16_t>(new_y * INV_GRID_CELL_SIZE);
        }

        break; // Bullet hit - stop checking this bullet
      }
    }
  }

  // Check zombie-player collision
  float query_range = static_cast<float>(GRID_CELL_SIZE) * 2.0f;
  const auto &nearby =
      engine->grid.queryCircle(player_cx, player_cy, query_range);

  for (const auto &entity : nearby) {
    if (entity.type != ENTITY_TYPE_ZOMBIE)
      continue;
    if (entity.index >= zCont->count)
      continue;
    if (zCont->health[entity.index] <= 0)
      continue;

    float zx = zCont->x_positions[entity.index];
    float zy = zCont->y_positions[entity.index];

    if (fabsf(player_x - zx) < (PLAYER_SIZE + ZOMBIE_SIZE) * 0.4f &&
        fabsf(player_y - zy) < (PLAYER_SIZE + ZOMBIE_SIZE) * 0.4f) {
      game_state->is_alive = false;
      std::cout << "GAME OVER! Killed by zombie!" << std::endl;
      std::cout << "Zombies killed: " << game_state->zombies_killed
                << std::endl;
      std::cout << "Press R to restart" << std::endl;
      break;
    }
  }
}
