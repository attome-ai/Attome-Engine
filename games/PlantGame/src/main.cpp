#include "ATMEngine.h"
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"
#include "stb_image.h"
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
#include <string>
#include <vector>

// --- Constants ---
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1020
#define PLAYER_SIZE 64 // Slightly larger for ship sprite
#define PLANET_SIZE 32 // Slightly larger for ship sprite
#define PLAYER_SPEED 800.0f
#define NUM_PLANETS 100000
#define BULLET_SIZE 26
#define BULLET_SPEED 600.0f
#define MAX_BULLETS 200000
// #define FIRE_RATE 0.05f       // 20 shots per second
// #define BULLETS_PER_SHOT 300  // Shoot 3 bullets at once
// #define BULLET_SPREAD 0.25f   // Spread angle in radians
// #define BULLET_DAMAGE 1000.0f // Damage per bullet
//  Shooting Constants
#define FIRE_RATE 0.05f
#define BULLETS_PER_SHOT 25
#define BULLET_SPREAD 0.05f
#define BULLET_LIFETIME 11600.0f
#define BULLET_DAMAGE 1000.0f
// --- Planet Type Stats ---
struct PlanetTypeStats {
  float speed;
  float health;
  int texture_idx; // Index into planet_texture_ids
};

// Types corresponding to ship2.png through ship6.png
static const PlanetTypeStats PLANET_STATS[5] = {
    {200.0f, 200.0f, 0}, // Type 0
    {200.0f, 175.0f, 1}, // Type 1
    {200.0f, 150.0f, 2}, // Type 2
    {200.0f, 250.0f, 3}, // Type 3
    {200.0f, 300.0f, 4}, // Type 4
};

// --- Game-specific entity types ---
enum GameEntityTypes {
  ENTITY_TYPE_PLAYER = 0,
  ENTITY_TYPE_PLANET,
  ENTITY_TYPE_BULLET,
  ENTITY_TYPE_COUNT
};

// --- Player Container ---
class PlayerContainer : public RenderableEntityContainer {
public:
  PlayerContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity) {}

  ~PlayerContainer() override {}

  uint32_t createEntity(float x, float y, int texture_id) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = PLAYER_SIZE;
    heights[index] = PLAYER_SIZE;
    texture_ids[index] = texture_id;
    rotations[index] = 0.0f;
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

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Bullet Container ---
class BulletContainer : public RenderableEntityContainer {
public:
  float *velocities_x;
  float *velocities_y;
  float *lifetimes;
  uint8_t *active;
  Engine *engine;

  // FREE LIST for O(1) allocation
  std::vector<int> free_list;

  BulletContainer(Engine *engine, int typeId, uint8_t defaultLayer,
                  int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity),
        engine(engine) {
    velocities_x = new float[capacity];
    velocities_y = new float[capacity];
    lifetimes = new float[capacity];
    active = new uint8_t[capacity];

    std::fill(velocities_x, velocities_x + capacity, 0.0f);
    std::fill(velocities_y, velocities_y + capacity, 0.0f);
    std::fill(lifetimes, lifetimes + capacity, 0.0f);
    std::fill(active, active + capacity, 0);

    free_list.reserve(capacity);
  }

  ~BulletContainer() override {
    delete[] velocities_x;
    delete[] velocities_y;
    delete[] lifetimes;
    delete[] active;
  }

  uint32_t createEntity(float x, float y, float vx, float vy, int texture_id) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    velocities_x[index] = vx;
    velocities_y[index] = vy;
    widths[index] = BULLET_SIZE;
    heights[index] = BULLET_SIZE;
    texture_ids[index] = texture_id;
    lifetimes[index] = BULLET_LIFETIME;
    active[index] = 1;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    z_indices[index] = 75; // Between planets and player

    // CRITICAL: Add to spatial grid for rendering!
    EntityRef ref;
    ref.type = type_id;
    ref.index = index;
    grid_node_indices[index] = engine->grid.add(ref, x, y);
    cell_x[index] = static_cast<uint16_t>(x * INV_GRID_CELL_SIZE);
    cell_y[index] = static_cast<uint16_t>(y * INV_GRID_CELL_SIZE);

    return index;
  }

  // O(1) find inactive using free list
  int findInactive() {
    if (!free_list.empty()) {
      int idx = free_list.back();
      free_list.pop_back();
      return idx;
    }
    return -1;
  }

  void activateBullet(int index, float x, float y, float vx, float vy) {
    x_positions[index] = x;
    y_positions[index] = y;
    velocities_x[index] = vx;
    velocities_y[index] = vy;
    lifetimes[index] = BULLET_LIFETIME;
    active[index] = 1;
    flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);

    // Update grid
    int32_t nodeIdx = grid_node_indices[index];
    if (nodeIdx != -1) {
      engine->grid.move(nodeIdx, x, y);
    }
    cell_x[index] = static_cast<uint16_t>(x * INV_GRID_CELL_SIZE);
    cell_y[index] = static_cast<uint16_t>(y * INV_GRID_CELL_SIZE);
  }

  void deactivateBullet(int index) {
    if (active[index] == 0)
      return; // Already inactive
    active[index] = 0;
    flags[index] &= ~static_cast<uint8_t>(EntityFlag::VISIBLE);
    x_positions[index] = -10000.0f;
    y_positions[index] = -10000.0f;

    // Update grid to move off-screen
    int32_t nodeIdx = grid_node_indices[index];
    if (nodeIdx != -1) {
      engine->grid.move(nodeIdx, -10000.0f, -10000.0f);
    }

    // Add to free list for O(1) reuse
    free_list.push_back(index);
  }

  void update(float delta_time) override {
    PROFILE_FUNCTION();
    delta_time = std::min(delta_time, 0.1f);

    for (int i = 0; i < count; ++i) {
      if (active[i] == 0)
        continue;

      // Move bullet
      x_positions[i] += velocities_x[i] * delta_time;
      y_positions[i] += velocities_y[i] * delta_time;

      // Update grid
      int32_t nodeIdx = grid_node_indices[i];
      if (nodeIdx != -1) {
        engine->grid.move(nodeIdx, x_positions[i], y_positions[i]);
      }
      cell_x[i] = static_cast<uint16_t>(x_positions[i] * INV_GRID_CELL_SIZE);
      cell_y[i] = static_cast<uint16_t>(y_positions[i] * INV_GRID_CELL_SIZE);

      // Decrease lifetime
      lifetimes[i] -= delta_time;
      if (lifetimes[i] <= 0 || x_positions[i] < 0 ||
          x_positions[i] > WORLD_WIDTH || y_positions[i] < 0 ||
          y_positions[i] > WORLD_HEIGHT) {
        deactivateBullet(i);
        continue;
      }

      flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    }
  }

protected:
  void resizeArrays(int newCapacity) override {
    if (newCapacity <= capacity)
      return;

    float *newVelX = new float[newCapacity];
    float *newVelY = new float[newCapacity];
    float *newLifetimes = new float[newCapacity];
    uint8_t *newActive = new uint8_t[newCapacity];

    if (count > 0) {
      std::copy(velocities_x, velocities_x + count, newVelX);
      std::copy(velocities_y, velocities_y + count, newVelY);
      std::copy(lifetimes, lifetimes + count, newLifetimes);
      std::copy(active, active + count, newActive);
    }
    std::fill(newVelX + count, newVelX + newCapacity, 0.0f);
    std::fill(newVelY + count, newVelY + newCapacity, 0.0f);
    std::fill(newLifetimes + count, newLifetimes + newCapacity, 0.0f);
    std::fill(newActive + count, newActive + newCapacity, 0);

    delete[] velocities_x;
    delete[] velocities_y;
    delete[] lifetimes;
    delete[] active;

    velocities_x = newVelX;
    velocities_y = newVelY;
    lifetimes = newLifetimes;
    active = newActive;

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Planet Container (OPTIMIZED) ---
class PlanetContainer : public RenderableEntityContainer {
public:
  float *speeds;
  float *health;
  float *max_health;
  uint8_t *planet_types;
  Engine *engine;

  // SINGLE global target instead of per-planet arrays!
  float global_target_x = 0.0f;
  float global_target_y = 0.0f;

  PlanetContainer(Engine *engine, int typeId, uint8_t defaultLayer,
                  int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity),
        engine(engine) {
    speeds = new float[capacity];
    health = new float[capacity];
    max_health = new float[capacity];

    planet_types = new uint8_t[capacity];

    std::fill(speeds, speeds + capacity, 50.0f);
    std::fill(health, health + capacity, 100.0f);
    std::fill(max_health, max_health + capacity, 100.0f);
    std::fill(planet_types, planet_types + capacity, 0);
  }

  ~PlanetContainer() override {
    delete[] speeds;
    delete[] health;
    delete[] max_health;

    delete[] planet_types;
  }

  uint32_t createEntity(float x, float y, int texture_id, uint8_t type) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = PLANET_SIZE;
    heights[index] = PLANET_SIZE;
    texture_ids[index] = texture_id;
    planet_types[index] = type;
    speeds[index] = PLANET_STATS[type].speed;
    health[index] = PLANET_STATS[type].health;
    max_health[index] = PLANET_STATS[type].health;
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
                    // Skip destroyed planets (health <= 0)
                    if (health[i] <= 0) {

                      return;
                    }

                    float &px = x_positions[i];
                    float &py = y_positions[i];
                    const float speed = speeds[i] * delta_time;

                    uint16_t oldCellX = cell_x[i];
                    uint16_t oldCellY = cell_y[i];

                    // Greedy pathfinding: move directly toward target
                    float dx = tx - px;
                    float dy = ty - py;
                    float dist = sqrtf(dx * dx + dy * dy);

                    // Calculate rotation angle toward target
                    rotations[i] = atan2f(dy, dx);

                    if (dist > 1.0f) {
                      float nx = dx / dist;
                      float ny = dy / dist;
                      px += nx * speed;
                      py += ny * speed;
                    }

                    // Clamp to world bounds
                    if (px < 0)
                      px = 0;
                    else if (px > WORLD_WIDTH - PLANET_SIZE)
                      px = WORLD_WIDTH - PLANET_SIZE;

                    if (py < 0)
                      py = 0;
                    else if (py > WORLD_HEIGHT - PLANET_SIZE)
                      py = WORLD_HEIGHT - PLANET_SIZE;

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
      std::copy(planet_types, planet_types + count, newTypes);
    }
    std::fill(newSpeeds + count, newSpeeds + newCapacity, 50.0f);
    std::fill(newHealth + count, newHealth + newCapacity, 100.0f);
    std::fill(newMaxHealth + count, newMaxHealth + newCapacity, 100.0f);
    std::fill(newTypes + count, newTypes + newCapacity, 0);

    delete[] speeds;
    delete[] health;
    delete[] max_health;
    delete[] planet_types;

    speeds = newSpeeds;
    health = newHealth;
    max_health = newMaxHealth;
    planet_types = newTypes;

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Game State ---
struct GameState {
  uint32_t player_index;
  // Player is invincible, always "alive"

  // Textures
  int player_texture_id;
  int planet_texture_ids[5];
  int bullet_texture_id;

  // Containers
  PlayerContainer *player_container;
  PlanetContainer *planet_container;
  BulletContainer *bullet_container;

  // Shooting
  float shoot_cooldown;
  bool mouse_pressed;

  // FPS tracking
  Uint64 last_fps_time;
  int frame_count;
  float current_fps;

  // Stats
  int hit_count;    // Number of times player got hit
  int killed_count; // Number of planets killed by bullets
};

// --- Function Declarations ---
void setup_game(Engine *engine, GameState *game_state);
void handle_input(Engine *engine, const bool *keyboard_state,
                  GameState *game_state, float delta_time);
void update_game(Engine *engine, GameState *game_state, float delta_time);
void check_collisions(Engine *engine, GameState *game_state);
SDL_Surface *load_image_to_surface(const char *filepath);

// --- Main Function ---
int main(int argc, char *argv[]) {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
    return 1;
  }
  srand(static_cast<unsigned int>(time(nullptr)));

  SDL_Window *window = SDL_CreateWindow("GAME ENGN - Invincible Ship",
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

  // Init ImGui Manually
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.FontGlobalScale = 2.0f;

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  // Setup Platform/Renderer backends
  if (engine->renderer) {
    ImGui_ImplSDL3_InitForSDLRenderer(engine->window, engine->renderer);
    ImGui_ImplSDLRenderer3_Init(engine->renderer);
  }

  // Register entity containers
  PlayerContainer *player_container =
      new PlayerContainer(ENTITY_TYPE_PLAYER, 0, 10);
  PlanetContainer *planet_container =
      new PlanetContainer(engine, ENTITY_TYPE_PLANET, 0, NUM_PLANETS + 500);
  BulletContainer *bullet_container =
      new BulletContainer(engine, ENTITY_TYPE_BULLET, 0, MAX_BULLETS);

  engine->entityManager.registerEntityType(player_container);
  engine->entityManager.registerEntityType(planet_container);
  engine->entityManager.registerEntityType(bullet_container);

  // Initialize game state (Local struct)
  GameState game_state = {};
  game_state.player_container = player_container;
  game_state.planet_container = planet_container;
  game_state.bullet_container = bullet_container;
  game_state.player_index = INVALID_ID;
  game_state.last_fps_time = SDL_GetTicks();
  game_state.hit_count = 0;
  game_state.killed_count = 0;
  game_state.shoot_cooldown = 0.0f;
  game_state.mouse_pressed = false;

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
    }

    // Process events
    while (SDL_PollEvent(&event)) {
      ImGui_ImplSDL3_ProcessEvent(&event);
      if (event.type == SDL_EVENT_QUIT)
        quit = true;
      else if (event.type == SDL_EVENT_KEY_DOWN) {
        if (event.key.scancode == SDL_SCANCODE_ESCAPE)
          quit = true;
      } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
        // Only process mouse input if ImGui doesn't want it
        ImGuiIO &io = ImGui::GetIO();
        if (event.button.button == SDL_BUTTON_LEFT && !io.WantCaptureMouse) {
          game_state.mouse_pressed = true;
        }
      } else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
        if (event.button.button == SDL_BUTTON_LEFT) {
          game_state.mouse_pressed = false;
        }
      }
    }

    const bool *keyboard_state = SDL_GetKeyboardState(NULL);

    handle_input(engine, keyboard_state, &game_state, delta_time);
    update_game(engine, &game_state, delta_time);
    check_collisions(engine, &game_state);

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

    // Render Setup
    if (engine->renderer) {
      // Start ImGui frame
      ImGui_ImplSDLRenderer3_NewFrame();
      ImGui_ImplSDL3_NewFrame();
      ImGui::NewFrame();

      // Render Stats Overlay
      ImGui::SetNextWindowPos(ImVec2(10, 10));
      ImGui::SetNextWindowSize(ImVec2(0, 0)); // Auto resize
      ImGui::Begin("Stats", NULL,
                   ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                       ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                       ImGuiWindowFlags_AlwaysAutoResize);
      ImGui::TextColored(ImVec4(1, 1, 0, 1), "FPS: %.1f",
                         game_state.current_fps);
      ImGui::TextColored(ImVec4(1, 0, 0, 1), "HITS: %d", game_state.hit_count);
      ImGui::TextColored(ImVec4(0, 1, 0, 1), "KILLS: %d",
                         game_state.killed_count);

      // Count active bullets
      int active_bullets = 0;
      for (int i = 0; i < game_state.bullet_container->count; ++i) {
        if (game_state.bullet_container->active[i]) {
          active_bullets++;
        }
      }
      ImGui::TextColored(ImVec4(1, 1, 0, 1), "Bullets: %d", active_bullets);

      ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1), "Planets: %d",
                         game_state.planet_container->count -
                             game_state.hit_count - game_state.killed_count);
      ImGui::End();

      ImGui::Render();

      SDL_SetRenderDrawColor(engine->renderer, 30, 30, 40, 255);
      SDL_RenderClear(engine->renderer);
      engine_render_scene(engine);

      // Render ImGui Over everything
      ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(),
                                            engine->renderer);

      engine_present(engine);
    }
  }
}

// Cleanup ImGui
ImGui_ImplSDLRenderer3_Shutdown();
ImGui_ImplSDL3_Shutdown();
ImGui::DestroyContext();

engine_destroy(engine);
return 0;
}

// Function to load image
SDL_Surface *load_image_to_surface(const char *filepath) {
  int w, h, c;
  unsigned char *data =
      stbi_load(filepath, &w, &h, &c, 4); // Force 4 channels (RGBA)
  if (!data) {
    std::cerr << "Failed to load image: " << filepath << std::endl;
    // Return a fallback surface (magenta square)
    SDL_Surface *surface = SDL_CreateSurface(32, 32, SDL_PIXELFORMAT_RGBA8888);
    SDL_FillSurfaceRect(surface, NULL,
                        SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format),
                                    NULL, 255, 0, 255, 255));
    return surface;
  }

  // Create SDL surface from data

  int pitch = w * 4;
  SDL_Surface *temp =
      SDL_CreateSurfaceFrom(w, h, SDL_PIXELFORMAT_RGBA32, data, pitch);
  if (temp) {
    // Create a copy so we can free stbi data
    SDL_Surface *copy = SDL_DuplicateSurface(temp);
    SDL_DestroySurface(temp);
    stbi_image_free(data);
    return copy;
  }
  stbi_image_free(data);
  return nullptr;
}

// ... (existing defines)

// ...

void setup_game(Engine *engine, GameState *game_state) {
  // 1. Load Textures with Atlas Packing
  char path_buffer[256];
  int atlas_x = 0;
  int atlas_y = 0; // Simple horizontal packing
  int padding = 2; // Pixel padding to avoid bleeding

  std::cout << "--- TEXTURE REGISTER DEBUG ---" << std::endl;

  // Player Ship (ship1.png)
  snprintf(path_buffer, sizeof(path_buffer), "resource/ship1.png");
  SDL_Surface *player_surf = load_image_to_surface(path_buffer);
  if (!player_surf)
    exit(1);
  game_state->player_texture_id =
      engine_register_texture(engine, player_surf, atlas_x, atlas_y, 0, 0);
  std::cout << "Player Tex ID: " << game_state->player_texture_id
            << " at X: " << atlas_x << std::endl;
  atlas_x += player_surf->w + padding;
  SDL_DestroySurface(player_surf);

  // Planet Ships (ship2.png - ship6.png)
  for (int i = 0; i < 5; ++i) {
    snprintf(path_buffer, sizeof(path_buffer), "resource/ship%d.png", i + 2);
    SDL_Surface *z_surf = load_image_to_surface(path_buffer);
    if (!z_surf)
      exit(1);
    game_state->planet_texture_ids[i] =
        engine_register_texture(engine, z_surf, atlas_x, atlas_y, 0, 0);
    std::cout << "Planet " << i
              << " Tex ID: " << game_state->planet_texture_ids[i]
              << " at X: " << atlas_x << std::endl;
    atlas_x += z_surf->w + padding;
    SDL_DestroySurface(z_surf);
  }

  // Bullet texture
  snprintf(path_buffer, sizeof(path_buffer), "resource/shoot1.png");
  SDL_Surface *bullet_surf = load_image_to_surface(path_buffer);
  if (!bullet_surf) {
    std::cerr << "Failed to load bullet texture, creating fallback."
              << std::endl;
    // Create a simple yellow circle for bullet
    bullet_surf =
        SDL_CreateSurface(BULLET_SIZE, BULLET_SIZE, SDL_PIXELFORMAT_RGBA8888);
    SDL_FillSurfaceRect(
        bullet_surf, NULL,
        SDL_MapRGBA(SDL_GetPixelFormatDetails(bullet_surf->format), NULL, 255,
                    255, 0, 255));
  }
  game_state->bullet_texture_id =
      engine_register_texture(engine, bullet_surf, atlas_x, atlas_y, 0, 0);
  std::cout << "Bullet Tex ID: " << game_state->bullet_texture_id
            << " at X: " << atlas_x << std::endl;
  atlas_x += bullet_surf->w + padding;
  SDL_DestroySurface(bullet_surf);

  std::cout << "------------------------------" << std::endl;

  // 2. Create Player
  game_state->player_index = game_state->player_container->createEntity(
      WORLD_WIDTH / 2.0f, WORLD_HEIGHT / 2.0f, game_state->player_texture_id);

  // 3. Create Planets
  for (int i = 0; i < NUM_PLANETS; ++i) {
    float x = static_cast<float>(rand() % WORLD_WIDTH);
    float y = static_cast<float>(rand() % WORLD_HEIGHT);

    // Ensure not spawning on top of player
    float dx = x - (WORLD_WIDTH / 2.0f);
    float dy = y - (WORLD_HEIGHT / 2.0f);
    if (dx * dx + dy * dy < 500 * 500) { // Keep clear area
      x += 1000.0f;
    }

    uint8_t type = rand() % 5;
    game_state->planet_container->createEntity(
        x, y, game_state->planet_texture_ids[type], type);
  }
}

void handle_input(Engine *engine, const bool *keyboard_state,
                  GameState *game_state, float delta_time) {
  if (game_state->player_index == INVALID_ID)
    return;

  float dx = 0.0f;
  float dy = 0.0f;

  if (keyboard_state[SDL_SCANCODE_W])
    dy -= 1.0f;
  if (keyboard_state[SDL_SCANCODE_S])
    dy += 1.0f;
  if (keyboard_state[SDL_SCANCODE_A])
    dx -= 1.0f;
  if (keyboard_state[SDL_SCANCODE_D])
    dx += 1.0f;

  if (dx != 0.0f || dy != 0.0f) {
    // Normalize
    float length = sqrtf(dx * dx + dy * dy);
    dx /= length;
    dy /= length;

    float move_speed = PLAYER_SPEED * delta_time;

    // Sprint
    if (keyboard_state[SDL_SCANCODE_LSHIFT])
      move_speed *= 2.0f;

    PlayerContainer *player = game_state->player_container;
    float new_x =
        player->x_positions[game_state->player_index] + dx * move_speed;
    float new_y =
        player->y_positions[game_state->player_index] + dy * move_speed;

    // Clamp to world
    if (new_x < 0)
      new_x = 0;
    if (new_x > WORLD_WIDTH - PLAYER_SIZE)
      new_x = WORLD_WIDTH - PLAYER_SIZE;
    if (new_y < 0)
      new_y = 0;
    if (new_y > WORLD_HEIGHT - PLAYER_SIZE)
      new_y = WORLD_HEIGHT - PLAYER_SIZE;

    player->x_positions[game_state->player_index] = new_x;
    player->y_positions[game_state->player_index] = new_y;

    // Update camera
    engine->camera.x = new_x + PLAYER_SIZE / 2.0f;
    engine->camera.y = new_y + PLAYER_SIZE / 2.0f;

    // Update grid
    int32_t nodeIdx = player->grid_node_indices[game_state->player_index];
    engine->grid.move(nodeIdx, new_x, new_y);
    player->cell_x[game_state->player_index] =
        static_cast<uint16_t>(new_x * INV_GRID_CELL_SIZE);
    player->cell_y[game_state->player_index] =
        static_cast<uint16_t>(new_y * INV_GRID_CELL_SIZE);
  }

  // Shooting mechanics
  if (game_state->shoot_cooldown > 0) {
    game_state->shoot_cooldown -= delta_time;
  }

  if (keyboard_state[SDL_SCANCODE_SPACE] || game_state->mouse_pressed) {
    if (game_state->shoot_cooldown <= 0) {
      // Shoot!
      PlayerContainer *player = game_state->player_container;
      float px =
          player->x_positions[game_state->player_index] + PLAYER_SIZE / 2.0f;
      float py =
          player->y_positions[game_state->player_index] + PLAYER_SIZE / 2.0f;

      // Get mouse position in world coordinates
      float mouse_x, mouse_y;
      SDL_GetMouseState(&mouse_x, &mouse_y);

      // Convert screen to world coordinates
      float world_mouse_x =
          engine->camera.x - (engine->camera.width / 2.0f) + mouse_x;
      float world_mouse_y =
          engine->camera.y - (engine->camera.height / 2.0f) + mouse_y;

      // Calculate direction from player to mouse
      float shoot_vx = world_mouse_x - px;
      float shoot_vy = world_mouse_y - py;
      float dist = sqrtf(shoot_vx * shoot_vx + shoot_vy * shoot_vy);

      // Normalize direction (if mouse is not exactly on player)
      if (dist > 1.0f) {
        shoot_vx /= dist;
        shoot_vy /= dist;
      } else {
        // Default to right if mouse is on player
        shoot_vx = 1.0f;
        shoot_vy = 0.0f;
      }

      // Shoot multiple bullets with spread
      BulletContainer *bullets = game_state->bullet_container;

      for (int i = 0; i < BULLETS_PER_SHOT; ++i) {
        // Calculate spread angle
        float spread_angle =
            (i - (BULLETS_PER_SHOT - 1) / 2.0f) * BULLET_SPREAD;
        float cos_spread = cosf(spread_angle);
        float sin_spread = sinf(spread_angle);

        // Rotate direction by spread angle
        float spread_vx = shoot_vx * cos_spread - shoot_vy * sin_spread;
        float spread_vy = shoot_vx * sin_spread + shoot_vy * cos_spread;

        int bullet_idx = bullets->findInactive();
        if (bullet_idx != -1) {
          bullets->activateBullet(
              bullet_idx, px - BULLET_SIZE / 2.0f, py - BULLET_SIZE / 2.0f,
              spread_vx * BULLET_SPEED, spread_vy * BULLET_SPEED);
        } else if (bullets->count < MAX_BULLETS) {
          bullets->createEntity(
              px - BULLET_SIZE / 2.0f, py - BULLET_SIZE / 2.0f,
              spread_vx * BULLET_SPEED, spread_vy * BULLET_SPEED,
              game_state->bullet_texture_id);
        }

        // Update rotation for the just activated/created bullet
        int last_idx = (bullet_idx != -1) ? bullet_idx : (bullets->count - 1);
        bullets->rotations[last_idx] =
            atan2f(spread_vy, spread_vx) + 3.14 / 2.0f;
      }

      game_state->shoot_cooldown = FIRE_RATE;
    }
  }
}

void update_game(Engine *engine, GameState *game_state, float delta_time) {
  // Update Planets Target (Player)
  if (game_state->player_index != INVALID_ID) {
    float px =
        game_state->player_container->x_positions[game_state->player_index];
    float py =
        game_state->player_container->y_positions[game_state->player_index];
    game_state->planet_container->setTarget(px, py);
  }
}

void check_collisions(Engine *engine, GameState *game_state) {
  if (game_state->player_index == INVALID_ID)
    return;

  PlayerContainer *player = game_state->player_container;
  PlanetContainer *planets = game_state->planet_container;

  float px = player->x_positions[game_state->player_index];
  float py = player->y_positions[game_state->player_index];
  float p_radius = PLAYER_SIZE / 2.2f; // Increased collision radius
  float p_center_x = px + PLAYER_SIZE / 2.0f;
  float p_center_y = py + PLAYER_SIZE / 2.0f;

  // Query planets near player
  // Broad phase: Circle query around player
  const std::vector<EntityRef> &nearby =
      engine->grid.queryCircle(p_center_x, p_center_y, p_radius + PLANET_SIZE);

  for (const auto &ref : nearby) {
    if (ref.type == ENTITY_TYPE_PLANET) {
      // Precise check
      float zx = planets->x_positions[ref.index];
      float zy = planets->y_positions[ref.index];
      float z_radius = PLANET_SIZE / 2.2f; // Increased collision radius
      float z_center_x = zx + PLANET_SIZE / 2.0f;
      float z_center_y = zy + PLANET_SIZE / 2.0f;

      float dx = p_center_x - z_center_x;
      float dy = p_center_y - z_center_y;
      float distSq = dx * dx + dy * dy;
      float combinedRadius = p_radius + z_radius;

      if (distSq < combinedRadius * combinedRadius) {
        // HIT!
        // 1. Increment Counter
        game_state->hit_count++;

        // 2. Destroy Planet
        planets->health[ref.index] = -1.0f; // Mark as destroyed

        // Move off-screen
        engine->grid.move(planets->grid_node_indices[ref.index], -10000.0f,
                          -10000.0f);
      }
    }
  }

  // Bullet-Planet Collision Detection
  BulletContainer *bullets = game_state->bullet_container;
  for (int b = 0; b < bullets->count; ++b) {
    if (bullets->active[b] == 0)
      continue; // Skip inactive bullets

    float bx = bullets->x_positions[b];
    float by = bullets->y_positions[b];
    float b_radius = BULLET_SIZE / 2.0f;

    // Query planets near bullet
    const std::vector<EntityRef> &nearby_planets =
        engine->grid.queryCircle(bx, by, b_radius);

    bool bullet_hit = false;

    for (const auto &ref : nearby_planets) {
      if (ref.type == ENTITY_TYPE_PLANET && !bullet_hit) {
        // Check if planet is active
        if (planets->health[ref.index] <= 0)
          continue;

        float zx = planets->x_positions[ref.index];
        float zy = planets->y_positions[ref.index];
        float z_radius = PLANET_SIZE / 2.2f;
        float z_center_x = zx + PLANET_SIZE / 2.0f;
        float z_center_y = zy + PLANET_SIZE / 2.0f;

        float dx = bx - z_center_x;
        float dy = by - z_center_y;
        float distSq = dx * dx + dy * dy;
        float combinedRadius = b_radius + z_radius;

        if (distSq < combinedRadius * combinedRadius) {
          planets->health[ref.index] -= BULLET_DAMAGE;

          if (planets->health[ref.index] <= 0) {
            // Destroyed!
            game_state->killed_count++;
          }
          // Move planet off-screen
          engine->grid.move(planets->grid_node_indices[ref.index], -10000,
                            -10000);

          // Deactivate bullet
          bullets->deactivateBullet(b);
          bullet_hit = true;
          break; // Bullet can only hit one planet
        }
      }
    }
  }
}
