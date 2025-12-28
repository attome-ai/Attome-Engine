#include "../game/ATMEngine.h"
#include "../game/stb_image.h"
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"
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
#define ZOMBIE_SIZE 32 // Slightly larger for ship sprite
#define DAMAGE_TEXT_SIZE 20
#define PLAYER_SPEED 600.0f
#define NUM_ZOMBIES 1000000
#define MAX_DAMAGE_TEXTS 500
#define DAMAGE_TEXT_LIFETIME 0.4f
#define DAMAGE_TEXT_FLOAT_SPEED 100.0f

// --- Zombie Type Stats ---
struct ZombieTypeStats {
  float speed;
  float health;
  int texture_idx; // Index into zombie_texture_ids
};

// Types corresponding to ship2.png through ship6.png
static const ZombieTypeStats ZOMBIE_STATS[5] = {
    {200.0f, 200.0f, 0}, // Type 0
    {200.0f, 175.0f, 1}, // Type 1
    {200.0f, 150.0f, 2}, // Type 2
    {200.0f, 250.0f, 3}, // Type 3
    {200.0f, 300.0f, 4}, // Type 4
};

// --- Game-specific entity types ---
enum GameEntityTypes {
  ENTITY_TYPE_PLAYER = 0,
  ENTITY_TYPE_ZOMBIE,
  ENTITY_TYPE_DAMAGE_TEXT, // Keep for now, maybe use for "Hit!" effects
  ENTITY_TYPE_COUNT
};

// --- Player Container ---
class PlayerContainer : public RenderableEntityContainer {
public:
  float *facing_angles;

  PlayerContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity) {
    facing_angles = new float[capacity];
    std::fill(facing_angles, facing_angles + capacity, 0.0f);
  }

  ~PlayerContainer() override { delete[] facing_angles; }

  uint32_t createEntity(float x, float y, int texture_id) {
    uint32_t index = RenderableEntityContainer::createEntity();
    if (index == INVALID_ID)
      return INVALID_ID;

    x_positions[index] = x;
    y_positions[index] = y;
    widths[index] = PLAYER_SIZE;
    heights[index] = PLAYER_SIZE;
    texture_ids[index] = texture_id;
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

    float *newAngles = new float[newCapacity];

    if (count > 0) {
      std::copy(facing_angles, facing_angles + count, newAngles);
    }
    std::fill(newAngles + count, newAngles + newCapacity, 0.0f);

    delete[] facing_angles;
    facing_angles = newAngles;

    RenderableEntityContainer::resizeArrays(newCapacity);
  }
};

// --- Zombie Container (OPTIMIZED) ---
class ZombieContainer : public RenderableEntityContainer {
public:
  float *speeds;
  float *health; // Used to kill zombies when they hit player
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
                    if (health[i] <= 0) {
                      // Move off-screen if dead
                      x_positions[i] = -10000.0f;
                      y_positions[i] = -10000.0f;
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
  // Player is invincible, always "alive"

  // Textures
  int player_texture_id;
  int zombie_texture_ids[5];

  // Containers
  PlayerContainer *player_container;
  ZombieContainer *zombie_container;
  DamageTextContainer *damage_text_container;

  // FPS tracking
  Uint64 last_fps_time;
  int frame_count;
  float current_fps;

  // Stats
  int hit_count; // Number of times player got hit
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
  ZombieContainer *zombie_container =
      new ZombieContainer(engine, ENTITY_TYPE_ZOMBIE, 0, NUM_ZOMBIES + 500);
  DamageTextContainer *damage_text_container = new DamageTextContainer(
      ENTITY_TYPE_DAMAGE_TEXT, 0, MAX_DAMAGE_TEXTS + 50);

  engine->entityManager.registerEntityType(player_container);
  engine->entityManager.registerEntityType(zombie_container);
  engine->entityManager.registerEntityType(damage_text_container);

  // Initialize game state (Local struct)
  GameState game_state = {};
  game_state.player_container = player_container;
  game_state.zombie_container = zombie_container;
  game_state.damage_text_container = damage_text_container;
  game_state.player_index = INVALID_ID;
  game_state.last_fps_time = SDL_GetTicks();
  game_state.hit_count = 0;

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
      ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1), "Zombies: %d",
                         game_state.zombie_container->count -
                             game_state.hit_count);
      ImGui::End();

      ImGui::Render();

      SDL_SetRenderDrawColor(engine->renderer, 30, 30, 40, 255);
      SDL_RenderClear(engine->renderer);
      engine_render_scene(engine);

      // Render ImGui Over everything
      ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(),
                                            engine->renderer);

      engine_present(engine);
    } else {
      // Vulkan path (if enabled, but we are primarily using SDL renderer in
      // this setup based on file)
      engine_render_scene(engine);
      engine_present(engine);
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

  // Zombie Ships (ship2.png - ship6.png)
  for (int i = 0; i < 5; ++i) {
    snprintf(path_buffer, sizeof(path_buffer), "resource/ship%d.png", i + 2);
    SDL_Surface *z_surf = load_image_to_surface(path_buffer);
    if (!z_surf)
      exit(1);
    game_state->zombie_texture_ids[i] =
        engine_register_texture(engine, z_surf, atlas_x, atlas_y, 0, 0);
    std::cout << "Zombie " << i
              << " Tex ID: " << game_state->zombie_texture_ids[i]
              << " at X: " << atlas_x << std::endl;
    atlas_x += z_surf->w + padding;
    SDL_DestroySurface(z_surf);
  }
  std::cout << "------------------------------" << std::endl;

  // 2. Create Player
  game_state->player_index = game_state->player_container->createEntity(
      WORLD_WIDTH / 2.0f, WORLD_HEIGHT / 2.0f, game_state->player_texture_id);

  // 3. Create Zombies
  for (int i = 0; i < NUM_ZOMBIES; ++i) {
    float x = static_cast<float>(rand() % WORLD_WIDTH);
    float y = static_cast<float>(rand() % WORLD_HEIGHT);

    // Ensure not spawning on top of player
    float dx = x - (WORLD_WIDTH / 2.0f);
    float dy = y - (WORLD_HEIGHT / 2.0f);
    if (dx * dx + dy * dy < 500 * 500) { // Keep clear area
      x += 1000.0f;
    }

    uint8_t type = rand() % 5;
    game_state->zombie_container->createEntity(
        x, y, game_state->zombie_texture_ids[type], type);
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
}

void update_game(Engine *engine, GameState *game_state, float delta_time) {
  // Update Zombies Target (Player)
  if (game_state->player_index != INVALID_ID) {
    float px =
        game_state->player_container->x_positions[game_state->player_index];
    float py =
        game_state->player_container->y_positions[game_state->player_index];
    game_state->zombie_container->setTarget(px, py);
  }
}

void check_collisions(Engine *engine, GameState *game_state) {
  if (game_state->player_index == INVALID_ID)
    return;

  PlayerContainer *player = game_state->player_container;
  ZombieContainer *zombies = game_state->zombie_container;

  float px = player->x_positions[game_state->player_index];
  float py = player->y_positions[game_state->player_index];
  float p_radius = PLAYER_SIZE / 2.2f; // Increased collision radius
  float p_center_x = px + PLAYER_SIZE / 2.0f;
  float p_center_y = py + PLAYER_SIZE / 2.0f;

  // Query zombies near player
  // Broad phase: Circle query around player
  const std::vector<EntityRef> &nearby =
      engine->grid.queryCircle(p_center_x, p_center_y, p_radius + ZOMBIE_SIZE);

  for (const auto &ref : nearby) {
    if (ref.type == ENTITY_TYPE_ZOMBIE) {
      // Precise check
      float zx = zombies->x_positions[ref.index];
      float zy = zombies->y_positions[ref.index];
      float z_radius = ZOMBIE_SIZE / 2.2f; // Increased collision radius
      float z_center_x = zx + ZOMBIE_SIZE / 2.0f;
      float z_center_y = zy + ZOMBIE_SIZE / 2.0f;

      float dx = p_center_x - z_center_x;
      float dy = p_center_y - z_center_y;
      float distSq = dx * dx + dy * dy;
      float combinedRadius = p_radius + z_radius;

      if (distSq < combinedRadius * combinedRadius) {
        // HIT!
        // 1. Increment Counter
        game_state->hit_count++;

        // 2. Kill Zombie
        zombies->health[ref.index] = -1.0f; // Mark as dead

        // Move off-screen immediately so it doesn't hit again
        engine->grid.move(zombies->grid_node_indices[ref.index], -10000.0f,
                          -10000.0f);
        zombies->x_positions[ref.index] = -10000.0f;
        zombies->y_positions[ref.index] = -10000.0f;
      }
    }
  }
}
