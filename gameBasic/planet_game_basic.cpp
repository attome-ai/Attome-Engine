#define _USE_MATH_DEFINES
#include "engineBasic/BasicEngine.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

/**
 * Zombie Game - Basic Engine Version
 *
 * Same game as zombie_game.cpp but using the unoptimized BasicEngine
 * to demonstrate performance impact of various optimizations.
 *
 * Uses 10,000 zombies instead of 1,000,000 because the basic engine
 * cannot handle the larger count at playable framerates.
 */

// --- Constants ---
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1020
#define PLAYER_SIZE 64
#define PLANET_SIZE 32
#define PLAYER_SPEED 600.0f
#define NUM_PLANETS 100000 // Reduced from 1,000,000 for basic engine

// Shooting Constants
#define BULLET_SIZE 26
#define BULLET_SPEED 600.0f
#define FIRE_RATE 0.05f
#define BULLETS_PER_SHOT 25
#define BULLET_SPREAD 0.05f
#define BULLET_LIFETIME 11600.0f
#define BULLET_DAMAGE 1000.0f

// --- Planet Type Stats ---
struct PlanetStats {
  float speed;
  float health;
  int texture_idx;
};

static const PlanetStats PLANET_STATS[5] = {
    {150.0f, 200.0f, 0}, // Type 0
    {150.0f, 175.0f, 1}, // Type 1
    {150.0f, 150.0f, 2}, // Type 2
    {150.0f, 250.0f, 3}, // Type 3
    {150.0f, 300.0f, 4}, // Type 4
};

// --- Game State ---
struct GameState {
  Entity *player = nullptr;

  // FPS tracking
  Uint64 last_fps_time = 0;
  int frame_count = 0;
  float current_fps = 0.0f;

  // Stats
  int hit_count = 0;
  int killed_count = 0;
  int planet_count = 0;

  // Shooting state
  float shoot_cooldown = 0.0f;
  bool mouse_pressed = false;
  int bullet_count = 0;
};

// --- Function Declarations ---
void setup_game(BasicEngine &engine, GameState &game_state);
void handle_input(BasicEngine &engine, const bool *keyboard_state,
                  GameState &game_state, float delta_time);
void update_game(BasicEngine &engine, GameState &game_state, float delta_time);
void check_collisions(BasicEngine &engine, GameState &game_state);

// --- Main Function ---
int main(int argc, char *argv[]) {
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
    return 1;
  }

  srand(static_cast<unsigned int>(time(nullptr)));

  // Create basic engine
  BasicEngine engine;
  if (!engine.initialize(WINDOW_WIDTH, WINDOW_HEIGHT)) {
    std::cerr << "Failed to initialize engine" << std::endl;
    SDL_Quit();
    return 1;
  }

  // Init ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.FontGlobalScale = 2.0f;

  ImGui::StyleColorsDark();
  ImGui_ImplSDL3_InitForSDLRenderer(engine.window, engine.renderer);
  ImGui_ImplSDLRenderer3_Init(engine.renderer);

  // Setup game
  GameState game_state;
  game_state.last_fps_time = SDL_GetTicks();
  setup_game(engine, game_state);

  // Set initial camera position
  if (game_state.player) {
    engine.camera.x = game_state.player->x + PLAYER_SIZE / 2.0f;
    engine.camera.y = game_state.player->y + PLAYER_SIZE / 2.0f;
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

    const bool *keyboard_state = SDL_GetKeyboardState(nullptr);

    // Game logic
    handle_input(engine, keyboard_state, game_state, delta_time);

    // Process bullets (Basic Engine way: brute force update)
    // Note: We need to do this manually here or in Entity::update.
    // Since Entity::update is virtual and generic, let's keep specific logic
    // here or let Entity handle movement. Entity::update handles basics, but we
    // need to manage cooldowns here.
    if (game_state.shoot_cooldown > 0) {
      game_state.shoot_cooldown -= delta_time;
    }

    update_game(engine, game_state, delta_time);
    check_collisions(engine, game_state);

    // Camera zoom controls
    if (keyboard_state[SDL_SCANCODE_EQUALS] ||
        keyboard_state[SDL_SCANCODE_KP_PLUS]) {
      engine.camera.width *= 0.98f;
      engine.camera.height *= 0.98f;
    }
    if (keyboard_state[SDL_SCANCODE_MINUS] ||
        keyboard_state[SDL_SCANCODE_KP_MINUS]) {
      engine.camera.width *= 1.02f;
      engine.camera.height *= 1.02f;
    }

    // Update engine (serial, unoptimized)
    engine.update(delta_time);

    // Start ImGui frame
    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    // Stats overlay
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::SetNextWindowSize(ImVec2(0, 0));
    ImGui::Begin("Stats", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::TextColored(ImVec4(1, 0, 0, 1), "BASIC ENGINE (UNOPTIMIZED)");
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "FPS: %.1f", game_state.current_fps);
    ImGui::TextColored(ImVec4(1, 0, 0, 1), "HITS: %d", game_state.hit_count);
    ImGui::TextColored(ImVec4(0, 1, 0, 1), "KILLS: %d",
                       game_state.killed_count);
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Bullets: %d",
                       game_state.bullet_count);
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1), "Planets: %d",
                       game_state.planet_count - game_state.killed_count);
    ImGui::End();

    ImGui::Render();

    // Render (individual draw calls, unoptimized)
    engine.render();

    // Render ImGui
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(),
                                          engine.renderer);

    SDL_RenderPresent(engine.renderer);
  }

  // Cleanup
  ImGui_ImplSDLRenderer3_Shutdown();
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();

  engine.shutdown();
  SDL_Quit();

  return 0;
}

void setup_game(BasicEngine &engine, GameState &game_state) {
  std::cout << "--- BASIC ENGINE TEXTURE LOADING ---" << std::endl;

  // Load textures individually (ANTI-OPTIMIZATION: no atlas)
  engine.player_texture = engine.loadTexture("resource/ship1.png");
  std::cout << "Loaded player texture" << std::endl;

  for (int i = 0; i < 5; ++i) {
    char path[256];
    snprintf(path, sizeof(path), "resource/ship%d.png", i + 2);
    engine.planet_textures[i] = engine.loadTexture(path);
    std::cout << "Loaded planet texture " << i << std::endl;
  }

  // Load bullet texture
  engine.bullet_texture = engine.loadTexture("resource/shoot1.png");
  if (!engine.bullet_texture) {
    std::cerr << "Failed to load bullet texture!" << std::endl;
  }

  std::cout << "-----------------------------------" << std::endl;

  // Create player
  game_state.player = engine.createEntity(BasicEntityType::PLAYER);
  game_state.player->x = BASIC_WORLD_WIDTH / 2.0f;
  game_state.player->y = BASIC_WORLD_HEIGHT / 2.0f;
  game_state.player->width = PLAYER_SIZE;
  game_state.player->height = PLAYER_SIZE;
  game_state.player->texture = engine.player_texture;
  game_state.player->z_index = 100;

  // Create planets
  std::cout << "Creating " << NUM_PLANETS << " planets..." << std::endl;
  for (int i = 0; i < NUM_PLANETS; ++i) {
    float x = static_cast<float>(rand() % BASIC_WORLD_WIDTH);
    float y = static_cast<float>(rand() % BASIC_WORLD_HEIGHT);

    // Keep clear area around player spawn
    float dx = x - (BASIC_WORLD_WIDTH / 2.0f);
    float dy = y - (BASIC_WORLD_HEIGHT / 2.0f);
    if (dx * dx + dy * dy < 500 * 500) {
      x += 1000.0f;
    }

    uint8_t type = rand() % 5;

    Entity *planet = engine.createEntity(BasicEntityType::PLANET);
    planet->x = x;
    planet->y = y;
    planet->width = PLANET_SIZE;
    planet->height = PLANET_SIZE;
    planet->texture = engine.planet_textures[type];
    planet->planet_type = type;
    planet->speed = PLANET_STATS[type].speed;
    planet->health = PLANET_STATS[type].health;
    planet->max_health = PLANET_STATS[type].health;
    planet->z_index = 50;
  }

  game_state.planet_count = NUM_PLANETS;
  std::cout << "Created " << NUM_PLANETS << " planets" << std::endl;
}

void handle_input(BasicEngine &engine, const bool *keyboard_state,
                  GameState &game_state, float delta_time) {
  if (!game_state.player)
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
    float length = std::sqrt(dx * dx + dy * dy);
    dx /= length;
    dy /= length;

    float move_speed = PLAYER_SPEED * delta_time;

    // Sprint
    if (keyboard_state[SDL_SCANCODE_LSHIFT])
      move_speed *= 2.0f;

    Entity *player = game_state.player;
    float new_x = player->x + dx * move_speed;
    float new_y = player->y + dy * move_speed;

    // Clamp to world
    if (new_x < 0)
      new_x = 0;
    if (new_x > BASIC_WORLD_WIDTH - PLAYER_SIZE)
      new_x = BASIC_WORLD_WIDTH - PLAYER_SIZE;
    if (new_y < 0)
      new_y = 0;
    if (new_y > BASIC_WORLD_HEIGHT - PLAYER_SIZE)
      new_y = BASIC_WORLD_HEIGHT - PLAYER_SIZE;

    player->x = new_x;
    player->y = new_y;

    // Update camera
    engine.camera.x = new_x + PLAYER_SIZE / 2.0f;
    engine.camera.y = new_y + PLAYER_SIZE / 2.0f;
  }

  // Shooting Logic
  float mx, my;
  SDL_GetMouseState(&mx, &my);
  bool mouseDown =
      (SDL_GetMouseState(NULL, NULL) & SDL_BUTTON_MASK(SDL_BUTTON_LEFT));

  if (mouseDown && game_state.shoot_cooldown <= 0.0f) {
    // Calculate spawn position (center of player)
    float px = game_state.player->x + PLAYER_SIZE / 2.0f;
    float py = game_state.player->y + PLAYER_SIZE / 2.0f;

    // Adjust mouse to world coordinates
    float wx = mx + (engine.camera.x - engine.camera.width / 2.0f);
    float wy = my + (engine.camera.y - engine.camera.height / 2.0f);

    // Direction
    float dir_x = wx - px;
    float dir_y = wy - py;
    float len = sqrtf(dir_x * dir_x + dir_y * dir_y);
    if (len > 0) {
      dir_x /= len;
      dir_y /= len;
    }

    // Base angle
    float base_angle = atan2f(dir_y, dir_x);

    for (int i = 0; i < BULLETS_PER_SHOT; ++i) {
      // Deterministic Fan Spread (Same as Optimized Engine)
      float spread_angle =
          base_angle + (i - (BULLETS_PER_SHOT - 1) / 2.0f) * BULLET_SPREAD;

      float vx = cosf(spread_angle);
      float vy = sinf(spread_angle);

      Entity *bullet = engine.createEntity(BasicEntityType::BULLET);
      bullet->x = px - BULLET_SIZE / 2.0f;
      bullet->y = py - BULLET_SIZE / 2.0f;
      bullet->width = BULLET_SIZE;
      bullet->height = BULLET_SIZE;
      bullet->texture = engine.bullet_texture;
      bullet->vx = vx * BULLET_SPEED;
      bullet->vy = vy * BULLET_SPEED;
      bullet->rotation =
          spread_angle + M_PI / 2.0f; // +90 deg if sprite points up
      bullet->lifetime = BULLET_LIFETIME;
      bullet->z_index = 80;
    }

    game_state.shoot_cooldown = FIRE_RATE;
  }
}

void update_game(BasicEngine &engine, GameState &game_state, float delta_time) {
  if (!game_state.player)
    return;

  float px = game_state.player->x;
  float py = game_state.player->y;

  // ANTI-OPTIMIZATION: Update ALL zombie targets every frame
  // Even zombies that are far away get updated
  for (Entity *entity : engine.entities) {
    if (entity->type == BasicEntityType::PLANET && entity->active) {
      entity->target_x = px;
      entity->target_y = py;
    } else if (entity->type == BasicEntityType::BULLET && entity->active) {
      // Update bullet position
      entity->x += entity->vx * delta_time;
      entity->y += entity->vy * delta_time;
      entity->lifetime -= delta_time;
      if (entity->lifetime <= 0) {
        entity->active = false;
      }
    }
  }
}

void check_collisions(BasicEngine &engine, GameState &game_state) {
  if (!game_state.player)
    return;

  Entity *player = game_state.player;
  float p_center_x = player->x + PLAYER_SIZE / 2.0f;
  float p_center_y = player->y + PLAYER_SIZE / 2.0f;
  float p_radius = PLAYER_SIZE / 2.2f;

  // ANTI-OPTIMIZATION: O(n) query for nearby planets
  // Optimized engine uses spatial grid for O(1) cell lookup
  float query_radius = p_radius + PLANET_SIZE;
  std::vector<Entity *> nearby =
      engine.queryEntitiesInRadius(p_center_x, p_center_y, query_radius);

  for (Entity *entity : nearby) {
    if (entity->type != BasicEntityType::PLANET || !entity->active)
      continue;

    // Precise circle collision
    float p_center_x_entity = entity->x + PLANET_SIZE / 2.0f;
    float p_center_y_entity = entity->y + PLANET_SIZE / 2.0f;
    float p_radius_entity = PLANET_SIZE / 2.2f;

    float dx = p_center_x - p_center_x_entity;
    float dy = p_center_y - p_center_y_entity;
    float dist_sq = dx * dx + dy * dy;
    float combined_radius = p_radius + p_radius_entity;

    if (dist_sq < combined_radius * combined_radius) {
      // HIT!
      game_state.hit_count++;

      // Destroy planet
      entity->health = -1.0f;
      entity->active = false;
      entity->visible = false;

      // Move off-screen
      entity->x = -10000.0f;
      entity->y = -10000.0f;
    }
  }

  // Bullet Collisions (O(N^2) effectively against planets closer to bullets)
  int active_bullets = 0;
  for (Entity *bullet : engine.entities) {
    if (bullet->type != BasicEntityType::BULLET || !bullet->active)
      continue;

    active_bullets++;

    float bx = bullet->x + BULLET_SIZE / 2.0f;
    float by = bullet->y + BULLET_SIZE / 2.0f;
    float b_radius = BULLET_SIZE / 2.0f;

    // Query planets near bullet (O(N) again!)
    float query_radius = b_radius + PLANET_SIZE / 2.2f;
    std::vector<Entity *> nearbyPlanets =
        engine.queryEntitiesInRadius(bx, by, query_radius);

    for (Entity *planet : nearbyPlanets) {
      if (planet->type != BasicEntityType::PLANET || !planet->active)
        continue;

      // Hit!
      planet->health -= BULLET_DAMAGE;
      if (planet->health <= 0) {
        game_state.killed_count++;
        planet->active = false;
        planet->visible = false;
        planet->x = -10000.0f;
        planet->y = -10000.0f;
      }

      bullet->active = false;
      bullet->visible = false;
      break; // Bullet hits one planet
    }
  }
  game_state.bullet_count = active_bullets;
}
