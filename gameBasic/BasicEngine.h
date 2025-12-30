#ifndef BASIC_ENGINE_H
#define BASIC_ENGINE_H

/**
 * BasicEngine - Intentionally UNOPTIMIZED 2D game engine
 *
 * This engine uses common naive approaches to demonstrate
 * how optimizations affect performance. DO NOT use this
 * as a template for actual game development.
 *
 * Anti-patterns implemented:
 * 1. OOP entity objects (AOS instead of SOA) - poor cache locality
 * 2. Individual textures per type (no atlas) - texture state changes
 * 3. No spatial partitioning - O(n) queries, O(n²) collision
 * 4. Individual draw calls - no batching
 * 5. Sort with pointer dereferencing - slow comparisons
 * 6. Single-threaded updates - no parallelism
 */

#include <SDL3/SDL.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

// World constants (same as optimized engine)
static constexpr int BASIC_WORLD_WIDTH = 50000;
static constexpr int BASIC_WORLD_HEIGHT = 50000;

// Entity types
enum class BasicEntityType { PLAYER, PLANET, BULLET, DAMAGE_TEXT, COUNT };

/**
 * Entity class - All data in one object (AOS pattern)
 *
 * ANTI-OPTIMIZATION: This causes poor cache locality because
 * iterating over entities means loading full objects, not just
 * the data we need (e.g., just positions for movement).
 */
class Entity {
public:
  // Transform
  float x = 0.0f;
  float y = 0.0f;
  float width = 32.0f;
  float height = 32.0f;

  // Rendering
  SDL_Texture *texture = nullptr; // Individual texture pointer
  int z_index = 0;
  bool visible = true;

  // State
  bool active = true;
  BasicEntityType type = BasicEntityType::PLANET;

  // Game-specific data (mixed in same object - bad for cache)
  float speed = 0.0f;
  float health = 100.0f;
  float max_health = 100.0f;
  float lifetime = 0.0f;
  uint8_t zombie_type = 0;

  // Target for movement (stored per-entity - wasteful)
  float target_x = 0.0f;
  float target_y = 0.0f;

  // Velocity (for bullets)
  float vx = 0.0f;
  float vy = 0.0f;

  // Rotation (radians)
  float rotation = 0.0f;

  Entity() = default;
  virtual ~Entity() = default;

  // Virtual methods add vtable overhead
  virtual void update(float delta_time);
  virtual void render(SDL_Renderer *renderer, float camera_x, float camera_y);

  // Collision helper
  bool intersects(const Entity &other) const {
    return x < other.x + other.width && x + width > other.x &&
           y < other.y + other.height && y + height > other.y;
  }

  float distanceToSq(const Entity &other) const {
    float dx = (x + width / 2) - (other.x + other.width / 2);
    float dy = (y + height / 2) - (other.y + other.height / 2);
    return dx * dx + dy * dy;
  }
};

/**
 * BasicCamera - Simple camera for view frustum
 */
struct BasicCamera {
  float x = 0.0f;
  float y = 0.0f;
  float width = 1600.0f;
  float height = 1020.0f;
};

/**
 * BasicEngine - Main engine class with no optimizations
 */
class BasicEngine {
public:
  // SDL resources
  SDL_Window *window = nullptr;
  SDL_Renderer *renderer = nullptr;

  // Camera
  BasicCamera camera;

  // Entity storage - uses pointer vector (ANTI-OPTIMIZATION)
  // This causes:
  // 1. Pointer chasing on every access
  // 2. Fragmented memory allocation
  // 3. Poor cache utilization
  std::vector<Entity *> entities;

  // Individual textures per entity type (ANTI-OPTIMIZATION)
  // No atlas means texture state changes between draw calls
  SDL_Texture *player_texture = nullptr;
  SDL_Texture *planet_textures[5] = {nullptr};
  SDL_Texture *bullet_texture = nullptr;
  SDL_Texture *damage_text_texture = nullptr;

  // Timing
  Uint64 last_frame_time = 0;
  float fps = 0.0f;

  // Constructor/Destructor
  BasicEngine();
  ~BasicEngine();

  // Initialization
  bool initialize(int window_width, int window_height);
  void shutdown();

  // Texture loading - individual textures (no atlas)
  SDL_Texture *loadTexture(const char *filepath);

  // Entity management
  Entity *createEntity(BasicEntityType type);
  void destroyEntity(Entity *entity);

  // Update all entities (SERIAL - no parallelism)
  void update(float delta_time);

  // Render all entities (INDIVIDUAL DRAW CALLS - no batching)
  void render();

  // Collision detection (O(n²) BRUTE FORCE)
  std::vector<Entity *> queryEntitiesInRadius(float x, float y, float radius);
  std::vector<Entity *> queryEntitiesInRect(float x1, float y1, float x2,
                                            float y2);

  // Get all entities of a type (O(n) linear scan)
  std::vector<Entity *> getEntitiesOfType(BasicEntityType type);

  // Calculate FPS
  void calculateFPS(float delta_time);
};

// Helper to load image (uses stb_image like the optimized version)
SDL_Surface *basic_load_image_to_surface(const char *filepath);

#endif // BASIC_ENGINE_H
