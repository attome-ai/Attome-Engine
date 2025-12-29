#include "BasicEngine.h"
#include "../game/stb_image.h"
#include <cmath>
#include <iostream>

/**
 * BasicEngine Implementation
 *
 * Every function here is intentionally UNOPTIMIZED to show
 * the performance impact of common naive approaches.
 */

// =============================================================================
// Entity Implementation
// =============================================================================

void Entity::update(float delta_time) {
  if (!active)
    return;

  // Virtual function overhead on every entity
  if (type == BasicEntityType::ZOMBIE) {
    // Zombie movement - calculate every frame (no caching)
    float dx = target_x - x;
    float dy = target_y - y;
    float dist = std::sqrt(dx * dx + dy * dy); // sqrt every frame

    if (dist > 1.0f) {
      // Normalize (division every frame)
      float nx = dx / dist;
      float ny = dy / dist;
      x += nx * speed * delta_time;
      y += ny * speed * delta_time;
    }

    // Boundary clamping
    if (x < 0)
      x = 0;
    if (x > BASIC_WORLD_WIDTH - width)
      x = BASIC_WORLD_WIDTH - width;
    if (y < 0)
      y = 0;
    if (y > BASIC_WORLD_HEIGHT - height)
      y = BASIC_WORLD_HEIGHT - height;
  } else if (type == BasicEntityType::DAMAGE_TEXT) {
    // Float upward
    y -= 100.0f * delta_time;
    lifetime -= delta_time;
    if (lifetime <= 0) {
      active = false;
      visible = false;
    }
  }
}

void Entity::render(SDL_Renderer *renderer, float camera_x, float camera_y) {
  if (!visible || !active || !texture)
    return;

  // Calculate screen position
  float screen_x = x - camera_x;
  float screen_y = y - camera_y;

  // Create destination rect
  SDL_FRect dest = {screen_x, screen_y, width, height};

  // ANTI-OPTIMIZATION: Individual draw call per entity
  // In the optimized engine, entities are batched by texture
  SDL_RenderTexture(renderer, texture, nullptr, &dest);
}

// =============================================================================
// BasicEngine Implementation
// =============================================================================

BasicEngine::BasicEngine() {
  // Reserve some space, but still uses dynamic allocation
  entities.reserve(20000);
}

BasicEngine::~BasicEngine() { shutdown(); }

bool BasicEngine::initialize(int window_width, int window_height) {
  // Create window
  window = SDL_CreateWindow("Basic Engine - UNOPTIMIZED", window_width,
                            window_height, 0);
  if (!window) {
    std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
    return false;
  }

  // Create renderer
  renderer = SDL_CreateRenderer(window, nullptr);
  if (!renderer) {
    std::cerr << "Failed to create renderer: " << SDL_GetError() << std::endl;
    SDL_DestroyWindow(window);
    return false;
  }

  // Initialize camera
  camera.x = BASIC_WORLD_WIDTH / 2.0f;
  camera.y = BASIC_WORLD_HEIGHT / 2.0f;
  camera.width = static_cast<float>(window_width);
  camera.height = static_cast<float>(window_height);

  last_frame_time = SDL_GetTicks();

  return true;
}

void BasicEngine::shutdown() {
  // Clean up entities
  for (Entity *entity : entities) {
    delete entity;
  }
  entities.clear();

  // Clean up textures - individual cleanup (more overhead)
  if (player_texture) {
    SDL_DestroyTexture(player_texture);
    player_texture = nullptr;
  }
  for (int i = 0; i < 5; ++i) {
    if (zombie_textures[i]) {
      SDL_DestroyTexture(zombie_textures[i]);
      zombie_textures[i] = nullptr;
    }
  }
  if (damage_text_texture) {
    SDL_DestroyTexture(damage_text_texture);
    damage_text_texture = nullptr;
  }

  if (renderer) {
    SDL_DestroyRenderer(renderer);
    renderer = nullptr;
  }
  if (window) {
    SDL_DestroyWindow(window);
    window = nullptr;
  }
}

// Load image to surface (same as optimized version)
SDL_Surface *basic_load_image_to_surface(const char *filepath) {
  int w, h, c;
  unsigned char *data = stbi_load(filepath, &w, &h, &c, 4);
  if (!data) {
    std::cerr << "Failed to load image: " << filepath << std::endl;
    // Return a fallback surface (magenta square)
    SDL_Surface *surface = SDL_CreateSurface(32, 32, SDL_PIXELFORMAT_RGBA8888);
    SDL_FillSurfaceRect(surface, nullptr,
                        SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format),
                                    nullptr, 255, 0, 255, 255));
    return surface;
  }

  int pitch = w * 4;
  SDL_Surface *temp =
      SDL_CreateSurfaceFrom(w, h, SDL_PIXELFORMAT_RGBA32, data, pitch);
  if (temp) {
    SDL_Surface *copy = SDL_DuplicateSurface(temp);
    SDL_DestroySurface(temp);
    stbi_image_free(data);
    return copy;
  }
  stbi_image_free(data);
  return nullptr;
}

SDL_Texture *BasicEngine::loadTexture(const char *filepath) {
  SDL_Surface *surface = basic_load_image_to_surface(filepath);
  if (!surface)
    return nullptr;

  // ANTI-OPTIMIZATION: Create individual texture (no atlas)
  // Each texture is a separate GPU resource, causing state changes
  SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
  SDL_DestroySurface(surface);

  return texture;
}

Entity *BasicEngine::createEntity(BasicEntityType type) {
  // ANTI-OPTIMIZATION: Heap allocation for every entity
  // Causes memory fragmentation and allocation overhead
  Entity *entity = new Entity();
  entity->type = type;
  entity->active = true;
  entity->visible = true;

  entities.push_back(entity);
  return entity;
}

void BasicEngine::destroyEntity(Entity *entity) {
  // ANTI-OPTIMIZATION: O(n) removal from vector
  auto it = std::find(entities.begin(), entities.end(), entity);
  if (it != entities.end()) {
    delete *it;
    entities.erase(it); // O(n) due to element shifting
  }
}

void BasicEngine::update(float delta_time) {
  // ANTI-OPTIMIZATION: Serial update loop
  // No parallelism, no SIMD, no vectorization hints
  // Also virtual function call overhead for each entity

  for (Entity *entity : entities) {
    if (entity->active) {
      entity->update(delta_time); // Virtual call overhead
    }
  }

  // Calculate FPS
  calculateFPS(delta_time);
}

void BasicEngine::render() {
  // Clear screen
  SDL_SetRenderDrawColor(renderer, 30, 30, 40, 255);
  SDL_RenderClear(renderer);

  // Calculate view bounds
  float view_x1 = camera.x - camera.width / 2.0f;
  float view_y1 = camera.y - camera.height / 2.0f;
  float view_x2 = camera.x + camera.width / 2.0f;
  float view_y2 = camera.y + camera.height / 2.0f;

  // ANTI-OPTIMIZATION: Collect and sort visible entities every frame
  // No spatial culling, just bounds check on ALL entities
  std::vector<Entity *> visible_entities;
  visible_entities.reserve(entities.size());

  for (Entity *entity : entities) {
    if (!entity->active || !entity->visible)
      continue;

    // Simple bounds check (no spatial grid)
    if (entity->x + entity->width >= view_x1 && entity->x <= view_x2 &&
        entity->y + entity->height >= view_y1 && entity->y <= view_y2) {
      visible_entities.push_back(entity);
    }
  }

  // ANTI-OPTIMIZATION: Sort with pointer dereferencing in comparator
  // Every comparison requires multiple memory lookups
  std::sort(visible_entities.begin(), visible_entities.end(),
            [](const Entity *a, const Entity *b) {
              // Multiple pointer dereferences per comparison
              if (a->z_index != b->z_index)
                return a->z_index < b->z_index;
              return a->y < b->y; // Additional deref
            });

  // ANTI-OPTIMIZATION: Individual draw calls (no batching)
  // Each call requires:
  // 1. Texture bind (if different)
  // 2. State setup
  // 3. Single quad submission
  // 4. Driver overhead

  for (Entity *entity : visible_entities) {
    entity->render(renderer, view_x1, view_y1);
  }

  // Note: SDL_RenderPresent is called by the game loop after ImGui rendering
}

std::vector<Entity *> BasicEngine::queryEntitiesInRadius(float x, float y,
                                                         float radius) {
  // ANTI-OPTIMIZATION: O(n) linear scan of ALL entities
  // Optimized engine would use spatial grid for O(1) cell lookup

  std::vector<Entity *> result;
  float radius_sq = radius * radius;

  for (Entity *entity : entities) {
    if (!entity->active)
      continue;

    float dx = (entity->x + entity->width / 2) - x;
    float dy = (entity->y + entity->height / 2) - y;
    float dist_sq = dx * dx + dy * dy;

    if (dist_sq <= radius_sq) {
      result.push_back(entity);
    }
  }

  return result;
}

std::vector<Entity *> BasicEngine::queryEntitiesInRect(float x1, float y1,
                                                       float x2, float y2) {
  // ANTI-OPTIMIZATION: O(n) linear scan
  std::vector<Entity *> result;

  for (Entity *entity : entities) {
    if (!entity->active)
      continue;

    if (entity->x + entity->width >= x1 && entity->x <= x2 &&
        entity->y + entity->height >= y1 && entity->y <= y2) {
      result.push_back(entity);
    }
  }

  return result;
}

std::vector<Entity *> BasicEngine::getEntitiesOfType(BasicEntityType type) {
  // ANTI-OPTIMIZATION: O(n) linear scan to find by type
  // Optimized engine has separate containers per type

  std::vector<Entity *> result;

  for (Entity *entity : entities) {
    if (entity->type == type && entity->active) {
      result.push_back(entity);
    }
  }

  return result;
}

void BasicEngine::calculateFPS(float delta_time) {
  // Simple FPS calculation
  fps = 0.95f * fps + 0.05f * (1.0f / delta_time);
}
