#include "ATMEngine.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr int kWindowW = 1600;
constexpr int kWindowH = 1020;
constexpr int kWorldW = 50000;
constexpr int kWorldH = 50000;

constexpr int16_t kPlayerSize = 50;
constexpr int16_t kEnemySize = 30;
constexpr int16_t kStaticSize = 30;
constexpr int16_t kBulletSize = 12;

constexpr int kTotalEnemyPlanets = 30000; // 50% dynamic + 50% hybrid
constexpr int kStaticPlanets = 20000;     // static planets just exist

constexpr float kPlayerSpeed = 850.0f;
constexpr float kShootCooldownSec = 0.05f;
constexpr float kBulletSpeed = 1400.0f;
constexpr float kBulletLifetimeSec = 2.4f;

class PlayerContainer : public RenderableEntityContainer {
public:
  explicit PlayerContainer(int initial_capacity)
      : RenderableEntityContainer(-1, 12, initial_capacity) {}
  void update(float delta_time) override { (void)delta_time; }
};

class DynamicEnemyContainer : public RenderableEntityContainer {
public:
  explicit DynamicEnemyContainer(Engine *engine_ptr, int initial_capacity)
      : RenderableEntityContainer(-1, 6, initial_capacity),
        engine(engine_ptr), speed(initial_capacity, 0.0f),
        phase(initial_capacity, 0.0f) {}

  uint32_t createEntity() override {
    const uint32_t slot = RenderableEntityContainer::createEntity();
    if (slot != INVALID_ID) {
      speed[slot] = 420.0f + static_cast<float>((slot * 17U) % 350U);
      phase[slot] = static_cast<float>((slot * 13U) % 360U) * 0.017453292f;
    }
    return slot;
  }

  void removeEntity(size_t index) override {
    if (index < static_cast<size_t>(count)) {
      const size_t last = static_cast<size_t>(count - 1);
      if (index < last) {
        speed[index] = speed[last];
        phase[index] = phase[last];
      }
    }
    RenderableEntityContainer::removeEntity(index);
  }

  void update(float delta_time) override {
    if (count <= 0) {
      last_updated = 0;
      return;
    }

    const float dt = std::min(delta_time, 0.05f);
    last_updated = count;
    for (int i = 0; i < count; ++i) {
      float &x = x_positions[i];
      float &y = y_positions[i];

      const float dx = target_x - x;
      const float dy = target_y - y;
      const float dist2 = dx * dx + dy * dy;
      if (dist2 > 1.0f) {
        const float inv_len = 1.0f / std::sqrt(dist2);
        x += dx * inv_len * speed[i] * dt;
        y += dy * inv_len * speed[i] * dt;
        rotations[i] = std::atan2(dy, dx);
      }

      x = std::clamp(x, 0.0f,
                     static_cast<float>(kWorldW - std::max<int16_t>(1, widths[i])));
      y = std::clamp(y, 0.0f,
                     static_cast<float>(kWorldH - std::max<int16_t>(1, heights[i])));

      phase[i] += dt * 1.2f;
      rotations[i] += std::sin(phase[i]) * 0.08f;

      const int32_t node = grid_node_indices[i];
      if (node != -1) {
        engine->grid.move(node, x, y);
      }

      uint16_t cx = 0;
      uint16_t cy = 0;
      engine->grid.getCellCoords(x, y, cx, cy);
      cell_x[i] = cx;
      cell_y[i] = cy;
    }
  }

protected:
  void resizeArrays(int new_capacity) override {
    RenderableEntityContainer::resizeArrays(new_capacity);
    speed.resize(new_capacity, count, 0.0f);
    phase.resize(new_capacity, count, 0.0f);
  }

public:
  Engine *engine{nullptr};
  DynamicArray<float> speed;
  DynamicArray<float> phase;
  float target_x{0.0f};
  float target_y{0.0f};
  int last_updated{0};
};

class HybridEnemyContainer : public RenderableEntityContainer {
public:
  explicit HybridEnemyContainer(Engine *engine_ptr, int initial_capacity)
      : RenderableEntityContainer(-1, 8, initial_capacity),
        engine(engine_ptr), speed(initial_capacity, 0.0f),
        phase(initial_capacity, 0.0f) {}

  uint32_t createEntity() override {
    const uint32_t slot = RenderableEntityContainer::createEntity();
    if (slot != INVALID_ID) {
      speed[slot] = 420.0f + static_cast<float>((slot * 19U) % 350U);
      phase[slot] = static_cast<float>((slot * 7U) % 360U) * 0.017453292f;
    }
    return slot;
  }

  void removeEntity(size_t index) override {
    if (index < static_cast<size_t>(count)) {
      const size_t last = static_cast<size_t>(count - 1);
      if (index < last) {
        speed[index] = speed[last];
        phase[index] = phase[last];
      }
    }
    RenderableEntityContainer::removeEntity(index);
  }

  void update(float delta_time) override { (void)delta_time; }

  void updateVisible(const std::vector<uint32_t> &active_indices,
                     float delta_time) override {
    last_active_count = static_cast<int>(active_indices.size());
    last_updated = last_active_count;

    if (active_indices.empty()) {
      return;
    }

    const float dt = std::min(delta_time, 0.05f);
    for (uint32_t idx : active_indices) {
      if (idx >= static_cast<uint32_t>(count)) {
        continue;
      }

      float &x = x_positions[idx];
      float &y = y_positions[idx];

      const float dx = target_x - x;
      const float dy = target_y - y;
      const float dist2 = dx * dx + dy * dy;
      if (dist2 > 1.0f) {
        const float inv_len = 1.0f / std::sqrt(dist2);
        x += dx * inv_len * speed[idx] * dt;
        y += dy * inv_len * speed[idx] * dt;
        rotations[idx] = std::atan2(dy, dx);
      }

      x = std::clamp(
          x, 0.0f, static_cast<float>(kWorldW - std::max<int16_t>(1, widths[idx])));
      y = std::clamp(
          y, 0.0f, static_cast<float>(kWorldH - std::max<int16_t>(1, heights[idx])));

      phase[idx] += dt * 1.4f;
      rotations[idx] += std::cos(phase[idx]) * 0.08f;

      const int32_t node = grid_node_indices[idx];
      if (node != -1) {
        engine->grid.move(node, x, y);
      }

      uint16_t cx = 0;
      uint16_t cy = 0;
      engine->grid.getCellCoords(x, y, cx, cy);
      cell_x[idx] = cx;
      cell_y[idx] = cy;
    }
  }

protected:
  void resizeArrays(int new_capacity) override {
    RenderableEntityContainer::resizeArrays(new_capacity);
    speed.resize(new_capacity, count, 0.0f);
    phase.resize(new_capacity, count, 0.0f);
  }

public:
  Engine *engine{nullptr};
  DynamicArray<float> speed;
  DynamicArray<float> phase;
  float target_x{0.0f};
  float target_y{0.0f};
  int last_active_count{0};
  int last_updated{0};
};

class StaticPlanetContainer : public RenderableEntityContainer {
public:
  explicit StaticPlanetContainer(int initial_capacity)
      : RenderableEntityContainer(-1, 3, initial_capacity) {}
  void update(float delta_time) override { (void)delta_time; }
};

class BulletContainer : public RenderableEntityContainer {
public:
  explicit BulletContainer(Engine *engine_ptr, int initial_capacity)
      : RenderableEntityContainer(-1, 10, initial_capacity),
        engine(engine_ptr), vx(initial_capacity, 0.0f),
        vy(initial_capacity, 0.0f), life(initial_capacity, 0.0f) {}

  uint32_t createEntity() override {
    const uint32_t slot = RenderableEntityContainer::createEntity();
    if (slot != INVALID_ID) {
      vx[slot] = 0.0f;
      vy[slot] = 0.0f;
      life[slot] = 0.0f;
    }
    return slot;
  }

  void removeEntity(size_t index) override {
    if (index < static_cast<size_t>(count)) {
      const size_t last = static_cast<size_t>(count - 1);
      if (index < last) {
        vx[index] = vx[last];
        vy[index] = vy[last];
        life[index] = life[last];
      }
    }
    RenderableEntityContainer::removeEntity(index);
  }

  void update(float delta_time) override {
    const float dt = std::min(delta_time, 0.05f);
    for (int i = 0; i < count;) {
      x_positions[i] += vx[i] * dt;
      y_positions[i] += vy[i] * dt;
      life[i] -= dt;

      const bool expired =
          life[i] <= 0.0f || x_positions[i] < 0.0f || y_positions[i] < 0.0f ||
          x_positions[i] > static_cast<float>(kWorldW) ||
          y_positions[i] > static_cast<float>(kWorldH);

      if (expired) {
        engine->entityManager.removeEntity(static_cast<uint32_t>(i), type_id,
                                           &engine->grid);
        continue;
      }

      const int32_t node = grid_node_indices[i];
      if (node != -1) {
        engine->grid.move(node, x_positions[i], y_positions[i]);
      }
      uint16_t cx = 0;
      uint16_t cy = 0;
      engine->grid.getCellCoords(x_positions[i], y_positions[i], cx, cy);
      cell_x[i] = cx;
      cell_y[i] = cy;
      ++i;
    }
  }

protected:
  void resizeArrays(int new_capacity) override {
    RenderableEntityContainer::resizeArrays(new_capacity);
    vx.resize(new_capacity, count, 0.0f);
    vy.resize(new_capacity, count, 0.0f);
    life.resize(new_capacity, count, 0.0f);
  }

public:
  Engine *engine{nullptr};
  DynamicArray<float> vx;
  DynamicArray<float> vy;
  DynamicArray<float> life;
};

struct GameState {
  Engine *engine{nullptr};

  PlayerContainer *player_container{nullptr};
  DynamicEnemyContainer *dynamic_container{nullptr};
  HybridEnemyContainer *hybrid_container{nullptr};
  StaticPlanetContainer *static_container{nullptr};
  BulletContainer *bullet_container{nullptr};

  int player_type{-1};
  int dynamic_type{-1};
  int hybrid_type{-1};
  int static_type{-1};
  int bullet_type{-1};

  int16_t tex_player{-1};
  int16_t tex_dynamic{-1};
  int16_t tex_hybrid{-1};
  int16_t tex_static{-1};
  int16_t tex_bullet{-1};

  EntityHandle player{};
  std::vector<EntityHandle> bullet_handles;

  float player_x{(kWorldW * 0.5f) - (kPlayerSize * 0.5f)};
  float player_y{(kWorldH * 0.5f) - (kPlayerSize * 0.5f)};

  bool mouse_pressed{false};
  float shoot_cooldown{0.0f};

  int hits{0};
  int kills{0};

  float console_timer{0.0f};
  int console_frames{0};
  float title_timer{0.0f};

  std::mt19937 rng{0x9E3779B9U};
};

static SDL_Surface *makeColorSurface(int width, int height, uint8_t r, uint8_t g,
                                     uint8_t b, uint8_t a = 255) {
  SDL_Surface *surface =
      SDL_CreateSurface(width, height, SDL_PIXELFORMAT_ABGR8888);
  if (!surface) {
    return nullptr;
  }
  SDL_FillSurfaceRect(surface, nullptr,
                      SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format),
                                  nullptr, r, g, b, a));
  return surface;
}

static float randomRange(std::mt19937 &rng, float min_v, float max_v) {
  std::uniform_real_distribution<float> dist(min_v, max_v);
  return dist(rng);
}

static bool resolveSlot(Engine *engine, const EntityHandle &handle,
                        uint32_t &slot) {
  if (!engine || !handle.isValid()) {
    return false;
  }
  return engine->entityManager.resolveEntitySlot(handle, slot);
}

static bool makeHandleFromSlot(Engine *engine, int type_id, uint32_t slot,
                               EntityHandle &out_handle) {
  out_handle = {};
  if (!engine || type_id < 0 ||
      type_id >= static_cast<int>(engine->entityManager.containers.size()) ||
      type_id >= static_cast<int>(engine->entityManager.type_states.size())) {
    return false;
  }

  auto *container = engine->entityManager.containers[type_id].get();
  if (!container || slot >= static_cast<uint32_t>(container->count)) {
    return false;
  }

  const uint32_t entity_id = container->entity_ids[slot];
  if (entity_id == INVALID_ID) {
    return false;
  }

  auto &state = engine->entityManager.type_states[type_id];
  if (entity_id >= state.entity_generations.size()) {
    return false;
  }

  const uint16_t generation = state.entity_generations[entity_id];
  out_handle =
      EntityHandle{static_cast<uint32_t>(type_id), entity_id, generation};
  return out_handle.isValid();
}

static bool setRenderable(GameState &state, RenderableEntityContainer *container,
                          const EntityHandle &handle, int16_t texture_id,
                          int16_t width, int16_t height, uint8_t z_index) {
  uint32_t slot = INVALID_SLOT;
  if (!resolveSlot(state.engine, handle, slot)) {
    return false;
  }
  if (!container || slot >= static_cast<uint32_t>(container->count)) {
    return false;
  }

  container->texture_ids[slot] = texture_id;
  container->widths[slot] = width;
  container->heights[slot] = height;
  container->z_indices[slot] = z_index;
  container->rotations[slot] = 0.0f;
  return true;
}

static EntityHandle spawnEntity(GameState &state, int type_id,
                                RenderableEntityContainer *container,
                                int16_t texture_id, int16_t width,
                                int16_t height, uint8_t z_index, float x,
                                float y) {
  EntityHandle handle = engine_create_entity(state.engine, type_id);
  if (!handle.isValid()) {
    return {};
  }

  if (!setRenderable(state, container, handle, texture_id, width, height,
                     z_index)) {
    engine_destroy_entity(state.engine, handle);
    return {};
  }

  engine_set_entity_position(state.engine, handle, x, y);
  return handle;
}

static void spawnEnemies(GameState &state) {
  const float center_x = static_cast<float>(kWorldW) * 0.5f;
  const float center_y = static_cast<float>(kWorldH) * 0.5f;

  for (int i = 0; i < kTotalEnemyPlanets; ++i) {
    const bool as_dynamic = (i % 2) == 0; // 50% dynamic, 50% hybrid
    int type_id = as_dynamic ? state.dynamic_type : state.hybrid_type;
    auto *container =
        as_dynamic ? static_cast<RenderableEntityContainer *>(state.dynamic_container)
                   : static_cast<RenderableEntityContainer *>(state.hybrid_container);
    const int16_t tex = as_dynamic ? state.tex_dynamic : state.tex_hybrid;

    float x = randomRange(state.rng, 0.0f, static_cast<float>(kWorldW - kEnemySize));
    float y = randomRange(state.rng, 0.0f, static_cast<float>(kWorldH - kEnemySize));

    const float dx = x - center_x;
    const float dy = y - center_y;
    if ((dx * dx + dy * dy) < (1400.0f * 1400.0f)) {
      x = std::clamp(x + 2200.0f, 0.0f, static_cast<float>(kWorldW - kEnemySize));
    }

    spawnEntity(state, type_id, container, tex, kEnemySize, kEnemySize, 24, x, y);
  }
}

static void spawnStaticPlanets(GameState &state) {
  for (int i = 0; i < kStaticPlanets; ++i) {
    float x = randomRange(state.rng, 0.0f, static_cast<float>(kWorldW - kStaticSize));
    float y = randomRange(state.rng, 0.0f, static_cast<float>(kWorldH - kStaticSize));
    spawnEntity(state, state.static_type, state.static_container, state.tex_static,
                kStaticSize, kStaticSize, 5, x, y);
  }
}

static void cleanupInvalidBullets(GameState &state) {
  size_t i = 0;
  while (i < state.bullet_handles.size()) {
    if (!engine_is_handle_valid(state.engine, state.bullet_handles[i])) {
      state.bullet_handles[i] = state.bullet_handles.back();
      state.bullet_handles.pop_back();
      continue;
    }
    ++i;
  }
}

static void shootIfNeeded(GameState &state, float dt) {
  state.shoot_cooldown = std::max(0.0f, state.shoot_cooldown - dt);
  const bool wants_shoot =
      state.mouse_pressed || SDL_GetKeyboardState(nullptr)[SDL_SCANCODE_SPACE];
  if (!wants_shoot || state.shoot_cooldown > 0.0f) {
    return;
  }

  float mouse_x = 0.0f;
  float mouse_y = 0.0f;
  SDL_GetMouseState(&mouse_x, &mouse_y);

  const float cam_left = state.engine->camera.x - (state.engine->camera.width * 0.5f);
  const float cam_top = state.engine->camera.y - (state.engine->camera.height * 0.5f);
  const float world_mouse_x = cam_left + mouse_x;
  const float world_mouse_y = cam_top + mouse_y;

  const float src_x = state.player_x + (kPlayerSize * 0.5f) - (kBulletSize * 0.5f);
  const float src_y = state.player_y + (kPlayerSize * 0.5f) - (kBulletSize * 0.5f);

  float dx = world_mouse_x - (src_x + kBulletSize * 0.5f);
  float dy = world_mouse_y - (src_y + kBulletSize * 0.5f);
  const float len2 = dx * dx + dy * dy;
  if (len2 < 1.0f) {
    dx = 1.0f;
    dy = 0.0f;
  } else {
    const float inv_len = 1.0f / std::sqrt(len2);
    dx *= inv_len;
    dy *= inv_len;
  }

  EntityHandle bullet =
      spawnEntity(state, state.bullet_type, state.bullet_container, state.tex_bullet,
                  kBulletSize, kBulletSize, 40, src_x, src_y);
  if (bullet.isValid()) {
    uint32_t slot = INVALID_SLOT;
    if (resolveSlot(state.engine, bullet, slot) &&
        slot < static_cast<uint32_t>(state.bullet_container->count)) {
      state.bullet_container->vx[slot] = dx * kBulletSpeed;
      state.bullet_container->vy[slot] = dy * kBulletSpeed;
      state.bullet_container->life[slot] = kBulletLifetimeSec;
      state.bullet_container->rotations[slot] = std::atan2(dy, dx);
      state.bullet_handles.push_back(bullet);
    } else {
      engine_destroy_entity(state.engine, bullet);
    }
  }

  state.shoot_cooldown = kShootCooldownSec;
}

static void checkPlayerEnemyCollisions(GameState &state) {
  const float player_center_x = state.player_x + (kPlayerSize * 0.5f);
  const float player_center_y = state.player_y + (kPlayerSize * 0.5f);
  const float player_radius = kPlayerSize * 0.42f;
  const float enemy_radius = kEnemySize * 0.5f;

  std::vector<EntityHandle> to_remove;
  auto &nearby = state.engine->grid.queryCircle(
      player_center_x, player_center_y, player_radius + enemy_radius + 8.0f);

  for (const EntityRef &ref : nearby) {
    if (ref.type != static_cast<uint32_t>(state.dynamic_type) &&
        ref.type != static_cast<uint32_t>(state.hybrid_type)) {
      continue;
    }

    auto *container = static_cast<RenderableEntityContainer *>(
        state.engine->entityManager.containers[ref.type].get());
    if (!container || ref.index >= static_cast<uint32_t>(container->count)) {
      continue;
    }

    const float ex = container->x_positions[ref.index] + (container->widths[ref.index] * 0.5f);
    const float ey = container->y_positions[ref.index] + (container->heights[ref.index] * 0.5f);
    const float dx = player_center_x - ex;
    const float dy = player_center_y - ey;
    const float r = player_radius + enemy_radius;
    if ((dx * dx + dy * dy) > (r * r)) {
      continue;
    }

    EntityHandle enemy{};
    if (makeHandleFromSlot(state.engine, static_cast<int>(ref.type), ref.index,
                           enemy)) {
      to_remove.push_back(enemy);
    }
  }

  for (const EntityHandle &enemy : to_remove) {
    if (engine_is_handle_valid(state.engine, enemy)) {
      engine_destroy_entity(state.engine, enemy);
      state.hits += 1;
    }
  }
}

static void checkBulletEnemyCollisions(GameState &state) {
  std::vector<EntityHandle> bullets_to_remove;
  std::vector<EntityHandle> enemies_to_remove;
  std::unordered_set<uint64_t> enemy_unique;

  size_t i = 0;
  while (i < state.bullet_handles.size()) {
    const EntityHandle bullet = state.bullet_handles[i];
    uint32_t bullet_slot = INVALID_SLOT;
    if (!resolveSlot(state.engine, bullet, bullet_slot) ||
        bullet_slot >= static_cast<uint32_t>(state.bullet_container->count)) {
      state.bullet_handles[i] = state.bullet_handles.back();
      state.bullet_handles.pop_back();
      continue;
    }

    const float bx =
        state.bullet_container->x_positions[bullet_slot] + (kBulletSize * 0.5f);
    const float by =
        state.bullet_container->y_positions[bullet_slot] + (kBulletSize * 0.5f);
    const float bullet_radius = kBulletSize * 0.5f;

    bool hit_any = false;
    auto &nearby = state.engine->grid.queryCircle(bx, by, bullet_radius + kEnemySize);
    for (const EntityRef &ref : nearby) {
      if (ref.type != static_cast<uint32_t>(state.dynamic_type) &&
          ref.type != static_cast<uint32_t>(state.hybrid_type)) {
        continue;
      }

      auto *container = static_cast<RenderableEntityContainer *>(
          state.engine->entityManager.containers[ref.type].get());
      if (!container || ref.index >= static_cast<uint32_t>(container->count)) {
        continue;
      }

      const float ex = container->x_positions[ref.index] + (container->widths[ref.index] * 0.5f);
      const float ey = container->y_positions[ref.index] + (container->heights[ref.index] * 0.5f);
      const float dx = bx - ex;
      const float dy = by - ey;
      const float rr = bullet_radius + (container->widths[ref.index] * 0.5f);
      if ((dx * dx + dy * dy) > (rr * rr)) {
        continue;
      }

      EntityHandle enemy{};
      if (makeHandleFromSlot(state.engine, static_cast<int>(ref.type), ref.index,
                             enemy)) {
        const uint64_t enemy_key =
            (static_cast<uint64_t>(enemy.type) << 48) ^
            (static_cast<uint64_t>(enemy.entity) << 16) ^
            static_cast<uint64_t>(enemy.generation);
        if (enemy_unique.insert(enemy_key).second) {
          enemies_to_remove.push_back(enemy);
        }
      }

      bullets_to_remove.push_back(bullet);
      hit_any = true;
      break;
    }

    if (hit_any) {
      state.bullet_handles[i] = state.bullet_handles.back();
      state.bullet_handles.pop_back();
      continue;
    }

    ++i;
  }

  for (const EntityHandle &enemy : enemies_to_remove) {
    if (engine_is_handle_valid(state.engine, enemy)) {
      engine_destroy_entity(state.engine, enemy);
      state.kills += 1;
    }
  }

  for (const EntityHandle &bullet : bullets_to_remove) {
    if (engine_is_handle_valid(state.engine, bullet)) {
      engine_destroy_entity(state.engine, bullet);
    }
  }
}

static void drawTopBar(Engine *engine) {
  SDL_SetRenderDrawBlendMode(engine->renderer, SDL_BLENDMODE_BLEND);
  SDL_SetRenderDrawColor(engine->renderer, 8, 10, 16, 180);
  SDL_FRect r = {10.0f, 10.0f, 560.0f, 36.0f};
  SDL_RenderFillRect(engine->renderer, &r);
  SDL_SetRenderDrawBlendMode(engine->renderer, SDL_BLENDMODE_NONE);
}

static void updateWindowTitle(GameState &state) {
  char title[512];
  std::snprintf(
      title, sizeof(title),
      "Planet OptionB (50%% Dynamic + 50%% Hybrid + Static) | FPS %.1f | Dyn %d | Hyb %d active %d | Sta %d | Bullets %d | Hits %d | Kills %d",
      state.engine->fps, state.dynamic_container->count,
      state.hybrid_container->count, state.hybrid_container->last_active_count,
      state.static_container->count, state.bullet_container->count, state.hits,
      state.kills);
  SDL_SetWindowTitle(state.engine->window, title);
}

} // namespace

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  Engine *engine = engine_create(kWindowW, kWindowH, kWorldW, kWorldH, 64);
  if (!engine) {
    SDL_Log("Failed to create engine");
    return 1;
  }

  GameState state;
  state.engine = engine;

  SDL_Surface *player_s = makeColorSurface(kPlayerSize, kPlayerSize, 72, 212, 255);
  SDL_Surface *dyn_s = makeColorSurface(kEnemySize, kEnemySize, 244, 82, 112);
  SDL_Surface *hyb_s = makeColorSurface(kEnemySize, kEnemySize, 80, 230, 185);
  SDL_Surface *sta_s = makeColorSurface(kStaticSize, kStaticSize, 114, 122, 160);
  SDL_Surface *bul_s = makeColorSurface(kBulletSize, kBulletSize, 255, 240, 132);

  if (!player_s || !dyn_s || !hyb_s || !sta_s || !bul_s) {
    SDL_Log("Surface creation failed");
    SDL_DestroySurface(player_s);
    SDL_DestroySurface(dyn_s);
    SDL_DestroySurface(hyb_s);
    SDL_DestroySurface(sta_s);
    SDL_DestroySurface(bul_s);
    engine_destroy(engine);
    return 1;
  }

  state.tex_player =
      static_cast<int16_t>(engine_register_texture(engine, player_s, 0, 0, 0, 0));
  state.tex_dynamic =
      static_cast<int16_t>(engine_register_texture(engine, dyn_s, 64, 0, 0, 0));
  state.tex_hybrid =
      static_cast<int16_t>(engine_register_texture(engine, hyb_s, 128, 0, 0, 0));
  state.tex_static =
      static_cast<int16_t>(engine_register_texture(engine, sta_s, 192, 0, 0, 0));
  state.tex_bullet =
      static_cast<int16_t>(engine_register_texture(engine, bul_s, 256, 0, 0, 0));

  SDL_DestroySurface(player_s);
  SDL_DestroySurface(dyn_s);
  SDL_DestroySurface(hyb_s);
  SDL_DestroySurface(sta_s);
  SDL_DestroySurface(bul_s);

  state.player_container = new PlayerContainer(8);
  state.dynamic_container =
      new DynamicEnemyContainer(engine, (kTotalEnemyPlanets / 2) + 1024);
  state.hybrid_container =
      new HybridEnemyContainer(engine, (kTotalEnemyPlanets / 2) + 1024);
  state.static_container = new StaticPlanetContainer(kStaticPlanets + 1024);
  state.bullet_container = new BulletContainer(engine, 8192);

  state.player_type = engine_register_dynamic_type(engine, state.player_container);
  state.dynamic_type = engine_register_dynamic_type(engine, state.dynamic_container);
  state.hybrid_type = engine_register_hybrid_type(engine, state.hybrid_container);
  state.static_type = engine_register_static_type(engine, state.static_container);
  state.bullet_type = engine_register_dynamic_type(engine, state.bullet_container);

  if (state.player_type < 0 || state.dynamic_type < 0 || state.hybrid_type < 0 ||
      state.static_type < 0 || state.bullet_type < 0) {
    SDL_Log("Failed to register containers");
    engine_destroy(engine);
    return 1;
  }

  state.player =
      spawnEntity(state, state.player_type, state.player_container, state.tex_player,
                  kPlayerSize, kPlayerSize, 100, state.player_x, state.player_y);
  if (!state.player.isValid()) {
    SDL_Log("Failed to create player");
    engine_destroy(engine);
    return 1;
  }

  spawnEnemies(state);
  spawnStaticPlanets(state);
  state.bullet_handles.reserve(10000);

  engine->camera.x = state.player_x + (kPlayerSize * 0.5f);
  engine->camera.y = state.player_y + (kPlayerSize * 0.5f);
  updateWindowTitle(state);

  SDL_Log("[PlanetOptionB] Started.");
  SDL_Log("[PlanetOptionB] Enemy planets: total=%d dynamic=%d hybrid=%d static=%d",
          kTotalEnemyPlanets, kTotalEnemyPlanets / 2, kTotalEnemyPlanets / 2,
          kStaticPlanets);
  SDL_Log("[PlanetOptionB] Controls: WASD/arrows move, mouse-left or Space shoot, Esc quit.");

  bool running = true;
  Uint64 last_tick = SDL_GetTicks();

  while (running) {
    const Uint64 now = SDL_GetTicks();
    float dt = static_cast<float>(now - last_tick) / 1000.0f;
    last_tick = now;
    if (dt <= 0.0f) {
      dt = 0.0001f;
    }
    if (dt > 0.05f) {
      dt = 0.05f;
    }

    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
      if (ev.type == SDL_EVENT_QUIT) {
        running = false;
      } else if (ev.type == SDL_EVENT_KEY_DOWN) {
        if (ev.key.key == SDLK_ESCAPE) {
          running = false;
        }
      } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
        if (ev.button.button == SDL_BUTTON_LEFT) {
          state.mouse_pressed = true;
        }
      } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_UP) {
        if (ev.button.button == SDL_BUTTON_LEFT) {
          state.mouse_pressed = false;
        }
      }
    }

    const bool *keys = SDL_GetKeyboardState(nullptr);
    float dx = 0.0f;
    float dy = 0.0f;
    if (keys[SDL_SCANCODE_W] || keys[SDL_SCANCODE_UP]) {
      dy -= 1.0f;
    }
    if (keys[SDL_SCANCODE_S] || keys[SDL_SCANCODE_DOWN]) {
      dy += 1.0f;
    }
    if (keys[SDL_SCANCODE_A] || keys[SDL_SCANCODE_LEFT]) {
      dx -= 1.0f;
    }
    if (keys[SDL_SCANCODE_D] || keys[SDL_SCANCODE_RIGHT]) {
      dx += 1.0f;
    }

    if (dx != 0.0f || dy != 0.0f) {
      const float len = std::sqrt(dx * dx + dy * dy);
      dx /= len;
      dy /= len;
      float speed = kPlayerSpeed;
      if (keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT]) {
        speed *= 2.0f;
      }
      state.player_x += dx * speed * dt;
      state.player_y += dy * speed * dt;
      state.player_x =
          std::clamp(state.player_x, 0.0f, static_cast<float>(kWorldW - kPlayerSize));
      state.player_y =
          std::clamp(state.player_y, 0.0f, static_cast<float>(kWorldH - kPlayerSize));
    }

    engine_set_entity_position(engine, state.player, state.player_x, state.player_y);
    state.dynamic_container->target_x = state.player_x;
    state.dynamic_container->target_y = state.player_y;
    state.hybrid_container->target_x = state.player_x;
    state.hybrid_container->target_y = state.player_y;

    shootIfNeeded(state, dt);

    engine->camera.x = state.player_x + (kPlayerSize * 0.5f);
    engine->camera.y = state.player_y + (kPlayerSize * 0.5f);

    engine_update(engine);
    cleanupInvalidBullets(state);
    checkPlayerEnemyCollisions(state);
    checkBulletEnemyCollisions(state);

    engine_render_scene(engine);
    drawTopBar(engine);
    engine_present(engine);

    state.console_frames += 1;
    state.console_timer += dt;
    state.title_timer += dt;

    if (state.console_timer >= 1.0f) {
      const float raw_fps = static_cast<float>(state.console_frames) / state.console_timer;
      SDL_Log(
          "[PlanetOptionB] FPS(raw)=%.1f FPS(smooth)=%.1f | dyn=%d upd=%d | hyb=%d act=%d upd=%d | static=%d | bullets=%d | hits=%d kills=%d",
          raw_fps, engine->fps, state.dynamic_container->count,
          state.dynamic_container->last_updated, state.hybrid_container->count,
          state.hybrid_container->last_active_count, state.hybrid_container->last_updated,
          state.static_container->count, state.bullet_container->count, state.hits,
          state.kills);
      state.console_timer = 0.0f;
      state.console_frames = 0;
    }

    if (state.title_timer >= 0.2f) {
      state.title_timer = 0.0f;
      updateWindowTitle(state);
    }
  }

  engine_destroy(engine);
  return 0;
}
