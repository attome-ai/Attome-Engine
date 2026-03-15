#include "TowerSwarmGame.h"

#include "InputManager.h"
#include "characters/CharacterDefinitions.h"
#include "entities/BaseEntity.h"
#include "entities/CreatureContainer.h"
#include "entities/EnemyContainer.h"
#include "entities/PickupContainer.h"
#include "entities/ProjectileContainer.h"
#include "entities/TileContainer.h"
#include "levels/SaveState.h"
#include "shop/RelicSystem.h"
#include "systems/Evolution.h"

#include "ATMEngine.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace tower_swarm {
namespace {

constexpr EntityHandle kDeployedRosterDead = 0xFFFFFFFEu;

SDL_Surface *create_colored_surface(int width, int height, Uint8 r, Uint8 g,
                                    Uint8 b) {
  SDL_Surface *surface =
      SDL_CreateSurface(width, height, SDL_PIXELFORMAT_RGBA8888);
  if (!surface) {
    return nullptr;
  }
  SDL_FillSurfaceRect(
      surface, nullptr,
      SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format), nullptr, r, g, b,
                  255));
  return surface;
}

struct PixelRgba final {
  std::uint8_t r = 0;
  std::uint8_t g = 0;
  std::uint8_t b = 0;
  std::uint8_t a = 0;
};

inline PixelRgba with_alpha(PixelRgba c, std::uint8_t a) {
  c.a = a;
  return c;
}

inline PixelRgba clamp_rgba(int r, int g, int b, int a) {
  PixelRgba out{};
  out.r = static_cast<std::uint8_t>(std::clamp(r, 0, 255));
  out.g = static_cast<std::uint8_t>(std::clamp(g, 0, 255));
  out.b = static_cast<std::uint8_t>(std::clamp(b, 0, 255));
  out.a = static_cast<std::uint8_t>(std::clamp(a, 0, 255));
  return out;
}

inline PixelRgba lerp_rgba(PixelRgba a, PixelRgba b, float t) {
  const float tt = std::clamp(t, 0.0f, 1.0f);
  const float it = 1.0f - tt;
  return clamp_rgba(static_cast<int>(std::lround(a.r * it + b.r * tt)),
                    static_cast<int>(std::lround(a.g * it + b.g * tt)),
                    static_cast<int>(std::lround(a.b * it + b.b * tt)),
                    static_cast<int>(std::lround(a.a * it + b.a * tt)));
}

inline void blend_over(PixelRgba &dst, PixelRgba src) {
  const int sa = static_cast<int>(src.a);
  if (sa <= 0) {
    return;
  }
  if (sa >= 255) {
    dst = src;
    return;
  }

  const int da = static_cast<int>(dst.a);
  const int inv = 255 - sa;
  const int outA = sa + (da * inv + 127) / 255;
  if (outA <= 0) {
    dst = PixelRgba{};
    return;
  }

  const int outR =
      (static_cast<int>(src.r) * sa +
       static_cast<int>(dst.r) * da * inv / 255 + 127) / outA;
  const int outG =
      (static_cast<int>(src.g) * sa +
       static_cast<int>(dst.g) * da * inv / 255 + 127) / outA;
  const int outB =
      (static_cast<int>(src.b) * sa +
       static_cast<int>(dst.b) * da * inv / 255 + 127) / outA;

  dst.r = static_cast<std::uint8_t>(std::clamp(outR, 0, 255));
  dst.g = static_cast<std::uint8_t>(std::clamp(outG, 0, 255));
  dst.b = static_cast<std::uint8_t>(std::clamp(outB, 0, 255));
  dst.a = static_cast<std::uint8_t>(std::clamp(outA, 0, 255));
}

inline void put_px(std::vector<PixelRgba> &px, int w, int h, int x, int y,
                   PixelRgba c) {
  if (x < 0 || y < 0 || x >= w || y >= h) {
    return;
  }
  blend_over(px[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                 static_cast<std::size_t>(x)],
             c);
}

inline PixelRgba rgba_from(Rgba8 c) {
  return PixelRgba{c.r, c.g, c.b, c.a};
}

Rgba8 character_color(CharacterId id);

PixelRgba tier_band_tint(int band) {
  switch (std::clamp(band, 0, evolution::kVisualBandCount - 1)) {
  case 0:
    return PixelRgba{255, 255, 255, 0};
  case 1: // green tint
    return PixelRgba{60, 255, 120, 120};
  case 2: // blue glow
    return PixelRgba{80, 170, 255, 140};
  case 3: // purple aura
    return PixelRgba{180, 80, 255, 150};
  case 4: // gold shimmer
    return PixelRgba{255, 220, 80, 160};
  case 5: // red corona
    return PixelRgba{255, 80, 80, 170};
  case 6: // void-black + white halo (handled specially)
    return PixelRgba{10, 10, 12, 220};
  }
  return PixelRgba{255, 255, 255, 0};
}

PixelRgba apply_tier_style(PixelRgba base, int band) {
  if (band <= 0) {
    return base;
  }
  if (band >= 6) {
    // Void: crush into near-black but keep subtle hue.
    const int lum =
        static_cast<int>(base.r) * 3 / 10 + static_cast<int>(base.g) * 6 / 10 +
        static_cast<int>(base.b) * 1 / 10;
    const int crushed = std::clamp(lum / 10 + 4, 0, 40);
    return PixelRgba{static_cast<std::uint8_t>(crushed),
                     static_cast<std::uint8_t>(crushed),
                     static_cast<std::uint8_t>(std::clamp(crushed + 2, 0, 50)),
                     base.a};
  }
  const PixelRgba tint = tier_band_tint(band);
  const float t = std::clamp(static_cast<float>(tint.a) / 255.0f, 0.0f, 1.0f);
  PixelRgba mixed = lerp_rgba(base, with_alpha(tint, base.a), t);
  mixed.a = base.a;
  return mixed;
}

bool shape_mask(CharacterId id, int w, int h, int x, int y) {
  const float cx = static_cast<float>(w - 1) * 0.5f;
  const float cy = static_cast<float>(h - 1) * 0.5f;
  const float fx = static_cast<float>(x) - cx;
  const float fy = static_cast<float>(y) - cy;
  const float r = static_cast<float>(std::min(w, h)) * 0.36f;

  switch (id) {
  case CharacterId::Brix: {
    const float ar = std::abs(fx);
    const float br = std::abs(fy);
    return (ar <= r && br <= r) &&
           !((fx > r * 0.2f && fy < -r * 0.2f) ||
             (fx < -r * 0.2f && fy > r * 0.2f));
  }
  case CharacterId::Flara: {
    // Teardrop / flame.
    const float rr = r * 1.05f;
    const float d2 = fx * fx + (fy + r * 0.15f) * (fy + r * 0.15f);
    if (d2 <= rr * rr) {
      return true;
    }
    // Pointy top.
    const float top = -r * 1.15f;
    if (fy < 0.0f && fy >= top) {
      const float t = (fy - top) / (0.0f - top);
      const float half = (1.0f - t) * r * 0.55f;
      return std::abs(fx) <= half;
    }
    return false;
  }
  case CharacterId::Mossling: {
    // Leaf-ish oval with a notch.
    const float rx = r * 0.95f;
    const float ry = r * 1.15f;
    const float v =
        (fx * fx) / (rx * rx) + ((fy + r * 0.10f) * (fy + r * 0.10f)) / (ry * ry);
    if (v > 1.0f) {
      return false;
    }
    return !(fy > r * 0.25f && std::abs(fx) < r * 0.15f);
  }
  case CharacterId::Glitch: {
    // Hex-ish (diamond + cut corners).
    const float ar = std::abs(fx);
    const float br = std::abs(fy);
    if (ar + br > r * 1.25f) {
      return false;
    }
    return !(ar > r * 0.9f && br > r * 0.35f);
  }
  case CharacterId::Ironjaw: {
    // Jaw block with teeth.
    const float ar = std::abs(fx);
    const float br = std::abs(fy);
    if (ar > r * 1.10f || br > r * 0.85f) {
      return false;
    }
    if (fy > r * 0.35f && (static_cast<int>(x / 4) % 2) == 0) {
      return false;
    }
    return true;
  }
  case CharacterId::Wraith: {
    // Slender diamond/arrow.
    const float ar = std::abs(fx);
    const float br = std::abs(fy);
    return ar * 0.75f + br <= r * 1.1f;
  }
  case CharacterId::Crystalis: {
    // Diamond crystal.
    const float ar = std::abs(fx);
    const float br = std::abs(fy);
    return ar + br <= r * 1.25f;
  }
  case CharacterId::Vex: {
    // Star-ish: circle with spikes.
    const float d = std::sqrt(fx * fx + fy * fy);
    if (d > r * 1.15f) {
      return false;
    }
    const float ang = std::atan2(fy, fx);
    const float spike = 0.22f * std::sin(5.0f * ang);
    return d <= r * (1.0f + spike);
  }
  case CharacterId::Orin: {
    // Shield-like circle + top cap.
    const float rr = r * 1.05f;
    if (fx * fx + fy * fy <= rr * rr) {
      return true;
    }
    return (fy < -r * 0.55f && std::abs(fx) <= r * 0.6f);
  }
  case CharacterId::NullSeed: {
    // Void orb.
    const float rr = r * 1.10f;
    return fx * fx + fy * fy <= rr * rr;
  }
  case CharacterId::Count:
    break;
  }

  return false;
}

void draw_character_sprite(std::vector<PixelRgba> &px, int w, int h, CharacterId id,
                           int band) {
  const PixelRgba base = apply_tier_style(rgba_from(character_color(id)), band);
  const PixelRgba outline = PixelRgba{10, 10, 10, 220};

  // Core fill + outline.
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (!shape_mask(id, w, h, x, y)) {
        continue;
      }
      put_px(px, w, h, x, y, base);

      bool edge = false;
      for (int oy = -1; oy <= 1 && !edge; oy++) {
        for (int ox = -1; ox <= 1; ox++) {
          if (ox == 0 && oy == 0) {
            continue;
          }
          if (!shape_mask(id, w, h, x + ox, y + oy)) {
            edge = true;
            break;
          }
        }
      }
      if (edge) {
        put_px(px, w, h, x, y, outline);
      }
    }
  }

  // Band effects (simple baked-in accents).
  if (band == 2 || band == 3 || band == 5 || band >= 6) {
    PixelRgba glow = tier_band_tint(band);
    glow.a = (band >= 6) ? 190 : 110;

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        if (shape_mask(id, w, h, x, y)) {
          continue;
        }
        bool near = false;
        for (int oy = -2; oy <= 2 && !near; oy++) {
          for (int ox = -2; ox <= 2; ox++) {
            if (shape_mask(id, w, h, x + ox, y + oy)) {
              near = true;
              break;
            }
          }
        }
        if (near) {
          put_px(px, w, h, x, y, glow);
        }
      }
    }
  }

  if (band == 4) {
    // Gold shimmer: sprinkle highlights.
    PixelRgba hi = PixelRgba{255, 255, 255, 90};
    for (int i = 0; i < 40; i++) {
      const int x = (i * 17) % w;
      const int y = (i * 29) % h;
      if (shape_mask(id, w, h, x, y)) {
        put_px(px, w, h, x, y, hi);
      }
    }
  }

  if (band >= 6) {
    // Halo ring.
    const float cx = static_cast<float>(w - 1) * 0.5f;
    const float cy = static_cast<float>(h - 1) * 0.5f;
    const float rr = static_cast<float>(std::min(w, h)) * 0.42f;
    const float rr2 = rr * rr;
    const float rrIn = (rr - 2.2f);
    const float rrIn2 = rrIn * rrIn;
    PixelRgba halo = PixelRgba{250, 250, 250, 120};
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        const float fx = static_cast<float>(x) - cx;
        const float fy = static_cast<float>(y) - cy;
        const float d2 = fx * fx + fy * fy;
        if (d2 <= rr2 && d2 >= rrIn2) {
          put_px(px, w, h, x, y, halo);
        }
      }
    }
  }
}

int register_creature_texture(Engine *engine, CharacterId id, int band) {
  if (!engine || !engine->renderer) {
    return -1;
  }

  const int size_px = std::max(8, kCreatureBaseSizePx);
  std::vector<PixelRgba> px(static_cast<std::size_t>(size_px) *
                                static_cast<std::size_t>(size_px),
                            PixelRgba{});
  draw_character_sprite(px, size_px, size_px, id, band);

  SDL_Surface *surface =
      SDL_CreateSurface(size_px, size_px, SDL_PIXELFORMAT_RGBA8888);
  if (!surface) {
    SDL_Log("TowerSwarm: SDL_CreateSurface failed: %s", SDL_GetError());
    return -1;
  }

  if (!SDL_LockSurface(surface)) {
    SDL_Log("TowerSwarm: SDL_LockSurface failed: %s", SDL_GetError());
    SDL_DestroySurface(surface);
    return -1;
  }

  Uint32 *dst = static_cast<Uint32 *>(surface->pixels);
  const int pitch_px = surface->pitch / 4;
  const SDL_PixelFormatDetails *fmt = SDL_GetPixelFormatDetails(surface->format);
  for (int y = 0; y < size_px; y++) {
    Uint32 *row = dst + y * pitch_px;
    for (int x = 0; x < size_px; x++) {
      const PixelRgba c =
          px[static_cast<std::size_t>(y) * static_cast<std::size_t>(size_px) +
             static_cast<std::size_t>(x)];
      row[x] = SDL_MapRGBA(fmt, nullptr, c.r, c.g, c.b, c.a);
    }
  }

  SDL_UnlockSurface(surface);

  const int texture_id =
      engine_register_texture(engine, surface, 0, 0, size_px, size_px);
  SDL_DestroySurface(surface);
  return texture_id;
}

Rgba8 character_color(CharacterId id) {
  switch (id) {
  case CharacterId::Brix:
    return kCreatureColorBrix;
  case CharacterId::Flara:
    return kCreatureColorFlara;
  case CharacterId::Mossling:
    return kCreatureColorMossling;
  case CharacterId::Glitch:
    return kCreatureColorGlitch;
  case CharacterId::Ironjaw:
    return kCreatureColorIronjaw;
  case CharacterId::Wraith:
    return kCreatureColorWraith;
  case CharacterId::Crystalis:
    return kCreatureColorCrystalis;
  case CharacterId::Vex:
    return kCreatureColorVex;
  case CharacterId::Orin:
    return kCreatureColorOrin;
  case CharacterId::NullSeed:
    return kCreatureColorNull;
  case CharacterId::Count:
    break;
  }
  return kCreatureColorBrix;
}

Rgba8 enemy_color(EnemyType t) {
  switch (t) {
  case EnemyType::Grub:
    return kEnemyColorGrub;
  case EnemyType::Hulk:
    return kEnemyColorHulk;
  case EnemyType::Scuttle:
    return kEnemyColorScuttle;
  case EnemyType::Driftwing:
    return kEnemyColorDriftwing;
  case EnemyType::Divide:
    return kEnemyColorDivide;
  case EnemyType::Vanguard:
    return kEnemyColorVanguard;
  case EnemyType::Mender:
    return kEnemyColorMender;
  case EnemyType::SiegeLord:
    return kEnemyColorBoss;
  case EnemyType::Count:
    break;
  }
  return kEnemyColorGrub;
}

int register_solid_texture(Engine *engine, int size_px, Rgba8 c) {
  if (!engine || !engine->renderer || size_px <= 0) {
    return -1;
  }
  SDL_Surface *surface =
      create_colored_surface(size_px, size_px, c.r, c.g, c.b);
  if (!surface) {
    return -1;
  }
  const int texture_id =
      engine_register_texture(engine, surface, 0, 0, size_px, size_px);
  SDL_DestroySurface(surface);
  return texture_id;
}

void set_color(SDL_Renderer *renderer, Rgba8 c) {
  SDL_SetRenderDrawColor(renderer, c.r, c.g, c.b, c.a);
}

bool point_in_rect(float x, float y, const SDL_FRect &r) {
  return x >= r.x && y >= r.y && x <= (r.x + r.w) && y <= (r.y + r.h);
}

float safe_zoom(const Engine *engine) {
  if (!engine || engine->camera.zoom <= kMinCameraZoomEpsilon) {
    return 1.0f;
  }
  return engine->camera.zoom;
}

void get_camera_world_rect(const Engine *engine, float &x1, float &y1,
                           float &x2, float &y2) {
  if (!engine) {
    x1 = y1 = x2 = y2 = 0.0f;
    return;
  }
  const float zoom = safe_zoom(engine);
  const float half_w = (engine->camera.width / zoom) * 0.5f;
  const float half_h = (engine->camera.height / zoom) * 0.5f;
  x1 = engine->camera.x - half_w;
  y1 = engine->camera.y - half_h;
  x2 = engine->camera.x + half_w;
  y2 = engine->camera.y + half_h;
}

Rgba8 biome_base_color(Biome biome) {
  switch (biome) {
  case Biome::VerdantFields:
    return kBiomeTileColorVerdantFields;
  case Biome::Ashlands:
    return kBiomeTileColorAshlands;
  case Biome::Frostmarsh:
    return kBiomeTileColorFrostmarsh;
  case Biome::Deepcore:
    return kBiomeTileColorDeepcore;
  case Biome::TheVoid:
    return kBiomeTileColorTheVoid;
  default:
    return kBiomeTileColorVerdantFields;
  }
}

std::uint32_t xorshift32(std::uint32_t &s) {
  std::uint32_t x = s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  s = x;
  return x;
}

Rarity roll_rarity(std::uint32_t &rng) {
  const std::uint32_t r = xorshift32(rng) % 100u;
  if (r < 60u) {
    return Rarity::Common;
  }
  if (r < 85u) {
    return Rarity::Rare;
  }
  if (r < 97u) {
    return Rarity::Epic;
  }
  return Rarity::Legendary;
}

int seed_cost_essence(Rarity rarity, int level_number) {
  const int lvl = std::max(1, level_number);
  switch (rarity) {
  case Rarity::Common:
    return inter_level_shop::kSeedCommonBaseCost +
           lvl * inter_level_shop::kSeedCommonPerLevelCost;
  case Rarity::Rare:
    return inter_level_shop::kSeedRareBaseCost +
           lvl * inter_level_shop::kSeedRarePerLevelCost;
  case Rarity::Epic:
    return inter_level_shop::kSeedEpicBaseCost +
           lvl * inter_level_shop::kSeedEpicPerLevelCost;
  case Rarity::Legendary:
    return inter_level_shop::kSeedLegendaryBaseCost +
           lvl * inter_level_shop::kSeedLegendaryPerLevelCost;
  }
  return inter_level_shop::kSeedCommonBaseCost +
         lvl * inter_level_shop::kSeedCommonPerLevelCost;
}

bool is_character_unlocked(const GameState &state, CharacterId cid) {
  return state.isCharacterUnlocked(cid);
}

int creature_size_px_for_tier(int tier) {
  return evolution::creatureSizePxForTier(tier);
}

float creature_pick_radius_px() {
  return static_cast<float>(
      evolution::creatureSizePxForTier(20, evolution::kEvolutionPulseScale));
}

void creature_top_left_for_cell(int col, int row, int tier, float &out_x,
                                float &out_y) {
  const int size_px = creature_size_px_for_tier(tier);

  const float cell_cx =
      static_cast<float>(col * kTileSizePx) + static_cast<float>(kTileSizePx) * 0.5f;
  const float cell_cy =
      static_cast<float>(row * kTileSizePx) + static_cast<float>(kTileSizePx) * 0.5f;

  const float half = static_cast<float>(size_px) * 0.5f;
  float x = cell_cx - half;
  float y = cell_cy - half;

  const float max_x =
      std::max(0.0f, static_cast<float>(kWorldWidthPx - size_px));
  const float max_y =
      std::max(0.0f, static_cast<float>(kWorldHeightPx - size_px));
  out_x = std::clamp(x, 0.0f, max_x);
  out_y = std::clamp(y, 0.0f, max_y);
}

} // namespace

TowerSwarmGame::TowerSwarmGame(Engine *engine) : engine_(engine) {
  biome_tile_texture_.fill(-1);
  for (auto &bands : creature_texture_) {
    bands.fill(-1);
  }
  projectile_texture_.fill(-1);
  enemy_texture_.fill(-1);
}

TowerSwarmGame::~TowerSwarmGame() = default;

int TowerSwarmGame::registerBiomeTileTexture(Biome biome) {
  if (!engine_ || !engine_->renderer) {
    return -1;
  }

  const Rgba8 c = biome_base_color(biome);
  SDL_Surface *surface =
      create_colored_surface(kTileSizePx, kTileSizePx, c.r, c.g, c.b);
  if (!surface) {
    SDL_Log("TowerSwarm: failed to create tile surface for biome %d",
            static_cast<int>(biome));
    return -1;
  }

  const int texture_id =
      engine_register_texture(engine_, surface, 0, 0, kTileSizePx, kTileSizePx);
  SDL_DestroySurface(surface);
  return texture_id;
}

void TowerSwarmGame::startLevel(std::int32_t level_number) {
  if (!engine_) {
    return;
  }

  auto destroy_all = [&](RenderableEntityContainer *container) {
    if (!container) {
      return;
    }
    std::vector<EntityHandle> ids;
    ids.reserve(static_cast<std::size_t>(std::max(0, container->count)));
    for (std::uint32_t slot = 0;
         slot < static_cast<std::uint32_t>(container->count); ++slot) {
      const EntityHandle id = container->getStableId(slot);
      if (id != INVALID_ID) {
        ids.push_back(id);
      }
    }
    for (const EntityHandle id : ids) {
      engine_destroy_entity(engine_, id, container->getTypeId());
    }
  };

  destroy_all(projectiles_);
  destroy_all(pickups_);
  destroy_all(enemies_);
  destroy_all(creatures_);

  std::fill(cell_occupant_.begin(), cell_occupant_.end(), INVALID_ID);
  deployed_roster_.assign(game_state_.roster.size(), INVALID_ID);
  if (game_state_.roster.empty()) {
    selected_roster_index_ = 0;
  } else {
    selected_roster_index_ =
        std::clamp(selected_roster_index_, 0,
                   static_cast<std::int32_t>(game_state_.roster.size()) - 1);
  }

  show_sell_confirm_ = false;
  pending_sell_creature_ = INVALID_ID;
  pending_sell_roster_index_ = -1;
  show_level_select_ = false;
  level_select_level_ = std::max(1, level_number);
  show_armory_ = false;
  armory_tab_ = ArmoryTab::Characters;
  show_armory_character_detail_ = false;
  show_armory_confirm_ = false;
  armory_confirm_kind_ = ArmoryConfirmKind::None;
  show_inter_level_ = false;
  inter_level_tab_ = InterLevelTab::Bazaar;
  inter_level_elapsed_sec_ = 0.0f;
  inter_level_rng_ = 0;
  bazaar_rerolled_ = false;
  show_bazaar_duplicate_confirm_ = false;
  pending_bazaar_offer_index_ = -1;
  repair_purchased_ = false;
  selected_creature_ = INVALID_ID;
  merge_drag_active_ = false;
  merge_drag_source_ = INVALID_ID;
  merge_cooldown_remaining_sec_ = 0.0f;
  auto_merge_idle_sec_ = 0.0f;
  merge_pairs_.clear();
  merge_anim_ = MergeAnim{};
  game_state_.floating_texts.clear();
  game_state_.screen_edge_glow_remaining_sec = 0.0f;
  game_state_.effect_zones.clear();

  ghost_valid_ = false;
  ghost_active_ = false;
  ghost_world_x_ = 0.0f;
  ghost_world_y_ = 0.0f;

  wave_buff_shop_.close();
  game_state_.clearWaveBuffs();
  RelicSystem::apply_all(game_state_);

  level_manager_.beginLevel(level_number);
  if (base_ && base_id_ != INVALID_ID) {
    const int max_hp = std::max(1, base_->getHpMax(base_id_));
    base_->resetHp(base_id_,
                   std::clamp<std::int32_t>(game_state_.base_hp, 0, max_hp));
    if (game_state_.next_level_base_hp_target !=
        std::numeric_limits<std::int32_t>::min()) {
      const int clamped_target = std::clamp<std::int32_t>(
          game_state_.next_level_base_hp_target, 0, max_hp);
      if (clamped_target > base_->getHp(base_id_)) {
        base_->resetHp(base_id_, clamped_target);
      }
    }
    game_state_.base_hp = base_->getHp(base_id_);
  }
  game_state_.next_level_base_hp_target =
      std::numeric_limits<std::int32_t>::min();

  const Biome biome = level_manager_.levelDef().biome;
  active_biome_ = biome;
  const int tile_tex = biome_tile_texture_[static_cast<std::size_t>(biome)];
  if (tiles_ && tile_tex >= 0) {
    for (std::uint32_t slot = 0;
         slot < static_cast<std::uint32_t>(tiles_->count); ++slot) {
      tiles_->texture_ids[slot] = static_cast<std::int16_t>(tile_tex);
    }
    engine_mark_static_dirty(engine_);
  }

  level_start_snapshot_ = SaveState::snapshotPersistent(game_state_);
  have_level_start_snapshot_ = true;
  last_level_state_ = level_manager_.state();
}

void TowerSwarmGame::applyWaveBuff(WaveBuffId id) {
  switch (id) {
  case WaveBuffId::Surge:
  case WaveBuffId::FrenziedBlood:
  case WaveBuffId::SlowTide:
  case WaveBuffId::Foresight:
  case WaveBuffId::EchoStrike:
  case WaveBuffId::IronSkin:
  case WaveBuffId::ApexHunter:
  case WaveBuffId::VoidPulse: {
    const auto &d = WaveBuffShop::def(id);
    game_state_.addOrRefreshWaveBuff(id, d.duration_waves);
    break;
  }
  case WaveBuffId::Fortify: {
    if (base_ && base_id_ != INVALID_ID) {
      base_->addMaxHp(base_id_, wave_shop::kFortifyBaseHpBonus, true);
    }
    break;
  }
  case WaveBuffId::Mend: {
    if (creatures_) {
      for (std::uint32_t slot = 0;
           slot < static_cast<std::uint32_t>(creatures_->count); ++slot) {
        if ((creatures_->flags[slot] &
             static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
          continue;
        }
        const float max_hp = std::max(0.0f, creatures_->hp_max[slot]);
        const float heal = max_hp * wave_shop::kMendHealFraction;
        creatures_->hp[slot] = std::clamp(creatures_->hp[slot] + heal, 0.0f, max_hp);
      }
    }
    break;
  }
  case WaveBuffId::EssenceCache: {
    const int before = std::max(0, game_state_.essence);
    const int gain = std::max(
        0, static_cast<int>(std::floor(static_cast<float>(before) *
                                       wave_shop::kEssenceCacheFraction)));
    game_state_.essence += gain;
    game_state_.essence_earned_this_level += gain;
    break;
  }
  case WaveBuffId::WildSeed: {
    if (!creatures_ || game_state_.roster.empty() || cell_occupant_.empty() ||
        grid_cols_ <= 0 || grid_rows_ <= 0) {
      break;
    }

    std::uint32_t rng =
        0x13579BDFu ^ static_cast<std::uint32_t>(level_manager_.levelDef().level_number * 2654435761u) ^
        static_cast<std::uint32_t>(level_manager_.waveIndex() + 1);
    auto next_u32 = [&](std::uint32_t &s) {
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
      return s;
    };

    const std::size_t choice =
        static_cast<std::size_t>(next_u32(rng) %
                                 static_cast<std::uint32_t>(game_state_.roster.size()));
    const CharacterId cid = game_state_.roster[choice].character;

    float base_cx = static_cast<float>(kWorldWidthPx) * 0.5f;
    float base_cy = static_cast<float>(kWorldHeightPx) * 0.5f;
    if (base_ && base_id_ != INVALID_ID) {
      const std::uint32_t bslot = base_->getSlot(base_id_);
      if (bslot != INVALID_ID && bslot < static_cast<std::uint32_t>(base_->count)) {
        base_cx = base_->x_positions[bslot] +
                  static_cast<float>(base_->widths[bslot]) * 0.5f;
        base_cy = base_->y_positions[bslot] +
                  static_cast<float>(base_->heights[bslot]) * 0.5f;
      }
    }

    const int base_col = std::clamp(static_cast<int>(base_cx / kTileSizePx), 0,
                                    std::max(0, grid_cols_ - 1));
    const int base_row = std::clamp(static_cast<int>(base_cy / kTileSizePx), 0,
                                    std::max(0, grid_rows_ - 1));

    const float avoid_r =
        kBaseRadiusPx + static_cast<float>(kCreatureBaseSizePx) * 1.25f;
    const float avoid_r2 = avoid_r * avoid_r;

    bool placed = false;
    for (int r = 0; !placed && r <= std::max(grid_cols_, grid_rows_); ++r) {
      const int min_c = std::max(0, base_col - r);
      const int max_c = std::min(std::max(0, grid_cols_ - 1), base_col + r);
      const int min_r = std::max(0, base_row - r);
      const int max_r = std::min(std::max(0, grid_rows_ - 1), base_row + r);

      for (int row = min_r; !placed && row <= max_r; ++row) {
        for (int col = min_c; !placed && col <= max_c; ++col) {
          if (r > 0 && col != min_c && col != max_c && row != min_r &&
              row != max_r) {
            continue;
          }

          const std::size_t idx =
              static_cast<std::size_t>(row * grid_cols_ + col);
          if (idx >= cell_occupant_.size()) {
            continue;
          }
          if (cell_occupant_[idx] != INVALID_ID) {
            continue;
          }

          const float cell_cx =
              static_cast<float>(col * kTileSizePx + kTileSizePx / 2);
          const float cell_cy =
              static_cast<float>(row * kTileSizePx + kTileSizePx / 2);
          const float dx = cell_cx - base_cx;
          const float dy = cell_cy - base_cy;
          if (dx * dx + dy * dy <= avoid_r2) {
            continue;
          }

          float px = 0.0f;
          float py = 0.0f;
          creature_top_left_for_cell(col, row, wave_shop::kWildSeedTier, px, py);

          const EntityHandle handle =
              creatures_->createCreature(px, py, cid, wave_shop::kWildSeedTier,
                                         0, -1);
          if (handle != INVALID_ID) {
            cell_occupant_[idx] = handle;
            placed = true;
          }
        }
      }
    }

    break;
  }
  case WaveBuffId::Count:
    break;
  }
}

void TowerSwarmGame::openInterLevel() {
  show_inter_level_ = true;
  inter_level_tab_ = InterLevelTab::Bazaar;
  bazaar_rerolled_ = false;
  repair_purchased_ = false;
  relic_pick_ = RelicId::None;
  inter_level_elapsed_sec_ = 0.0f;
  show_bazaar_duplicate_confirm_ = false;
  pending_bazaar_offer_index_ = -1;

  const std::uint32_t level_n = static_cast<std::uint32_t>(
      std::max<std::int32_t>(1, level_manager_.levelDef().level_number));
  inter_level_rng_ = 0x51EAD123u ^ (level_n * 2654435761u);

  if (game_state_.roster.empty()) {
    forge_selected_ = CharacterId::Brix;
  } else {
    forge_selected_ = game_state_.roster.front().character;
  }

  rerollBazaar();
}

void TowerSwarmGame::rerollBazaar() {
  const int level_n = std::max(1, level_manager_.levelDef().level_number);
  if (inter_level_rng_ == 0) {
    inter_level_rng_ = 0xC0DEF00Du ^ (static_cast<std::uint32_t>(level_n) * 1597334677u);
  }

  std::array<bool, static_cast<std::size_t>(CharacterId::Count)> used{};
  used.fill(false);

  auto build_pool = [&](Rarity rarity,
                         std::array<CharacterId, static_cast<std::size_t>(CharacterId::Count)> &pool,
                         std::size_t &pool_count) {
    pool_count = 0;
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(CharacterId::Count); ++i) {
      const CharacterId cid = static_cast<CharacterId>(i);
      if (cid == CharacterId::Count) {
        continue;
      }
      const CharacterDefinition &def = get_character_def(cid);
      if (def.rarity != rarity) {
        continue;
      }
      if (!is_character_unlocked(game_state_, cid)) {
        continue;
      }
      pool[pool_count] = cid;
      pool_count += 1;
    }
  };

  auto degrade = [&](Rarity r) {
    switch (r) {
    case Rarity::Legendary:
      return Rarity::Epic;
    case Rarity::Epic:
      return Rarity::Rare;
    case Rarity::Rare:
      return Rarity::Common;
    case Rarity::Common:
    default:
      return Rarity::Common;
    }
  };

  for (std::size_t i = 0; i < bazaar_offers_.size(); ++i) {
    const Rarity rolled = roll_rarity(inter_level_rng_);
    Rarity rarity = rolled;
    std::array<CharacterId, static_cast<std::size_t>(CharacterId::Count)> pool{};
    std::size_t pool_count = 0;
    for (int step = 0; step < 4; ++step) {
      build_pool(rarity, pool, pool_count);
      if (pool_count > 0) {
        break;
      }
      const Rarity next = degrade(rarity);
      if (next == rarity) {
        break;
      }
      rarity = next;
    }

    CharacterId cid = CharacterId::Brix;
    if (pool_count > 0) {
      cid = pool[static_cast<std::size_t>(xorshift32(inter_level_rng_) % pool_count)];
    }

    for (int tries = 0; tries < 12; ++tries) {
      const std::size_t idx = static_cast<std::size_t>(cid);
      if (idx < used.size() && !used[idx]) {
        used[idx] = true;
        break;
      }
      if (pool_count > 0) {
        cid = pool[static_cast<std::size_t>(xorshift32(inter_level_rng_) % pool_count)];
      }
    }

    bazaar_offers_[i].character = cid;
    bazaar_offers_[i].rarity = rarity;
    bazaar_offers_[i].cost_essence = seed_cost_essence(rarity, level_n);
    bazaar_offers_[i].purchased = false;
  }
}

bool TowerSwarmGame::initialize() {
  if (!engine_) {
    return false;
  }

  if (!SaveState::load(game_state_)) {
    game_state_.resetToNewProfile();
    (void)SaveState::save(game_state_);
  }
  grid_cols_ = kWorldWidthPx / kTileSizePx;
  grid_rows_ = kWorldHeightPx / kTileSizePx;
  path_grid_.reset(grid_cols_, grid_rows_);
  cell_occupant_.assign(
      static_cast<std::size_t>(std::max(0, grid_cols_ * grid_rows_)),
      INVALID_ID);
  deployed_roster_.assign(game_state_.roster.size(), INVALID_ID);
  selected_roster_index_ = 0;
  show_sell_confirm_ = false;
  ghost_valid_ = false;
  ghost_active_ = false;

  camera_.initialize(engine_, kWorldWidthPx, kWorldHeightPx);

  for (std::size_t i = 0; i < static_cast<std::size_t>(Biome::Count); ++i) {
    const auto biome = static_cast<Biome>(static_cast<std::uint8_t>(i));
    biome_tile_texture_[i] = registerBiomeTileTexture(biome);
  }

  const int active_tex =
      biome_tile_texture_[static_cast<std::size_t>(active_biome_)];
  if (active_tex < 0) {
    SDL_Log("TowerSwarm: missing tile texture for biome.");
    return false;
  }

  const int cols = kWorldWidthPx / kTileSizePx;
  const int rows = kWorldHeightPx / kTileSizePx;
  const int tile_count = std::max(1, cols * rows);

  auto tiles = std::make_unique<TileContainer>(0, 0, tile_count);
  engine_register_static_type(engine_, tiles.get());
  tiles_ = tiles.release();
  tiles_->buildSolidTilemap(cols, rows, kTileSizePx, active_tex);

  base_texture_id_ = register_solid_texture(engine_, kBaseSizePx, kBaseColor);
  if (base_texture_id_ < 0) {
    SDL_Log("TowerSwarm: failed to register base texture.");
    return false;
  }

  auto base = std::make_unique<BaseEntity>(0, 0, 1);
  engine_register_static_type(engine_, base.get());
  base_ = base.release();
  const float base_x =
      kWorldWidthPx * 0.5f - static_cast<float>(kBaseSizePx) * 0.5f;
  const float base_y =
      kWorldHeightPx * 0.5f - static_cast<float>(kBaseSizePx) * 0.5f;
  base_id_ = base_->createBase(engine_, base_x, base_y, level::kBaseHp,
                               base_texture_id_, kBaseSizePx);
  if (base_id_ == INVALID_ID) {
    SDL_Log("TowerSwarm: failed to create base entity.");
    return false;
  }

  for (std::size_t i = 0; i < static_cast<std::size_t>(CharacterId::Count); ++i) {
    const auto id = static_cast<CharacterId>(static_cast<std::uint8_t>(i));
    for (int band = 0; band < evolution::kVisualBandCount; ++band) {
      creature_texture_[i][static_cast<std::size_t>(band)] =
          register_creature_texture(engine_, id, band);
      if (creature_texture_[i][static_cast<std::size_t>(band)] < 0) {
        SDL_Log("TowerSwarm: failed to register creature texture %d band %d",
                static_cast<int>(id), band);
        return false;
      }
    }
    projectile_texture_[i] =
        register_solid_texture(engine_, kProjectileSizePx, character_color(id));
    if (projectile_texture_[i] < 0) {
      SDL_Log("TowerSwarm: failed to register projectile texture %d",
              static_cast<int>(id));
      return false;
    }
  }

  for (std::size_t i = 0; i < static_cast<std::size_t>(EnemyType::Count); ++i) {
    const auto t = static_cast<EnemyType>(static_cast<std::uint8_t>(i));
    enemy_texture_[i] =
        register_solid_texture(engine_, kEnemyBaseSizePx, enemy_color(t));
    if (enemy_texture_[i] < 0) {
      SDL_Log("TowerSwarm: failed to register enemy texture %d",
              static_cast<int>(t));
      return false;
    }
  }

  projectile_texture_id_ =
      projectile_texture_[static_cast<std::size_t>(CharacterId::Brix)];
  pickup_texture_id_ =
      register_solid_texture(engine_, kPickupSizePx, kPickupColor);
  if (projectile_texture_id_ < 0 || pickup_texture_id_ < 0) {
    SDL_Log("TowerSwarm: failed to register projectile/pickup textures.");
    return false;
  }

  auto creatures =
      std::make_unique<CreatureContainer>(engine_, 0, 0, kCreaturePoolCapacity);
  engine_register_hybrid_type(engine_, creatures.get());
  creatures_ = creatures.release();
  creatures_->bindGameState(&game_state_);
  creatures_->bindBase(base_, base_id_);
  creatures_->bindPathGrid(&path_grid_);
  creatures_->setCharacterTextures(&creature_texture_);
  creatures_->setProjectileTextures(&projectile_texture_);

  auto pickups =
      std::make_unique<PickupContainer>(engine_, 0, 0, kPickupPoolCapacity);
  engine_register_dynamic_type(engine_, pickups.get());
  pickups_ = pickups.release();
  pickups_->bindCreatures(creatures_);
  pickups_->bindGameState(&game_state_);
  pickups_->setTexture(pickup_texture_id_);

  auto enemies = std::make_unique<EnemyContainer>(engine_, 0, 0, kEnemyPoolCapacity);
  engine_register_dynamic_type(engine_, enemies.get());
  enemies_ = enemies.release();
  enemies_->bindBase(base_, base_id_);
  enemies_->bindCreatures(creatures_);
  enemies_->bindPickups(pickups_);
  enemies_->bindGameState(&game_state_);
  enemies_->setEnemyTextures(&enemy_texture_);

  auto projectiles = std::make_unique<ProjectileContainer>(engine_, 0, 0,
                                                           kProjectilePoolCapacity);
  engine_register_dynamic_type(engine_, projectiles.get());
  projectiles_ = projectiles.release();
  projectiles_->bindEnemies(enemies_);
  projectiles_->bindCreatures(creatures_);
  projectiles_->bindGameState(&game_state_);
  projectiles_->setTexture(projectile_texture_id_);

  creatures_->bindEnemies(enemies_);
  creatures_->bindProjectiles(projectiles_);

  level_manager_.bind(engine_, base_, base_id_, enemies_,
                      &game_state_);
  show_main_menu_ = true;

  engine_mark_static_dirty(engine_);

  SDL_SetWindowTitle(engine_->window, "Tower Swarm");
  return true;
}

void TowerSwarmGame::tick(float dt, const InputManager &input) {
  if (!engine_) {
    return;
  }

  if (input.wasPressed(SDL_SCANCODE_G)) {
    show_debug_grid_ = !show_debug_grid_;
  }

  if (show_main_menu_) {
    if (input.wasPressed(SDL_SCANCODE_RETURN) ||
        input.wasPressed(SDL_SCANCODE_KP_ENTER) ||
        input.wasPressed(SDL_SCANCODE_SPACE) ||
        input.wasMousePressed(SDL_BUTTON_LEFT)) {
      show_main_menu_ = false;
      startLevel(game_state_.level_number);
    } else if (input.wasPressed(SDL_SCANCODE_N)) {
      game_state_.resetToNewProfile();
      (void)SaveState::save(game_state_);
    }

    camera_.tick(dt, input, engine_, kWorldWidthPx, kWorldHeightPx);
    return;
  }

  if (base_ && base_id_ != INVALID_ID) {
    game_state_.base_hp = std::max(0, base_->getHp(base_id_));
  }

  if (creatures_ && !deployed_roster_.empty()) {
    for (std::size_t i = 0; i < deployed_roster_.size(); ++i) {
      const EntityHandle h = deployed_roster_[i];
      if (h == INVALID_ID || h == kDeployedRosterDead) {
        continue;
      }
      if (!engine_is_handle_valid(engine_, h, creatures_->getTypeId())) {
        deployed_roster_[i] = kDeployedRosterDead;
        if (selected_creature_ == h) {
          selected_creature_ = INVALID_ID;
        }
        if (drag_candidate_ == h) {
          drag_candidate_ = INVALID_ID;
        }
      }
    }
  }

  {
    const float step = std::max(0.0f, dt);
    game_state_.screen_edge_glow_remaining_sec =
        std::max(0.0f, game_state_.screen_edge_glow_remaining_sec - step);
    merge_cooldown_remaining_sec_ =
        std::max(0.0f, merge_cooldown_remaining_sec_ - step);
    if (merge_anim_.active) {
      merge_anim_.elapsed_sec =
          std::max(0.0f, merge_anim_.elapsed_sec + step);
    }
    if (!game_state_.floating_texts.empty()) {
      for (std::size_t i = 0; i < game_state_.floating_texts.size();) {
        FloatingText &t = game_state_.floating_texts[i];
        t.remaining_sec = std::max(0.0f, t.remaining_sec - step);
        t.world_y -= evolution::kEvolutionFloatingTextRisePxPerSec * step;
        if (t.remaining_sec <= 0.0f) {
          game_state_.floating_texts[i] = game_state_.floating_texts.back();
          game_state_.floating_texts.pop_back();
          continue;
        }
        ++i;
      }
    }
  }

  if (creatures_ && grid_cols_ > 0 && grid_rows_ > 0 && !cell_occupant_.empty()) {
    std::fill(cell_occupant_.begin(), cell_occupant_.end(), INVALID_ID);
    for (std::uint32_t slot = 0;
         slot < static_cast<std::uint32_t>(creatures_->count); ++slot) {
      const EntityHandle id = creatures_->getStableId(slot);
      if (id == INVALID_ID) {
        continue;
      }

      const float half = static_cast<float>(creatures_->widths[slot]) * 0.5f;
      const float cx = creatures_->x_positions[slot] + half;
      const float cy = creatures_->y_positions[slot] + half;
      const int col = std::clamp(static_cast<int>(cx / kTileSizePx), 0,
                                 std::max(0, grid_cols_ - 1));
      const int row = std::clamp(static_cast<int>(cy / kTileSizePx), 0,
                                 std::max(0, grid_rows_ - 1));
      const std::size_t idx = static_cast<std::size_t>(row * grid_cols_ + col);
      if (idx >= cell_occupant_.size()) {
        continue;
      }

      if (cell_occupant_[idx] == INVALID_ID) {
        cell_occupant_[idx] = id;
        continue;
      }

      if (cell_occupant_[idx] == id) {
        continue;
      }

      // Collision fallback: prefer the higher-tier creature, otherwise keep the
      // existing occupant for determinism.
      const EntityHandle cur = cell_occupant_[idx];
      if (engine_is_handle_valid(engine_, cur, creatures_->getTypeId())) {
        const std::uint32_t cur_slot = creatures_->getSlot(cur);
        if (cur_slot != INVALID_ID &&
            cur_slot < static_cast<std::uint32_t>(creatures_->count)) {
          const int cur_t = std::max(1, creatures_->tier[cur_slot]);
          const int new_t = std::max(1, creatures_->tier[slot]);
          if (new_t > cur_t) {
            cell_occupant_[idx] = id;
          }
          continue;
        }
      }
      cell_occupant_[idx] = id;
    }
  }

  const LevelManagerState level_state = level_manager_.state();
  const bool allow_gameplay_input =
      (level_state == LevelManagerState::Playing ||
       level_state == LevelManagerState::WaveClear);

  if (!show_sell_confirm_ && !show_level_select_) {
    if (level_state == LevelManagerState::Failed) {
      if (input.wasPressed(SDL_SCANCODE_R) ||
          input.wasPressed(SDL_SCANCODE_RETURN)) {
        if (have_level_start_snapshot_) {
          SaveState::restorePersistent(game_state_, level_start_snapshot_);
          (void)SaveState::save(game_state_);
        }
        startLevel(level_manager_.levelDef().level_number);
        return;
      }
    }
  }

  if (allow_gameplay_input && !show_sell_confirm_ && !show_level_select_ &&
      input.wasPressed(SDL_SCANCODE_TAB) && !game_state_.roster.empty()) {
    selected_roster_index_ =
        (selected_roster_index_ + 1) %
        static_cast<std::int32_t>(game_state_.roster.size());
  }

  const float zoom = safe_zoom(engine_);
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  get_camera_world_rect(engine_, x1, y1, x2, y2);

  const float mouse_sx = static_cast<float>(input.mouseX());
  const float mouse_sy = static_cast<float>(input.mouseY());
  const float mouse_wx = x1 + (mouse_sx / zoom);
  const float mouse_wy = y1 + (mouse_sy / zoom);

  const bool mouse_over_ui = mouse_sy <= static_cast<float>(kHudTopBarHeightPx);

  if (!show_sell_confirm_ && show_level_select_) {
    const std::int32_t max_level =
        std::max<std::int32_t>(1, game_state_.max_level_reached);
    level_select_level_ =
        std::clamp(level_select_level_, 1, max_level);

    if (input.wasPressed(SDL_SCANCODE_ESCAPE)) {
      show_level_select_ = false;
    } else if (input.wasPressed(SDL_SCANCODE_LEFT) ||
               input.wasPressed(SDL_SCANCODE_A)) {
      level_select_level_ =
          std::max<std::int32_t>(1, level_select_level_ - 1);
    } else if (input.wasPressed(SDL_SCANCODE_RIGHT) ||
               input.wasPressed(SDL_SCANCODE_D)) {
      level_select_level_ =
          std::min<std::int32_t>(max_level, level_select_level_ + 1);
    } else if (input.wasPressed(SDL_SCANCODE_RETURN)) {
      startLevel(level_select_level_);
      return;
    }

    if (input.wasMousePressed(SDL_BUTTON_LEFT)) {
      const float w = engine_->camera.width;
      const float h = engine_->camera.height;
      const float pw = static_cast<float>(kConfirmDialogWidthPx);
      const float ph = static_cast<float>(kConfirmDialogHeightPx);
      const float px = (w - pw) * 0.5f;
      const float py = (h - ph) * 0.5f;

      const float bw = static_cast<float>(kConfirmDialogButtonWidthPx);
      const float bh = static_cast<float>(kConfirmDialogButtonHeightPx);
      const float gap = static_cast<float>(kConfirmDialogButtonGapPx);

      const float buttons_y = py + ph - bh - static_cast<float>(kHudPaddingPx);
      const float buttons_x = px + (pw - (2.0f * bw + gap)) * 0.5f;
      const SDL_FRect play_rect = {buttons_x, buttons_y, bw, bh};
      const SDL_FRect back_rect = {buttons_x + bw + gap, buttons_y, bw, bh};

      if (point_in_rect(mouse_sx, mouse_sy, play_rect)) {
        startLevel(level_select_level_);
        return;
      }
      if (point_in_rect(mouse_sx, mouse_sy, back_rect)) {
        show_level_select_ = false;
      }
    }
  }

  if (!show_sell_confirm_ && show_level_select_) {
    ghost_valid_ = false;
    ghost_active_ = false;
    ghost_world_x_ = 0.0f;
    ghost_world_y_ = 0.0f;
    camera_.tick(dt, input, engine_, kWorldWidthPx, kWorldHeightPx);
    return;
  }

  if (!show_sell_confirm_ && !show_level_select_ &&
      level_state == LevelManagerState::Failed) {
    if (input.wasMousePressed(SDL_BUTTON_LEFT)) {
      const float w = engine_->camera.width;
      const float h = engine_->camera.height;
      const float pw = static_cast<float>(kConfirmDialogWidthPx);
      const float ph = static_cast<float>(kConfirmDialogHeightPx);
      const float px = (w - pw) * 0.5f;
      const float py = (h - ph) * 0.5f;

      const float bw = static_cast<float>(kConfirmDialogButtonWidthPx);
      const float bh = static_cast<float>(kConfirmDialogButtonHeightPx);
      const float gap = static_cast<float>(kConfirmDialogButtonGapPx);

      const float buttons_y = py + ph - bh - static_cast<float>(kHudPaddingPx);
      const float buttons_x = px + (pw - (2.0f * bw + gap)) * 0.5f;
      const SDL_FRect primary_rect = {buttons_x, buttons_y, bw, bh};
      const SDL_FRect secondary_rect = {buttons_x + bw + gap, buttons_y, bw, bh};

      if (point_in_rect(mouse_sx, mouse_sy, primary_rect)) {
        if (have_level_start_snapshot_) {
          SaveState::restorePersistent(game_state_, level_start_snapshot_);
          (void)SaveState::save(game_state_);
        }
        startLevel(level_manager_.levelDef().level_number);
        return;
      } else if (point_in_rect(mouse_sx, mouse_sy, secondary_rect)) {
        if (have_level_start_snapshot_) {
          SaveState::restorePersistent(game_state_, level_start_snapshot_);
          (void)SaveState::save(game_state_);
        }
        show_level_select_ = true;
        const std::int32_t max_level =
            std::max<std::int32_t>(1, game_state_.max_level_reached);
        level_select_level_ = std::clamp<std::int32_t>(
            level_manager_.levelDef().level_number, 1, max_level);
      }
    }

    ghost_valid_ = false;
    ghost_active_ = false;
    ghost_world_x_ = 0.0f;
    ghost_world_y_ = 0.0f;

    camera_.tick(dt, input, engine_, kWorldWidthPx, kWorldHeightPx);
    return;
  }

  if (!show_sell_confirm_ && !show_level_select_ && show_armory_) {
    const float w = engine_->camera.width;
    const float h = engine_->camera.height;
    const float margin = static_cast<float>(kHudPaddingPx);
    const float panel_w =
        std::min(1040.0f, std::max(0.0f, w - 2.0f * margin));
    const float panel_h =
        std::min(720.0f, std::max(0.0f, h - 2.0f * margin));
    const float panel_x = (w - panel_w) * 0.5f;
    const float panel_y = (h - panel_h) * 0.5f;

    const float header_h = 56.0f;
    const float tab_h = 32.0f;
    const float inner_pad = 14.0f;

    const float tab_y = panel_y + header_h;
    const float tab_w = 140.0f;
    const float tab_gap = 8.0f;
    const SDL_FRect tab_chars = {panel_x + inner_pad, tab_y, tab_w, tab_h};
    const SDL_FRect tab_masteries = {tab_chars.x + tab_w + tab_gap, tab_y, tab_w,
                                     tab_h};
    const SDL_FRect tab_relics = {tab_masteries.x + tab_w + tab_gap, tab_y, tab_w,
                                  tab_h};
    const SDL_FRect tab_cosmetics = {tab_relics.x + tab_w + tab_gap, tab_y, tab_w,
                                     tab_h};

    const SDL_FRect close_rect = {
        panel_x + panel_w - inner_pad - 96.0f,
        panel_y + (header_h - 28.0f) * 0.5f,
        96.0f,
        28.0f,
    };

    auto stars_for_level = [&](int level) -> int {
      if (level <= 0) {
        return 0;
      }
      const std::size_t idx =
          static_cast<std::size_t>(std::max(0, level - 1));
      if (idx >= game_state_.stars_per_level.size()) {
        return 0;
      }
      return std::clamp<int>(game_state_.stars_per_level[idx], 0, 3);
    };

    auto character_requirements_met = [&](CharacterId cid) -> bool {
      if (game_state_.isCharacterUnlocked(cid)) {
        return false;
      }
      const CharacterDefinition &def = get_character_def(cid);
      if (cid == CharacterId::Orin) {
        return stars_for_level(unlocks::kOrinUnlockLevel) >= 3;
      }
      return game_state_.max_level_reached >= def.unlock_level;
    };

    if (input.wasPressed(SDL_SCANCODE_ESCAPE)) {
      if (show_armory_confirm_) {
        show_armory_confirm_ = false;
        armory_confirm_kind_ = ArmoryConfirmKind::None;
      } else if (show_armory_character_detail_) {
        show_armory_character_detail_ = false;
      } else {
        show_armory_ = false;
      }
      ghost_valid_ = false;
      ghost_active_ = false;
      ghost_world_x_ = 0.0f;
      ghost_world_y_ = 0.0f;
      camera_.tick(dt, input, engine_, kWorldWidthPx, kWorldHeightPx);
      return;
    }

    if (show_armory_confirm_) {
      if (input.wasMousePressed(SDL_BUTTON_LEFT)) {
        const float pw = static_cast<float>(kConfirmDialogWidthPx);
        const float ph = static_cast<float>(kConfirmDialogHeightPx);
        const float px = (w - pw) * 0.5f;
        const float py = (h - ph) * 0.5f;

        const float bw = static_cast<float>(kConfirmDialogButtonWidthPx);
        const float bh = static_cast<float>(kConfirmDialogButtonHeightPx);
        const float gap = static_cast<float>(kConfirmDialogButtonGapPx);

        const float buttons_y = py + ph - bh - static_cast<float>(kHudPaddingPx);
        const float buttons_x = px + (pw - (2.0f * bw + gap)) * 0.5f;
        const SDL_FRect yes_rect = {buttons_x, buttons_y, bw, bh};
        const SDL_FRect no_rect = {buttons_x + bw + gap, buttons_y, bw, bh};

        if (point_in_rect(mouse_sx, mouse_sy, yes_rect)) {
          switch (armory_confirm_kind_) {
          case ArmoryConfirmKind::UnlockCharacter: {
            const CharacterId cid = armory_confirm_character_;
            const CharacterDefinition &def = get_character_def(cid);
            const int cost = std::max(0, def.unlock_shards);
            if (!game_state_.isCharacterUnlocked(cid) && cost > 0 &&
                game_state_.shards >= cost && character_requirements_met(cid)) {
              game_state_.shards -= cost;
              game_state_.unlockCharacter(cid);
              game_state_.sanitizeCharacterUnlocks();
              (void)SaveState::save(game_state_);
            }
            break;
          }
          case ArmoryConfirmKind::BuyMasteryRank: {
            const MasteryId mid = armory_confirm_mastery_;
            const int cur = game_state_.masteryRank(mid);
            const int max_r = GameState::masteryMaxRanks(mid);
            if (cur < max_r) {
              const int cost = std::max(0, GameState::masteryNextRankCost(mid, cur));
              if (cost > 0 && game_state_.shards >= cost) {
                game_state_.shards -= cost;
                const std::size_t idx = static_cast<std::size_t>(mid);
                if (idx < game_state_.mastery_ranks.size()) {
                  game_state_.mastery_ranks[idx] = static_cast<std::uint8_t>(
                      std::clamp(cur + 1, 0, max_r));
                }
                game_state_.sanitizeMasteries();
                (void)SaveState::save(game_state_);
              }
            }
            break;
          }
          case ArmoryConfirmKind::UnlockRelic: {
            const RelicId rid = armory_confirm_relic_;
            const std::size_t idx = static_cast<std::size_t>(rid);
            if (rid != RelicId::None && rid != RelicId::Count &&
                idx < game_state_.relic_unlocked.size() &&
                game_state_.relic_unlocked[idx] == 0) {
              const RelicDef &def = RelicSystem::def(rid);
              const int cost = std::max(0, def.shard_cost);
              if (cost > 0 && game_state_.shards >= cost) {
                game_state_.shards -= cost;
                game_state_.relic_unlocked[idx] = 1;
                RelicSystem::sanitizePersistent(game_state_);
                (void)SaveState::save(game_state_);
              }
            }
            break;
          }
          case ArmoryConfirmKind::None:
            break;
          }

          show_armory_confirm_ = false;
          armory_confirm_kind_ = ArmoryConfirmKind::None;
        } else if (point_in_rect(mouse_sx, mouse_sy, no_rect)) {
          show_armory_confirm_ = false;
          armory_confirm_kind_ = ArmoryConfirmKind::None;
        }
      }

      ghost_valid_ = false;
      ghost_active_ = false;
      ghost_world_x_ = 0.0f;
      ghost_world_y_ = 0.0f;
      camera_.tick(dt, input, engine_, kWorldWidthPx, kWorldHeightPx);
      return;
    }

    const float content_x = panel_x + inner_pad;
    const float content_y = tab_y + tab_h + inner_pad;
    const float content_w = std::max(0.0f, panel_w - 2.0f * inner_pad);
    const float content_h = std::max(0.0f, panel_h - header_h - tab_h - inner_pad * 2.0f);

    const SDL_FRect detail_rect = {
        panel_x + (panel_w - std::min(640.0f, panel_w - 2.0f * inner_pad)) * 0.5f,
        panel_y + (panel_h - std::min(420.0f, panel_h - 2.0f * inner_pad)) * 0.5f,
        std::min(640.0f, panel_w - 2.0f * inner_pad),
        std::min(420.0f, panel_h - 2.0f * inner_pad),
    };

    const SDL_FRect detail_close = {
        detail_rect.x + detail_rect.w - 92.0f,
        detail_rect.y + 12.0f,
        80.0f,
        28.0f,
    };

    const SDL_FRect detail_unlock = {
        detail_rect.x + detail_rect.w - 200.0f,
        detail_rect.y + detail_rect.h - 46.0f,
        180.0f,
        34.0f,
    };

    if (input.wasMousePressed(SDL_BUTTON_LEFT)) {
      if (show_armory_character_detail_) {
        if (!point_in_rect(mouse_sx, mouse_sy, detail_rect)) {
          show_armory_character_detail_ = false;
        } else if (point_in_rect(mouse_sx, mouse_sy, detail_close)) {
          show_armory_character_detail_ = false;
        } else {
          const CharacterId cid = armory_selected_character_;
          const CharacterDefinition &def = get_character_def(cid);
          const int cost = std::max(0, def.unlock_shards);
          if (cost > 0 && !game_state_.isCharacterUnlocked(cid) &&
              character_requirements_met(cid) && game_state_.shards >= cost &&
              point_in_rect(mouse_sx, mouse_sy, detail_unlock)) {
            show_armory_confirm_ = true;
            armory_confirm_kind_ = ArmoryConfirmKind::UnlockCharacter;
            armory_confirm_character_ = cid;
          }
        }
      } else {
        if (point_in_rect(mouse_sx, mouse_sy, close_rect)) {
          show_armory_ = false;
          ghost_valid_ = false;
          ghost_active_ = false;
          ghost_world_x_ = 0.0f;
          ghost_world_y_ = 0.0f;
          camera_.tick(dt, input, engine_, kWorldWidthPx, kWorldHeightPx);
          return;
        }

        if (point_in_rect(mouse_sx, mouse_sy, tab_chars)) {
          armory_tab_ = ArmoryTab::Characters;
        } else if (point_in_rect(mouse_sx, mouse_sy, tab_masteries)) {
          armory_tab_ = ArmoryTab::Masteries;
        } else if (point_in_rect(mouse_sx, mouse_sy, tab_relics)) {
          armory_tab_ = ArmoryTab::Relics;
        } else if (point_in_rect(mouse_sx, mouse_sy, tab_cosmetics)) {
          armory_tab_ = ArmoryTab::Cosmetics;
        } else if (armory_tab_ == ArmoryTab::Characters) {
          const int cols = 5;
          const float gap = 10.0f;
          const float card_w =
              cols > 0 ? std::max(0.0f, (content_w - gap * (cols - 1)) /
                                             static_cast<float>(cols))
                       : 0.0f;
          const float card_h =
              std::max(0.0f, (content_h - gap) * 0.5f);
          for (int i = 0; i < static_cast<int>(CharacterId::Count); ++i) {
            const int row = i / cols;
            const int col = i % cols;
            const SDL_FRect card = {content_x + static_cast<float>(col) * (card_w + gap),
                                    content_y + static_cast<float>(row) * (card_h + gap),
                                    card_w, card_h};
            if (!point_in_rect(mouse_sx, mouse_sy, card)) {
              continue;
            }
            armory_selected_character_ =
                static_cast<CharacterId>(static_cast<std::uint8_t>(i));
            show_armory_character_detail_ = true;
            break;
          }
        } else if (armory_tab_ == ArmoryTab::Masteries) {
          const float row_h = 58.0f;
          const float gap = 8.0f;
          for (int i = 0; i < static_cast<int>(MasteryId::Count); ++i) {
            const SDL_FRect rr = {content_x,
                                  content_y + static_cast<float>(i) * (row_h + gap),
                                  content_w, row_h};
            if (!point_in_rect(mouse_sx, mouse_sy, rr)) {
              continue;
            }

            const auto mid = static_cast<MasteryId>(static_cast<std::uint8_t>(i));
            const int cur = game_state_.masteryRank(mid);
            const int max_r = GameState::masteryMaxRanks(mid);
            if (cur >= max_r) {
              break;
            }
            const int cost = std::max(0, GameState::masteryNextRankCost(mid, cur));
            if (cost <= 0 || game_state_.shards < cost) {
              break;
            }

            const SDL_FRect buy = {rr.x + rr.w - 130.0f, rr.y + 14.0f, 120.0f, 30.0f};
            if (!point_in_rect(mouse_sx, mouse_sy, buy)) {
              break;
            }
            show_armory_confirm_ = true;
            armory_confirm_kind_ = ArmoryConfirmKind::BuyMasteryRank;
            armory_confirm_mastery_ = mid;
            break;
          }
        } else if (armory_tab_ == ArmoryTab::Relics) {
          const int cols = 4;
          const float gap = 10.0f;
          const float card_w =
              cols > 0 ? std::max(0.0f, (content_w - gap * (cols - 1)) /
                                             static_cast<float>(cols))
                       : 0.0f;
          const float card_h = 68.0f;
          for (std::size_t i = 0; i < RelicSystem::kRelicCount; ++i) {
            const int row = static_cast<int>(i / static_cast<std::size_t>(cols));
            const int col = static_cast<int>(i % static_cast<std::size_t>(cols));
            const SDL_FRect rr = {content_x + static_cast<float>(col) * (card_w + gap),
                                  content_y + static_cast<float>(row) * (card_h + gap),
                                  card_w, card_h};
            if (!point_in_rect(mouse_sx, mouse_sy, rr)) {
              continue;
            }
            const RelicId rid = static_cast<RelicId>(i);
            if (game_state_.isRelicUnlocked(rid)) {
              break;
            }
            const RelicDef &def = RelicSystem::def(rid);
            const int cost = std::max(0, def.shard_cost);
            if (cost <= 0 || game_state_.shards < cost) {
              break;
            }
            show_armory_confirm_ = true;
            armory_confirm_kind_ = ArmoryConfirmKind::UnlockRelic;
            armory_confirm_relic_ = rid;
            break;
          }
        }
      }
    }

    ghost_valid_ = false;
    ghost_active_ = false;
    ghost_world_x_ = 0.0f;
    ghost_world_y_ = 0.0f;
    camera_.tick(dt, input, engine_, kWorldWidthPx, kWorldHeightPx);
    return;
  }

  if (!show_sell_confirm_ && !show_level_select_ && show_inter_level_ &&
      level_state == LevelManagerState::LevelClear) {
    const float step = std::max(0.0f, dt);
    inter_level_elapsed_sec_ =
        std::max(0.0f, inter_level_elapsed_sec_ + step);

    if (input.wasPressed(SDL_SCANCODE_1)) {
      inter_level_tab_ = InterLevelTab::Bazaar;
    } else if (input.wasPressed(SDL_SCANCODE_2)) {
      inter_level_tab_ = InterLevelTab::Forge;
    } else if (input.wasPressed(SDL_SCANCODE_3)) {
      inter_level_tab_ = InterLevelTab::Relics;
    } else if (input.wasPressed(SDL_SCANCODE_4)) {
      inter_level_tab_ = InterLevelTab::Repair;
    }

    const float w = engine_->camera.width;
    const float h = engine_->camera.height;
    const float margin = static_cast<float>(kHudPaddingPx);

    const float panel_w = std::max(0.0f, w - 2.0f * margin);
    const float panel_h = std::max(0.0f, h - 2.0f * margin);

    const float slide_sec = 0.4f;
    float t = slide_sec > 0.0f
                  ? std::clamp(inter_level_elapsed_sec_ / slide_sec, 0.0f, 1.0f)
                  : 1.0f;
    t = t * t * (3.0f - 2.0f * t); // smoothstep

    const float panel_end_x = margin;
    const float panel_x = w + (panel_end_x - w) * t;
    const float panel_y = margin;

    const float header_h = 72.0f;
    const float tab_h = 32.0f;
    const float inner_pad = 14.0f;
    const float roster_h = 72.0f;
    const float button_h = 38.0f;

    const float content_top = panel_y + header_h + tab_h + inner_pad;
    const float button_y = panel_y + panel_h - button_h - inner_pad;
    const float roster_y = button_y - roster_h - inner_pad;
    const float content_h = std::max(0.0f, roster_y - content_top);

    const float content_x = panel_x + inner_pad;
    const float content_w = std::max(0.0f, panel_w - 2.0f * inner_pad);

    const float results_w = std::min(360.0f, content_w * 0.36f);
    const float shop_gap = inner_pad;
    const float shop_x = content_x + results_w + shop_gap;
    const float shop_w = std::max(0.0f, content_w - results_w - shop_gap);

    const float tab_y = panel_y + header_h;
    const float tab_w = 120.0f;
    const float tab_gap = 8.0f;
    const SDL_FRect tab_bazaar = {panel_x + inner_pad, tab_y, tab_w, tab_h};
    const SDL_FRect tab_forge = {tab_bazaar.x + tab_w + tab_gap, tab_y, tab_w,
                                 tab_h};
    const SDL_FRect tab_relics = {tab_forge.x + tab_w + tab_gap, tab_y, tab_w,
                                  tab_h};
    const SDL_FRect tab_repair = {tab_relics.x + tab_w + tab_gap, tab_y, tab_w,
                                  tab_h};

    const float btn_w = 150.0f;
    const float btn_gap = 12.0f;
    const SDL_FRect btn_levels = {panel_x + inner_pad, button_y, btn_w, button_h};
    const SDL_FRect btn_armory = {btn_levels.x + btn_w + btn_gap, button_y, btn_w,
                                  button_h};
    const SDL_FRect btn_next = {panel_x + panel_w - inner_pad - btn_w, button_y,
                                btn_w, button_h};
    const SDL_FRect btn_replay = {btn_next.x - btn_gap - btn_w, button_y, btn_w,
                                  button_h};

    auto already_owned = [&](CharacterId cid) -> bool {
      for (const RosterEntry &re : game_state_.roster) {
        if (re.character == cid) {
          return true;
        }
      }
      return false;
    };

    auto buy_offer = [&](std::size_t offer_index) {
      if (offer_index >= bazaar_offers_.size()) {
        return;
      }
      BazaarOffer &offer = bazaar_offers_[offer_index];
      if (offer.purchased) {
        return;
      }
      if (game_state_.essence < offer.cost_essence) {
        return;
      }

      RosterEntry re{};
      re.character = offer.character;
      re.tier = 1;
      re.kills = 0;
      re.seed_cost_essence = offer.cost_essence;
      game_state_.roster.push_back(re);

      offer.purchased = true;
      game_state_.essence -= offer.cost_essence;
      (void)SaveState::save(game_state_);
    };

    auto try_upgrade_node = [&](CharacterId cid, UpgradeNode node, int max_rank) {
      if (max_rank <= 0) {
        return;
      }

      int current = 0;
      for (const RosterEntry &re : game_state_.roster) {
        if (re.character != cid) {
          continue;
        }
        current = std::max<int>(
            current,
            re.upgrades[static_cast<std::size_t>(node)]);
      }
      current = std::clamp(current, 0, max_rank);
      if (current >= max_rank) {
        return;
      }

      const int level_n = std::max(1, level_manager_.levelDef().level_number);
      const float level_mod = 1.0f + static_cast<float>(level_n) * 0.05f;
      const float raw =
          static_cast<float>((current + 1) * inter_level_shop::kUpgradeCostBase) *
          level_mod;
      const int cost = std::max(0, static_cast<int>(std::lround(raw)));
      if (game_state_.essence < cost) {
        return;
      }

      const std::uint8_t next_rank =
          static_cast<std::uint8_t>(std::clamp(current + 1, 0, max_rank));
      for (RosterEntry &re : game_state_.roster) {
        if (re.character != cid) {
          continue;
        }
        const std::size_t idx = static_cast<std::size_t>(node);
        re.upgrades[idx] =
            std::max<std::uint8_t>(re.upgrades[idx], next_rank);
      }
      game_state_.essence -= cost;
      (void)SaveState::save(game_state_);
      if (creatures_) {
        creatures_->recalcStatsForCharacter(cid);
      }
    };

    auto open_level_select = [&]() {
      show_inter_level_ = false;
      show_bazaar_duplicate_confirm_ = false;
      pending_bazaar_offer_index_ = -1;
      show_level_select_ = true;
      const std::int32_t max_level =
          std::max<std::int32_t>(1, game_state_.max_level_reached);
      level_select_level_ = std::clamp<std::int32_t>(
          level_manager_.levelDef().level_number, 1, max_level);
    };

    if (!show_bazaar_duplicate_confirm_) {
      if (input.wasPressed(SDL_SCANCODE_RETURN) || input.wasPressed(SDL_SCANCODE_N)) {
        startLevel(level_manager_.levelDef().level_number + 1);
        return;
      }
      if (input.wasPressed(SDL_SCANCODE_R)) {
        startLevel(level_manager_.levelDef().level_number);
        return;
      }
      if (input.wasPressed(SDL_SCANCODE_L)) {
        open_level_select();
        camera_.tick(dt, input, engine_, kWorldWidthPx, kWorldHeightPx);
        return;
      }
    }

    if (input.wasMousePressed(SDL_BUTTON_LEFT)) {
      if (show_bazaar_duplicate_confirm_) {
        const float pw = static_cast<float>(kConfirmDialogWidthPx);
        const float ph = static_cast<float>(kConfirmDialogHeightPx);
        const float px = (w - pw) * 0.5f;
        const float py = (h - ph) * 0.5f;

        const float bw = static_cast<float>(kConfirmDialogButtonWidthPx);
        const float bh = static_cast<float>(kConfirmDialogButtonHeightPx);
        const float gap = static_cast<float>(kConfirmDialogButtonGapPx);

        const float buttons_y = py + ph - bh - static_cast<float>(kHudPaddingPx);
        const float buttons_x = px + (pw - (2.0f * bw + gap)) * 0.5f;
        const SDL_FRect yes_rect = {buttons_x, buttons_y, bw, bh};
        const SDL_FRect no_rect = {buttons_x + bw + gap, buttons_y, bw, bh};

        if (point_in_rect(mouse_sx, mouse_sy, yes_rect)) {
          if (pending_bazaar_offer_index_ >= 0 &&
              pending_bazaar_offer_index_ <
                  static_cast<std::int32_t>(bazaar_offers_.size())) {
            buy_offer(static_cast<std::size_t>(pending_bazaar_offer_index_));
          }
          show_bazaar_duplicate_confirm_ = false;
          pending_bazaar_offer_index_ = -1;
        } else if (point_in_rect(mouse_sx, mouse_sy, no_rect)) {
          show_bazaar_duplicate_confirm_ = false;
          pending_bazaar_offer_index_ = -1;
        }
      } else {
        if (point_in_rect(mouse_sx, mouse_sy, tab_bazaar)) {
          inter_level_tab_ = InterLevelTab::Bazaar;
        } else if (point_in_rect(mouse_sx, mouse_sy, tab_forge)) {
          inter_level_tab_ = InterLevelTab::Forge;
        } else if (point_in_rect(mouse_sx, mouse_sy, tab_relics)) {
          inter_level_tab_ = InterLevelTab::Relics;
        } else if (point_in_rect(mouse_sx, mouse_sy, tab_repair)) {
          inter_level_tab_ = InterLevelTab::Repair;
        } else if (point_in_rect(mouse_sx, mouse_sy, btn_next)) {
          startLevel(level_manager_.levelDef().level_number + 1);
          return;
        } else if (point_in_rect(mouse_sx, mouse_sy, btn_replay)) {
          startLevel(level_manager_.levelDef().level_number);
          return;
        } else if (point_in_rect(mouse_sx, mouse_sy, btn_levels)) {
          open_level_select();
          camera_.tick(dt, input, engine_, kWorldWidthPx, kWorldHeightPx);
          return;
        } else if (point_in_rect(mouse_sx, mouse_sy, btn_armory)) {
          show_armory_ = true;
          armory_tab_ = ArmoryTab::Characters;
          show_armory_character_detail_ = false;
          show_armory_confirm_ = false;
          armory_confirm_kind_ = ArmoryConfirmKind::None;
          camera_.tick(dt, input, engine_, kWorldWidthPx, kWorldHeightPx);
          return;
        } else {
          const SDL_FRect shop_rect = {shop_x, content_top, shop_w, content_h};

          if (inter_level_tab_ == InterLevelTab::Bazaar) {
            const SDL_FRect reroll_rect = {shop_rect.x + shop_rect.w - 144.0f,
                                           shop_rect.y, 144.0f, 28.0f};
            if (point_in_rect(mouse_sx, mouse_sy, reroll_rect)) {
              const bool reroll_unlocked =
                  game_state_.player_level >= meta::kPlayerLevelUnlockReroll;
              if (reroll_unlocked && !bazaar_rerolled_ &&
                  game_state_.essence >= inter_level_shop::kRerollCostEssence) {
                game_state_.essence -= inter_level_shop::kRerollCostEssence;
                bazaar_rerolled_ = true;
                rerollBazaar();
                (void)SaveState::save(game_state_);
              }
            } else {
              const float offer_top = shop_rect.y + 40.0f;
              const float gap = 12.0f;
              const float card_w =
                  std::max(0.0f, (shop_rect.w - gap) * 0.5f);
              const float card_h = 132.0f;
              for (std::size_t i = 0; i < bazaar_offers_.size(); ++i) {
                const int row = static_cast<int>(i / 2);
                const int col = static_cast<int>(i % 2);
                SDL_FRect card = {shop_rect.x + static_cast<float>(col) * (card_w + gap),
                                  offer_top + static_cast<float>(row) * (card_h + gap),
                                  card_w, card_h};
                if (!point_in_rect(mouse_sx, mouse_sy, card)) {
                  continue;
                }
                const BazaarOffer &offer = bazaar_offers_[i];
                if (offer.purchased) {
                  break;
                }
                if (game_state_.essence < offer.cost_essence) {
                  break;
                }
                if (already_owned(offer.character)) {
                  show_bazaar_duplicate_confirm_ = true;
                  pending_bazaar_offer_index_ = static_cast<std::int32_t>(i);
                } else {
                  buy_offer(i);
                }
                break;
              }
            }
          } else if (inter_level_tab_ == InterLevelTab::Forge) {
            std::array<bool, static_cast<std::size_t>(CharacterId::Count)> seen{};
            seen.fill(false);
            std::vector<CharacterId> owned;
            owned.reserve(game_state_.roster.size());
            for (const RosterEntry &re : game_state_.roster) {
              const std::size_t idx = static_cast<std::size_t>(re.character);
              if (idx >= seen.size() || seen[idx]) {
                continue;
              }
              seen[idx] = true;
              owned.push_back(re.character);
            }
            if (owned.empty()) {
              owned.push_back(CharacterId::Brix);
            }

            const float list_w = 150.0f;
            const float list_x = shop_rect.x;
            const float list_y = shop_rect.y;
            const float btn_h = 28.0f;
            const float btn_gap = 6.0f;

            bool selected_changed = false;
            for (std::size_t i = 0; i < owned.size(); ++i) {
              const SDL_FRect br = {list_x, list_y + static_cast<float>(i) * (btn_h + btn_gap),
                                    list_w, btn_h};
              if (point_in_rect(mouse_sx, mouse_sy, br)) {
                forge_selected_ = owned[i];
                selected_changed = true;
                break;
              }
            }

            if (!selected_changed) {
              const float nodes_x = shop_rect.x + list_w + inner_pad;
              const float nodes_y = shop_rect.y + 36.0f;
              const float nodes_w = std::max(0.0f, shop_rect.w - list_w - inner_pad);
              const float node_h = 32.0f;
              const float node_gap = 10.0f;

              struct NodeDef final {
                UpgradeNode node;
                int max_rank;
              };
              constexpr std::array<NodeDef, 5> kNodes{
                  NodeDef{UpgradeNode::Strike, inter_level_shop::kUpgradeStrikeMaxRanks},
                  NodeDef{UpgradeNode::Vitality, inter_level_shop::kUpgradeVitalityMaxRanks},
                  NodeDef{UpgradeNode::Reach, inter_level_shop::kUpgradeReachMaxRanks},
                  NodeDef{UpgradeNode::Tempo, inter_level_shop::kUpgradeTempoMaxRanks},
                  NodeDef{UpgradeNode::Signature, inter_level_shop::kUpgradeSignatureMaxRanks},
              };

              for (std::size_t n = 0; n < kNodes.size(); ++n) {
                const SDL_FRect nr = {nodes_x, nodes_y + static_cast<float>(n) * (node_h + node_gap),
                                      nodes_w, node_h};
                if (point_in_rect(mouse_sx, mouse_sy, nr)) {
                  try_upgrade_node(forge_selected_, kNodes[n].node, kNodes[n].max_rank);
                  break;
                }
              }
            }
          } else if (inter_level_tab_ == InterLevelTab::Relics) {
            auto commit_relics = [&]() {
              RelicSystem::sanitizePersistent(game_state_);
              (void)SaveState::save(game_state_);
            };

            auto clear_equipped = [&](RelicId id) {
              if (id == RelicId::None) {
                return;
              }
              for (RelicId &slot_id : game_state_.equipped_relics) {
                if (slot_id == id) {
                  slot_id = RelicId::None;
                }
              }
            };

            const float slots_y = shop_rect.y + 80.0f;
            const float slot_h = 38.0f;
            const float slot_gap = 10.0f;
            const float slot_w =
                std::max(0.0f, (shop_rect.w - 2.0f * slot_gap) / 3.0f);
            bool clicked_any = false;

            for (int s = 0; s < relics::kSlotCount; ++s) {
              const SDL_FRect sr = {shop_rect.x + static_cast<float>(s) * (slot_w + slot_gap),
                                    slots_y, slot_w, slot_h};
              if (!point_in_rect(mouse_sx, mouse_sy, sr)) {
                continue;
              }
              clicked_any = true;

              if (!RelicSystem::isSlotUnlocked(s, game_state_.player_level)) {
                break;
              }

              if (relic_pick_ != RelicId::None) {
                clear_equipped(relic_pick_);
                game_state_.equipped_relics[static_cast<std::size_t>(s)] = relic_pick_;
                relic_pick_ = RelicId::None;
                commit_relics();
                break;
              }

              const RelicId cur = game_state_.equipped_relics[static_cast<std::size_t>(s)];
              if (cur != RelicId::None) {
                game_state_.equipped_relics[static_cast<std::size_t>(s)] = RelicId::None;
                commit_relics();
              }
              break;
            }

            if (!clicked_any) {
              const float list_top = slots_y + slot_h + 24.0f;
              const float card_h = 60.0f;
              const float gap = 10.0f;
              const int cols = 2;
              const float card_w =
                  cols > 0 ? std::max(0.0f, (shop_rect.w - gap) / 2.0f) : 0.0f;

              for (std::size_t i = 0; i < RelicSystem::kRelicCount; ++i) {
                const RelicId id = static_cast<RelicId>(i);
                const int row = static_cast<int>(i / static_cast<std::size_t>(cols));
                const int col = static_cast<int>(i % static_cast<std::size_t>(cols));
                SDL_FRect rr = {shop_rect.x + static_cast<float>(col) * (card_w + gap),
                                list_top + static_cast<float>(row) * (card_h + gap),
                                card_w, card_h};
                if (!point_in_rect(mouse_sx, mouse_sy, rr)) {
                  continue;
                }
                clicked_any = true;

                const RelicDef &def = RelicSystem::def(id);
                const bool unlocked = game_state_.isRelicUnlocked(id);

                if (!unlocked) {
                  show_armory_ = true;
                  armory_tab_ = ArmoryTab::Relics;
                  show_armory_character_detail_ = false;
                  relic_pick_ = RelicId::None;

                  const int cost = std::max(0, def.shard_cost);
                  if (cost > 0 && game_state_.shards >= cost) {
                    show_armory_confirm_ = true;
                    armory_confirm_kind_ = ArmoryConfirmKind::UnlockRelic;
                    armory_confirm_relic_ = id;
                  }
                  break;
                }

                if (relic_pick_ == id) {
                  relic_pick_ = RelicId::None;
                } else {
                  relic_pick_ = id;
                }
                break;
              }
            }

            if (!clicked_any) {
              relic_pick_ = RelicId::None;
            }
          } else if (inter_level_tab_ == InterLevelTab::Repair) {
            if (!repair_purchased_) {
              const float opt_x = shop_rect.x;
              const float opt_y = shop_rect.y + 36.0f;
              const float opt_w = shop_rect.w;
              const float opt_h = 34.0f;
              const float opt_gap = 10.0f;

              struct RepairOpt final {
                int cost;
                int add_hp;
                bool full;
              };
              constexpr std::array<RepairOpt, 3> kOpts{
                  RepairOpt{inter_level_shop::kRepairRestore20Cost,
                            inter_level_shop::kRepairRestore20Hp, false},
                  RepairOpt{inter_level_shop::kRepairRestore50Cost,
                            inter_level_shop::kRepairRestore50Hp, false},
                  RepairOpt{inter_level_shop::kRepairFullRestoreCost,
                            inter_level_shop::kRepairFullRestoreHp, true},
              };

              const int base_hp_now = std::max(0, game_state_.base_hp);
              for (std::size_t i = 0; i < kOpts.size(); ++i) {
                const SDL_FRect rr = {opt_x, opt_y + static_cast<float>(i) * (opt_h + opt_gap),
                                      opt_w, opt_h};
                if (!point_in_rect(mouse_sx, mouse_sy, rr)) {
                  continue;
                }
                const RepairOpt &opt = kOpts[i];
                if (game_state_.essence < opt.cost) {
                  break;
                }
                game_state_.essence -= opt.cost;
                if (opt.full) {
                  game_state_.next_level_base_hp_target =
                      std::numeric_limits<std::int32_t>::max();
                } else {
                  game_state_.next_level_base_hp_target =
                      base_hp_now + opt.add_hp;
                }
                repair_purchased_ = true;
                (void)SaveState::save(game_state_);
                break;
              }
            }
          }
        }
      }
    }

    ghost_valid_ = false;
    ghost_active_ = false;
    ghost_world_x_ = 0.0f;
    ghost_world_y_ = 0.0f;

    camera_.tick(dt, input, engine_, kWorldWidthPx, kWorldHeightPx);
    return;
  }

  bool wave_shop_consumed_left_click = false;
  if (allow_gameplay_input && level_state == LevelManagerState::WaveClear &&
      wave_buff_shop_.isOpen()) {
    wave_shop_consumed_left_click = wave_buff_shop_.tick(
        input, engine_->camera.width, engine_->camera.height);

    WaveBuffId chosen = WaveBuffId::Surge;
    bool skipped = false;
    if (wave_buff_shop_.consumeSelection(chosen, skipped)) {
      if (!skipped) {
        applyWaveBuff(chosen);
      }
    }
  }

  if (show_sell_confirm_) {
    if (input.wasMousePressed(SDL_BUTTON_LEFT)) {
      const float w = engine_->camera.width;
      const float h = engine_->camera.height;
      const float pw = static_cast<float>(kConfirmDialogWidthPx);
      const float ph = static_cast<float>(kConfirmDialogHeightPx);
      const float px = (w - pw) * 0.5f;
      const float py = (h - ph) * 0.5f;

      const float bw = static_cast<float>(kConfirmDialogButtonWidthPx);
      const float bh = static_cast<float>(kConfirmDialogButtonHeightPx);
      const float gap = static_cast<float>(kConfirmDialogButtonGapPx);

      const float buttons_y = py + ph - bh - static_cast<float>(kHudPaddingPx);
      const float buttons_x = px + (pw - (2.0f * bw + gap)) * 0.5f;
      const SDL_FRect sell_rect = {buttons_x, buttons_y, bw, bh};
      const SDL_FRect cancel_rect = {buttons_x + bw + gap, buttons_y, bw, bh};

      if (point_in_rect(mouse_sx, mouse_sy, sell_rect)) {
        const int ri = pending_sell_roster_index_;
        if (ri >= 0 && ri < static_cast<int>(game_state_.roster.size()) &&
            creatures_ && pending_sell_creature_ != INVALID_ID) {
          const std::uint32_t cslot = creatures_->getSlot(pending_sell_creature_);
          if (cslot != INVALID_ID &&
              cslot < static_cast<std::uint32_t>(creatures_->count)) {
            const float ccx = creatures_->x_positions[cslot] +
                              static_cast<float>(creatures_->widths[cslot]) * 0.5f;
            const float ccy =
                creatures_->y_positions[cslot] +
                static_cast<float>(creatures_->heights[cslot]) * 0.5f;
            const int col = std::clamp(static_cast<int>(ccx / kTileSizePx), 0,
                                       std::max(0, grid_cols_ - 1));
            const int row = std::clamp(static_cast<int>(ccy / kTileSizePx), 0,
                                       std::max(0, grid_rows_ - 1));
            const std::size_t idx =
                static_cast<std::size_t>(row * grid_cols_ + col);
            if (idx < cell_occupant_.size() &&
                cell_occupant_[idx] == pending_sell_creature_) {
              cell_occupant_[idx] = INVALID_ID;
            }
          }

          const int cost = game_state_.roster[static_cast<std::size_t>(ri)].seed_cost_essence;
          const int refund = std::max(
              0, static_cast<int>(
                     static_cast<float>(cost) * inter_level_shop::kSellRefundFraction));
          game_state_.essence += refund;
          game_state_.roster.erase(game_state_.roster.begin() + ri);

          if (ri >= 0 && ri < static_cast<int>(deployed_roster_.size())) {
            deployed_roster_.erase(deployed_roster_.begin() + ri);
          }

          for (std::uint32_t slot = 0;
               creatures_ && slot < static_cast<std::uint32_t>(creatures_->count);
               ++slot) {
            if ((creatures_->flags[slot] &
                 static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
              continue;
            }
            if (creatures_->roster_index[slot] > ri) {
              creatures_->roster_index[slot] -= 1;
            }
          }

          engine_destroy_entity(engine_, pending_sell_creature_,
                                creatures_->getTypeId());
          if (selected_roster_index_ >=
              static_cast<std::int32_t>(game_state_.roster.size())) {
            selected_roster_index_ =
                std::max(0, static_cast<int>(game_state_.roster.size()) - 1);
          }

          (void)SaveState::save(game_state_);
        }
        show_sell_confirm_ = false;
        pending_sell_creature_ = INVALID_ID;
        pending_sell_roster_index_ = -1;
      } else if (point_in_rect(mouse_sx, mouse_sy, cancel_rect)) {
        show_sell_confirm_ = false;
        pending_sell_creature_ = INVALID_ID;
        pending_sell_roster_index_ = -1;
      }
    }
  } else if (creatures_ && input.wasMousePressed(SDL_BUTTON_RIGHT) &&
             !mouse_over_ui) {
    const float pick_radius = creature_pick_radius_px();
    const auto &refs = engine_->grid.queryCircle(mouse_wx, mouse_wy, pick_radius);
    EntityHandle picked = INVALID_ID;
    for (const EntityRef &ref : refs) {
      if (static_cast<int>(ref.type) != creatures_->getTypeId()) {
        continue;
      }
      if (!engine_is_handle_valid(engine_, ref.index, creatures_->getTypeId())) {
        continue;
      }
      const std::uint32_t cslot = creatures_->getSlot(ref.index);
      if (cslot == INVALID_ID ||
          cslot >= static_cast<std::uint32_t>(creatures_->count)) {
        continue;
      }
      if ((creatures_->flags[cslot] &
           static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
        continue;
      }
      const float left = creatures_->x_positions[cslot];
      const float top = creatures_->y_positions[cslot];
      const float w = static_cast<float>(creatures_->widths[cslot]);
      const float h = static_cast<float>(creatures_->heights[cslot]);
      if (mouse_wx >= left && mouse_wx <= (left + w) && mouse_wy >= top &&
          mouse_wy <= (top + h)) {
        picked = ref.index;
        break;
      }
    }
    if (picked != INVALID_ID) {
      const int ri = creatures_->getRosterIndex(picked);
      if (ri >= 0 && ri < static_cast<int>(game_state_.roster.size())) {
        show_sell_confirm_ = true;
        pending_sell_creature_ = picked;
        pending_sell_roster_index_ = ri;
      }
    }
  }

  auto cancel_merge_drag = [&]() {
    if (!merge_drag_active_ || !creatures_ || merge_drag_source_ == INVALID_ID) {
      merge_drag_active_ = false;
      merge_drag_source_ = INVALID_ID;
      ghost_valid_ = false;
      ghost_active_ = false;
      ghost_world_x_ = 0.0f;
      ghost_world_y_ = 0.0f;
      return;
    }
    if (engine_is_handle_valid(engine_, merge_drag_source_,
                              creatures_->getTypeId())) {
      const std::uint32_t slot = creatures_->getSlot(merge_drag_source_);
      if (slot != INVALID_ID &&
          slot < static_cast<std::uint32_t>(creatures_->count)) {
        creatures_->flags[slot] |=
            static_cast<std::uint8_t>(EntityFlag::VISIBLE);
        creatures_->state[slot] = CreatureState::Idle;
      }
    }
    merge_drag_active_ = false;
    merge_drag_source_ = INVALID_ID;
    ghost_valid_ = false;
    ghost_active_ = false;
    ghost_world_x_ = 0.0f;
    ghost_world_y_ = 0.0f;
  };

  auto begin_merge_anim = [&](EntityHandle a, EntityHandle b,
                              EntityHandle target) -> bool {
    if (!creatures_ || !engine_ || a == INVALID_ID || b == INVALID_ID ||
        a == b || target == INVALID_ID) {
      return false;
    }
    if (!engine_is_handle_valid(engine_, a, creatures_->getTypeId()) ||
        !engine_is_handle_valid(engine_, b, creatures_->getTypeId()) ||
        !engine_is_handle_valid(engine_, target, creatures_->getTypeId())) {
      return false;
    }

    const std::uint32_t aslot = creatures_->getSlot(a);
    const std::uint32_t bslot = creatures_->getSlot(b);
    const std::uint32_t tslot = creatures_->getSlot(target);
    if (aslot == INVALID_ID || bslot == INVALID_ID || tslot == INVALID_ID ||
        aslot >= static_cast<std::uint32_t>(creatures_->count) ||
        bslot >= static_cast<std::uint32_t>(creatures_->count) ||
        tslot >= static_cast<std::uint32_t>(creatures_->count)) {
      return false;
    }

    const CharacterId cid = creatures_->character[aslot];
    const int tier_value = std::max(1, creatures_->tier[aslot]);
    if (creatures_->character[bslot] != cid ||
        std::max(1, creatures_->tier[bslot]) != tier_value) {
      return false;
    }
    const CreatureState astate = creatures_->state[aslot];
    const CreatureState bstate = creatures_->state[bslot];
    const CreatureState tstate = creatures_->state[tslot];
    if (!((astate == CreatureState::Idle) || (astate == CreatureState::Dragging)) ||
        bstate != CreatureState::Idle || tstate != CreatureState::Idle) {
      return false;
    }
    if ((creatures_->flags[bslot] &
         static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0 ||
        (creatures_->flags[tslot] &
         static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
      return false;
    }

    const float ahalf = static_cast<float>(creatures_->widths[aslot]) * 0.5f;
    const float bhalf = static_cast<float>(creatures_->widths[bslot]) * 0.5f;
    const float thalf = static_cast<float>(creatures_->widths[tslot]) * 0.5f;
    const float acx = creatures_->x_positions[aslot] + ahalf;
    const float acy = creatures_->y_positions[aslot] + ahalf;
    const float bcx = creatures_->x_positions[bslot] + bhalf;
    const float bcy = creatures_->y_positions[bslot] + bhalf;
    const float tcx = creatures_->x_positions[tslot] + thalf;
    const float tcy = creatures_->y_positions[tslot] + thalf;

    const int acol = std::clamp(static_cast<int>(acx / kTileSizePx), 0,
                                std::max(0, grid_cols_ - 1));
    const int arow = std::clamp(static_cast<int>(acy / kTileSizePx), 0,
                                std::max(0, grid_rows_ - 1));
    const int bcol = std::clamp(static_cast<int>(bcx / kTileSizePx), 0,
                                std::max(0, grid_cols_ - 1));
    const int brow = std::clamp(static_cast<int>(bcy / kTileSizePx), 0,
                                std::max(0, grid_rows_ - 1));
    const int tcol = std::clamp(static_cast<int>(tcx / kTileSizePx), 0,
                                std::max(0, grid_cols_ - 1));
    const int trow = std::clamp(static_cast<int>(tcy / kTileSizePx), 0,
                                std::max(0, grid_rows_ - 1));

    if (std::abs(acol - bcol) > 1 || std::abs(arow - brow) > 1 ||
        (acol == bcol && arow == brow)) {
      return false;
    }

    const std::size_t a_idx = static_cast<std::size_t>(arow * grid_cols_ + acol);
    const std::size_t b_idx = static_cast<std::size_t>(brow * grid_cols_ + bcol);
    const std::size_t t_idx = static_cast<std::size_t>(trow * grid_cols_ + tcol);
    if (a_idx >= cell_occupant_.size() || b_idx >= cell_occupant_.size() ||
        t_idx >= cell_occupant_.size()) {
      return false;
    }

    const int ri_a = creatures_->roster_index[aslot];
    const int ri_b = creatures_->roster_index[bslot];
    if (ri_a < 0 || ri_b < 0 || ri_a == ri_b) {
      return false;
    }

    const int keep = std::min(ri_a, ri_b);
    const int remove = std::max(ri_a, ri_b);

    const int kills_a = std::max(0, creatures_->kills[aslot]);
    const int kills_b = std::max(0, creatures_->kills[bslot]);
    int new_kills = static_cast<int>(std::floor(
        static_cast<float>(kills_a + kills_b) / merge::kKillInheritanceDivisor));
    const int new_tier = tier_value + 1;
    if (game_state_.isRelicEquipped(RelicId::MergersGift)) {
      int need = evolution::killsNeededForNextTier(new_tier);
      if (need > 0) {
        const float mult =
            std::clamp(game_state_.rapidGrowthKillThresholdMultiplier(), 0.01f, 1.0f);
        need = std::max(
            1, static_cast<int>(std::floor(static_cast<float>(need) * mult)));
      }
      const int min_inherited =
          (need > 0)
              ? std::max(
                    0, static_cast<int>(std::floor(
                           static_cast<double>(need) *
                           static_cast<double>(relics::kMergersGiftProgressInheritance))))
              : 0;
      new_kills = std::max(new_kills, min_inherited);
    }

    creatures_->flags[aslot] &= ~static_cast<std::uint8_t>(EntityFlag::VISIBLE);
    creatures_->flags[bslot] &= ~static_cast<std::uint8_t>(EntityFlag::VISIBLE);
    creatures_->state[aslot] = CreatureState::Merging;
    creatures_->state[bslot] = CreatureState::Merging;
    creatures_->state_time_sec[aslot] = merge::kAnimationSec;
    creatures_->state_time_sec[bslot] = merge::kAnimationSec;

    merge_anim_ = MergeAnim{};
    merge_anim_.active = true;
    merge_anim_.elapsed_sec = 0.0f;
    merge_anim_.a = a;
    merge_anim_.b = b;
    merge_anim_.character = cid;
    merge_anim_.a_cx = acx;
    merge_anim_.a_cy = acy;
    merge_anim_.b_cx = bcx;
    merge_anim_.b_cy = bcy;
    merge_anim_.target_cx = tcx;
    merge_anim_.target_cy = tcy;
    merge_anim_.keep_roster_index = keep;
    merge_anim_.remove_roster_index = remove;
    merge_anim_.new_tier = new_tier;
    merge_anim_.new_kills = std::max(0, new_kills);
    merge_anim_.cell_a_idx = a_idx;
    merge_anim_.cell_b_idx = b_idx;
    merge_anim_.target_cell_idx = t_idx;

    merge_drag_active_ = false;
    merge_drag_source_ = INVALID_ID;
    auto_merge_idle_sec_ = 0.0f;
    selected_creature_ = INVALID_ID;
    ghost_valid_ = false;
    ghost_active_ = false;
    ghost_world_x_ = 0.0f;
    ghost_world_y_ = 0.0f;
    return true;
  };

  auto complete_merge_anim = [&]() {
    if (!merge_anim_.active) {
      return;
    }

    merge_anim_.active = false;

    if (!creatures_ || !engine_) {
      merge_anim_ = MergeAnim{};
      return;
    }

    const int keep = merge_anim_.keep_roster_index;
    const int remove = merge_anim_.remove_roster_index;
    if (keep < 0 || remove < 0 || keep == remove ||
        keep >= static_cast<int>(game_state_.roster.size()) ||
        remove >= static_cast<int>(game_state_.roster.size()) ||
        keep >= static_cast<int>(deployed_roster_.size()) ||
        remove >= static_cast<int>(deployed_roster_.size())) {
      merge_anim_ = MergeAnim{};
      return;
    }

    if (engine_is_handle_valid(engine_, merge_anim_.a, creatures_->getTypeId())) {
      engine_destroy_entity(engine_, merge_anim_.a, creatures_->getTypeId());
    }
    if (engine_is_handle_valid(engine_, merge_anim_.b, creatures_->getTypeId())) {
      engine_destroy_entity(engine_, merge_anim_.b, creatures_->getTypeId());
    }

    if (merge_anim_.cell_a_idx < cell_occupant_.size() &&
        cell_occupant_[merge_anim_.cell_a_idx] == merge_anim_.a) {
      cell_occupant_[merge_anim_.cell_a_idx] = INVALID_ID;
    }
    if (merge_anim_.cell_b_idx < cell_occupant_.size() &&
        cell_occupant_[merge_anim_.cell_b_idx] == merge_anim_.b) {
      cell_occupant_[merge_anim_.cell_b_idx] = INVALID_ID;
    }

    game_state_.roster[static_cast<std::size_t>(keep)].tier =
        std::max(1, merge_anim_.new_tier);
    game_state_.roster[static_cast<std::size_t>(keep)].kills =
        std::max(0, merge_anim_.new_kills);

    game_state_.roster.erase(game_state_.roster.begin() + remove);
    deployed_roster_.erase(deployed_roster_.begin() + remove);

    for (std::uint32_t slot = 0;
         slot < static_cast<std::uint32_t>(creatures_->count); ++slot) {
      if ((creatures_->flags[slot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
        continue;
      }
      if (creatures_->roster_index[slot] > remove) {
        creatures_->roster_index[slot] -= 1;
      }
    }

    if (selected_roster_index_ > remove) {
      selected_roster_index_ -= 1;
    } else if (selected_roster_index_ == remove) {
      selected_roster_index_ = keep;
    }
    selected_roster_index_ =
        std::clamp(selected_roster_index_, 0,
                   std::max(0, static_cast<int>(game_state_.roster.size()) - 1));

    const std::size_t t_idx = merge_anim_.target_cell_idx;
    if (t_idx < cell_occupant_.size() && grid_cols_ > 0) {
      const int col = static_cast<int>(t_idx % static_cast<std::size_t>(grid_cols_));
      const int row = static_cast<int>(t_idx / static_cast<std::size_t>(grid_cols_));
      const RosterEntry &re =
          game_state_.roster[static_cast<std::size_t>(keep)];
      float x = 0.0f;
      float y = 0.0f;
      creature_top_left_for_cell(col, row, re.tier, x, y);
      const EntityHandle handle =
          creatures_->createCreature(x, y, re.character, re.tier, re.kills, keep);
      deployed_roster_[static_cast<std::size_t>(keep)] = handle;
      if (handle != INVALID_ID) {
        cell_occupant_[t_idx] = handle;
        selected_creature_ = handle;
        game_state_.essence += merge::kEssenceBonus;
        game_state_.essence_earned_this_level += merge::kEssenceBonus;
        game_state_.merges_this_level += 1;
        game_state_.lifetime_merges = std::max(0, game_state_.lifetime_merges + 1);
        game_state_.recomputeMetaProgression();

        bool started_recursive = false;
        if (game_state_.isRelicEquipped(RelicId::RecursiveMerge)) {
          static std::uint32_t rng = 0xC0DECAFEu;
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          const float roll = static_cast<float>(rng) / 4294967295.0f;
          if (roll < relics::kRecursiveMergeSecondMergeChance) {
            EntityHandle neighbor = INVALID_ID;
            const CharacterId cid = re.character;
            const int tier_value = std::max(1, re.tier);

            for (int dy = -1; dy <= 1 && neighbor == INVALID_ID; ++dy) {
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) {
                  continue;
                }
                const int nc = col + dx;
                const int nr = row + dy;
                if (nc < 0 || nr < 0 || nc >= grid_cols_ || nr >= grid_rows_) {
                  continue;
                }
                const std::size_t nidx =
                    static_cast<std::size_t>(nr * grid_cols_ + nc);
                if (nidx >= cell_occupant_.size()) {
                  continue;
                }
                const EntityHandle cand = cell_occupant_[nidx];
                if (cand == INVALID_ID || cand == handle ||
                    !engine_is_handle_valid(engine_, cand, creatures_->getTypeId())) {
                  continue;
                }
                const std::uint32_t cslot = creatures_->getSlot(cand);
                if (cslot == INVALID_ID ||
                    cslot >= static_cast<std::uint32_t>(creatures_->count)) {
                  continue;
                }
                if ((creatures_->flags[cslot] &
                     static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
                  continue;
                }
                if (creatures_->state[cslot] != CreatureState::Idle) {
                  continue;
                }
                if (creatures_->character[cslot] != cid ||
                    std::max(1, creatures_->tier[cslot]) != tier_value) {
                  continue;
                }
                neighbor = cand;
                break;
              }
            }

            if (neighbor != INVALID_ID) {
              started_recursive = begin_merge_anim(handle, neighbor, handle);
            }
          }
        }
        if (started_recursive) {
          merge_cooldown_remaining_sec_ = 0.0f;
          return;
        }
      } else {
        cell_occupant_[t_idx] = INVALID_ID;
        selected_creature_ = INVALID_ID;
      }
    }
    merge_cooldown_remaining_sec_ = game_state_.synthesisMergeCooldownSec();

    merge_anim_ = MergeAnim{};
  };

  if (merge_anim_.active && merge_anim_.elapsed_sec >= merge::kAnimationSec) {
    complete_merge_anim();
  }

  merge_pairs_.clear();
  if (!merge_anim_.active && allow_gameplay_input &&
      level_state == LevelManagerState::WaveClear && !show_sell_confirm_ &&
      !show_level_select_ && creatures_ && grid_cols_ > 0 && grid_rows_ > 0 &&
      !cell_occupant_.empty()) {
    for (int row = 0; row < grid_rows_; ++row) {
      for (int col = 0; col < grid_cols_; ++col) {
        const std::size_t idx =
            static_cast<std::size_t>(row * grid_cols_ + col);
        if (idx >= cell_occupant_.size()) {
          continue;
        }
        const EntityHandle a = cell_occupant_[idx];
        if (a == INVALID_ID ||
            !engine_is_handle_valid(engine_, a, creatures_->getTypeId())) {
          continue;
        }
        const std::uint32_t aslot = creatures_->getSlot(a);
        if (aslot == INVALID_ID ||
            aslot >= static_cast<std::uint32_t>(creatures_->count)) {
          continue;
        }
        if ((creatures_->flags[aslot] &
             static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
          continue;
        }
        if (creatures_->state[aslot] != CreatureState::Idle) {
          continue;
        }
        const CharacterId cid = creatures_->character[aslot];
        const int t = std::max(1, creatures_->tier[aslot]);

        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
              continue;
            }
            const int nc = col + dx;
            const int nr = row + dy;
            if (nc < 0 || nr < 0 || nc >= grid_cols_ || nr >= grid_rows_) {
              continue;
            }
            const std::size_t nidx =
                static_cast<std::size_t>(nr * grid_cols_ + nc);
            if (nidx <= idx || nidx >= cell_occupant_.size()) {
              continue;
            }

            const EntityHandle b = cell_occupant_[nidx];
            if (b == INVALID_ID ||
                !engine_is_handle_valid(engine_, b, creatures_->getTypeId())) {
              continue;
            }
            const std::uint32_t bslot = creatures_->getSlot(b);
            if (bslot == INVALID_ID ||
                bslot >= static_cast<std::uint32_t>(creatures_->count)) {
              continue;
            }
            if ((creatures_->flags[bslot] &
                 static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
              continue;
            }
            if (creatures_->state[bslot] != CreatureState::Idle) {
              continue;
            }
            if (creatures_->character[bslot] != cid ||
                std::max(1, creatures_->tier[bslot]) != t) {
              continue;
            }
            const int ri_a = creatures_->roster_index[aslot];
            const int ri_b = creatures_->roster_index[bslot];
            if (ri_a < 0 || ri_b < 0 || ri_a == ri_b) {
              continue;
            }
            merge_pairs_.push_back({a, b});
          }
        }
      }
    }
  } else {
    auto_merge_idle_sec_ = 0.0f;
  }

  if (!merge_anim_.active && !merge_drag_active_ &&
      merge_cooldown_remaining_sec_ <= 0.0f && allow_gameplay_input &&
      level_state == LevelManagerState::WaveClear && !show_sell_confirm_ &&
      !show_level_select_ && creatures_) {
    if (merge_pairs_.empty()) {
      auto_merge_idle_sec_ = 0.0f;
    } else {
      auto_merge_idle_sec_ = std::min(
          merge::kAutoMergeIdleSec,
          auto_merge_idle_sec_ + std::max(0.0f, dt));
      if (auto_merge_idle_sec_ >= merge::kAutoMergeIdleSec) {
        const EntityHandle a = merge_pairs_.front().first;
        const EntityHandle b = merge_pairs_.front().second;
        (void)begin_merge_anim(a, b, b);
        auto_merge_idle_sec_ = 0.0f;
      }
    }
  }

  if (merge_drag_active_ && creatures_) {
    int drag_col = -1;
    int drag_row = -1;
    bool drag_drop_valid = false;
    ghost_active_ = false;
    ghost_valid_ = false;

    if (merge_drag_source_ != INVALID_ID &&
        engine_is_handle_valid(engine_, merge_drag_source_, creatures_->getTypeId()) &&
        grid_cols_ > 0 && grid_rows_ > 0) {
      const std::uint32_t sslot = creatures_->getSlot(merge_drag_source_);
      if (sslot != INVALID_ID &&
          sslot < static_cast<std::uint32_t>(creatures_->count)) {
        const int tier = std::max(1, creatures_->tier[sslot]);
        drag_col = std::clamp(static_cast<int>(mouse_wx / kTileSizePx), 0,
                              std::max(0, grid_cols_ - 1));
        drag_row = std::clamp(static_cast<int>(mouse_wy / kTileSizePx), 0,
                              std::max(0, grid_rows_ - 1));

        creature_top_left_for_cell(drag_col, drag_row, tier, ghost_world_x_,
                                   ghost_world_y_);
        ghost_active_ = true;

        const std::size_t idx =
            static_cast<std::size_t>(drag_row * grid_cols_ + drag_col);
        if (idx < cell_occupant_.size()) {
          const EntityHandle occ = cell_occupant_[idx];
          drag_drop_valid = (occ == INVALID_ID || occ == merge_drag_source_);
        }

        float base_cx = static_cast<float>(kWorldWidthPx) * 0.5f;
        float base_cy = static_cast<float>(kWorldHeightPx) * 0.5f;
        if (base_ && base_id_ != INVALID_ID) {
          const std::uint32_t bslot = base_->getSlot(base_id_);
          if (bslot != INVALID_ID &&
              bslot < static_cast<std::uint32_t>(base_->count)) {
            const float half = static_cast<float>(base_->widths[bslot]) * 0.5f;
            base_cx = base_->x_positions[bslot] + half;
            base_cy = base_->y_positions[bslot] + half;
          }
        }

        const float size_px = static_cast<float>(creature_size_px_for_tier(tier));
        const float avoid_r = kBaseRadiusPx + size_px * 0.5f * 1.25f;
        const float avoid_r2 = avoid_r * avoid_r;
        const float cell_cx =
            static_cast<float>(drag_col * kTileSizePx) + static_cast<float>(kTileSizePx) * 0.5f;
        const float cell_cy =
            static_cast<float>(drag_row * kTileSizePx) + static_cast<float>(kTileSizePx) * 0.5f;
        const float dx = cell_cx - base_cx;
        const float dy = cell_cy - base_cy;
        if (dx * dx + dy * dy <= avoid_r2) {
          drag_drop_valid = false;
        }

        if (path_grid_.cols() == grid_cols_ && path_grid_.rows() == grid_rows_ &&
            !path_grid_.isWalkable(drag_col, drag_row)) {
          drag_drop_valid = false;
        }
      }
    }

    ghost_valid_ = drag_drop_valid;

    if (input.wasPressed(SDL_SCANCODE_ESCAPE)) {
      cancel_merge_drag();
    } else if (input.wasMouseReleased(SDL_BUTTON_LEFT)) {
      EntityHandle drop = INVALID_ID;
      const float pick_radius = creature_pick_radius_px();
      const auto &refs =
          engine_->grid.queryCircle(mouse_wx, mouse_wy, pick_radius);
      for (const EntityRef &ref : refs) {
        if (static_cast<int>(ref.type) != creatures_->getTypeId()) {
          continue;
        }
        if (ref.index == merge_drag_source_) {
          continue;
        }
        if (!engine_is_handle_valid(engine_, ref.index, creatures_->getTypeId())) {
          continue;
        }
        const std::uint32_t cslot = creatures_->getSlot(ref.index);
        if (cslot == INVALID_ID ||
            cslot >= static_cast<std::uint32_t>(creatures_->count)) {
          continue;
        }
        if ((creatures_->flags[cslot] &
             static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
          continue;
        }
        const float left = creatures_->x_positions[cslot];
        const float top = creatures_->y_positions[cslot];
        const float w = static_cast<float>(creatures_->widths[cslot]);
        const float h = static_cast<float>(creatures_->heights[cslot]);
        if (mouse_wx >= left && mouse_wx <= (left + w) && mouse_wy >= top &&
            mouse_wy <= (top + h)) {
          drop = ref.index;
          break;
        }
      }

      const bool can_merge_drop =
          (level_state == LevelManagerState::WaveClear) &&
          drop != INVALID_ID && merge_cooldown_remaining_sec_ <= 0.0f;
      if (can_merge_drop && begin_merge_anim(merge_drag_source_, drop, drop)) {
        merge_drag_active_ = false;
        merge_drag_source_ = INVALID_ID;
        ghost_valid_ = false;
        ghost_active_ = false;
        ghost_world_x_ = 0.0f;
        ghost_world_y_ = 0.0f;
      } else if (drag_drop_valid && drag_col >= 0 && drag_row >= 0 &&
                 creatures_->moveToCell(merge_drag_source_, drag_col, drag_row)) {
        const std::uint32_t sslot = creatures_->getSlot(merge_drag_source_);
        if (sslot != INVALID_ID &&
            sslot < static_cast<std::uint32_t>(creatures_->count)) {
          creatures_->flags[sslot] |= static_cast<std::uint8_t>(EntityFlag::VISIBLE);
          creatures_->state[sslot] = CreatureState::Idle;
        }
        creatures_->ensureAttackCooldownAtLeast(merge_drag_source_,
                                                movement_ai::kPlayerDragStunSec);
        selected_creature_ = merge_drag_source_;
        merge_drag_active_ = false;
        merge_drag_source_ = INVALID_ID;
        ghost_valid_ = false;
        ghost_active_ = false;
        ghost_world_x_ = 0.0f;
        ghost_world_y_ = 0.0f;
      } else {
        cancel_merge_drag();
      }
    }
  }

  if (!merge_anim_.active && !merge_drag_active_ && allow_gameplay_input &&
      !show_sell_confirm_ && !show_level_select_ && creatures_) {
    if (drag_candidate_ != INVALID_ID) {
      if (!input.isMouseDown(SDL_BUTTON_LEFT) ||
          input.wasMouseReleased(SDL_BUTTON_LEFT) ||
          input.wasPressed(SDL_SCANCODE_ESCAPE)) {
        drag_candidate_ = INVALID_ID;
      } else {
        const float dx = mouse_wx - drag_candidate_start_wx_;
        const float dy = mouse_wy - drag_candidate_start_wy_;
        const float thresh = movement_ai::kPlayerDragStartThresholdPx;
        if (dx * dx + dy * dy >= thresh * thresh) {
          if (engine_is_handle_valid(engine_, drag_candidate_,
                                    creatures_->getTypeId())) {
            const std::uint32_t slot = creatures_->getSlot(drag_candidate_);
            if (slot != INVALID_ID &&
                slot < static_cast<std::uint32_t>(creatures_->count) &&
                (creatures_->flags[slot] &
                 static_cast<std::uint8_t>(EntityFlag::VISIBLE)) != 0) {
              merge_drag_active_ = true;
              merge_drag_source_ = drag_candidate_;
              drag_candidate_ = INVALID_ID;
              creatures_->flags[slot] &=
                  ~static_cast<std::uint8_t>(EntityFlag::VISIBLE);
              creatures_->state[slot] = CreatureState::Dragging;
              selected_creature_ = merge_drag_source_;
              auto_merge_idle_sec_ = 0.0f;
            } else {
              drag_candidate_ = INVALID_ID;
            }
          } else {
            drag_candidate_ = INVALID_ID;
          }
        }
      }
    }
  } else {
    drag_candidate_ = INVALID_ID;
  }

  bool placed_this_click = false;
  if (!merge_drag_active_) {
    ghost_valid_ = false;
    ghost_active_ = false;
  }
  if (!merge_anim_.active && !merge_drag_active_ && !show_sell_confirm_ &&
      !mouse_over_ui && creatures_ &&
      !game_state_.roster.empty() && !deployed_roster_.empty()) {
    selected_roster_index_ =
        std::clamp(selected_roster_index_, 0,
                   static_cast<std::int32_t>(deployed_roster_.size()) - 1);
    if (deployed_roster_[static_cast<std::size_t>(selected_roster_index_)] ==
        INVALID_ID) {
      const int col = std::clamp(static_cast<int>(mouse_wx / kTileSizePx), 0,
                                 std::max(0, grid_cols_ - 1));
      const int row = std::clamp(static_cast<int>(mouse_wy / kTileSizePx), 0,
                                 std::max(0, grid_rows_ - 1));
      const std::size_t idx =
          static_cast<std::size_t>(row * grid_cols_ + col);

      const RosterEntry &re =
          game_state_.roster[static_cast<std::size_t>(selected_roster_index_)];
      creature_top_left_for_cell(col, row, re.tier, ghost_world_x_, ghost_world_y_);
      ghost_active_ = true;

      ghost_valid_ = (idx < cell_occupant_.size()) && (cell_occupant_[idx] == INVALID_ID);

      if (ghost_valid_ && input.wasMousePressed(SDL_BUTTON_LEFT) &&
          !wave_shop_consumed_left_click) {
        const EntityHandle handle = creatures_->createCreature(
            ghost_world_x_, ghost_world_y_, re.character, re.tier, re.kills,
            selected_roster_index_);
        if (handle != INVALID_ID) {
          deployed_roster_[static_cast<std::size_t>(selected_roster_index_)] = handle;
          selected_creature_ = handle;
          if (idx < cell_occupant_.size()) {
            cell_occupant_[idx] = handle;
          }
          placed_this_click = true;
        }
      }
    }
  }

  if (!merge_anim_.active && !merge_drag_active_ && allow_gameplay_input &&
      !show_sell_confirm_ && !show_level_select_ &&
      input.wasMousePressed(SDL_BUTTON_LEFT) && !mouse_over_ui &&
      !wave_shop_consumed_left_click && !placed_this_click && creatures_) {
    const float pick_radius = creature_pick_radius_px();
    const auto &refs =
        engine_->grid.queryCircle(mouse_wx, mouse_wy, pick_radius);
    EntityHandle picked = INVALID_ID;
    for (const EntityRef &ref : refs) {
      if (static_cast<int>(ref.type) != creatures_->getTypeId()) {
        continue;
      }
      if (!engine_is_handle_valid(engine_, ref.index, creatures_->getTypeId())) {
        continue;
      }
      const std::uint32_t cslot = creatures_->getSlot(ref.index);
      if (cslot == INVALID_ID ||
          cslot >= static_cast<std::uint32_t>(creatures_->count)) {
        continue;
      }
      if ((creatures_->flags[cslot] &
           static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
        continue;
      }
      const float left = creatures_->x_positions[cslot];
      const float top = creatures_->y_positions[cslot];
      const float w = static_cast<float>(creatures_->widths[cslot]);
      const float h = static_cast<float>(creatures_->heights[cslot]);
      if (mouse_wx >= left && mouse_wx <= (left + w) && mouse_wy >= top &&
          mouse_wy <= (top + h)) {
        picked = ref.index;
        break;
      }
    }
    selected_creature_ = picked;
    if (picked != INVALID_ID) {
      drag_candidate_ = picked;
      drag_candidate_start_wx_ = mouse_wx;
      drag_candidate_start_wy_ = mouse_wy;
    } else {
      drag_candidate_ = INVALID_ID;
    }
  }

  float level_step = dt;
  if (level_state == LevelManagerState::WaveClear &&
      (merge_drag_active_ || merge_anim_.active)) {
    level_step = 0.0f;
  }
  level_manager_.tick(level_step);

  const LevelManagerState new_level_state = level_manager_.state();
  if (new_level_state != last_level_state_) {
    if (new_level_state == LevelManagerState::WaveClear &&
        last_level_state_ == LevelManagerState::Playing) {
      const std::uint32_t level_n = static_cast<std::uint32_t>(
          std::max<std::int32_t>(1, level_manager_.levelDef().level_number));
      const std::uint32_t wave_n = static_cast<std::uint32_t>(
          std::max<std::int32_t>(0, level_manager_.waveIndex()));
      const std::uint32_t seed =
          0xC0FFEEu ^ (level_n * 2654435761u) ^ (wave_n * 1597334677u);
      wave_buff_shop_.open(seed);
    }

    if (new_level_state == LevelManagerState::Playing &&
        last_level_state_ == LevelManagerState::WaveClear) {
      wave_buff_shop_.close();

      if (game_state_.hasWaveBuff(WaveBuffId::ApexHunter) && creatures_) {
        std::int32_t best_ri = -1;
        int best_kills = -1;
        for (std::uint32_t slot = 0;
             slot < static_cast<std::uint32_t>(creatures_->count); ++slot) {
          if ((creatures_->flags[slot] &
               static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          const int ri = creatures_->roster_index[slot];
          if (ri < 0) {
            continue;
          }
          const int k = creatures_->kills[slot];
          if (k > best_kills) {
            best_kills = k;
            best_ri = ri;
          }
        }
        game_state_.apex_hunter_roster_index = best_ri;
      }
    }

    if (new_level_state == LevelManagerState::LevelClear) {
      openInterLevel();
      (void)SaveState::save(game_state_);
    }
    if (new_level_state == LevelManagerState::Failed ||
        new_level_state == LevelManagerState::LevelClear) {
      show_sell_confirm_ = false;
      pending_sell_creature_ = INVALID_ID;
      pending_sell_roster_index_ = -1;
      show_level_select_ = false;
      merge_drag_active_ = false;
      merge_drag_source_ = INVALID_ID;
      drag_candidate_ = INVALID_ID;
      merge_pairs_.clear();
      merge_anim_ = MergeAnim{};
      merge_cooldown_remaining_sec_ = 0.0f;
      auto_merge_idle_sec_ = 0.0f;
      if (creatures_) {
        for (std::uint32_t slot = 0;
             slot < static_cast<std::uint32_t>(creatures_->count); ++slot) {
          creatures_->flags[slot] |=
              static_cast<std::uint8_t>(EntityFlag::VISIBLE);
          if (creatures_->state[slot] == CreatureState::Dragging ||
              creatures_->state[slot] == CreatureState::Merging) {
            creatures_->state[slot] = CreatureState::Idle;
            creatures_->state_time_sec[slot] = 0.0f;
          }
        }
      }
      wave_buff_shop_.close();
      ghost_valid_ = false;
      ghost_active_ = false;
      ghost_world_x_ = 0.0f;
      ghost_world_y_ = 0.0f;
    }
    last_level_state_ = new_level_state;
  }
  camera_.tick(dt, input, engine_, kWorldWidthPx, kWorldHeightPx);
}

void TowerSwarmGame::renderHUD(const InputManager &input) {
  if (!engine_) {
    return;
  }
  const int hp = (base_id_ != INVALID_ID && base_) ? base_->getHp(base_id_) : 0;
  const int hp_max =
      (base_id_ != INVALID_ID && base_) ? base_->getHpMax(base_id_) : 0;
  hud_.render(engine_, camera_, input, show_debug_grid_, hp, hp_max);

  SDL_Renderer *r = engine_->renderer;
  if (!r) {
    return;
  }

  SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);

  if (show_main_menu_) {
    SDL_FRect overlay = {0.0f, 0.0f, engine_->camera.width, engine_->camera.height};
    set_color(r, kModalOverlayColor);
    SDL_RenderFillRect(r, &overlay);

    const float panel_w = 560.0f;
    const float panel_h = 260.0f;
    const float panel_x = (engine_->camera.width - panel_w) * 0.5f;
    const float panel_y = (engine_->camera.height - panel_h) * 0.5f;
    SDL_FRect panel = {panel_x, panel_y, panel_w, panel_h};
    set_color(r, kModalPanelColor);
    SDL_RenderFillRect(r, &panel);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &panel);

    set_color(r, kModalButtonTextColor);
    float tx = panel_x + static_cast<float>(kModalPanelTextInsetXPx);
    float ty = panel_y + static_cast<float>(kModalPanelTextInsetYPx);
    SDL_RenderDebugTextFormat(r, tx, ty, "TOWER SWARM");
    ty += static_cast<float>(kModalPanelLineStepPx) * 2.0f;

    SDL_RenderDebugTextFormat(r, tx, ty, "Enter / Space / Click: Play");
    ty += static_cast<float>(kModalPanelLineStepPx);
    SDL_RenderDebugTextFormat(r, tx, ty, "N: New Profile");
    ty += static_cast<float>(kModalPanelLineStepPx);
    SDL_RenderDebugTextFormat(r, tx, ty, "Esc: Quit");
    ty += static_cast<float>(kModalPanelLineStepPx) * 2.0f;

    SDL_RenderDebugTextFormat(
        r, tx, ty, "Profile: Level %d | Essence %d | Max Level %d",
        std::max(1, game_state_.level_number), std::max(0, game_state_.essence),
        std::max(1, game_state_.max_level_reached));

    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_NONE);
    return;
  }

  float cam_x1 = 0.0f;
  float cam_y1 = 0.0f;
  float cam_x2 = 0.0f;
  float cam_y2 = 0.0f;
  get_camera_world_rect(engine_, cam_x1, cam_y1, cam_x2, cam_y2);
  const float zoom = safe_zoom(engine_);

  const LevelManagerState level_state = level_manager_.state();

  if (!show_inter_level_ && level_state != LevelManagerState::Failed &&
      level_state != LevelManagerState::LevelClear &&
      !game_state_.active_wave_buffs.empty()) {
    const float icon = 32.0f;
    const float gap = 6.0f;
    const float pad = static_cast<float>(kHudPaddingPx);
    const float y = std::max(
        0.0f,
        (static_cast<float>(kHudTopBarHeightPx) - icon) * 0.5f);

    float x = engine_->camera.width - pad - icon;
    for (const ActiveWaveBuff &b : game_state_.active_wave_buffs) {
      if (b.remaining_waves <= 0) {
        continue;
      }
      SDL_FRect rr = {x, y, icon, icon};
      set_color(r, WaveBuffShop::iconColor(b.id));
      SDL_RenderFillRect(r, &rr);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &rr);

      const char *glyph = WaveBuffShop::iconGlyph(b.id);
      set_color(r, Rgba8{0, 0, 0, 200});
      SDL_RenderDebugTextFormat(r, rr.x + 7.0f, rr.y + 11.0f, "%s", glyph);
      set_color(r, kModalButtonTextColor);
      SDL_RenderDebugTextFormat(r, rr.x + 6.0f, rr.y + 10.0f, "%s", glyph);

      set_color(r, Rgba8{0, 0, 0, 200});
      SDL_RenderDebugTextFormat(r, rr.x + rr.w - 10.0f, rr.y + rr.h - 12.0f,
                                "%d", b.remaining_waves);
      set_color(r, kHudTextColor);
      SDL_RenderDebugTextFormat(r, rr.x + rr.w - 11.0f, rr.y + rr.h - 13.0f,
                                "%d", b.remaining_waves);

      x -= icon + gap;
      if (x < pad) {
        break;
      }
    }
  }

  if (!show_sell_confirm_ && !show_level_select_ &&
      level_state != LevelManagerState::Failed &&
      level_state != LevelManagerState::LevelClear &&
      ghost_active_) {
    const float sx = (ghost_world_x_ - cam_x1) * zoom;
    const float sy = (ghost_world_y_ - cam_y1) * zoom;
    int ghost_size_px = kCreatureBaseSizePx;
    if (selected_roster_index_ >= 0 &&
        static_cast<std::size_t>(selected_roster_index_) <
            game_state_.roster.size()) {
      const RosterEntry &re = game_state_.roster[static_cast<std::size_t>(
          selected_roster_index_)];
      ghost_size_px = creature_size_px_for_tier(re.tier);
    }
    const float sw = static_cast<float>(ghost_size_px) * zoom;
    const float sh = static_cast<float>(ghost_size_px) * zoom;
    SDL_FRect rect = {sx, sy, sw, sh};
    set_color(r, ghost_valid_ ? kGhostValidColor : kGhostInvalidColor);
    SDL_RenderFillRect(r, &rect);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &rect);
  }

  if (!show_sell_confirm_ && !show_level_select_ &&
      level_state != LevelManagerState::Failed &&
      level_state != LevelManagerState::LevelClear) {
    if (enemies_) {
      struct WraithReveal final {
        float cx{0.0f};
        float cy{0.0f};
        float r2{0.0f};
      };
      std::array<WraithReveal, 64> wraiths{};
      std::size_t wraith_count = 0;
      const float wraith_mult =
          game_state_.isRelicEquipped(RelicId::VoidLens)
              ? std::max(1.0f, relics::kVoidLensHpBarRevealRangeMultiplier)
              : 1.0f;
      if (creatures_) {
        for (std::uint32_t cslot = 0;
             cslot < static_cast<std::uint32_t>(creatures_->count); ++cslot) {
          if ((creatures_->flags[cslot] &
               static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          if (creatures_->character[cslot] != CharacterId::Wraith) {
            continue;
          }
          if (wraith_count >= wraiths.size()) {
            break;
          }
          const float half = static_cast<float>(creatures_->widths[cslot]) * 0.5f;
          const float cx = creatures_->x_positions[cslot] + half;
          const float cy = creatures_->y_positions[cslot] + half;
          const float r_world = std::max(0.0f, creatures_->attack_range_px[cslot] *
                                                   wraith_mult);
          wraiths[wraith_count++] =
              WraithReveal{cx, cy, r_world * r_world};
        }
      }

      for (std::uint32_t slot = 0;
           slot < static_cast<std::uint32_t>(enemies_->count); ++slot) {
        if ((enemies_->flags[slot] &
             static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
          continue;
        }

        const float ex = enemies_->x_positions[slot];
        const float ey = enemies_->y_positions[slot];
        const float ew = static_cast<float>(enemies_->widths[slot]);
        const float eh = static_cast<float>(enemies_->heights[slot]);

        if (ex > cam_x2 || (ex + ew) < cam_x1 || ey > cam_y2 ||
            (ey + eh) < cam_y1) {
          continue;
        }

        const float hp_max_f = std::max(1.0f, enemies_->hp_max[slot]);
        const float frac = std::clamp(enemies_->hp[slot] / hp_max_f, 0.0f, 1.0f);

        const float bar_w_world = ew;
        const float bar_h_world = static_cast<float>(kEnemyHpBarHeightPx);
        const float bar_x_world = ex;
        const float bar_y_world =
            ey - static_cast<float>(kEnemyHpBarOffsetYPx) - bar_h_world;

        const float sx = (bar_x_world - cam_x1) * zoom;
        const float sy = (bar_y_world - cam_y1) * zoom;
        const float sw = bar_w_world * zoom;
        const float sh = bar_h_world * zoom;

        SDL_FRect outer = {sx, sy, sw, sh};
        set_color(r, kEnemyHpBarBackColor);
        SDL_RenderFillRect(r, &outer);

        const float inset = static_cast<float>(kEnemyHpBarInsetPx) * zoom;
        SDL_FRect fill = {sx + inset, sy + inset,
                          std::max(0.0f, sw - 2.0f * inset) * frac,
                          std::max(0.0f, sh - 2.0f * inset)};
        set_color(r, kEnemyHpBarFillColor);
        SDL_RenderFillRect(r, &fill);

        set_color(r, kEnemyHpBarOutlineColor);
        SDL_RenderRect(r, &outer);

        bool reveal_hp = false;
        if (wraith_count > 0) {
          const float ecx = ex + ew * 0.5f;
          const float ecy = ey + eh * 0.5f;
          for (std::size_t i = 0; i < wraith_count; ++i) {
            const float dx = ecx - wraiths[i].cx;
            const float dy = ecy - wraiths[i].cy;
            if (dx * dx + dy * dy <= wraiths[i].r2) {
              reveal_hp = true;
              break;
            }
          }
        }
        if (reveal_hp) {
          const int hp_i =
              std::max(0, static_cast<int>(std::ceil(enemies_->hp[slot])));
          const int hp_max_i = std::max(
              0, static_cast<int>(std::ceil(std::max(0.0f, enemies_->hp_max[slot]))));
          const float ty = std::max(0.0f, sy - 14.0f);
          set_color(r, kHudTextColor);
          SDL_RenderDebugTextFormat(r, sx, ty, "%d/%d", hp_i, hp_max_i);
        }
      }
    }

    if (creatures_ && enemies_) {
      const float mouse_sx = static_cast<float>(input.mouseX());
      const float mouse_sy = static_cast<float>(input.mouseY());
      const float mouse_wx = cam_x1 + (mouse_sx / zoom);
      const float mouse_wy = cam_y1 + (mouse_sy / zoom);

      const float pick_radius = creature_pick_radius_px();
      const auto &refs =
          engine_->grid.queryCircle(mouse_wx, mouse_wy, pick_radius);
      EntityHandle hovered = INVALID_ID;
      for (const EntityRef &ref : refs) {
        if (static_cast<int>(ref.type) != creatures_->getTypeId()) {
          continue;
        }
        if (!engine_is_handle_valid(engine_, ref.index, creatures_->getTypeId())) {
          continue;
        }
        const std::uint32_t cslot = creatures_->getSlot(ref.index);
        if (cslot == INVALID_ID ||
            cslot >= static_cast<std::uint32_t>(creatures_->count)) {
          continue;
        }
        if ((creatures_->flags[cslot] &
             static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
          continue;
        }
        const float cx = creatures_->x_positions[cslot];
        const float cy = creatures_->y_positions[cslot];
        const float cw = static_cast<float>(creatures_->widths[cslot]);
        const float ch = static_cast<float>(creatures_->heights[cslot]);
        if (mouse_wx >= cx && mouse_wx <= (cx + cw) && mouse_wy >= cy &&
            mouse_wy <= (cy + ch)) {
          hovered = ref.index;
          break;
        }
      }

      if (hovered != INVALID_ID) {
        const std::uint32_t cslot = creatures_->getSlot(hovered);
        if (cslot != INVALID_ID &&
            cslot < static_cast<std::uint32_t>(creatures_->count)) {
          const float half = static_cast<float>(creatures_->widths[cslot]) * 0.5f;
          const float wcx = creatures_->x_positions[cslot] + half;
          const float wcy = creatures_->y_positions[cslot] + half;
          const float radius_world = std::max(0.0f, creatures_->attack_range_px[cslot]);

          const float scx = (wcx - cam_x1) * zoom;
          const float scy = (wcy - cam_y1) * zoom;
          const float sr = radius_world * zoom;

          if (sr > 1.0f && kRangeIndicatorSegments >= 6) {
            set_color(r, kRangeIndicatorColor);
            const float two_pi = 2.0f * std::acos(-1.0f);
            const float step = two_pi / static_cast<float>(kRangeIndicatorSegments);
            float prev_x = scx + sr;
            float prev_y = scy;
            for (int i = 1; i <= kRangeIndicatorSegments; ++i) {
              const float a = step * static_cast<float>(i);
              const float x = scx + std::cos(a) * sr;
              const float y = scy + std::sin(a) * sr;
              SDL_RenderLine(r, prev_x, prev_y, x, y);
              prev_x = x;
              prev_y = y;
            }
          }
        }
      }
    }
  }

  if (!show_sell_confirm_ && !show_level_select_ &&
      level_state != LevelManagerState::Failed &&
      level_state != LevelManagerState::LevelClear) {
    if (level_state == LevelManagerState::WaveClear && creatures_) {
      const float time = std::max(0.0f, game_state_.level_time_sec);
      const float two_pi = 2.0f * std::acos(-1.0f);

      if (!merge_pairs_.empty() && !merge_anim_.active) {
        const float pulse =
            0.5f + 0.5f * std::sin(time * two_pi * kMergeLinkPulseHz);
        Rgba8 c = kMergeLinkColor;
        c.a = static_cast<std::uint8_t>(std::clamp(
            static_cast<float>(kMergeLinkColor.a) * (0.35f + 0.65f * pulse),
            0.0f, 255.0f));
        set_color(r, c);

        for (const auto &p : merge_pairs_) {
          const EntityHandle a = p.first;
          const EntityHandle b = p.second;
          if (!engine_is_handle_valid(engine_, a, creatures_->getTypeId()) ||
              !engine_is_handle_valid(engine_, b, creatures_->getTypeId())) {
            continue;
          }
          const std::uint32_t aslot = creatures_->getSlot(a);
          const std::uint32_t bslot = creatures_->getSlot(b);
          if (aslot == INVALID_ID || bslot == INVALID_ID ||
              aslot >= static_cast<std::uint32_t>(creatures_->count) ||
              bslot >= static_cast<std::uint32_t>(creatures_->count)) {
            continue;
          }
          if ((creatures_->flags[aslot] &
               static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0 ||
              (creatures_->flags[bslot] &
               static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }

          const float ahalf =
              static_cast<float>(creatures_->widths[aslot]) * 0.5f;
          const float bhalf =
              static_cast<float>(creatures_->widths[bslot]) * 0.5f;
          const float acx = creatures_->x_positions[aslot] + ahalf;
          const float acy = creatures_->y_positions[aslot] + ahalf;
          const float bcx = creatures_->x_positions[bslot] + bhalf;
          const float bcy = creatures_->y_positions[bslot] + bhalf;

          const float asx = (acx - cam_x1) * zoom;
          const float asy = (acy - cam_y1) * zoom;
          const float bsx = (bcx - cam_x1) * zoom;
          const float bsy = (bcy - cam_y1) * zoom;
          SDL_RenderLine(r, asx, asy, bsx, bsy);
        }
      }

      if (merge_drag_active_ && merge_drag_source_ != INVALID_ID &&
          engine_is_handle_valid(engine_, merge_drag_source_,
                                 creatures_->getTypeId())) {
        const std::uint32_t s = creatures_->getSlot(merge_drag_source_);
        if (s != INVALID_ID && s < static_cast<std::uint32_t>(creatures_->count)) {
          const float half = static_cast<float>(creatures_->widths[s]) * 0.5f;
          const float scx = creatures_->x_positions[s] + half;
          const float scy = creatures_->y_positions[s] + half;
          const float ssx = (scx - cam_x1) * zoom;
          const float ssy = (scy - cam_y1) * zoom;

          const float mx = static_cast<float>(input.mouseX());
          const float my = static_cast<float>(input.mouseY());

          set_color(r, kMergeLinkColor);
          SDL_RenderLine(r, ssx, ssy, mx, my);

          const float sw = static_cast<float>(kCreatureBaseSizePx) * zoom;
          const float sh = static_cast<float>(kCreatureBaseSizePx) * zoom;
          SDL_FRect rect = {mx - sw * 0.5f, my - sh * 0.5f, sw, sh};
          Rgba8 c = character_color(creatures_->character[s]);
          c.a = 170;
          set_color(r, c);
          SDL_RenderFillRect(r, &rect);
          set_color(r, kHudBorderColor);
          SDL_RenderRect(r, &rect);
        }
      }

      if (merge_anim_.active && merge::kAnimationSec > 0.0f) {
        const float t = std::clamp(merge_anim_.elapsed_sec / merge::kAnimationSec,
                                   0.0f, 1.0f);
        const float ax =
            merge_anim_.a_cx + (merge_anim_.target_cx - merge_anim_.a_cx) * t;
        const float ay =
            merge_anim_.a_cy + (merge_anim_.target_cy - merge_anim_.a_cy) * t;
        const float bx =
            merge_anim_.b_cx + (merge_anim_.target_cx - merge_anim_.b_cx) * t;
        const float by =
            merge_anim_.b_cy + (merge_anim_.target_cy - merge_anim_.b_cy) * t;

        const float asx = (ax - cam_x1) * zoom;
        const float asy = (ay - cam_y1) * zoom;
        const float bsx = (bx - cam_x1) * zoom;
        const float bsy = (by - cam_y1) * zoom;

        const float sw = static_cast<float>(kCreatureBaseSizePx) * zoom;
        const float sh = static_cast<float>(kCreatureBaseSizePx) * zoom;
        SDL_FRect arect = {asx - sw * 0.5f, asy - sh * 0.5f, sw, sh};
        SDL_FRect brect = {bsx - sw * 0.5f, bsy - sh * 0.5f, sw, sh};

        Rgba8 c = character_color(merge_anim_.character);
        c.a = 190;
        set_color(r, c);
        SDL_RenderFillRect(r, &arect);
        SDL_RenderFillRect(r, &brect);
        set_color(r, kHudBorderColor);
        SDL_RenderRect(r, &arect);
        SDL_RenderRect(r, &brect);
      }
    }

    if (creatures_) {
      const float two_pi = 2.0f * std::acos(-1.0f);
      const float step =
          (kRangeIndicatorSegments >= 6)
              ? (two_pi / static_cast<float>(kRangeIndicatorSegments))
              : 0.0f;

      for (std::uint32_t slot = 0;
           slot < static_cast<std::uint32_t>(creatures_->count); ++slot) {
        if ((creatures_->flags[slot] &
             static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
          continue;
        }
        if (creatures_->state[slot] != CreatureState::Evolving) {
          continue;
        }
        const float remain = std::max(0.0f, creatures_->state_time_sec[slot]);
        if (remain <= 0.0f || evolution::kEvolutionAnimSec <= 0.0f) {
          continue;
        }

        const float half = static_cast<float>(creatures_->widths[slot]) * 0.5f;
        const float wcx = creatures_->x_positions[slot] + half;
        const float wcy = creatures_->y_positions[slot] + half;

        const float progress =
            std::clamp(1.0f - (remain / evolution::kEvolutionAnimSec), 0.0f, 1.0f);
        const float pulse =
            1.0f + (evolution::kEvolutionPulseScale - 1.0f) *
                       std::sin(std::acos(-1.0f) * progress);
        const float base_r =
            static_cast<float>(kCreatureBaseSizePx) * 0.5f *
            evolution::sizeMultiplierForTier(creatures_->tier[slot]);
        const float radius_world = std::max(0.0f, base_r * pulse);

        const float scx = (wcx - cam_x1) * zoom;
        const float scy = (wcy - cam_y1) * zoom;
        const float sr = radius_world * zoom;
        if (sr <= 1.0f || step <= 0.0f) {
          continue;
        }

        set_color(r, kEvolutionPulseColor);
        float prev_x = scx + sr;
        float prev_y = scy;
        for (int i = 1; i <= kRangeIndicatorSegments; ++i) {
          const float a = step * static_cast<float>(i);
          const float x = scx + std::cos(a) * sr;
          const float y = scy + std::sin(a) * sr;
          SDL_RenderLine(r, prev_x, prev_y, x, y);
          prev_x = x;
          prev_y = y;
        }
      }
    }

    if (!game_state_.floating_texts.empty()) {
      for (const FloatingText &t : game_state_.floating_texts) {
        const float sx = (t.world_x - cam_x1) * zoom;
        const float sy = (t.world_y - cam_y1) * zoom;
        const float a = std::clamp(
            t.remaining_sec / std::max(0.001f, evolution::kEvolutionFloatingTextSec),
            0.0f, 1.0f);
        Rgba8 c = kHudTextColor;
        c.a = static_cast<std::uint8_t>(
            std::clamp(a * 255.0f, 0.0f, 255.0f));
        set_color(r, c);
        SDL_RenderDebugTextFormat(r, sx, sy, "%s", t.text.c_str());
      }
    }

    if (creatures_ && selected_creature_ != INVALID_ID) {
      if (!engine_is_handle_valid(engine_, selected_creature_,
                                  creatures_->getTypeId())) {
        selected_creature_ = INVALID_ID;
      } else {
        const std::uint32_t cslot = creatures_->getSlot(selected_creature_);
        if (cslot == INVALID_ID ||
            cslot >= static_cast<std::uint32_t>(creatures_->count)) {
          selected_creature_ = INVALID_ID;
        } else {
          const CharacterId cid = creatures_->character[cslot];
          const int t = std::max(1, creatures_->tier[cslot]);
          const int k = std::max(0, creatures_->kills[cslot]);
          int prev_need = (t > 1) ? evolution::killsNeededForNextTier(t - 1) : 0;
          int next_need = evolution::killsNeededForNextTier(t);
          if (next_need > 0) {
            const float mult =
                std::clamp(game_state_.rapidGrowthKillThresholdMultiplier(), 0.01f, 1.0f);
            next_need = std::max(
                1, static_cast<int>(std::floor(static_cast<float>(next_need) * mult)));
            if (prev_need > 0) {
              prev_need = std::max(
                  0, static_cast<int>(std::floor(static_cast<float>(prev_need) * mult)));
            }
          }
          const int denom = std::max(1, next_need - prev_need);
          const float frac =
              std::clamp(static_cast<float>(k - prev_need) / static_cast<float>(denom),
                         0.0f, 1.0f);

          const float pw = static_cast<float>(kSelectedCreaturePanelWidthPx);
          const float ph = static_cast<float>(kSelectedCreaturePanelHeightPx);
          const float px =
              std::max(0.0f, engine_->camera.width - pw - static_cast<float>(kHudPaddingPx));
          const float py =
              static_cast<float>(kHudTopBarHeightPx + kHudPaddingPx);
          SDL_FRect panel = {px, py, pw, ph};
          set_color(r, kModalPanelColor);
          SDL_RenderFillRect(r, &panel);
          set_color(r, kHudBorderColor);
          SDL_RenderRect(r, &panel);

          set_color(r, kHudTextColor);
          SDL_RenderDebugTextFormat(
              r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
              panel.y + static_cast<float>(kModalPanelTextInsetYPx),
              "%s  (%s)", to_string(cid).data(), get_stage_name(cid, t).data());
          SDL_RenderDebugTextFormat(
              r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
              panel.y + static_cast<float>(kModalPanelTextInsetYPx + kModalPanelLineStepPx),
              "Tier %d  |  Kills %d / %d", t, k, next_need);

          const float bar_h = static_cast<float>(kSelectedCreaturePanelBarHeightPx);
          const float bar_w =
              pw - 2.0f * static_cast<float>(kModalPanelTextInsetXPx);
          const float bar_x = panel.x + static_cast<float>(kModalPanelTextInsetXPx);
          const float bar_y =
              panel.y + ph - bar_h - static_cast<float>(kHudPaddingPx);
          SDL_FRect outer = {bar_x, bar_y, bar_w, bar_h};
          set_color(r, kSelectedCreatureBarBackColor);
          SDL_RenderFillRect(r, &outer);

          const float inset = static_cast<float>(kSelectedCreaturePanelBarInsetPx);
          SDL_FRect fill = {bar_x + inset, bar_y + inset,
                            std::max(0.0f, bar_w - 2.0f * inset) * frac,
                            std::max(0.0f, bar_h - 2.0f * inset)};
          set_color(r, kSelectedCreatureBarFillColor);
          SDL_RenderFillRect(r, &fill);

          set_color(r, kHudBorderColor);
          SDL_RenderRect(r, &outer);
        }
      }
    }
  }

  set_color(r, kHudTextColor);
  SDL_RenderDebugTextFormat(
      r, static_cast<float>(kHudPaddingPx),
      static_cast<float>(kHudTopBarHeightPx + kHudSecondaryTextOffsetYPx),
      "Level %d | Wave %d/%d | Essence %d | Roster %d/%d%s",
      static_cast<int>(level_manager_.levelDef().level_number),
      static_cast<int>(level_manager_.waveIndex() + 1),
      static_cast<int>(level_manager_.waveCount()),
      game_state_.essence,
      static_cast<int>(selected_roster_index_ + 1),
      static_cast<int>(game_state_.roster.size()),
      show_sell_confirm_ ? " | SELL CONFIRM" : "");

  if (const char *banner = level_manager_.bannerText()) {
    const std::size_t len = std::strlen(banner);
    const float text_w =
        static_cast<float>(len) *
        static_cast<float>(SDL_DEBUG_TEXT_FONT_CHARACTER_SIZE);
    const float x =
        std::max(0.0f, (engine_->camera.width - text_w) * 0.5f);
    const float y = static_cast<float>(kHudTopBarHeightPx + kHudPaddingPx * 2);
    set_color(r, kHudTextColor);
    SDL_RenderDebugTextFormat(r, x, y, "%s", banner);
  }

  if (!show_sell_confirm_ && !show_level_select_ &&
      level_state == LevelManagerState::WaveClear) {
    if (wave_buff_shop_.isOpen()) {
      wave_buff_shop_.render(r, input, engine_->camera.width, engine_->camera.height,
                             level_manager_.graceRemainingSec());
    } else {
      set_color(r, kHudTextColor);
      SDL_RenderDebugTextFormat(
          r, static_cast<float>(kHudPaddingPx),
          static_cast<float>(kHudTopBarHeightPx + kHudPaddingPx * 2 +
                             kModalPanelLineStepPx),
          "Next wave in %.1fs (place creatures)",
          level_manager_.graceRemainingSec());
    }
  }

  if (show_sell_confirm_) {
    const float w = engine_->camera.width;
    const float h = engine_->camera.height;

    SDL_FRect overlay = {0.0f, 0.0f, w, h};
    set_color(r, kModalOverlayColor);
    SDL_RenderFillRect(r, &overlay);

    const float pw = static_cast<float>(kConfirmDialogWidthPx);
    const float ph = static_cast<float>(kConfirmDialogHeightPx);
    const float px = (w - pw) * 0.5f;
    const float py = (h - ph) * 0.5f;
    SDL_FRect panel = {px, py, pw, ph};
    set_color(r, kModalPanelColor);
    SDL_RenderFillRect(r, &panel);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &panel);

    const float bw = static_cast<float>(kConfirmDialogButtonWidthPx);
    const float bh = static_cast<float>(kConfirmDialogButtonHeightPx);
    const float gap = static_cast<float>(kConfirmDialogButtonGapPx);

    const float buttons_y = py + ph - bh - static_cast<float>(kHudPaddingPx);
    const float buttons_x = px + (pw - (2.0f * bw + gap)) * 0.5f;
    const SDL_FRect sell_rect = {buttons_x, buttons_y, bw, bh};
    const SDL_FRect cancel_rect = {buttons_x + bw + gap, buttons_y, bw, bh};

    const float mx = static_cast<float>(input.mouseX());
    const float my = static_cast<float>(input.mouseY());

    set_color(r, point_in_rect(mx, my, sell_rect) ? kModalButtonHoverColor
                                                  : kModalButtonColor);
    SDL_RenderFillRect(r, &sell_rect);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &sell_rect);

    set_color(r, point_in_rect(mx, my, cancel_rect) ? kModalButtonHoverColor
                                                    : kModalButtonColor);
    SDL_RenderFillRect(r, &cancel_rect);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &cancel_rect);

    set_color(r, kModalButtonTextColor);
    SDL_RenderDebugTextFormat(r, sell_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
                              sell_rect.y + static_cast<float>(kModalButtonTextInsetYPx),
                              "SELL");
    SDL_RenderDebugTextFormat(
        r, cancel_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
        cancel_rect.y + static_cast<float>(kModalButtonTextInsetYPx),
                              "CANCEL");

    const int ri = pending_sell_roster_index_;
    if (ri >= 0 && ri < static_cast<int>(game_state_.roster.size())) {
      const RosterEntry &re = game_state_.roster[static_cast<std::size_t>(ri)];
      const int refund = std::max(
          0, static_cast<int>(static_cast<float>(re.seed_cost_essence) *
                              inter_level_shop::kSellRefundFraction));
      SDL_RenderDebugTextFormat(
          r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
          panel.y + static_cast<float>(kModalPanelTextInsetYPx),
          "Sell %.*s? Refund %d essence",
          static_cast<int>(to_string(re.character).size()),
          to_string(re.character).data(), refund);
    } else {
      SDL_RenderDebugTextFormat(r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
                                panel.y + static_cast<float>(kModalPanelTextInsetYPx),
                                "Sell creature?");
    }
  } else if (show_level_select_) {
    const float w = engine_->camera.width;
    const float h = engine_->camera.height;

    SDL_FRect overlay = {0.0f, 0.0f, w, h};
    set_color(r, kModalOverlayColor);
    SDL_RenderFillRect(r, &overlay);

    const float pw = static_cast<float>(kConfirmDialogWidthPx);
    const float ph = static_cast<float>(kConfirmDialogHeightPx);
    const float px = (w - pw) * 0.5f;
    const float py = (h - ph) * 0.5f;
    SDL_FRect panel = {px, py, pw, ph};
    set_color(r, kModalPanelColor);
    SDL_RenderFillRect(r, &panel);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &panel);

    const float bw = static_cast<float>(kConfirmDialogButtonWidthPx);
    const float bh = static_cast<float>(kConfirmDialogButtonHeightPx);
    const float gap = static_cast<float>(kConfirmDialogButtonGapPx);

    const float buttons_y = py + ph - bh - static_cast<float>(kHudPaddingPx);
    const float buttons_x = px + (pw - (2.0f * bw + gap)) * 0.5f;
    const SDL_FRect play_rect = {buttons_x, buttons_y, bw, bh};
    const SDL_FRect back_rect = {buttons_x + bw + gap, buttons_y, bw, bh};

    const float mx = static_cast<float>(input.mouseX());
    const float my = static_cast<float>(input.mouseY());

    set_color(r, point_in_rect(mx, my, play_rect) ? kModalButtonHoverColor
                                                  : kModalButtonColor);
    SDL_RenderFillRect(r, &play_rect);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &play_rect);

    set_color(r, point_in_rect(mx, my, back_rect) ? kModalButtonHoverColor
                                                  : kModalButtonColor);
    SDL_RenderFillRect(r, &back_rect);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &back_rect);

    set_color(r, kModalButtonTextColor);
    SDL_RenderDebugTextFormat(
        r, play_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
        play_rect.y + static_cast<float>(kModalButtonTextInsetYPx), "PLAY");
    SDL_RenderDebugTextFormat(
        r, back_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
        back_rect.y + static_cast<float>(kModalButtonTextInsetYPx), "BACK");

    const std::int32_t max_level =
        std::max<std::int32_t>(1, game_state_.max_level_reached);
    const std::int32_t level = std::clamp<std::int32_t>(level_select_level_, 1, max_level);
    int stars = 0;
    const std::size_t idx =
        static_cast<std::size_t>(std::max<std::int32_t>(0, level - 1));
    if (idx < game_state_.stars_per_level.size()) {
      stars = static_cast<int>(game_state_.stars_per_level[idx]);
    }

    set_color(r, kHudTextColor);
    SDL_RenderDebugTextFormat(
        r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
        panel.y + static_cast<float>(kModalPanelTextInsetYPx),
        "LEVEL SELECT  (Unlocked 1-%d)", static_cast<int>(max_level));
    SDL_RenderDebugTextFormat(
        r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
        panel.y + static_cast<float>(kModalPanelTextInsetYPx + kModalPanelLineStepPx),
        "Level %d  | Stars %d", static_cast<int>(level), stars);
    SDL_RenderDebugTextFormat(
        r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
        panel.y + static_cast<float>(kModalPanelTextInsetYPx + kModalPanelLineStepPx * 2),
        "Left/Right to change, Enter to play");
  } else if (show_armory_) {
    const float w = engine_->camera.width;
    const float h = engine_->camera.height;

    const float mx = static_cast<float>(input.mouseX());
    const float my = static_cast<float>(input.mouseY());

    SDL_FRect overlay = {0.0f, 0.0f, w, h};
    set_color(r, kModalOverlayColor);
    SDL_RenderFillRect(r, &overlay);

    const float margin = static_cast<float>(kHudPaddingPx);
    const float panel_w =
        std::min(1040.0f, std::max(0.0f, w - 2.0f * margin));
    const float panel_h =
        std::min(720.0f, std::max(0.0f, h - 2.0f * margin));
    const float panel_x = (w - panel_w) * 0.5f;
    const float panel_y = (h - panel_h) * 0.5f;
    SDL_FRect panel = {panel_x, panel_y, panel_w, panel_h};
    set_color(r, kModalPanelColor);
    SDL_RenderFillRect(r, &panel);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &panel);

    const float header_h = 56.0f;
    const float tab_h = 32.0f;
    const float inner_pad = 14.0f;

    const float tab_y = panel_y + header_h;
    const float tab_w = 140.0f;
    const float tab_gap = 8.0f;
    const SDL_FRect tab_chars = {panel_x + inner_pad, tab_y, tab_w, tab_h};
    const SDL_FRect tab_masteries = {tab_chars.x + tab_w + tab_gap, tab_y, tab_w,
                                     tab_h};
    const SDL_FRect tab_relics = {tab_masteries.x + tab_w + tab_gap, tab_y, tab_w,
                                  tab_h};
    const SDL_FRect tab_cosmetics = {tab_relics.x + tab_w + tab_gap, tab_y, tab_w,
                                     tab_h};

    const SDL_FRect close_rect = {
        panel_x + panel_w - inner_pad - 96.0f,
        panel_y + (header_h - 28.0f) * 0.5f,
        96.0f,
        28.0f,
    };

    auto rarity_name = [&](Rarity rr) -> const char * {
      switch (rr) {
      case Rarity::Common:
        return "Common";
      case Rarity::Rare:
        return "Rare";
      case Rarity::Epic:
        return "Epic";
      case Rarity::Legendary:
        return "Legendary";
      }
      return "Common";
    };

    auto mastery_name = [&](MasteryId id) -> const char * {
      switch (id) {
      case MasteryId::EchoFoundation:
        return "Echo Foundation";
      case MasteryId::NexusVault:
        return "Nexus Vault";
      case MasteryId::RapidGrowth:
        return "Rapid Growth";
      case MasteryId::KineticSwarm:
        return "Kinetic Swarm";
      case MasteryId::SynthesisMastery:
        return "Synthesis Mastery";
      case MasteryId::IronResolve:
        return "Iron Resolve";
      case MasteryId::VoidAppetite:
        return "Void Appetite";
      case MasteryId::ShardEye:
        return "Shard Eye";
      case MasteryId::Count:
        break;
      }
      return "Mastery";
    };

    auto mastery_effect = [&](MasteryId id) -> const char * {
      switch (id) {
      case MasteryId::EchoFoundation:
        return "+20 essence / rank";
      case MasteryId::NexusVault:
        return "+10 base HP / rank";
      case MasteryId::RapidGrowth:
        return "Evolve threshold -5% / rank";
      case MasteryId::KineticSwarm:
        return "Move speed +5% / rank";
      case MasteryId::SynthesisMastery:
        return "Merge cooldown -1s / rank";
      case MasteryId::IronResolve:
        return "+5% creature HP per PL>20 / rank";
      case MasteryId::VoidAppetite:
        return "Essence drops +8% / rank";
      case MasteryId::ShardEye:
        return "+1 shard on first clear / rank";
      case MasteryId::Count:
        break;
      }
      return "";
    };

    auto stars_for_level = [&](int level) -> int {
      if (level <= 0) {
        return 0;
      }
      const std::size_t idx =
          static_cast<std::size_t>(std::max(0, level - 1));
      if (idx >= game_state_.stars_per_level.size()) {
        return 0;
      }
      return std::clamp<int>(game_state_.stars_per_level[idx], 0, 3);
    };

    auto character_requirements_met = [&](CharacterId cid) -> bool {
      if (game_state_.isCharacterUnlocked(cid)) {
        return false;
      }
      const CharacterDefinition &def = get_character_def(cid);
      if (cid == CharacterId::Orin) {
        return stars_for_level(unlocks::kOrinUnlockLevel) >= 3;
      }
      return game_state_.max_level_reached >= def.unlock_level;
    };

    auto draw_tab = [&](const SDL_FRect &rr, const char *label, bool active) {
      const bool hover = point_in_rect(mx, my, rr);
      set_color(r, (active || hover) ? kModalButtonHoverColor : kModalButtonColor);
      SDL_RenderFillRect(r, &rr);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &rr);
      set_color(r, kModalButtonTextColor);
      SDL_RenderDebugTextFormat(
          r, rr.x + static_cast<float>(kModalButtonTextInsetXPx),
          rr.y + static_cast<float>(kModalButtonTextInsetYPx), "%s", label);
    };

    auto draw_button = [&](const SDL_FRect &rr, const char *label) {
      const bool hover = point_in_rect(mx, my, rr);
      set_color(r, hover ? kModalButtonHoverColor : kModalButtonColor);
      SDL_RenderFillRect(r, &rr);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &rr);
      set_color(r, kModalButtonTextColor);
      SDL_RenderDebugTextFormat(
          r, rr.x + static_cast<float>(kModalButtonTextInsetXPx),
          rr.y + static_cast<float>(kModalButtonTextInsetYPx), "%s", label);
    };

    set_color(r, kHudTextColor);
    SDL_RenderDebugTextFormat(
        r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
        panel.y + static_cast<float>(kModalPanelTextInsetYPx),
        "ARMORY  |  Shards %d  |  Player Lv %d (XP %d)",
        std::max(0, game_state_.shards), std::max(1, game_state_.player_level),
        std::max(0, game_state_.player_xp));

    draw_button(close_rect, "CLOSE");

    draw_tab(tab_chars, "CHARS", armory_tab_ == ArmoryTab::Characters);
    draw_tab(tab_masteries, "MASTERY", armory_tab_ == ArmoryTab::Masteries);
    draw_tab(tab_relics, "RELICS", armory_tab_ == ArmoryTab::Relics);
    draw_tab(tab_cosmetics, "COSMETIC", armory_tab_ == ArmoryTab::Cosmetics);

    const float content_x = panel_x + inner_pad;
    const float content_y = tab_y + tab_h + inner_pad;
    const float content_w = std::max(0.0f, panel_w - 2.0f * inner_pad);
    const float content_h =
        std::max(0.0f, panel_h - header_h - tab_h - inner_pad * 2.0f);

    if (armory_tab_ == ArmoryTab::Characters) {
      const int cols = 5;
      const float gap = 10.0f;
      const float card_w =
          cols > 0 ? std::max(0.0f, (content_w - gap * (cols - 1)) /
                                         static_cast<float>(cols))
                   : 0.0f;
      const float card_h = std::max(0.0f, (content_h - gap) * 0.5f);

      for (int i = 0; i < static_cast<int>(CharacterId::Count); ++i) {
        const auto cid = static_cast<CharacterId>(static_cast<std::uint8_t>(i));
        const CharacterDefinition &def = get_character_def(cid);
        const bool unlocked = game_state_.isCharacterUnlocked(cid);
        const bool req_met = character_requirements_met(cid);

        const int row = i / cols;
        const int col = i % cols;
        SDL_FRect card = {content_x + static_cast<float>(col) * (card_w + gap),
                          content_y + static_cast<float>(row) * (card_h + gap),
                          card_w, card_h};

        const bool hover = point_in_rect(mx, my, card);
        Rgba8 bg = unlocked ? (hover ? kWaveShopCardHoverColor : kWaveShopCardColor)
                            : Rgba8{14, 18, 30, 190};
        set_color(r, bg);
        SDL_RenderFillRect(r, &card);
        set_color(r, kWaveShopCardBorderColor);
        SDL_RenderRect(r, &card);

        SDL_FRect icon = {card.x + 10.0f, card.y + 10.0f, 34.0f, 34.0f};
        Rgba8 ic = character_color(cid);
        ic.a = unlocked ? 220 : 90;
        set_color(r, ic);
        SDL_RenderFillRect(r, &icon);
        set_color(r, kHudBorderColor);
        SDL_RenderRect(r, &icon);

        set_color(r, kHudTextColor);
        float tx = card.x + 52.0f;
        float ty = card.y + 10.0f;
        SDL_RenderDebugTextFormat(
            r, tx, ty, "%.*s (%s)",
            static_cast<int>(to_string(cid).size()), to_string(cid).data(),
            rarity_name(def.rarity));
        ty += static_cast<float>(kModalPanelLineStepPx);

        if (unlocked) {
          SDL_RenderDebugTextFormat(r, tx, ty, "UNLOCKED");
        } else {
          const int cost = std::max(0, def.unlock_shards);
          if (cid == CharacterId::Orin) {
            SDL_RenderDebugTextFormat(r, tx, ty, "Req: 3★ L%d  |  %d shards%s",
                                      unlocks::kOrinUnlockLevel, cost,
                                      req_met ? "" : " (locked)");
          } else {
            SDL_RenderDebugTextFormat(r, tx, ty, "Req: L%d  |  %d shards%s",
                                      def.unlock_level, cost,
                                      req_met ? "" : " (locked)");
          }
        }
      }

      set_color(r, kHudTextColor);
      SDL_RenderDebugTextFormat(
          r, content_x,
          content_y + content_h - static_cast<float>(kModalPanelLineStepPx),
          "Click a character for details");
    } else if (armory_tab_ == ArmoryTab::Masteries) {
      const float row_h = 58.0f;
      const float gap = 8.0f;
      for (int i = 0; i < static_cast<int>(MasteryId::Count); ++i) {
        const auto mid = static_cast<MasteryId>(static_cast<std::uint8_t>(i));
        const int cur = game_state_.masteryRank(mid);
        const int max_r = GameState::masteryMaxRanks(mid);
        const int cost =
            (cur < max_r) ? std::max(0, GameState::masteryNextRankCost(mid, cur)) : 0;
        SDL_FRect rr = {content_x, content_y + static_cast<float>(i) * (row_h + gap),
                        content_w, row_h};
        const bool hover = point_in_rect(mx, my, rr);
        set_color(r, hover ? kWaveShopCardHoverColor : kWaveShopCardColor);
        SDL_RenderFillRect(r, &rr);
        set_color(r, kWaveShopCardBorderColor);
        SDL_RenderRect(r, &rr);

        set_color(r, kHudTextColor);
        SDL_RenderDebugTextFormat(
            r, rr.x + static_cast<float>(kWaveShopCardTextInsetXPx),
            rr.y + static_cast<float>(kWaveShopCardTextInsetYPx),
            "%s  (%s)  Rank %d/%d", mastery_name(mid), mastery_effect(mid), cur,
            max_r);

        const SDL_FRect buy = {rr.x + rr.w - 130.0f, rr.y + 14.0f, 120.0f, 30.0f};
        const bool can_buy = (cur < max_r) && cost > 0 && game_state_.shards >= cost;
        set_color(r, can_buy ? (point_in_rect(mx, my, buy) ? kModalButtonHoverColor
                                                          : kModalButtonColor)
                             : Rgba8{20, 26, 42, 140});
        SDL_RenderFillRect(r, &buy);
        set_color(r, kHudBorderColor);
        SDL_RenderRect(r, &buy);
        set_color(r, kModalButtonTextColor);
        if (cur >= max_r) {
          SDL_RenderDebugTextFormat(
              r, buy.x + static_cast<float>(kModalButtonTextInsetXPx),
              buy.y + static_cast<float>(kModalButtonTextInsetYPx), "MAX");
        } else {
          SDL_RenderDebugTextFormat(
              r, buy.x + static_cast<float>(kModalButtonTextInsetXPx),
              buy.y + static_cast<float>(kModalButtonTextInsetYPx), "BUY %d",
              cost);
        }
      }
    } else if (armory_tab_ == ArmoryTab::Relics) {
      const int cols = 4;
      const float gap = 10.0f;
      const float card_w =
          cols > 0 ? std::max(0.0f, (content_w - gap * (cols - 1)) /
                                         static_cast<float>(cols))
                   : 0.0f;
      const float card_h = 68.0f;
      for (std::size_t i = 0; i < RelicSystem::kRelicCount; ++i) {
        const RelicId rid = static_cast<RelicId>(i);
        const RelicDef &def = RelicSystem::def(rid);
        const bool unlocked = game_state_.isRelicUnlocked(rid);

        const int row = static_cast<int>(i / static_cast<std::size_t>(cols));
        const int col = static_cast<int>(i % static_cast<std::size_t>(cols));
        SDL_FRect rr = {content_x + static_cast<float>(col) * (card_w + gap),
                        content_y + static_cast<float>(row) * (card_h + gap),
                        card_w, card_h};
        const bool hover = point_in_rect(mx, my, rr);
        set_color(r, unlocked ? (hover ? kWaveShopCardHoverColor : kWaveShopCardColor)
                              : Rgba8{14, 18, 30, 190});
        SDL_RenderFillRect(r, &rr);
        set_color(r, kWaveShopCardBorderColor);
        SDL_RenderRect(r, &rr);

        set_color(r, kHudTextColor);
        float tx = rr.x + static_cast<float>(kWaveShopCardTextInsetXPx);
        float ty = rr.y + static_cast<float>(kWaveShopCardTextInsetYPx);
        SDL_RenderDebugTextFormat(r, tx, ty, "%s", def.name);
        ty += static_cast<float>(kModalPanelLineStepPx);
        if (unlocked) {
          SDL_RenderDebugTextFormat(r, tx, ty, "UNLOCKED");
        } else {
          SDL_RenderDebugTextFormat(r, tx, ty, "Cost %d shards",
                                    std::max(0, def.shard_cost));
        }
      }
    } else {
      set_color(r, kHudTextColor);
      SDL_RenderDebugTextFormat(
          r, content_x, content_y,
          "Cosmetics are not available in this build.");
    }

    if (show_armory_character_detail_) {
      SDL_FRect o2 = {0.0f, 0.0f, w, h};
      set_color(r, Rgba8{0, 0, 0, 200});
      SDL_RenderFillRect(r, &o2);

      const SDL_FRect detail_rect = {
          panel_x + (panel_w - std::min(640.0f, panel_w - 2.0f * inner_pad)) * 0.5f,
          panel_y + (panel_h - std::min(420.0f, panel_h - 2.0f * inner_pad)) * 0.5f,
          std::min(640.0f, panel_w - 2.0f * inner_pad),
          std::min(420.0f, panel_h - 2.0f * inner_pad),
      };
      set_color(r, kModalPanelColor);
      SDL_RenderFillRect(r, &detail_rect);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &detail_rect);

      const SDL_FRect detail_close = {
          detail_rect.x + detail_rect.w - 92.0f,
          detail_rect.y + 12.0f,
          80.0f,
          28.0f,
      };
      draw_button(detail_close, "CLOSE");

      const CharacterId cid = armory_selected_character_;
      const CharacterDefinition &def = get_character_def(cid);
      const bool unlocked = game_state_.isCharacterUnlocked(cid);
      const bool req_met = character_requirements_met(cid);
      const int cost = std::max(0, def.unlock_shards);

      set_color(r, kHudTextColor);
      float tx = detail_rect.x + static_cast<float>(kModalPanelTextInsetXPx);
      float ty = detail_rect.y + static_cast<float>(kModalPanelTextInsetYPx);
      SDL_RenderDebugTextFormat(r, tx, ty, "%.*s  (%s)",
                                static_cast<int>(to_string(cid).size()),
                                to_string(cid).data(), rarity_name(def.rarity));
      ty += static_cast<float>(kModalPanelLineStepPx);
      SDL_RenderDebugTextFormat(r, tx, ty, "%.*s",
                                static_cast<int>(def.lore.size()),
                                def.lore.data());
      ty += static_cast<float>(kModalPanelLineStepPx * 2);
      SDL_RenderDebugTextFormat(r, tx, ty, "Stages: %.*s | %.*s | %.*s",
                                static_cast<int>(def.stage_names[0].size()),
                                def.stage_names[0].data(),
                                static_cast<int>(def.stage_names[1].size()),
                                def.stage_names[1].data(),
                                static_cast<int>(def.stage_names[2].size()),
                                def.stage_names[2].data());
      ty += static_cast<float>(kModalPanelLineStepPx);

      const CharacterBaseStats &bs = def.base;
      SDL_RenderDebugTextFormat(r, tx, ty, "Stats: HP %.0f  DMG %.1f  RNG %.0f",
                                bs.base_hp, bs.base_damage, bs.base_range_px);
      ty += static_cast<float>(kModalPanelLineStepPx);
      SDL_RenderDebugTextFormat(r, tx, ty, "APS %.2f  Move %.0f",
                                bs.base_attack_rate_per_sec,
                                bs.base_move_speed_px_per_sec);
      ty += static_cast<float>(kModalPanelLineStepPx);
      if (bs.splash_radius_px > 0.0f) {
        SDL_RenderDebugTextFormat(r, tx, ty, "Splash: %.0fpx",
                                  bs.splash_radius_px);
        ty += static_cast<float>(kModalPanelLineStepPx);
      } else if (bs.aura_radius_px > 0.0f) {
        SDL_RenderDebugTextFormat(
            r, tx, ty, "Aura: %.0fpx  AtkSpd %+0.0f%%  Dmg %+0.0f%%",
            bs.aura_radius_px, bs.aura_attack_speed * 100.0f,
            bs.aura_damage * 100.0f);
        ty += static_cast<float>(kModalPanelLineStepPx);
      } else if (bs.slow_field_radius_px > 0.0f) {
        SDL_RenderDebugTextFormat(r, tx, ty, "Field: Slow %.0fpx",
                                  bs.slow_field_radius_px);
        ty += static_cast<float>(kModalPanelLineStepPx);
      } else if (bs.drain_radius_px > 0.0f) {
        SDL_RenderDebugTextFormat(r, tx, ty, "Aura: Drain %.0fpx",
                                  bs.drain_radius_px);
        ty += static_cast<float>(kModalPanelLineStepPx);
      }

      if (unlocked) {
        SDL_RenderDebugTextFormat(r, tx, ty, "Status: UNLOCKED");
      } else if (cid == CharacterId::Orin) {
        SDL_RenderDebugTextFormat(r, tx, ty, "Req: 3★ L%d  |  Cost %d shards",
                                  unlocks::kOrinUnlockLevel, cost);
      } else {
        SDL_RenderDebugTextFormat(r, tx, ty, "Req: L%d  |  Cost %d shards",
                                  def.unlock_level, cost);
      }

      const SDL_FRect unlock_rect = {
          detail_rect.x + detail_rect.w - 200.0f,
          detail_rect.y + detail_rect.h - 46.0f,
          180.0f,
          34.0f,
      };
      const bool can_unlock =
          !unlocked && req_met && cost > 0 && game_state_.shards >= cost;
      set_color(r, can_unlock ? (point_in_rect(mx, my, unlock_rect) ? kModalButtonHoverColor
                                                                    : kModalButtonColor)
                              : Rgba8{20, 26, 42, 140});
      SDL_RenderFillRect(r, &unlock_rect);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &unlock_rect);
      set_color(r, kModalButtonTextColor);
      SDL_RenderDebugTextFormat(
          r, unlock_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
          unlock_rect.y + static_cast<float>(kModalButtonTextInsetYPx),
          unlocked ? "UNLOCKED" : (req_met ? "UNLOCK" : "LOCKED"));
    }

    if (show_armory_confirm_) {
      SDL_FRect o2 = {0.0f, 0.0f, w, h};
      set_color(r, Rgba8{0, 0, 0, 200});
      SDL_RenderFillRect(r, &o2);

      const float pw = static_cast<float>(kConfirmDialogWidthPx);
      const float ph = static_cast<float>(kConfirmDialogHeightPx);
      const float px = (w - pw) * 0.5f;
      const float py = (h - ph) * 0.5f;
      SDL_FRect cp = {px, py, pw, ph};
      set_color(r, kModalPanelColor);
      SDL_RenderFillRect(r, &cp);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &cp);

      const float bw = static_cast<float>(kConfirmDialogButtonWidthPx);
      const float bh = static_cast<float>(kConfirmDialogButtonHeightPx);
      const float gap = static_cast<float>(kConfirmDialogButtonGapPx);

      const float buttons_y = py + ph - bh - static_cast<float>(kHudPaddingPx);
      const float buttons_x = px + (pw - (2.0f * bw + gap)) * 0.5f;
      const SDL_FRect yes_rect = {buttons_x, buttons_y, bw, bh};
      const SDL_FRect no_rect = {buttons_x + bw + gap, buttons_y, bw, bh};

      set_color(r, point_in_rect(mx, my, yes_rect) ? kModalButtonHoverColor
                                                   : kModalButtonColor);
      SDL_RenderFillRect(r, &yes_rect);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &yes_rect);
      set_color(r, point_in_rect(mx, my, no_rect) ? kModalButtonHoverColor
                                                  : kModalButtonColor);
      SDL_RenderFillRect(r, &no_rect);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &no_rect);

      set_color(r, kModalButtonTextColor);
      SDL_RenderDebugTextFormat(r, yes_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
                                yes_rect.y + static_cast<float>(kModalButtonTextInsetYPx),
                                "YES");
      SDL_RenderDebugTextFormat(r, no_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
                                no_rect.y + static_cast<float>(kModalButtonTextInsetYPx),
                                "NO");

      set_color(r, kHudTextColor);
      switch (armory_confirm_kind_) {
      case ArmoryConfirmKind::UnlockCharacter: {
        const CharacterId cid = armory_confirm_character_;
        const CharacterDefinition &def = get_character_def(cid);
        SDL_RenderDebugTextFormat(
            r, cp.x + static_cast<float>(kModalPanelTextInsetXPx),
            cp.y + static_cast<float>(kModalPanelTextInsetYPx),
            "Unlock %.*s for %d shards?",
            static_cast<int>(to_string(cid).size()), to_string(cid).data(),
            std::max(0, def.unlock_shards));
        break;
      }
      case ArmoryConfirmKind::BuyMasteryRank: {
        const MasteryId mid = armory_confirm_mastery_;
        const int cur = game_state_.masteryRank(mid);
        const int cost = std::max(0, GameState::masteryNextRankCost(mid, cur));
        SDL_RenderDebugTextFormat(
            r, cp.x + static_cast<float>(kModalPanelTextInsetXPx),
            cp.y + static_cast<float>(kModalPanelTextInsetYPx),
            "Buy %s rank %d for %d shards?",
            mastery_name(mid), std::clamp(cur + 1, 1, 3), cost);
        break;
      }
      case ArmoryConfirmKind::UnlockRelic: {
        const RelicId rid = armory_confirm_relic_;
        const RelicDef &def = RelicSystem::def(rid);
        SDL_RenderDebugTextFormat(
            r, cp.x + static_cast<float>(kModalPanelTextInsetXPx),
            cp.y + static_cast<float>(kModalPanelTextInsetYPx),
            "Unlock %s for %d shards?",
            def.name, std::max(0, def.shard_cost));
        break;
      }
      case ArmoryConfirmKind::None:
        SDL_RenderDebugTextFormat(
            r, cp.x + static_cast<float>(kModalPanelTextInsetXPx),
            cp.y + static_cast<float>(kModalPanelTextInsetYPx),
            "Confirm purchase?");
        break;
      }
    }
  } else if (show_inter_level_ &&
             level_state == LevelManagerState::LevelClear) {
    const float w = engine_->camera.width;
    const float h = engine_->camera.height;

    SDL_FRect overlay = {0.0f, 0.0f, w, h};
    set_color(r, kModalOverlayColor);
    SDL_RenderFillRect(r, &overlay);

    const float margin = static_cast<float>(kHudPaddingPx);
    const float panel_w = std::max(0.0f, w - 2.0f * margin);
    const float panel_h = std::max(0.0f, h - 2.0f * margin);

    const float slide_sec = 0.4f;
    float t = slide_sec > 0.0f
                  ? std::clamp(inter_level_elapsed_sec_ / slide_sec, 0.0f, 1.0f)
                  : 1.0f;
    t = t * t * (3.0f - 2.0f * t); // smoothstep

    const float panel_x = w + (margin - w) * t;
    const float panel_y = margin;
    SDL_FRect panel = {panel_x, panel_y, panel_w, panel_h};
    set_color(r, kModalPanelColor);
    SDL_RenderFillRect(r, &panel);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &panel);

    const float header_h = 72.0f;
    const float tab_h = 32.0f;
    const float inner_pad = 14.0f;
    const float roster_h = 72.0f;
    const float button_h = 38.0f;

    const float content_top = panel_y + header_h + tab_h + inner_pad;
    const float button_y = panel_y + panel_h - button_h - inner_pad;
    const float roster_y = button_y - roster_h - inner_pad;
    const float content_h = std::max(0.0f, roster_y - content_top);

    const float content_x = panel_x + inner_pad;
    const float content_w = std::max(0.0f, panel_w - 2.0f * inner_pad);
    const float results_w = std::min(360.0f, content_w * 0.36f);
    const float shop_gap = inner_pad;
    const float shop_x = content_x + results_w + shop_gap;
    const float shop_w = std::max(0.0f, content_w - results_w - shop_gap);

    const float tab_y = panel_y + header_h;
    const float tab_w = 120.0f;
    const float tab_gap = 8.0f;
    const SDL_FRect tab_bazaar = {panel_x + inner_pad, tab_y, tab_w, tab_h};
    const SDL_FRect tab_forge = {tab_bazaar.x + tab_w + tab_gap, tab_y, tab_w,
                                 tab_h};
    const SDL_FRect tab_relics = {tab_forge.x + tab_w + tab_gap, tab_y, tab_w,
                                  tab_h};
    const SDL_FRect tab_repair = {tab_relics.x + tab_w + tab_gap, tab_y, tab_w,
                                  tab_h};

    const float btn_w = 150.0f;
    const float btn_gap = 12.0f;
    const SDL_FRect btn_levels = {panel_x + inner_pad, button_y, btn_w, button_h};
    const SDL_FRect btn_armory = {btn_levels.x + btn_w + btn_gap, button_y, btn_w,
                                  button_h};
    const SDL_FRect btn_next = {panel_x + panel_w - inner_pad - btn_w, button_y,
                                btn_w, button_h};
    const SDL_FRect btn_replay = {btn_next.x - btn_gap - btn_w, button_y, btn_w,
                                  button_h};

    const float mx = static_cast<float>(input.mouseX());
    const float my = static_cast<float>(input.mouseY());

    auto rarity_name = [&](Rarity r) -> const char * {
      switch (r) {
      case Rarity::Common:
        return "Common";
      case Rarity::Rare:
        return "Rare";
      case Rarity::Epic:
        return "Epic";
      case Rarity::Legendary:
        return "Legendary";
      }
      return "Common";
    };

    auto draw_tab = [&](const SDL_FRect &rect, const char *label, bool active) {
      const bool hover = point_in_rect(mx, my, rect);
      set_color(r, (active || hover) ? kModalButtonHoverColor : kModalButtonColor);
      SDL_RenderFillRect(r, &rect);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &rect);
      set_color(r, kModalButtonTextColor);
      SDL_RenderDebugTextFormat(
          r, rect.x + static_cast<float>(kModalButtonTextInsetXPx),
          rect.y + static_cast<float>(kModalButtonTextInsetYPx), "%s", label);
    };

    auto draw_button = [&](const SDL_FRect &rect, const char *label) {
      const bool hover = point_in_rect(mx, my, rect);
      set_color(r, hover ? kModalButtonHoverColor : kModalButtonColor);
      SDL_RenderFillRect(r, &rect);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &rect);
      set_color(r, kModalButtonTextColor);
      SDL_RenderDebugTextFormat(
          r, rect.x + static_cast<float>(kModalButtonTextInsetXPx),
          rect.y + static_cast<float>(kModalButtonTextInsetYPx), "%s", label);
    };

    set_color(r, kHudTextColor);
    const int level_n = std::max(1, level_manager_.levelDef().level_number);
    const int stars = std::clamp(level_manager_.lastLevelStars(), 0, 3);
    const int quiet_bonus =
        std::max(0, game_state_.the_quiet_bonus_stars_cosmetic);

    const int reveal = std::clamp(
        static_cast<int>(std::floor(inter_level_elapsed_sec_ / 0.35f)), 0, stars);
    char star_line[9] = {'[', '-', ']', '[', '-', ']', '[', '-', ']'};
    if (reveal >= 1) {
      star_line[1] = '*';
    }
    if (reveal >= 2) {
      star_line[4] = '*';
    }
    if (reveal >= 3) {
      star_line[7] = '*';
    }

    if (quiet_bonus > 0) {
      SDL_RenderDebugTextFormat(
          r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
          panel.y + static_cast<float>(kModalPanelTextInsetYPx),
          "LEVEL %d COMPLETE  Stars %d (+%d)  %c%c%c%c%c%c%c%c%c", level_n,
          stars, quiet_bonus, star_line[0], star_line[1], star_line[2],
          star_line[3], star_line[4], star_line[5], star_line[6], star_line[7],
          star_line[8]);
    } else {
      SDL_RenderDebugTextFormat(
          r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
          panel.y + static_cast<float>(kModalPanelTextInsetYPx),
          "LEVEL %d COMPLETE  Stars %d  %c%c%c%c%c%c%c%c%c", level_n, stars,
          star_line[0], star_line[1], star_line[2], star_line[3], star_line[4],
          star_line[5], star_line[6], star_line[7], star_line[8]);
    }
    SDL_RenderDebugTextFormat(
        r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
        panel.y + static_cast<float>(kModalPanelTextInsetYPx + kModalPanelLineStepPx),
        "Essence %d  |  Base HP %d", game_state_.essence, game_state_.base_hp);

    draw_tab(tab_bazaar, "BAZAAR", inter_level_tab_ == InterLevelTab::Bazaar);
    draw_tab(tab_forge, "FORGE", inter_level_tab_ == InterLevelTab::Forge);
    draw_tab(tab_relics, "RELICS", inter_level_tab_ == InterLevelTab::Relics);
    draw_tab(tab_repair, "REPAIR", inter_level_tab_ == InterLevelTab::Repair);

    const SDL_FRect results_rect = {content_x, content_top, results_w, content_h};
    const SDL_FRect shop_rect = {shop_x, content_top, shop_w, content_h};
    set_color(r, Rgba8{10, 14, 26, 120});
    SDL_RenderFillRect(r, &results_rect);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &results_rect);

    int best_kills = -1;
    CharacterId best_cid = CharacterId::Brix;
    int best_tier = 1;
    for (const RosterEntry &re : game_state_.roster) {
      if (re.kills > best_kills) {
        best_kills = re.kills;
        best_cid = re.character;
        best_tier = re.tier;
      }
    }

    const float rx = results_rect.x + static_cast<float>(kModalPanelTextInsetXPx);
    float ry = results_rect.y + static_cast<float>(kModalPanelTextInsetYPx);
    SDL_RenderDebugTextFormat(r, rx, ry, "RESULTS");
    ry += static_cast<float>(kModalPanelLineStepPx);
    SDL_RenderDebugTextFormat(r, rx, ry, "Killed %d",
                              game_state_.enemies_killed_this_level);
    ry += static_cast<float>(kModalPanelLineStepPx);
    SDL_RenderDebugTextFormat(r, rx, ry, "Essence +%d",
                              game_state_.essence_earned_this_level);
    ry += static_cast<float>(kModalPanelLineStepPx);
    SDL_RenderDebugTextFormat(r, rx, ry, "Time %.1fs",
                              game_state_.level_time_sec);
    ry += static_cast<float>(kModalPanelLineStepPx);
    SDL_RenderDebugTextFormat(r, rx, ry, "Evolutions %d  |  Merges %d",
                              game_state_.evolutions_this_level,
                              game_state_.merges_this_level);
    ry += static_cast<float>(kModalPanelLineStepPx);
    if (quiet_bonus > 0) {
      SDL_RenderDebugTextFormat(r, rx, ry, "The Quiet +%d stars (cosmetic)",
                                quiet_bonus);
      ry += static_cast<float>(kModalPanelLineStepPx);
    }
    if (game_state_.isRelicEquipped(RelicId::ShardHunger)) {
      const int bonus_shards = std::max(
          0, (std::max(0, game_state_.enemies_killed_this_level) /
                  relics::kShardHungerKillsStep) *
                 relics::kShardHungerBonusShardsPer100Kills);
      if (bonus_shards > 0) {
        SDL_RenderDebugTextFormat(r, rx, ry, "Shard Hunger +%d shards",
                                  bonus_shards);
        ry += static_cast<float>(kModalPanelLineStepPx);
      }
    }
    SDL_RenderDebugTextFormat(r, rx, ry, "Best %.*s  T%d  %d kills",
                              static_cast<int>(to_string(best_cid).size()),
                              to_string(best_cid).data(), best_tier,
                              std::max(0, best_kills));

    set_color(r, Rgba8{10, 14, 26, 90});
    SDL_RenderFillRect(r, &shop_rect);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &shop_rect);

    if (inter_level_tab_ == InterLevelTab::Bazaar) {
      const SDL_FRect reroll_rect = {shop_rect.x + shop_rect.w - 144.0f,
                                     shop_rect.y, 144.0f, 28.0f};
      const bool reroll_hover = point_in_rect(mx, my, reroll_rect);
      const bool reroll_unlocked =
          game_state_.player_level >= meta::kPlayerLevelUnlockReroll;
      const bool can_reroll =
          reroll_unlocked && !bazaar_rerolled_ &&
          game_state_.essence >= inter_level_shop::kRerollCostEssence;
      Rgba8 reroll_color = can_reroll ? (reroll_hover ? kModalButtonHoverColor
                                                      : kModalButtonColor)
                                      : Rgba8{20, 26, 42, 140};
      set_color(r, reroll_color);
      SDL_RenderFillRect(r, &reroll_rect);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &reroll_rect);
      set_color(r, kModalButtonTextColor);
      SDL_RenderDebugTextFormat(
          r, reroll_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
          reroll_rect.y + static_cast<float>(kModalButtonTextInsetYPx),
          !reroll_unlocked ? "REROLL (PL%d)" : (bazaar_rerolled_ ? "REROLLED" : "REROLL (%d)"),
          !reroll_unlocked ? meta::kPlayerLevelUnlockReroll
                           : inter_level_shop::kRerollCostEssence);

      const float offer_top = shop_rect.y + 40.0f;
      const float gap = 12.0f;
      const float card_w = std::max(0.0f, (shop_rect.w - gap) * 0.5f);
      const float card_h = 132.0f;

      for (std::size_t i = 0; i < bazaar_offers_.size(); ++i) {
        const int row = static_cast<int>(i / 2);
        const int col = static_cast<int>(i % 2);
        SDL_FRect card = {
            shop_rect.x + static_cast<float>(col) * (card_w + gap),
            offer_top + static_cast<float>(row) * (card_h + gap), card_w,
            card_h};
        const BazaarOffer &offer = bazaar_offers_[i];
        const bool hover = point_in_rect(mx, my, card);
        const bool afford = game_state_.essence >= offer.cost_essence;
        Rgba8 base =
            offer.purchased
                ? Rgba8{14, 18, 30, 160}
                : (!afford ? Rgba8{14, 18, 30, 190}
                           : (hover ? kWaveShopCardHoverColor : kWaveShopCardColor));
        set_color(r, base);
        SDL_RenderFillRect(r, &card);
        set_color(r, kWaveShopCardBorderColor);
        SDL_RenderRect(r, &card);

        set_color(r, kHudTextColor);
        const float tx = card.x + static_cast<float>(kWaveShopCardTextInsetXPx);
        float ty = card.y + static_cast<float>(kWaveShopCardTextInsetYPx);
        SDL_RenderDebugTextFormat(
            r, tx, ty, "%.*s (%s)",
            static_cast<int>(to_string(offer.character).size()),
            to_string(offer.character).data(), rarity_name(offer.rarity));
        ty += static_cast<float>(kModalPanelLineStepPx);
        if (offer.purchased) {
          SDL_RenderDebugTextFormat(r, tx, ty, "Purchased");
        } else if (!afford) {
          SDL_RenderDebugTextFormat(r, tx, ty, "Cost %d  (Need %d)",
                                    offer.cost_essence,
                                    offer.cost_essence - game_state_.essence);
        } else {
          SDL_RenderDebugTextFormat(r, tx, ty, "Cost %d  | Click to buy",
                                    offer.cost_essence);
        }
      }
    } else if (inter_level_tab_ == InterLevelTab::Forge) {
      set_color(r, kHudTextColor);
      SDL_RenderDebugTextFormat(
          r, shop_rect.x + static_cast<float>(kModalPanelTextInsetXPx),
          shop_rect.y + static_cast<float>(kModalPanelTextInsetYPx), "FORGE");

      std::array<bool, static_cast<std::size_t>(CharacterId::Count)> seen{};
      seen.fill(false);
      std::vector<CharacterId> owned;
      owned.reserve(game_state_.roster.size());
      for (const RosterEntry &re : game_state_.roster) {
        const std::size_t idx = static_cast<std::size_t>(re.character);
        if (idx >= seen.size() || seen[idx]) {
          continue;
        }
        seen[idx] = true;
        owned.push_back(re.character);
      }
      if (owned.empty()) {
        owned.push_back(CharacterId::Brix);
      }

      auto rank_for = [&](CharacterId cid, UpgradeNode node, int max_rank) -> int {
        int v = 0;
        for (const RosterEntry &re : game_state_.roster) {
          if (re.character != cid) {
            continue;
          }
          v = std::max<int>(v, re.upgrades[static_cast<std::size_t>(node)]);
        }
        return std::clamp(v, 0, std::max(0, max_rank));
      };

      const int lvl = std::max(1, level_manager_.levelDef().level_number);
      const float level_mod = 1.0f + static_cast<float>(lvl) * 0.05f;

      const float list_w = 150.0f;
      const float list_x = shop_rect.x;
      const float list_y = shop_rect.y + 36.0f;
      const float row_h = 28.0f;
      const float row_gap = 6.0f;
      for (std::size_t i = 0; i < owned.size(); ++i) {
        const SDL_FRect br = {list_x, list_y + static_cast<float>(i) * (row_h + row_gap),
                              list_w, row_h};
        const bool hover = point_in_rect(mx, my, br);
        const bool active = owned[i] == forge_selected_;
        set_color(r, (active || hover) ? kModalButtonHoverColor : kModalButtonColor);
        SDL_RenderFillRect(r, &br);
        set_color(r, kHudBorderColor);
        SDL_RenderRect(r, &br);
        set_color(r, kModalButtonTextColor);
        SDL_RenderDebugTextFormat(
            r, br.x + static_cast<float>(kModalButtonTextInsetXPx),
            br.y + static_cast<float>(kModalButtonTextInsetYPx), "%.*s",
            static_cast<int>(to_string(owned[i]).size()), to_string(owned[i]).data());
      }

      const float nodes_x = shop_rect.x + list_w + inner_pad;
      float ny = shop_rect.y + 36.0f;
      const float nodes_w = std::max(0.0f, shop_rect.w - list_w - inner_pad);

      struct NodeRow final {
        const char *label;
        UpgradeNode node;
        int max_rank;
      };
      constexpr std::array<NodeRow, 5> kNodes{
          NodeRow{"Strike", UpgradeNode::Strike, inter_level_shop::kUpgradeStrikeMaxRanks},
          NodeRow{"Vitality", UpgradeNode::Vitality, inter_level_shop::kUpgradeVitalityMaxRanks},
          NodeRow{"Reach", UpgradeNode::Reach, inter_level_shop::kUpgradeReachMaxRanks},
          NodeRow{"Tempo", UpgradeNode::Tempo, inter_level_shop::kUpgradeTempoMaxRanks},
          NodeRow{"Signature", UpgradeNode::Signature, inter_level_shop::kUpgradeSignatureMaxRanks},
      };

      for (const NodeRow &row : kNodes) {
        const int cur = rank_for(forge_selected_, row.node, row.max_rank);
        const bool maxed = cur >= row.max_rank;
        const float raw = static_cast<float>((cur + 1) * inter_level_shop::kUpgradeCostBase) *
                          level_mod;
        const int cost = std::max(0, static_cast<int>(std::lround(raw)));
        SDL_FRect rr = {nodes_x, ny, nodes_w, 32.0f};
        const bool hover = point_in_rect(mx, my, rr);
        set_color(r, maxed ? Rgba8{14, 18, 30, 150}
                           : (hover ? kWaveShopCardHoverColor : kWaveShopCardColor));
        SDL_RenderFillRect(r, &rr);
        set_color(r, kWaveShopCardBorderColor);
        SDL_RenderRect(r, &rr);
        set_color(r, kHudTextColor);
        SDL_RenderDebugTextFormat(
            r, rr.x + static_cast<float>(kWaveShopCardTextInsetXPx),
            rr.y + static_cast<float>(kModalButtonTextInsetYPx),
            "%s  %d/%d   %s", row.label, cur, row.max_rank,
            maxed ? "MAX" : "");
        if (!maxed) {
          SDL_RenderDebugTextFormat(
              r, rr.x + nodes_w - 150.0f,
              rr.y + static_cast<float>(kModalButtonTextInsetYPx), "Cost %d",
              cost);
        }
        ny += 42.0f;
      }
    } else if (inter_level_tab_ == InterLevelTab::Relics) {
      set_color(r, kHudTextColor);
      SDL_RenderDebugTextFormat(
          r, shop_rect.x + static_cast<float>(kModalPanelTextInsetXPx),
          shop_rect.y + static_cast<float>(kModalPanelTextInsetYPx), "RELICS");
      SDL_RenderDebugTextFormat(
          r, shop_rect.x + static_cast<float>(kModalPanelTextInsetXPx),
          shop_rect.y +
              static_cast<float>(kModalPanelTextInsetYPx + kModalPanelLineStepPx),
          "Shards %d  |  Slots %d/%d", game_state_.shards,
          RelicSystem::unlockedSlotCount(game_state_.player_level),
          relics::kSlotCount);

      const float help_y =
          shop_rect.y + static_cast<float>(kModalPanelTextInsetYPx) +
          2.0f * static_cast<float>(kModalPanelLineStepPx);
      if (relic_pick_ != RelicId::None) {
        const std::string_view nm = to_string(relic_pick_);
        SDL_RenderDebugTextFormat(
            r, shop_rect.x + static_cast<float>(kModalPanelTextInsetXPx), help_y,
            "Selected: %.*s  (click a slot to equip)", static_cast<int>(nm.size()),
            nm.data());
      } else {
        SDL_RenderDebugTextFormat(
            r, shop_rect.x + static_cast<float>(kModalPanelTextInsetXPx), help_y,
            "Click a relic, then a slot. Click a slot to unequip. Unlock in ARMORY.");
      }

      const float slots_y = shop_rect.y + 80.0f;
      const float slot_h = 38.0f;
      const float slot_gap = 10.0f;
      const float slot_w =
          std::max(0.0f, (shop_rect.w - 2.0f * slot_gap) / 3.0f);

      for (int s = 0; s < relics::kSlotCount; ++s) {
        SDL_FRect sr = {shop_rect.x + static_cast<float>(s) * (slot_w + slot_gap),
                        slots_y, slot_w, slot_h};
        const bool hover = point_in_rect(mx, my, sr);
        const bool unlocked =
            RelicSystem::isSlotUnlocked(s, game_state_.player_level);
        set_color(r, unlocked ? (hover ? kWaveShopCardHoverColor : kWaveShopCardColor)
                              : Rgba8{14, 18, 30, 170});
        SDL_RenderFillRect(r, &sr);
        set_color(r, kWaveShopCardBorderColor);
        SDL_RenderRect(r, &sr);

        set_color(r, kHudTextColor);
        const RelicId cur = game_state_.equipped_relics[static_cast<std::size_t>(s)];
        const std::string_view rn = (cur != RelicId::None) ? to_string(cur) : "Empty";
        if (unlocked) {
          SDL_RenderDebugTextFormat(
              r, sr.x + static_cast<float>(kWaveShopCardTextInsetXPx),
              sr.y + static_cast<float>(kModalButtonTextInsetYPx), "Slot %d: %.*s",
              s + 1, static_cast<int>(rn.size()), rn.data());
        } else {
          const int need_lvl =
              (s == 1) ? relics::kSlot2UnlockPlayerLevel : relics::kSlot3UnlockPlayerLevel;
          SDL_RenderDebugTextFormat(
              r, sr.x + static_cast<float>(kWaveShopCardTextInsetXPx),
              sr.y + static_cast<float>(kModalButtonTextInsetYPx), "Slot %d: Locked (Lv %d)",
              s + 1, need_lvl);
        }
      }

      const float list_top = slots_y + slot_h + 24.0f;
      const float card_h = 60.0f;
      const float gap = 10.0f;
      const int cols = 2;
      const float card_w =
          cols > 0 ? std::max(0.0f, (shop_rect.w - gap) / 2.0f) : 0.0f;

      for (std::size_t i = 0; i < RelicSystem::kRelicCount; ++i) {
        const RelicId id = static_cast<RelicId>(i);
        const int row = static_cast<int>(i / static_cast<std::size_t>(cols));
        const int col = static_cast<int>(i % static_cast<std::size_t>(cols));
        SDL_FRect rr = {shop_rect.x + static_cast<float>(col) * (card_w + gap),
                        list_top + static_cast<float>(row) * (card_h + gap), card_w,
                        card_h};
        const bool hover = point_in_rect(mx, my, rr);

        const RelicDef &def = RelicSystem::def(id);
        const bool unlocked = game_state_.isRelicUnlocked(id);
        const bool equipped = game_state_.isRelicEquipped(id);
        const bool picked = relic_pick_ == id;
        const int cost = std::max(0, def.shard_cost);

        Rgba8 bg = unlocked ? (hover ? kWaveShopCardHoverColor : kWaveShopCardColor)
                            : (hover ? Rgba8{18, 22, 36, 200}
                                     : Rgba8{14, 18, 30, 190});
        set_color(r, bg);
        SDL_RenderFillRect(r, &rr);
        set_color(r, (equipped || picked) ? kModalButtonHoverColor
                                          : kWaveShopCardBorderColor);
        SDL_RenderRect(r, &rr);

        set_color(r, kHudTextColor);
        const float tx = rr.x + static_cast<float>(kWaveShopCardTextInsetXPx);
        float ty = rr.y + static_cast<float>(kWaveShopCardTextInsetYPx);
        const std::string_view nm = to_string(id);
        SDL_RenderDebugTextFormat(r, tx, ty, "%.*s", static_cast<int>(nm.size()),
                                  nm.data());
        ty += static_cast<float>(kModalPanelLineStepPx);
        if (!unlocked) {
          SDL_RenderDebugTextFormat(
              r, tx, ty, "Locked  Cost %d  (Armory)", cost);
        } else if (equipped) {
          SDL_RenderDebugTextFormat(r, tx, ty, "Equipped");
        } else {
          SDL_RenderDebugTextFormat(r, tx, ty, "Unlocked");
        }
      }
    } else if (inter_level_tab_ == InterLevelTab::Repair) {
      set_color(r, kHudTextColor);
      SDL_RenderDebugTextFormat(
          r, shop_rect.x + static_cast<float>(kModalPanelTextInsetXPx),
          shop_rect.y + static_cast<float>(kModalPanelTextInsetYPx), "REPAIR");

      const int base_now = std::max(0, game_state_.base_hp);
      const int base_max_next = level::kBaseHp;
      int next_target = base_now;
      if (game_state_.next_level_base_hp_target !=
          std::numeric_limits<std::int32_t>::min()) {
        next_target = game_state_.next_level_base_hp_target;
      }
      const int next_hp =
          std::clamp<std::int32_t>(next_target, 0, base_max_next);
      SDL_RenderDebugTextFormat(
          r, shop_rect.x + static_cast<float>(kModalPanelTextInsetXPx),
          shop_rect.y + static_cast<float>(kModalPanelTextInsetYPx + kModalPanelLineStepPx),
          "Next level start HP: %d / %d", next_hp, base_max_next);

      struct Opt final {
        const char *label;
        int cost;
        int add;
        bool full;
      };
      constexpr std::array<Opt, 3> kOpts{
          Opt{"+20 HP", inter_level_shop::kRepairRestore20Cost,
              inter_level_shop::kRepairRestore20Hp, false},
          Opt{"+50 HP", inter_level_shop::kRepairRestore50Cost,
              inter_level_shop::kRepairRestore50Hp, false},
          Opt{"FULL", inter_level_shop::kRepairFullRestoreCost,
              inter_level_shop::kRepairFullRestoreHp, true},
      };

      const float opt_x = shop_rect.x;
      float opt_y = shop_rect.y + 60.0f;
      const float opt_w = shop_rect.w;
      for (std::size_t i = 0; i < kOpts.size(); ++i) {
        SDL_FRect rr = {opt_x, opt_y, opt_w, 34.0f};
        const bool hover = point_in_rect(mx, my, rr);
        const bool afford = game_state_.essence >= kOpts[i].cost;
        set_color(r, repair_purchased_ ? Rgba8{14, 18, 30, 150}
                                       : (!afford ? Rgba8{14, 18, 30, 190}
                                                  : (hover ? kWaveShopCardHoverColor
                                                           : kWaveShopCardColor)));
        SDL_RenderFillRect(r, &rr);
        set_color(r, kWaveShopCardBorderColor);
        SDL_RenderRect(r, &rr);
        set_color(r, kHudTextColor);
        SDL_RenderDebugTextFormat(
            r, rr.x + static_cast<float>(kWaveShopCardTextInsetXPx),
            rr.y + static_cast<float>(kModalButtonTextInsetYPx),
            "%s  (Cost %d)%s", kOpts[i].label, kOpts[i].cost,
            repair_purchased_ ? "  Purchased" : "");
        opt_y += 44.0f;
      }
    }

    SDL_FRect roster = {panel_x + inner_pad, roster_y,
                        std::max(0.0f, panel_w - 2.0f * inner_pad), roster_h};
    set_color(r, Rgba8{10, 14, 26, 120});
    SDL_RenderFillRect(r, &roster);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &roster);

    const float cell = static_cast<float>(kCreatureBaseSizePx);
    const float gap = 8.0f;
    float cx = roster.x + 10.0f;
    for (std::size_t i = 0; i < game_state_.roster.size(); ++i) {
      if (cx + cell > roster.x + roster.w - 10.0f) {
        break;
      }
      const RosterEntry &re = game_state_.roster[i];
      SDL_FRect c = {cx, roster.y + (roster.h - cell) * 0.5f, cell, cell};
      set_color(r, character_color(re.character));
      SDL_RenderFillRect(r, &c);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &c);
      set_color(r, kHudTextColor);
      SDL_RenderDebugTextFormat(r, c.x + 4.0f, c.y + 4.0f, "%d", re.tier);
      cx += cell + gap;
    }

    draw_button(btn_levels, "LEVELS");
    draw_button(btn_armory, "ARMORY");
    draw_button(btn_replay, "REPLAY");
    draw_button(btn_next, "NEXT");

    if (show_bazaar_duplicate_confirm_) {
      SDL_FRect o2 = {0.0f, 0.0f, w, h};
      set_color(r, Rgba8{0, 0, 0, 200});
      SDL_RenderFillRect(r, &o2);

      const float pw = static_cast<float>(kConfirmDialogWidthPx);
      const float ph = static_cast<float>(kConfirmDialogHeightPx);
      const float px = (w - pw) * 0.5f;
      const float py = (h - ph) * 0.5f;
      SDL_FRect cp = {px, py, pw, ph};
      set_color(r, kModalPanelColor);
      SDL_RenderFillRect(r, &cp);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &cp);

      const float bw = static_cast<float>(kConfirmDialogButtonWidthPx);
      const float bh = static_cast<float>(kConfirmDialogButtonHeightPx);
      const float g = static_cast<float>(kConfirmDialogButtonGapPx);
      const float by = py + ph - bh - static_cast<float>(kHudPaddingPx);
      const float bx = px + (pw - (2.0f * bw + g)) * 0.5f;
      const SDL_FRect yes_rect = {bx, by, bw, bh};
      const SDL_FRect no_rect = {bx + bw + g, by, bw, bh};

      set_color(r, point_in_rect(mx, my, yes_rect) ? kModalButtonHoverColor
                                                   : kModalButtonColor);
      SDL_RenderFillRect(r, &yes_rect);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &yes_rect);
      set_color(r, point_in_rect(mx, my, no_rect) ? kModalButtonHoverColor
                                                  : kModalButtonColor);
      SDL_RenderFillRect(r, &no_rect);
      set_color(r, kHudBorderColor);
      SDL_RenderRect(r, &no_rect);
      set_color(r, kModalButtonTextColor);
      SDL_RenderDebugTextFormat(
          r, yes_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
          yes_rect.y + static_cast<float>(kModalButtonTextInsetYPx), "BUY");
      SDL_RenderDebugTextFormat(
          r, no_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
          no_rect.y + static_cast<float>(kModalButtonTextInsetYPx), "CANCEL");

      set_color(r, kHudTextColor);
      if (pending_bazaar_offer_index_ >= 0 &&
          pending_bazaar_offer_index_ <
              static_cast<std::int32_t>(bazaar_offers_.size())) {
        const BazaarOffer &offer =
            bazaar_offers_[static_cast<std::size_t>(pending_bazaar_offer_index_)];
        SDL_RenderDebugTextFormat(
            r, cp.x + static_cast<float>(kModalPanelTextInsetXPx),
            cp.y + static_cast<float>(kModalPanelTextInsetYPx),
            "Already owned. Buy duplicate %.*s?",
            static_cast<int>(to_string(offer.character).size()),
            to_string(offer.character).data());
      } else {
        SDL_RenderDebugTextFormat(
            r, cp.x + static_cast<float>(kModalPanelTextInsetXPx),
            cp.y + static_cast<float>(kModalPanelTextInsetYPx),
            "Already owned. Buy duplicate?");
      }
    }
  } else if (level_state == LevelManagerState::Failed ||
             level_state == LevelManagerState::LevelClear) {
    const float w = engine_->camera.width;
    const float h = engine_->camera.height;

    SDL_FRect overlay = {0.0f, 0.0f, w, h};
    set_color(r, kModalOverlayColor);
    SDL_RenderFillRect(r, &overlay);

    const float pw = static_cast<float>(kConfirmDialogWidthPx);
    const float ph = static_cast<float>(kConfirmDialogHeightPx);
    const float px = (w - pw) * 0.5f;
    const float py = (h - ph) * 0.5f;
    SDL_FRect panel = {px, py, pw, ph};
    set_color(r, kModalPanelColor);
    SDL_RenderFillRect(r, &panel);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &panel);

    const float bw = static_cast<float>(kConfirmDialogButtonWidthPx);
    const float bh = static_cast<float>(kConfirmDialogButtonHeightPx);
    const float gap = static_cast<float>(kConfirmDialogButtonGapPx);

    const float buttons_y = py + ph - bh - static_cast<float>(kHudPaddingPx);
    const float buttons_x = px + (pw - (2.0f * bw + gap)) * 0.5f;
    const SDL_FRect primary_rect = {buttons_x, buttons_y, bw, bh};
    const SDL_FRect secondary_rect = {buttons_x + bw + gap, buttons_y, bw, bh};

    const float mx = static_cast<float>(input.mouseX());
    const float my = static_cast<float>(input.mouseY());

    set_color(r, point_in_rect(mx, my, primary_rect) ? kModalButtonHoverColor
                                                     : kModalButtonColor);
    SDL_RenderFillRect(r, &primary_rect);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &primary_rect);

    set_color(r, point_in_rect(mx, my, secondary_rect) ? kModalButtonHoverColor
                                                       : kModalButtonColor);
    SDL_RenderFillRect(r, &secondary_rect);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &secondary_rect);

    set_color(r, kModalButtonTextColor);
    SDL_RenderDebugTextFormat(
        r, secondary_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
        secondary_rect.y + static_cast<float>(kModalButtonTextInsetYPx),
        "LEVELS");

    set_color(r, kHudTextColor);
    if (level_state == LevelManagerState::Failed) {
      SDL_RenderDebugTextFormat(
          r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
          panel.y + static_cast<float>(kModalPanelTextInsetYPx),
          "LEVEL FAILED");
      SDL_RenderDebugTextFormat(
          r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
          panel.y + static_cast<float>(kModalPanelTextInsetYPx + kModalPanelLineStepPx),
          "Press R/Enter to retry");
      SDL_RenderDebugTextFormat(
          r, primary_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
          primary_rect.y + static_cast<float>(kModalButtonTextInsetYPx),
          "RETRY");
    } else {
      const int stars = level_manager_.lastLevelStars();
      SDL_RenderDebugTextFormat(
          r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
          panel.y + static_cast<float>(kModalPanelTextInsetYPx),
          "LEVEL COMPLETE  (Stars %d)", stars);
      SDL_RenderDebugTextFormat(
          r, panel.x + static_cast<float>(kModalPanelTextInsetXPx),
          panel.y + static_cast<float>(kModalPanelTextInsetYPx + kModalPanelLineStepPx),
          "Press N/Enter for next level");
      SDL_RenderDebugTextFormat(
          r, primary_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
          primary_rect.y + static_cast<float>(kModalButtonTextInsetYPx),
          "NEXT");
    }
  }

  if (game_state_.screen_edge_glow_remaining_sec > 0.0f) {
    const float frac =
        std::clamp(game_state_.screen_edge_glow_remaining_sec /
                       std::max(0.001f, evolution::kScreenEdgeGlowSec),
                   0.0f, 1.0f);
    Rgba8 c = kEvolutionPulseColor;
    c.a = static_cast<std::uint8_t>(
        std::clamp(70.0f * frac, 0.0f, 255.0f));
    SDL_FRect overlay = {0.0f, 0.0f, engine_->camera.width, engine_->camera.height};
    set_color(r, c);
    SDL_RenderFillRect(r, &overlay);
  }

  SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_NONE);
}

} // namespace tower_swarm
