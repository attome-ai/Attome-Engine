#include "screens/HUD.h"

#include "CameraController.h"
#include "Constants.h"
#include "InputManager.h"

#include "ATMEngine.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>

namespace tower_swarm {
namespace {

void set_color(SDL_Renderer *renderer, Rgba8 c) {
  SDL_SetRenderDrawColor(renderer, c.r, c.g, c.b, c.a);
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

void render_top_bar(Engine *engine, bool show_debug_grid) {
  if (!engine || !engine->renderer) {
    return;
  }

  SDL_Renderer *r = engine->renderer;

  SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);

  set_color(r, kHudTopBarColor);
  SDL_FRect bar = {0.0f, 0.0f, engine->camera.width,
                   static_cast<float>(kHudTopBarHeightPx)};
  SDL_RenderFillRect(r, &bar);

  set_color(r, kHudBorderColor);
  SDL_RenderRect(r, &bar);

  set_color(r, kHudTextColor);
  const float text_y =
      std::max(0.0f, (static_cast<float>(kHudTopBarHeightPx) -
                      static_cast<float>(SDL_DEBUG_TEXT_FONT_CHARACTER_SIZE)) *
                         0.5f);

  SDL_RenderDebugTextFormat(
      r, static_cast<float>(kHudPaddingPx), text_y,
      "Tower Swarm | Camera %.0f, %.0f | Grid %s (G) | Pan WASD/Arrows, drag RMB/MMB",
      engine->camera.x, engine->camera.y, show_debug_grid ? "ON" : "OFF");

  SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_NONE);
}

void render_debug_grid(Engine *engine) {
  if (!engine || !engine->renderer) {
    return;
  }

  const float zoom = safe_zoom(engine);

  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  get_camera_world_rect(engine, x1, y1, x2, y2);

  const float world_w = engine->world_bounds.w;
  const float world_h = engine->world_bounds.h;

  x1 = std::clamp(x1, 0.0f, world_w);
  y1 = std::clamp(y1, 0.0f, world_h);
  x2 = std::clamp(x2, 0.0f, world_w);
  y2 = std::clamp(y2, 0.0f, world_h);

  const int first_col =
      std::max(0, static_cast<int>(std::floor(x1 / kTileSizePx)));
  const int last_col =
      std::max(first_col, static_cast<int>(std::ceil(x2 / kTileSizePx)));
  const int first_row =
      std::max(0, static_cast<int>(std::floor(y1 / kTileSizePx)));
  const int last_row =
      std::max(first_row, static_cast<int>(std::ceil(y2 / kTileSizePx)));

  SDL_Renderer *r = engine->renderer;
  SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
  set_color(r, kDebugGridColor);

  const float screen_w = engine->camera.width;
  const float screen_h = engine->camera.height;

  for (int col = first_col; col <= last_col; ++col) {
    const float wx = static_cast<float>(col * kTileSizePx);
    const float sx = (wx - x1) * zoom;
    if (sx < -1.0f || sx > screen_w + 1.0f) {
      continue;
    }
    SDL_RenderLine(r, sx, 0.0f, sx, screen_h);
  }

  for (int row = first_row; row <= last_row; ++row) {
    const float wy = static_cast<float>(row * kTileSizePx);
    const float sy = (wy - y1) * zoom;
    if (sy < -1.0f || sy > screen_h + 1.0f) {
      continue;
    }
    SDL_RenderLine(r, 0.0f, sy, screen_w, sy);
  }

  SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_NONE);
}

void render_base_hp_bar(Engine *engine, int base_hp, int base_hp_max) {
  if (!engine || !engine->renderer || base_hp_max <= 0) {
    return;
  }

  SDL_Renderer *r = engine->renderer;
  SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);

  const float w = static_cast<float>(kBaseHpBarWidthPx);
  const float h = static_cast<float>(kBaseHpBarHeightPx);
  const float x = (engine->camera.width - w) * 0.5f;
  const float y =
      engine->camera.height - h - static_cast<float>(kBaseHpBarMarginBottomPx);

  SDL_FRect outer = {x, y, w, h};
  set_color(r, kBaseHpBarBackColor);
  SDL_RenderFillRect(r, &outer);

  const float frac =
      std::clamp(static_cast<float>(base_hp) / static_cast<float>(base_hp_max),
                 0.0f, 1.0f);
  const float inset = static_cast<float>(kBaseHpBarInsetPx);
  SDL_FRect fill = {x + inset, y + inset, (w - 2.0f * inset) * frac,
                    h - 2.0f * inset};
  set_color(r, kBaseHpBarFillColor);
  SDL_RenderFillRect(r, &fill);

  set_color(r, kBaseHpBarOutlineColor);
  SDL_RenderRect(r, &outer);

  const float marker_x2 =
      x + inset + (w - 2.0f * inset) * level::kStar2Threshold;
  const float marker_x3 =
      x + inset + (w - 2.0f * inset) * level::kStar3Threshold;
  set_color(r, kBaseHpBarMarkerColor);
  SDL_RenderLine(r, marker_x2, y + inset, marker_x2, y + h - inset);
  SDL_RenderLine(r, marker_x3, y + inset, marker_x3, y + h - inset);

  set_color(r, kHudTextColor);
  SDL_RenderDebugTextFormat(
      r, x, y - static_cast<float>(kBaseHpBarLabelOffsetYPx),
      "Base HP %d / %d", base_hp, base_hp_max);

  SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_NONE);
}

} // namespace

void HUD::render(Engine *engine, const CameraController &camera,
                 const InputManager &input, bool show_debug_grid, int base_hp,
                 int base_hp_max) {
  (void)camera;
  (void)input;

  if (show_debug_grid) {
    render_debug_grid(engine);
  }
  render_top_bar(engine, show_debug_grid);
  render_base_hp_bar(engine, base_hp, base_hp_max);
}

} // namespace tower_swarm
