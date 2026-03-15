#include "CameraController.h"

#include "Constants.h"
#include "InputManager.h"

#include "ATMEngine.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>

namespace tower_swarm {
namespace {

float safe_zoom(const Engine *engine) {
  if (!engine || engine->camera.zoom <= kMinCameraZoomEpsilon) {
    return 1.0f;
  }
  return engine->camera.zoom;
}

void clamp_camera_center(const Engine *engine, float zoom, int world_width_px,
                         int world_height_px, float &x, float &y) {
  if (!engine) {
    return;
  }

  const float half_w = (engine->camera.width / zoom) * 0.5f;
  const float half_h = (engine->camera.height / zoom) * 0.5f;

  const float world_w = static_cast<float>(world_width_px);
  const float world_h = static_cast<float>(world_height_px);

  if (world_w <= 2.0f * half_w) {
    x = world_w * 0.5f;
  } else {
    x = std::clamp(x, half_w, world_w - half_w);
  }

  if (world_h <= 2.0f * half_h) {
    y = world_h * 0.5f;
  } else {
    y = std::clamp(y, half_h, world_h - half_h);
  }
}

float smooth_alpha(float rate, float dt) {
  if (dt <= 0.0f) {
    return 1.0f;
  }
  const float t = 1.0f - std::exp(-rate * dt);
  return std::clamp(t, 0.0f, 1.0f);
}

} // namespace

void CameraController::initialize(Engine *engine, int world_width_px,
                                  int world_height_px) {
  if (!engine) {
    return;
  }

  engine->camera.zoom = kCameraDefaultZoom;

  target_x_ = static_cast<float>(world_width_px) * 0.5f;
  target_y_ = static_cast<float>(world_height_px) * 0.5f;

  clamp_camera_center(engine, safe_zoom(engine), world_width_px, world_height_px,
                      target_x_, target_y_);
  engine->camera.x = target_x_;
  engine->camera.y = target_y_;
}

void CameraController::tick(float dt, const InputManager &input, Engine *engine,
                            int world_width_px, int world_height_px) {
  if (!engine) {
    return;
  }

  const float zoom = safe_zoom(engine);
  const float pan = (kCameraPanSpeedPxPerSec * std::max(dt, 0.0f)) / zoom;

  float next_target_x = target_x_;
  float next_target_y = target_y_;

  if (input.isDown(SDL_SCANCODE_A) || input.isDown(SDL_SCANCODE_LEFT)) {
    next_target_x -= pan;
  }
  if (input.isDown(SDL_SCANCODE_D) || input.isDown(SDL_SCANCODE_RIGHT)) {
    next_target_x += pan;
  }
  if (input.isDown(SDL_SCANCODE_W) || input.isDown(SDL_SCANCODE_UP)) {
    next_target_y -= pan;
  }
  if (input.isDown(SDL_SCANCODE_S) || input.isDown(SDL_SCANCODE_DOWN)) {
    next_target_y += pan;
  }

  if (input.isMouseDown(SDL_BUTTON_MIDDLE) ||
      input.isMouseDown(SDL_BUTTON_RIGHT)) {
    next_target_x -= static_cast<float>(input.mouseDeltaX()) / zoom;
    next_target_y -= static_cast<float>(input.mouseDeltaY()) / zoom;
  }

  clamp_camera_center(engine, zoom, world_width_px, world_height_px,
                      next_target_x, next_target_y);

  target_x_ = next_target_x;
  target_y_ = next_target_y;

  const float a = smooth_alpha(kCameraSmoothRate, dt);
  engine->camera.x = engine->camera.x + (target_x_ - engine->camera.x) * a;
  engine->camera.y = engine->camera.y + (target_y_ - engine->camera.y) * a;

  clamp_camera_center(engine, zoom, world_width_px, world_height_px,
                      engine->camera.x, engine->camera.y);
}

} // namespace tower_swarm
