#include "InputManager.h"

#include <algorithm>
#include <cmath>

namespace tower_swarm {
namespace {

std::size_t clamp_mouse_button_index(std::uint8_t button) {
  if (button == 0) {
    return 0;
  }
  std::size_t idx = static_cast<std::size_t>(button);
  if (idx >= InputManager::kMouseButtonSlots) {
    idx = InputManager::kMouseButtonSlots - 1;
  }
  return idx;
}

} // namespace

void InputManager::beginFrame() {
  pressed_.fill(0);
  released_.fill(0);
  mouse_pressed_.fill(0);
  mouse_released_.fill(0);
  mouse_dx_ = 0;
  mouse_dy_ = 0;
  wheel_y_ = 0.0f;
}

void InputManager::handleEvent(const SDL_Event &event) {
  switch (event.type) {
  case SDL_EVENT_KEY_DOWN: {
    const SDL_Scancode sc = event.key.scancode;
    if (sc >= 0 && sc < SDL_SCANCODE_COUNT) {
      const std::size_t idx = static_cast<std::size_t>(sc);
      if (!event.key.repeat && down_[idx] == 0) {
        pressed_[idx] = 1;
      }
      down_[idx] = 1;
    }
    break;
  }
  case SDL_EVENT_KEY_UP: {
    const SDL_Scancode sc = event.key.scancode;
    if (sc >= 0 && sc < SDL_SCANCODE_COUNT) {
      const std::size_t idx = static_cast<std::size_t>(sc);
      released_[idx] = 1;
      down_[idx] = 0;
    }
    break;
  }
  case SDL_EVENT_MOUSE_MOTION: {
    mouse_x_ = static_cast<int>(std::lround(event.motion.x));
    mouse_y_ = static_cast<int>(std::lround(event.motion.y));
    mouse_dx_ += static_cast<int>(std::lround(event.motion.xrel));
    mouse_dy_ += static_cast<int>(std::lround(event.motion.yrel));
    break;
  }
  case SDL_EVENT_MOUSE_BUTTON_DOWN: {
    const std::size_t idx = clamp_mouse_button_index(event.button.button);
    mouse_x_ = static_cast<int>(std::lround(event.button.x));
    mouse_y_ = static_cast<int>(std::lround(event.button.y));
    if (mouse_down_[idx] == 0) {
      mouse_pressed_[idx] = 1;
    }
    mouse_down_[idx] = 1;
    break;
  }
  case SDL_EVENT_MOUSE_BUTTON_UP: {
    const std::size_t idx = clamp_mouse_button_index(event.button.button);
    mouse_x_ = static_cast<int>(std::lround(event.button.x));
    mouse_y_ = static_cast<int>(std::lround(event.button.y));
    mouse_released_[idx] = 1;
    mouse_down_[idx] = 0;
    break;
  }
  case SDL_EVENT_MOUSE_WHEEL: {
    wheel_y_ += event.wheel.y;
    break;
  }
  default:
    break;
  }
}

bool InputManager::isDown(SDL_Scancode scancode) const {
  if (scancode < 0 || scancode >= SDL_SCANCODE_COUNT) {
    return false;
  }
  return down_[static_cast<std::size_t>(scancode)] != 0;
}

bool InputManager::wasPressed(SDL_Scancode scancode) const {
  if (scancode < 0 || scancode >= SDL_SCANCODE_COUNT) {
    return false;
  }
  return pressed_[static_cast<std::size_t>(scancode)] != 0;
}

bool InputManager::wasReleased(SDL_Scancode scancode) const {
  if (scancode < 0 || scancode >= SDL_SCANCODE_COUNT) {
    return false;
  }
  return released_[static_cast<std::size_t>(scancode)] != 0;
}

bool InputManager::isMouseDown(std::uint8_t button) const {
  return mouse_down_[clamp_mouse_button_index(button)] != 0;
}

bool InputManager::wasMousePressed(std::uint8_t button) const {
  return mouse_pressed_[clamp_mouse_button_index(button)] != 0;
}

bool InputManager::wasMouseReleased(std::uint8_t button) const {
  return mouse_released_[clamp_mouse_button_index(button)] != 0;
}

} // namespace tower_swarm

