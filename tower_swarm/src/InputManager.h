#pragma once

#include <SDL3/SDL.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace tower_swarm {

class InputManager final {
public:
  static constexpr std::size_t kMouseButtonSlots = 8;

  void beginFrame();
  void handleEvent(const SDL_Event &event);

  bool isDown(SDL_Scancode scancode) const;
  bool wasPressed(SDL_Scancode scancode) const;
  bool wasReleased(SDL_Scancode scancode) const;

  bool isMouseDown(std::uint8_t button) const;
  bool wasMousePressed(std::uint8_t button) const;
  bool wasMouseReleased(std::uint8_t button) const;

  int mouseX() const { return mouse_x_; }
  int mouseY() const { return mouse_y_; }
  int mouseDeltaX() const { return mouse_dx_; }
  int mouseDeltaY() const { return mouse_dy_; }
  float wheelY() const { return wheel_y_; }

private:
  std::array<std::uint8_t, SDL_SCANCODE_COUNT> down_{};
  std::array<std::uint8_t, SDL_SCANCODE_COUNT> pressed_{};
  std::array<std::uint8_t, SDL_SCANCODE_COUNT> released_{};

  std::array<std::uint8_t, kMouseButtonSlots> mouse_down_{};
  std::array<std::uint8_t, kMouseButtonSlots> mouse_pressed_{};
  std::array<std::uint8_t, kMouseButtonSlots> mouse_released_{};

  int mouse_x_{0};
  int mouse_y_{0};
  int mouse_dx_{0};
  int mouse_dy_{0};
  float wheel_y_{0.0f};
};

} // namespace tower_swarm

