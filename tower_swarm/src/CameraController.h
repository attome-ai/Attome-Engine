#pragma once

struct Engine;

namespace tower_swarm {

class InputManager;

class CameraController final {
public:
  void initialize(Engine *engine, int world_width_px, int world_height_px);
  void tick(float dt, const InputManager &input, Engine *engine,
            int world_width_px, int world_height_px);

  float targetX() const { return target_x_; }
  float targetY() const { return target_y_; }

private:
  float target_x_{0.0f};
  float target_y_{0.0f};
};

} // namespace tower_swarm
