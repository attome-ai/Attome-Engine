#pragma once

struct Engine;

namespace tower_swarm {

class CameraController;
class InputManager;

class HUD final {
public:
  void render(Engine *engine, const CameraController &camera,
              const InputManager &input, bool show_debug_grid, int base_hp,
              int base_hp_max);
};

} // namespace tower_swarm
