#pragma once

#include "levels/GameState.h"

#include <array>
#include <cstdint>

struct SDL_Renderer;

namespace tower_swarm {

class InputManager;

struct WaveBuffCardDef final {
  WaveBuffId id{WaveBuffId::Surge};
  const char *name{""};
  const char *description{""};
  std::int32_t duration_waves{0};
};

class WaveBuffShop final {
public:
  void open(std::uint32_t seed);
  void close();
  bool isOpen() const { return open_; }

  bool tick(const InputManager &input, float screen_w, float screen_h);
  void render(SDL_Renderer *r, const InputManager &input, float screen_w,
              float screen_h, float timer_sec) const;

  bool consumeSelection(WaveBuffId &out_selected, bool &out_skipped);

  std::array<WaveBuffId, 3> currentDraw() const { return draw_; }

  static const WaveBuffCardDef &def(WaveBuffId id);
  static Rgba8 iconColor(WaveBuffId id);
  static const char *iconGlyph(WaveBuffId id);

private:
  void randomizeDraw(std::uint32_t seed);

  bool open_{false};
  std::array<WaveBuffId, 3> draw_{};
  std::int32_t hovered_index_{-1};

  bool result_ready_{false};
  WaveBuffId selected_{WaveBuffId::Surge};
  bool skipped_{false};
};

} // namespace tower_swarm
