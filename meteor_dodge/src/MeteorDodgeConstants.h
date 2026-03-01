#pragma once

namespace meteor_dodge {

inline constexpr int WINDOW_WIDTH = 960;
inline constexpr int WINDOW_HEIGHT = 560;
inline constexpr int WORLD_WIDTH = 2000;
inline constexpr int WORLD_HEIGHT = 1200;
inline constexpr int GRID_CELL_SIZE = 64;

inline constexpr int PLAYER_WIDTH = 58;
inline constexpr int PLAYER_HEIGHT = 24;
inline constexpr float PLAYER_SPEED = 540.0f;

inline constexpr int METEOR_WIDTH = 24;
inline constexpr int METEOR_HEIGHT = 24;
inline constexpr int INITIAL_METEOR_COUNT = 44;
inline constexpr int MAX_METEOR_COUNT = 80;

inline constexpr float METEOR_MIN_SPEED = 140.0f;
inline constexpr float METEOR_MAX_SPEED = 320.0f;
inline constexpr float METEOR_DRIFT_RANGE = 68.0f;

inline constexpr int TITLE_UPDATE_MS = 180;

} // namespace meteor_dodge
