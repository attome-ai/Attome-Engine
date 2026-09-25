#pragma once

// Game tuning values. Each ATM_TUNABLE is a mutable global whose compiled-in
// value is the default; config/meteor_dodge.json overrides it at startup and
// while the game runs (JSON keys: "meteor_dodge.<NAME>").
//
// Read once at startup (need a restart): WINDOW_*, WORLD_*, GRID_CELL_SIZE,
// PLAYER_WIDTH/HEIGHT, METEOR_WIDTH/HEIGHT, INITIAL_METEOR_COUNT,
// MAX_METEOR_COUNT. Speeds and drift apply live (new meteor speeds are picked
// on respawn).

#include "ATMConfig.h"

namespace meteor_dodge {
ATM_TUNABLE_SECTION("meteor_dodge");

ATM_TUNABLE(int, WINDOW_WIDTH, 960);
ATM_TUNABLE(int, WINDOW_HEIGHT, 560);
ATM_TUNABLE(int, WORLD_WIDTH, 2000);
ATM_TUNABLE(int, WORLD_HEIGHT, 1200);
ATM_TUNABLE(int, GRID_CELL_SIZE, 64);

ATM_TUNABLE(int, PLAYER_WIDTH, 58);
ATM_TUNABLE(int, PLAYER_HEIGHT, 24);
ATM_TUNABLE(float, PLAYER_SPEED, 540.0f);

ATM_TUNABLE(int, METEOR_WIDTH, 24);
ATM_TUNABLE(int, METEOR_HEIGHT, 24);
ATM_TUNABLE(int, INITIAL_METEOR_COUNT, 44);
ATM_TUNABLE(int, MAX_METEOR_COUNT, 80);

ATM_TUNABLE(float, METEOR_MIN_SPEED, 140.0f);
ATM_TUNABLE(float, METEOR_MAX_SPEED, 320.0f);
ATM_TUNABLE(float, METEOR_DRIFT_RANGE, 68.0f);

ATM_TUNABLE(int, TITLE_UPDATE_MS, 180);

} // namespace meteor_dodge
