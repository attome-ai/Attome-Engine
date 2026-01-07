#pragma once

#include "ATMEngine/GameState.h"

// ImGui initialization and shutdown
void init_imgui(GameState* state);
void shutdown_imgui();

// UI rendering functions for different game states
void render_auth_ui(GameState* state);
void render_title_ui(GameState* state);
void render_game_ui(GameState* state);
void render_pause_ui(GameState* state);
void render_game_over_ui(GameState* state);
