	// GameUI.h
#pragma once

#include "GameState.h"  // Will define the GameState struct and enums
#include "imgui.h"

// UI initialization and cleanup
void ui_init(SDL_Window* window, SDL_Renderer* renderer);
void ui_shutdown();

// Process events for ImGui
bool ui_process_event(SDL_Event* event);

// Start a new ImGui frame
void ui_new_frame();

// Render different UI states
void ui_render_title_screen(GameState* state);
void ui_render_ranked_screen(GameState* state);
void ui_render_shop_screen(GameState* state);
void ui_render_settings_screen(GameState* state);
void ui_render_pause_menu(GameState* state);
void ui_render_game_over(GameState* state);
void ui_render_in_game_hud(GameState* state);

// Main UI render function that dispatches to the appropriate screen based on game state
void ui_render(GameState* state);
