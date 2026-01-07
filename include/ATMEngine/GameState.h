#pragma once

#include <SDL3/SDL.h>
#include "ATMEngine.h"
#include "ATMNetwork.h"
#include "ATMProtocol.h"

// Game state enum for UI flow
enum class GameStateEnum {
    GAME_STATE_AUTH,
    GAME_STATE_TITLE,
    GAME_STATE_PLAYING,
    GAME_STATE_PAUSED,
    GAME_STATE_GAME_OVER
} ;

// Player entity data structure (used only for player)
struct PlayerData {
    float speed;
    int current_frame;
    int texture_ids[3];
    Uint64 animation_timer;
    SDL_Scancode keys_pressed[4];
} ;

// Generic entity data structure (used by all test entity types)
struct GenericEntityData {
    float speed;
    float direction[2];   // Normalized direction vector
    Uint8 color[3];       // RGB color values
    int behavior_flags;   // Flags to control entity behavior
} ;

// Main game state structure
struct GameState {
    Engine* engine;
    int player_entity;
    int player_type_id;
    int score;
    bool game_over;
    Uint64 last_spawn_time;

    // Game state and UI
    GameStateEnum current_state;
    float player_health;
    int max_health;
    Uint64 game_time_start;
    bool show_controls;
    bool show_debug_info;

    // Animation variables for UI
    float title_animation;
    float score_animation;
    float health_animation;
    float damage_flash;

    // Tracking for entity types and textures
    int num_entity_types;
    int* entity_type_ids;
    int* entity_counts;
    int* entity_texture_ids;
    Uint64* entity_update_times;

    // ImGui-related fields
    float fps;
    float fps_history[100];
    int fps_history_index;

    // Authentication-related fields
    UDPNode network;

    SocketAddress serverAddr;
    bool isConnected;
    uint16_t userId;
    char errorMessage[256];
    bool isAuthenticating;
    bool showLoginForm;
    bool showRegisterForm;
    char loginUsername[MAX_USERNAME_LENGTH];
    char loginPassword[MAX_PASSWORD_LENGTH];
    char registerUsername[MAX_USERNAME_LENGTH];
    char registerPassword[MAX_PASSWORD_LENGTH];
    char loggedInUsername[MAX_USERNAME_LENGTH];
    bool isGuest;
    Uint64 lastHeartbeatTime;

    GameState()
    {
        // Initialize ImGui-related fields in game state
        fps = 0.0f;
        fps_history_index = 0;
        for (int i = 0; i < 100; i++) {
           fps_history[i] = 0.0f;
        }

        // Initialize game state variables
        current_state = GameStateEnum::GAME_STATE_AUTH;  // Start with authentication
        max_health = 100;
        player_health = max_health;
        show_controls = false;
        show_debug_info = false;
        title_animation = 0.0f;
        score_animation = 0.0f;
        health_animation = (float)max_health;
        damage_flash = 0.0f;

        isConnected = false;
        userId = 0;
        errorMessage[0] = '\0';
        isAuthenticating = false;
        showLoginForm = true;
        showRegisterForm = false;
        loginUsername[0] = '\0';
        loginPassword[0] = '\0';
        registerUsername[0] = '\0';
        registerPassword[0] = '\0';
        loggedInUsername[0] = '\0';
        isGuest = false;
        lastHeartbeatTime = 0;
    }
} ;
