// EntitySystem.h
#pragma once

#include "ATMEngine.h"
#include "ATMEngine/GameState.h"

// Constants for entity system
#define ENTITY_WIDTH 32
#define ENTITY_HEIGHT 32
#define PLAYER_WIDTH 32
#define PLAYER_HEIGHT 32

// Entity data structures
typedef struct {
    float speed;
    float direction[2];   // Normalized direction vector
    Uint8 color[3];       // RGB color values
    int behavior_flags;   // Flags to control entity behavior
} GenericEntityData;

typedef struct {
    float speed;
    int current_frame;
    int texture_ids[3];
    Uint64 animation_timer;
    SDL_Scancode keys_pressed[4];
} PlayerData;

// Entity update functions
void player_entity_update(EntityChunk* chunk, int count, float delta_time);
void generic_entity_update(EntityChunk* chunk, int count, float delta_time);

// Entity initialization
void init_entities(GameState* state);

