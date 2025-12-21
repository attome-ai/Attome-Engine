#define SDL_MAIN_USE_CALLBACKS 1

#include "ATMCommon.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <SDL3_image/SDL_image.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "ATMEngine.h"

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1020
#define WORLD_WIDTH 500000
#define WORLD_HEIGHT 100000
#define GRID_CELL_SIZE 512  // Added grid cell size parameter

#define PLAYER_WIDTH 64
#define PLAYER_HEIGHT 64
#define OBSTACLE_WIDTH 48
#define OBSTACLE_HEIGHT 48
#define NUM_OBSTACLES 100000

// Define texture IDs
#define TEXTURE_MAP 0
#define TEXTURE_PLAYER1 1
#define TEXTURE_PLAYER2 2
#define TEXTURE_PLAYER3 3
#define TEXTURE_OBSTACLE1 4
#define TEXTURE_OBSTACLE2 5
#define TEXTURE_OBSTACLE3 6
#define TEXTURE_OBSTACLE4 7
#define TEXTURE_COUNT 8

typedef struct {
    Engine* engine;
    int player_entity;
    int obstacle_entities[NUM_OBSTACLES];
    float obstacle_speeds[NUM_OBSTACLES];
    float obstacle_directions[NUM_OBSTACLES][2]; // x,y direction vectors
    int score;
    bool game_over;
    Uint64 last_spawn_time;
    SDL_Scancode keys_pressed[4]; // Track WASD keys
    Uint64 animation_timer;       // For player animation
    int current_player_frame;     // Current animation frame
    int player_texture_ids[3];    // Store player texture IDs
    int obstacle_texture_ids[4];  // Store obstacle texture IDs
} GameState;

// Function to load a texture from file
SDL_Surface* load_texture(const char* filename) {
    SDL_Surface* surface = NULL;

    
    // Attempt to load the image
    surface = IMG_Load(filename);
    if (!surface) {
        SDL_Log("Failed to load texture %s: %s", filename, SDL_GetError());
        // Try fallback - create a colored surface
        surface = SDL_CreateSurface(64, 64, SDL_PIXELFORMAT_RGBA32);
        SDL_FillSurfaceRect(surface, NULL, SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format),0, 255, 0, 255, 255));
    }

    return surface;
}

// Create a colored surface (fallback if textures fail to load)
SDL_Surface* create_colored_surface(int width, int height, Uint8 r, Uint8 g, Uint8 b) {
    SDL_Surface* surface = SDL_CreateSurface(width, height, SDL_PIXELFORMAT_RGBA32);
    if (!surface) {
        SDL_Log("Failed to create surface: %s", SDL_GetError());
        return NULL;
    }

    SDL_FillSurfaceRect(surface, NULL, SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format),0, r, g, b, 255));
    return surface;
}

// Initialize obstacles with random positions and directions
void init_obstacles(GameState* state) {
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        // Random position within world bounds
        float x = (float)(rand() % (WORLD_WIDTH - OBSTACLE_WIDTH));
        float y = (float)(rand() % (WORLD_HEIGHT - OBSTACLE_HEIGHT));

        // Place obstacles away from player's starting position
        if (x > WINDOW_WIDTH / 2 - 100 && x < WINDOW_WIDTH / 2 + 100) {
            x += 150;
        }
        if (y > WINDOW_HEIGHT / 2 - 100 && y < WINDOW_HEIGHT / 2 + 100) {
            y += 150;
        }

        // Randomly select an obstacle texture
        int texture_id = state->obstacle_texture_ids[rand() % 4];

        // Create obstacle entity
        state->obstacle_entities[i] = engine_add_entity(state->engine,
            x, y,
            OBSTACLE_WIDTH, OBSTACLE_HEIGHT,
            texture_id, 1); // texture_id, layer 1

        // Random movement speed and direction
        state->obstacle_speeds[i] = 50.0f + (rand() % 100);

        // Normalized direction vector
        float dir_x = (float)((rand() % 200) - 100) / 100.0f;
        float dir_y = (float)((rand() % 200) - 100) / 100.0f;

        // Normalize
        float length = SDL_sqrtf(dir_x * dir_x + dir_y * dir_y);
        if (length > 0) {
            dir_x /= length;
            dir_y /= length;
        }
        else {
            dir_x = 1.0f;
            dir_y = 0.0f;
        }

        state->obstacle_directions[i][0] = dir_x;
        state->obstacle_directions[i][1] = dir_y;
    }
}

// Check if two entities are colliding - Fixed for chunked entity architecture
bool check_collision(Engine* engine, int entity1, int entity2) {
    EntityManager* em = &engine->entities;

    // Check if entity IDs are valid
    if (!em->isValidEntity(entity1) || !em->isValidEntity(entity2)) {
        return false;
    }

    // Get chunk and local indices for both entities
    int chunk_idx1, local_idx1, chunk_idx2, local_idx2;
    em->getChunkIndices(entity1, &chunk_idx1, &local_idx1);
    em->getChunkIndices(entity2, &chunk_idx2, &local_idx2);

    // Get entity coordinates
    float x1 = em->chunks[chunk_idx1]->x[local_idx1];
    float y1 = em->chunks[chunk_idx1]->y[local_idx1];
    float r1 = em->chunks[chunk_idx1]->right[local_idx1];
    float b1 = em->chunks[chunk_idx1]->bottom[local_idx1];

    float x2 = em->chunks[chunk_idx2]->x[local_idx2];
    float y2 = em->chunks[chunk_idx2]->y[local_idx2];
    float r2 = em->chunks[chunk_idx2]->right[local_idx2];
    float b2 = em->chunks[chunk_idx2]->bottom[local_idx2];

    // Check for AABB collision
    return !(r1 <= x2 || r2 <= x1 || b1 <= y2 || b2 <= y1);
}

// Get entity position - utility function to simplify entity access
bool get_entity_position(Engine* engine, int entity_id, float* x, float* y) {
    EntityManager* em = &engine->entities;

    if (!em->isValidEntity(entity_id)) {
        return false;
    }

    int chunk_idx, local_idx;
    em->getChunkIndices(entity_id, &chunk_idx, &local_idx);

    *x = em->chunks[chunk_idx]->x[local_idx];
    *y = em->chunks[chunk_idx]->y[local_idx];

    return true;
}

// Check if entity is in view rect
bool is_entity_in_view(Engine* engine, int entity_id, SDL_FRect* view_rect) {
    EntityManager* em = &engine->entities;

    if (!em->isValidEntity(entity_id)) {
        return false;
    }

    int chunk_idx, local_idx;
    em->getChunkIndices(entity_id, &chunk_idx, &local_idx);

    float x = em->chunks[chunk_idx]->x[local_idx];
    float y = em->chunks[chunk_idx]->y[local_idx];
    float right = em->chunks[chunk_idx]->right[local_idx];
    float bottom = em->chunks[chunk_idx]->bottom[local_idx];

    return !(right < view_rect->x || x > view_rect->x + view_rect->w ||
        bottom < view_rect->y || y > view_rect->y + view_rect->h);
}

// Updates player animation
void update_player_animation(GameState* state) {
    Uint64 current_time = SDL_GetTicks();

    // Update animation frame every 200ms
    if (current_time - state->animation_timer > 200) {
        state->current_player_frame = (state->current_player_frame + 1) % 3; // 3 player textures

        // Update player texture ID
        int entity_id = state->player_entity;
        int texture_id = state->player_texture_ids[state->current_player_frame];

        // Update the entity's texture
        EntityManager* em = &state->engine->entities;
        if (em->isValidEntity(entity_id)) {
            int chunk_idx, local_idx;
            em->getChunkIndices(entity_id, &chunk_idx, &local_idx);
            em->chunks[chunk_idx]->texture_id[local_idx] = texture_id;
        }
        
        // Reset timer
        state->animation_timer = current_time;
    }
}

SDL_AppResult SDL_AppInit(void** appstate, int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        SDL_Log("Could not initialize SDL: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }
    
    


    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Create game state
    GameState* state = (GameState*)SDL_malloc(sizeof(GameState));
    if (!state) {
        SDL_Log("Failed to allocate game state");
        SDL_Quit();
        return SDL_APP_FAILURE;
    }

    // Initialize game state with cell size parameter
    state->engine = engine_create(WINDOW_WIDTH, WINDOW_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT, GRID_CELL_SIZE);
    if (!state->engine) {
        SDL_Log("Failed to create engine");
        SDL_free(state);
        SDL_Quit();
        return SDL_APP_FAILURE;
    }
    IMG_Load;
    // Load textures from files
    SDL_Surface* map_surface = load_texture("textures/np1.png");
    SDL_Surface* player_surface1 = load_texture("textures/np1.png");
    SDL_Surface* player_surface2 = load_texture("textures/np1.png");
    SDL_Surface* player_surface3 = load_texture("textures/np1.png");
    SDL_Surface* obstacle_surface1 = load_texture("textures/np1.png");
    SDL_Surface* obstacle_surface2 = load_texture("textures/np1.png");
    SDL_Surface* obstacle_surface3 = load_texture("textures/np1.png");
    SDL_Surface* obstacle_surface4 = load_texture("textures/np1.png");

    // Fallback if textures failed to load
    if (!player_surface1 || !player_surface2 || !player_surface3) {
        SDL_Log("Failed to load player textures, using fallback");
        if (player_surface1) SDL_DestroySurface(player_surface1);
        if (player_surface2) SDL_DestroySurface(player_surface2);
        if (player_surface3) SDL_DestroySurface(player_surface3);

        player_surface1 = create_colored_surface(PLAYER_WIDTH, PLAYER_HEIGHT, 0, 0, 255); // Blue
        player_surface2 = create_colored_surface(PLAYER_WIDTH, PLAYER_HEIGHT, 0, 100, 255); // Light blue
        player_surface3 = create_colored_surface(PLAYER_WIDTH, PLAYER_HEIGHT, 100, 100, 255); // Different blue
    }

    if (!obstacle_surface1 || !obstacle_surface2 || !obstacle_surface3 || !obstacle_surface4) {
        SDL_Log("Failed to load obstacle textures, using fallback");
        if (obstacle_surface1) SDL_DestroySurface(obstacle_surface1);
        if (obstacle_surface2) SDL_DestroySurface(obstacle_surface2);
        if (obstacle_surface3) SDL_DestroySurface(obstacle_surface3);
        if (obstacle_surface4) SDL_DestroySurface(obstacle_surface4);

        obstacle_surface1 = create_colored_surface(OBSTACLE_WIDTH, OBSTACLE_HEIGHT, 255, 0, 0); // Red
        obstacle_surface2 = create_colored_surface(OBSTACLE_WIDTH, OBSTACLE_HEIGHT, 255, 50, 0); // Orange-red
        obstacle_surface3 = create_colored_surface(OBSTACLE_WIDTH, OBSTACLE_HEIGHT, 200, 0, 0); // Dark red
        obstacle_surface4 = create_colored_surface(OBSTACLE_WIDTH, OBSTACLE_HEIGHT, 255, 100, 100); // Light red
    }

    if (!map_surface) {
        map_surface = create_colored_surface(WORLD_WIDTH, WORLD_HEIGHT, 200, 200, 200); // Gray background
    }

    // Add textures to engine
    int map_texture = engine_add_texture(state->engine, map_surface, 0, 0);
    int player_texture1 = engine_add_texture(state->engine, player_surface1, 0, 0);
    int player_texture2 = engine_add_texture(state->engine, player_surface2, 0, 0);
    int player_texture3 = engine_add_texture(state->engine, player_surface3, 0, 0);
    int obstacle_texture1 = engine_add_texture(state->engine, obstacle_surface1, 0, 0);
    int obstacle_texture2 = engine_add_texture(state->engine, obstacle_surface2, 0, 0);
    int obstacle_texture3 = engine_add_texture(state->engine, obstacle_surface3, 0, 0);
    int obstacle_texture4 = engine_add_texture(state->engine, obstacle_surface4, 0, 0);

    // Store texture IDs in game state
    state->player_texture_ids[0] = player_texture1;
    state->player_texture_ids[1] = player_texture2;
    state->player_texture_ids[2] = player_texture3;
    state->obstacle_texture_ids[0] = obstacle_texture1;
    state->obstacle_texture_ids[1] = obstacle_texture2;
    state->obstacle_texture_ids[2] = obstacle_texture3;
    state->obstacle_texture_ids[3] = obstacle_texture4;

    // Clean up surfaces
    SDL_DestroySurface(map_surface);
    SDL_DestroySurface(player_surface1);
    SDL_DestroySurface(player_surface2);
    SDL_DestroySurface(player_surface3);
    SDL_DestroySurface(obstacle_surface1);
    SDL_DestroySurface(obstacle_surface2);
    SDL_DestroySurface(obstacle_surface3);
    SDL_DestroySurface(obstacle_surface4);

    // Create player entity at center of screen
    state->player_entity = engine_add_entity(state->engine,
        WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2,
        PLAYER_WIDTH, PLAYER_HEIGHT,
        player_texture1, 2); // Start with first player texture, layer 2 (above obstacles)

    if (state->player_entity < 0) {
        SDL_Log("Failed to create player entity");
        engine_destroy(state->engine);
        SDL_free(state);
        SDL_Quit();
        return SDL_APP_FAILURE;
    }

    // Initialize obstacles
    SDL_Log("Creating %d obstacles...", NUM_OBSTACLES);
    init_obstacles(state);
    SDL_Log("Obstacles created successfully");

    // Init game state
    state->score = 0;
    state->game_over = false;
    state->last_spawn_time = SDL_GetTicks();
    state->animation_timer = SDL_GetTicks();
    state->current_player_frame = 0;

    // Set camera to follow player
    engine_set_camera_position(state->engine, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);

    // Clear key tracking
    for (int i = 0; i < 4; i++) {
        state->keys_pressed[i] = SDL_SCANCODE_UNKNOWN;
    }

    // Initialize last_frame_time to current time
    state->engine->last_frame_time = SDL_GetTicks();

    *appstate = state;
    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appstate) {
    GameState* state = (GameState*)appstate;
    Engine* engine = state->engine;

    // Get time delta
    Uint64 current_time = SDL_GetTicks();
    Uint64 delta_time_ms = current_time - engine->last_frame_time;
    float delta_time = delta_time_ms / 1000.0f;

    // Calculate and print FPS (using a rolling average for stability)
    static Uint64 fps_last_time = 0;
    static int fps_frames = 0;

    fps_frames++;

    if (current_time - fps_last_time >= 1000) {
        // We've reached a second, calculate and print FPS
        float fps = fps_frames * 1000.0f / (current_time - fps_last_time);
        SDL_Log("FPS: %.2f, Active entities: %d", fps, engine->entities.total_count);

        // Reset counters
        fps_last_time = current_time;
        fps_frames = 0;
    }

    if (state->game_over) {
        engine->last_frame_time = current_time;
        return SDL_APP_CONTINUE;
    }

    // Update player animation
    update_player_animation(state);

    // Player movement based on key presses
    float player_speed = 200.0f; // Pixels per second
    float move_x = 0.0f;
    float move_y = 0.0f;

    // Process keyboard state
    if (state->keys_pressed[0]) move_y -= 1.0f; // W - up
    if (state->keys_pressed[1]) move_x -= 1.0f; // A - left
    if (state->keys_pressed[2]) move_y += 1.0f; // S - down
    if (state->keys_pressed[3]) move_x += 1.0f; // D - right

    // Normalize diagonal movement
    if (move_x != 0.0f && move_y != 0.0f) {
        float length = SDL_sqrtf(move_x * move_x + move_y * move_y);
        move_x /= length;
        move_y /= length;
    }

    // Get player position
    float player_x, player_y;
    if (!get_entity_position(engine, state->player_entity, &player_x, &player_y)) {
        SDL_Log("Player entity is invalid!");
        engine_update(engine);
        engine_render(engine);
        // Update last frame time for proper delta time calculation
        engine->last_frame_time = current_time;
        return SDL_APP_CONTINUE;
    }

    player_x += move_x * player_speed * delta_time;
    player_y += move_y * player_speed * delta_time;

    // Keep player within world bounds
    player_x = SDL_clamp(player_x, 0, WORLD_WIDTH - PLAYER_WIDTH);
    player_y = SDL_clamp(player_y, 0, WORLD_HEIGHT - PLAYER_HEIGHT);

    engine_set_entity_position(engine, state->player_entity, player_x, player_y);

    // Update obstacles - only process visible obstacles for better performance
    SDL_FRect visible_rect = engine_get_visible_rect(engine);

    // Add some padding to the visible rect to include nearby obstacles
    visible_rect.x -= 200;
    visible_rect.y -= 200;
    visible_rect.w += 400;
    visible_rect.h += 400;

    // Process obstacles with spatial culling
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        int entity_id = state->obstacle_entities[i];

        // Skip if not in view (spatial culling)
        if (!is_entity_in_view(engine, entity_id, &visible_rect)) {
            continue;
        }

        // Get obstacle position
        float obstacle_x, obstacle_y;
        if (!get_entity_position(engine, entity_id, &obstacle_x, &obstacle_y)) {
            continue;
        }

        // Move obstacle
        obstacle_x += state->obstacle_directions[i][0] * state->obstacle_speeds[i] * delta_time;
        obstacle_y += state->obstacle_directions[i][1] * state->obstacle_speeds[i] * delta_time;

        // Bounce off world boundaries
        if (obstacle_x < 0 || obstacle_x > WORLD_WIDTH - OBSTACLE_WIDTH) {
            state->obstacle_directions[i][0] *= -1;
            obstacle_x = SDL_clamp(obstacle_x, 0, WORLD_WIDTH - OBSTACLE_WIDTH);
        }
        if (obstacle_y < 0 || obstacle_y > WORLD_HEIGHT - OBSTACLE_HEIGHT) {
            state->obstacle_directions[i][1] *= -1;
            obstacle_y = SDL_clamp(obstacle_y, 0, WORLD_HEIGHT - OBSTACLE_HEIGHT);
        }

        engine_set_entity_position(engine, entity_id, obstacle_x, obstacle_y);

        // Check collision with player - only for active obstacles
        if (check_collision(engine, state->player_entity, entity_id)) {
            state->game_over = true;
            SDL_Log("Game Over! Score: %d", state->score);
            break;
        }
    }

    // Update camera to follow player
    engine_set_camera_position(engine, player_x + PLAYER_WIDTH / 2, player_y + PLAYER_HEIGHT / 2);

    // Increment score
    if (!state->game_over) {
        state->score += 1;
    }

    // Update and render the engine
    engine_update(engine);
    engine_render(engine);

    // Update last frame time for proper delta time calculation
    engine->last_frame_time = current_time;

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event) {
    GameState* state = (GameState*)appstate;

    switch (event->type) {
    case SDL_EVENT_KEY_DOWN:
        if (state->game_over && event->key.scancode == SDL_SCANCODE_R) {
            // Restart game on 'R' press when game over
            state->game_over = false;
            state->score = 0;

            // Reset player position
            engine_set_entity_position(state->engine, state->player_entity,
                WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);

            // Reinitialize obstacles
            init_obstacles(state);
        }

        // Track WASD keys using scancodes
        if (event->key.scancode == SDL_SCANCODE_W) state->keys_pressed[0] = SDL_SCANCODE_W;
        if (event->key.scancode == SDL_SCANCODE_A) state->keys_pressed[1] = SDL_SCANCODE_A;
        if (event->key.scancode == SDL_SCANCODE_S) state->keys_pressed[2] = SDL_SCANCODE_S;
        if (event->key.scancode == SDL_SCANCODE_D) state->keys_pressed[3] = SDL_SCANCODE_D;
        break;

    case SDL_EVENT_KEY_UP:
        // Untrack WASD keys
        if (event->key.scancode == SDL_SCANCODE_W) state->keys_pressed[0] = SDL_SCANCODE_UNKNOWN;
        if (event->key.scancode == SDL_SCANCODE_A) state->keys_pressed[1] = SDL_SCANCODE_UNKNOWN;
        if (event->key.scancode == SDL_SCANCODE_S) state->keys_pressed[2] = SDL_SCANCODE_UNKNOWN;
        if (event->key.scancode == SDL_SCANCODE_D) state->keys_pressed[3] = SDL_SCANCODE_UNKNOWN;
        break;

    case SDL_EVENT_QUIT:
        return SDL_APP_FAILURE;
    }

    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult result) {
    GameState* state = (GameState*)appstate;

    if (state) {
        if (state->engine) {
            engine_destroy(state->engine);
        }
        SDL_free(state);
    }

    SDL_Quit();
}