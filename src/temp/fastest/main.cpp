#define SDL_MAIN_USE_CALLBACKS 1

#include "ATMCommon.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
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
#define WORLD_WIDTH 240000
#define WORLD_HEIGHT 180000

#define PLAYER_WIDTH 32
#define PLAYER_HEIGHT 32
#define OBSTACLE_WIDTH 24
#define OBSTACLE_HEIGHT 24
#define NUM_OBSTACLES 1000000

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
} GameState;

// Function to create a colored surface
SDL_Surface* create_colored_surface(int width, int height, Uint8 r, Uint8 g, Uint8 b) {
    SDL_Surface* surface = SDL_CreateSurface(width, height, SDL_PIXELFORMAT_RGBA32);
    if (!surface) {
        SDL_Log("Failed to create surface: %s", SDL_GetError());
        return NULL;
    }

    SDL_FillSurfaceRect(surface, NULL, SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format), 0, r, g, b, 255));
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

        // Create obstacle entity
        state->obstacle_entities[i] = engine_add_entity(state->engine,
            x, y,
            OBSTACLE_WIDTH, OBSTACLE_HEIGHT,
            1, 1); // texture_id 1, layer 1

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

// Check if two entities are colliding
bool check_collision(Engine* engine, int entity1, int entity2) {
    if (entity1 < 0 || entity2 < 0 ||
        entity1 >= engine->entities.count ||
        entity2 >= engine->entities.count) {
        return false;
    }

    float x1 = engine->entities.x[entity1];
    float y1 = engine->entities.y[entity1];
    float r1 = engine->entities.right[entity1];
    float b1 = engine->entities.bottom[entity1];

    float x2 = engine->entities.x[entity2];
    float y2 = engine->entities.y[entity2];
    float r2 = engine->entities.right[entity2];
    float b2 = engine->entities.bottom[entity2];

    return !(r1 < x2 || r2 < x1 || b1 < y2 || b2 < y1);
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

    // Initialize game state
    state->engine = engine_create(WINDOW_WIDTH, WINDOW_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT);
    if (!state->engine) {
        SDL_Log("Failed to create engine");
        SDL_free(state);
        SDL_Quit();
        return SDL_APP_FAILURE;
    }

    // Create textures
    SDL_Surface* player_surface = create_colored_surface(PLAYER_WIDTH, PLAYER_HEIGHT, 0, 0, 255); // Blue
    SDL_Surface* obstacle_surface = create_colored_surface(OBSTACLE_WIDTH, OBSTACLE_HEIGHT, 255, 0, 0); // Red

    // Add textures to engine
    int player_texture = engine_add_texture(state->engine, player_surface, 0, 0);
    int obstacle_texture = engine_add_texture(state->engine, obstacle_surface, 0, 0);

    // Clean up surfaces
    SDL_DestroySurface(player_surface);
    SDL_DestroySurface(obstacle_surface);

    // Create player entity at center of screen
    state->player_entity = engine_add_entity(state->engine,
        WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2,
        PLAYER_WIDTH, PLAYER_HEIGHT,
        0, 2); // texture_id 0, layer 2 (above obstacles)

    // Initialize obstacles
    init_obstacles(state);

    // Init game state
    state->score = 0;
    state->game_over = false;
    state->last_spawn_time = SDL_GetTicks();

    // Set camera to follow player
    engine_set_camera_position(state->engine, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);

    // Clear key tracking
    for (int i = 0; i < 4; i++) {
        state->keys_pressed[i] = SDL_SCANCODE_UNKNOWN;
    }

    *appstate = state;
    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appstate) {

    GameState* state = (GameState*)appstate;
    Engine* engine = state->engine;

    // Get time delta
    Uint64 current_time = SDL_GetTicks();
    Uint64 delta_time_ms = current_time - engine->last_frame_time;
    double delta_time = delta_time_ms / 1000.0f;


    // Update last frame time for next iteration
    engine->last_frame_time = current_time;
    // Calculate and print FPS (using a rolling average for stability)
    static Uint64 fps_last_time = 0;
    static int fps_frames = 0;

    fps_frames++;

    if (current_time - fps_last_time >= 1000) {
        // We've reached a second, calculate and print FPS
        float fps = fps_frames * 1000.0f / (current_time - fps_last_time);
        SDL_Log("FPS: %.2f", fps);

        // Reset counters
        fps_last_time = current_time;
        fps_frames = 0;
    }

    if (state->game_over) {
        // Game over logic - could display message, wait for restart, etc.
        return SDL_APP_CONTINUE;
    }

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

    // Update player position
    float player_x = engine->entities.x[state->player_entity];
    float player_y = engine->entities.y[state->player_entity];

    player_x += move_x * player_speed * delta_time;
    player_y += move_y * player_speed * delta_time;

    // Keep player within world bounds
    if (player_x < 0) player_x = 0;
    if (player_y < 0) player_y = 0;
    if (player_x > WORLD_WIDTH - PLAYER_WIDTH) player_x = WORLD_WIDTH - PLAYER_WIDTH;
    if (player_y > WORLD_HEIGHT - PLAYER_HEIGHT) player_y = WORLD_HEIGHT - PLAYER_HEIGHT;

    engine_set_entity_position(engine, state->player_entity, player_x, player_y);

    // Update obstacles
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        double obstacle_x = engine->entities.x[state->obstacle_entities[i]];
        double obstacle_y = engine->entities.y[state->obstacle_entities[i]];

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

        engine_set_entity_position(engine, state->obstacle_entities[i], obstacle_x, obstacle_y);

        // Check collision with player
        if (check_collision(engine, state->player_entity, state->obstacle_entities[i])) {
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