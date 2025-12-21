#define SDL_MAIN_USE_CALLBACKS 1

#include "ATMCommon.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include "SDL3_image/SDL_image.h"
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
#define WORLD_WIDTH 50000
#define WORLD_HEIGHT 10000
#define GRID_CELL_SIZE 512  // Spatial grid cell size

#define PLAYER_WIDTH 32
#define PLAYER_HEIGHT 32
#define OBSTACLE_WIDTH 32
#define OBSTACLE_HEIGHT 32
#define NUM_OBSTACLES 100000

// Entity type IDs
#define ENTITY_TYPE_DEFAULT -1
#define ENTITY_TYPE_PLAYER 1
#define ENTITY_TYPE_OBSTACLE 2

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

// Player entity data structure
typedef struct {
    float speed;
    int current_frame;
    int texture_ids[3];
    Uint64 animation_timer;
    SDL_Scancode keys_pressed[4];
} PlayerData;

// Obstacle entity data structure
typedef struct {
    float speed;
    float direction[2]; // x,y normalized vector
} ObstacleData;

typedef struct {
    Engine* engine;
    int player_entity;
    int score;
    bool game_over;
    Uint64 last_spawn_time;
    int obstacle_texture_ids[4];
} GameState;

// Function declarations
SDL_Surface* load_texture(const char* filename);
SDL_Surface* create_colored_surface(int width, int height, Uint8 r, Uint8 g, Uint8 b);
void init_obstacles(GameState* state);
bool check_collision(Engine* engine, int entity1, int entity2);
bool is_entity_in_view(Engine* engine, int entity_id, SDL_FRect* view_rect);

// Entity type update functions
void player_entity_update(EntityChunk* chunk, int count, float delta_time);
void obstacle_entity_update(EntityChunk* chunk, int count, float delta_time);

// Function to load a texture from file
SDL_Surface* load_texture(const char* filename) {
    SDL_Surface* surface = IMG_Load(filename);
    if (!surface) {
        SDL_Log("Failed to load texture %s: %s", filename, SDL_GetError());
        // Create a colored surface as fallback
        surface = SDL_CreateSurface(64, 64, SDL_PIXELFORMAT_RGBA32);
        SDL_FillSurfaceRect(surface, NULL, SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format),0, 0, 255, 0, 255));
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

// Initialize obstacles with random positions, speeds, and directions
void init_obstacles(GameState* state) {
    Engine* engine = state->engine;

    // Pre-allocate obstacles in chunks with their specific type
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

        // Create obstacle entity with specific entity type
        int entity_id = engine_add_entity_with_type(engine, ENTITY_TYPE_OBSTACLE,
            x, y,
            OBSTACLE_WIDTH, OBSTACLE_HEIGHT,
            texture_id, 1);

        // Get the type-specific data and initialize it
        ObstacleData* data = (ObstacleData*)engine_get_entity_type_data(engine, entity_id);
        if (data) {
            // Random movement speed
            data->speed = 50.0f + (rand() % 100);

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

            data->direction[0] = dir_x;
            data->direction[1] = dir_y;
        }
    }
}

// Check if two entities are colliding using spatial grid optimization
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

    // AABB collision check
    return !(r1 <= x2 || r2 <= x1 || b1 <= y2 || b2 <= y1);
}

// Check if entity is in view rect - used for culling
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

// Player entity update function - processes an entire chunk of player entities
void player_entity_update(EntityChunk* chunk, int count, float delta_time) {
    // Skip empty chunks
    if (count == 0) return;

    // Get current time once for all entities in the chunk
    Uint64 current_time = SDL_GetTicks();

    // Process all player entities in the chunk
    for (int i = 0; i < count; i++) {
        // Skip inactive entities
        if (!chunk->active[i]) continue;

        // Get type-specific data for this entity
        PlayerData* data = (PlayerData*)((uint8_t*)chunk->type_data + i * sizeof(PlayerData));

        // Update animation
        if (current_time - data->animation_timer > 200) {
            data->current_frame = (data->current_frame + 1) % 3;
            chunk->texture_id[i] = data->texture_ids[data->current_frame];
            data->animation_timer = current_time;
        }

        // Process movement based on key presses
        float move_x = 0.0f;
        float move_y = 0.0f;

        if (data->keys_pressed[0]) move_y -= 1.0f; // W - up
        if (data->keys_pressed[1]) move_x -= 1.0f; // A - left
        if (data->keys_pressed[2]) move_y += 1.0f; // S - down
        if (data->keys_pressed[3]) move_x += 1.0f; // D - right

        // Normalize diagonal movement
        if (move_x != 0.0f && move_y != 0.0f) {
            float length = SDL_sqrtf(move_x * move_x + move_y * move_y);
            move_x /= length;
            move_y /= length;
        }

        // Apply movement
        float new_x = chunk->x[i] + move_x * data->speed * delta_time;
        float new_y = chunk->y[i] + move_y * data->speed * delta_time;

        // Clamp to world bounds
        new_x = SDL_clamp(new_x, 0, WORLD_WIDTH - chunk->width[i]);
        new_y = SDL_clamp(new_y, 0, WORLD_HEIGHT - chunk->height[i]);

        // Update position
        chunk->x[i] = new_x;
        chunk->y[i] = new_y;

        // Update precomputed right/bottom values
        chunk->right[i] = new_x + chunk->width[i];
        chunk->bottom[i] = new_y + chunk->height[i];
    }
}

// Obstacle entity update function - processes an entire chunk of obstacle entities
void obstacle_entity_update(EntityChunk* chunk, int count, float delta_time) {
    // Skip empty chunks
    if (count == 0) return;

    // Process all obstacle entities in the chunk
    for (int i = 0; i < count; i++) {
        // Skip inactive entities
        if (!chunk->active[i]) continue;

        // Get type-specific data for this entity
        ObstacleData* data = (ObstacleData*)((uint8_t*)chunk->type_data + i * sizeof(ObstacleData));

        // Calculate movement
        float new_x = chunk->x[i] + data->direction[0] * data->speed * delta_time;
        float new_y = chunk->y[i] + data->direction[1] * data->speed * delta_time;

        // Bounce off world boundaries
        bool bounce_x = false;
        bool bounce_y = false;

        if (new_x < 0 || new_x > WORLD_WIDTH - chunk->width[i]) {
            data->direction[0] *= -1;
            bounce_x = true;
        }

        if (new_y < 0 || new_y > WORLD_HEIGHT - chunk->height[i]) {
            data->direction[1] *= -1;
            bounce_y = true;
        }

        // Apply corrected position after bounce
        if (bounce_x) {
            new_x = SDL_clamp(new_x, 0, WORLD_WIDTH - chunk->width[i]);
        }

        if (bounce_y) {
            new_y = SDL_clamp(new_y, 0, WORLD_HEIGHT - chunk->height[i]);
        }

        // Update position
        chunk->x[i] = new_x;
        chunk->y[i] = new_y;

        // Update precomputed right/bottom values
        chunk->right[i] = new_x + chunk->width[i];
        chunk->bottom[i] = new_y + chunk->height[i];
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

    // Initialize game engine with spatial grid cell size parameter
    state->engine = engine_create(WINDOW_WIDTH, WINDOW_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT, GRID_CELL_SIZE);
    if (!state->engine) {
        SDL_Log("Failed to create engine");
        SDL_free(state);
        SDL_Quit();
        return SDL_APP_FAILURE;
    }

    // Register entity types with the engine
    engine_register_entity_type(state->engine, ENTITY_TYPE_PLAYER, player_entity_update, sizeof(PlayerData));
    engine_register_entity_type(state->engine, ENTITY_TYPE_OBSTACLE, obstacle_entity_update, sizeof(ObstacleData));

    // Load textures
    SDL_Surface* map_surface = create_colored_surface(100, 100, 200, 200, 200);
    SDL_Surface* player_surface1 = create_colored_surface(PLAYER_WIDTH, PLAYER_HEIGHT, 0, 0, 255);
    SDL_Surface* player_surface2 = create_colored_surface(PLAYER_WIDTH, PLAYER_HEIGHT, 0, 100, 255);
    SDL_Surface* player_surface3 = create_colored_surface(PLAYER_WIDTH, PLAYER_HEIGHT, 100, 100, 255);
    SDL_Surface* obstacle_surface1 = create_colored_surface(OBSTACLE_WIDTH, OBSTACLE_HEIGHT, 255, 0, 0);
    SDL_Surface* obstacle_surface2 = create_colored_surface(OBSTACLE_WIDTH, OBSTACLE_HEIGHT, 255, 50, 0);
    SDL_Surface* obstacle_surface3 = create_colored_surface(OBSTACLE_WIDTH, OBSTACLE_HEIGHT, 200, 0, 0);
    SDL_Surface* obstacle_surface4 = create_colored_surface(OBSTACLE_WIDTH, OBSTACLE_HEIGHT, 255, 100, 100);

    // Add textures to engine
    int map_texture = engine_add_texture(state->engine, map_surface, 0, 0);
    int player_texture1 = engine_add_texture(state->engine, player_surface1, 0, 0);
    int player_texture2 = engine_add_texture(state->engine, player_surface2, 0, 0);
    int player_texture3 = engine_add_texture(state->engine, player_surface3, 0, 0);
    int obstacle_texture1 = engine_add_texture(state->engine, obstacle_surface1, 0, 0);
    int obstacle_texture2 = engine_add_texture(state->engine, obstacle_surface2, 0, 0);
    int obstacle_texture3 = engine_add_texture(state->engine, obstacle_surface3, 0, 0);
    int obstacle_texture4 = engine_add_texture(state->engine, obstacle_surface4, 0, 0);

    // Clean up surfaces
    SDL_DestroySurface(map_surface);
    SDL_DestroySurface(player_surface1);
    SDL_DestroySurface(player_surface2);
    SDL_DestroySurface(player_surface3);
    SDL_DestroySurface(obstacle_surface1);
    SDL_DestroySurface(obstacle_surface2);
    SDL_DestroySurface(obstacle_surface3);
    SDL_DestroySurface(obstacle_surface4);

    // Store texture IDs for obstacles
    state->obstacle_texture_ids[0] = obstacle_texture1;
    state->obstacle_texture_ids[1] = obstacle_texture2;
    state->obstacle_texture_ids[2] = obstacle_texture3;
    state->obstacle_texture_ids[3] = obstacle_texture4;

    // Create player entity at center of screen with specific entity type
    state->player_entity = engine_add_entity_with_type(state->engine,
        ENTITY_TYPE_PLAYER,
        WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2,
        PLAYER_WIDTH, PLAYER_HEIGHT,
        player_texture1, 2); // Layer 2 (above obstacles)

    // Initialize player data
    PlayerData* player_data = (PlayerData*)engine_get_entity_type_data(state->engine, state->player_entity);
    if (player_data) {
        player_data->speed = 200.0f;
        player_data->current_frame = 0;
        player_data->animation_timer = SDL_GetTicks();
        player_data->texture_ids[0] = player_texture1;
        player_data->texture_ids[1] = player_texture2;
        player_data->texture_ids[2] = player_texture3;

        // Clear key tracking
        for (int i = 0; i < 4; i++) {
            player_data->keys_pressed[i] = SDL_SCANCODE_UNKNOWN;
        }
    }

    // Initialize obstacles
    SDL_Log("Creating %d obstacles...", NUM_OBSTACLES);
    init_obstacles(state);
    SDL_Log("Obstacles created successfully");

    // Init game state
    state->score = 0;
    state->game_over = false;
    state->last_spawn_time = SDL_GetTicks();

    // Set camera to follow player
    engine_set_camera_position(state->engine, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);

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
        float fps = fps_frames * 1000.0f / (current_time - fps_last_time);
        SDL_Log("FPS: %.2f, Active entities: %d", fps, engine->entities.total_count);
        fps_last_time = current_time;
        fps_frames = 0;
    }

    if (!state->game_over) {
        // Get player position for camera update
        int chunk_idx, local_idx;
        engine->entities.getChunkIndices(state->player_entity, &chunk_idx, &local_idx);

        if (chunk_idx < engine->entities.chunk_count) {
            EntityChunk* chunk = engine->entities.chunks[chunk_idx];

            if (local_idx < chunk->count) {
                float player_x = chunk->x[local_idx];
                float player_y = chunk->y[local_idx];

                // Update camera to follow player
                engine_set_camera_position(engine, player_x + PLAYER_WIDTH / 2, player_y + PLAYER_HEIGHT / 2);
            }
        }

        // Let the engine handle all entity updates through type-specific update functions
        engine_update(engine);

        // Check for collision between player and obstacles
        // More efficient, only checking visible obstacles
        SDL_FRect visible_rect = engine_get_visible_rect(engine);

        // Add padding to include entities just outside view
        visible_rect.x -= 200;
        visible_rect.y -= 200;
        visible_rect.w += 400;
        visible_rect.h += 400;

        // Use the spatial grid for efficient querying
        const int MAX_VISIBLE = 10000;
        int* visible_entities = (int*)SDL_aligned_alloc(MAX_VISIBLE * sizeof(int), CACHE_LINE_SIZE);
        int visible_count = 0;

        spatial_grid_query(&engine->grid, visible_rect, visible_entities, &visible_count, MAX_VISIBLE, &engine->entities);

        // Check collision only with visible obstacles
        for (int i = 0; i < visible_count; i++) {
            int entity_id = visible_entities[i];

            // Skip player entity and non-obstacles
            if (entity_id == state->player_entity) continue;

            // Get entity's chunk and check if it's an obstacle
            int chunk_idx, local_idx;
            engine->entities.getChunkIndices(entity_id, &chunk_idx, &local_idx);

            if (chunk_idx < engine->entities.chunk_count) {
                EntityChunk* chunk = engine->entities.chunks[chunk_idx];

                // We only care about active obstacles
                if (chunk->type_id == ENTITY_TYPE_OBSTACLE && chunk->active[local_idx]) {
                    if (check_collision(engine, state->player_entity, entity_id)) {
                        state->game_over = true;
                        SDL_Log("Game Over! Score: %d", state->score);
                        break;
                    }
                }
            }
        }

        SDL_aligned_free(visible_entities);

        // Increment score
        state->score += 1;
    }

    // Render updated scene
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

        // Update key state in player data
        if (!state->game_over) {
            PlayerData* player_data = (PlayerData*)engine_get_entity_type_data(state->engine, state->player_entity);
            if (player_data) {
                if (event->key.scancode == SDL_SCANCODE_W) player_data->keys_pressed[0] = SDL_SCANCODE_W;
                if (event->key.scancode == SDL_SCANCODE_A) player_data->keys_pressed[1] = SDL_SCANCODE_A;
                if (event->key.scancode == SDL_SCANCODE_S) player_data->keys_pressed[2] = SDL_SCANCODE_S;
                if (event->key.scancode == SDL_SCANCODE_D) player_data->keys_pressed[3] = SDL_SCANCODE_D;
            }
        }
        break;

    case SDL_EVENT_KEY_UP:
        // Update key state in player data
        if (!state->game_over) {
            PlayerData* player_data = (PlayerData*)engine_get_entity_type_data(state->engine, state->player_entity);
            if (player_data) {
                if (event->key.scancode == SDL_SCANCODE_W) player_data->keys_pressed[0] = SDL_SCANCODE_UNKNOWN;
                if (event->key.scancode == SDL_SCANCODE_A) player_data->keys_pressed[1] = SDL_SCANCODE_UNKNOWN;
                if (event->key.scancode == SDL_SCANCODE_S) player_data->keys_pressed[2] = SDL_SCANCODE_UNKNOWN;
                if (event->key.scancode == SDL_SCANCODE_D) player_data->keys_pressed[3] = SDL_SCANCODE_UNKNOWN;
            }
        }
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