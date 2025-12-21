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

// Configuration for window and world
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1020
#define WORLD_WIDTH 50000
#define WORLD_HEIGHT 10000
#define GRID_CELL_SIZE 1024  // Spatial grid cell size

// Entity performance test configuration
#define NUM_ENTITY_TYPES 1500                // Total different entity types to create
#define MIN_ENTITIES_PER_TYPE 1        // Minimum entities per type
#define MAX_ENTITIES_PER_TYPE 100       // Maximum entities per type
#define ENTITY_WIDTH 32                    // Width of entities
#define ENTITY_HEIGHT 32                   // Height of entities

// Player configuration
#define PLAYER_WIDTH 32
#define PLAYER_HEIGHT 32
#define ENTITY_TYPE_PLAYER 1               // Keep player as special entity type
#define ENTITY_TYPE_FIRST_TEST 1         // First ID for test entity types

// Generic entity data structure (used by all test entity types)
typedef struct {
    float speed;
    float direction[2];   // Normalized direction vector
    Uint8 color[3];       // RGB color values
    int behavior_flags;   // Flags to control entity behavior
} GenericEntityData;

// Player entity data structure (used only for player)
typedef struct {
    float speed;
    int current_frame;
    int texture_ids[3];
    Uint64 animation_timer;
    SDL_Scancode keys_pressed[4];
} PlayerData;

typedef struct {
    Engine* engine;
    int player_entity;
    int score;
    bool game_over;
    Uint64 last_spawn_time;

    // Tracking for entity types and textures
    int num_entity_types;               // Actual number of entity types created
    int* entity_type_ids;               // Array of entity type IDs
    int* entity_counts;                 // How many entities of each type
    int* entity_texture_ids;            // Texture ID for each entity type
    Uint64* entity_update_times;        // Performance tracking
} GameState;

// Function declarations
SDL_Surface* load_texture(const char* filename);
SDL_Surface* create_colored_surface(int width, int height, Uint8 r, Uint8 g, Uint8 b);
void init_entities(GameState* state);
bool check_collision(Engine* engine, int entity1, int entity2);
bool is_entity_in_view(Engine* engine, int entity_id, SDL_FRect* view_rect);

// Entity update functions
void player_entity_update(EntityChunk* chunk, int count, float delta_time);
void generic_entity_update(EntityChunk* chunk, int count, float delta_time);

// Function to load a texture from file
SDL_Surface* load_texture(const char* filename) {
    SDL_Surface* surface = IMG_Load(filename);
    if (!surface) {
        SDL_Log("Failed to load texture %s: %s", filename, SDL_GetError());
        // Create a colored surface as fallback
        surface = SDL_CreateSurface(64, 64, SDL_PIXELFORMAT_RGBA32);
        SDL_FillSurfaceRect(surface, NULL, SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format), 0, 0, 255, 0, 255));
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
    SDL_FillSurfaceRect(surface, NULL, SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format), NULL, r, g, b, 255));
    return surface;
}
// Generic entity update function - processes an entire chunk of entities
void generic_entity_update(EntityChunk* chunk, int count, float delta_time) {
    // Skip empty chunks
    if (count == 0) return;

    // Process all entities in the chunk
    for (int i = 0; i < count; i++) {
        // Skip inactive entities
        if (!chunk->active[i]) continue;

        // Get type-specific data for this entity
        GenericEntityData* data = (GenericEntityData*)((uint8_t*)chunk->type_data + i * sizeof(GenericEntityData));

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

        // Apply any behavior based on behavior flags (extensible for future)
        if (data->behavior_flags & 0x1) {
            // Example: Orbit behavior
            float cx = WORLD_WIDTH / 2;
            float cy = WORLD_HEIGHT / 2;
            float dx = new_x - cx;
            float dy = new_y - cy;
            float dist = SDL_sqrtf(dx * dx + dy * dy);
            if (dist > 0) {
                float nx = -dy / dist;
                float ny = dx / dist;
                data->direction[0] = 0.9f * data->direction[0] + 0.1f * nx;
                data->direction[1] = 0.9f * data->direction[1] + 0.1f * ny;
                // Normalize the direction
                float len = SDL_sqrtf(data->direction[0] * data->direction[0] +
                    data->direction[1] * data->direction[1]);
                if (len > 0) {
                    data->direction[0] /= len;
                    data->direction[1] /= len;
                }
            }
        }
    }
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

// Initialize entities with random positions, speeds, and directions
void init_entities(GameState* state) {
    Engine* engine = state->engine;

    SDL_Log("Creating entities for %d types...", state->num_entity_types);

    // Loop through each entity type
    for (int type_idx = 0; type_idx < state->num_entity_types; type_idx++) {
        int entity_type_id = state->entity_type_ids[type_idx];
        int entity_count = state->entity_counts[type_idx];
        int texture_id = state->entity_texture_ids[type_idx];

        SDL_Log("Creating %d entities of type %d with texture %d",
            entity_count, entity_type_id, texture_id);

        // Create entities of this type
        for (int i = 0; i < entity_count; i++) {
            // Random position within world bounds
            float x = (float)(rand() % (WORLD_WIDTH - ENTITY_WIDTH));
            float y = (float)(rand() % (WORLD_HEIGHT - ENTITY_HEIGHT));

            // Place entities away from player's starting position
            if (x > WINDOW_WIDTH / 2 - 200 && x < WINDOW_WIDTH / 2 + 200) {
                x += 250;
            }
            if (y > WINDOW_HEIGHT / 2 - 200 && y < WINDOW_HEIGHT / 2 + 200) {
                y += 250;
            }

            // Create entity with specific entity type
            int entity_id = engine_add_entity_with_type(engine, entity_type_id,
                x, y, ENTITY_WIDTH, ENTITY_HEIGHT, texture_id, 1);

            // Get the type-specific data and initialize it
            GenericEntityData* data = (GenericEntityData*)engine_get_entity_type_data(engine, entity_id);
            if (data) {
                // Random movement speed
                data->speed = 30.0f + (rand() % 150) + ((float)type_idx * 5.0f);

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

                // Store the color (matches texture color)
                data->color[0] = (type_idx * 25) % 255;
                data->color[1] = (type_idx * 40) % 255;
                data->color[2] = (type_idx * 60) % 255;

                // Set behavior flags (every third type gets orbit behavior)
                data->behavior_flags = (type_idx % 3 == 0) ? 0x1 : 0x0;
            }
        }
    }

    SDL_Log("All entities created successfully");
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

    // Zero-initialize
    memset(state, 0, sizeof(GameState));

    // Initialize game engine with spatial grid cell size parameter
    state->engine = engine_create(WINDOW_WIDTH, WINDOW_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT, GRID_CELL_SIZE);
    if (!state->engine) {
        SDL_Log("Failed to create engine");
        SDL_free(state);
        SDL_Quit();
        return SDL_APP_FAILURE;
    }

    // Register player entity type with the engine
    engine_register_entity_type(state->engine, ENTITY_TYPE_PLAYER, player_entity_update, sizeof(PlayerData));

    // Determine actual number of entity types to create based on configuration 
    // (limited by NUM_ENTITY_TYPES)
    state->num_entity_types = NUM_ENTITY_TYPES;

    // Allocate arrays for entity type tracking
    state->entity_type_ids = (int*)SDL_malloc(state->num_entity_types * sizeof(int));
    state->entity_counts = (int*)SDL_malloc(state->num_entity_types * sizeof(int));
    state->entity_texture_ids = (int*)SDL_malloc(state->num_entity_types * sizeof(int));
    state->entity_update_times = (Uint64*)SDL_malloc(state->num_entity_types * sizeof(Uint64));

    if (!state->entity_type_ids || !state->entity_counts ||
        !state->entity_texture_ids || !state->entity_update_times) {
        SDL_Log("Failed to allocate memory for entity tracking");
        // Free any successfully allocated memory
        if (state->entity_type_ids) SDL_free(state->entity_type_ids);
        if (state->entity_counts) SDL_free(state->entity_counts);
        if (state->entity_texture_ids) SDL_free(state->entity_texture_ids);
        if (state->entity_update_times) SDL_free(state->entity_update_times);
        engine_destroy(state->engine);
        SDL_free(state);
        SDL_Quit();
        return SDL_APP_FAILURE;
    }

    // Create player textures
    SDL_Surface* player_surface1 = create_colored_surface(PLAYER_WIDTH, PLAYER_HEIGHT, 0, 0, 255);
    SDL_Surface* player_surface2 = create_colored_surface(PLAYER_WIDTH, PLAYER_HEIGHT, 0, 100, 255);
    SDL_Surface* player_surface3 = create_colored_surface(PLAYER_WIDTH, PLAYER_HEIGHT, 100, 100, 255);

    // Add player textures to engine
    int player_texture1 = engine_add_texture(state->engine, player_surface1, 0, 0);
    int player_texture2 = engine_add_texture(state->engine, player_surface2, 0, 0);
    int player_texture3 = engine_add_texture(state->engine, player_surface3, 0, 0);

    // Clean up player surfaces
    SDL_DestroySurface(player_surface1);
    SDL_DestroySurface(player_surface2);
    SDL_DestroySurface(player_surface3);

    // Create player entity
    state->player_entity = engine_add_entity_with_type(state->engine,
        ENTITY_TYPE_PLAYER,
        WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2,
        PLAYER_WIDTH, PLAYER_HEIGHT,
        player_texture1, 2); // Layer 2 (above other entities)

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

    // Create textures and register entity types for each test entity type
    for (int i = 0; i < state->num_entity_types; i++) {
        // Create a unique entity type ID
        int type_id =  i;
        state->entity_type_ids[i] = type_id;

        // Register this entity type with the engine
        engine_register_entity_type(state->engine, type_id, generic_entity_update, sizeof(GenericEntityData));

        // Generate unique color for this entity type
        Uint8 r = (i * 25) % 255;
        Uint8 g = (i * 40) % 255;
        Uint8 b = (i * 60) % 255;

        // Create surface with this color
        SDL_Surface* entity_surface = create_colored_surface(ENTITY_WIDTH, ENTITY_HEIGHT, r, g, b);

        int atlas_columns = 16;  // Adjust based on your atlas size
        int x_pos = (i % atlas_columns) * ENTITY_WIDTH;
        int y_pos = (i / atlas_columns) * ENTITY_HEIGHT;

        // Add the texture to the engine with unique position
        int texture_id = engine_add_texture(state->engine, entity_surface, x_pos, y_pos);
        state->entity_texture_ids[i] = texture_id;

        // Clean up surface
        SDL_DestroySurface(entity_surface);

        int count = MIN_ENTITIES_PER_TYPE;
        if (MAX_ENTITIES_PER_TYPE > MIN_ENTITIES_PER_TYPE) {
            count += rand() % (MAX_ENTITIES_PER_TYPE - MIN_ENTITIES_PER_TYPE);
        }
        state->entity_counts[i] = count;

        // Initialize update time tracking
        state->entity_update_times[i] = 0;
    }

    // Initialize test entities
    init_entities(state);

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
    static Uint64 performance_report_timer = 0;

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
        // Measure the start time
        Uint64 update_start_time = SDL_GetPerformanceCounter();

        // Update all entities
        engine_update(engine);

        // Measure the end time
        Uint64 update_end_time = SDL_GetPerformanceCounter();

        // Calculate the elapsed time in milliseconds
        float update_time_ms = (update_end_time - update_start_time) * 1000.0f /
            SDL_GetPerformanceFrequency();

        // Update performance stats (entity update times are tracked in the engine)
        for (int i = 0; i < state->num_entity_types; i++) {
            int type_id = state->entity_type_ids[i];
            int type_idx = engine->type_id_to_index[type_id];
            if (type_idx >= 0 && type_idx < engine->entity_type_count) {
                state->entity_update_times[i] =
                    engine->entity_types[type_idx].last_update_time * 1000000; // Store in microseconds
            }
        }

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

            // Reinitialize entities
            init_entities(state);
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

        // Free memory for entity tracking arrays
        if (state->entity_type_ids) SDL_free(state->entity_type_ids);
        if (state->entity_counts) SDL_free(state->entity_counts);
        if (state->entity_texture_ids) SDL_free(state->entity_texture_ids);
        if (state->entity_update_times) SDL_free(state->entity_update_times);

        SDL_free(state);
    }

    SDL_Quit();
}