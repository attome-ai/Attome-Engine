#define SDL_MAIN_USE_CALLBACKS 1

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "ATMEngine.h"


// Game state type
typedef struct Game {
    Engine* engine;

    // Benchmark system
    bool benchmark_running;
    Uint64 benchmark_start_time;
    Uint64 last_report_time;
    double total_ms;
    int total_frames;
    int frames_since_report;
    int target_entity_count;
    int current_entity_count;
    int benchmark_duration_seconds;
    int entities_per_frame;
} Game;

// Game functions
Game* game_create(int window_width, int window_height);
void game_destroy(Game* game);
void game_update(Game* game);
void game_render(Game* game);
void game_handle_event(Game* game, SDL_Event* event);

// Benchmark functions
void game_start_benchmark(Game* game, int entity_count, int duration_seconds);
void game_update_benchmark(Game* game);


// ====== GAME.C ======
#include <stdio.h>
#include <stdlib.h>

// Generate random entity for benchmark
void generate_random_entity(Game* game) {
    Engine* engine = game->engine;
    float x = (float)(rand() % (int)engine->world_bounds.w);
    float y = (float)(rand() % (int)engine->world_bounds.h);
    int width = 16 + rand() % 48;
    int height = 16 + rand() % 48;
    int texture_id = rand() % 8; // Assuming 8 textures
    int layer = rand() % 5; // 5 layers

    engine_add_entity(engine, x, y, width, height, texture_id, layer);
}

Game* game_create(int window_width, int window_height) {
    Game* game = (Game*)SDL_malloc(sizeof(Game));
    if (!game) return NULL;
    memset(game, 0, sizeof(Game));

    // Create engine with large world
    game->engine = engine_create(window_width, window_height, 8000, 8000);
    if (!game->engine) {
        SDL_free(game);
        return NULL;
    }

    // Create some dummy textures for benchmark
    for (int i = 0; i < 8; i++) {
        // Create a colored surface
        SDL_Surface* surface = SDL_CreateSurface(64, 64, SDL_PIXELFORMAT_RGBA8888);

        // Fill with a color
        Uint32 color = SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format), 0,
            rand() % 255, rand() % 255, rand() % 255, 255);
        SDL_FillSurfaceRect(surface, NULL, color);

        // Add to engine
        engine_add_texture(game->engine, surface, (i % 4) * 64, (i / 4) * 64);

        // Clean up
        SDL_DestroySurface(surface);
    }

    // Initialize benchmark data
    game->benchmark_running = false;
    game->target_entity_count = 5000;
    game->current_entity_count = 0;
    game->benchmark_duration_seconds = 10;
    game->entities_per_frame = 200;

    return game;
}

void game_destroy(Game* game) {
    if (game) {
        if (game->engine) {
            engine_destroy(game->engine);
        }
        SDL_free(game);
    }
}

void game_update(Game* game) {
    // Update benchmark if running
    if (game->benchmark_running) {
        game_update_benchmark(game);
    }

    // Update engine
    engine_update(game->engine);
}

void game_render(Game* game) {
    engine_render(game->engine);
}

void game_handle_event(Game* game, SDL_Event* event) {
    if (!game || !event) return;

    Engine* engine = game->engine;

    switch (event->type) {
    case SDL_EVENT_KEY_DOWN:
        // Camera controls
        if (event->key.scancode == SDL_SCANCODE_W) {
            engine_set_camera_position(engine, engine->camera.x, engine->camera.y - 100.0f);
        }
        else if (event->key.scancode == SDL_SCANCODE_S) {
            engine_set_camera_position(engine, engine->camera.x, engine->camera.y + 100.0f);
        }
        else if (event->key.scancode == SDL_SCANCODE_A) {
            engine_set_camera_position(engine, engine->camera.x - 100.0f, engine->camera.y);
        }
        else if (event->key.scancode == SDL_SCANCODE_D) {
            engine_set_camera_position(engine, engine->camera.x + 100.0f, engine->camera.y);
        }
        else if (event->key.scancode == SDL_SCANCODE_Q) {
            engine_set_camera_zoom(engine, engine->camera.zoom * 0.9f);
        }
        else if (event->key.scancode == SDL_SCANCODE_E) {
            engine_set_camera_zoom(engine, engine->camera.zoom * 1.1f);
        }
        // Start benchmark
        else if (event->key.scancode == SDL_SCANCODE_B) {
            if (!game->benchmark_running) {
                game_start_benchmark(game, 5000, 10);
            }
        }
        break;
    }
}

void game_start_benchmark(Game* game, int entity_count, int duration_seconds) {
    printf("Starting benchmark with target %d entities for %d seconds...\n",
        entity_count, duration_seconds);

    // Reset entity count in engine
    game->engine->entities.count = 0;

    // Set benchmark parameters
    game->benchmark_running = true;
    game->target_entity_count = entity_count;
    game->current_entity_count = 0;
    game->benchmark_duration_seconds = duration_seconds;
    game->benchmark_start_time = SDL_GetTicks();
    game->last_report_time = game->benchmark_start_time;
    game->total_ms = 0.0;
    game->total_frames = 0;
    game->frames_since_report = 0;
}

void game_update_benchmark(Game* game) {
    // Add entities gradually
    int entities_to_add = game->entities_per_frame;
    if (game->current_entity_count + entities_to_add > game->target_entity_count) {
        entities_to_add = game->target_entity_count - game->current_entity_count;
    }

    for (int i = 0; i < entities_to_add; i++) {
        generate_random_entity(game);
        game->current_entity_count++;
    }

    // Move camera to simulate world exploration
    Engine* engine = game->engine;
    engine_set_camera_position(engine, engine->camera.x + 2.0f, engine->camera.y);

    if (engine->camera.x > engine->world_bounds.w) {
        engine_set_camera_position(engine, 0.0f, engine->camera.y + 200.0f);

        if (engine->camera.y > engine->world_bounds.h) {
            engine_set_camera_position(engine, 0.0f, 0.0f);
        }
    }

    // Time frame execution
    Uint64 frame_start = SDL_GetPerformanceCounter();

    // The actual update happens in the main loop

    // Calculate frame time
    Uint64 frame_end = SDL_GetPerformanceCounter();
    double frame_ms = (frame_end - frame_start) * 1000.0 / SDL_GetPerformanceFrequency();

    // Skip first 100 frames (warmup)
    game->total_frames++;
    if (game->total_frames > 100) {
        game->total_ms += frame_ms;
        game->frames_since_report++;
    }

    // Report FPS every second
    Uint64 current_time = SDL_GetTicks();
    if (current_time - game->last_report_time >= 1000) {
        double elapsed_seconds = (current_time - game->last_report_time) / 1000.0;
        double current_fps = game->frames_since_report / elapsed_seconds;

        // Get visible entities count
        int visible_count = engine_get_visible_entities_count(game->engine);

        printf("Entities: %d/%d - FPS: %.2f (%.2f ms/frame) - Visible: %d\n",
            game->current_entity_count, game->target_entity_count,
            current_fps,
            game->frames_since_report > 0 ? (game->total_ms / game->frames_since_report) : 0,
            visible_count);

        game->frames_since_report = 0;
        game->last_report_time = current_time;
    }

    // Check if benchmark is complete
    if (game->current_entity_count >= game->target_entity_count &&
        current_time - game->benchmark_start_time >= game->benchmark_duration_seconds * 1000) {

        // Calculate final results
        double avg_ms = game->total_ms / (game->total_frames - 100); // Exclude warmup frames
        double avg_fps = 1000.0 / avg_ms;

        printf("Benchmark Complete:\n");
        printf("- Entities: %d\n", game->current_entity_count);
        printf("- Duration: %d seconds\n", (int)((current_time - game->benchmark_start_time) / 1000));
        printf("- Total Frames: %d\n", game->total_frames);
        printf("- Average Frame Time: %.2f ms\n", avg_ms);
        printf("- Average FPS: %.2f\n", avg_fps);

        // End benchmark
        game->benchmark_running = false;
    }
}



#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

// Main game structure
Game* game = NULL;

// SDL callbacks
SDL_AppResult SDL_AppInit(void** appstate, int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        SDL_Log("Could not initialize SDL: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    // Create game
    game = game_create(800, 600);
    if (!game) {
        SDL_Log("Failed to initialize game");
        SDL_Quit();
        return SDL_APP_FAILURE;
    }

    // Set app state
    *appstate = game;

    // Start benchmark automatically (for web)
    game_start_benchmark(game, 200000, 30);

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appstate) {
    Game* game = (Game*)appstate;

    // Update and render game
    game_update(game);
    game_render(game);

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event) {
    Game* game = (Game*)appstate;

    switch (event->type) {
    case SDL_EVENT_QUIT:
        return SDL_APP_FAILURE;

    case SDL_EVENT_KEY_DOWN:
        if (event->key.scancode == SDL_SCANCODE_ESCAPE) {
            return SDL_APP_FAILURE;
        }
        game_handle_event(game, event);
        break;
    }

    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult result) {
    Game* game = (Game*)appstate;

    if (game) {
        game_destroy(game);
    }

    SDL_Quit();
}