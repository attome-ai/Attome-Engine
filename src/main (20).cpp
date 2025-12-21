#define SDL_MAIN_USE_CALLBACKS 1

#include "ATMCommon.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include "SDL3_image/SDL_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

// ImGui includes
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "ATMEngine.h"

// Configuration for window and world
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1020
#define WORLD_WIDTH 50000
#define WORLD_HEIGHT 10000
#define GRID_CELL_SIZE 512  // Spatial grid cell size

// Entity performance test configuration
#define NUM_ENTITY_TYPES 1                // Total different entity types to create
#define MIN_ENTITIES_PER_TYPE 10000        // Minimum entities per type
#define MAX_ENTITIES_PER_TYPE 10000       // Maximum entities per type
#define ENTITY_WIDTH 32                    // Width of entities
#define ENTITY_HEIGHT 32                   // Height of entities

// Player configuration
#define PLAYER_WIDTH 32
#define PLAYER_HEIGHT 32
#define ENTITY_TYPE_PLAYER 0               // Keep player as special entity type

// Game state enum for UI flow
typedef enum {
    GAME_STATE_TITLE,
    GAME_STATE_PLAYING,
    GAME_STATE_PAUSED,
    GAME_STATE_GAME_OVER
} GameStateEnum;

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
    int num_entity_types;               // Actual number of entity types created
    int* entity_type_ids;               // Array of entity type IDs
    int* entity_counts;                 // How many entities of each type
    int* entity_texture_ids;            // Texture ID for each entity type
    Uint64* entity_update_times;        // Performance tracking

    // ImGui-related fields
    float fps;                          // Current FPS value
    float fps_history[100];             // Store FPS history for plotting
    int fps_history_index;              // Current index in the FPS history array
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

// UI rendering functions
void RenderTitleScreen(GameState* state);
void RenderGameHUD(GameState* state, float delta_time);
void RenderDebugInfo(GameState* state);
void RenderGameOverScreen(GameState* state);
void RenderPauseScreen(GameState* state);

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
    SDL_FillSurfaceRect(surface, NULL, SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format),0, r, g, b, 255));
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

        // Create entities of this type with better distribution
        for (int i = 0; i < entity_count; i++) {
            // Calculate a grid position for better distribution
            int grid_columns = (int)SDL_sqrt((float)entity_count);
            int grid_rows = (entity_count + grid_columns - 1) / grid_columns;

            int grid_x = i % grid_columns;
            int grid_y = i / grid_columns;

            // Add some randomness
            float rand_offset_x = (float)((rand() % 400) - 200);
            float rand_offset_y = (float)((rand() % 400) - 200);

            // Calculate position to ensure distribution across world
            float spacing_x = WORLD_WIDTH / grid_columns;
            float spacing_y = WORLD_HEIGHT / grid_rows;

            float x = spacing_x * grid_x + (spacing_x / 2) + rand_offset_x;
            float y = spacing_y * grid_y + (spacing_y / 2) + rand_offset_y;

            // Clamp to world bounds
            x = SDL_clamp(x, 0, WORLD_WIDTH - ENTITY_WIDTH);
            y = SDL_clamp(y, 0, WORLD_HEIGHT - ENTITY_HEIGHT);

            // Keep a clear area around the player's starting position
            float player_start_x = WINDOW_WIDTH / 2;
            float player_start_y = WINDOW_HEIGHT / 2;
            float clear_radius = 300.0f;

            float dx = x - player_start_x;
            float dy = y - player_start_y;
            float dist = SDL_sqrtf(dx * dx + dy * dy);

            if (dist < clear_radius) {
                // Move away from player start
                if (dist > 0) {
                    x = player_start_x + (dx / dist) * clear_radius;
                    y = player_start_y + (dy / dist) * clear_radius;
                }
                else {
                    // Random direction if at exact same position
                    float angle = (float)(rand() % 360) * 3.14159f / 180.0f;
                    x = player_start_x + SDL_cosf(angle) * clear_radius;
                    y = player_start_y + SDL_sinf(angle) * clear_radius;
                }

                // Clamp again after adjustment
                x = SDL_clamp(x, 0, WORLD_WIDTH - ENTITY_WIDTH);
                y = SDL_clamp(y, 0, WORLD_HEIGHT - ENTITY_HEIGHT);
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

// Title screen rendering
void RenderTitleScreen(GameState* state) {
    ImGuiIO& io = ImGui::GetIO();

    // Calculate animation values
    state->title_animation = 0.5f + 0.5f * sinf(SDL_GetTicks() / 300.0f);

    // Center the title window
    ImVec2 center = ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);
    ImVec2 window_size = ImVec2(400, 300);
    ImGui::SetNextWindowPos(ImVec2(center.x - window_size.x / 2, center.y - window_size.y / 2));
    ImGui::SetNextWindowSize(window_size);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 20));
    ImGui::Begin("##TitleScreen", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);

    // Title with pulsing animation
    ImGui::PushFont(io.Fonts->Fonts[0]); // Assuming default font

    ImVec4 title_color = ImVec4(0.3f, 0.5f, 0.9f, 0.8f + 0.2f * state->title_animation);
    ImGui::PushStyleColor(ImGuiCol_Text, title_color);

    float title_scale = 1.8f + 0.1f * state->title_animation;
    ImVec2 text_size = ImGui::CalcTextSize("ENTITY ENGINE");
    ImVec2 pos = ImGui::GetCursorPos();
    ImGui::SetCursorPos(ImVec2(
        window_size.x / 2 - text_size.x * title_scale / 2,
        pos.y + 30
    ));

    ImGui::SetWindowFontScale(title_scale);
    ImGui::Text("ENTITY ENGINE");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();
    ImGui::PopFont();

    // Subtitle
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 30);
    ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("Performance Demo").x / 2);
    ImGui::Text("Performance Demo");

    // Add some space before buttons
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 40);

    // Center and style the buttons
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 8));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 16));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.25f, 0.65f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.35f, 0.8f, 0.9f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.3f, 0.7f, 1.0f));

    ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("  Start Game  ").x / 2 - 10);
    if (ImGui::Button("  Start Game  ", ImVec2(0, 0))) {
        state->current_state = GAME_STATE_PLAYING;
        state->game_time_start = SDL_GetTicks();
        state->player_health = state->max_health +100000;
        state->score = 0;
    }

    ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("  Controls  ").x / 2 - 10);
    if (ImGui::Button("  Controls  ", ImVec2(0, 0))) {
        state->show_controls = !state->show_controls;
    }

    ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("  Exit  ").x / 2 - 10);
    if (ImGui::Button("  Exit  ", ImVec2(0, 0))) {
        SDL_Event quit_event;
        quit_event.type = SDL_EVENT_QUIT;
        SDL_PushEvent(&quit_event);
    }

    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);

    // Show controls if requested
    if (state->show_controls) {
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 20);
        ImGui::Separator();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 10);
        ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("Controls").x / 2);
        ImGui::Text("Controls");
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5);
        ImGui::Columns(2, "ControlsColumns");
        ImGui::Text("Move Up"); ImGui::NextColumn(); ImGui::Text("W"); ImGui::NextColumn();
        ImGui::Text("Move Left"); ImGui::NextColumn(); ImGui::Text("A"); ImGui::NextColumn();
        ImGui::Text("Move Down"); ImGui::NextColumn(); ImGui::Text("S"); ImGui::NextColumn();
        ImGui::Text("Move Right"); ImGui::NextColumn(); ImGui::Text("D"); ImGui::NextColumn();
        ImGui::Text("Toggle Debug"); ImGui::NextColumn(); ImGui::Text("F1"); ImGui::NextColumn();
        ImGui::Text("Pause"); ImGui::NextColumn(); ImGui::Text("ESC"); ImGui::NextColumn();
        ImGui::Columns(1);
    }

    ImGui::End();
    ImGui::PopStyleVar();
}

// In-game HUD rendering
void RenderGameHUD(GameState* state, float delta_time) {
    ImGuiIO& io = ImGui::GetIO();
    Engine* engine = state->engine;

    // Smooth animations
    state->score_animation = state->score_animation * 0.9f + state->score * 0.1f;
    state->health_animation = state->health_animation * 0.9f + state->player_health * 0.1f;

    // Update damage flash effect (decreases over time)
    if (state->damage_flash > 0) {
        state->damage_flash -= delta_time * 2.0f;
        if (state->damage_flash < 0) state->damage_flash = 0;
    }

    // Top-left HUD panel
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::SetNextWindowSize(ImVec2(200, 120));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 2.0f);
    ImGui::Begin("Game HUD", NULL,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar);

    // Score with animation
    ImGui::Text("SCORE:");
    ImGui::SameLine();

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.9f, 0.2f, 1.0f));
    ImGui::SetWindowFontScale(1.4f);
    ImGui::Text("%.0f", state->score_animation);
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();

    // Elapsed time
    Uint64 elapsed_time = SDL_GetTicks() - state->game_time_start;
    int minutes = (elapsed_time / 60000) % 60;
    int seconds = (elapsed_time / 1000) % 60;

    ImGui::Text("TIME: %02d:%02d", minutes, seconds);

    // Health bar
    ImGui::Text("HEALTH:");
    float health_ratio = state->health_animation / state->max_health;

    // Health bar colors based on health percentage
    ImVec4 health_color;
    if (health_ratio > 0.7f) {
        health_color = ImVec4(0.0f, 0.8f, 0.1f, 1.0f); // Green for high health
    }
    else if (health_ratio > 0.3f) {
        health_color = ImVec4(0.9f, 0.7f, 0.0f, 1.0f); // Yellow for medium health
    }
    else {
        health_color = ImVec4(0.9f, 0.1f, 0.1f, 1.0f); // Red for low health

        // Pulse animation for low health
        if (health_ratio < 0.3f) {
            float pulse = 0.5f + 0.5f * sinf(SDL_GetTicks() / 200.0f);
            health_color.w = 0.5f + 0.5f * pulse;
        }
    }

    // If taking damage, flash the health bar
    if (state->damage_flash > 0) {
        health_color = ImVec4(1.0f, 0.0f, 0.0f, state->damage_flash);
    }

    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, health_color);
    ImGui::ProgressBar(health_ratio, ImVec2(-1.0f, 0.0f));
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleVar();

    // Bottom-right controls reminder (fades out after a few seconds)
    static Uint64 controls_fade_timer = 0;
    if (state->game_time_start > 0 && SDL_GetTicks() - state->game_time_start < 5000) {
        float alpha = 1.0f - ((float)(SDL_GetTicks() - state->game_time_start) / 5000.0f);

        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 180, io.DisplaySize.y - 100));
        ImGui::SetNextWindowSize(ImVec2(170, 90));
        ImGui::SetNextWindowBgAlpha(alpha * 0.7f);

        ImGui::Begin("Controls Reminder", NULL,
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar);

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, alpha));
        ImGui::Text("Controls:");
        ImGui::Text("WASD - Move");
        ImGui::Text("F1 - Toggle Debug");
        ImGui::Text("ESC - Pause");
        ImGui::PopStyleColor();

        ImGui::End();
    }
}

// Debug performance info panel
void RenderDebugInfo(GameState* state) {
    Engine* engine = state->engine;

    // Performance window (right side)
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 310, 10));
    ImGui::SetNextWindowSize(ImVec2(300, 400)); // Increased height for entity visualization
    ImGui::Begin("Debug Info", NULL, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("FPS: %.2f", state->fps);
    ImGui::Text("Active Entities: %d", engine->entities.total_count);

    // Player position with coordinate styling
    ImGui::Text("Player Position:");
    int chunk_idx, local_idx;
    engine->entities.getChunkIndices(state->player_entity, &chunk_idx, &local_idx);

    if (chunk_idx < engine->entities.chunk_count) {
        EntityChunk* chunk = engine->entities.chunks[chunk_idx];
        if (local_idx < chunk->count) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f),
                "(%.1f, %.1f)", chunk->x[local_idx], chunk->y[local_idx]);
        }
    }

    // Camera position
    ImGui::Text("Camera Position: (%.1f, %.1f)", engine->camera.x, engine->camera.y);
    ImGui::Text("Camera Viewport: %.1f x %.1f", engine->camera.width, engine->camera.height);

    // Plot FPS history with improved styling
    ImGui::PlotLines("FPS History", state->fps_history, 100, state->fps_history_index,
        NULL, 0.0f, 120.0f, ImVec2(280, 80));

    ImGui::Separator();

    // Entity count by type (collapsible section)
    if (ImGui::CollapsingHeader("Entity Types", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Columns(2, "EntityTypesColumns");
        ImGui::Text("Type"); ImGui::NextColumn(); ImGui::Text("Count"); ImGui::NextColumn();
        ImGui::Separator();

        for (int i = 0; i < state->num_entity_types; i++) {
            if (i % 10 == 0) { // Show only every 10th type to avoid cluttering
                ImGui::Text("Type %d", state->entity_type_ids[i]);
                ImGui::NextColumn();
                ImGui::Text("%d", state->entity_counts[i]);
                ImGui::NextColumn();
            }
        }

        ImGui::Columns(1);
    }

    // Mini-map visualization (new section)
    if (ImGui::CollapsingHeader("World Map", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImVec2 mini_map_size(280, 140);
        ImGui::BeginChild("MiniMap", mini_map_size, true);
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        ImVec2 mini_map_pos = ImGui::GetCursorScreenPos();
        float map_scale_x = mini_map_size.x / WORLD_WIDTH;
        float map_scale_y = mini_map_size.y / WORLD_HEIGHT;

        // Draw world boundary
        draw_list->AddRect(
            ImVec2(mini_map_pos.x, mini_map_pos.y),
            ImVec2(mini_map_pos.x + mini_map_size.x, mini_map_pos.y + mini_map_size.y),
            IM_COL32(255, 255, 255, 128)
        );

        // Draw visible area (camera view)
        ImVec2 view_min(
            mini_map_pos.x + engine->camera.x * map_scale_x,
            mini_map_pos.y + engine->camera.y * map_scale_y
        );
        ImVec2 view_max(
            mini_map_pos.x + (engine->camera.x + engine->camera.width) * map_scale_x,
            mini_map_pos.y + (engine->camera.y + engine->camera.height) * map_scale_y
        );
        draw_list->AddRect(view_min, view_max, IM_COL32(0, 255, 0, 196), 0, 0, 2.0f);

        // Draw player position
        if (chunk_idx < engine->entities.chunk_count) {
            EntityChunk* chunk = engine->entities.chunks[chunk_idx];
            if (local_idx < chunk->count) {
                float player_x = chunk->x[local_idx];
                float player_y = chunk->y[local_idx];
                ImVec2 player_pos(
                    mini_map_pos.x + player_x * map_scale_x,
                    mini_map_pos.y + player_y * map_scale_y
                );
                draw_list->AddCircleFilled(player_pos, 3.0f, IM_COL32(0, 0, 255, 255));
            }
        }

        // Sample some entities to draw on mini-map (to avoid drawing thousands)
        const int max_sample = 100;
        int total_entities = engine->entities.total_count;
        int sample_step = total_entities > max_sample ? total_entities / max_sample : 1;

        for (int chunk_i = 0; chunk_i < engine->entities.chunk_count; chunk_i++) {
            EntityChunk* chunk = engine->entities.chunks[chunk_i];
            for (int i = 0; i < chunk->count; i += sample_step) {
                if (!chunk->active[i] || chunk->type_id == ENTITY_TYPE_PLAYER) continue;

                float entity_x = chunk->x[i];
                float entity_y = chunk->y[i];
                ImVec2 entity_pos(
                    mini_map_pos.x + entity_x * map_scale_x,
                    mini_map_pos.y + entity_y * map_scale_y
                );

                // Get entity color if available
                ImU32 entity_color = IM_COL32(255, 0, 0, 128);
                if (chunk->type_data) {
                    // Try to get the color from GenericEntityData
                    GenericEntityData* data = (GenericEntityData*)((uint8_t*)chunk->type_data + i * sizeof(GenericEntityData));
                    entity_color = IM_COL32(data->color[0], data->color[1], data->color[2], 128);
                }

                draw_list->AddCircleFilled(entity_pos, 1.5f, entity_color);
            }
        }

        ImGui::EndChild();
    }

    ImGui::End();
}

// Game over screen
void RenderGameOverScreen(GameState* state) {
    ImGuiIO& io = ImGui::GetIO();

    // Calculate animation values
    state->title_animation = 0.5f + 0.5f * sinf(SDL_GetTicks() / 300.0f);

    // Center the game over window
    ImVec2 center = ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);
    ImVec2 window_size = ImVec2(400, 300);
    ImGui::SetNextWindowPos(ImVec2(center.x - window_size.x / 2, center.y - window_size.y / 2));
    ImGui::SetNextWindowSize(window_size);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 20));
    ImGui::Begin("##GameOverScreen", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);

    // Game Over title with animation
    ImGui::PushFont(io.Fonts->Fonts[0]); // Assuming default font

    ImVec4 title_color = ImVec4(0.9f, 0.3f, 0.3f, 0.8f + 0.2f * state->title_animation);
    ImGui::PushStyleColor(ImGuiCol_Text, title_color);

    float title_scale = 1.8f + 0.1f * state->title_animation;
    ImVec2 text_size = ImGui::CalcTextSize("GAME OVER");
    ImVec2 pos = ImGui::GetCursorPos();
    ImGui::SetCursorPos(ImVec2(
        window_size.x / 2 - text_size.x * title_scale / 2,
        pos.y + 30
    ));

    ImGui::SetWindowFontScale(title_scale);
    ImGui::Text("GAME OVER");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();
    ImGui::PopFont();

    // Display final score
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 40);
    ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("Final Score: ").x / 2 - 20);

    ImGui::Text("Final Score: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.2f, 1.0f), "%d", state->score);

    // Add some space before buttons
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 40);

    // Center and style the buttons
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 8));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 16));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.25f, 0.65f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.35f, 0.8f, 0.9f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.3f, 0.7f, 1.0f));

    ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("  Play Again  ").x / 2 - 10);
    if (ImGui::Button("  Play Again  ", ImVec2(0, 0))) {
        state->current_state = GAME_STATE_PLAYING;
        state->game_time_start = SDL_GetTicks();
        state->player_health = state->max_health;
        state->score = 0;

        // Reset player position
        engine_set_entity_position(state->engine, state->player_entity,
            WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);

        // Reinitialize entities
        init_entities(state);
    }

    ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("  Main Menu  ").x / 2 - 10);
    if (ImGui::Button("  Main Menu  ", ImVec2(0, 0))) {
        state->current_state = GAME_STATE_TITLE;
    }

    ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("  Exit  ").x / 2 - 10);
    if (ImGui::Button("  Exit  ", ImVec2(0, 0))) {
        SDL_Event quit_event;
        quit_event.type = SDL_EVENT_QUIT;
        SDL_PushEvent(&quit_event);
    }

    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);

    ImGui::End();
    ImGui::PopStyleVar();
}

// Pause menu screen
void RenderPauseScreen(GameState* state) {
    ImGuiIO& io = ImGui::GetIO();

    // Center the pause window
    ImVec2 center = ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);
    ImVec2 window_size = ImVec2(300, 250);
    ImGui::SetNextWindowPos(ImVec2(center.x - window_size.x / 2, center.y - window_size.y / 2));
    ImGui::SetNextWindowSize(window_size);

    ImGui::Begin("##PauseScreen", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);

    // Pause title
    ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("PAUSED").x / 2);
    ImGui::SetWindowFontScale(1.5f);
    ImGui::Text("PAUSED");
    ImGui::SetWindowFontScale(1.0f);

    // Add some space before buttons
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 20);

    // Center and style the buttons
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 8));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 12));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.25f, 0.65f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.35f, 0.8f, 0.9f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.3f, 0.7f, 1.0f));

    ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("  Resume  ").x / 2 - 10);
    if (ImGui::Button("  Resume  ", ImVec2(0, 0))) {
        state->current_state = GAME_STATE_PLAYING;
    }

    ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("  Restart  ").x / 2 - 10);
    if (ImGui::Button("  Restart  ", ImVec2(0, 0))) {
        state->current_state = GAME_STATE_PLAYING;
        state->game_time_start = SDL_GetTicks();
        state->player_health = state->max_health;
        state->score = 0;

        // Reset player position
        engine_set_entity_position(state->engine, state->player_entity,
            WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);

        // Reinitialize entities
        init_entities(state);
    }

    ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("  Main Menu  ").x / 2 - 10);
    if (ImGui::Button("  Main Menu  ", ImVec2(0, 0))) {
        state->current_state = GAME_STATE_TITLE;
    }

    ImGui::SetCursorPosX(window_size.x / 2 - ImGui::CalcTextSize("  Exit  ").x / 2 - 10);
    if (ImGui::Button("  Exit  ", ImVec2(0, 0))) {
        SDL_Event quit_event;
        quit_event.type = SDL_EVENT_QUIT;
        SDL_PushEvent(&quit_event);
    }

    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);

    ImGui::End();
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

    // Initialize ImGui-related fields in game state
    state->fps = 0.0f;
    state->fps_history_index = 0;
    for (int i = 0; i < 100; i++) {
        state->fps_history[i] = 0.0f;
    }

    // Initialize game state variables
    state->current_state = GAME_STATE_TITLE;
    state->max_health = 100;
    state->player_health = state->max_health;
    state->show_controls = false;
    state->show_debug_info = true; // Default to showing debug info for now
    state->title_animation = 0.0f;
    state->score_animation = 0.0f;
    state->health_animation = (float)state->max_health;
    state->damage_flash = 0.0f;

    // Initialize game engine with spatial grid cell size parameter
    state->engine = engine_create(WINDOW_WIDTH, WINDOW_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT, GRID_CELL_SIZE);
    if (!state->engine) {
        SDL_Log("Failed to create engine");
        SDL_free(state);
        SDL_Quit();
        return SDL_APP_FAILURE;
    }

    // Get window and renderer from engine (already created in engine_create)
    SDL_Window* window = state->engine->window;
    SDL_Renderer* renderer = state->engine->renderer;

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_IsSRGB;      // Enable Docking

    // Set up custom ImGui game style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();

    // More game-like rounded corners
    style.WindowRounding = 8.0f;
    style.FrameRounding = 4.0f;
    style.PopupRounding = 4.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding = 4.0f;
    style.TabRounding = 4.0f;

    // Slightly larger padding for better touch/click targets
    style.WindowPadding = ImVec2(10, 10);
    style.FramePadding = ImVec2(6, 4);
    style.ItemSpacing = ImVec2(8, 6);

    // Thicker borders for visibility
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;

    // Adjust colors for a game UI feel with blue accent color
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.08f, 0.08f, 0.12f, 0.95f);
    colors[ImGuiCol_Header] = ImVec4(0.20f, 0.25f, 0.58f, 0.55f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.33f, 0.75f, 0.80f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.24f, 0.31f, 0.70f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.20f, 0.25f, 0.58f, 0.55f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.33f, 0.75f, 0.80f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.24f, 0.31f, 0.70f, 1.00f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.13f, 0.14f, 0.30f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.17f, 0.19f, 0.40f, 1.00f);
    colors[ImGuiCol_PlotLines] = ImVec4(0.30f, 0.65f, 0.95f, 1.00f);
    colors[ImGuiCol_Text] = ImVec4(0.90f, 0.90f, 0.90f, 1.00f);
    colors[ImGuiCol_Border] = ImVec4(0.30f, 0.30f, 0.50f, 0.60f);

    // Setup Platform/Renderer backends using the engine's window and renderer
    ImGui_ImplSDL3_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer3_Init(renderer);

    // Engine already has renderer and window set up in engine_create

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
        ImGui_ImplSDLRenderer3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return SDL_APP_FAILURE;
    }

    // Create player textures
    SDL_Surface* player_surface1 = create_colored_surface(PLAYER_WIDTH, PLAYER_HEIGHT, 0, 0, 255);

    // Add player textures to engine
    int player_texture1 = engine_add_texture(state->engine, player_surface1, 0, 0);

    // Clean up player surfaces
    SDL_DestroySurface(player_surface1);

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

        // Clear key tracking
        for (int i = 0; i < 4; i++) {
            player_data->keys_pressed[i] = SDL_SCANCODE_UNKNOWN;
        }
    }

    // Create textures and register entity types for each test entity type
    for (int i = 0; i < state->num_entity_types; i++) {
        // Create a unique entity type ID
        int type_id = ENTITY_TYPE_PLAYER+3 + i; // Start at 2 because ENTITY_TYPE_PLAYER is 1
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

    // Center camera on player at start
    float camera_x = WINDOW_WIDTH / 2 - WINDOW_WIDTH / 2;
    float camera_y = WINDOW_HEIGHT / 2 - WINDOW_HEIGHT / 2;
    engine_set_camera_position(state->engine, camera_x, camera_y);

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

    // Calculate FPS (using a rolling average for stability)
    static Uint64 fps_last_time = 0;
    static int fps_frames = 0;
    static Uint64 performance_report_timer = 0;

    fps_frames++;

    if (current_time - fps_last_time >= 1000) {
        state->fps = fps_frames * 1000.0f / (current_time - fps_last_time);

        // Update FPS history for plotting
        state->fps_history[state->fps_history_index] = state->fps;
        state->fps_history_index = (state->fps_history_index + 1) % 100;

        SDL_Log("FPS: %.2f, Active entities: %d", state->fps, engine->entities.total_count);
        fps_last_time = current_time;
        fps_frames = 0;
    }

    // Start the Dear ImGui frame
    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    // Handle different game states for UI and logic
    switch (state->current_state) {
    case GAME_STATE_TITLE:
        // Render title screen - no game updates
        RenderTitleScreen(state);
        break;

    case GAME_STATE_PLAYING:
        // Update game logic
        if (!state->game_over) {
            // Get player position for camera update
            int chunk_idx, local_idx;
            engine->entities.getChunkIndices(state->player_entity, &chunk_idx, &local_idx);

            if (chunk_idx < engine->entities.chunk_count) {
                EntityChunk* chunk = engine->entities.chunks[chunk_idx];

                if (local_idx < chunk->count) {
                    float player_x = chunk->x[local_idx];
                    float player_y = chunk->y[local_idx];

                    // Update camera to follow player - properly center the view
                    float camera_x = player_x + PLAYER_WIDTH / 2 - WINDOW_WIDTH / 2;
                    float camera_y = player_y + PLAYER_HEIGHT / 2 - WINDOW_HEIGHT / 2;
                    engine_set_camera_position(engine, camera_x, camera_y);
                }
            }

            // Update all entities
            engine_update(engine);

            // Increment score while playing
            state->score += 1;

            // Randomly reduce health occasionally for testing health bar
            if (rand() % 300 == 0) {
                state->player_health -= 5;
                state->damage_flash = 1.0f; // Set damage flash effect

                if (state->player_health <= 0) {
                    state->player_health = 0;
                    state->current_state = GAME_STATE_GAME_OVER;
                }
            }
        }

        // Render game HUD
        RenderGameHUD(state, delta_time);

        // Render debug info if enabled
        if (state->show_debug_info) {
            RenderDebugInfo(state);
        }
        break;

    case GAME_STATE_PAUSED:
        // Render pause menu
        RenderPauseScreen(state);
        break;

    case GAME_STATE_GAME_OVER:
        // Render game over screen
        RenderGameOverScreen(state);
        break;
    }

    // Render updated scene
    engine_render(engine);

    // Render ImGui
    ImGui::Render();
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), engine->renderer);

    // Present the rendered frame
    SDL_RenderPresent(engine->renderer);

    // Update last frame time for proper delta time calculation
    engine->last_frame_time = current_time;

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event) {
    GameState* state = (GameState*)appstate;

    // Process ImGui events
    ImGui_ImplSDL3_ProcessEvent(event);

    // If ImGui is capturing mouse/keyboard, don't process game events
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse || io.WantCaptureKeyboard)
        return SDL_APP_CONTINUE;

    switch (event->type) {
    case SDL_EVENT_KEY_DOWN:
        // Toggle debug info with F1
        if (event->key.scancode == SDL_SCANCODE_F1) {
            state->show_debug_info = !state->show_debug_info;
        }

        // Handle ESC key for pausing/returning to title
        if (event->key.scancode == SDL_SCANCODE_ESCAPE) {
            if (state->current_state == GAME_STATE_PLAYING) {
                state->current_state = GAME_STATE_PAUSED;
            }
            else if (state->current_state == GAME_STATE_PAUSED) {
                state->current_state = GAME_STATE_PLAYING;
            }
        }

        // Restart on 'R' press when game over
        if (state->current_state == GAME_STATE_GAME_OVER && event->key.scancode == SDL_SCANCODE_R) {
            state->current_state = GAME_STATE_PLAYING;
            state->game_over = false;
            state->score = 0;
            state->player_health = state->max_health;

            // Reset player position
            engine_set_entity_position(state->engine, state->player_entity,
                WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);

            // Reinitialize entities
            init_entities(state);
        }

        // Update key state in player data (only when playing)
        if (state->current_state == GAME_STATE_PLAYING) {
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
        if (state->current_state == GAME_STATE_PLAYING) {
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
        // Clean up ImGui first before engine destroys renderer and window
        ImGui_ImplSDLRenderer3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();

        // Destroy engine (this will also destroy window and renderer)
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