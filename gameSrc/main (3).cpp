
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
#include "ATMNetwork.h"
#include "ATMProtocol.h" // Include the protocol definitions

// SIMD detection macros
#if defined(__EMSCRIPTEN__)
    // Emscripten SIMD detection
#if defined(__wasm_simd128__)
#define HAS_WASM_SIMD 1
#include <wasm_simd128.h>
#else
#define HAS_WASM_SIMD 0
#endif
#define HAS_SSE 0
#elif defined(__SSE__) && !defined(__EMSCRIPTEN__)
    // x86/x64 SSE detection
#define HAS_SSE 1
#include <xmmintrin.h> // SSE
#if defined(__SSE2__)
#include <emmintrin.h> // SSE2
#endif
#else
#define HAS_SSE 0
#endif

// Configuration for window and world
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1020
#define WORLD_WIDTH 50000
#define WORLD_HEIGHT 50000

// Entity performance test configuration
#define NUM_ENTITY_TYPES 1000                // Total different entity types to create
#define MIN_ENTITIES_PER_TYPE 1000       // Minimum entities per type
#define MAX_ENTITIES_PER_TYPE 1000       // Maximum entities per type
#define ENTITY_WIDTH 32                    // Width of entities
#define ENTITY_HEIGHT 32                   // Height of entities
#define ADD_ENTITIES_AT_ONCE 1000
#define ENTITIES_TO_REMOVE_AT_ONCE 1000
#define PLAYER_WIDTH 32
#define PLAYER_HEIGHT 32

// Game state enum for UI flow
typedef enum {
    GAME_STATE_AUTH,          // New authentication state (initial state)
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
    int player_type_id;   // Added to track player entity type ID
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

    // Authentication-related fields
    ReliableOrderedUDP network;        // Network client for authentication (changed from DefaultUDP)
    SocketAddress serverAddr;         // Server address
    bool isConnected;                 // Connected to server flag
    uint16_t userId;                  // User ID from server
    char errorMessage[256];           // Authentication error message
    bool isAuthenticating;            // Authentication in progress flag
    bool showLoginForm;               // Show login form flag
    bool showRegisterForm;            // Show register form flag
    char loginUsername[MAX_USERNAME_LENGTH];      // Login username
    char loginPassword[MAX_PASSWORD_LENGTH];      // Login password
    char registerUsername[MAX_USERNAME_LENGTH];   // Register username
    char registerPassword[MAX_PASSWORD_LENGTH];   // Register password
    char loggedInUsername[MAX_USERNAME_LENGTH];   // Username after successful login
    bool isGuest;                     // Whether logged in as guest
    Uint64 lastHeartbeatTime;         // Time of last heartbeat
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

// Authentication functions
void initializeNetwork(GameState* state);
void sendLoginRequest(GameState* state);
void sendRegisterRequest(GameState* state);
void sendGuestLoginRequest(GameState* state);
void handleAuthenticationMessages(GameState* state);
void sendHeartbeatIfNeeded(GameState* state);
void sendDisconnectMessage(GameState* state);

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
    SDL_FillSurfaceRect(surface, NULL, SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format), 0, r, g, b, 255));
    return surface;
}

// Generic entity update function - processes an entire chunk of entities
void generic_entity_update(EntityChunk* chunk, int count, float delta_time) {
    // Skip empty chunks
    if (count == 0) return;

#if HAS_SSE
    // SSE implementation for x86/x64
    // Constants for vectorization
    const __m128 zero = _mm_setzero_ps();
    const __m128 world_width = _mm_set1_ps(WORLD_WIDTH);
    const __m128 world_height = _mm_set1_ps(WORLD_HEIGHT);
    const __m128 neg_one = _mm_set1_ps(-1.0f);
    const __m128 point_nine = _mm_set1_ps(0.9f);
    const __m128 point_one = _mm_set1_ps(0.1f);
    const __m128 delta = _mm_set1_ps(delta_time);
    const __m128 center_x = _mm_set1_ps(WORLD_WIDTH / 2);
    const __m128 center_y = _mm_set1_ps(WORLD_HEIGHT / 2);

    // Process entities in blocks of 4 (SSE width)
    int i = 0;
    for (; i + 3 < count; i += 4) {
        // Load entity data for 4 entities
        GenericEntityData* data0 = (GenericEntityData*)((uint8_t*)chunk->type_data + i * sizeof(GenericEntityData));
        GenericEntityData* data1 = (GenericEntityData*)((uint8_t*)chunk->type_data + (i + 1) * sizeof(GenericEntityData));
        GenericEntityData* data2 = (GenericEntityData*)((uint8_t*)chunk->type_data + (i + 2) * sizeof(GenericEntityData));
        GenericEntityData* data3 = (GenericEntityData*)((uint8_t*)chunk->type_data + (i + 3) * sizeof(GenericEntityData));

        // Load positions
        __m128 x = _mm_set_ps(chunk->x[i + 3], chunk->x[i + 2], chunk->x[i + 1], chunk->x[i]);
        __m128 y = _mm_set_ps(chunk->y[i + 3], chunk->y[i + 2], chunk->y[i + 1], chunk->y[i]);

        // Load widths and heights
        __m128 width = _mm_set_ps(chunk->width[i + 3], chunk->width[i + 2], chunk->width[i + 1], chunk->width[i]);
        __m128 height = _mm_set_ps(chunk->height[i + 3], chunk->height[i + 2], chunk->height[i + 1], chunk->height[i]);

        // Load directions and speeds
        __m128 dir_x = _mm_set_ps(
            data3->direction[0], data2->direction[0],
            data1->direction[0], data0->direction[0]
        );

        __m128 dir_y = _mm_set_ps(
            data3->direction[1], data2->direction[1],
            data1->direction[1], data0->direction[1]
        );

        __m128 speed = _mm_set_ps(
            data3->speed, data2->speed, data1->speed, data0->speed
        );

        // Calculate movement
        __m128 speed_time = _mm_mul_ps(speed, delta);
        __m128 dx = _mm_mul_ps(dir_x, speed_time);
        __m128 dy = _mm_mul_ps(dir_y, speed_time);
        __m128 new_x = _mm_add_ps(x, dx);
        __m128 new_y = _mm_add_ps(y, dy);

        // Calculate world boundaries
        __m128 max_x = _mm_sub_ps(world_width, width);
        __m128 max_y = _mm_sub_ps(world_height, height);

        // Check for boundary collisions
        __m128 x_min_cmp = _mm_cmplt_ps(new_x, zero);
        __m128 x_max_cmp = _mm_cmpgt_ps(new_x, max_x);
        __m128 y_min_cmp = _mm_cmplt_ps(new_y, zero);
        __m128 y_max_cmp = _mm_cmpgt_ps(new_y, max_y);

        // Combine collision masks
        __m128 x_bounce = _mm_or_ps(x_min_cmp, x_max_cmp);
        __m128 y_bounce = _mm_or_ps(y_min_cmp, y_max_cmp);

        // Apply direction reflection where needed
        __m128 dir_x_reflect = _mm_and_ps(x_bounce, neg_one);
        __m128 dir_y_reflect = _mm_and_ps(y_bounce, neg_one);
        __m128 dir_x_keep = _mm_andnot_ps(x_bounce, _mm_set1_ps(1.0f));
        __m128 dir_y_keep = _mm_andnot_ps(y_bounce, _mm_set1_ps(1.0f));

        __m128 dir_x_factor = _mm_or_ps(dir_x_reflect, dir_x_keep);
        __m128 dir_y_factor = _mm_or_ps(dir_y_reflect, dir_y_keep);

        dir_x = _mm_mul_ps(dir_x, dir_x_factor);
        dir_y = _mm_mul_ps(dir_y, dir_y_factor);

        // Clamp positions to world boundaries
        new_x = _mm_max_ps(zero, new_x);
        new_x = _mm_min_ps(max_x, new_x);
        new_y = _mm_max_ps(zero, new_y);
        new_y = _mm_min_ps(max_y, new_y);

        // Calculate right/bottom values
        __m128 right = _mm_add_ps(new_x, width);
        __m128 bottom = _mm_add_ps(new_y, height);

        // Process orbit behavior
        __m128 behavior_mask = _mm_set_ps(
            (data3->behavior_flags & 0x1) ? -1.0f : 0.0f,
            (data2->behavior_flags & 0x1) ? -1.0f : 0.0f,
            (data1->behavior_flags & 0x1) ? -1.0f : 0.0f,
            (data0->behavior_flags & 0x1) ? -1.0f : 0.0f
        );

        // Only perform orbit calculations if at least one entity needs it
        if (_mm_movemask_ps(behavior_mask)) {
            // Calculate vector to center
            __m128 dx_center = _mm_sub_ps(new_x, center_x);
            __m128 dy_center = _mm_sub_ps(new_y, center_y);

            // Distance calculation
            __m128 dist_sq = _mm_add_ps(
                _mm_mul_ps(dx_center, dx_center),
                _mm_mul_ps(dy_center, dy_center)
            );
            __m128 dist = _mm_sqrt_ps(dist_sq);

            // Avoid division by zero - create a safe denominator
            __m128 valid_dist = _mm_cmpgt_ps(dist, _mm_set1_ps(0.001f));
            __m128 safe_dist = _mm_or_ps(
                _mm_and_ps(valid_dist, dist),
                _mm_andnot_ps(valid_dist, _mm_set1_ps(1.0f))
            );

            // Perpendicular normalized vector components (-dy/dist, dx/dist)
            __m128 nx = _mm_div_ps(_mm_mul_ps(dy_center, neg_one), safe_dist);
            __m128 ny = _mm_div_ps(dx_center, safe_dist);

            // Mix with current direction (0.9*current + 0.1*orbit)
            __m128 new_dir_x = _mm_add_ps(
                _mm_mul_ps(dir_x, point_nine),
                _mm_mul_ps(nx, point_one)
            );
            __m128 new_dir_y = _mm_add_ps(
                _mm_mul_ps(dir_y, point_nine),
                _mm_mul_ps(ny, point_one)
            );

            // Apply only to entities with orbit behavior flag
            dir_x = _mm_or_ps(
                _mm_and_ps(behavior_mask, new_dir_x),
                _mm_andnot_ps(behavior_mask, dir_x)
            );
            dir_y = _mm_or_ps(
                _mm_and_ps(behavior_mask, new_dir_y),
                _mm_andnot_ps(behavior_mask, dir_y)
            );

            // Normalize direction vector
            __m128 len_sq = _mm_add_ps(
                _mm_mul_ps(dir_x, dir_x),
                _mm_mul_ps(dir_y, dir_y)
            );
            __m128 len = _mm_sqrt_ps(len_sq);
            __m128 valid_len = _mm_cmpgt_ps(len, _mm_set1_ps(0.001f));
            __m128 safe_len = _mm_or_ps(
                _mm_and_ps(valid_len, len),
                _mm_andnot_ps(valid_len, _mm_set1_ps(1.0f))
            );
            dir_x = _mm_div_ps(dir_x, safe_len);
            dir_y = _mm_div_ps(dir_y, safe_len);
        }

        // Store results back to memory - only for active entities
        float x_out[4], y_out[4], right_out[4], bottom_out[4], dir_x_out[4], dir_y_out[4];
        _mm_storeu_ps(x_out, new_x);
        _mm_storeu_ps(y_out, new_y);
        _mm_storeu_ps(right_out, right);
        _mm_storeu_ps(bottom_out, bottom);
        _mm_storeu_ps(dir_x_out, dir_x);
        _mm_storeu_ps(dir_y_out, dir_y);

        // Only update active entities
        chunk->x[i] = x_out[0];
        chunk->y[i] = y_out[0];
        chunk->right[i] = right_out[0];
        chunk->bottom[i] = bottom_out[0];
        data0->direction[0] = dir_x_out[0];
        data0->direction[1] = dir_y_out[0];

        chunk->x[i + 1] = x_out[1];
        chunk->y[i + 1] = y_out[1];
        chunk->right[i + 1] = right_out[1];
        chunk->bottom[i + 1] = bottom_out[1];
        data1->direction[0] = dir_x_out[1];
        data1->direction[1] = dir_y_out[1];

        chunk->x[i + 2] = x_out[2];
        chunk->y[i + 2] = y_out[2];
        chunk->right[i + 2] = right_out[2];
        chunk->bottom[i + 2] = bottom_out[2];
        data2->direction[0] = dir_x_out[2];
        data2->direction[1] = dir_y_out[2];

        chunk->x[i + 3] = x_out[3];
        chunk->y[i + 3] = y_out[3];
        chunk->right[i + 3] = right_out[3];
        chunk->bottom[i + 3] = bottom_out[3];
        data3->direction[0] = dir_x_out[3];
        data3->direction[1] = dir_y_out[3];
    }

#elif HAS_WASM_SIMD
    // WebAssembly SIMD implementation
    // Note: This would require implementation using WebAssembly SIMD intrinsics
    // For now, we'll use the scalar fallback, but this section could be expanded
    // to use WebAssembly SIMD in the future
    int i = 0;
#else
    // No SIMD available, start with scalar code directly
    int i = 0;
#endif


    // Process remaining entities using scalar code (either all entities if no SIMD, 
    // or just the remainder if SIMD was used)
    for (; i < count; i++) {
        GenericEntityData* data = (GenericEntityData*)((uint8_t*)chunk->type_data + i * sizeof(GenericEntityData));

        float new_x = chunk->x[i] + data->direction[0] * data->speed * delta_time;
        float new_y = chunk->y[i] + data->direction[1] * data->speed * delta_time;

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

        if (bounce_x) {
            new_x = SDL_clamp(new_x, 0, WORLD_WIDTH - chunk->width[i]);
        }

        if (bounce_y) {
            new_y = SDL_clamp(new_y, 0, WORLD_HEIGHT - chunk->height[i]);
        }

        chunk->x[i] = new_x;
        chunk->y[i] = new_y;
        chunk->right[i] = new_x + chunk->width[i];
        chunk->bottom[i] = new_y + chunk->height[i];

        if (data->behavior_flags & 0x1) {
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

        if (data->keys_pressed[0])
            move_y -= 1.0f; // W - up
        if (data->keys_pressed[1])
            move_x -= 1.0f; // A - left
        if (data->keys_pressed[2])
            move_y += 1.0f; // S - down
        if (data->keys_pressed[3])
            move_x += 1.0f; // D - right

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
                x, y, ENTITY_WIDTH, ENTITY_HEIGHT, texture_id, 1, entity_type_id);

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

// Network initialization function
void initializeNetwork(GameState* state) {
    // Initialize network with proper parameters
    SDL_Log("Attempting to initialize network...");

    // Initialize with any port (0) to allow the system to choose a free port
    state->isConnected = state->network.init("0.0.0.0", 0);

    if (!state->isConnected) {
#ifdef _WIN32
        SDL_Log("Network initialization failed: WSA error %d", WSAGetLastError());
#else
        SDL_Log("Network initialization failed: %s", strerror(errno));
#endif
        SDL_snprintf(state->errorMessage, sizeof(state->errorMessage),
            "Failed to connect to network. Server features unavailable.");
        return;
    }

    // Set server address - assuming server is running on localhost:7777
    state->serverAddr.setIPv4("127.0.0.1", 7777);

    SDL_Log("Network initialized successfully on port %d. Server at %s:%d",
        state->network.getPort(),
        state->serverAddr.getIPString().c_str(),
        state->serverAddr.getPort());
}

// Function to send login request
void sendLoginRequest(GameState* state) {
    if (!state->isConnected) {
        SDL_snprintf(state->errorMessage, sizeof(state->errorMessage),
            "Not connected to server. Try offline mode.");
        return;
    }

    state->isAuthenticating = true;
    SDL_snprintf(state->errorMessage, sizeof(state->errorMessage),
        "Logging in...");

    // For initial message when user has no ID yet, use a temporary ID
    // Server will assign real ID after authentication
    uint16_t tempUserId = 1;

    // Create a packet using ReliableOrderedUDP (unlike DefaultUDP, this adds header info)
    UDPPacket packet = state->network.preparePacket(tempUserId);
    if (!packet.bufferHandle) {
        SDL_snprintf(state->errorMessage, sizeof(state->errorMessage),
            "Failed to allocate network buffer");
        state->isAuthenticating = false;
        return;
    }

    // Get pointer to message area after the ReliableOrderedHeader
    LoginRequestMsg* msg = reinterpret_cast<LoginRequestMsg*>(
        packet.bufferHandle.data() + sizeof(ReliableOrderedHeader));

    // Set up message
    msg->header.type = MSG_LOGIN_REQUEST;
    msg->header.length = sizeof(LoginRequestMsg);

    SDL_strlcpy(msg->username, state->loginUsername, MAX_USERNAME_LENGTH);
    SDL_strlcpy(msg->password, state->loginPassword, MAX_PASSWORD_LENGTH);

    // Set packet data length (include both header and message)
    packet.dataLength = sizeof(ReliableOrderedHeader) + sizeof(LoginRequestMsg);

    // Send the packet
    if (state->network.sendPacket(std::move(packet)) <= 0) {
        SDL_snprintf(state->errorMessage, sizeof(state->errorMessage),
            "Failed to send login request");
        state->isAuthenticating = false;
        return;
    }

    SDL_Log("Sent login request for user: %s", state->loginUsername);
}

// Function to send register request
void sendRegisterRequest(GameState* state) {
    if (!state->isConnected) {
        SDL_snprintf(state->errorMessage, sizeof(state->errorMessage),
            "Not connected to server. Try offline mode.");
        return;
    }

    state->isAuthenticating = true;
    SDL_snprintf(state->errorMessage, sizeof(state->errorMessage),
        "Registering...");

    // For initial message when user has no ID yet, use a temporary ID
    uint16_t tempUserId = 1;

    // Create a packet using ReliableOrderedUDP
    UDPPacket packet = state->network.preparePacket(tempUserId);
    if (!packet.bufferHandle) {
        SDL_snprintf(state->errorMessage, sizeof(state->errorMessage),
            "Failed to allocate network buffer");
        state->isAuthenticating = false;
        return;
    }

    // Get pointer to message area after the ReliableOrderedHeader
    RegisterRequestMsg* msg = reinterpret_cast<RegisterRequestMsg*>(
        packet.bufferHandle.data() + sizeof(ReliableOrderedHeader));

    // Set up message
    msg->header.type = MSG_REGISTER_REQUEST;
    msg->header.length = sizeof(RegisterRequestMsg);

    SDL_strlcpy(msg->username, state->registerUsername, MAX_USERNAME_LENGTH);
    SDL_strlcpy(msg->password, state->registerPassword, MAX_PASSWORD_LENGTH);

    // Set packet data length (include both header and message)
    packet.dataLength = sizeof(ReliableOrderedHeader) + sizeof(RegisterRequestMsg);

    // Send the packet
    if (state->network.sendPacket(std::move(packet)) <= 0) {
        SDL_snprintf(state->errorMessage, sizeof(state->errorMessage),
            "Failed to send registration request");
        state->isAuthenticating = false;
        return;
    }

    SDL_Log("Sent registration request for user: %s", state->registerUsername);
}

// Function to send guest login request
void sendGuestLoginRequest(GameState* state) {
    if (!state->isConnected) {
        SDL_snprintf(state->errorMessage, sizeof(state->errorMessage),
            "Not connected to server. Try offline mode.");
        return;
    }

    state->isAuthenticating = true;
    SDL_snprintf(state->errorMessage, sizeof(state->errorMessage),
        "Logging in as guest...");

    // For initial message when user has no ID yet, use a temporary ID
    uint16_t tempUserId = 1;

    // Create a packet using ReliableOrderedUDP
    UDPPacket packet = state->network.preparePacket(tempUserId);
    if (!packet.bufferHandle) {
        SDL_snprintf(state->errorMessage, sizeof(state->errorMessage),
            "Failed to allocate network buffer");
        state->isAuthenticating = false;
        return;
    }

    // Get pointer to message area after the ReliableOrderedHeader
    MessageHeader* msg = reinterpret_cast<MessageHeader*>(
        packet.bufferHandle.data() + sizeof(ReliableOrderedHeader));

    // Set up message - guest login is just a header
    msg->type = MSG_GUEST_LOGIN_REQUEST;
    msg->length = sizeof(MessageHeader);

    // Set packet data length (include both ReliableOrderedHeader and message)
    packet.dataLength = sizeof(ReliableOrderedHeader) + sizeof(MessageHeader);

    // Send the packet
    if (state->network.sendPacket(std::move(packet)) <= 0) {
        SDL_snprintf(state->errorMessage, sizeof(state->errorMessage),
            "Failed to send guest login request");
        state->isAuthenticating = false;
        return;
    }

    SDL_Log("Sent guest login request");
}

// Function to handle authentication messages
void handleAuthenticationMessages(GameState* state) {
    // Receive and process incoming packets
    UDPPacket packet = state->network.receive();
    while (packet.bufferHandle) {
        // First let ReliableOrderedUDP process the packet to update its internal state
        bool processedByNetwork = state->network.processPacket(packet);

        // Then check if it contains a message we're interested in
        if (packet.dataLength >= sizeof(ReliableOrderedHeader) + sizeof(MessageHeader)) {
            // Get the message header, skipping the ReliableOrderedHeader
            MessageHeader* header = reinterpret_cast<MessageHeader*>(
                packet.bufferHandle.data() + sizeof(ReliableOrderedHeader));

            if (header->type == MSG_LOGIN_RESPONSE &&
                packet.dataLength >= sizeof(ReliableOrderedHeader) + sizeof(LoginResponseMsg)) {
                // Cast to login response message
                LoginResponseMsg* response = reinterpret_cast<LoginResponseMsg*>(
                    packet.bufferHandle.data() + sizeof(ReliableOrderedHeader));

                if (response->status == STATUS_SUCCESS) {
                    // Authentication successful
                    state->userId = response->userId;
                    state->isAuthenticating = false;

                    // Store username for welcome message based on which form we were using
                    if (state->showLoginForm) {
                        SDL_strlcpy(state->loggedInUsername, state->loginUsername, MAX_USERNAME_LENGTH);
                        state->isGuest = false;
                    }
                    else if (state->showRegisterForm) {
                        SDL_strlcpy(state->loggedInUsername, state->registerUsername, MAX_USERNAME_LENGTH);
                        state->isGuest = false;
                    }
                    else {
                        SDL_snprintf(state->loggedInUsername, MAX_USERNAME_LENGTH, "Guest_%d", response->userId);
                        state->isGuest = true;
                    }

                    // Add the user ID to the network client so we can send messages with it
                    std::array<uint8_t, 32> dummyKey = {};  // Client doesn't use encryption
                    state->network.addUser(state->userId, dummyKey);

                    // Transition to title screen
                    state->current_state = GAME_STATE_TITLE;
                    state->lastHeartbeatTime = SDL_GetTicks();

                    SDL_Log("Authentication successful. User ID: %d", state->userId);
                }
                else {
                    // Authentication failed
                    SDL_strlcpy(state->errorMessage, response->errorMsg, sizeof(state->errorMessage));
                    state->isAuthenticating = false;

                    SDL_Log("Authentication failed: %s", response->errorMsg);
                }
            }
        }

        // Get next packet
        packet = state->network.receive();
    }

    // Execute network tasks (retransmissions, etc.)
    state->network.execute();
}

// Function to send heartbeat to keep session alive
void sendHeartbeatIfNeeded(GameState* state) {
    // Only send heartbeat if we're authenticated
    if (!state->isConnected || state->userId == 0) {
        return;
    }

    // Send heartbeat every 30 seconds
    Uint64 currentTime = SDL_GetTicks();
    if (currentTime - state->lastHeartbeatTime >= 30000) {
        // Create heartbeat message using ReliableOrderedUDP
        UDPPacket packet = state->network.preparePacket(state->userId);
        if (!packet.bufferHandle) {
            return;
        }

        // Get pointer to message area after the ReliableOrderedHeader
        HeartbeatMsg* msg = reinterpret_cast<HeartbeatMsg*>(
            packet.bufferHandle.data() + sizeof(ReliableOrderedHeader));

        // Set up message
        msg->header.type = MSG_HEARTBEAT;
        msg->header.length = sizeof(HeartbeatMsg);
        msg->userId = state->userId;

        // Set packet data length (include both ReliableOrderedHeader and message)
        packet.dataLength = sizeof(ReliableOrderedHeader) + sizeof(HeartbeatMsg);

        // Send the packet
        state->network.sendPacket(std::move(packet));
        state->lastHeartbeatTime = currentTime;

        SDL_Log("Sent heartbeat for user ID: %d", state->userId);
    }

    // Periodically process the network to handle retransmissions
    state->network.execute();
}

// Function to send disconnect message
void sendDisconnectMessage(GameState* state) {
    if (!state->isConnected || state->userId == 0) {
        return;
    }

    // Create disconnect message using ReliableOrderedUDP
    UDPPacket packet = state->network.preparePacket(state->userId);
    if (!packet.bufferHandle) {
        return;
    }

    // Get pointer to message area after the ReliableOrderedHeader
    HeartbeatMsg* msg = reinterpret_cast<HeartbeatMsg*>(
        packet.bufferHandle.data() + sizeof(ReliableOrderedHeader));

    // Set up message (using same format as heartbeat)
    msg->header.type = MSG_DISCONNECT;
    msg->header.length = sizeof(HeartbeatMsg);
    msg->userId = state->userId;

    // Set packet data length (include both ReliableOrderedHeader and message)
    packet.dataLength = sizeof(ReliableOrderedHeader) + sizeof(HeartbeatMsg);

    // Send the packet
    state->network.sendPacket(std::move(packet));

    SDL_Log("Sent disconnect for user ID: %d", state->userId);
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
    state->current_state = GAME_STATE_AUTH;  // Start with authentication
    state->max_health = 100;
    state->player_health = state->max_health;
    state->show_controls = false;
    state->show_debug_info = false;
    state->title_animation = 0.0f;
    state->score_animation = 0.0f;
    state->health_animation = (float)state->max_health;
    state->damage_flash = 0.0f;

    // Initialize networking and authentication fields
    state->isConnected = false;
    state->userId = 0;
    state->errorMessage[0] = '\0';
    state->isAuthenticating = false;
    state->showLoginForm = true;
    state->showRegisterForm = false;
    state->loginUsername[0] = '\0';
    state->loginPassword[0] = '\0';
    state->registerUsername[0] = '\0';
    state->registerPassword[0] = '\0';
    state->loggedInUsername[0] = '\0';
    state->isGuest = false;
    state->lastHeartbeatTime = 0;

    // Initialize network
    initializeNetwork(state);

    // Initialize game engine with spatial grid cell size parameter
    state->engine = engine_create(WINDOW_WIDTH, WINDOW_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT, GRID_CELL_SIZE);
    if (!state->engine) {
        SDL_Log("Failed to create engine");
        SDL_free(state);
        SDL_Quit();
        return SDL_APP_FAILURE;
    }

    SDL_Log("engine created successfully");
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

    // Register player entity type with the engine and store the ID
    state->player_type_id = engine_register_entity_type(state->engine, player_entity_update, sizeof(PlayerData));
    SDL_Log("Registered player entity type with ID: %d", state->player_type_id);

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
    SDL_Surface* player_surface2 = create_colored_surface(PLAYER_WIDTH, PLAYER_HEIGHT, 0, 100, 255);
    SDL_Surface* player_surface3 = create_colored_surface(PLAYER_WIDTH, PLAYER_HEIGHT, 100, 100, 255);

    // Add player textures to engine
    int player_texture1 = engine_register_texture(state->engine, player_surface1, 0, 0);
    int player_texture2 = engine_register_texture(state->engine, player_surface2, 0, 0);
    int player_texture3 = engine_register_texture(state->engine, player_surface3, 0, 0);

    // Clean up player surfaces
    SDL_DestroySurface(player_surface1);
    SDL_DestroySurface(player_surface2);
    SDL_DestroySurface(player_surface3);

    // Create player entity using the stored player type ID
    state->player_entity = engine_add_entity_with_type(state->engine,
        state->player_type_id,  // Use the explicitly stored player type ID
        WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2,
        PLAYER_WIDTH, PLAYER_HEIGHT,
        player_texture1, 2, 100); // Layer 2 (above other entities)

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
        // Register this entity type with the engine
        state->entity_type_ids[i] = engine_register_entity_type(state->engine, generic_entity_update, sizeof(GenericEntityData));
        SDL_Log("Registered entity type %d with ID: %d", i, state->entity_type_ids[i]);

        // Generate unique color for this entity type - ensure enough contrast
        Uint8 r = 50 + (i * 25) % 205;  // Adjusted to avoid too dark colors
        Uint8 g = 50 + (i * 40) % 205;
        Uint8 b = 50 + (i * 60) % 205;

        // Create surface with this color
        SDL_Surface* entity_surface = create_colored_surface(ENTITY_WIDTH, ENTITY_HEIGHT, r, g, b);

        int atlas_columns = 16;  // Adjust based on your atlas size
        int x_pos = (i % atlas_columns) * ENTITY_WIDTH;
        int y_pos = (i / atlas_columns) * ENTITY_HEIGHT;

        // Add the texture to the engine with unique position
        int texture_id = engine_register_texture(state->engine, entity_surface, x_pos, y_pos);
        state->entity_texture_ids[i] = texture_id;
        SDL_Log("Created texture ID %d for entity type %d", texture_id, i);

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

    // Check for authentication messages and send heartbeat if needed
    if (state->isConnected) {
        if (state->current_state == GAME_STATE_AUTH) {
            handleAuthenticationMessages(state);
        }
        else if (state->userId > 0) {
            sendHeartbeatIfNeeded(state);
        }
    }

    // Handle different game states
    if (state->current_state == GAME_STATE_AUTH) {
        // Authentication UI
        // Draw a gradient background similar to the title screen
        ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(WINDOW_WIDTH, WINDOW_HEIGHT));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("AuthBackground", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);

        // Draw background - a gradient from dark blue to black
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 window_pos = ImGui::GetWindowPos();
        ImVec2 window_size = ImGui::GetWindowSize();

        draw_list->AddRectFilledMultiColor(
            window_pos,
            ImVec2(window_pos.x + window_size.x, window_pos.y + window_size.y),
            ImColor(0.05f, 0.05f, 0.1f, 1.0f),   // Top-left (dark blue)
            ImColor(0.1f, 0.1f, 0.2f, 1.0f),     // Top-right (slightly lighter blue)
            ImColor(0.02f, 0.02f, 0.05f, 1.0f),  // Bottom-right (almost black)
            ImColor(0.0f, 0.0f, 0.0f, 1.0f)      // Bottom-left (black)
        );

        // Add some particles/stars to the background
        for (int i = 0; i < 100; i++) {
            float x = window_pos.x + (float)(sin(SDL_GetTicks() * 0.001f + i * 0.5f) + 1.0f) * window_size.x * 0.5f;
            float y = window_pos.y + (float)(cos(SDL_GetTicks() * 0.001f + i * 0.3f) + 1.0f) * window_size.y * 0.5f;
            float brightness = (float)(sin(SDL_GetTicks() * 0.002f + i) * 0.5f + 0.5f);
            draw_list->AddCircleFilled(ImVec2(x, y), 1.0f + brightness * 2.0f,
                ImColor(0.5f + brightness * 0.5f, 0.5f + brightness * 0.5f, 0.8f + brightness * 0.2f, 0.7f));
        }

        ImGui::PopStyleVar();
        ImGui::End();

        // Authentication panel
        float auth_width = 400.0f;
        float auth_height = 450.0f;
        ImGui::SetNextWindowPos(ImVec2(WINDOW_WIDTH * 0.5f, WINDOW_HEIGHT * 0.5f),
            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(auth_width, auth_height));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 12.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 20));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.07f, 0.07f, 0.15f, 0.94f));

        ImGui::Begin("Authentication", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse);

        // Game title with stylized text
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);

        // Title text with shadow effect
        ImVec2 title_pos = ImGui::GetCursorPos();
        ImGui::SetCursorPos(ImVec2(title_pos.x + 3, title_pos.y + 3));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 0.7f));
        ImGui::SetWindowFontScale(3.0f);
        ImGui::Text("TDTower");
        ImGui::PopStyleColor();

        ImGui::SetCursorPos(title_pos);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.7f, 1.0f, 1.0f));
        ImGui::SetWindowFontScale(3.0f);
        ImGui::Text("TDTower");
        ImGui::PopStyleColor();
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopFont();

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Error message display
        if (strlen(state->errorMessage) > 0) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
            ImGui::TextWrapped("%s", state->errorMessage);
            ImGui::PopStyleColor();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
        }

        // Toggle between login and register forms
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
        if (ImGui::Button("Login", ImVec2(auth_width * 0.3f, 30))) {
            state->showLoginForm = true;
            state->showRegisterForm = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Register", ImVec2(auth_width * 0.3f, 30))) {
            state->showLoginForm = false;
            state->showRegisterForm = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Guest", ImVec2(auth_width * 0.3f, 30))) {
            // Skip form, just send guest login request
            sendGuestLoginRequest(state);
        }
        ImGui::PopStyleVar();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (state->showLoginForm) {
            // Login form
            ImGui::Text("Login to Your Account");
            ImGui::Spacing();

            ImGui::PushItemWidth(auth_width * 0.8f);

            // Username input
            ImGui::Text("Username:");
            ImGui::InputText("##LoginUsername", state->loginUsername, MAX_USERNAME_LENGTH);

            // Password input
            ImGui::Text("Password:");
            ImGui::InputText("##LoginPassword", state->loginPassword, MAX_PASSWORD_LENGTH,
                ImGuiInputTextFlags_Password);

            ImGui::PopItemWidth();
            ImGui::Spacing();

            // Login button
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 8));
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.7f, 0.8f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.5f, 0.8f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.6f, 0.9f, 1.0f));

            if (ImGui::Button("Login", ImVec2(auth_width * 0.8f, 40)) && !state->isAuthenticating) {
                if (strlen(state->loginUsername) == 0 || strlen(state->loginPassword) == 0) {
                    SDL_strlcpy(state->errorMessage, "Please enter both username and password",
                        sizeof(state->errorMessage));
                }
                else {
                    sendLoginRequest(state);
                }
            }

            ImGui::PopStyleColor(3);
            ImGui::PopStyleVar(2);
        }
        else if (state->showRegisterForm) {
            // Register form
            ImGui::Text("Create a New Account");
            ImGui::Spacing();

            ImGui::PushItemWidth(auth_width * 0.8f);

            // Username input
            ImGui::Text("Username:");
            ImGui::InputText("##RegisterUsername", state->registerUsername, MAX_USERNAME_LENGTH);

            // Password input
            ImGui::Text("Password:");
            ImGui::InputText("##RegisterPassword", state->registerPassword, MAX_PASSWORD_LENGTH,
                ImGuiInputTextFlags_Password);

            ImGui::PopItemWidth();
            ImGui::Spacing();

            // Register button
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 8));
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.4f, 0.8f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.5f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.8f, 0.6f, 1.0f));

            if (ImGui::Button("Create Account", ImVec2(auth_width * 0.8f, 40)) && !state->isAuthenticating) {
                if (strlen(state->registerUsername) == 0 || strlen(state->registerPassword) == 0) {
                    SDL_strlcpy(state->errorMessage, "Please enter both username and password",
                        sizeof(state->errorMessage));
                }
                else if (strlen(state->registerPassword) < 4) {
                    SDL_strlcpy(state->errorMessage, "Password must be at least 4 characters",
                        sizeof(state->errorMessage));
                }
                else {
                    sendRegisterRequest(state);
                }
            }

            ImGui::PopStyleColor(3);
            ImGui::PopStyleVar(2);
        }

        // Add offline mode option when network connection fails
        if (!state->isConnected) {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
            ImGui::TextWrapped("Server connection failed. You can still play in offline mode.");
            ImGui::PopStyleColor();

            ImGui::Spacing();
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 8));
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.5f, 0.1f, 0.8f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.6f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.9f, 0.7f, 0.3f, 1.0f));

            if (ImGui::Button("Play Offline", ImVec2(auth_width * 0.8f, 40))) {
                // Set up offline session
                SDL_strlcpy(state->loggedInUsername, "Offline Player", MAX_USERNAME_LENGTH);
                state->isGuest = true;
                state->userId = 0; // No user ID for offline mode
                state->current_state = GAME_STATE_TITLE;
            }

            ImGui::PopStyleColor(3);
            ImGui::PopStyleVar(2);
        }

        // Note about server
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
        if (state->isConnected) {
            ImGui::Text("Connected to server at %s:%d",
                state->serverAddr.getIPString().c_str(),
                state->serverAddr.getPort());
        }
        else {
            ImGui::TextWrapped("Server connection failed. Some features may be unavailable.");
        }
        ImGui::PopStyleColor();

        ImGui::End();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
    }
    else if (state->current_state == GAME_STATE_TITLE) {
        // Main Menu UI
        // Make the menu fill the entire window
        ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(WINDOW_WIDTH, WINDOW_HEIGHT));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("MainMenuBackground", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);

        // Draw background - a gradient from dark blue to black
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 window_pos = ImGui::GetWindowPos();
        ImVec2 window_size = ImGui::GetWindowSize();

        draw_list->AddRectFilledMultiColor(
            window_pos,
            ImVec2(window_pos.x + window_size.x, window_pos.y + window_size.y),
            ImColor(0.05f, 0.05f, 0.1f, 1.0f),   // Top-left (dark blue)
            ImColor(0.1f, 0.1f, 0.2f, 1.0f),     // Top-right (slightly lighter blue)
            ImColor(0.02f, 0.02f, 0.05f, 1.0f),  // Bottom-right (almost black)
            ImColor(0.0f, 0.0f, 0.0f, 1.0f)      // Bottom-left (black)
        );

        // Add some particles/stars to the background
        for (int i = 0; i < 100; i++) {
            float x = window_pos.x + (float)(sin(current_time * 0.001f + i * 0.5f) + 1.0f) * window_size.x * 0.5f;
            float y = window_pos.y + (float)(cos(current_time * 0.001f + i * 0.3f) + 1.0f) * window_size.y * 0.5f;
            float brightness = (float)(sin(current_time * 0.002f + i) * 0.5f + 0.5f);
            draw_list->AddCircleFilled(ImVec2(x, y), 1.0f + brightness * 2.0f,
                ImColor(0.5f + brightness * 0.5f, 0.5f + brightness * 0.5f, 0.8f + brightness * 0.2f, 0.7f));
        }

        ImGui::PopStyleVar();
        ImGui::End();

        // Menu window
        float menu_width = 400.0f;
        float menu_height = 450.0f;
        ImGui::SetNextWindowPos(ImVec2(WINDOW_WIDTH * 0.5f, WINDOW_HEIGHT * 0.5f),
            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(menu_width, menu_height));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 12.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 20));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.07f, 0.07f, 0.15f, 0.94f));

        ImGui::Begin("MainMenu", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);

        // Game title with stylized text
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]); // Use default font - in a real game you'd load a custom font

        // Title text with shadow effect
        ImVec2 title_pos = ImGui::GetCursorPos();
        ImGui::SetCursorPos(ImVec2(title_pos.x + 3, title_pos.y + 3));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 0.7f));
        ImGui::SetWindowFontScale(3.0f);
        ImGui::Text("TDTower");
        ImGui::PopStyleColor();

        ImGui::SetCursorPos(title_pos);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.7f, 1.0f, 1.0f));
        ImGui::SetWindowFontScale(3.0f);
        ImGui::Text("TDTower");
        ImGui::PopStyleColor();
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopFont();

        // Welcome message for logged-in user
        if (strlen(state->loggedInUsername) > 0) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.8f, 1.0f, 1.0f));
            if (state->isGuest) {
                ImGui::Text("Welcome, %s", state->loggedInUsername);
            }
            else {
                ImGui::Text("Welcome back, %s", state->loggedInUsername);
            }
            ImGui::PopStyleColor();
        }

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Spacing();

        // Menu buttons
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 10));
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.25f, 0.6f, 0.8f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.35f, 0.7f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.45f, 0.8f, 1.0f));

        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
        ImGui::SetWindowFontScale(1.5f);

        if (ImGui::Button("Start Game", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
            state->current_state = GAME_STATE_PLAYING;
            state->player_health = state->max_health;
            state->score = 0;
            SDL_Log("Game started!");
        }

        ImGui::Spacing();
        ImGui::Spacing();

        if (ImGui::Button("Ranked", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
            // Ranked mode would be implemented here
            SDL_Log("Ranked mode selected - not implemented yet");
        }

        ImGui::Spacing();
        ImGui::Spacing();

        if (ImGui::Button("Shop", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
            // Shop functionality would be implemented here
            SDL_Log("Shop selected - not implemented yet");
        }

        ImGui::Spacing();
        ImGui::Spacing();

        if (ImGui::Button("Settings", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
            // Settings functionality would be implemented here
            SDL_Log("Settings selected - not implemented yet");
        }

        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopFont();

        ImGui::PopStyleColor(3);
        ImGui::PopStyleVar(2);

        // Version info at bottom
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 30);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
        ImGui::Text("Version 1.0.0");
        ImGui::PopStyleColor();

        ImGui::End();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
    }
    else if (state->current_state == GAME_STATE_PLAYING) {
        // Update game logic
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

        // Display HUD with health, score, etc.
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::SetNextWindowSize(ImVec2(300, 100));
        ImGui::Begin("HUD", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBringToFrontOnFocus);

        ImGui::Text("Health: %d / %d", (int)state->player_health, state->max_health);
        ImGui::ProgressBar(state->player_health / (float)state->max_health, ImVec2(-1, 0), "");
        ImGui::Text("Score: %d", state->score);
        ImGui::End();
    }
    else if (state->current_state == GAME_STATE_PAUSED) {
        // Pause menu
        ImGui::SetNextWindowPos(ImVec2(WINDOW_WIDTH * 0.5f, WINDOW_HEIGHT * 0.5f),
            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(300, 200));
        ImGui::Begin("PauseMenu", nullptr,
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        ImGui::Text("Game Paused");
        ImGui::Spacing();

        if (ImGui::Button("Resume", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
            state->current_state = GAME_STATE_PLAYING;
        }

        ImGui::Spacing();

        if (ImGui::Button("Main Menu", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
            state->current_state = GAME_STATE_TITLE;
        }

        ImGui::End();
    }
    else if (state->current_state == GAME_STATE_GAME_OVER) {
        // Game over screen
        ImGui::SetNextWindowPos(ImVec2(WINDOW_WIDTH * 0.5f, WINDOW_HEIGHT * 0.5f),
            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(300, 200));
        ImGui::Begin("GameOver", nullptr,
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        ImGui::Text("Game Over");
        ImGui::Text("Final Score: %d", state->score);
        ImGui::Spacing();

        if (ImGui::Button("Restart", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
            state->current_state = GAME_STATE_PLAYING;
            state->player_health = state->max_health;
            state->score = 0;
            // Reinitialize entities
            init_entities(state);
        }

        ImGui::Spacing();

        if (ImGui::Button("Main Menu", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
            state->current_state = GAME_STATE_TITLE;
        }

        ImGui::End();
    }

    // Render updated scene (only if not in title screen or auth screen)
    if (state->current_state != GAME_STATE_TITLE && state->current_state != GAME_STATE_AUTH) {
        engine_render(engine);
    }

    // Render ImGui
    ImGui::Render();
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), engine->renderer);

    // Present the rendered frame
    SDL_RenderPresent(engine->renderer);

    // Update last frame time for proper delta time calculation
    engine->last_frame_time = current_time;

    return SDL_APP_CONTINUE;
}

#include <unordered_map>

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event) {
    GameState* state = (GameState*)appstate;
    Engine* engine = state->engine;

    // Process ImGui events
    ImGui_ImplSDL3_ProcessEvent(event);

    // If ImGui is capturing mouse/keyboard, don't process game events
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse || io.WantCaptureKeyboard)
        return SDL_APP_CONTINUE;

    switch (event->type) {
    case SDL_EVENT_KEY_DOWN:
    {
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

            // Reinitialize entities
            init_entities(state);
        }

        // Add multiple entities when 'A' is pressed
        if (event->key.scancode == SDL_SCANCODE_A) {
            int added_count = 0;

            // Get player position to ensure new entities are placed away from player
            int player_chunk_idx, player_local_idx;
            engine->entities.getChunkIndices(state->player_entity, &player_chunk_idx, &player_local_idx);
            float player_x = engine->entities.chunks[player_chunk_idx]->x[player_local_idx];
            float player_y = engine->entities.chunks[player_chunk_idx]->y[player_local_idx];

            // Add ADD_ENTITIES_AT_ONCE entities
            for (int n = 0; n < ADD_ENTITIES_AT_ONCE; n++) {
                // Randomly select entity type from available types (excluding player type)
                int type_idx = -1;
                do {
                    type_idx = rand() % state->num_entity_types;
                } while (state->entity_type_ids[type_idx] == state->player_type_id);

                int entity_type_id = state->entity_type_ids[type_idx];
                int texture_id = state->entity_texture_ids[type_idx];

                // Random position within world bounds, away from player
                float x, y;
                do {
                    x = (float)(rand() % (WORLD_WIDTH - ENTITY_WIDTH));
                    y = (float)(rand() % (WORLD_HEIGHT - ENTITY_HEIGHT));
                } while (SDL_fabsf(x - player_x) < 200 && SDL_fabsf(y - player_y) < 200);

                // Create entity with specific entity type
                int entity_id = engine_add_entity_with_type(engine, entity_type_id,
                    x, y, ENTITY_WIDTH, ENTITY_HEIGHT, texture_id, 1, entity_type_id);

                if (entity_id >= 0) {
                    // Initialize entity data
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

                    added_count++;
                    SDL_Log("Added entity of type %d (index %d)", entity_type_id, entity_id);
                }
            }

            SDL_Log("Total entities added: %d, total entities: %d",
                added_count, engine->entities.total_count);
        }

        // Remove multiple entities when 'B' is pressed
        if (event->key.scancode == SDL_SCANCODE_B) {
            int removed_count = 0;

            // First, build a map of entity types and their counts
            std::unordered_map<int, std::vector<int>> entityTypeMap;

            // Scan all chunks to find entities of different types
            for (int chunk_idx = 0; chunk_idx < engine->entities.chunk_count; chunk_idx++) {
                EntityChunk* chunk = engine->entities.chunks[chunk_idx];

                // Skip empty chunks
                if (chunk->count == 0)
                    continue;

                // Skip player entities
                if (chunk->type_id == state->player_type_id)
                    continue;

                // For each entity in the chunk, store its global index
                for (int local_idx = 0; local_idx < chunk->count; local_idx++) {
                    int entity_idx = chunk_idx * ENTITY_CHUNK_SIZE + local_idx;
                    entityTypeMap[chunk->type_id].push_back(entity_idx);
                }
            }

            // Try to remove ENTITIES_TO_REMOVE_AT_ONCE entities
            for (int n = 0; n < ENTITIES_TO_REMOVE_AT_ONCE; n++) {
                // If no valid entity types remain, stop
                if (entityTypeMap.empty())
                    break;

                // Get a list of available type IDs
                std::vector<int> available_types;
                for (const auto& pair : entityTypeMap) {
                    if (!pair.second.empty()) {
                        available_types.push_back(pair.first);
                    }
                }

                // Randomly select an entity type
                int random_type_idx = rand() % available_types.size();
                int selected_type_id = available_types[random_type_idx];

                // Get a random entity of this type
                auto& entity_indices = entityTypeMap[selected_type_id];
                int random_entity_idx = rand() % entity_indices.size();
                int entity_to_remove = entity_indices[random_entity_idx];

                // Remove the entity from our tracking
                entity_indices.erase(entity_indices.begin() + random_entity_idx);

                // If this was the last entity of this type, remove the type
                if (entity_indices.empty()) {
                    entityTypeMap.erase(selected_type_id);
                }

                // Mark entity for removal in the engine
                engine_remove_entity(engine, entity_to_remove);

                SDL_Log("Removed entity %d of type %d", entity_to_remove, selected_type_id);
                removed_count++;
            }

            if (removed_count > 0) {
                SDL_Log("Total entities removed: %d, total remaining: %d",
                    removed_count, engine->entities.total_count - engine->pending_removals.size());
            }
            else {
                SDL_Log("No valid entities to remove");
            }
        }

        // Update key state in player data (only when playing)
        PlayerData* player_data = (PlayerData*)engine_get_entity_type_data(state->engine, state->player_entity);
        if (player_data) {
            if (event->key.scancode == SDL_SCANCODE_W) player_data->keys_pressed[0] = SDL_SCANCODE_W;
            if (event->key.scancode == SDL_SCANCODE_A) player_data->keys_pressed[1] = SDL_SCANCODE_A;
            if (event->key.scancode == SDL_SCANCODE_S) player_data->keys_pressed[2] = SDL_SCANCODE_S;
            if (event->key.scancode == SDL_SCANCODE_D) player_data->keys_pressed[3] = SDL_SCANCODE_D;
        }
        break;
    }
    case SDL_EVENT_KEY_UP:
    {
        // Update key state in player data
        PlayerData* player_data = (PlayerData*)engine_get_entity_type_data(state->engine, state->player_entity);
        if (player_data) {
            if (event->key.scancode == SDL_SCANCODE_W) player_data->keys_pressed[0] = SDL_SCANCODE_UNKNOWN;
            if (event->key.scancode == SDL_SCANCODE_A) player_data->keys_pressed[1] = SDL_SCANCODE_UNKNOWN;
            if (event->key.scancode == SDL_SCANCODE_S) player_data->keys_pressed[2] = SDL_SCANCODE_UNKNOWN;
            if (event->key.scancode == SDL_SCANCODE_D) player_data->keys_pressed[3] = SDL_SCANCODE_UNKNOWN;
        }
        break;
    }
    case SDL_EVENT_QUIT:
        // Send disconnect message if connected
        sendDisconnectMessage(state);
        return SDL_APP_FAILURE;
    }

    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult result) {
    GameState* state = (GameState*)appstate;

    if (state) {
        // Send disconnect message
        sendDisconnectMessage(state);

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

        // Shutdown network
        state->network.shutdown();

        SDL_free(state);
    }

    SDL_Quit();
}