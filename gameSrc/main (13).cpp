#define SDL_MAIN_USE_CALLBACKS 1

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include "ATMEngine.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <future>

// --- Constants ---
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1020
#define NUM_OBSTACLES 1
#define NUM_ITEMS 0
#define PLAYER_SIZE toGameUnit(32)
#define OBSTACLE_SIZE toGameUnit(32)  
#define ITEM_SIZE toGameUnit(24)
#define PLAYER_SPEED toGameUnit(30)
#define PLAYER_MAX_HEALTH 100.0f
#define OBSTACLE_DAMAGE 10.0f

// --- Game-specific entity types ---
enum GameEntityTypes {
    ENTITY_TYPE_PLAYER = 0,
    ENTITY_TYPE_OBSTACLE,
    ENTITY_TYPE_ITEM,
    ENTITY_TYPE_COUNT
};

// --- Player Entity Container ---
class PlayerEntityContainer : public RenderableEntityContainer {
public:
    int32_t* speeds;  // Changed from uint32_t to int32_t
    uint32_t* health;
    bool* isMoving;

    PlayerEntityContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
        : RenderableEntityContainer(typeId, defaultLayer, initialCapacity)
    {
        speeds = new int32_t[capacity];  // Changed from uint32_t to int32_t
        health = new uint32_t[capacity];
        isMoving = new bool[capacity];
        std::fill(speeds, speeds + capacity, 0);  // Changed to 0
        std::fill(health, health + capacity, 0.0f);
        std::fill(isMoving, isMoving + capacity, false);
    }

    ~PlayerEntityContainer() override {
        delete[] speeds;
        delete[] health;
        delete[] isMoving;
    }

    uint32_t createEntity() override {
        uint32_t index = RenderableEntityContainer::createEntity();
        if (index == INVALID_ID) return INVALID_ID;

        speeds[index] = PLAYER_SPEED;
        health[index] = PLAYER_MAX_HEALTH;
        isMoving[index] = false;
        flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
        z_indices[index] = 10;
        return index;
    }

    uint32_t createEntity(int32_t x, int32_t y, int width, int height, int texture_id) {  // Changed from float to int32_t
        uint32_t index = createEntity();
        if (index == INVALID_ID) return INVALID_ID;

        // Store position directly as int32_t
        x_positions[index] = x;
        y_positions[index] = y;
        speeds[index] = PLAYER_SPEED;

        // Store dimensions
        widths[index] = width;
        heights[index] = height;

        texture_ids[index] = texture_id;

        return index;
    }

    void removeEntity(size_t index) override {
        if (index >= count) return;
        size_t last = count - 1;
        if (index < last) {
            speeds[index] = speeds[last];
            health[index] = health[last];
            isMoving[index] = isMoving[last];
        }
        RenderableEntityContainer::removeEntity(index);
    }

    void update(uint32_t delta_time) override {
        for (int i = 0; i < count; ++i) {
            flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
        }
    }

protected:
    void resizeArrays(int newCapacity) override {
        if (newCapacity <= capacity) return;

        int32_t* newSpeeds = new int32_t[newCapacity];  // Changed from uint32_t to int32_t
        uint32_t* newHealth = new uint32_t[newCapacity];
        bool* newIsMoving = new bool[newCapacity];

        if (count > 0) {
            std::copy(speeds, speeds + count, newSpeeds);
            std::copy(health, health + count, newHealth);
            std::copy(isMoving, isMoving + count, newIsMoving);
        }
        std::fill(newSpeeds + count, newSpeeds + newCapacity, 0);  // Changed to 0
        std::fill(newHealth + count, newHealth + newCapacity, 0.0f);
        std::fill(newIsMoving + count, newIsMoving + newCapacity, false);

        delete[] speeds;
        delete[] health;
        delete[] isMoving;

        speeds = newSpeeds;
        health = newHealth;
        isMoving = newIsMoving;

        RenderableEntityContainer::resizeArrays(newCapacity);
    }
};

// --- Item Entity Container ---
class ItemEntityContainer : public RenderableEntityContainer {
public:
    int* item_types;
    int* values;

    ItemEntityContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
        : RenderableEntityContainer(typeId, defaultLayer, initialCapacity) {
        item_types = new int[capacity];
        values = new int[capacity];
        std::fill(item_types, item_types + capacity, 0);
        std::fill(values, values + capacity, 0);
    }

    ~ItemEntityContainer() override {
        delete[] item_types;
        delete[] values;
    }

    uint32_t createEntity() override {
        uint32_t index = RenderableEntityContainer::createEntity();
        if (index == INVALID_ID) return INVALID_ID;
        item_types[index] = 0;
        values[index] = 10;
        flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
        z_indices[index] = 1;
        return index;
    }

    uint32_t createEntity(int32_t x, int32_t y, int width, int height, int texture_id, int value, int type = 0) {  // Changed from uint32_t to int32_t
        uint32_t index = RenderableEntityContainer::createEntity();
        if (index == INVALID_ID) return INVALID_ID;

        // Store position directly as int32_t
        x_positions[index] = x;
        y_positions[index] = y;

        // Store dimensions
        widths[index] = width;
        heights[index] = height;

        texture_ids[index] = texture_id;
        values[index] = value;
        item_types[index] = type;
        flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
        z_indices[index] = 1;

        return index;
    }

    void removeEntity(size_t index) override {
        if (index >= count) return;
        size_t last = count - 1;
        if (index < last) {
            item_types[index] = item_types[last];
            values[index] = values[last];
        }
        RenderableEntityContainer::removeEntity(index);
    }

    void update(uint32_t delta_time) override {
        for (int i = 0; i < count; ++i) {
            flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
        }
    }

protected:
    void resizeArrays(int newCapacity) override {
        if (newCapacity <= capacity) return;
        int* newItemTypes = new int[newCapacity];
        int* newValues = new int[newCapacity];
        if (count > 0) {
            std::copy(item_types, item_types + count, newItemTypes);
            std::copy(values, values + count, newValues);
        }
        std::fill(newItemTypes + count, newItemTypes + newCapacity, 0);
        std::fill(newValues + count, newValues + newCapacity, 0);
        delete[] item_types;
        delete[] values;
        item_types = newItemTypes;
        values = newValues;
        RenderableEntityContainer::resizeArrays(newCapacity);
    }
};

// --- Obstacle Entity Container ---
class ObstacleEntityContainer : public RenderableEntityContainer {
public:
    int* obstacle_types;
    bool* damaging;
    int32_t* speeds;  // Changed from uint32_t to int32_t
    float* dir_x;
    float* dir_y;
    Engine* engine;

    ObstacleEntityContainer(Engine* engine, int typeId, uint8_t defaultLayer, int initialCapacity)
        : RenderableEntityContainer(typeId, defaultLayer, initialCapacity), engine(engine)
    {
        obstacle_types = new int[capacity];
        damaging = new bool[capacity];
        speeds = new int32_t[capacity];  // Changed from uint32_t to int32_t
        dir_x = new float[capacity];
        dir_y = new float[capacity];
        std::fill(obstacle_types, obstacle_types + capacity, 0);
        std::fill(damaging, damaging + capacity, false);
        std::fill(speeds, speeds + capacity, 0);  // Changed to 0
        std::fill(dir_x, dir_x + capacity, 0.0f);
        std::fill(dir_y, dir_y + capacity, 0.0f);
    }

    ~ObstacleEntityContainer() override {
        delete[] obstacle_types;
        delete[] damaging;
        delete[] speeds;
        delete[] dir_x;
        delete[] dir_y;
    }

    uint32_t createEntity() override {
        uint32_t index = RenderableEntityContainer::createEntity();
        if (index == INVALID_ID) return INVALID_ID;

        obstacle_types[index] = 0;
        damaging[index] = (rand() % 4 == 0); // 25% chance
        speeds[index] = toGameUnit(50.0f + static_cast<float>(rand() % 101)); // 50-150 pixels/sec

        float dx = 15;
        float dy = 0;
        float len = sqrt(dx * dx + dy * dy);
        dir_x[index] = (len > 0.001f) ? dx / len : 1.0f;
        dir_y[index] = (len > 0.001f) ? dy / len : 0.0f;

        flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
        z_indices[index] = 5;

        return index;
    }

    uint32_t createEntity(int32_t x, int32_t y, int width, int height, int texture_id,  // Changed from float to int32_t
        bool is_damaging = false, int32_t speed = 50.0f,  // Changed from float to int32_t
        float direction_x = 150.0f, float direction_y = 0.0f) {
        uint32_t index = RenderableEntityContainer::createEntity();
        if (index == INVALID_ID) return INVALID_ID;

        // Store position directly as int32_t
        x_positions[index] = x;
        y_positions[index] = y;

        // Store dimensions
        widths[index] = width;
        heights[index] = height;

        texture_ids[index] = texture_id;
        obstacle_types[index] = 0;
        damaging[index] = is_damaging;
        speeds[index] = speed;

        // Normalize direction
        float len = sqrt(direction_x * direction_x + direction_y * direction_y);
        dir_x[index] = (len > 0.001f) ? direction_x / len : 1.0f;
        dir_y[index] = (len > 0.001f) ? direction_y / len : 0.0f;

        flags[index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
        z_indices[index] = 5;
        return index;
    }

    void removeEntity(size_t index) override {
        if (index >= count) return;
        size_t last = count - 1;
        if (index < last) {
            obstacle_types[index] = obstacle_types[last];
            damaging[index] = damaging[last];
            speeds[index] = speeds[last];
            dir_x[index] = dir_x[last];
            dir_y[index] = dir_y[last];
        }
        RenderableEntityContainer::removeEntity(index);
    }

    void update(uint32_t delta_time) override {
        // Determine number of hardware threads available
        const unsigned int num_threads = std::thread::hardware_concurrency();
        const int entities_per_thread = (count + num_threads - 1) / num_threads;

        // Define worker function that processes a range of entities
        auto process_entities = [&](int start, int end) {
            for (int i = start; i < end && i < count; ++i) {
                flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);

                // Get current fixed-point position
                int32_t& pos_x = x_positions[i];
                int32_t& pos_y = y_positions[i];
                float& direction_x = dir_x[i];
                float& direction_y = dir_y[i];

                // Get width, height, and speed in fixed-point
                const int32_t width = toGameUnit(widths[i]);
                const int32_t height = toGameUnit(heights[i]);
                const int32_t speed = speeds[i];

                // Update position
                pos_x = pos_x + (direction_x * speed * delta_time) / 1000;
                pos_y = pos_y + (direction_y * speed * delta_time) / 1000;

                // Handle world boundaries
                const int32_t max_x = WORLD_WIDTH - width;
                if (pos_x < 0 || pos_x > max_x) {
                    direction_x = -direction_x;
                    pos_x = std::clamp(pos_x, 0, max_x);
                }

                const int32_t max_y = WORLD_HEIGHT - height;
                if (pos_y < 0 || pos_y > max_y) {
                    direction_y = -direction_y;
                    pos_y = std::clamp(pos_y, 0, max_y);
                }
            }
            };

        // Launch worker threads
        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);

        for (unsigned int t = 0; t < num_threads; ++t) {
            int start = t * entities_per_thread;
            int end = start + entities_per_thread;
            futures.push_back(std::async(std::launch::async, process_entities, start, end));
        }

        // Wait for all threads to complete
        for (auto& future : futures) {
            future.wait();
        }
    }

protected:
    void resizeArrays(int newCapacity) override {
        if (newCapacity <= capacity) return;

        int* newObstacleTypes = new int[newCapacity];
        bool* newDamaging = new bool[newCapacity];
        int32_t* newSpeeds = new int32_t[newCapacity];  // Changed from uint32_t to int32_t
        float* newDirX = new float[newCapacity];
        float* newDirY = new float[newCapacity];

        if (count > 0) {
            std::copy(obstacle_types, obstacle_types + count, newObstacleTypes);
            std::copy(damaging, damaging + count, newDamaging);
            std::copy(speeds, speeds + count, newSpeeds);
            std::copy(dir_x, dir_x + count, newDirX);
            std::copy(dir_y, dir_y + count, newDirY);
        }
        std::fill(newObstacleTypes + count, newObstacleTypes + newCapacity, 0);
        std::fill(newDamaging + count, newDamaging + newCapacity, false);
        std::fill(newSpeeds + count, newSpeeds + newCapacity, 0);  // Changed to 0
        std::fill(newDirX + count, newDirX + newCapacity, 0.0f);
        std::fill(newDirY + count, newDirY + newCapacity, 0.0f);

        delete[] obstacle_types;
        delete[] damaging;
        delete[] speeds;
        delete[] dir_x;
        delete[] dir_y;

        obstacle_types = newObstacleTypes;
        damaging = newDamaging;
        speeds = newSpeeds;
        dir_x = newDirX;
        dir_y = newDirY;

        RenderableEntityContainer::resizeArrays(newCapacity);
    }
};

// --- Game State ---
struct GameState {
    uint32_t player_entity_index;
    int player_texture_id;
    int obstacle_texture_id;
    int item_texture_id;
    int score;
    bool game_over;
    bool game_won;
    int items_collected;
    int total_items;
    PlayerEntityContainer* player_container;
    ObstacleEntityContainer* obstacle_container;
    ItemEntityContainer* item_container;
    uint64_t last_fps_time;
    int frame_count;
    float current_fps;
    bool quit;
    uint64_t last_time;
    int frameCounter;
};

// --- Global variables for callbacks ---
Engine* g_engine = nullptr;
GameState g_game_state;

// --- Function Declarations ---
void setup_game(Engine* engine, GameState* game_state);
void handle_input(const Uint8* keyboard_state, Engine* engine, GameState* game_state, uint32_t delta_time_ms);
void check_collisions(Engine* engine, GameState* game_state);
void reset_game(Engine* engine, GameState* game_state);
SDL_Surface* create_colored_surface(int width, int height, Uint8 r, Uint8 g, Uint8 b);

SDL_Surface* create_colored_surface(int width, int height, Uint8 r, Uint8 g, Uint8 b) {
    SDL_Surface* surface = SDL_CreateSurface(width, height, SDL_PIXELFORMAT_RGBA8888);
    if (!surface) return nullptr;
    SDL_FillSurfaceRect(surface, NULL, SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format), 0, r, g, b, 255));
    return surface;
}

void setup_game(Engine* engine, GameState* game_state) {
    game_state->score = 0;
    game_state->game_over = false;
    game_state->game_won = false;
    game_state->items_collected = 0;
    game_state->total_items = NUM_ITEMS;
    game_state->player_entity_index = INVALID_ID;

    // Register Textures
    SDL_Surface* player_surface = create_colored_surface(fixedPointToFloat(PLAYER_SIZE), fixedPointToFloat(PLAYER_SIZE), 0, 0, 255);
    game_state->player_texture_id = engine->registerTexture(player_surface, 0, 0, fixedPointToFloat(PLAYER_SIZE), fixedPointToFloat(PLAYER_SIZE));
    SDL_DestroySurface(player_surface);

    SDL_Surface* obstacle_surface = create_colored_surface(fixedPointToFloat(OBSTACLE_SIZE), fixedPointToFloat(OBSTACLE_SIZE), 255, 0, 0);
    game_state->obstacle_texture_id = engine->registerTexture(obstacle_surface, 0, 0, fixedPointToFloat(OBSTACLE_SIZE), fixedPointToFloat(OBSTACLE_SIZE));
    SDL_DestroySurface(obstacle_surface);

    SDL_Surface* item_surface = create_colored_surface(fixedPointToFloat(ITEM_SIZE), fixedPointToFloat(ITEM_SIZE), 255, 255, 0);
    game_state->item_texture_id = engine->registerTexture(item_surface, 0, 0, fixedPointToFloat(ITEM_SIZE), fixedPointToFloat(ITEM_SIZE));
    SDL_DestroySurface(item_surface);

    // Create Player
    int32_t start_x = WORLD_WIDTH / 2 - PLAYER_SIZE / 2;  // Changed from float to int32_t
    int32_t start_y = WORLD_HEIGHT / 2 - PLAYER_SIZE / 2;  // Changed from float to int32_t
    game_state->player_entity_index = game_state->player_container->createEntity(
        start_x, start_y, fixedPointToFloat(PLAYER_SIZE), fixedPointToFloat(PLAYER_SIZE), game_state->player_texture_id
    );

    // Create Obstacles
    ObstacleEntityContainer* oCont = game_state->obstacle_container;

    // Divide world into grid for distribution
    const int32_t dist_grid_width = 2;  // Changed from uint32_t to int32_t
    const int32_t dist_grid_height = 2;  // Changed from uint32_t to int32_t
    const int32_t cell_width = (WORLD_WIDTH / dist_grid_width);  // Changed from uint32_t to int32_t
    const int32_t cell_height = (WORLD_HEIGHT / dist_grid_height);  // Changed from uint32_t to int32_t
    const uint32_t obstacles_per_cell = NUM_OBSTACLES / (dist_grid_width * dist_grid_height);
    const uint32_t obstacles_remainder = NUM_OBSTACLES % (dist_grid_width * dist_grid_height);

    int32_t player_center_x = WORLD_WIDTH / 2;  // Changed from uint32_t to int32_t
    int32_t player_center_y = WORLD_HEIGHT / 2;  // Changed from uint32_t to int32_t
    float safe_radius_sq = pow(fixedPointToFloat(PLAYER_SIZE) * 2, 2);

    for (int grid_y = 0; grid_y < dist_grid_height; ++grid_y) {
        for (int grid_x = 0; grid_x < dist_grid_width; ++grid_x) {
            int cell_obstacles = obstacles_per_cell;
            if ((grid_y * dist_grid_width + grid_x) < obstacles_remainder) {
                cell_obstacles++;
            }

            int32_t cell_min_x = grid_x * cell_width;  // Changed from uint32_t to int32_t
            int32_t cell_min_y = grid_y * cell_height;  // Changed from uint32_t to int32_t

            for (int i = 0; i < cell_obstacles; ++i) {
                int32_t margin = OBSTACLE_SIZE / 2;  // Changed from uint32_t to int32_t
                int32_t x_pos = cell_min_x + margin + (SDL_randf() * (cell_width - OBSTACLE_SIZE - margin * 2));  // Changed from long long to int32_t
                int32_t y_pos = cell_min_y + margin + (SDL_randf() * (cell_height - OBSTACLE_SIZE - margin * 2));  // Changed from long long to int32_t

                // Keep obstacles away from player
                float start_dist_sq = sqrt(pow(x_pos - player_center_x, 2) + pow(y_pos - player_center_y, 2));
                if (start_dist_sq < safe_radius_sq) {
                    float angle = atan2f(y_pos - player_center_y, x_pos - player_center_x);
                    float safe_dist = fixedPointToFloat(PLAYER_SIZE) * 10.0f + (rand() % 50);
                    x_pos = player_center_x + cosf(angle) * safe_dist;
                    y_pos = player_center_y + sinf(angle) * safe_dist;

                    // If still problematic, try alternative position
                    if ((pow(x_pos - player_center_x, 2) + pow(y_pos - player_center_y, 2) < safe_radius_sq) ||
                        x_pos < 0 || x_pos > WORLD_WIDTH - OBSTACLE_SIZE ||
                        y_pos < 0 || y_pos > WORLD_HEIGHT - OBSTACLE_SIZE) {
                        int quadrant = rand() % 4;
                        switch (quadrant) {
                        case 0: // Top-left
                            x_pos = (rand() % (WORLD_WIDTH / 4));
                            y_pos = (rand() % (WORLD_HEIGHT / 4));
                            break;
                        case 1: // Top-right
                            x_pos = (WORLD_WIDTH * 3 / 4 + rand() % (WORLD_WIDTH / 4));
                            y_pos = (rand() % (WORLD_HEIGHT / 4));
                            break;
                        case 2: // Bottom-left
                            x_pos = (rand() % (WORLD_WIDTH / 4));
                            y_pos = (WORLD_HEIGHT * 3 / 4 + rand() % (WORLD_HEIGHT / 4));
                            break;
                        case 3: // Bottom-right
                            x_pos = (WORLD_WIDTH * 3 / 4 + rand() % (WORLD_WIDTH / 4));
                            y_pos = (WORLD_HEIGHT * 3 / 4 + rand() % (WORLD_HEIGHT / 4));
                            break;
                        }
                    }
                }

                // Clamp to world bounds
                x_pos = std::max(0, std::min(WORLD_WIDTH - OBSTACLE_SIZE, x_pos));
                y_pos = std::max(0, std::min(WORLD_HEIGHT - OBSTACLE_SIZE, y_pos));

                bool is_damaging = (rand() % 4 == 0);
                int32_t speed = toGameUnit(7);  // Changed from uint32_t to int32_t
                oCont->createEntity(x_pos, y_pos, fixedPointToFloat(OBSTACLE_SIZE), fixedPointToFloat(OBSTACLE_SIZE),
                    game_state->obstacle_texture_id, is_damaging, speed, 15.0f, 0.0f);
            }
        }
    }

    // Create Items
    ItemEntityContainer* iCont = game_state->item_container;
    for (int i = 0; i < NUM_ITEMS; ++i) {
        int32_t x_pos = rand() % (WORLD_WIDTH - ITEM_SIZE);  // Changed from uint32_t to int32_t
        int32_t y_pos = rand() % (WORLD_HEIGHT - ITEM_SIZE);  // Changed from uint32_t to int32_t
        int value = 10 + (rand() % 11);
        iCont->createEntity(x_pos, y_pos, fixedPointToFloat(ITEM_SIZE), fixedPointToFloat(ITEM_SIZE), game_state->item_texture_id, value);
    }
}

void handle_input(const Uint8* keyboard_state, Engine* engine, GameState* game_state, uint32_t delta_time_ms) {
    if (game_state->player_entity_index == INVALID_ID) return;

    PlayerEntityContainer* pCont = game_state->player_container;
    uint32_t player_idx = game_state->player_entity_index;

    int32_t speed_fixed = pCont->speeds[player_idx]; // Player speed in fixed-point units per second
    float move_norm_x = 0.0f;
    float move_norm_y = 0.0f;

    if (keyboard_state[SDL_SCANCODE_W] || keyboard_state[SDL_SCANCODE_UP])    move_norm_y -= 1.0f;
    if (keyboard_state[SDL_SCANCODE_S] || keyboard_state[SDL_SCANCODE_DOWN])  move_norm_y += 1.0f;
    if (keyboard_state[SDL_SCANCODE_A] || keyboard_state[SDL_SCANCODE_LEFT])  move_norm_x -= 1.0f;
    if (keyboard_state[SDL_SCANCODE_D] || keyboard_state[SDL_SCANCODE_RIGHT]) move_norm_x += 1.0f;


    if (move_norm_x != 0.0f || move_norm_y != 0.0f) {
        pCont->isMoving[player_idx] = true;
    }
    // --- Camera Zoom (Handle potential fixed-point truncation) ---
    // Let's assume camera.width/height are fixed-point uint32_t
    // Multiplying fixed-point by float needs care to avoid truncation
    if (keyboard_state[SDL_SCANCODE_EQUALS] || keyboard_state[SDL_SCANCODE_KP_PLUS]) {
        const float zoom_factor = 0.99f; // Make zoom changes smaller per frame
        engine->camera.width = static_cast<uint32_t>(static_cast<double>(engine->camera.width) * zoom_factor); // Use double for intermediate calc
        engine->camera.height = static_cast<uint32_t>(static_cast<double>(engine->camera.height) * zoom_factor);
    }
    if (keyboard_state[SDL_SCANCODE_MINUS] || keyboard_state[SDL_SCANCODE_KP_MINUS]) {
        const float zoom_factor = 1.01f; // Make zoom changes smaller per frame
        engine->camera.width = static_cast<uint32_t>(static_cast<double>(engine->camera.width) * zoom_factor);
        engine->camera.height = static_cast<uint32_t>(static_cast<double>(engine->camera.height) * zoom_factor);
    }
    // Clamp camera zoom to prevent excessive zoom in/out which might exacerbate issues
    const uint32_t min_cam_dim = toGameUnit(100); // Example minimum dimension
    const uint32_t max_cam_dim = toGameUnit(WINDOW_WIDTH * 4); // Example max dimension
    engine->camera.width = std::clamp(engine->camera.width, min_cam_dim, max_cam_dim);
    engine->camera.height = std::clamp(engine->camera.height, min_cam_dim, max_cam_dim);


    pCont->isMoving[player_idx] = (move_norm_x != 0.0f || move_norm_y != 0.0f);
    // pCont->flags[player_idx] |= static_cast<uint8_t>(EntityFlag::VISIBLE); // Not needed here

    if (pCont->isMoving[player_idx]) {
        // Normalize direction vector (float)
        float length = sqrt(move_norm_x * move_norm_x + move_norm_y * move_norm_y);
        if (length > 0.001f) {
            move_norm_x /= length;
            move_norm_y /= length;
        }
        else {
            pCont->isMoving[player_idx] = false;
            return; // Avoid division by zero or tiny length
        }

        // Get current position (fixed-point) & dimensions (pixels)
        int32_t current_x = pCont->x_positions[player_idx];
        int32_t current_y = pCont->y_positions[player_idx];
        int current_width = pCont->widths[player_idx];
        int current_height = pCont->heights[player_idx];

        float delta_seconds = static_cast<float>(delta_time_ms) / 1000.0f;
        float speed_float = fixedPointToFloat(speed_fixed); // Pixels per second
        float displacement_x_pixels = move_norm_x * speed_float * delta_seconds;
        float displacement_y_pixels = move_norm_y * speed_float * delta_seconds;
        int32_t move_x = floatToFixedPoint(displacement_x_pixels);
        int32_t move_y = floatToFixedPoint(displacement_y_pixels);

        int32_t next_x = current_x + move_x;
        int32_t next_y = current_y + move_y;

        // Clamp to world bounds (fixed-point)
        // Ensure WORLD_WIDTH/HEIGHT are the fixed-point values
        const int32_t world_boundary_x = WORLD_WIDTH - toGameUnit(current_width);
        const int32_t world_boundary_y = WORLD_HEIGHT - toGameUnit(current_height);
        next_x = std::max((int32_t)0, std::min(world_boundary_x, next_x));
        next_y = std::max((int32_t)0, std::min(world_boundary_y, next_y));

        // Store new fixed-point position
        pCont->x_positions[player_idx] = next_x;
        pCont->y_positions[player_idx] = next_y;

        // Update camera smoothly based on new fixed-point player position
        engine->camera.x = next_x + toGameUnit(current_width) / 2;
        engine->camera.y = next_y + toGameUnit(current_height) / 2;
    }
}

void check_collisions(Engine* engine, GameState* game_state) {
    // Implementation remains empty in the original code
}

void reset_game(Engine* engine, GameState* game_state) {
    // Clear existing entities
    for (int i = 0; i < game_state->item_container->count; i++) {
        engine->entityManager.removeEntity(i, ENTITY_TYPE_ITEM);
    }
    for (int i = 0; i < game_state->obstacle_container->count; i++) {
        engine->entityManager.removeEntity(i, ENTITY_TYPE_OBSTACLE);
    }
    if (game_state->player_entity_index != INVALID_ID) {
        engine->entityManager.removeEntity(game_state->player_entity_index, ENTITY_TYPE_PLAYER);
    }

    // Setup game again
    setup_game(engine, game_state);

    if (game_state->player_entity_index != INVALID_ID) {
        // Get player position in fixed-point format
        int32_t player_x = game_state->player_container->x_positions[game_state->player_entity_index];  // Changed from float to int32_t
        int32_t player_y = game_state->player_container->y_positions[game_state->player_entity_index];  // Changed from float to int32_t
        int player_width = game_state->player_container->widths[game_state->player_entity_index];
        int player_height = game_state->player_container->heights[game_state->player_entity_index];

        // Update camera position
        engine->camera.x = player_x + toGameUnit(player_width) / 2;
        engine->camera.y = player_y + toGameUnit(player_height) / 2;

        game_state->player_container->flags[game_state->player_entity_index] |=
            static_cast<uint8_t>(EntityFlag::VISIBLE);
    }
    else {
        engine->camera.x = WORLD_WIDTH / 2;
        engine->camera.y = WORLD_HEIGHT / 2;
    }
}

// --- SDL callback functions ---

struct AppState {
    Engine* engine;
    GameState game_state;
};

SDL_AppResult SDL_AppInit(void** appstate, int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return SDL_APP_FAILURE;
    }
    srand(static_cast<unsigned int>(time(nullptr)));

    // Create app state
    AppState* state = new AppState();
    *appstate = state;

    state->engine = new Engine(WINDOW_WIDTH, WINDOW_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT);
    if (!state->engine) {
        std::cerr << "Failed to create engine" << std::endl;
        SDL_Quit();
        delete state;
        *appstate = nullptr;
        return SDL_APP_FAILURE;
    }

    // --- Register Entity Types ---
    PlayerEntityContainer* player_container = new PlayerEntityContainer(ENTITY_TYPE_PLAYER, 0, 10);
    ObstacleEntityContainer* obstacle_container = new ObstacleEntityContainer(state->engine, ENTITY_TYPE_OBSTACLE, 0, NUM_OBSTACLES + 100);
    ItemEntityContainer* item_container = new ItemEntityContainer(ENTITY_TYPE_ITEM, 0, NUM_ITEMS + 100);

    state->engine->entityManager.registerEntityType(player_container);
    state->engine->entityManager.registerEntityType(obstacle_container);
    state->engine->entityManager.registerEntityType(item_container);

    // --- Initialize Game State ---
    state->game_state = {};
    state->game_state.player_container = player_container;
    state->game_state.obstacle_container = obstacle_container;
    state->game_state.item_container = item_container;
    state->game_state.player_entity_index = INVALID_ID;
    state->game_state.last_fps_time = SDL_GetTicks();
    state->game_state.frame_count = 0;
    state->game_state.current_fps = 0.0f;
    state->game_state.quit = false;
    state->game_state.last_time = SDL_GetTicks();
    state->game_state.frameCounter = 0;

    // For backward compatibility with any existing code
    g_engine = state->engine;
    g_game_state = state->game_state;

    setup_game(state->engine, &state->game_state);

    if (state->game_state.player_entity_index != INVALID_ID) {
        // Set camera position to player - using int32_t
        int32_t player_x = player_container->x_positions[state->game_state.player_entity_index];
        int32_t player_y = player_container->y_positions[state->game_state.player_entity_index];

        // Add half the player size to center the camera
        int32_t half_width = toGameUnit(player_container->widths[state->game_state.player_entity_index]) / 2;
        int32_t half_height = toGameUnit(player_container->heights[state->game_state.player_entity_index]) / 2;

        state->engine->camera.x = player_x + half_width;
        state->engine->camera.y = player_y + half_height;

        player_container->flags[state->game_state.player_entity_index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    }
    else {
        state->engine->camera.x = WORLD_WIDTH / 2;
        state->engine->camera.y = WORLD_HEIGHT / 2;
    }

    // Camera dimensions should be in fixed point as well
    state->engine->camera.width = toGameUnit(WINDOW_WIDTH);
    state->engine->camera.height = toGameUnit(WINDOW_HEIGHT);
    state->engine->camera.zoom = 1; // 1.0 in fixed-point format

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event) {
    AppState* state = static_cast<AppState*>(appstate);

    if (event->type == SDL_EVENT_QUIT) {
        state->game_state.quit = true;
    }
    else if (event->type == SDL_EVENT_KEY_DOWN) {
        if (event->key.scancode == SDL_SCANCODE_ESCAPE) {
            state->game_state.quit = true;
        }
        else if (event->key.scancode == SDL_SCANCODE_R &&
            (state->game_state.game_over || state->game_state.game_won)) {
            reset_game(state->engine, &state->game_state);
        }
    }

    // Update globals for backward compatibility
    g_game_state = state->game_state;

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appstate) {
    AppState* state = static_cast<AppState*>(appstate);

    uint64_t current_time = SDL_GetTicks();
    uint32_t delta_time_ms = static_cast<uint32_t>(current_time - state->game_state.last_time);
    state->game_state.last_time = current_time;

    // Calculate FPS
    state->game_state.frame_count++;
    uint64_t time_since_last_fps = current_time - state->game_state.last_fps_time;
    if (time_since_last_fps >= 1000) {
        state->game_state.current_fps = static_cast<float>(state->game_state.frame_count * 1000.0f) / static_cast<float>(time_since_last_fps);
        state->game_state.last_fps_time = current_time;
        state->game_state.frame_count = 0;

        std::cout << "FPS: " << state->game_state.current_fps << std::endl;
    }

    const Uint8* keyboard_state = (uint8_t*)SDL_GetKeyboardState(NULL);

    // Handle Input and Game Logic
    if (!state->game_state.game_over) {
        handle_input(keyboard_state, state->engine, &state->game_state, delta_time_ms);

        if (state->game_state.frameCounter > 20) { // Start collisions after initial frames
            check_collisions(state->engine, &state->game_state);

            if (state->game_state.player_entity_index != INVALID_ID &&
                state->game_state.player_container->health[state->game_state.player_entity_index] <= 0) {
                state->game_state.game_over = true;
            }

            if (state->game_state.items_collected >= state->game_state.total_items) {
                state->game_state.game_won = true;
            }
        }
    }

    state->engine->update();
    state->engine->renderScene();
    state->engine->present();

    state->game_state.frameCounter++;

    // Update globals for backward compatibility
    g_game_state = state->game_state;

    return state->game_state.quit ? SDL_APP_FAILURE : SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult result) {
    if (appstate) {
        AppState* state = static_cast<AppState*>(appstate);
        delete state->engine;
        delete state;
    }
    SDL_Quit();
}