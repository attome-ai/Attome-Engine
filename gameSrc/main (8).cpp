
#include <SDL3/SDL.h>
#include "ATMEngine.h" // Use the provided engine header
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <iomanip>
#include <future>

// --- Constants ---
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1020

// --- Entity configuration ---
#define NUM_OBSTACLES 3000000  // Reduced from 1,000,000 for better performance
#define NUM_ITEMS 100       // Reduced from 500 for better performance
#define PLAYER_SIZE 32
#define OBSTACLE_SIZE 32
#define ITEM_SIZE 24
#define PLAYER_SPEED 350.0f // Pixels per second
#define PLAYER_MAX_HEALTH 100.0f
#define OBSTACLE_DAMAGE 10.0f // Damage per second when colliding

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
    float* speeds;
    float* health;
    bool* isMoving;

    PlayerEntityContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
        : RenderableEntityContainer(typeId, defaultLayer, initialCapacity)
    {
        speeds = new float[capacity];
        health = new float[capacity];
        isMoving = new bool[capacity];
        std::fill(speeds, speeds + capacity, 0.0f);
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

    void update(float delta_time) override {
        // Any player-specific updates would go here
        for (int i = 0; i < count; ++i) {
            // Ensure player is always visible
            flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
        }
    }

protected:
    void resizeArrays(int newCapacity) override {
        if (newCapacity <= capacity) return;

        float* newSpeeds = new float[newCapacity];
        float* newHealth = new float[newCapacity];
        bool* newIsMoving = new bool[newCapacity];

        if (count > 0) {
            std::copy(speeds, speeds + count, newSpeeds);
            std::copy(health, health + count, newHealth);
            std::copy(isMoving, isMoving + count, newIsMoving);
        }
        std::fill(newSpeeds + count, newSpeeds + newCapacity, 0.0f);
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

    void removeEntity(size_t index) override {
        if (index >= count) return;
        size_t last = count - 1;
        if (index < last) {
            item_types[index] = item_types[last];
            values[index] = values[last];
        }
        RenderableEntityContainer::removeEntity(index);
    }

    void update(float delta_time) override {
        // Make sure items are always visible
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
    float* speeds;
    float* dir_x;
    float* dir_y;
    int* behavior_flags;
    Engine* engine;
    ObstacleEntityContainer(Engine* engine, int typeId, uint8_t defaultLayer, int initialCapacity)
        : RenderableEntityContainer(typeId, defaultLayer, initialCapacity), engine(engine)
    {
        obstacle_types = new int[capacity];
        damaging = new bool[capacity];
        speeds = new float[capacity];
        dir_x = new float[capacity];
        dir_y = new float[capacity];
        behavior_flags = new int[capacity];
        std::fill(obstacle_types, obstacle_types + capacity, 0);
        std::fill(damaging, damaging + capacity, false);
        std::fill(speeds, speeds + capacity, 0.0f);
        std::fill(dir_x, dir_x + capacity, 0.0f);
        std::fill(dir_y, dir_y + capacity, 0.0f);
        std::fill(behavior_flags, behavior_flags + capacity, 0);
    }

    ~ObstacleEntityContainer() override {
        delete[] obstacle_types;
        delete[] damaging;
        delete[] speeds;
        delete[] dir_x;
        delete[] dir_y;
        delete[] behavior_flags;
    }

    uint32_t createEntity() override {
        uint32_t index = RenderableEntityContainer::createEntity();
        if (index == INVALID_ID) return INVALID_ID;

        obstacle_types[index] = 0;
        damaging[index] = (rand() % 4 == 0); // 25% chance
        speeds[index] = 50.0f + static_cast<float>(rand() % 101); // 50-150 pixels/sec

        float dx = static_cast<float>((rand() % 200) - 100);
        float dy = static_cast<float>((rand() % 200) - 100);
        float len = sqrt(dx * dx + dy * dy);
        if (len > 0.001f) {
            dir_x[index] = dx / len;
            dir_y[index] = dy / len;
        }
        else {
            dir_x[index] = 1.0f; dir_y[index] = 0.0f;
        }
        behavior_flags[index] = (rand() % 5 == 0) ? 0x1 : 0x0; // 1 in 5 orbits

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
            behavior_flags[index] = behavior_flags[last];
        }
        RenderableEntityContainer::removeEntity(index);
    }

    void update(float delta_time) override {

        PROFILE_SCOPE("UPDATE XXX");
        delta_time = std::min(delta_time, 0.1f); // Cap delta time

        const float half_world_width = WORLD_WIDTH * 0.5f;
        const float half_world_height = WORLD_HEIGHT * 0.5f;
        const float blend_factor = 0.1f;
        const float blend_factor_inv = 1.0f - blend_factor;

        // Determine number of hardware threads available
        const unsigned int num_threads = std::thread::hardware_concurrency();
        const int entities_per_thread = (count + num_threads - 1) / num_threads;

        // Define worker function that processes a range of entities
        auto process_entities = [&](int start, int end) {
            for (int i = start; i < end && i < count; ++i) {
                // Make sure every obstacle is visible
                flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);

                // Load all data for this entity at once to improve cache locality
                float& pos_x = x_positions[i];
                float& pos_y = y_positions[i];
                float& direction_x = dir_x[i];
                float& direction_y = dir_y[i];
                const int width = widths[i];
                const int height = heights[i];
                const float speed = speeds[i];
                const bool is_orbiting = behavior_flags[i] & 0x1;

                // Precalculate movement
                pos_x += direction_x * speed * delta_time;
                pos_y += direction_y * speed * delta_time;

                // Handle X boundary
                const float max_x = WORLD_WIDTH - width;
                if (pos_x < 0 || pos_x > max_x) {
                    direction_x = -direction_x;
                    pos_x = std::clamp(pos_x,0.0f,max_x);
                }

                // Handle Y boundary
                const float max_y = WORLD_HEIGHT - height;
                if (pos_y < 0 || pos_y > max_y) {
                    direction_y = -direction_y;
                    pos_y = std::clamp(pos_y, 0.0f, max_y);

                }
            }
            };

        // Launch worker threads and collect futures
        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);

        for (unsigned int t = 0; t < num_threads; ++t) {
            int start = t * entities_per_thread;
            int end = start + entities_per_thread;
            futures.push_back(std::async(std::launch::async, process_entities, start, end));
        }

        // Wait for all futures to complete
        for (auto& future : futures) {
            future.wait();
        }
    }

protected:
    void resizeArrays(int newCapacity) override {
        if (newCapacity <= capacity) return;

        int* newObstacleTypes = new int[newCapacity];
        bool* newDamaging = new bool[newCapacity];
        float* newSpeeds = new float[newCapacity];
        float* newDirX = new float[newCapacity];
        float* newDirY = new float[newCapacity];
        int* newBehaviorFlags = new int[newCapacity];

        if (count > 0) {
            std::copy(obstacle_types, obstacle_types + count, newObstacleTypes);
            std::copy(damaging, damaging + count, newDamaging);
            std::copy(speeds, speeds + count, newSpeeds);
            std::copy(dir_x, dir_x + count, newDirX);
            std::copy(dir_y, dir_y + count, newDirY);
            std::copy(behavior_flags, behavior_flags + count, newBehaviorFlags);
        }
        std::fill(newObstacleTypes + count, newObstacleTypes + newCapacity, 0);
        std::fill(newDamaging + count, newDamaging + newCapacity, false);
        std::fill(newSpeeds + count, newSpeeds + newCapacity, 0.0f);
        std::fill(newDirX + count, newDirX + newCapacity, 0.0f);
        std::fill(newDirY + count, newDirY + newCapacity, 0.0f);
        std::fill(newBehaviorFlags + count, newBehaviorFlags + newCapacity, 0);

        delete[] obstacle_types;
        delete[] damaging;
        delete[] speeds;
        delete[] dir_x;
        delete[] dir_y;
        delete[] behavior_flags;

        obstacle_types = newObstacleTypes;
        damaging = newDamaging;
        speeds = newSpeeds;
        dir_x = newDirX;
        dir_y = newDirY;
        behavior_flags = newBehaviorFlags;

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
    Uint64 last_fps_time;
    int frame_count;
    float current_fps;
};




void draw_debug_player(SDL_Renderer* renderer, Engine* engine, GameState* game_state) {
    if (game_state->player_entity_index == INVALID_ID) return;

    PlayerEntityContainer* pCont = game_state->player_container;
    uint32_t player_idx = game_state->player_entity_index;

    // Get world coordinates
    float world_x = pCont->x_positions[player_idx];
    float world_y = pCont->y_positions[player_idx];
    int width = pCont->widths[player_idx];
    int height = pCont->heights[player_idx];

    // Convert to screen coordinates with alternative method
    float screen_x = (world_x - engine->camera.x) * engine->camera.zoom + (engine->camera.width / 2.0f);
    float screen_y = (world_y - engine->camera.y) * engine->camera.zoom + (engine->camera.height / 2.0f);
    float screen_w = width * engine->camera.zoom;
    float screen_h = height * engine->camera.zoom;

    // Draw debug rectangle for player in a bright color that stands out
    SDL_FRect debug_rect = { screen_x, screen_y, screen_w, screen_h };
    SDL_SetRenderDrawColor(renderer, 255, 0, 255, 255); // Magenta
    SDL_RenderFillRect(renderer, &debug_rect);

    // Draw crosshair at player center
    float center_x = screen_x + screen_w / 2;
    float center_y = screen_y + screen_h / 2;
    SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255); // Yellow
    SDL_RenderLine(renderer, center_x - 10, center_y, center_x + 10, center_y);
    SDL_RenderLine(renderer, center_x, center_y - 10, center_x, center_y + 10);
}

// --- Function Declarations ---
void setup_game(Engine* engine, GameState* game_state);
void handle_input(const Uint8* keyboard_state, Engine* engine, GameState* game_state, float delta_time);
void check_collisions(Engine* engine, GameState* game_state);
void reset_game(Engine* engine, GameState* game_state);
SDL_Surface* create_colored_surface(int width, int height, Uint8 r, Uint8 g, Uint8 b);
// New debugging function
void debug_entity_visibility(Engine* engine, GameState* game_state);

// --- Main Function ---
int main(int argc, char* argv[]) {

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return 1;
    }
    srand(static_cast<unsigned int>(time(nullptr)));

    // Create engine with explicit renderer flags to disable vsync
    SDL_Window* window = SDL_CreateWindow("2D Game Engine - SDL3", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    if (!window) {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    // Create renderer explicitly with ACCELERATED flag but WITHOUT PRESENTVSYNC
    SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
    if (!renderer) {
        std::cerr << "Failed to create renderer: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Destroy the temporary window and renderer
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);

    // Now create the engine with the correct setup
    Engine* engine = engine_create(WINDOW_WIDTH, WINDOW_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT, GRID_CELL_SIZE);
    if (!engine) {
        std::cerr << "Failed to create engine" << std::endl;
        SDL_Quit();
        return 1;
    }

    // --- Register Entity Types ---
    PlayerEntityContainer* player_container = new PlayerEntityContainer(ENTITY_TYPE_PLAYER, 0, 10);
    ObstacleEntityContainer* obstacle_container = new ObstacleEntityContainer(engine, ENTITY_TYPE_OBSTACLE, 0, NUM_OBSTACLES + 100);
    ItemEntityContainer* item_container = new ItemEntityContainer(ENTITY_TYPE_ITEM, 0, NUM_ITEMS + 100);

    engine->entityManager.registerEntityType(player_container);
    engine->entityManager.registerEntityType(obstacle_container);
    engine->entityManager.registerEntityType(item_container);

    // --- Initialize Game State ---
    GameState game_state = {};
    game_state.player_container = player_container;
    game_state.obstacle_container = obstacle_container;
    game_state.item_container = item_container;
    game_state.player_entity_index = INVALID_ID;
    game_state.last_fps_time = SDL_GetTicks();
    game_state.frame_count = 0;
    game_state.current_fps = 0.0f;

    setup_game(engine, &game_state);

    if (game_state.player_entity_index != INVALID_ID) {
        // Set camera position to player
        float player_x = player_container->x_positions[game_state.player_entity_index];
        float player_y = player_container->y_positions[game_state.player_entity_index];
        engine->camera.x = player_x + PLAYER_SIZE / 2.0f;
        engine->camera.y = player_y + PLAYER_SIZE / 2.0f;

        // Force player visibility
        player_container->flags[game_state.player_entity_index] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
    }
    else {
        engine->camera.x = WORLD_WIDTH / 2.0f;
        engine->camera.y = WORLD_HEIGHT / 2.0f;
    }

    // --- Game Loop ---
    bool quit = false;
    SDL_Event event;
    Uint64 last_time = SDL_GetTicks();
    int frameCounter = 0;

    while (!quit) {
        Uint64 current_time = SDL_GetTicks();
        float delta_time = std::min((current_time - last_time) / 1000.0f, 0.1f);
        last_time = current_time;

        // --- Calculate and Log FPS ---
        game_state.frame_count++;
        Uint64 time_since_last_fps = current_time - game_state.last_fps_time;
        if (time_since_last_fps >= 1000) {
            game_state.current_fps = static_cast<float>(game_state.frame_count * 1000.0f) / static_cast<float>(time_since_last_fps);
            game_state.last_fps_time = current_time;
            game_state.frame_count = 0;
            std::cout << "FPS: " << std::fixed << std::setprecision(2) << game_state.current_fps << std::endl;
        }
        const Uint8* keyboard_state = (uint8_t*)SDL_GetKeyboardState(NULL);

        // --- Process Events ---
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) quit = true;
            else if (event.type == SDL_EVENT_KEY_DOWN) {
                if (event.key.scancode == SDL_SCANCODE_ESCAPE) quit = true;
                else if (event.key.scancode == SDL_SCANCODE_R && (game_state.game_over || game_state.game_won)) {
                    reset_game(engine, &game_state);
                }
                else if (event.key.scancode == SDL_SCANCODE_D &&
                    (keyboard_state[SDL_SCANCODE_LCTRL] || keyboard_state[SDL_SCANCODE_RCTRL])) {
                    // Debug key combination - print entity info
                    debug_entity_visibility(engine, &game_state);
                }
            }
        }

        // --- Handle Input ---
        if (!game_state.game_over && !game_state.game_won) {
            handle_input(keyboard_state, engine, &game_state, delta_time);
        }

        // --- Game Logic ---
        if (!game_state.game_over && !game_state.game_won) {
            // Force visibility flags for all entity types
            // This is the key fix - ensure ALL entities have their visibility flags set properly
            if (game_state.player_entity_index != INVALID_ID) {
                game_state.player_container->flags[game_state.player_entity_index] |=
                    static_cast<uint8_t>(EntityFlag::VISIBLE);
            }

            // Force visibility on a sample of obstacles and items
            const int sample_size = std::min(10, game_state.obstacle_container->count);
            for (int i = 0; i < sample_size; i++) {
                game_state.obstacle_container->flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
            }

            const int item_sample = std::min(10, game_state.item_container->count);
            for (int i = 0; i < item_sample; i++) {
                game_state.item_container->flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
            }

            // Collision detection - disabled initially for testing
            if (frameCounter > 20) { // Only start collisions after initial frames
                try {
                    check_collisions(engine, &game_state);

                    if (game_state.player_entity_index != INVALID_ID &&
                        game_state.player_container->health[game_state.player_entity_index] <= 0) {
                        game_state.game_over = true;
                    }

                    if (game_state.items_collected >= game_state.total_items) {
                        game_state.game_won = true;
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Exception during collision check: " << e.what() << std::endl;
                }
            }
        }

        // --- Update Engine ---
        // Debug player position before and after engine update
        if (game_state.player_entity_index != INVALID_ID) {
            float before_x = game_state.player_container->x_positions[game_state.player_entity_index];
            float before_y = game_state.player_container->y_positions[game_state.player_entity_index];

            engine_update(engine);

            float after_x = game_state.player_container->x_positions[game_state.player_entity_index];
            float after_y = game_state.player_container->y_positions[game_state.player_entity_index];

            // Only print if position changed
            if (before_x != after_x || before_y != after_y) {
                std::cout << "Player position changed during engine_update: ("
                    << before_x << "," << before_y << ") -> ("
                    << after_x << "," << after_y << ")" << std::endl;
            }
        }
        else {
            engine_update(engine);
        }

        // Always update grid with current player position after engine update
        if (game_state.player_entity_index != INVALID_ID) {
            float player_x = game_state.player_container->x_positions[game_state.player_entity_index];
            float player_y = game_state.player_container->y_positions[game_state.player_entity_index];
            EntityRef player_ref = { ENTITY_TYPE_PLAYER, static_cast<uint32_t>(game_state.player_entity_index) };
        }

        // --- Rendering ---
        // Clear before rendering new frame
        SDL_SetRenderDrawColor(engine->renderer, 0, 0, 0, 255);
        SDL_RenderClear(engine->renderer);

        // Render with engine
        engine_render_scene(engine);



        // Present the renderer
        engine_present(engine);

        frameCounter++;
    }

    engine_destroy(engine);
    SDL_Quit();
    return 0;
}

// Helper function to debug entity visibility
void debug_entity_visibility(Engine* engine, GameState* game_state) {
    std::cout << "\n==== ENTITY VISIBILITY DEBUG ====" << std::endl;

    // Player
    if (game_state->player_entity_index != INVALID_ID) {
        PlayerEntityContainer* pCont = game_state->player_container;
        uint32_t player_idx = game_state->player_entity_index;
        bool is_visible = (pCont->flags[player_idx] & static_cast<uint8_t>(EntityFlag::VISIBLE)) != 0;

        std::cout << "Player (ID:" << player_idx << "):" << std::endl;
        std::cout << "  Position: (" << pCont->x_positions[player_idx] << ", " << pCont->y_positions[player_idx] << ")" << std::endl;
        std::cout << "  Size: " << pCont->widths[player_idx] << "x" << pCont->heights[player_idx] << std::endl;
        std::cout << "  Texture ID: " << pCont->texture_ids[player_idx] << std::endl;
        std::cout << "  Visible: " << (is_visible ? "YES" : "NO") << std::endl;
        std::cout << "  Health: " << pCont->health[player_idx] << std::endl;
    }
    else {
        std::cout << "Player: INVALID" << std::endl;
    }

    // Obstacles (first few)
    std::cout << "\nObstacles: " << game_state->obstacle_container->count << " total" << std::endl;
    const int obstacle_sample = std::min(5, game_state->obstacle_container->count);
    for (int i = 0; i < obstacle_sample; i++) {
        ObstacleEntityContainer* oCont = game_state->obstacle_container;
        bool is_visible = (oCont->flags[i] & static_cast<uint8_t>(EntityFlag::VISIBLE)) != 0;

        std::cout << "  Obstacle " << i << ":" << std::endl;
        std::cout << "    Position: (" << oCont->x_positions[i] << ", " << oCont->y_positions[i] << ")" << std::endl;
        std::cout << "    Size: " << oCont->widths[i] << "x" << oCont->heights[i] << std::endl;
        std::cout << "    Texture ID: " << oCont->texture_ids[i] << std::endl;
        std::cout << "    Visible: " << (is_visible ? "YES" : "NO") << std::endl;
    }

    // Items (first few)
    std::cout << "\nItems: " << game_state->item_container->count << " total" << std::endl;
    const int item_sample = std::min(5, game_state->item_container->count);
    for (int i = 0; i < item_sample; i++) {
        ItemEntityContainer* iCont = game_state->item_container;
        bool is_visible = (iCont->flags[i] & static_cast<uint8_t>(EntityFlag::VISIBLE)) != 0;

        std::cout << "  Item " << i << ":" << std::endl;
        std::cout << "    Position: (" << iCont->x_positions[i] << ", " << iCont->y_positions[i] << ")" << std::endl;
        std::cout << "    Size: " << iCont->widths[i] << "x" << iCont->heights[i] << std::endl;
        std::cout << "    Texture ID: " << iCont->texture_ids[i] << std::endl;
        std::cout << "    Visible: " << (is_visible ? "YES" : "NO") << std::endl;
    }

    // Camera info
    std::cout << "\nCamera:" << std::endl;
    std::cout << "  Position: (" << engine->camera.x << ", " << engine->camera.y << ")" << std::endl;
    std::cout << "  Size: " << engine->camera.width << "x" << engine->camera.height << std::endl;
    std::cout << "  Zoom: " << engine->camera.zoom << std::endl;

    std::cout << "================================\n" << std::endl;
}

// --- Function Implementations ---
SDL_Surface* create_colored_surface(int width, int height, Uint8 r, Uint8 g, Uint8 b) {
    SDL_Surface* surface = SDL_CreateSurface(width, height, SDL_PIXELFORMAT_RGBA8888);
    if (!surface) { SDL_Log("Failed to create surface: %s", SDL_GetError()); return nullptr; }

    SDL_FillSurfaceRect(surface, NULL, SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format), 0, r, g, b, 255));
    return surface;
}

void setup_game(Engine* engine, GameState* game_state) {
    std::cout << "Setting up game..." << std::endl;

    game_state->score = 0;
    game_state->game_over = false;
    game_state->game_won = false;
    game_state->items_collected = 0;
    game_state->total_items = NUM_ITEMS;
    game_state->player_entity_index = INVALID_ID;

    // --- Register Textures ---
    SDL_Surface* player_surface = create_colored_surface(PLAYER_SIZE, PLAYER_SIZE, 0, 0, 255);
    // FIX: Make sure the texture creation succeeded and is a valid ID
    game_state->player_texture_id = engine_register_texture(engine, player_surface, 0, 0, PLAYER_SIZE, PLAYER_SIZE);
    SDL_DestroySurface(player_surface);
    std::cout << "Player texture ID: " << game_state->player_texture_id << std::endl;

    SDL_Surface* obstacle_surface = create_colored_surface(OBSTACLE_SIZE, OBSTACLE_SIZE, 255, 0, 0);
    game_state->obstacle_texture_id = engine_register_texture(engine, obstacle_surface, 0, 0, OBSTACLE_SIZE, OBSTACLE_SIZE);
    SDL_DestroySurface(obstacle_surface);
    std::cout << "Obstacle texture ID: " << game_state->obstacle_texture_id << std::endl;

    SDL_Surface* item_surface = create_colored_surface(ITEM_SIZE, ITEM_SIZE, 255, 255, 0);
    game_state->item_texture_id = engine_register_texture(engine, item_surface, 0, 0, ITEM_SIZE, ITEM_SIZE);
    SDL_DestroySurface(item_surface);
    std::cout << "Item texture ID: " << game_state->item_texture_id << std::endl;

    // Verify textures were created successfully
    if (game_state->player_texture_id < 0 ||
        game_state->obstacle_texture_id < 0 ||
        game_state->item_texture_id < 0) {
        std::cerr << "WARNING: Failed to create one or more textures!" << std::endl;
    }

    // --- Create Player ---
    uint32_t player_idx = engine->entityManager.createEntity(ENTITY_TYPE_PLAYER);
    if (player_idx != INVALID_ID) {
        game_state->player_entity_index = player_idx;
        PlayerEntityContainer* pCont = game_state->player_container;

        // Set initial position
        float start_x = WORLD_WIDTH / 2 - PLAYER_SIZE / 2;
        float start_y = WORLD_HEIGHT / 2 - PLAYER_SIZE / 2;
        pCont->x_positions[player_idx] = start_x;
        pCont->y_positions[player_idx] = start_y;
        pCont->widths[player_idx] = PLAYER_SIZE;
        pCont->heights[player_idx] = PLAYER_SIZE;
        pCont->texture_ids[player_idx] = game_state->player_texture_id;
        pCont->health[player_idx] = PLAYER_MAX_HEALTH;
        pCont->flags[player_idx] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
        pCont->z_indices[player_idx] = 10; // Ensure player is on top

        EntityRef player_ref = { ENTITY_TYPE_PLAYER, static_cast<uint32_t>(player_idx) };



        std::cout << "Player created at position: " << pCont->x_positions[player_idx]
            << ", " << pCont->y_positions[player_idx] << std::endl;
        std::cout << "Player size: " << pCont->widths[player_idx]
            << " x " << pCont->heights[player_idx] << std::endl;
        std::cout << "Player texture ID: " << pCont->texture_ids[player_idx] << std::endl;
        std::cout << "Player is visible: "
            << ((pCont->flags[player_idx] & static_cast<uint8_t>(EntityFlag::VISIBLE)) ? "YES" : "NO") << std::endl;
    }
    else {
        std::cerr << "Failed to create player entity!" << std::endl;
    }

    std::cout << "Creating obstacles with improved distribution..." << std::endl;
    ObstacleEntityContainer* oCont = game_state->obstacle_container;
    int obstacles_created = 0;

    // Divide the world into a grid of cells for distribution
    const int dist_grid_width = 50;  // More cells for finer distribution
    const int dist_grid_height = 50;
    const float cell_width = WORLD_WIDTH / (float)dist_grid_width;
    const float cell_height = WORLD_HEIGHT / (float)dist_grid_height;
    const int obstacles_per_cell = NUM_OBSTACLES / (dist_grid_width * dist_grid_height);
    const int obstacles_remainder = NUM_OBSTACLES % (dist_grid_width * dist_grid_height);

    // Create obstacles distributed across the grid
    for (int grid_y = 0; grid_y < dist_grid_height; ++grid_y) {
        for (int grid_x = 0; grid_x < dist_grid_width; ++grid_x) {
            int cell_obstacles = obstacles_per_cell;
            // Distribute remainder evenly
            if ((grid_y * dist_grid_width + grid_x) < obstacles_remainder) {
                cell_obstacles++;
            }

            // Calculate position within this grid cell
            float cell_min_x = grid_x * cell_width;
            float cell_min_y = grid_y * cell_height;

            // Keep track of player safe zone
            float player_center_x = WORLD_WIDTH / 2.0f;
            float player_center_y = WORLD_HEIGHT / 2.0f;
            float safe_radius_sq = pow(PLAYER_SIZE * 10.0f, 2);

            for (int i = 0; i < cell_obstacles; ++i) {
                uint32_t obstacle_idx = engine->entityManager.createEntity(ENTITY_TYPE_OBSTACLE);
                if (obstacle_idx != INVALID_ID) {
                    // Random position within the cell, keeping margin from edges
                    float margin = OBSTACLE_SIZE * 0.5f;
                    float x_pos = cell_min_x + margin + (static_cast<float>(rand()) / RAND_MAX) * (cell_width - OBSTACLE_SIZE - margin * 2);
                    float y_pos = cell_min_y + margin + (static_cast<float>(rand()) / RAND_MAX) * (cell_height - OBSTACLE_SIZE - margin * 2);

                    // Keep obstacles away from player start position (center of world)
                    float start_dist_sq = pow(x_pos - player_center_x, 2) + pow(y_pos - player_center_y, 2);
                    if (start_dist_sq < safe_radius_sq) {
                        // Move it away radially
                        float angle = atan2f(y_pos - player_center_y, x_pos - player_center_x);
                        float safe_dist = PLAYER_SIZE * 10.0f + (rand() % 50); // Add randomness to avoid clustering
                        x_pos = player_center_x + cosf(angle) * safe_dist;
                        y_pos = player_center_y + sinf(angle) * safe_dist;

                        // If we're still inside the safe zone (could happen due to numerical issues)
                        // or outside the world bounds, try a completely different position
                        if ((pow(x_pos - player_center_x, 2) + pow(y_pos - player_center_y, 2) < safe_radius_sq) ||
                            x_pos < 0 || x_pos > WORLD_WIDTH - OBSTACLE_SIZE ||
                            y_pos < 0 || y_pos > WORLD_HEIGHT - OBSTACLE_SIZE) {

                            // Try to find another spot in this cell
                            int retry = 0;
                            bool found = false;
                            while (retry < 5 && !found) {  // Limit retries
                                x_pos = cell_min_x + margin + (static_cast<float>(rand()) / RAND_MAX) * (cell_width - OBSTACLE_SIZE - margin * 2);
                                y_pos = cell_min_y + margin + (static_cast<float>(rand()) / RAND_MAX) * (cell_height - OBSTACLE_SIZE - margin * 2);

                                start_dist_sq = pow(x_pos - player_center_x, 2) + pow(y_pos - player_center_y, 2);
                                if (start_dist_sq >= safe_radius_sq) {
                                    found = true;
                                }
                                retry++;
                            }

                            // If we still can't find a good spot, place it elsewhere in the world
                            if (!found) {
                                // Pick a random quadrant that's not the center
                                int quadrant = rand() % 4;
                                switch (quadrant) {
                                case 0: // Top-left
                                    x_pos = (float)(rand() % (WORLD_WIDTH / 4));
                                    y_pos = (float)(rand() % (WORLD_HEIGHT / 4));
                                    break;
                                case 1: // Top-right
                                    x_pos = (float)(WORLD_WIDTH * 3 / 4 + rand() % (WORLD_WIDTH / 4));
                                    y_pos = (float)(rand() % (WORLD_HEIGHT / 4));
                                    break;
                                case 2: // Bottom-left
                                    x_pos = (float)(rand() % (WORLD_WIDTH / 4));
                                    y_pos = (float)(WORLD_HEIGHT * 3 / 4 + rand() % (WORLD_HEIGHT / 4));
                                    break;
                                case 3: // Bottom-right
                                    x_pos = (float)(WORLD_WIDTH * 3 / 4 + rand() % (WORLD_WIDTH / 4));
                                    y_pos = (float)(WORLD_HEIGHT * 3 / 4 + rand() % (WORLD_HEIGHT / 4));
                                    break;
                                }
                            }
                        }
                    }

                    // Final position clamping
                    x_pos = std::max(0.0f, std::min((float)WORLD_WIDTH - OBSTACLE_SIZE, x_pos));
                    y_pos = std::max(0.0f, std::min((float)WORLD_HEIGHT - OBSTACLE_SIZE, y_pos));

                    oCont->x_positions[obstacle_idx] = x_pos;
                    oCont->y_positions[obstacle_idx] = y_pos;
                    oCont->widths[obstacle_idx] = OBSTACLE_SIZE;
                    oCont->heights[obstacle_idx] = OBSTACLE_SIZE;
                    oCont->texture_ids[obstacle_idx] = game_state->obstacle_texture_id;
                    oCont->flags[obstacle_idx] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
                    oCont->z_indices[obstacle_idx] = 5; // Ensure z-index is set

                    // Randomize movement properties
                    oCont->damaging[obstacle_idx] = (rand() % 4 == 0); // 25% chance of being damaging
                    oCont->speeds[obstacle_idx] = 50.0f + static_cast<float>(rand() % 101); // 50-150 pixels/sec

                    float dx = static_cast<float>((rand() % 200) - 100);
                    float dy = static_cast<float>((rand() % 200) - 100);
                    float len = sqrt(dx * dx + dy * dy);
                    if (len > 0.001f) {
                        oCont->dir_x[obstacle_idx] = dx / len;
                        oCont->dir_y[obstacle_idx] = dy / len;
                    }
                    else {
                        oCont->dir_x[obstacle_idx] = 1.0f; oCont->dir_y[obstacle_idx] = 0.0f;
                    }
                    oCont->behavior_flags[obstacle_idx] = (rand() % 5 == 0) ? 0x1 : 0x0; // 1 in 5 orbits

                    EntityRef obstacle_ref = { ENTITY_TYPE_OBSTACLE, obstacle_idx };

                    obstacles_created++;

                    // Print progress every 100k obstacles
                    if (obstacles_created % 100000 == 0) {
                        std::cout << "Created " << obstacles_created << " obstacles so far..." << std::endl;
                    }
                }
            }
        }
    }
    std::cout << "Created " << obstacles_created << " obstacles with improved distribution" << std::endl;

    // --- Create Items ---
    std::cout << "Creating items..." << std::endl;
    ItemEntityContainer* iCont = game_state->item_container;
    int items_created = 0;

    for (int i = 0; i < NUM_ITEMS; ++i) {
        uint32_t item_idx = engine->entityManager.createEntity(ENTITY_TYPE_ITEM);
        if (item_idx != INVALID_ID) {
            float x_pos = rand() % (WORLD_WIDTH - ITEM_SIZE);
            float y_pos = rand() % (WORLD_HEIGHT - ITEM_SIZE);
            iCont->x_positions[item_idx] = x_pos;
            iCont->y_positions[item_idx] = y_pos;
            iCont->widths[item_idx] = ITEM_SIZE;
            iCont->heights[item_idx] = ITEM_SIZE;
            iCont->texture_ids[item_idx] = game_state->item_texture_id;
            iCont->values[item_idx] = 10 + (rand() % 11);
            iCont->flags[item_idx] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
            iCont->z_indices[item_idx] = 1; // Ensure z-index is set

            EntityRef item_ref = { ENTITY_TYPE_ITEM, item_idx};

            items_created++;
        }
    }
    std::cout << "Created " << items_created << " items" << std::endl;

    // Print summary
    std::cout << "Game initialization summary:" << std::endl;
    std::cout << "Player entity index: " << game_state->player_entity_index << std::endl;
    std::cout << "Number of obstacles: " << game_state->obstacle_container->count << "/" << NUM_OBSTACLES << std::endl;
    std::cout << "Number of items: " << game_state->item_container->count << "/" << NUM_ITEMS << std::endl;
}

// Handle player input with improved camera handling
void handle_input(const Uint8* keyboard_state, Engine* engine, GameState* game_state, float delta_time) {
    if (game_state->player_entity_index == INVALID_ID) return;

    PlayerEntityContainer* pCont = game_state->player_container;
    uint32_t player_idx = game_state->player_entity_index;

    float speed_pixels_per_sec = pCont->speeds[player_idx];
    float move_norm_x = 0.0f; // Normalized direction component
    float move_norm_y = 0.0f;

    if (keyboard_state[SDL_SCANCODE_W] || keyboard_state[SDL_SCANCODE_UP])    move_norm_y -= 1.0f;
    if (keyboard_state[SDL_SCANCODE_S] || keyboard_state[SDL_SCANCODE_DOWN])  move_norm_y += 1.0f;
    if (keyboard_state[SDL_SCANCODE_A] || keyboard_state[SDL_SCANCODE_LEFT])  move_norm_x -= 1.0f;
    if (keyboard_state[SDL_SCANCODE_D] || keyboard_state[SDL_SCANCODE_RIGHT]) move_norm_x += 1.0f;

    // Add camera zoom controls
    if (keyboard_state[SDL_SCANCODE_EQUALS] || keyboard_state[SDL_SCANCODE_KP_PLUS]) {
        // Zoom in
        float new_zoom = engine->camera.zoom * 1.05f;
        engine->camera.width *= 1.1;
        engine->camera.height *= 1.1;

    }
    if (keyboard_state[SDL_SCANCODE_MINUS] || keyboard_state[SDL_SCANCODE_KP_MINUS]) {
        engine->camera.width *= 0.9;
        engine->camera.height *= 0.9;
    }
    if (keyboard_state[SDL_SCANCODE_0] || keyboard_state[SDL_SCANCODE_KP_0])
    {
    }

    pCont->isMoving[player_idx] = (move_norm_x != 0.0f || move_norm_y != 0.0f);

    // Ensure player visibility flag is set
    pCont->flags[player_idx] |= static_cast<uint8_t>(EntityFlag::VISIBLE);

    if (pCont->isMoving[player_idx]) {
        // Normalize direction vector
        float length = sqrt(move_norm_x * move_norm_x + move_norm_y * move_norm_y);
        if (length > 0.001f) { // Avoid division by zero
            move_norm_x /= length;
            move_norm_y /= length;
        }
        else {
            pCont->isMoving[player_idx] = false;
            return;
        }

        // Calculate movement
        float move_x = move_norm_x * speed_pixels_per_sec * delta_time;
        float move_y = move_norm_y * speed_pixels_per_sec * delta_time;

        // Get current position
        float current_x = pCont->x_positions[player_idx];
        float current_y = pCont->y_positions[player_idx];

        // Calculate new position
        float next_x = current_x + move_x;
        float next_y = current_y + move_y;

        // Clamp to world bounds
        next_x = std::max(0.0f, std::min(static_cast<float>(WORLD_WIDTH - pCont->widths[player_idx]), next_x));
        next_y = std::max(0.0f, std::min(static_cast<float>(WORLD_HEIGHT - pCont->heights[player_idx]), next_y));

        // Store updated position
        pCont->x_positions[player_idx] = next_x;
        pCont->y_positions[player_idx] = next_y;

        // Update spatial grid - FIX: Actually update the grid when player moves
        EntityRef player_ref = { ENTITY_TYPE_PLAYER, player_idx };

        // Update camera position to center on player
        engine->camera.x = next_x + pCont->widths[player_idx] / 2.0f;
        engine->camera.y = next_y + pCont->heights[player_idx] / 2.0f;

        std::cout << " player x : " << next_x << " player y " << next_y << std::endl;

    }
}

// Safe collision check with error handling
void check_collisions(Engine* engine, GameState* game_state) {
    if (game_state->player_entity_index == INVALID_ID) return;

    PlayerEntityContainer* pCont = game_state->player_container;
    ObstacleEntityContainer* oCont = game_state->obstacle_container;
    ItemEntityContainer* iCont = game_state->item_container;
    uint32_t player_idx = game_state->player_entity_index;

    // Validate player index
    if (player_idx >= pCont->count) {
        std::cerr << "Invalid player index: " << player_idx << std::endl;
        return;
    }

    // Get player position
    float player_x = pCont->x_positions[player_idx];
    float player_y = pCont->y_positions[player_idx];
    int player_width = pCont->widths[player_idx];
    int player_height = pCont->heights[player_idx];

    // Query the spatial grid for nearby entities
    float player_pos_x = player_x + player_width / 2.0f;
    float player_pos_y = player_y + player_height / 2.0f;
    float query_range = std::max(player_width, player_height) * 2.0f;

    try {
        const auto& nearby_entities = engine->grid.queryCircle(player_pos_x, player_pos_y, query_range);

        // Store items to remove to avoid removing during iteration
        std::vector<EntityRef> items_to_remove;

        // Check collision with each nearby entity
        for (const auto& entity : nearby_entities) {
            // Skip player itself
            if (entity.type == ENTITY_TYPE_PLAYER && entity.index == static_cast<int32_t>(player_idx)) continue;

            // Skip invalid entities
            if (entity.type < 0 || entity.index < 0) continue;

            // Process different entity types
            if (entity.type == ENTITY_TYPE_OBSTACLE) {
                // Validate obstacle index
                if (entity.index >= oCont->count) continue;

                // Get obstacle position and size
                float obstacle_x = oCont->x_positions[entity.index];
                float obstacle_y = oCont->y_positions[entity.index];
                int obstacle_width = oCont->widths[entity.index];
                int obstacle_height = oCont->heights[entity.index];

                // Check for AABB collision
                if (player_x < obstacle_x + obstacle_width &&
                    player_x + player_width > obstacle_x &&
                    player_y < obstacle_y + obstacle_height &&
                    player_y + player_height > obstacle_y) {

                    // Handle collision with obstacle
                    if (oCont->damaging[entity.index]) {
                        // Apply damage based on delta time
                        pCont->health[player_idx] -= OBSTACLE_DAMAGE * 0.016f;
                    }
                }
            }
            else if (entity.type == ENTITY_TYPE_ITEM) {
                // Validate item index
                if (entity.index >= iCont->count) continue;

                // Get item position and size
                float item_x = iCont->x_positions[entity.index];
                float item_y = iCont->y_positions[entity.index];
                int item_width = iCont->widths[entity.index];
                int item_height = iCont->heights[entity.index];

                // Check for AABB collision
                if (player_x < item_x + item_width &&
                    player_x + player_width > item_x &&
                    player_y < item_y + item_height &&
                    player_y + player_height > item_y) {

                    // Collect the item
                    game_state->score += iCont->values[entity.index];
                    game_state->items_collected++;

                    // Mark item for removal instead of removing immediately
                    EntityRef item_ref = { ENTITY_TYPE_ITEM, entity.index };
                    items_to_remove.push_back(item_ref);

                    // Mark the item as invisible immediately
                    iCont->flags[entity.index] &= ~static_cast<uint8_t>(EntityFlag::VISIBLE);
                }
            }
        }

        // Process items to remove after iteration is complete
        for (const auto& item_ref : items_to_remove) {
            // Add to engine's pending removals
            engine->pending_removals.push_back(item_ref);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in collision detection: " << e.what() << std::endl;
    }
}

void reset_game(Engine* engine, GameState* game_state) {
    std::cout << "Resetting game..." << std::endl;

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
        // Set camera to player
        float player_x = game_state->player_container->x_positions[game_state->player_entity_index];
        float player_y = game_state->player_container->y_positions[game_state->player_entity_index];
        engine->camera.x = player_x + PLAYER_SIZE / 2.0f;
        engine->camera.y = player_y + PLAYER_SIZE / 2.0f;

        // Force visibility flag
        game_state->player_container->flags[game_state->player_entity_index] |=
            static_cast<uint8_t>(EntityFlag::VISIBLE);
    }
    else {
        engine->camera.x = WORLD_WIDTH / 2.0f;
        engine->camera.y = WORLD_HEIGHT / 2.0;
    }
}

