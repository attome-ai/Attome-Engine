#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3_image/SDL_image.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <random>

#include "ATMEngine.h" // Include the new high-performance engine header

// Game constants
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1020
#define WORLD_WIDTH 50000.0f
#define WORLD_HEIGHT 10000.0f
#define GRID_CELL_SIZE 512.0f

#define PLAYER_SIZE 32.0f
#define OBSTACLE_SIZE 32.0f
#define NUM_OBSTACLES 1000000

// Physics layers
#define LAYER_DEFAULT 0
#define LAYER_PLAYER 1
#define LAYER_OBSTACLE 2

// Component structs for entity type handlers
struct ALIGN_TO_CACHE PlayerComponent {
    float speed;
    int currentFrame;
    int textureIds[3];
    uint64_t animationTimer;
    bool keysPressed[4];  // W, A, S, D
};

struct ALIGN_TO_CACHE ObstacleComponent {
    float speed;
    glm::vec3 direction;
    float bounceTimer;
};

// Game state
struct GameState {
    Engine* engine;
    int playerEntity;
    int score;
    bool gameOver;
    uint64_t lastSpawnTime;
    int obstacleTextureIds[4];

    // Handler registrations
    int playerHandlerId;
    int obstacleHandlerId;

    // Material IDs
    int playerMaterial;
    int obstacleMaterial;
};

// Forward declarations
void InitializeGame(GameState* state);
void UpdateGame(GameState* state, float deltaTime);
void CleanupGame(GameState* state);
void InitializeObstacles(GameState* state);
void HandleInput(GameState* state, const SDL_Event& event);
bool ProcessCollisions(GameState* state);
void ProcessPlayerMovement(GameState* state, float deltaTime);

// Entity type update functions
void  PlayerUpdateFunction(void* data, size_t count, float deltaTime);
void  ObstacleUpdateFunction(void* data, size_t count, float deltaTime);

// Create a solid color surface with optional alpha
SDL_Surface* CreateSolidColorSurface(int width, int height, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    SDL_Surface* surface = SDL_CreateSurface(width, height, SDL_PIXELFORMAT_RGBA32);
    if (!surface) {
        printf("Failed to create surface: %s\n", SDL_GetError());
        return nullptr;
    }

    SDL_FillSurfaceRect(surface, nullptr, SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format),0, r, g, b, a));
    return surface;
}

// Player update function - SIMD optimized
void  PlayerUpdateFunction(void* data, size_t count, float deltaTime) {
    // This function processes batches of player entities
    EntityChunk* chunk = static_cast<EntityChunk*>(data);

    // Process SIMD_WIDTH entities at a time where possible
    int fullBatches = count / SIMD_WIDTH;

    for (int batchIdx = 0; batchIdx < fullBatches; batchIdx++) {
        int startIdx = batchIdx * SIMD_WIDTH;

        // Prefetch next batch
        if (batchIdx + 1 < fullBatches) {
            PREFETCH(&chunk->hot.position[startIdx + SIMD_WIDTH]);
        }

        // Process entities using SIMD where applicable
        for (int i = 0; i < SIMD_WIDTH; i++) {
            int idx = startIdx + i;

            // Skip inactive entities
            if (!chunk->hot.active[idx]) continue;

            // Get component data
            PlayerComponent* player = (PlayerComponent*) (
                static_cast<uint8_t*>(chunk->type_data) + idx * chunk->type_data_stride
                );

            // Get current time
            uint64_t currentTime = SDL_GetTicks();

            // Animate sprite
            if (currentTime - player->animationTimer > 150) {
                player->currentFrame = (player->currentFrame + 1) % 3;
                // Update texture would happen in rendering
                player->animationTimer = currentTime;
            }

            // Calculate movement
            glm::vec3 moveDir(0.0f);

            if (player->keysPressed[0]) moveDir.z -= 1.0f; // W - forward
            if (player->keysPressed[1]) moveDir.x -= 1.0f; // A - left
            if (player->keysPressed[2]) moveDir.z += 1.0f; // S - back
            if (player->keysPressed[3]) moveDir.x += 1.0f; // D - right

            // Normalize if needed
            if (glm::length(moveDir) > 0.001f) {
                moveDir = glm::normalize(moveDir);
            }

            // Apply movement
            glm::vec3 newPos = chunk->hot.position[idx] + moveDir * player->speed * deltaTime;

            // Clamp position to world bounds
            newPos.x = glm::clamp(newPos.x, 0.0f, WORLD_WIDTH - PLAYER_SIZE);
            newPos.z = glm::clamp(newPos.z, 0.0f, WORLD_HEIGHT - PLAYER_SIZE);

            // Update position
            chunk->hot.position[idx] = newPos;
        }
    }

    // Handle remaining entities
    for (int i = fullBatches * SIMD_WIDTH; i < count; i++) {
        // Skip inactive entities
        if (!chunk->hot.active[i]) continue;

        // Get component data
        PlayerComponent* player = (PlayerComponent*)(
            static_cast<uint8_t*>(chunk->type_data) + i * chunk->type_data_stride
            );

        // Get current time
        uint64_t currentTime = SDL_GetTicks();

        // Animate sprite
        if (currentTime - player->animationTimer > 150) {
            player->currentFrame = (player->currentFrame + 1) % 3;
            // Update texture would happen in rendering
            player->animationTimer = currentTime;
        }

        // Calculate movement
        glm::vec3 moveDir(0.0f);

        if (player->keysPressed[0]) moveDir.z -= 1.0f; // W - forward
        if (player->keysPressed[1]) moveDir.x -= 1.0f; // A - left
        if (player->keysPressed[2]) moveDir.z += 1.0f; // S - back
        if (player->keysPressed[3]) moveDir.x += 1.0f; // D - right

        // Normalize if needed
        if (glm::length(moveDir) > 0.001f) {
            moveDir = glm::normalize(moveDir);
        }

        // Apply movement
        glm::vec3 newPos = chunk->hot.position[i] + moveDir * player->speed * deltaTime;

        // Clamp position to world bounds
        newPos.x = glm::clamp(newPos.x, 0.0f, WORLD_WIDTH - PLAYER_SIZE);
        newPos.z = glm::clamp(newPos.z, 0.0f, WORLD_HEIGHT - PLAYER_SIZE);

        // Update position
        chunk->hot.position[i] = newPos;
    }
}

// Obstacle update function - SIMD optimized
void  ObstacleUpdateFunction(void* data, size_t count, float deltaTime) {
    EntityChunk* chunk = static_cast<EntityChunk*>(data);

    // Skip empty chunks
    if (count == 0) return;

    // Process in cache-friendly batches with SIMD where possible
    int fullBatches = count / SIMD_WIDTH;

    for (int batchIdx = 0; batchIdx < fullBatches; batchIdx++) {
        int startIdx = batchIdx * SIMD_WIDTH;

        // Prefetch next batch data
        if (batchIdx + 1 < fullBatches) {
            PREFETCH(&chunk->hot.position[startIdx + SIMD_WIDTH]);
            PREFETCH(static_cast<uint8_t*>(chunk->type_data) + (startIdx + SIMD_WIDTH) * chunk->type_data_stride);
        }

        // Prepare SIMD arrays for position updates
        SimdFloat posX, posZ, dirX, dirZ, speed;

        // Process batch
        for (int i = 0; i < SIMD_WIDTH; i++) {
            int idx = startIdx + i;

            // Skip inactive entities
            if (!chunk->hot.active[idx]) continue;

            // Get obstacle data
            ObstacleComponent* obstacle = (ObstacleComponent*)(
                static_cast<uint8_t*>(chunk->type_data) + idx * chunk->type_data_stride
                );

            // Calculate movement
            glm::vec3 newPos = chunk->hot.position[idx] + obstacle->direction * obstacle->speed * deltaTime;

            // Bounce off world boundaries
            bool bounceX = false;
            bool bounceZ = false;

            if (newPos.x < 0.0f || newPos.x > WORLD_WIDTH - OBSTACLE_SIZE) {
                obstacle->direction.x *= -1.0f;
                bounceX = true;
            }

            if (newPos.z < 0.0f || newPos.z > WORLD_HEIGHT - OBSTACLE_SIZE) {
                obstacle->direction.z *= -1.0f;
                bounceZ = true;
            }

            // Apply corrected position after bounce
            if (bounceX) {
                newPos.x = glm::clamp(newPos.x, 0.0f, WORLD_WIDTH - OBSTACLE_SIZE);
            }

            if (bounceZ) {
                newPos.z = glm::clamp(newPos.z, 0.0f, WORLD_HEIGHT - OBSTACLE_SIZE);
            }

            // Update position
            chunk->hot.position[idx] = newPos;
        }
    }

    // Handle remaining obstacles
    for (int i = fullBatches * SIMD_WIDTH; i < count; i++) {
        // Skip inactive entities
        if (!chunk->hot.active[i]) continue;

        // Get obstacle data
        ObstacleComponent* obstacle = (ObstacleComponent*)(
            static_cast<uint8_t*>(chunk->type_data) + i * chunk->type_data_stride
            );

        // Calculate movement
        glm::vec3 newPos = chunk->hot.position[i] + obstacle->direction * obstacle->speed * deltaTime;

        // Bounce off world boundaries
        bool bounceX = false;
        bool bounceZ = false;

        if (newPos.x < 0.0f || newPos.x > WORLD_WIDTH - OBSTACLE_SIZE) {
            obstacle->direction.x *= -1.0f;
            bounceX = true;
        }

        if (newPos.z < 0.0f || newPos.z > WORLD_HEIGHT - OBSTACLE_SIZE) {
            obstacle->direction.z *= -1.0f;
            bounceZ = true;
        }

        // Apply corrected position after bounce
        if (bounceX) {
            newPos.x = glm::clamp(newPos.x, 0.0f, WORLD_WIDTH - OBSTACLE_SIZE);
        }

        if (bounceZ) {
            newPos.z = glm::clamp(newPos.z, 0.0f, WORLD_HEIGHT - OBSTACLE_SIZE);
        }

        // Update position
        chunk->hot.position[i] = newPos;
    }
}

// Create entity type handler for player entities
EntityTypeHandler* CreatePlayerTypeHandler() {
    EntityTypeHandler* handler = new EntityTypeHandler();
    handler->type_id = 1; // Player type ID
    handler->update_func = PlayerUpdateFunction;
    return handler;
}

// Create entity type handler for obstacle entities
EntityTypeHandler* CreateObstacleTypeHandler() {
    EntityTypeHandler* handler = new EntityTypeHandler();
    handler->type_id = 2; // Obstacle type ID
    handler->update_func = ObstacleUpdateFunction;
    return handler;
}

// Initialize the game
void InitializeGame(GameState* state) {
    // Create engine with optimized grid cell size
    state->engine = new Engine(
        WINDOW_WIDTH, WINDOW_HEIGHT,
        WORLD_WIDTH, WORLD_HEIGHT, WORLD_HEIGHT, // World dimensions in 3D
        GRID_CELL_SIZE,
        NUM_OBSTACLES + 1 // Total entity count (obstacles + player)
    );

    // Register entity type handlers
    EntityTypeHandler* playerHandler = CreatePlayerTypeHandler();
    EntityTypeHandler* obstacleHandler = CreateObstacleTypeHandler();

    state->playerHandlerId = state->engine->entities.registerEntityType(playerHandler);
    state->obstacleHandlerId = state->engine->entities.registerEntityType(obstacleHandler);

    // Configure physics collision layers
    state->engine->registerPhysicsLayer(LAYER_PLAYER, (1 << LAYER_OBSTACLE)); // Player collides with obstacles
    state->engine->registerPhysicsLayer(LAYER_OBSTACLE, (1 << LAYER_PLAYER)); // Obstacles collide with player

    // Create materials and textures
    SDL_Surface* playerSurface1 = CreateSolidColorSurface(PLAYER_SIZE, PLAYER_SIZE, 0, 0, 255);
    SDL_Surface* playerSurface2 = CreateSolidColorSurface(PLAYER_SIZE, PLAYER_SIZE, 0, 100, 255);
    SDL_Surface* playerSurface3 = CreateSolidColorSurface(PLAYER_SIZE, PLAYER_SIZE, 100, 100, 255);

    SDL_Surface* obstacleSurface1 = CreateSolidColorSurface(OBSTACLE_SIZE, OBSTACLE_SIZE, 255, 0, 0);
    SDL_Surface* obstacleSurface2 = CreateSolidColorSurface(OBSTACLE_SIZE, OBSTACLE_SIZE, 255, 50, 0);
    SDL_Surface* obstacleSurface3 = CreateSolidColorSurface(OBSTACLE_SIZE, OBSTACLE_SIZE, 200, 0, 0);
    SDL_Surface* obstacleSurface4 = CreateSolidColorSurface(OBSTACLE_SIZE, OBSTACLE_SIZE, 255, 100, 100);

    // Add textures to engine
    int playerTexture1 = state->engine->addTexture( state->engine->renderer, playerSurface1);
    int playerTexture2 = state->engine->addTexture(state->engine->renderer, playerSurface2);
    int playerTexture3 = state->engine->addTexture(state->engine->renderer, playerSurface3);

    int obstacleTexture1 = state->engine->addTexture(state->engine->renderer, obstacleSurface1);
    int obstacleTexture2 = state->engine->addTexture(state->engine->renderer, obstacleSurface2);
    int obstacleTexture3 = state->engine->addTexture(state->engine->renderer, obstacleSurface3);
    int obstacleTexture4 = state->engine->addTexture(state->engine->renderer,  obstacleSurface4);

    // Store obstacle texture IDs
    state->obstacleTextureIds[0] = obstacleTexture1;
    state->obstacleTextureIds[1] = obstacleTexture2;
    state->obstacleTextureIds[2] = obstacleTexture3;
    state->obstacleTextureIds[3] = obstacleTexture4;

    // Create materials
    state->playerMaterial = state->engine->addMaterial(
        playerTexture1, -1, -1,
        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
        glm::vec4(0.0f), 0.0f
    );

    state->obstacleMaterial = state->engine->addMaterial(
        obstacleTexture1, -1, -1,
        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
        glm::vec4(0.0f), 0.0f
    );

    // Clean up surfaces
    SDL_DestroySurface(playerSurface1);
    SDL_DestroySurface(playerSurface2);
    SDL_DestroySurface(playerSurface3);
    SDL_DestroySurface(obstacleSurface1);
    SDL_DestroySurface(obstacleSurface2);
    SDL_DestroySurface(obstacleSurface3);
    SDL_DestroySurface(obstacleSurface4);

    // Create player entity
    glm::vec3 playerPos(WINDOW_WIDTH / 2.0f, 0.0f, WINDOW_HEIGHT / 2.0f);
    glm::vec3 playerScale(PLAYER_SIZE, PLAYER_SIZE, PLAYER_SIZE);
    state->playerEntity = state->engine->addEntity(
        state->playerHandlerId,
        playerPos,
        playerScale,
        -1, // No mesh, 2D game
        state->playerMaterial
    );

    // Set player physics layer
    state->engine->setEntityPhysicsLayer(state->playerEntity, LAYER_PLAYER);

    // Initialize player component data
    PlayerComponent* playerComp = static_cast<PlayerComponent*>(
        state->engine->entities.getEntityTypeData(state->playerEntity)
        );

    if (playerComp) {
        playerComp->speed = 200.0f;
        playerComp->currentFrame = 0;
        playerComp->animationTimer = SDL_GetTicks();
        playerComp->textureIds[0] = playerTexture1;
        playerComp->textureIds[1] = playerTexture2;
        playerComp->textureIds[2] = playerTexture3;

        // Clear key states
        for (int i = 0; i < 4; i++) {
            playerComp->keysPressed[i] = false;
        }
    }

    // Initialize obstacles
    InitializeObstacles(state);

    // Initialize game state
    state->score = 0;
    state->gameOver = false;
    state->lastSpawnTime = SDL_GetTicks();

    // Set up camera
    state->engine->setCameraPosition(glm::vec3(WINDOW_WIDTH / 2.0f, 1000.0f, WINDOW_HEIGHT / 2.0f));
    state->engine->setCameraTarget(glm::vec3(WINDOW_WIDTH / 2.0f, 0.0f, WINDOW_HEIGHT / 2.0f));
    state->engine->setCameraUp(glm::vec3(0.0f, 0.0f, -1.0f)); // Z-up for 2D-like view

    // Enable engine optimizations
    state->engine->optimizeMemoryLayout();
}

void InitializeObstacles(GameState* state) {
    // Use a modern random number generator for better quality
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> xDist(0.0f, WORLD_WIDTH - OBSTACLE_SIZE);
    std::uniform_real_distribution<float> zDist(0.0f, WORLD_HEIGHT - OBSTACLE_SIZE);
    std::uniform_real_distribution<float> speedDist(50.0f, 150.0f);
    std::uniform_real_distribution<float> dirDist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> textureDist(0, 3);

    // Pre-allocate obstacles in cache-friendly batches
    for (int batchStart = 0; batchStart < NUM_OBSTACLES; batchStart += ENTITY_BATCH_SIZE) {
        int batchEnd = std::min(batchStart + ENTITY_BATCH_SIZE, NUM_OBSTACLES);
        int batchSize = batchEnd - batchStart;

        // Batch allocate AABB structures for spatial partitioning
        AABB* aabbBatch = static_cast<AABB*>(SDL_aligned_alloc(CACHE_LINE_SIZE, batchSize * sizeof(AABB)));
        glm::vec3* posBatch = static_cast<glm::vec3*>(SDL_aligned_alloc(CACHE_LINE_SIZE, batchSize * sizeof(glm::vec3)));
        glm::vec3* scaleBatch = static_cast<glm::vec3*>(SDL_aligned_alloc(CACHE_LINE_SIZE, batchSize * sizeof(glm::vec3)));
        int* materialBatch = static_cast<int*>(SDL_aligned_alloc(CACHE_LINE_SIZE, batchSize * sizeof(int)));

        // Fill batch arrays
        for (int i = 0; i < batchSize; i++) {
            // Generate random position
            float x = xDist(gen);
            float z = zDist(gen);

            // Ensure obstacles are not too close to the player's starting position
            if (abs(x - WINDOW_WIDTH / 2.0f) < 200.0f && abs(z - WINDOW_HEIGHT / 2.0f) < 200.0f) {
                x += (x < WINDOW_WIDTH / 2.0f) ? -200.0f : 200.0f;
                z += (z < WINDOW_HEIGHT / 2.0f) ? -200.0f : 200.0f;
            }

            // Store position and scale for batch creation
            posBatch[i] = glm::vec3(x, 0.0f, z);
            scaleBatch[i] = glm::vec3(OBSTACLE_SIZE, OBSTACLE_SIZE, OBSTACLE_SIZE);

            // Set obstacle material
            int textureIdx = textureDist(gen);
            materialBatch[i] = state->obstacleMaterial;

            // Compute AABB for spatial grid insertion
            aabbBatch[i].min = posBatch[i] - scaleBatch[i] * 0.5f;
            aabbBatch[i].max = posBatch[i] + scaleBatch[i] * 0.5f;
        }

        // Batch create obstacles for better cache coherence
        int* obstacleIds = new int[batchSize];

        // Add entities in batch
        for (int i = 0; i < batchSize; i++) {
            obstacleIds[i] = state->engine->addEntity(
                state->obstacleHandlerId,
                posBatch[i],
                scaleBatch[i],
                -1, // No mesh, 2D game
                materialBatch[i]
            );

            // Set physics layer
            state->engine->setEntityPhysicsLayer(obstacleIds[i], LAYER_OBSTACLE);

            // Initialize obstacle component data
            ObstacleComponent* obstacle = static_cast<ObstacleComponent*>(
                state->engine->entities.getEntityTypeData(obstacleIds[i])
                );

            if (obstacle) {
                // Random movement speed
                obstacle->speed = speedDist(gen);

                // Normalized direction vector
                glm::vec3 dir(dirDist(gen), 0.0f, dirDist(gen));
                if (glm::length(dir) < 0.001f) {
                    dir = glm::vec3(1.0f, 0.0f, 0.0f);
                }
                else {
                    dir = glm::normalize(dir);
                }

                obstacle->direction = dir;
                obstacle->bounceTimer = 0.0f;
            }
        }

        // Free batch memory
        SDL_aligned_free(aabbBatch);
        SDL_aligned_free(posBatch);
        SDL_aligned_free(scaleBatch);
        SDL_aligned_free(materialBatch);
        delete[] obstacleIds;
    }

    // Optimize spatial partitioning after adding all obstacles
    state->engine->morton_grid->optimizeCellSize();
}

bool ProcessCollisions(GameState* state) {
    // Get player position and create bounding box
    glm::vec3 playerPos = state->engine->entities.getEntityPosition(state->playerEntity);

    // Create player AABB
    AABB playerBox;
    playerBox.min = glm::vec3(playerPos.x, 0.0f, playerPos.z);
    playerBox.max = glm::vec3(playerPos.x + PLAYER_SIZE, PLAYER_SIZE, playerPos.z + PLAYER_SIZE);

    // Query nearby obstacles using morton-ordered grid for cache-friendly access
    const int MAX_NEARBY = 1024;
    int* nearbyEntities = new int[MAX_NEARBY];

    int count = state->engine->queryMortonRegion(
        playerBox.min - glm::vec3(GRID_CELL_SIZE * 0.25f),
        playerBox.max + glm::vec3(GRID_CELL_SIZE * 0.25f),
        nearbyEntities,
        MAX_NEARBY
    );

    // Check collisions in batches
    bool collision = false;
    AABB obstacleBox;

    // Use SIMD optimized collision checking where possible
    for (int i = 0; i < count; i++) {
        int entityId = nearbyEntities[i];

        // Skip player entity
        if (entityId == state->playerEntity) continue;

        // Only check against obstacles
        int chunkIdx, localIdx;
        state->engine->entities.getChunkIndices(entityId, &chunkIdx, &localIdx);

        EntityChunk* chunk = state->engine->entities.chunks[chunkIdx];
        if (chunk->type_id != state->obstacleHandlerId) continue;

        // Get obstacle position
        glm::vec3 obstaclePos = chunk->hot.position[localIdx];

        // Create obstacle AABB
        obstacleBox.min = glm::vec3(obstaclePos.x, 0.0f, obstaclePos.z);
        obstacleBox.max = glm::vec3(obstaclePos.x + OBSTACLE_SIZE, OBSTACLE_SIZE, obstaclePos.z + OBSTACLE_SIZE);

        // Check AABB overlap
        if (playerBox.overlaps(obstacleBox)) {
            collision = true;
            break;
        }
    }

    delete[] nearbyEntities;
    return collision;
}

void ProcessPlayerMovement(GameState* state, float deltaTime) {
    // Get player component data
    PlayerComponent* player = static_cast<PlayerComponent*>(
        state->engine->entities.getEntityTypeData(state->playerEntity)
        );

    if (!player) return;

    // Calculate movement vector
    glm::vec3 moveDir(0.0f);

    if (player->keysPressed[0]) moveDir.z -= 1.0f; // W - forward
    if (player->keysPressed[1]) moveDir.x -= 1.0f; // A - left
    if (player->keysPressed[2]) moveDir.z += 1.0f; // S - back
    if (player->keysPressed[3]) moveDir.x += 1.0f; // D - right

    // Normalize if needed
    if (glm::length(moveDir) > 0.001f) {
        moveDir = glm::normalize(moveDir);

        // Get current position
        glm::vec3 currentPos = state->engine->entities.getEntityPosition(state->playerEntity);

        // Calculate new position
        glm::vec3 newPos = currentPos + moveDir * player->speed * deltaTime;

        // Clamp to world bounds
        newPos.x = glm::clamp(newPos.x, 0.0f, WORLD_WIDTH - PLAYER_SIZE);
        newPos.z = glm::clamp(newPos.z, 0.0f, WORLD_HEIGHT - PLAYER_SIZE);

        // Update position
        state->engine->setEntityPosition(state->playerEntity, newPos);

        // Update camera to follow player
        state->engine->setCameraPosition(glm::vec3(newPos.x + PLAYER_SIZE / 2, 1000.0f, newPos.z + PLAYER_SIZE / 2));
        state->engine->setCameraTarget(glm::vec3(newPos.x + PLAYER_SIZE / 2, 0.0f, newPos.z + PLAYER_SIZE / 2));
    }
}

void UpdateGame(GameState* state, float deltaTime) {
    if (state->gameOver) return;

    // Process input and movement directly
    ProcessPlayerMovement(state, deltaTime);

    // Update engine - will process all entities using registered update functions
    state->engine->update();

    // Check for collisions
    if (ProcessCollisions(state)) {
        state->gameOver = true;
        printf("Game Over! Score: %d\n", state->score);
        return;
    }

    // Increment score
    state->score++;
}

void HandleInput(GameState* state, const SDL_Event& event) {
    // Get player component data
    PlayerComponent* player = static_cast<PlayerComponent*>(
        state->engine->entities.getEntityTypeData(state->playerEntity)
        );

    if (!player) return;

    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
        if (state->gameOver && event.key.scancode == SDL_SCANCODE_R) {
            // Restart game
            state->gameOver = false;
            state->score = 0;

            // Reset player position
            state->engine->setEntityPosition(
                state->playerEntity,
                glm::vec3(WINDOW_WIDTH / 2.0f, 0.0f, WINDOW_HEIGHT / 2.0f)
            );

            // Reset camera
            state->engine->setCameraPosition(glm::vec3(WINDOW_WIDTH / 2.0f, 1000.0f, WINDOW_HEIGHT / 2.0f));
            state->engine->setCameraTarget(glm::vec3(WINDOW_WIDTH / 2.0f, 0.0f, WINDOW_HEIGHT / 2.0f));

            // Reinitialize obstacles
            InitializeObstacles(state);
        }

        // Update key state in player data
        if (!state->gameOver) {
            if (event.key.scancode == SDL_SCANCODE_W) player->keysPressed[0] = true;
            if (event.key.scancode == SDL_SCANCODE_A) player->keysPressed[1] = true;
            if (event.key.scancode == SDL_SCANCODE_S) player->keysPressed[2] = true;
            if (event.key.scancode == SDL_SCANCODE_D) player->keysPressed[3] = true;
        }
        break;

    case SDL_EVENT_KEY_UP:
        // Update key state in player data
        if (!state->gameOver) {
            if (event.key.scancode == SDL_SCANCODE_W) player->keysPressed[0] = false;
            if (event.key.scancode == SDL_SCANCODE_A) player->keysPressed[1] = false;
            if (event.key.scancode == SDL_SCANCODE_S) player->keysPressed[2] = false;
            if (event.key.scancode == SDL_SCANCODE_D) player->keysPressed[3] = false;
        }
        break;

    default:
        break;
    }
}

void CleanupGame(GameState* state) {
    delete state->engine;
}

int main(int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("Could not initialize SDL: %s\n", SDL_GetError());
        return 1;
    }

    // Seed random number generator
    srand(static_cast<unsigned>(time(nullptr)));

    // Create game state
    GameState state;

    // Initialize game
    InitializeGame(&state);

    // Main game loop
    bool running = true;
    uint64_t lastFrameTime = SDL_GetTicks();

    while (running) {
        // Calculate delta time
        uint64_t currentTime = SDL_GetTicks();
        float deltaTime = (currentTime - lastFrameTime) / 1000.0f;
        lastFrameTime = currentTime;

        // Cap delta time to prevent large jumps
        if (deltaTime > 0.1f) deltaTime = 0.1f;

        // Process events
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            }
            else {
                HandleInput(&state, event);
            }
        }

        // Update game
        UpdateGame(&state, deltaTime);

        // Render game
        state.engine->render();

        // Calculate and print FPS
        static uint64_t fpsLastTime = 0;
        static int fpsFrames = 0;

        fpsFrames++;

        if (currentTime - fpsLastTime >= 1000) {
            float fps = fpsFrames * 1000.0f / (currentTime - fpsLastTime);
            printf("FPS: %.2f, Active entities: %d, Score: %d\n",
                fps, state.engine->entities.total_entity_count, state.score);
            fpsLastTime = currentTime;
            fpsFrames = 0;
        }
    }

    // Cleanup
    CleanupGame(&state);
    SDL_Quit();

    return 0;
}