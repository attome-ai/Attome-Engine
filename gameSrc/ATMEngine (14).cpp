#include "ATMEngine.h"
#include <future>

// EntityContainer implementation
EntityContainer::EntityContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
    : type_id(typeId)
    , default_layer(defaultLayer)
    , capacity(initialCapacity)
    , count(0), containerFlag((uint8_t)ContainerFlag::UPDATEABLE)
{
    // Allocate base arrays
    flags = new uint8_t[capacity];
    x_positions = new int32_t[capacity];  // Changed from uint32_t to int32_t
    y_positions = new int32_t[capacity];  // Changed from uint32_t to int32_t
}

EntityContainer::~EntityContainer() {
    delete[] x_positions;
    delete[] y_positions;
    delete[] flags;
}

uint32_t EntityContainer::createEntity() {
    if (count >= capacity) {
        // Resize arrays to accommodate more entities
        int newCapacity = capacity * 2;
        resizeArrays(newCapacity);
    }

    size_t index = count++;
    x_positions[index] = 0;  // Changed to fixed-point 0
    y_positions[index] = 0;  // Changed to fixed-point 0
    flags[index] = static_cast<uint8_t>(EntityFlag::NONE);

    return index;
}

void EntityContainer::removeEntity(size_t index) {
    PROFILE_FUNCTION();
    if (index >= count) return;

    // Move last entity to removed position
    size_t last = count - 1;
    if (index < last) {
        x_positions[index] = x_positions[last];
        y_positions[index] = y_positions[last];
        flags[index] = flags[last];
    }

    count--;
}

void EntityContainer::resizeArrays(int newCapacity) {
    PROFILE_FUNCTION();
    if (newCapacity <= capacity) return;

    // Create new arrays
    uint8_t* newFlags = new uint8_t[newCapacity];
    uint32_t* newEntityIds = new uint32_t[newCapacity];
    uint32_t* newParentIds = new uint32_t[newCapacity];
    uint32_t* newFirstChildIds = new uint32_t[newCapacity];
    uint32_t* newNextSiblingIds = new uint32_t[newCapacity];
    int32_t* newXPositions = new int32_t[newCapacity];  // Changed from uint32_t to int32_t
    int32_t* newYPositions = new int32_t[newCapacity];  // Changed from uint32_t to int32_t

    // Copy existing data
    for (int i = 0; i < count; i++) {
        newXPositions[i] = x_positions[i];
        newYPositions[i] = y_positions[i];
        newFlags[i] = flags[i];
    }

    // Delete old arrays
    delete[] flags;
    delete[] x_positions;  // Add this line
    delete[] y_positions;  // Add this line

    // Assign new arrays
    x_positions = newXPositions;
    y_positions = newYPositions;
    flags = newFlags;

    // Update capacity
    capacity = newCapacity;
}

// RenderableEntityContainer implementation
RenderableEntityContainer::RenderableEntityContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
    : EntityContainer(typeId, defaultLayer, initialCapacity)
{
    // Allocate renderable arrays
    widths = new int16_t[capacity];
    heights = new int16_t[capacity];
    texture_ids = new int16_t[capacity];
    z_indices = new uint8_t[capacity];
    this->containerFlag |= (uint8_t)ContainerFlag::RENDERABLE;
}

RenderableEntityContainer::~RenderableEntityContainer() {
    PROFILE_FUNCTION();
    delete[] widths;
    delete[] heights;
    delete[] texture_ids;
    delete[] z_indices;
}

uint32_t RenderableEntityContainer::createEntity() {
    uint32_t index = EntityContainer::createEntity();
    if (index == INVALID_ID) return INVALID_ID;

    widths[index] = 0;
    heights[index] = 0;
    texture_ids[index] = 0;
    z_indices[index] = 0;
    return index;
}

void RenderableEntityContainer::removeEntity(size_t index) {
    PROFILE_FUNCTION();
    if (index >= count) return;

    size_t last = count - 1;
    if (index < last) {
        widths[index] = widths[last];
        heights[index] = heights[last];
        texture_ids[index] = texture_ids[last];
        z_indices[index] = z_indices[last];
    }

    EntityContainer::removeEntity(index);
}

void RenderableEntityContainer::resizeArrays(int newCapacity) {
    PROFILE_FUNCTION();
    if (newCapacity <= capacity) return;

    // Resize base class arrays first
    EntityContainer::resizeArrays(newCapacity);

    int16_t* newWidths = new int16_t[newCapacity];
    int16_t* newHeights = new int16_t[newCapacity];
    int16_t* newTextureIds = new int16_t[newCapacity];
    uint8_t* newZIndices = new uint8_t[newCapacity];

    // Copy existing data
    for (int i = 0; i < count; i++) {
        newWidths[i] = widths[i];
        newHeights[i] = heights[i];
        newTextureIds[i] = texture_ids[i];
        newZIndices[i] = z_indices[i];
    }

    // Delete old arrays
    delete[] widths;
    delete[] heights;
    delete[] texture_ids;
    delete[] z_indices;

    // Assign new arrays
    widths = newWidths;
    heights = newHeights;
    texture_ids = newTextureIds;
    z_indices = newZIndices;
}

// Layer implementation
Layer::Layer(int id) : layer_id(id), is_active(true) {
    PROFILE_FUNCTION();
}

void Layer::update(uint32_t delta_time) {  // Changed from float to uint32_t
    PROFILE_FUNCTION();
    if (!is_active) return;
    for (auto container : entity_containers) {
        if (container->containerFlag & (uint8_t)ContainerFlag::UPDATEABLE)
            container->update(delta_time);
    }
}

void Layer::addEntityContainer(EntityContainer* container) {
    PROFILE_FUNCTION();
    entity_containers.push_back(container);
}

// EntityManager implementation
EntityManager::EntityManager() : next_entity_id(0) {
    PROFILE_FUNCTION();
    layers.push_back(std::make_unique<Layer>(0));
}

int EntityManager::registerEntityType(EntityContainer* container) {
    PROFILE_FUNCTION();
    int type_id = containers.size();
    containers.emplace_back(container);

    uint8_t layer_index = container->getDefaultLayer();
    if (layer_index >= layers.size()) {
        layers.resize(layer_index + 1);
    }
    if (!layers[layer_index]) {
        layers[layer_index] = std::make_unique<Layer>(layer_index);
    }
    layers[layer_index]->addEntityContainer(container);

    return type_id;
}

uint32_t EntityManager::createEntity(int type_id) {
    PROFILE_FUNCTION();
    if (type_id >= containers.size()) return INVALID_ID;
    uint32_t index = containers[type_id]->createEntity();
    if (index != INVALID_ID) {
        // Could store a mapping from index to next_entity_id if needed
        next_entity_id++;
    }
    return index;
}

void EntityManager::removeEntity(uint32_t index, int type_id) {
    PROFILE_FUNCTION();
    if (type_id >= containers.size()) return;
    containers[type_id]->removeEntity(index);
}

void EntityManager::update(uint32_t delta_time) {  // Changed from float to uint32_t
    PROFILE_FUNCTION();
    for (auto& layer : layers) {
        if (layer) layer->update(delta_time);
    }
}

// Engine implementation
Engine::Engine(int window_width, int window_height, int world_width, int world_height)
    : atlas(nullptr, 2048, 2048)  // Temporarily set renderer to nullptr, will set after creation
{
    PROFILE_FUNCTION();

    // Create window
    window = SDL_CreateWindow("2D Game Engine - Vulkan", window_width, window_height, SDL_WINDOW_VULKAN);
    if (!window) {
        throw std::runtime_error("Failed to create SDL window");
    }

    // Create Vulkan renderer
    vulkanRenderer = new VulkanRenderer();
    if (!vulkanRenderer->initialize(window, window_width, window_height)) {
        SDL_DestroyWindow(window);
        delete vulkanRenderer;
        throw std::runtime_error("Failed to initialize Vulkan renderer");
    }

    // Now properly initialize TextureAtlas with the renderer
    atlas.setRenderer(vulkanRenderer);

    // Init world bounds
    world_bounds.x = 0;
    world_bounds.y = 0;
    world_bounds.w = world_width;
    world_bounds.h = world_height;

    // Init camera with fixed-point positions
    camera.x = 0;
    camera.y = 0;
    camera.width = window_width * POSITION_UNIT;
    camera.height = window_height * POSITION_UNIT;
    camera.zoom = 1.0f;

    // Init timing
    last_frame_time = SDL_GetTicks();
    fps = 0.0f;
}

Engine::~Engine() {
    PROFILE_FUNCTION();

    // Clean up resources
    vulkanRenderer->cleanup();
    delete vulkanRenderer;
    SDL_DestroyWindow(window);
}

int Engine::registerTexture(SDL_Surface* surface, int x, int y, int width, int height) {
    PROFILE_FUNCTION();
    return atlas.registerTexture(surface, x, y, width, height);
}

void Engine::renderScene() {
    PROFILE_FUNCTION();

    // Begin vulkan frame
    vulkanRenderer->beginFrame();

    // Calculate camera bounds in fixed-point
    const int32_t cx1 = camera.x - camera.width / 2;
    const int32_t cy1 = camera.y - camera.height / 2;
    const int32_t cx2 = camera.x + camera.width / 2;
    const int32_t cy2 = camera.y + camera.height / 2;

    // Add some margin for culling to prevent pop-in (100 units in fixed-point)
    const int32_t margin = toGameUnit(100);

    // Query the spatial grid for entities in the visible area
    std::vector<EntityRef>& visible_entities = grid.queryRect(
        cx1 - margin, cy1 - margin, cx2 + margin, cy2 + margin);

    // Sort entities by Z-index for correct rendering order
    std::sort(visible_entities.begin(), visible_entities.end(),
        [this](const EntityRef& a, const EntityRef& b) {
            // Skip non-renderable entities
            auto containerA = entityManager.containers[a.type].get();
            auto containerB = entityManager.containers[b.type].get();

            bool renderableA = containerA->containerFlag & (uint8_t)ContainerFlag::RENDERABLE;
            bool renderableB = containerB->containerFlag & (uint8_t)ContainerFlag::RENDERABLE;

            // Non-renderable entities go to the end
            if (!renderableA && renderableB) return false;
            if (renderableA && !renderableB) return true;
            if (!renderableA && !renderableB) return false;

            // Both are renderable, cast to RenderableEntityContainer
            RenderableEntityContainer* rContainerA = reinterpret_cast<RenderableEntityContainer*>(containerA);
            RenderableEntityContainer* rContainerB = reinterpret_cast<RenderableEntityContainer*>(containerB);

            // Compare z-indices first
            uint8_t zIndexA = rContainerA->z_indices[a.index];
            uint8_t zIndexB = rContainerB->z_indices[b.index];

            if (zIndexA != zIndexB) return zIndexA < zIndexB;

            // If z-indices are equal, compare types
            if (a.type != b.type) return a.type < b.type;

            // If types are equal, compare indices
            return a.index < b.index;
        });

    // Track the current texture and z-index
    int currentTextureId = -1;
    int currentZIndex = -1;

    // Process only entities returned by the grid query
    for (const EntityRef& entity : visible_entities) {
        auto container = entityManager.containers[entity.type].get();
        if (!(container->containerFlag & (uint8_t)ContainerFlag::RENDERABLE))
            continue;

        RenderableEntityContainer* renderableContainer =
            static_cast<RenderableEntityContainer*>(container);

        // Convert fixed-point positions and dimensions to float for rendering
        const int32_t screen_x = renderableContainer->x_positions[entity.index] - cx1;
        const int32_t screen_y = renderableContainer->y_positions[entity.index] - cy1;
        const int screen_w = renderableContainer->widths[entity.index];
        const int screen_h = renderableContainer->heights[entity.index];

        // Skip if outside camera view (final precise culling)
        if (screen_x + (int32_t)toGameUnit(screen_w) < 0 || screen_x > cx2 ||
            screen_y + (int32_t)toGameUnit(screen_h) < 0 || screen_y > cy2)
            continue;

        int textureId = renderableContainer->texture_ids[entity.index];
        int zIndex = renderableContainer->z_indices[entity.index];

        // If we encounter a new texture or z-index, create a new batch
        if (textureId != currentTextureId || zIndex != currentZIndex) {
            // Tell the renderer to switch to this texture
            vulkanRenderer->beginBatch(textureId);
            currentTextureId = textureId;
            currentZIndex = zIndex;
        }

        // Convert screen coordinates to normalized device coordinates (-1 to 1)
        float screenWidth = fixedPointToFloat(camera.width);
        float screenHeight = fixedPointToFloat(camera.height);

        float ndcX = (fixedPointToFloat(screen_x) / screenWidth) * 2.0f - 1.0f;
        float ndcY = (fixedPointToFloat(screen_y) / screenHeight) * 2.0f - 1.0f;
        float ndcWidth = (screen_w / screenWidth) * 2.0f;
        float ndcHeight = (screen_h / screenHeight) * 2.0f;

        // Get texture region
        uint16_t tx1, ty1, tx2, ty2;
        atlas.getRegion(textureId, tx1, ty1, tx2, ty2);

        // Create quad vertices
        Vertex topLeft = {
            {ndcX, ndcY},
            {tx1, ty1},
            {1.0f, 1.0f, 1.0f, 1.0f}
        };

        Vertex topRight = {
            {ndcX + ndcWidth, ndcY},
            {tx2, ty1},
            {1.0f, 1.0f, 1.0f, 1.0f}
        };

        Vertex bottomRight = {
            {ndcX + ndcWidth, ndcY + ndcHeight},
            {tx2, ty2},
            {1.0f, 1.0f, 1.0f, 1.0f}
        };

        Vertex bottomLeft = {
            {ndcX, ndcY + ndcHeight},
            {tx1, ty2},
            {1.0f, 1.0f, 1.0f, 1.0f}
        };

        // Add the quad to the current batch
        vulkanRenderer->addQuadToBatch(topLeft, topRight, bottomRight, bottomLeft);
    }

    // The renderer will flush any remaining batches in endFrame
    vulkanRenderer->endFrame();
}
void Engine::processPendingRemovals() {
    PROFILE_FUNCTION();

    if (pending_removals.empty()) return;

    for (const auto& ref : pending_removals) {
        if (ref.type < 0 || ref.type >= entityManager.containers.size()) continue;

        auto container = entityManager.containers[ref.type].get();
        if (!container) continue;

        container->removeEntity(ref.index);
    }

    pending_removals.clear();
}

void Engine::update() {
    PROFILE_FUNCTION();

    // Calculate delta time
    uint64_t current_time = SDL_GetTicks();
    uint32_t delta_time = static_cast<uint32_t>(current_time - last_frame_time);  // Changed from float to uint32_t
    last_frame_time = current_time;

    grid.rebuild_grid(this);

    // Update all entity layers
    entityManager.update(delta_time);

    // Process pending removals
    processPendingRemovals();
}

void Engine::setEntityZIndex(uint32_t entity_idx, int type_id, uint8_t z_index) {
    PROFILE_FUNCTION();

    if (type_id >= entityManager.containers.size()) return;

    auto container = entityManager.containers[type_id].get();
    if (!container) return;

    RenderableEntityContainer* renderable = dynamic_cast<RenderableEntityContainer*>(container);
    if (!renderable || entity_idx >= renderable->getCount()) return;

    renderable->z_indices[entity_idx] = z_index;
}

void Engine::present() {
    PROFILE_FUNCTION();
    // Nothing needed here since endFrame is called in renderScene
}

void SpatialGrid::rebuild_grid(Engine* engine) {
    PROFILE_FUNCTION();
    clearAll();
    constexpr uint32_t CHUNK_SIZE = 25000; // Adjust based on your needs
    std::vector<std::future<void>> futures;

    // Process each container
    for (uint32_t containerIdx = 0; containerIdx < engine->entityManager.containers.size(); containerIdx++) {
        auto& container = engine->entityManager.containers[containerIdx];
        uint32_t entityCount = container->count;

        // If container has enough entities, split into chunks for parallel processing
        if (entityCount > CHUNK_SIZE) {
            uint32_t numChunks = (entityCount + CHUNK_SIZE - 1) / CHUNK_SIZE;

            for (uint32_t chunk = 0; chunk < numChunks; chunk++) {
                uint32_t startIdx = chunk * CHUNK_SIZE;
                uint32_t endIdx = std::min(startIdx + CHUNK_SIZE, entityCount);

                // Launch async task for each chunk
                futures.push_back(std::async(std::launch::async,
                    [this, containerIdx, startIdx, endIdx, x_positions = container->x_positions,
                    y_positions = container->y_positions]() {
                        for (uint32_t i = startIdx; i < endIdx; i++) {
                            // Calculate grid cell coordinates using fixed-point positions
                            uint16_t cellX = x_positions[i] / GRID_CELL_SIZE;
                            uint16_t cellY = y_positions[i] / GRID_CELL_SIZE;

                            // Call add2 which now handles atomic operations
                            add2({ containerIdx, i }, cellX, cellY);
                        }
                    }));
            }
        }
        else {
            // Process small containers directly in this thread
            for (uint32_t i = 0; i < entityCount; i++) {
                uint16_t cellX = container->x_positions[i] / GRID_CELL_SIZE;
                uint16_t cellY = container->y_positions[i] / GRID_CELL_SIZE;
                add2({ containerIdx, i }, cellX, cellY);
            }
        }
    }

    // Wait for all async tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    futures.clear();

    for (int i = 0; i < cells.size(); i++) {
        for (int j = 0; j < cells[i].size(); j++) {
            cells[i][j].count = std::min(cells[i][j].count.operator int(), MAX_ENTITIES_PER_CELL);
        }
    }
}