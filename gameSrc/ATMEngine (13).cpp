
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
    PROFILE_FUNCTION();
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

// RenderBatch implementation
RenderBatch::RenderBatch(int textureId, int zIndex, int initialVertexCapacity)
    : texture_id(textureId)
{
    PROFILE_FUNCTION();
    vertices.reserve(initialVertexCapacity);
    indices.reserve(initialVertexCapacity * 1.5);
}

RenderBatch::~RenderBatch() {
}

RenderBatch::RenderBatch(RenderBatch&& other) noexcept
    : texture_id(other.texture_id)
    , vertices(std::move(other.vertices))
    , indices(std::move(other.indices))
{
    PROFILE_FUNCTION();
}

RenderBatch& RenderBatch::operator=(RenderBatch&& other) noexcept {
    PROFILE_FUNCTION();
    if (this != &other) {
        texture_id = other.texture_id;
        vertices = std::move(other.vertices);
        indices = std::move(other.indices);
    }
    return *this;
}

void RenderBatch::addQuad(float x, float y, float w, float h, SDL_FRect tex_region) {
    PROFILE_FUNCTION();


    // Ensure we have enough space
    const uint64_t base_vert = vertices.size();

    vertices.resize(vertices.size() + 4);
    indices.resize(indices.size() + 6);

    // Use direct memory access for better performance
    SDL_Vertex* v = &vertices.data()[vertices.size() - 4];

    // Top-left
    v[0].position.x = x;
    v[0].position.y = y;
    v[0].color.a = 1;
    v[0].color.b = 1;
    v[0].color.r = 1;
    v[0].color.g = 1;
    v[0].tex_coord.x = tex_region.x;
    v[0].tex_coord.y = tex_region.y;

    // Top-right
    v[1].position.x = x + w;
    v[1].position.y = y;
    v[1].color.a = 1;
    v[1].color.b = 1;
    v[1].color.r = 1;
    v[1].color.g = 1;
    v[1].tex_coord.x = tex_region.x + tex_region.w;
    v[1].tex_coord.y = tex_region.y;

    // Bottom-right
    v[2].position.x = x + w;
    v[2].position.y = y + h;
    v[2].color.a = 1;
    v[2].color.b = 1;
    v[2].color.r = 1;
    v[2].color.g = 1;
    v[2].tex_coord.x = tex_region.x + tex_region.w;
    v[2].tex_coord.y = tex_region.y + tex_region.h;

    // Bottom-left
    v[3].position.x = x;
    v[3].position.y = y + h;
    v[3].color.a = 1;
    v[3].color.b = 1;
    v[3].color.r = 1;
    v[3].color.g = 1;
    v[3].tex_coord.x = tex_region.x;
    v[3].tex_coord.y = tex_region.y + tex_region.h;

    // Add indices
    int* idx = &indices.data()[indices.size() - 6];
    idx[0] = base_vert;
    idx[1] = base_vert + 1;
    idx[2] = base_vert + 2;
    idx[3] = base_vert;
    idx[4] = base_vert + 2;
    idx[5] = base_vert + 3;
}

void RenderBatch::clear() {
    PROFILE_FUNCTION();
    vertices.clear();
    indices.clear();
}

// RenderBatchManager implementation
RenderBatchManager::RenderBatchManager(int initialBatchCount) : needsSort(false) {
    batches.reserve(initialBatchCount * 2); // Reserve extra space to minimize reallocations
}

void RenderBatchManager::addQuad(int textureId, int zIndex, float x, float y, float w, float h,
    SDL_FRect tex_region) {
    PROFILE_FUNCTION();
    // Get or create a batch for this texture/z-index combination
    RenderBatch& batch = getBatch(textureId, zIndex);
    batch.addQuad(x, y, w, h, tex_region);
}

RenderBatch& RenderBatchManager::getBatch(int textureId, int zIndex) {
    PROFILE_FUNCTION();
    BatchKey key = createKey(textureId, zIndex);

    // Try to find existing batch
    auto it = batchMap.find(key);
    if (it != batchMap.end()) {
        return batches[it->second];
    }

    // Create new batch
    size_t newIndex = batches.size();
    batches.emplace_back(textureId, zIndex);
    batchMap[key] = newIndex;
    needsSort = true; // New batch might change sorting order

    return batches[newIndex];
}

void RenderBatchManager::clear() {
    PROFILE_FUNCTION();
    for (auto& batch : batches) {
        batch.clear();
    }
    // Don't clear the map - reuse the same batches
}

void RenderBatchManager::sortIfNeeded() {
    PROFILE_FUNCTION();
    if (!needsSort) return;

    // Implement sorting if needed
    needsSort = false;
}

const std::vector<RenderBatch>& RenderBatchManager::getBatches() {
    PROFILE_FUNCTION();
    return batches;
}

size_t RenderBatchManager::getBatchCount() const {
    PROFILE_FUNCTION();
    return batches.size();
}

// Engine implementation
Engine::Engine(int window_width, int window_height, int world_width, int world_height)
    : renderBatchManager(8)
    , atlas(nullptr, 2048, 2048)  // Temporarily set renderer to nullptr, will set after creation
{
    PROFILE_FUNCTION();

    // Create window and renderer
    window = SDL_CreateWindow("2D Game Engine - SDL3", window_width, window_height, 0);
    if (!window) {
        throw std::runtime_error("Failed to create SDL window");
    }

    renderer = SDL_CreateRenderer(window, NULL);
    if (!renderer) {
        SDL_DestroyWindow(window);
        throw std::runtime_error("Failed to create SDL renderer");
    }

    // Now properly initialize TextureAtlas with the renderer
    atlas = TextureAtlas(renderer, 2048, 2048);

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

    // Clean up SDL resources
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
}

int Engine::registerTexture(SDL_Surface* surface, int x, int y, int width, int height) {
    PROFILE_FUNCTION();
    return atlas.registerTexture(surface, x, y, width, height);
}

void Engine::renderScene() {
    PROFILE_FUNCTION();

    SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
    SDL_RenderClear(renderer);

    // Clear batches from previous frame
    {
        PROFILE_SCOPE("render scene p1");
        renderBatchManager.clear();
    }

    {
        PROFILE_SCOPE("render scene p2");

        // Calculate camera bounds in fixed-point
        const int cx1 = camera.x - camera.width / 2;
        const int cy1 = camera.y - camera.height / 2;
        const int cx2 = camera.x + camera.width / 2;
        const int cy2 = camera.y + camera.height / 2;

        // Add some margin for culling to prevent pop-in (100 units in fixed-point)
        const int32_t margin = toGameUnit(100);

        // Query the spatial grid for entities in the visible area
        std::vector<EntityRef>& visible_entities = grid.queryRect(
            cx1 - margin, cy1 - margin, cx2 + margin, cy2 + margin);

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

        // Process only entities returned by the grid query
        for (const EntityRef& entity : visible_entities) {
            auto container = entityManager.containers[entity.type].get();
            if (!(container->containerFlag & (uint8_t)ContainerFlag::RENDERABLE))
                continue;

            RenderableEntityContainer* renderableContainer =
                static_cast<RenderableEntityContainer*>(container);

            // Convert fixed-point positions and dimensions to float for rendering
            const int screen_x = renderableContainer->x_positions[entity.index] - cx1;
            const int screen_y = renderableContainer->y_positions[entity.index] - cy1;
            const int screen_w = renderableContainer->widths[entity.index];
            const int screen_h = renderableContainer->heights[entity.index];

            // Skip if outside camera view (final precise culling)
            if (screen_x + (int)toGameUnit(screen_w) < 0 || screen_x > cx2 ||
                screen_y + (int)toGameUnit(screen_h) < 0 || screen_y > cy2)
                continue;

            // Get texture region using TextureAtlas class
            SDL_FRect texRegion = atlas.getRegion(renderableContainer->texture_ids[entity.index]);

            if(entity.type == 1)
            std::cout << fixedPointToFloat(screen_x) << std::endl;
            // Add quad to appropriate batch
            renderBatchManager.addQuad(
                renderableContainer->texture_ids[entity.index],
                renderableContainer->z_indices[entity.index],
                fixedPointToFloat(screen_x), fixedPointToFloat(screen_y),
                renderableContainer->widths[entity.index], renderableContainer->heights[entity.index],
                texRegion
            );
        }
    }

    {
        PROFILE_SCOPE("render scene p3");
        // Render all batches in correct z-order
        const auto& batches = renderBatchManager.getBatches();

        for (const auto& batch : batches) {
            if (batch.vertices.size() == 0) continue;

            // Set the texture for this batch using TextureAtlas class
            SDL_Texture* texture = atlas.getTexture(batch.texture_id);

            SDL_SetTextureScaleMode(texture, SDL_SCALEMODE_LINEAR);
            // Render the batch in one draw call
            SDL_RenderGeometry(
                renderer,
                texture,
                batch.vertices.data(),
                batch.vertices.size(),
                batch.indices.data(),
                batch.indices.size()
            );
        }
    }
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
    SDL_RenderPresent(renderer);
}

// TextureAtlas implementation
TextureAtlas::TextureAtlas(SDL_Renderer* renderer, int width, int height, int initialCapacity)
    : renderer(renderer)
    , texture_count(0)
    , texture_capacity(initialCapacity)
    , region_count(0)
    , region_capacity(64)
{
    PROFILE_FUNCTION();
    // Allocate texture array with alignment for better cache performance
    textures = static_cast<SDL_Texture**>(SDL_aligned_alloc(CACHE_LINE_SIZE,
        texture_capacity * sizeof(SDL_Texture*)));

    // Allocate region array with alignment
    regions = static_cast<SDL_FRect*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
        region_capacity * sizeof(SDL_FRect)));

    // Create a texture for the atlas
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_TARGET, width, height);
    SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_BLEND);

    // Add the texture to the array
    textures[0] = texture;
    texture_count = 1;
}

TextureAtlas::~TextureAtlas() {
    PROFILE_FUNCTION();
    // Free textures
    for (int i = 0; i < texture_count; i++) {
        SDL_DestroyTexture(textures[i]);
    }

    // Free arrays
    SDL_aligned_free(textures);
    SDL_aligned_free(regions);
}

TextureAtlas::TextureAtlas(TextureAtlas&& other) noexcept
    : textures(other.textures)
    , texture_count(other.texture_count)
    , texture_capacity(other.texture_capacity)
    , regions(other.regions)
    , region_count(other.region_count)
    , region_capacity(other.region_capacity)
    , renderer(other.renderer)
{
    PROFILE_FUNCTION();
    other.textures = nullptr;
    other.regions = nullptr;
    other.texture_count = 0;
    other.region_count = 0;
}

TextureAtlas& TextureAtlas::operator=(TextureAtlas&& other) noexcept {
    PROFILE_FUNCTION();
    if (this != &other) {
        // Free current resources
        for (int i = 0; i < texture_count; i++) {
            SDL_DestroyTexture(textures[i]);
        }
        SDL_aligned_free(textures);
        SDL_aligned_free(regions);

        // Move other's resources
        textures = other.textures;
        texture_count = other.texture_count;
        texture_capacity = other.texture_capacity;
        regions = other.regions;
        region_count = other.region_count;
        region_capacity = other.region_capacity;
        renderer = other.renderer;

        // Null out other's pointers
        other.textures = nullptr;
        other.regions = nullptr;
        other.texture_count = 0;
        other.region_count = 0;
    }
    return *this;
}

int TextureAtlas::registerTexture(SDL_Surface* surface, int x, int y, int width, int height) {
    PROFILE_FUNCTION();

    int texture_id = region_count;

    // Ensure capacity
    ensureRegionCapacity(texture_id + 1);

    // Calculate normalized UV coordinates
    float atlas_width, atlas_height;
    SDL_GetTextureSize(textures[0], &atlas_width, &atlas_height);

    // Use the provided width/height or the surface dimensions if not specified
    int tex_width = (width > 0) ? width : surface->w;
    int tex_height = (height > 0) ? height : surface->h;

    SDL_FRect region = {
        (float)x / atlas_width,
        (float)y / atlas_height,
        (float)tex_width / atlas_width,
        (float)tex_height / atlas_height
    };

    regions[texture_id] = region;
    region_count++;

    // Copy surface to atlas texture
    SDL_Texture* temp = SDL_CreateTextureFromSurface(renderer, surface);

    // Set render target to atlas
    SDL_Texture* old_target = SDL_GetRenderTarget(renderer);
    SDL_SetRenderTarget(renderer, textures[0]);

    // Copy texture to atlas
    SDL_FRect dest = { (float)x, (float)y, (float)tex_width, (float)tex_height };
    SDL_RenderTexture(renderer, temp, NULL, &dest);

    // Reset render target
    SDL_SetRenderTarget(renderer, old_target);

    // Clean up
    SDL_DestroyTexture(temp);

    return texture_id;
}

SDL_FRect TextureAtlas::getRegion(int textureId) const {
    PROFILE_FUNCTION();
    if (textureId >= 0 && textureId < region_count) {
        return regions[textureId];
    }
    // Return empty region if invalid texture ID
    return { 0, 0, 1, 1 };
}

SDL_Texture* TextureAtlas::getTexture(int textureId) const {
    PROFILE_FUNCTION();
    // Currently, we use only the first texture in the atlas
    // This simplifies batch rendering while still allowing future expansion
    return textures[0];
}

void TextureAtlas::ensureTextureCapacity(int needed) {
    PROFILE_FUNCTION();
    if (needed <= texture_capacity) return;

    int new_capacity = texture_capacity * 2;
    while (new_capacity < needed) new_capacity *= 2;

    SDL_Texture** new_textures = static_cast<SDL_Texture**>(SDL_aligned_alloc(CACHE_LINE_SIZE,
        new_capacity * sizeof(SDL_Texture*)));

    // Copy existing textures
    memcpy(new_textures, textures, texture_count * sizeof(SDL_Texture*));

    SDL_aligned_free(textures);
    textures = new_textures;
    texture_capacity = new_capacity;
}

void TextureAtlas::ensureRegionCapacity(int needed) {
    if (needed <= region_capacity) return;

    int new_capacity = region_capacity * 2;
    while (new_capacity < needed) new_capacity *= 2;

    SDL_FRect* new_regions = static_cast<SDL_FRect*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
        new_capacity * sizeof(SDL_FRect)));

    // Copy existing regions
    memcpy(new_regions, regions, region_count * sizeof(SDL_FRect));

    SDL_aligned_free(regions);
    regions = new_regions;
    region_capacity = new_capacity;
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