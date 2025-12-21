#include "ATMEngine.h"

// EntityContainer implementation
EntityContainer::EntityContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
    : type_id(typeId)
    , default_layer(defaultLayer)
    , capacity(initialCapacity)
    , count(0), containerFlag((uint8_t)ContainerFlag::UPDATEABLE)
{
    // Allocate base arrays
    flags = new uint8_t[capacity];
    x_positions = new float[capacity];
    y_positions = new float[capacity];
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
    x_positions[index] = 0.0f;  // Changed from 0 to 0.0f
    y_positions[index] = 0.0f;  // Changed from 0 to 0.0f
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

    float* newXPositions = new float[newCapacity];  // Changed from int32_t to float
    float* newYPositions = new float[newCapacity];  // Changed from int32_t to float
    // Copy existing data
    for (int i = 0; i < count; i++) {
        newXPositions[i] = x_positions[i];
        newYPositions[i] = y_positions[i];
        newFlags[i] = flags[i];
    }

    // Delete old arrays
    delete[] flags;

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

void Layer::update(float delta_time) {
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
    for(int i = 0;i < MAX_LAYERS; i++)
    {
        layers.push_back(std::make_unique<Layer>(i));
    }
}

int EntityManager::registerEntityType(EntityContainer* container) {
    PROFILE_FUNCTION();
    int type_id = containers.size();
    containers.emplace_back(container);

    uint8_t layer_index = container->getDefaultLayer();

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

void EntityManager::update(float delta_time) {
    PROFILE_FUNCTION();
    for (auto& layer : layers) {
        layer->update(delta_time);
    }
}

// RenderBatch implementation
RenderBatch::RenderBatch(int textureId, int zIndex, int initialVertexCapacity)
    : texture_id(textureId)
{
    PROFILE_FUNCTION();

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
    const uint64_t& base_vert = vertices.size();

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

    return batches[newIndex];
}

void RenderBatchManager::clear() {
    PROFILE_FUNCTION();
    for (auto& batch : batches) {
        batch.clear();
    }
    // Don't clear the map - reuse the same batches
}


const std::vector<RenderBatch>& RenderBatchManager::getBatches() {
    PROFILE_FUNCTION();
    return batches;
}

size_t RenderBatchManager::getBatchCount() const {
    PROFILE_FUNCTION();
    return batches.size();
}











// Process entities marked for removal
void process_pending_removals(Engine* engine) {
    PROFILE_FUNCTION();

    if (engine->pending_removals.empty()) return;

    for (const auto& ref : engine->pending_removals) {
        if (ref.type < 0 || ref.type >= engine->entityManager.containers.size()) continue;

        auto container = engine->entityManager.containers[ref.type].get();
        if (!container) continue;

        container->removeEntity(ref.index);
    }

    engine->pending_removals.clear();
}

// Update the engine state
void engine_update(Engine* engine) {
    PROFILE_FUNCTION();

    // Calculate delta time
    Uint64 current_time = SDL_GetTicks();
    float delta_time = (current_time - engine->last_frame_time) / 1000.0f;
    engine->last_frame_time = current_time;

    // Smooth FPS calculation
    engine->fps = 0.95f * engine->fps + 0.05f * (1.0f / delta_time);

    engine->grid.rebuild_grid(engine);


    // Update all entity layers
    engine->entityManager.update(delta_time);

    // Process pending removals
    process_pending_removals(engine);
}

// Set entity z_index
void engine_set_entity_z_index(Engine* engine, uint32_t entity_idx, int type_id, uint8_t z_index) {
    PROFILE_FUNCTION();

    if (type_id >= engine->entityManager.containers.size()) return;

    auto container = engine->entityManager.containers[type_id].get();
    if (!container) return;

    RenderableEntityContainer* renderable = dynamic_cast<RenderableEntityContainer*>(container);
    if (!renderable || entity_idx >= renderable->getCount()) return;

    renderable->z_indices[entity_idx] = z_index;
}

// Present the renderer
void engine_present(Engine* engine) {
    PROFILE_FUNCTION();
    SDL_RenderPresent(engine->renderer);
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

// Helper function to get texture region - now a wrapper around TextureAtlas method
SDL_FRect get_texture_region(const TextureAtlas& atlas, int16_t texture_id) {
    PROFILE_FUNCTION();
    return atlas.getRegion(texture_id);
}

// Engine initialization with TextureAtlas class
Engine* engine_create(int window_width, int window_height, int world_width, int world_height, int cell_size) {
    PROFILE_FUNCTION();
    Engine* engine = static_cast<Engine*>(malloc(sizeof(Engine)));
    if (!engine) return NULL;

    // Use placement new to properly initialize C++ members
    new (&engine->grid) SpatialGrid();
    new (&engine->entityManager) EntityManager();
    new (&engine->pending_removals) std::vector<EntityRef>();
    new (&engine->renderBatchManager) RenderBatchManager(8);

    // Create window and renderer
    engine->window = SDL_CreateWindow("2D Game Engine - SDL3", window_width, window_height, 0);
    if (!engine->window) {
        engine->renderBatchManager.~RenderBatchManager();
        engine->entityManager.~EntityManager();
        engine->pending_removals.~vector();
        free(engine);
        return NULL;
    }

    engine->renderer = SDL_CreateRenderer(engine->window, NULL);
    if (!engine->renderer) {
        SDL_DestroyWindow(engine->window);
        engine->renderBatchManager.~RenderBatchManager();
        engine->entityManager.~EntityManager();
        engine->pending_removals.~vector();
        free(engine);
        return NULL;
    }

    // Initialize the TextureAtlas with placement new
    new (&engine->atlas) TextureAtlas(engine->renderer, 2048, 2048);

    // Init world bounds
    engine->world_bounds.x = 0;
    engine->world_bounds.y = 0;
    engine->world_bounds.w = world_width;
    engine->world_bounds.h = world_height;

    // Init camera
    engine->camera.x = 0;
    engine->camera.y = 0;
    engine->camera.width = window_width;
    engine->camera.height = window_height;
    engine->camera.zoom = 1.0f;

    // Init timing
    engine->last_frame_time = SDL_GetTicks();
    engine->fps = 0.0f;

    return engine;
}

// Clean up the engine resources
void engine_destroy(Engine* engine) {
    PROFILE_FUNCTION();
    if (!engine) return;

    // Call destructors for C++ members in reverse order of construction
    engine->atlas.~TextureAtlas();
    engine->renderBatchManager.~RenderBatchManager();
    engine->pending_removals.~vector();
    engine->entityManager.~EntityManager();
    engine->grid.~SpatialGrid();

    // Destroy SDL resources
    SDL_DestroyRenderer(engine->renderer);
    SDL_DestroyWindow(engine->window);

    // Free the engine struct
    free(engine);
}

// Register a texture with the engine - now using TextureAtlas class
int engine_register_texture(Engine* engine, SDL_Surface* surface, int x, int y, int width, int height) {
    PROFILE_FUNCTION();
    return engine->atlas.registerTexture(surface, x, y, width, height);
}

void engine_render_scene(Engine* engine) {
    PROFILE_FUNCTION();

    SDL_SetRenderDrawColor(engine->renderer, 20, 20, 20, 255);
    SDL_RenderClear(engine->renderer);
    // Clear batches from previous frame
    {
        PROFILE_SCOPE("render scene p1");
        engine->renderBatchManager.clear();
    }

    {
        PROFILE_SCOPE("render scene p2");

        const float& x1 = engine->camera.x - engine->camera.width / 2;
        const float& y1 = engine->camera.y - engine->camera.height / 2;
        const float& x2 = engine->camera.x + engine->camera.width / 2;
        const float& y2 = engine->camera.y + engine->camera.height / 2;

        // Query the spatial grid for entities in the visible area
        std::vector<EntityRef> visible_entities = engine->grid.queryRect(
            x1 - 100, y1 - 100, x2 + 100, y2 + 1);

        
        // Sort entities by their ID for better cache coherence and rendering efficiency
        std::sort(visible_entities.begin(), visible_entities.end(),
            [](const EntityRef& a, const EntityRef& b) {
                // First sort by type, then by index within type
                return (a.type == b.type) ? (a.index < b.index) : (a.type < b.type);
            });

        // Process only entities returned by the grid query
        for (const EntityRef& entity : visible_entities) {
            auto container = engine->entityManager.containers[entity.type].get();
            if (!(container->containerFlag & (uint8_t)ContainerFlag::RENDERABLE))
                continue;

            RenderableEntityContainer* renderableContainer =
                static_cast<RenderableEntityContainer*>(container);

            // Calculate screen coordinates
            // Add half screen width/height to center the camera view
            float x = renderableContainer->x_positions[entity.index] - engine->camera.x + engine->camera.width / 2;
            float y = renderableContainer->y_positions[entity.index] - engine->camera.y + engine->camera.height / 2;
            float w = renderableContainer->widths[entity.index];
            float h = renderableContainer->heights[entity.index];

            // Skip if outside camera view (final precise culling)
            if (x + w < 0 || x > engine->camera.width || y + h < 0 || y > engine->camera.height)
                continue;

            // Get texture region using TextureAtlas class
            SDL_FRect texRegion = engine->atlas.getRegion(renderableContainer->texture_ids[entity.index]);

            // Add quad to appropriate batch
            engine->renderBatchManager.addQuad(
                renderableContainer->texture_ids[entity.index],
                renderableContainer->z_indices[entity.index],
                x, y, w, h,
                texRegion
            );
        }
    }

    {
        PROFILE_SCOPE("render scene p3");
        // Render all batches in correct z-order
        const auto& batches = engine->renderBatchManager.getBatches();

        for (const auto& batch : batches) {
            if (batch.vertices.size() == 0) continue;

            // Set the texture for this batch using TextureAtlas class
            SDL_Texture* texture = engine->atlas.getTexture(batch.texture_id);
            SDL_SetTextureScaleMode(texture, SDL_SCALEMODE_NEAREST);

            // Render the batch in one draw call
            SDL_RenderGeometry(
                engine->renderer,
                texture,
                batch.vertices.data(),
                batch.vertices.size(),
                batch.indices.data(),
                batch.indices.size()
            );
        }
    }
}
#include <future>

// Helper struct to store grid entries for parallel processing
struct GridEntry {
    EntityRef entityId;
    uint16_t x;
    uint16_t y;
};
std::vector<std::future<void>> futures;

void SpatialGrid::rebuild_grid(Engine* engine) {
    PROFILE_FUNCTION();
    clearAll();

    constexpr uint32_t CHUNK_SIZE = 25000; // Adjust based on your needs
    futures.clear();

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
                            // Calculate grid cell coordinates
                            uint16_t cellX = (uint16_t)(x_positions[i] * INV_GRID_CELL_SIZE);
                            uint16_t cellY = (uint16_t)(y_positions[i] * INV_GRID_CELL_SIZE);

                            // Call add2 which now handles atomic operations
                            add2({ containerIdx, i }, cellX, cellY);
                        }
                    }));
            }
        }
        else {
            // Process small containers directly in this thread
            for (uint32_t i = 0; i < entityCount; i++) {
                add2({ containerIdx, i },
                    (uint16_t)(container->x_positions[i] * INV_GRID_CELL_SIZE),
                    (uint16_t)(container->y_positions[i] * INV_GRID_CELL_SIZE));
            }
        }
    }

    // Wait for all async tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    futures.clear();

    for(int i = 0; i < cells.size(); i++)
    {
        for(int y = 0; y < cells.size();y++)
        {
            cells[i][y].count = std::min(cells[i][y].count.operator int(), MAX_ENTITIES_PER_CELL);
        }
    }
}

