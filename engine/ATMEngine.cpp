#include "ATMEngine.h"
#include <algorithm>
#include <execution>
#include <future>
#include <numeric>
#include <thread>
#define STB_IMAGE_IMPLEMENTATION
#include "SDL3_image/SDL_image.h"
#include "stb_image.h"


// EntityContainer implementation
EntityContainer::EntityContainer(int typeId, uint8_t defaultLayer,
                                 int initialCapacity)
    : flags(initialCapacity), entity_ids(initialCapacity),
      parent_ids(initialCapacity, INVALID_ID),
      first_child_ids(initialCapacity, INVALID_ID),
      next_sibling_ids(initialCapacity, INVALID_ID),
      x_positions(initialCapacity, 0.0f), y_positions(initialCapacity, 0.0f),
      cell_x(initialCapacity), cell_y(initialCapacity),
      grid_node_indices(initialCapacity, -1),
      containerFlag((uint8_t)ContainerFlag::UPDATEABLE), type_id(typeId),
      default_layer(defaultLayer), capacity(initialCapacity), count(0) {
  // All arrays initialized via member initializer list - RAII handles cleanup
}

EntityContainer::~EntityContainer() {
  PROFILE_FUNCTION();
  // RAII: DynamicArray destructors automatically free memory
}

uint32_t EntityContainer::createEntity() {
  if (count >= capacity) {
    // Resize arrays to accommodate more entities
    int newCapacity = capacity * 2;
    resizeArrays(newCapacity);
  }

  size_t index = count++;
  x_positions[index] = 0.0f;
  y_positions[index] = 0.0f;
  flags[index] = static_cast<uint8_t>(EntityFlag::NONE);
  entity_ids[index] = index;
  parent_ids[index] = INVALID_ID;
  first_child_ids[index] = INVALID_ID;
  next_sibling_ids[index] = INVALID_ID;
  cell_x[index] = 0;
  cell_y[index] = 0;
  grid_node_indices[index] = -1;

  return index;
}

void EntityContainer::removeEntity(size_t index) {
  PROFILE_FUNCTION();
  if (index >= count)
    return;

  // Move last entity to removed position
  size_t last = count - 1;
  if (index < last) {
    x_positions[index] = x_positions[last];
    y_positions[index] = y_positions[last];
    flags[index] = flags[last];
    entity_ids[index] = index; // Update to new position
    parent_ids[index] = parent_ids[last];
    first_child_ids[index] = first_child_ids[last];
    next_sibling_ids[index] = next_sibling_ids[last];
    cell_x[index] = cell_x[last];
    cell_y[index] = cell_y[last];
    grid_node_indices[index] = grid_node_indices[last];
  }

  count--;
}

void EntityContainer::resizeArrays(int newCapacity) {
  PROFILE_FUNCTION();
  if (newCapacity <= capacity)
    return;

  // RAII: Use DynamicArray::resize() - handles allocation, copy, and cleanup
  flags.resize(newCapacity, count);
  entity_ids.resize(newCapacity, count);
  parent_ids.resize(newCapacity, count, INVALID_ID);
  first_child_ids.resize(newCapacity, count, INVALID_ID);
  next_sibling_ids.resize(newCapacity, count, INVALID_ID);
  x_positions.resize(newCapacity, count, 0.0f);
  y_positions.resize(newCapacity, count, 0.0f);
  cell_x.resize(newCapacity, count);
  cell_y.resize(newCapacity, count);
  grid_node_indices.resize(newCapacity, count, -1);

  // Update capacity
  capacity = newCapacity;
}

// RenderableEntityContainer implementation
RenderableEntityContainer::RenderableEntityContainer(int typeId,
                                                     uint8_t defaultLayer,
                                                     int initialCapacity)
    : EntityContainer(typeId, defaultLayer, initialCapacity),
      widths(initialCapacity), heights(initialCapacity),
      texture_ids(initialCapacity), z_indices(initialCapacity),
      rotations(initialCapacity, 0.0f) {
  this->containerFlag |= (uint8_t)ContainerFlag::RENDERABLE;
}

RenderableEntityContainer::~RenderableEntityContainer() {
  PROFILE_FUNCTION();
  // RAII: DynamicArray destructors automatically free memory
}

uint32_t RenderableEntityContainer::createEntity() {
  uint32_t index = EntityContainer::createEntity();
  if (index == INVALID_ID)
    return INVALID_ID;

  widths[index] = 0;
  heights[index] = 0;
  texture_ids[index] = 0;
  z_indices[index] = 0;
  rotations[index] = 0.0f;
  return index;
}

void RenderableEntityContainer::removeEntity(size_t index) {
  PROFILE_FUNCTION();
  if (index >= count)
    return;

  size_t last = count - 1;
  if (index < last) {
    widths[index] = widths[last];
    heights[index] = heights[last];
    texture_ids[index] = texture_ids[last];
    z_indices[index] = z_indices[last];
    rotations[index] = rotations[last];
  }

  EntityContainer::removeEntity(index);
}

void RenderableEntityContainer::resizeArrays(int newCapacity) {
  PROFILE_FUNCTION();
  if (newCapacity <= capacity)
    return;

  // IMPORTANT: EntityContainer::resizeArrays handles resizing the base arrays
  EntityContainer::resizeArrays(newCapacity);

  // RAII: Use DynamicArray::resize() for renderable-specific arrays
  widths.resize(newCapacity, count);
  heights.resize(newCapacity, count);
  texture_ids.resize(newCapacity, count);
  z_indices.resize(newCapacity, count);
  rotations.resize(newCapacity, count, 0.0f);
}

// Layer implementation
Layer::Layer(int id) : layer_id(id), is_active(true) { PROFILE_FUNCTION(); }

void Layer::update(float delta_time) {
  PROFILE_FUNCTION();
  if (!is_active)
    return;
  for (auto container : entity_containers) {
    if (container->containerFlag & (uint8_t)ContainerFlag::UPDATEABLE)
      container->update(delta_time);
  }
}

void Layer::addEntityContainer(EntityContainer *container) {
  PROFILE_FUNCTION();
  entity_containers.push_back(container);
}

// EntityManager implementation
EntityManager::EntityManager() : next_entity_id(0) {
  PROFILE_FUNCTION();
  layers.push_back(std::make_unique<Layer>(0));
}

int EntityManager::registerEntityType(EntityContainer *container) {
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
  if (type_id >= containers.size())
    return INVALID_ID;
  uint32_t index = containers[type_id]->createEntity();
  if (index != INVALID_ID) {
    // Could store a mapping from index to next_entity_id if needed
    next_entity_id++;
  }
  return index;
}

void EntityManager::removeEntity(uint32_t index, int type_id) {
  PROFILE_FUNCTION();
  if (type_id >= containers.size())
    return;
  containers[type_id]->removeEntity(index);
}

void EntityManager::update(float delta_time) {
  PROFILE_FUNCTION();
  for (auto &layer : layers) {
    if (layer)
      layer->update(delta_time);
  }
}

// RenderBatch implementation
RenderBatch::RenderBatch(int textureId, int zIndex, int initialVertexCapacity)
    : texture_id(textureId), z_index(zIndex) {
  PROFILE_FUNCTION();
  vertices.reserve(initialVertexCapacity);
  indices.reserve(initialVertexCapacity * 1.5);
}

RenderBatch::~RenderBatch() {}

RenderBatch::RenderBatch(RenderBatch &&other) noexcept
    : texture_id(other.texture_id), vertices(std::move(other.vertices)),
      indices(std::move(other.indices)) {
  PROFILE_FUNCTION();
}

RenderBatch &RenderBatch::operator=(RenderBatch &&other) noexcept {
  PROFILE_FUNCTION();
  if (this != &other) {

    texture_id = other.texture_id;
    vertices = std::move(other.vertices);
    indices = std::move(other.indices);
  }
  return *this;
}

void RenderBatch::addQuad(float x, float y, float w, float h,
                          SDL_FRect tex_region) {
  // Ensure we have enough space
  const uint64_t &base_vert = vertices.size();

  vertices.resize(vertices.size() + 4);
  indices.resize(indices.size() + 6);

  // Use direct memory access for better performance
  SDL_Vertex *v = &vertices.data()[vertices.size() - 4];

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
  int *idx = &indices.data()[indices.size() - 6];
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

RenderBatchManager::RenderBatchManager(int initialBatchCount)
    : needsSort(false) {
  batches.reserve(initialBatchCount *
                  2); // Reserve extra space to minimize reallocations
}

void RenderBatchManager::addQuad(int textureId, int zIndex, float x, float y,
                                 float w, float h, SDL_FRect tex_region) {
  // Get or create a batch for this texture/z-index combination
  RenderBatch &batch = getBatch(textureId, zIndex);
  batch.addQuad(x, y, w, h, tex_region);
}

RenderBatch &RenderBatchManager::getBatch(int textureId, int zIndex) {
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
  for (auto &batch : batches) {
    batch.clear();
  }
  // Don't clear the map - reuse the same batches
}

const std::vector<RenderBatch> &RenderBatchManager::getBatches() {
  PROFILE_FUNCTION();
  if (needsSort) {
    sortIfNeeded();
  }
  return batches;
}

void RenderBatchManager::sortIfNeeded() {
  PROFILE_FUNCTION();
  std::sort(batches.begin(), batches.end(),
            [](const RenderBatch &a, const RenderBatch &b) {
              if (a.z_index != b.z_index)
                return a.z_index < b.z_index;
              return a.texture_id < b.texture_id;
            });

  // Rebuild the map labels to reflect new indices
  batchMap.clear();
  for (size_t i = 0; i < batches.size(); ++i) {
    batchMap[createKey(batches[i].texture_id, batches[i].z_index)] = i;
  }
  needsSort = false;
}

size_t RenderBatchManager::getBatchCount() const {
  PROFILE_FUNCTION();
  return batches.size();
}

// Process entities marked for removal
void process_pending_removals(Engine *engine) {
  PROFILE_FUNCTION();

  if (engine->pending_removals.empty())
    return;

  for (const auto &ref : engine->pending_removals) {
    if (ref.type < 0 || ref.type >= engine->entityManager.containers.size())
      continue;

    auto container = engine->entityManager.containers[ref.type].get();
    if (!container)
      continue;

    container->removeEntity(ref.index);
  }

  engine->pending_removals.clear();
}

// Update the engine state
void engine_update(Engine *engine) {
  PROFILE_FUNCTION();

  // Calculate delta time
  Uint64 current_time = SDL_GetTicks();
  float delta_time = (current_time - engine->last_frame_time) / 1000.0f;
  engine->last_frame_time = current_time;

  // Smooth FPS calculation
  engine->fps = 0.95f * engine->fps + 0.05f * (1.0f / delta_time);

  // Grid is now updated incrementally during entity updates - no full rebuild
  // needed!

  // Update all entity layers
  engine->entityManager.update(delta_time);

  // Process pending removals
  process_pending_removals(engine);
}

// Set entity z_index
void engine_set_entity_z_index(Engine *engine, uint32_t entity_idx, int type_id,
                               uint8_t z_index) {
  PROFILE_FUNCTION();

  if (type_id >= engine->entityManager.containers.size())
    return;

  auto container = engine->entityManager.containers[type_id].get();
  if (!container)
    return;

  RenderableEntityContainer *renderable =
      dynamic_cast<RenderableEntityContainer *>(container);
  if (!renderable || entity_idx >= renderable->getCount())
    return;

  renderable->z_indices[entity_idx] = z_index;
}

// Present the renderer
void engine_present(Engine *engine) {
  PROFILE_FUNCTION();
  // Present with SDL
  SDL_RenderPresent(engine->renderer);
}

// TextureAtlas implementation
TextureAtlas::TextureAtlas(SDL_Renderer *renderer, int width, int height,
                           int initialCapacity)
    : renderer(renderer), texture_count(0), texture_capacity(initialCapacity),
      region_count(0), region_capacity(64) {
  PROFILE_FUNCTION();
  // Allocate texture array with alignment for better cache performance
  textures = static_cast<SDL_Texture **>(SDL_aligned_alloc(
      CACHE_LINE_SIZE, texture_capacity * sizeof(SDL_Texture *)));

  // Allocate region array with alignment
  regions = static_cast<SDL_FRect *>(
      SDL_aligned_alloc(CACHE_LINE_SIZE, region_capacity * sizeof(SDL_FRect)));

  // Create a texture for the atlas
  SDL_Texture *texture =
      SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
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

TextureAtlas::TextureAtlas(TextureAtlas &&other) noexcept
    : textures(other.textures), texture_count(other.texture_count),
      texture_capacity(other.texture_capacity), regions(other.regions),
      region_count(other.region_count), region_capacity(other.region_capacity),
      renderer(other.renderer) {
  PROFILE_FUNCTION();
  other.textures = nullptr;
  other.regions = nullptr;
  other.texture_count = 0;
  other.region_count = 0;
}

TextureAtlas &TextureAtlas::operator=(TextureAtlas &&other) noexcept {
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

int TextureAtlas::registerTexture(SDL_Surface *surface, int x, int y, int width,
                                  int height) {
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

  SDL_FRect region = {(float)x / atlas_width, (float)y / atlas_height,
                      (float)tex_width / atlas_width,
                      (float)tex_height / atlas_height};

  regions[texture_id] = region;
  region_count++;

  // Copy surface to atlas texture
  SDL_Texture *temp = SDL_CreateTextureFromSurface(renderer, surface);

  // Set render target to atlas
  SDL_Texture *old_target = SDL_GetRenderTarget(renderer);
  SDL_SetRenderTarget(renderer, textures[0]);

  // Copy texture to atlas
  SDL_FRect dest = {(float)x, (float)y, (float)tex_width, (float)tex_height};
  SDL_RenderTexture(renderer, temp, NULL, &dest);

  // Reset render target
  SDL_SetRenderTarget(renderer, old_target);

  // Clean up
  SDL_DestroyTexture(temp);

  return texture_id;
}

SDL_FRect TextureAtlas::getRegion(int textureId) const {
  if (textureId >= 0 && textureId < region_count) {
    return regions[textureId];
  }
  // Return empty region if invalid texture ID
  return {0, 0, 1, 1};
}

SDL_Texture *TextureAtlas::getTexture(int textureId) const {
  PROFILE_FUNCTION();
  // Currently, we use only the first texture in the atlas
  // This simplifies batch rendering while still allowing future expansion
  return textures[0];
}

void TextureAtlas::ensureTextureCapacity(int needed) {
  PROFILE_FUNCTION();
  if (needed <= texture_capacity)
    return;

  int new_capacity = texture_capacity * 2;
  while (new_capacity < needed)
    new_capacity *= 2;

  SDL_Texture **new_textures = static_cast<SDL_Texture **>(
      SDL_aligned_alloc(CACHE_LINE_SIZE, new_capacity * sizeof(SDL_Texture *)));

  // Copy existing textures
  memcpy(new_textures, textures, texture_count * sizeof(SDL_Texture *));

  SDL_aligned_free(textures);
  textures = new_textures;
  texture_capacity = new_capacity;
}

void TextureAtlas::ensureRegionCapacity(int needed) {
  if (needed <= region_capacity)
    return;

  int new_capacity = region_capacity * 2;
  while (new_capacity < needed)
    new_capacity *= 2;

  SDL_FRect *new_regions = static_cast<SDL_FRect *>(
      SDL_aligned_alloc(CACHE_LINE_SIZE, new_capacity * sizeof(SDL_FRect)));

  // Copy existing regions
  memcpy(new_regions, regions, region_count * sizeof(SDL_FRect));

  SDL_aligned_free(regions);
  regions = new_regions;
  region_capacity = new_capacity;
}

// Helper function to get texture region - now a wrapper around TextureAtlas
// method
SDL_FRect get_texture_region(const TextureAtlas &atlas, int16_t texture_id) {
  PROFILE_FUNCTION();
  return atlas.getRegion(texture_id);
}

// Engine initialization with TextureAtlas class
Engine *engine_create(int window_width, int window_height, int world_width,
                      int world_height, int cell_size) {
  PROFILE_FUNCTION();

  Engine *engine = static_cast<Engine *>(malloc(sizeof(Engine)));
  if (!engine)
    return NULL;

  // Use placement new to properly initialize C++ members
  new (&engine->grid) SpatialGrid();
  new (&engine->entityManager) EntityManager();
  new (&engine->pending_removals) std::vector<EntityRef>();
  new (&engine->renderBatchManager) RenderBatchManager(8);

  // Create window
  Uint32 windowFlags = SDL_WINDOW_HIGH_PIXEL_DENSITY;

  engine->window = SDL_CreateWindow("2D Game Engine", window_width,
                                    window_height, windowFlags);
  if (!engine->window) {
    engine->renderBatchManager.~RenderBatchManager();
    engine->entityManager.~EntityManager();
    engine->pending_removals.~vector();
    free(engine);
    return NULL;
  }
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

  // Initialize SDL renderer
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
  engine->world_bounds.w = (float)world_width;
  engine->world_bounds.h = (float)world_height;

  // Init camera
  engine->camera.x = 0;
  engine->camera.y = 0;
  engine->camera.width = (float)window_width;
  engine->camera.height = (float)window_height;
  engine->camera.zoom = 1.0f;

  // Init timing
  engine->last_frame_time = SDL_GetTicks();
  engine->fps = 0.0f;

  return engine;
}
// Clean up the engine resources
void engine_destroy(Engine *engine) {
  PROFILE_FUNCTION();
  if (!engine)
    return;

  // Call destructors for C++ members in reverse order of construction
  engine->atlas.~TextureAtlas();
  engine->renderBatchManager.~RenderBatchManager();

  // Destroy SDL resources
  if (engine->renderer) {
    SDL_DestroyRenderer(engine->renderer);
  }

  SDL_DestroyWindow(engine->window);

  // Free the engine struct
  free(engine);
}
int engine_register_texture(Engine *engine, SDL_Surface *surface, int x, int y,
                            int width, int height) {
  PROFILE_FUNCTION();
  // Use SDL texture management
  return engine->atlas.registerTexture(surface, x, y, width, height);
}

void engine_render_scene(Engine *engine) {
  PROFILE_FUNCTION();

  // Modern 2D Aesthetic: Clear to dark gray
  SDL_SetRenderDrawColor(engine->renderer, 15, 15, 20, 255);
  SDL_RenderClear(engine->renderer);

  engine->renderBatchManager.clear();

  const float x1 = engine->camera.x - engine->camera.width / 2.0f;
  const float y1 = engine->camera.y - engine->camera.height / 2.0f;
  const float x2 = engine->camera.x + engine->camera.width / 2.0f;
  const float y2 = engine->camera.y + engine->camera.height / 2.0f;

  // 1. Thread-safe Parallel Query
  std::vector<EntityRef> &visible_entities =
      engine->grid.queryRect(x1 - 50, y1 - 50, x2 + 50, y2 + 50);
  if (visible_entities.empty())
    return;

  // 2. OPTIMIZED SORTING: Pre-compute sort keys to eliminate pointer derefs
  // during comparison. Key layout: (z_index << 56) | (type << 48) | index
  // This reduces ~40M pointer dereferences to ~N (one per entity during key
  // building)

  // Pre-computed sort key for zero-cost comparisons during sort
  struct SortableEntity {
    uint64_t sort_key;
    EntityRef ref;

    bool operator<(const SortableEntity &other) const {
      return sort_key < other.sort_key;
    }
  };

  // Reuse buffer across frames to avoid allocation
  static thread_local std::vector<SortableEntity> sortable_entities;
  sortable_entities.clear();
  sortable_entities.reserve(visible_entities.size());

  // Build sort keys - ONE pointer deref per entity (not per comparison)
  for (const auto &entity : visible_entities) {
    auto rCont = static_cast<RenderableEntityContainer *>(
        engine->entityManager.containers[entity.type].get());

    uint64_t key =
        (static_cast<uint64_t>(rCont->z_indices[entity.index]) << 56) |
        (static_cast<uint64_t>(entity.type) << 48) |
        static_cast<uint64_t>(entity.index);

    sortable_entities.push_back({key, entity});
  }

  // Sort on pre-computed keys - ZERO pointer derefs during sort
  std::sort(sortable_entities.begin(), sortable_entities.end());

  // 3. Build a SINGLE unified batch - preserves sort order for proper layering
  // Since all textures are in the atlas, we render with one draw call
  static std::vector<SDL_Vertex> unified_vertices;
  static std::vector<int> unified_indices;
  unified_vertices.clear();
  unified_indices.clear();
  unified_vertices.reserve(sortable_entities.size() * 4);
  unified_indices.reserve(sortable_entities.size() * 6);

  for (const auto &se : sortable_entities) {
    const auto &entity = se.ref;
    auto rCont = static_cast<RenderableEntityContainer *>(
        engine->entityManager.containers[entity.type].get());
    float x = rCont->x_positions[entity.index] - x1;
    float y = rCont->y_positions[entity.index] - y1;
    float w = rCont->widths[entity.index];
    float h = rCont->heights[entity.index];

    if (x + w < 0 || x > engine->camera.width || y + h < 0 ||
        y > engine->camera.height)
      continue;

    SDL_FRect texRegion =
        engine->atlas.getRegion(rCont->texture_ids[entity.index]);

    // Add quad directly to unified batch
    int base_vert = unified_vertices.size();

    // Vertices
    // Vertices - Manual Rotation
    SDL_Vertex v;
    v.color = {1, 1, 1, 1};

    float angle = rCont->rotations[entity.index];
    float cx = x + w * 0.5f;
    float cy = y + h * 0.5f;
    float c = cosf(angle);
    float s = sinf(angle);

    // Lambda to rotate point around center
    auto rotate = [&](float vx, float vy) -> SDL_FPoint {
      return {cx + (vx - cx) * c - (vy - cy) * s,
              cy + (vx - cx) * s + (vy - cy) * c};
    };

    // Top-left
    v.position = rotate(x, y);
    v.tex_coord = {texRegion.x, texRegion.y};
    unified_vertices.push_back(v);

    // Top-right
    v.position = rotate(x + w, y);
    v.tex_coord = {texRegion.x + texRegion.w, texRegion.y};
    unified_vertices.push_back(v);

    // Bottom-right
    v.position = rotate(x + w, y + h);
    v.tex_coord = {texRegion.x + texRegion.w, texRegion.y + texRegion.h};
    unified_vertices.push_back(v);

    // Bottom-left
    v.position = rotate(x, y + h);
    v.tex_coord = {texRegion.x, texRegion.y + texRegion.h};
    unified_vertices.push_back(v);

    // Indices (two triangles)
    unified_indices.push_back(base_vert);
    unified_indices.push_back(base_vert + 1);
    unified_indices.push_back(base_vert + 2);
    unified_indices.push_back(base_vert);
    unified_indices.push_back(base_vert + 2);
    unified_indices.push_back(base_vert + 3);
  }

  // 4. Single draw call with atlas texture
  if (!unified_vertices.empty()) {
    SDL_Texture *texture = engine->atlas.getTexture(0);
    SDL_SetTextureScaleMode(texture, SDL_SCALEMODE_NEAREST);
    SDL_RenderGeometry(engine->renderer, texture, unified_vertices.data(),
                       (int)unified_vertices.size(), unified_indices.data(),
                       (int)unified_indices.size());
  }
}

std::vector<EntityRef> &SpatialGrid::queryRect(float x1, float y1, float x2,
                                               float y2) {
  PROFILE_FUNCTION();
  queryResult.clear();

  const uint16_t minCellX =
      static_cast<uint16_t>(std::max(0.0f, x1 * INV_GRID_CELL_SIZE));
  const uint16_t minCellY =
      static_cast<uint16_t>(std::max(0.0f, y1 * INV_GRID_CELL_SIZE));
  const uint16_t maxCellX = static_cast<uint16_t>(std::min(
      static_cast<float>(GRID_CELL_WIDTH - 1), x2 * INV_GRID_CELL_SIZE));
  const uint16_t maxCellY = static_cast<uint16_t>(std::min(
      static_cast<float>(GRID_CELL_HEIGHT - 1), y2 * INV_GRID_CELL_SIZE));

  for (uint16_t cy = minCellY; cy <= maxCellY; ++cy) {
    int32_t rowBase = cy * GRID_CELL_WIDTH;
    for (uint16_t cx = minCellX; cx <= maxCellX; ++cx) {
      int32_t nodeIdx = cell_heads[rowBase + cx];

      while (nodeIdx != -1) {
        const GridNode &node = nodes[nodeIdx];
        queryResult.push_back(node.entity);
        nodeIdx = node.next;
      }
    }
  }
  return queryResult;
}
void SpatialGrid::rebuild_grid(Engine *engine) {
  PROFILE_FUNCTION();
  clearAll();

  // Re-add all entities to the grid
  int container_count = engine->entityManager.containers.size();

  // Serial for now to avoid contention on the linked list (or use locks?
  // Node allocation is thread-unsafe if not atomic).
  // The 'add' method modifies global 'cell_heads' and 'nodes' vector.
  // It is NOT thread-safe without locking.
  // Previous version used atomic fetch_add on cells.
  // Intrusive list generally hard to build in parallel without per-cell locks.
  // Given we have 50k entities, serial add might be slow?
  // 50k simple appends is fast. 0.5ms?
  // Let's try serial query first.

  for (int i = 0; i < container_count; ++i) {
    auto container = engine->entityManager.containers[i].get();
    if (!container || container->count == 0)
      continue;

    int count = container->count;
    for (int j = 0; j < count; ++j) {
      float x = container->x_positions[j];
      float y = container->y_positions[j];
      EntityRef ref = {(uint32_t)i, (uint32_t)j};

      int32_t nodeIdx = engine->grid.add(ref, x, y);
      container->grid_node_indices[j] = nodeIdx;

      // Update cell coords
      uint16_t cx, cy;
      engine->grid.getCellCoords(x, y, cx, cy);
      container->cell_x[j] = cx;
      container->cell_y[j] = cy;
    }
  }
}
