#include "../game/ATMEngine.h"
#include <algorithm>
#include <cmath>
#include <execution>
#include <future>
#include <numeric>
#include <thread>

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

void EntityContainer::updateVisible(const std::vector<uint32_t> &active_indices,
                                    float delta_time) {
  // Compatibility fallback: preserve old behavior when a custom visible-only
  // update path is not implemented by the container.
  if (!active_indices.empty()) {
    update(delta_time);
  }
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
  flags[index] = static_cast<uint8_t>(EntityFlag::VISIBLE);
  entity_ids[index] = INVALID_ID;
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
    entity_ids[index] = entity_ids[last];
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

int EntityManager::registerEntityType(EntityContainer *container,
                                      ObjectRuntimeKind kind) {
  PROFILE_FUNCTION();
  if (!container) {
    return -1;
  }

  int type_id = containers.size();
  container->type_id = type_id;
  containers.emplace_back(container);
  type_states.emplace_back();
  type_states.back().runtime_kind = kind;

  switch (kind) {
  case ObjectRuntimeKind::Dynamic:
    dynamic_type_ids.push_back(type_id);
    break;
  case ObjectRuntimeKind::Static:
    static_type_ids.push_back(type_id);
    container->containerFlag &= ~(uint8_t)ContainerFlag::UPDATEABLE;
    break;
  case ObjectRuntimeKind::Hybrid:
    hybrid_type_ids.push_back(type_id);
    break;
  default:
    dynamic_type_ids.push_back(type_id);
    break;
  }

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

int EntityManager::registerEntityType(EntityContainer *container) {
  return registerEntityType(container, ObjectRuntimeKind::Dynamic);
}

int EntityManager::registerDynamicEntityType(EntityContainer *container) {
  return registerEntityType(container, ObjectRuntimeKind::Dynamic);
}

int EntityManager::registerStaticEntityType(EntityContainer *container) {
  return registerEntityType(container, ObjectRuntimeKind::Static);
}

int EntityManager::registerHybridEntityType(EntityContainer *container) {
  return registerEntityType(container, ObjectRuntimeKind::Hybrid);
}

EntityHandle EntityManager::createEntityHandle(int type_id) {
  PROFILE_FUNCTION();
  if (type_id < 0 || type_id >= static_cast<int>(containers.size())) {
    return {};
  }

  auto *container = containers[type_id].get();
  if (!container) {
    return {};
  }

  const uint32_t slot = container->createEntity();
  if (slot == INVALID_ID) {
    return {};
  }

  auto &state = type_states[type_id];
  uint32_t entity_id = INVALID_ID;
  if (!state.free_entity_ids.empty()) {
    entity_id = state.free_entity_ids.back();
    state.free_entity_ids.pop_back();
  } else {
    entity_id = state.next_entity_id++;
  }

  if (entity_id >= state.entity_to_slot.size()) {
    state.entity_to_slot.resize(entity_id + 1, INVALID_SLOT);
  }
  if (entity_id >= state.entity_generations.size()) {
    state.entity_generations.resize(entity_id + 1, 1);
  }
  if (state.entity_generations[entity_id] == 0) {
    state.entity_generations[entity_id] = 1;
  }

  state.entity_to_slot[entity_id] = slot;
  container->entity_ids[slot] = entity_id;
  next_entity_id++;

  return EntityHandle{static_cast<uint32_t>(type_id), entity_id,
                      state.entity_generations[entity_id]};
}

uint32_t EntityManager::createEntity(int type_id) {
  const EntityHandle handle = createEntityHandle(type_id);
  uint32_t slot = INVALID_ID;
  if (!resolveEntitySlot(handle, slot)) {
    return INVALID_ID;
  }
  return slot;
}

bool EntityManager::resolveEntitySlot(const EntityHandle &handle,
                                      uint32_t &outSlot) const {
  outSlot = INVALID_SLOT;
  if (!handle.isValid()) {
    return false;
  }

  if (handle.type >= containers.size() || handle.type >= type_states.size()) {
    return false;
  }

  const auto *container = containers[handle.type].get();
  if (!container) {
    return false;
  }

  const auto &state = type_states[handle.type];
  if (handle.entity >= state.entity_to_slot.size() ||
      handle.entity >= state.entity_generations.size()) {
    return false;
  }

  if (state.entity_generations[handle.entity] != handle.generation) {
    return false;
  }

  const uint32_t slot = state.entity_to_slot[handle.entity];
  if (slot == INVALID_SLOT || slot >= static_cast<uint32_t>(container->count)) {
    return false;
  }

  if (container->entity_ids[slot] != handle.entity) {
    return false;
  }

  outSlot = slot;
  return true;
}

bool EntityManager::isHandleValid(const EntityHandle &handle) const {
  uint32_t slot = INVALID_SLOT;
  return resolveEntitySlot(handle, slot);
}

bool EntityManager::removeEntity(const EntityHandle &handle, SpatialGrid *grid) {
  uint32_t slot = INVALID_SLOT;
  if (!resolveEntitySlot(handle, slot)) {
    return false;
  }

  removeEntity(slot, static_cast<int>(handle.type), grid);
  return true;
}

void EntityManager::removeEntity(uint32_t index, int type_id, SpatialGrid *grid) {
  PROFILE_FUNCTION();
  if (type_id < 0 || type_id >= static_cast<int>(containers.size()) ||
      type_id >= static_cast<int>(type_states.size())) {
    return;
  }

  auto *container = containers[type_id].get();
  if (!container || index >= static_cast<uint32_t>(container->count)) {
    return;
  }

  auto &state = type_states[type_id];
  const ObjectRuntimeKind runtime_kind = state.runtime_kind;

  const uint32_t last = static_cast<uint32_t>(container->count - 1);
  const uint32_t removed_entity_id = container->entity_ids[index];
  const bool swapped = index < last;
  const uint32_t moved_entity_id =
      swapped ? container->entity_ids[last] : INVALID_ID;

  const int32_t removed_grid_node = container->grid_node_indices[index];
  const int32_t moved_grid_node =
      swapped ? container->grid_node_indices[last] : -1;

  container->removeEntity(index);

  if (grid && runtime_kind != ObjectRuntimeKind::Static) {
    if (removed_grid_node != -1) {
      grid->remove(removed_grid_node);
    }

    if (swapped && moved_grid_node != -1) {
      grid->updateNodeEntity(moved_grid_node,
                             EntityRef{static_cast<uint32_t>(type_id), index});
    }
  }

  if (removed_entity_id != INVALID_ID) {
    if (removed_entity_id < state.entity_to_slot.size()) {
      state.entity_to_slot[removed_entity_id] = INVALID_SLOT;
    }
    if (removed_entity_id < state.entity_generations.size()) {
      uint16_t &generation = state.entity_generations[removed_entity_id];
      generation = static_cast<uint16_t>(generation + 1u);
      if (generation == 0) {
        generation = 1;
      }
    }
    state.free_entity_ids.push_back(removed_entity_id);
  }

  if (swapped && moved_entity_id != INVALID_ID &&
      moved_entity_id < state.entity_to_slot.size()) {
    state.entity_to_slot[moved_entity_id] = index;
  }
}

ObjectRuntimeKind EntityManager::getRuntimeKind(int type_id) const {
  if (type_id < 0 || type_id >= static_cast<int>(type_states.size())) {
    return ObjectRuntimeKind::Dynamic;
  }
  return type_states[type_id].runtime_kind;
}

void EntityManager::updateDynamic(float delta_time) {
  PROFILE_FUNCTION();
  for (const int type_id : dynamic_type_ids) {
    if (type_id < 0 || type_id >= static_cast<int>(containers.size())) {
      continue;
    }
    auto *container = containers[type_id].get();
    if (!container) {
      continue;
    }
    if (container->containerFlag & (uint8_t)ContainerFlag::UPDATEABLE) {
      container->update(delta_time);
    }
  }
}

void EntityManager::updateHybrid(Engine *engine, float delta_time, float x1,
                                 float y1, float x2, float y2) {
  PROFILE_FUNCTION();
  if (!engine || hybrid_type_ids.empty()) {
    return;
  }

  std::vector<EntityRef> &active_refs = engine->grid.queryRect(x1, y1, x2, y2);
  if (active_refs.empty()) {
    return;
  }

  static thread_local std::vector<std::vector<uint32_t>> active_indices_by_type;
  static thread_local std::vector<int> touched_types;

  if (active_indices_by_type.size() < containers.size()) {
    active_indices_by_type.resize(containers.size());
  }
  touched_types.clear();

  for (const EntityRef &ref : active_refs) {
    if (ref.type >= type_states.size()) {
      continue;
    }
    if (type_states[ref.type].runtime_kind != ObjectRuntimeKind::Hybrid) {
      continue;
    }

    auto &indices = active_indices_by_type[ref.type];
    if (indices.empty()) {
      touched_types.push_back(static_cast<int>(ref.type));
    }
    indices.push_back(ref.index);
  }

  for (const int type_id : touched_types) {
    auto *container = containers[type_id].get();
    if (!container) {
      active_indices_by_type[type_id].clear();
      continue;
    }

    if (container->containerFlag & (uint8_t)ContainerFlag::UPDATEABLE) {
      container->updateVisible(active_indices_by_type[type_id], delta_time);
    }
    active_indices_by_type[type_id].clear();
  }
}

void EntityManager::update(float delta_time) {
  updateDynamic(delta_time);
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

int64_t StaticChunkCache::makeChunkKey(int32_t chunk_x, int32_t chunk_y) {
  return (static_cast<int64_t>(chunk_x) << 32) ^
         static_cast<uint32_t>(chunk_y);
}

void StaticChunkCache::clear() {
  chunk_index_by_key.clear();
  chunks.clear();
  visible_chunk_indices.clear();
  needs_full_rebuild = true;
}

void StaticChunkCache::markAllDirty() {
  for (auto &chunk : chunks) {
    chunk.dirty = true;
  }
}

int32_t StaticChunkCache::worldToChunk(float world_value) const {
  return static_cast<int32_t>(
      std::floor(world_value / static_cast<float>(STATIC_CHUNK_SIZE)));
}

void StaticChunkCache::rebuildRefs(Engine *engine) {
  PROFILE_FUNCTION();
  chunk_index_by_key.clear();
  chunks.clear();
  visible_chunk_indices.clear();

  if (!engine) {
    needs_full_rebuild = false;
    return;
  }

  auto ensureChunk = [&](int32_t chunk_x, int32_t chunk_y) -> StaticChunk & {
    const int64_t key = makeChunkKey(chunk_x, chunk_y);
    auto it = chunk_index_by_key.find(key);
    if (it != chunk_index_by_key.end()) {
      return chunks[it->second];
    }

    const size_t index = chunks.size();
    chunk_index_by_key[key] = index;
    chunks.push_back(StaticChunk{});
    chunks.back().chunk_x = chunk_x;
    chunks.back().chunk_y = chunk_y;
    chunks.back().dirty = true;
    return chunks.back();
  };

  const std::vector<int> &static_types = engine->entityManager.getStaticTypeIds();
  for (const int type_id : static_types) {
    if (type_id < 0 ||
        type_id >= static_cast<int>(engine->entityManager.containers.size())) {
      continue;
    }

    auto *container = engine->entityManager.containers[type_id].get();
    if (!container || container->count <= 0) {
      continue;
    }

    if (!(container->containerFlag & (uint8_t)ContainerFlag::RENDERABLE)) {
      continue;
    }

    for (int i = 0; i < container->count; ++i) {
      const int32_t chunk_x = worldToChunk(container->x_positions[i]);
      const int32_t chunk_y = worldToChunk(container->y_positions[i]);
      StaticChunk &chunk = ensureChunk(chunk_x, chunk_y);
      chunk.refs.push_back(
          EntityRef{static_cast<uint32_t>(type_id), static_cast<uint32_t>(i)});
      chunk.dirty = true;
    }
  }

  needs_full_rebuild = false;
}

std::vector<size_t> &StaticChunkCache::queryVisible(float x1, float y1, float x2,
                                                    float y2) {
  visible_chunk_indices.clear();
  if (chunks.empty()) {
    return visible_chunk_indices;
  }

  const int32_t min_chunk_x = worldToChunk(x1);
  const int32_t min_chunk_y = worldToChunk(y1);
  const int32_t max_chunk_x = worldToChunk(x2);
  const int32_t max_chunk_y = worldToChunk(y2);

  for (int32_t chunk_y = min_chunk_y; chunk_y <= max_chunk_y; ++chunk_y) {
    for (int32_t chunk_x = min_chunk_x; chunk_x <= max_chunk_x; ++chunk_x) {
      const int64_t key = makeChunkKey(chunk_x, chunk_y);
      auto it = chunk_index_by_key.find(key);
      if (it != chunk_index_by_key.end()) {
        visible_chunk_indices.push_back(it->second);
      }
    }
  }

  return visible_chunk_indices;
}

// Process entities marked for removal
void process_pending_removals(Engine *engine) {
  PROFILE_FUNCTION();

  if (engine->pending_removals.empty()) {
    return;
  }

  bool static_changed = false;

  // Swap-delete invalidates higher indices first; remove in descending index
  // order per type to avoid skipping entities.
  std::sort(engine->pending_removals.begin(), engine->pending_removals.end(),
            [](const EntityRef &a, const EntityRef &b) {
              if (a.type != b.type) {
                return a.type < b.type;
              }
              return a.index > b.index;
            });

  for (const auto &ref : engine->pending_removals) {
    if (ref.type >= engine->entityManager.containers.size()) {
      continue;
    }

    if (engine->entityManager.getRuntimeKind(static_cast<int>(ref.type)) ==
        ObjectRuntimeKind::Static) {
      static_changed = true;
    }

    engine->entityManager.removeEntity(ref.index, static_cast<int>(ref.type),
                                       &engine->grid);
  }

  if (static_changed) {
    engine->staticChunkCache.markNeedsFullRebuild();
  }

  engine->pending_removals.clear();
}

// Update the engine state
void engine_update(Engine *engine) {
  PROFILE_FUNCTION();

  // Calculate delta time
  Uint64 current_time = SDL_GetTicks();
  float delta_time = (current_time - engine->last_frame_time) / 1000.0f;
  if (delta_time <= 0.0f) {
    delta_time = 0.0001f;
  }
  engine->last_frame_time = current_time;

  // Smooth FPS calculation
  engine->fps = 0.95f * engine->fps + 0.05f * (1.0f / delta_time);

  // Option B pipeline:
  // 1) Dynamic types are always updated globally.
  engine->entityManager.updateDynamic(delta_time);

  // 2) Hybrid types update only inside camera activation bounds.
  static constexpr float HYBRID_ACTIVATION_MARGIN = 150.0f;
  const float x1 = engine->camera.x - engine->camera.width / 2.0f;
  const float y1 = engine->camera.y - engine->camera.height / 2.0f;
  const float x2 = engine->camera.x + engine->camera.width / 2.0f;
  const float y2 = engine->camera.y + engine->camera.height / 2.0f;
  engine->entityManager.updateHybrid(
      engine, delta_time, x1 - HYBRID_ACTIVATION_MARGIN,
      y1 - HYBRID_ACTIVATION_MARGIN, x2 + HYBRID_ACTIVATION_MARGIN,
      y2 + HYBRID_ACTIVATION_MARGIN);

  // Process pending removals
  process_pending_removals(engine);
}

void engine_update_entity_types(Engine *engine, float delta_time) {
  (void)engine;
  (void)delta_time;
  // Backward-compatibility no-op:
  // engine_update() now owns the full dynamic/hybrid update pipeline.
}

int engine_register_dynamic_type(Engine *engine, EntityContainer *container) {
  if (!engine) {
    return -1;
  }
  return engine->entityManager.registerDynamicEntityType(container);
}

int engine_register_static_type(Engine *engine, EntityContainer *container) {
  if (!engine) {
    return -1;
  }
  const int type_id = engine->entityManager.registerStaticEntityType(container);
  engine->staticChunkCache.markNeedsFullRebuild();
  return type_id;
}

int engine_register_hybrid_type(Engine *engine, EntityContainer *container) {
  if (!engine) {
    return -1;
  }
  return engine->entityManager.registerHybridEntityType(container);
}

EntityHandle engine_create_entity(Engine *engine, int type_id) {
  if (!engine) {
    return {};
  }

  EntityHandle handle = engine->entityManager.createEntityHandle(type_id);
  uint32_t slot = INVALID_SLOT;
  if (!engine->entityManager.resolveEntitySlot(handle, slot)) {
    return {};
  }

  auto *container = engine->entityManager.containers[type_id].get();
  if (!container) {
    return {};
  }

  const ObjectRuntimeKind kind = engine->entityManager.getRuntimeKind(type_id);
  if (kind == ObjectRuntimeKind::Static) {
    engine->staticChunkCache.markNeedsFullRebuild();
    return handle;
  }

  const float x = container->x_positions[slot];
  const float y = container->y_positions[slot];
  const EntityRef ref{static_cast<uint32_t>(type_id), slot};
  const int32_t node_idx = engine->grid.add(ref, x, y);
  container->grid_node_indices[slot] = node_idx;
  uint16_t cx = 0;
  uint16_t cy = 0;
  engine->grid.getCellCoords(x, y, cx, cy);
  container->cell_x[slot] = cx;
  container->cell_y[slot] = cy;

  return handle;
}

void engine_destroy_entity(Engine *engine, const EntityHandle &handle) {
  if (!engine || !handle.isValid()) {
    return;
  }

  if (handle.type >= engine->entityManager.containers.size()) {
    return;
  }

  const ObjectRuntimeKind kind =
      engine->entityManager.getRuntimeKind(static_cast<int>(handle.type));
  if (engine->entityManager.removeEntity(handle, &engine->grid) &&
      kind == ObjectRuntimeKind::Static) {
    engine->staticChunkCache.markNeedsFullRebuild();
  }
}

bool engine_is_handle_valid(Engine *engine, const EntityHandle &handle) {
  if (!engine) {
    return false;
  }
  return engine->entityManager.isHandleValid(handle);
}

bool engine_set_entity_position(Engine *engine, const EntityHandle &handle,
                                float x, float y) {
  if (!engine) {
    return false;
  }

  uint32_t slot = INVALID_SLOT;
  if (!engine->entityManager.resolveEntitySlot(handle, slot)) {
    return false;
  }

  auto *container = engine->entityManager.containers[handle.type].get();
  if (!container) {
    return false;
  }

  container->x_positions[slot] = x;
  container->y_positions[slot] = y;

  const ObjectRuntimeKind kind =
      engine->entityManager.getRuntimeKind(static_cast<int>(handle.type));
  if (kind == ObjectRuntimeKind::Static) {
    engine->staticChunkCache.markNeedsFullRebuild();
    return true;
  }

  int32_t node_idx = container->grid_node_indices[slot];
  const EntityRef ref{handle.type, slot};
  if (node_idx == -1) {
    node_idx = engine->grid.add(ref, x, y);
    container->grid_node_indices[slot] = node_idx;
  } else {
    engine->grid.move(node_idx, x, y);
  }

  uint16_t cx = 0;
  uint16_t cy = 0;
  engine->grid.getCellCoords(x, y, cx, cy);
  container->cell_x[slot] = cx;
  container->cell_y[slot] = cy;
  return true;
}

bool engine_set_entity_visible(Engine *engine, const EntityHandle &handle,
                               bool visible) {
  if (!engine) {
    return false;
  }

  uint32_t slot = INVALID_SLOT;
  if (!engine->entityManager.resolveEntitySlot(handle, slot)) {
    return false;
  }

  auto *container = engine->entityManager.containers[handle.type].get();
  if (!container) {
    return false;
  }

  if (visible) {
    container->flags[slot] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
  } else {
    container->flags[slot] &= ~static_cast<uint8_t>(EntityFlag::VISIBLE);
  }

  if (engine->entityManager.getRuntimeKind(static_cast<int>(handle.type)) ==
      ObjectRuntimeKind::Static) {
    engine->staticChunkCache.markNeedsFullRebuild();
  }
  return true;
}

bool engine_set_entity_z_index(Engine *engine, const EntityHandle &handle,
                               uint8_t z_index) {
  if (!engine) {
    return false;
  }

  uint32_t slot = INVALID_SLOT;
  if (!engine->entityManager.resolveEntitySlot(handle, slot)) {
    return false;
  }

  if (handle.type >= engine->entityManager.containers.size()) {
    return false;
  }

  auto *container = engine->entityManager.containers[handle.type].get();
  auto *renderable = dynamic_cast<RenderableEntityContainer *>(container);
  if (!renderable || slot >= static_cast<uint32_t>(renderable->count)) {
    return false;
  }

  renderable->z_indices[slot] = z_index;
  if (engine->entityManager.getRuntimeKind(static_cast<int>(handle.type)) ==
      ObjectRuntimeKind::Static) {
    engine->staticChunkCache.markNeedsFullRebuild();
  }
  return true;
}

void engine_mark_static_dirty(Engine *engine) {
  if (!engine) {
    return;
  }
  engine->staticChunkCache.markNeedsFullRebuild();
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
  if (engine->entityManager.getRuntimeKind(type_id) ==
      ObjectRuntimeKind::Static) {
    engine->staticChunkCache.markNeedsFullRebuild();
  }
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

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    SDL_Log("SDL_Init failed: %s", SDL_GetError());
  }

  // Use placement new to properly initialize C++ members
  new (&engine->grid) SpatialGrid();
  new (&engine->entityManager) EntityManager();
  new (&engine->pending_removals) std::vector<EntityRef>();
  new (&engine->staticChunkCache) StaticChunkCache();
  new (&engine->renderBatchManager) RenderBatchManager(8);

  // Create window
#ifdef __ANDROID__
  Uint32 windowFlags = SDL_WINDOW_OPENGL;
#else
  Uint32 windowFlags = SDL_WINDOW_HIGH_PIXEL_DENSITY;
#endif

  engine->window = SDL_CreateWindow("2D Game Engine", window_width,
                                    window_height, windowFlags);
  if (!engine->window) {
    SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
    engine->renderBatchManager.~RenderBatchManager();
    engine->staticChunkCache.~StaticChunkCache();
    engine->entityManager.~EntityManager();
    engine->pending_removals.~vector();
    engine->grid.~SpatialGrid();
    free(engine);
    return NULL;
  }

  // Initialize SDL renderer
  engine->renderer = SDL_CreateRenderer(engine->window, NULL);
  if (!engine->renderer) {
    SDL_Log("SDL_CreateRenderer failed: %s", SDL_GetError());
    SDL_DestroyWindow(engine->window);
    engine->renderBatchManager.~RenderBatchManager();
    engine->staticChunkCache.~StaticChunkCache();
    engine->entityManager.~EntityManager();
    engine->pending_removals.~vector();
    engine->grid.~SpatialGrid();
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
  engine->staticChunkCache.~StaticChunkCache();
  engine->entityManager.~EntityManager();
  engine->pending_removals.~vector();
  engine->grid.~SpatialGrid();

  // Destroy SDL resources
  if (engine->renderer) {
    SDL_DestroyRenderer(engine->renderer);
  }

  SDL_DestroyWindow(engine->window);

  // Free the engine struct
  free(engine);

  SDL_Quit();
}
int engine_register_texture(Engine *engine, SDL_Surface *surface, int x, int y,
                            int width, int height) {
  PROFILE_FUNCTION();
  // Use SDL texture management
  return engine->atlas.registerTexture(surface, x, y, width, height);
}

static inline bool isEntityVisible(const EntityContainer *container,
                                   uint32_t index) {
  if (!container || index >= static_cast<uint32_t>(container->count)) {
    return false;
  }
  return (container->flags[index] & static_cast<uint8_t>(EntityFlag::VISIBLE)) !=
         0;
}

static void appendQuadToBuffers(std::vector<SDL_Vertex> &vertices,
                                std::vector<int> &indices, float x, float y,
                                float w, float h, float rotation_radians,
                                const SDL_FRect &tex_region) {
  const int base_vert = static_cast<int>(vertices.size());
  SDL_Vertex v;
  v.color = {1, 1, 1, 1};

  const float cx = x + w * 0.5f;
  const float cy = y + h * 0.5f;
  const float c = std::cos(rotation_radians);
  const float s = std::sin(rotation_radians);
  auto rotate = [&](float vx, float vy) -> SDL_FPoint {
    return {cx + (vx - cx) * c - (vy - cy) * s,
            cy + (vx - cx) * s + (vy - cy) * c};
  };

  v.position = rotate(x, y);
  v.tex_coord = {tex_region.x, tex_region.y};
  vertices.push_back(v);

  v.position = rotate(x + w, y);
  v.tex_coord = {tex_region.x + tex_region.w, tex_region.y};
  vertices.push_back(v);

  v.position = rotate(x + w, y + h);
  v.tex_coord = {tex_region.x + tex_region.w, tex_region.y + tex_region.h};
  vertices.push_back(v);

  v.position = rotate(x, y + h);
  v.tex_coord = {tex_region.x, tex_region.y + tex_region.h};
  vertices.push_back(v);

  indices.push_back(base_vert);
  indices.push_back(base_vert + 1);
  indices.push_back(base_vert + 2);
  indices.push_back(base_vert);
  indices.push_back(base_vert + 2);
  indices.push_back(base_vert + 3);
}

static void rebuildStaticChunkMesh(Engine *engine, StaticChunk &chunk) {
  chunk.vertices.clear();
  chunk.indices.clear();

  struct SortableEntity {
    uint64_t sort_key;
    EntityRef ref;
    bool operator<(const SortableEntity &other) const {
      return sort_key < other.sort_key;
    }
  };

  std::vector<SortableEntity> sortable_entities;
  sortable_entities.reserve(chunk.refs.size());

  for (const EntityRef &ref : chunk.refs) {
    if (ref.type >= engine->entityManager.containers.size()) {
      continue;
    }

    auto *base = engine->entityManager.containers[ref.type].get();
    if (!base || ref.index >= static_cast<uint32_t>(base->count)) {
      continue;
    }
    if (!(base->containerFlag & (uint8_t)ContainerFlag::RENDERABLE)) {
      continue;
    }
    if (!isEntityVisible(base, ref.index)) {
      continue;
    }

    auto *renderable = static_cast<RenderableEntityContainer *>(base);
    const uint64_t sort_key =
        (static_cast<uint64_t>(renderable->z_indices[ref.index]) << 56) |
        (static_cast<uint64_t>(ref.type) << 48) |
        static_cast<uint64_t>(ref.index);
    sortable_entities.push_back(SortableEntity{sort_key, ref});
  }

  std::sort(sortable_entities.begin(), sortable_entities.end());

  chunk.vertices.reserve(sortable_entities.size() * 4);
  chunk.indices.reserve(sortable_entities.size() * 6);

  for (const SortableEntity &se : sortable_entities) {
    const EntityRef &ref = se.ref;
    auto *renderable = static_cast<RenderableEntityContainer *>(
        engine->entityManager.containers[ref.type].get());
    const float x = renderable->x_positions[ref.index];
    const float y = renderable->y_positions[ref.index];
    const float w = renderable->widths[ref.index];
    const float h = renderable->heights[ref.index];
    const float angle = renderable->rotations[ref.index];
    SDL_FRect tex_region = engine->atlas.getRegion(renderable->texture_ids[ref.index]);
    appendQuadToBuffers(chunk.vertices, chunk.indices, x, y, w, h, angle,
                        tex_region);
  }

  chunk.dirty = false;
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

  // Pre-computed sort key for zero-cost comparisons during sort.
  struct SortableEntity {
    uint64_t sort_key;
    EntityRef ref;

    bool operator<(const SortableEntity &other) const {
      return sort_key < other.sort_key;
    }
  };

  // Reuse buffers across frames to avoid allocations.
  static thread_local std::vector<SortableEntity> sortable_entities;
  static thread_local std::vector<SDL_Vertex> unified_vertices;
  static thread_local std::vector<int> unified_indices;

  sortable_entities.clear();
  unified_vertices.clear();
  unified_indices.clear();

  // Static pass: chunked cached geometry.
  if (engine->staticChunkCache.needs_full_rebuild) {
    engine->staticChunkCache.rebuildRefs(engine);
  }

  std::vector<size_t> &visible_static_chunks =
      engine->staticChunkCache.queryVisible(x1, y1, x2, y2);
  for (size_t chunk_idx : visible_static_chunks) {
    auto &chunks = engine->staticChunkCache.getChunks();
    if (chunk_idx >= chunks.size()) {
      continue;
    }

    StaticChunk &chunk = chunks[chunk_idx];
    if (chunk.dirty) {
      rebuildStaticChunkMesh(engine, chunk);
    }
    if (chunk.vertices.empty()) {
      continue;
    }

    const int base_vert = static_cast<int>(unified_vertices.size());
    unified_vertices.reserve(unified_vertices.size() + chunk.vertices.size());
    for (const SDL_Vertex &world_v : chunk.vertices) {
      SDL_Vertex v = world_v;
      v.position.x -= x1;
      v.position.y -= y1;
      unified_vertices.push_back(v);
    }
    unified_indices.reserve(unified_indices.size() + chunk.indices.size());
    for (int idx : chunk.indices) {
      unified_indices.push_back(base_vert + idx);
    }
  }

  // Dynamic + hybrid pass: moving-grid visible query.
  std::vector<EntityRef> &visible_entities =
      engine->grid.queryRect(x1 - 50, y1 - 50, x2 + 50, y2 + 50);
  sortable_entities.reserve(visible_entities.size());
  for (const auto &entity : visible_entities) {
    if (entity.type >= engine->entityManager.containers.size()) {
      continue;
    }
    if (engine->entityManager.getRuntimeKind(static_cast<int>(entity.type)) ==
        ObjectRuntimeKind::Static) {
      continue;
    }

    auto *base = engine->entityManager.containers[entity.type].get();
    if (!base || entity.index >= static_cast<uint32_t>(base->count)) {
      continue;
    }
    if (!(base->containerFlag & (uint8_t)ContainerFlag::RENDERABLE)) {
      continue;
    }
    if (!isEntityVisible(base, entity.index)) {
      continue;
    }

    auto *renderable = static_cast<RenderableEntityContainer *>(base);
    const uint64_t key =
        (static_cast<uint64_t>(renderable->z_indices[entity.index]) << 56) |
        (static_cast<uint64_t>(entity.type) << 48) |
        static_cast<uint64_t>(entity.index);
    sortable_entities.push_back({key, entity});
  }

  std::sort(sortable_entities.begin(), sortable_entities.end());

  for (const auto &se : sortable_entities) {
    const auto &entity = se.ref;
    auto *rCont = static_cast<RenderableEntityContainer *>(
        engine->entityManager.containers[entity.type].get());
    if (!rCont || entity.index >= static_cast<uint32_t>(rCont->count)) {
      continue;
    }

    float x = rCont->x_positions[entity.index] - x1;
    float y = rCont->y_positions[entity.index] - y1;
    float w = rCont->widths[entity.index];
    float h = rCont->heights[entity.index];

    if (x + w < 0 || x > engine->camera.width || y + h < 0 ||
        y > engine->camera.height)
      continue;

    SDL_FRect texRegion =
        engine->atlas.getRegion(rCont->texture_ids[entity.index]);

    appendQuadToBuffers(unified_vertices, unified_indices, x, y, w, h,
                        rCont->rotations[entity.index], texRegion);
  }

  // Single draw call with atlas texture.
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
    if (engine->entityManager.getRuntimeKind(i) == ObjectRuntimeKind::Static) {
      continue;
    }

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
