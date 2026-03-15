#include "ATMEngine.h"
#include <algorithm>
#include <cmath>

EntityContainer::EntityContainer(int typeId, uint8_t defaultLayer,
                                 int initialCapacity)
    : flags(initialCapacity), entity_ids(initialCapacity, INVALID_ID),
      parent_ids(initialCapacity, INVALID_ID),
      first_child_ids(initialCapacity, INVALID_ID),
      next_sibling_ids(initialCapacity, INVALID_ID),
      x_positions(initialCapacity, 0.0f), y_positions(initialCapacity, 0.0f),
      cell_x(initialCapacity), cell_y(initialCapacity),
      grid_node_indices(initialCapacity, -1),
      slot_to_id(initialCapacity, INVALID_ID),
      id_to_slot(initialCapacity, INVALID_ID), free_ids(), next_id(0),
      containerFlag((uint8_t)ContainerFlag::UPDATEABLE), type_id(typeId),
      default_layer(defaultLayer), capacity(initialCapacity), count(0) {
  free_ids.reserve(initialCapacity);
}

EntityContainer::~EntityContainer() {
  PROFILE_FUNCTION();
}

void EntityContainer::updateVisible(const std::vector<uint32_t> &active_slots,
                                    float delta_time) {
  if (!active_slots.empty()) {
    update(delta_time);
  }
}

EntityHandle EntityContainer::createEntity() {
  if (count >= capacity) {
    resizeArrays(capacity > 0 ? capacity * 2 : 1);
  }

  const uint32_t slot = count++;
  EntityHandle id = INVALID_ID;
  if (!free_ids.empty()) {
    id = free_ids.back();
    free_ids.pop_back();
  } else {
    id = next_id++;
  }

  if (id >= id_to_slot.size()) {
    id_to_slot.resize(id + 1, INVALID_ID);
  }

  id_to_slot[id] = slot;
  slot_to_id[slot] = id;
  entity_ids[slot] = id;
  x_positions[slot] = 0.0f;
  y_positions[slot] = 0.0f;
  flags[slot] = static_cast<uint8_t>(EntityFlag::NONE);
  parent_ids[slot] = INVALID_ID;
  first_child_ids[slot] = INVALID_ID;
  next_sibling_ids[slot] = INVALID_ID;
  cell_x[slot] = 0;
  cell_y[slot] = 0;
  grid_node_indices[slot] = -1;
  return id;
}

void EntityContainer::removeEntity(EntityHandle id) {
  PROFILE_FUNCTION();
  if (!isAlive(id)) {
    return;
  }

  const uint32_t slot = id_to_slot[id];
  const uint32_t last = static_cast<uint32_t>(count - 1);
  if (slot != last) {
    const EntityHandle moved_id = slot_to_id[last];
    swapSlots(slot, last);
    id_to_slot[moved_id] = slot;
  }

  count--;
  id_to_slot[id] = INVALID_ID;
  free_ids.push_back(id);
  if (count < capacity) {
    slot_to_id[count] = INVALID_ID;
    entity_ids[count] = INVALID_ID;
    grid_node_indices[count] = -1;
  }
}

void EntityContainer::swapSlots(uint32_t a, uint32_t b) {
  if (a == b)
    return;

  std::swap(flags[a], flags[b]);
  std::swap(entity_ids[a], entity_ids[b]);
  std::swap(parent_ids[a], parent_ids[b]);
  std::swap(first_child_ids[a], first_child_ids[b]);
  std::swap(next_sibling_ids[a], next_sibling_ids[b]);
  std::swap(x_positions[a], x_positions[b]);
  std::swap(y_positions[a], y_positions[b]);
  std::swap(cell_x[a], cell_x[b]);
  std::swap(cell_y[a], cell_y[b]);
  std::swap(grid_node_indices[a], grid_node_indices[b]);
  std::swap(slot_to_id[a], slot_to_id[b]);
}

EntityHandle EntityContainer::getStableId(uint32_t slot) const {
  if (slot >= static_cast<uint32_t>(count))
    return INVALID_ID;
  return slot_to_id[slot];
}

uint32_t EntityContainer::getSlot(EntityHandle id) const {
  if (id >= id_to_slot.size())
    return INVALID_ID;
  return id_to_slot[id];
}

bool EntityContainer::isAlive(EntityHandle id) const {
  return id < id_to_slot.size() && id_to_slot[id] != INVALID_ID;
}

void EntityContainer::resizeArrays(int newCapacity) {
  PROFILE_FUNCTION();
  if (newCapacity <= capacity)
    return;

  flags.resize(newCapacity, count);
  entity_ids.resize(newCapacity, count, INVALID_ID);
  parent_ids.resize(newCapacity, count, INVALID_ID);
  first_child_ids.resize(newCapacity, count, INVALID_ID);
  next_sibling_ids.resize(newCapacity, count, INVALID_ID);
  x_positions.resize(newCapacity, count, 0.0f);
  y_positions.resize(newCapacity, count, 0.0f);
  cell_x.resize(newCapacity, count);
  cell_y.resize(newCapacity, count);
  grid_node_indices.resize(newCapacity, count, -1);
  slot_to_id.resize(newCapacity, count, INVALID_ID);
  capacity = newCapacity;
}

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
}

EntityHandle RenderableEntityContainer::createEntity() {
  const EntityHandle id = EntityContainer::createEntity();
  if (id == INVALID_ID)
    return INVALID_ID;

  const uint32_t slot = getSlot(id);
  widths[slot] = 0;
  heights[slot] = 0;
  texture_ids[slot] = 0;
  z_indices[slot] = 0;
  rotations[slot] = 0.0f;
  return id;
}

void RenderableEntityContainer::swapSlots(uint32_t a, uint32_t b) {
  if (a == b)
    return;

  std::swap(widths[a], widths[b]);
  std::swap(heights[a], heights[b]);
  std::swap(texture_ids[a], texture_ids[b]);
  std::swap(z_indices[a], z_indices[b]);
  std::swap(rotations[a], rotations[b]);
  EntityContainer::swapSlots(a, b);
}

void RenderableEntityContainer::resizeArrays(int newCapacity) {
  PROFILE_FUNCTION();
  if (newCapacity <= capacity)
    return;

  EntityContainer::resizeArrays(newCapacity);
  widths.resize(newCapacity, count);
  heights.resize(newCapacity, count);
  texture_ids.resize(newCapacity, count);
  z_indices.resize(newCapacity, count);
  rotations.resize(newCapacity, count, 0.0f);
}

Layer::Layer(int id) : layer_id(id), is_active(true) {
  PROFILE_FUNCTION();
}

void Layer::update(float delta_time) {
  PROFILE_FUNCTION();
  if (!is_active)
    return;
  for (auto *container : entity_containers) {
    if (container->containerFlag & (uint8_t)ContainerFlag::UPDATEABLE) {
      container->update(delta_time);
    }
  }
}

void Layer::addEntityContainer(EntityContainer *container) {
  PROFILE_FUNCTION();
  entity_containers.push_back(container);
}

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

  const int type_id = static_cast<int>(containers.size());
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
  }

  const uint8_t layer_index = container->getDefaultLayer();
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

EntityHandle EntityManager::createEntity(int type_id) {
  PROFILE_FUNCTION();
  if (type_id < 0 || type_id >= static_cast<int>(containers.size()))
    return INVALID_ID;

  const EntityHandle id = containers[type_id]->createEntity();
  if (id != INVALID_ID) {
    next_entity_id++;
  }
  return id;
}

void EntityManager::removeEntity(EntityHandle id, int type_id, SpatialGrid *grid) {
  PROFILE_FUNCTION();
  if (type_id < 0 || type_id >= static_cast<int>(containers.size()) ||
      type_id >= static_cast<int>(type_states.size())) {
    return;
  }

  auto *container = containers[type_id].get();
  if (!container || !container->isAlive(id)) {
    return;
  }

  const uint32_t slot = container->getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<uint32_t>(container->count)) {
    return;
  }

  if (grid && type_states[type_id].runtime_kind != ObjectRuntimeKind::Static) {
    const int32_t node_idx = container->grid_node_indices[slot];
    if (node_idx != -1) {
      grid->remove(node_idx);
    }
  }

  container->removeEntity(id);
}

bool EntityManager::isHandleValid(EntityHandle id, int type_id) const {
  if (type_id < 0 || type_id >= static_cast<int>(containers.size())) {
    return false;
  }
  const auto *container = containers[type_id].get();
  return container && container->isAlive(id);
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
    if (container &&
        (container->containerFlag & (uint8_t)ContainerFlag::UPDATEABLE)) {
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

  static thread_local std::vector<std::vector<uint32_t>> active_slots_by_type;
  static thread_local std::vector<int> touched_types;

  if (active_slots_by_type.size() < containers.size()) {
    active_slots_by_type.resize(containers.size());
  }
  touched_types.clear();

  for (const EntityRef &ref : active_refs) {
    if (ref.type >= type_states.size()) {
      continue;
    }
    if (type_states[ref.type].runtime_kind != ObjectRuntimeKind::Hybrid) {
      continue;
    }

    auto *container = containers[ref.type].get();
    if (!container) {
      continue;
    }

    const uint32_t slot = container->getSlot(ref.index);
    if (slot == INVALID_ID || slot >= static_cast<uint32_t>(container->count)) {
      continue;
    }
    if ((container->flags[slot] & static_cast<uint8_t>(EntityFlag::VISIBLE)) ==
        0) {
      continue;
    }

    auto &slots = active_slots_by_type[ref.type];
    if (slots.empty()) {
      touched_types.push_back(static_cast<int>(ref.type));
    }
    slots.push_back(slot);
  }

  for (const int type_id : touched_types) {
    auto *container = containers[type_id].get();
    if (!container) {
      active_slots_by_type[type_id].clear();
      continue;
    }

    if (container->containerFlag & (uint8_t)ContainerFlag::UPDATEABLE) {
      container->updateVisible(active_slots_by_type[type_id], delta_time);
    }
    active_slots_by_type[type_id].clear();
  }
}

void EntityManager::update(float delta_time) {
  updateDynamic(delta_time);
}

RenderBatch::RenderBatch(int textureId, int zIndex, int initialVertexCapacity)
    : texture_id(textureId), z_index(zIndex) {
  PROFILE_FUNCTION();
  vertices.reserve(initialVertexCapacity);
  indices.reserve(initialVertexCapacity * 3 / 2);
}

RenderBatch::~RenderBatch() {}

RenderBatch::RenderBatch(RenderBatch &&other) noexcept
    : texture_id(other.texture_id), z_index(other.z_index),
      vertices(std::move(other.vertices)), indices(std::move(other.indices)) {
  PROFILE_FUNCTION();
}

RenderBatch &RenderBatch::operator=(RenderBatch &&other) noexcept {
  PROFILE_FUNCTION();
  if (this != &other) {
    texture_id = other.texture_id;
    z_index = other.z_index;
    vertices = std::move(other.vertices);
    indices = std::move(other.indices);
  }
  return *this;
}

void RenderBatch::addQuad(float x, float y, float w, float h,
                          SDL_FRect tex_region) {
  const uint64_t base_vert = vertices.size();
  vertices.resize(vertices.size() + 4);
  indices.resize(indices.size() + 6);

  SDL_Vertex *v = &vertices[vertices.size() - 4];

  v[0].position = {x, y};
  v[0].color = {1, 1, 1, 1};
  v[0].tex_coord = {tex_region.x, tex_region.y};

  v[1].position = {x + w, y};
  v[1].color = {1, 1, 1, 1};
  v[1].tex_coord = {tex_region.x + tex_region.w, tex_region.y};

  v[2].position = {x + w, y + h};
  v[2].color = {1, 1, 1, 1};
  v[2].tex_coord = {tex_region.x + tex_region.w, tex_region.y + tex_region.h};

  v[3].position = {x, y + h};
  v[3].color = {1, 1, 1, 1};
  v[3].tex_coord = {tex_region.x, tex_region.y + tex_region.h};

  int *idx = &indices[indices.size() - 6];
  idx[0] = static_cast<int>(base_vert);
  idx[1] = static_cast<int>(base_vert + 1);
  idx[2] = static_cast<int>(base_vert + 2);
  idx[3] = static_cast<int>(base_vert);
  idx[4] = static_cast<int>(base_vert + 2);
  idx[5] = static_cast<int>(base_vert + 3);
}

void RenderBatch::clear() {
  PROFILE_FUNCTION();
  vertices.clear();
  indices.clear();
}

RenderBatchManager::RenderBatchManager(int initialBatchCount)
    : needsSort(false) {
  batches.reserve(initialBatchCount * 2);
}

void RenderBatchManager::addQuad(int textureId, int zIndex, float x, float y,
                                 float w, float h, SDL_FRect tex_region) {
  RenderBatch &batch = getBatch(textureId, zIndex);
  batch.addQuad(x, y, w, h, tex_region);
}

RenderBatch &RenderBatchManager::getBatch(int textureId, int zIndex) {
  BatchKey key = createKey(textureId, zIndex);
  auto it = batchMap.find(key);
  if (it != batchMap.end()) {
    return batches[it->second];
  }

  const size_t newIndex = batches.size();
  batches.emplace_back(textureId, zIndex);
  batchMap[key] = newIndex;
  needsSort = true;
  return batches[newIndex];
}

void RenderBatchManager::clear() {
  PROFILE_FUNCTION();
  for (auto &batch : batches) {
    batch.clear();
  }
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

// SECTION: static_cache_helpers
namespace {

constexpr float HYBRID_ACTIVATION_MARGIN = 150.0f;

float get_safe_zoom(const Engine *engine) {
  if (!engine || engine->camera.zoom <= 0.0001f) {
    return 1.0f;
  }
  return engine->camera.zoom;
}

void get_camera_world_rect(const Engine *engine, float &x1, float &y1,
                           float &x2, float &y2) {
  if (!engine) {
    x1 = y1 = x2 = y2 = 0.0f;
    return;
  }

  const float zoom = get_safe_zoom(engine);
  const float half_width = (engine->camera.width / zoom) * 0.5f;
  const float half_height = (engine->camera.height / zoom) * 0.5f;

  x1 = engine->camera.x - half_width;
  y1 = engine->camera.y - half_height;
  x2 = engine->camera.x + half_width;
  y2 = engine->camera.y + half_height;
}

void update_cell_coords(EntityContainer *container, uint32_t slot, float x,
                        float y) {
  if (!container || slot >= static_cast<uint32_t>(container->count)) {
    return;
  }

  int cell_x = static_cast<int>(x * INV_GRID_CELL_SIZE);
  int cell_y = static_cast<int>(y * INV_GRID_CELL_SIZE);
  cell_x = std::clamp(cell_x, 0, static_cast<int>(GRID_CELL_WIDTH) - 1);
  cell_y = std::clamp(cell_y, 0, static_cast<int>(GRID_CELL_HEIGHT) - 1);

  container->cell_x[slot] = static_cast<uint16_t>(cell_x);
  container->cell_y[slot] = static_cast<uint16_t>(cell_y);
}

RenderableEntityContainer *as_renderable(EntityContainer *container) {
  if (!container ||
      (container->containerFlag & static_cast<uint8_t>(ContainerFlag::RENDERABLE)) ==
          0) {
    return nullptr;
  }
  return static_cast<RenderableEntityContainer *>(container);
}

const RenderableEntityContainer *as_renderable(const EntityContainer *container) {
  if (!container ||
      (container->containerFlag & static_cast<uint8_t>(ContainerFlag::RENDERABLE)) ==
          0) {
    return nullptr;
  }
  return static_cast<const RenderableEntityContainer *>(container);
}

bool is_entity_renderable_visible(const EntityContainer *container,
                                  uint32_t slot) {
  const auto *renderable = as_renderable(container);
  if (!renderable || slot >= static_cast<uint32_t>(renderable->count)) {
    return false;
  }

  return (renderable->flags[slot] & static_cast<uint8_t>(EntityFlag::VISIBLE)) !=
         0;
}

bool intersects_rect(const RenderableEntityContainer *container, uint32_t slot,
                     float x1, float y1, float x2, float y2) {
  if (!container || slot >= static_cast<uint32_t>(container->count)) {
    return false;
  }

  const float entity_x = container->x_positions[slot];
  const float entity_y = container->y_positions[slot];
  const float entity_w = static_cast<float>(container->widths[slot]);
  const float entity_h = static_cast<float>(container->heights[slot]);

  return entity_x < x2 && (entity_x + entity_w) > x1 && entity_y < y2 &&
         (entity_y + entity_h) > y1;
}

void add_world_quad(RenderBatchManager &batch_manager, const TextureAtlas &atlas,
                    const RenderableEntityContainer *container, uint32_t slot) {
  const int texture_id = container->texture_ids[slot];
  const int z_index = container->z_indices[slot];
  const SDL_FRect region = get_texture_region(atlas, texture_id);

  batch_manager.addQuad(texture_id, z_index, container->x_positions[slot],
                        container->y_positions[slot],
                        static_cast<float>(container->widths[slot]),
                        static_cast<float>(container->heights[slot]), region);
}

void add_screen_quad(RenderBatchManager &batch_manager, const TextureAtlas &atlas,
                     const RenderableEntityContainer *container, uint32_t slot,
                     float cam_x1, float cam_y1, float zoom) {
  const float world_x = container->x_positions[slot];
  const float world_y = container->y_positions[slot];
  const float width = static_cast<float>(container->widths[slot]);
  const float height = static_cast<float>(container->heights[slot]);
  const int texture_id = container->texture_ids[slot];
  const int z_index = container->z_indices[slot];
  const SDL_FRect region = get_texture_region(atlas, texture_id);

  batch_manager.addQuad(texture_id, z_index, (world_x - cam_x1) * zoom,
                        (world_y - cam_y1) * zoom, width * zoom, height * zoom,
                        region);
}

void append_transformed_batch(RenderBatch &dst, const RenderBatch &src,
                              float cam_x1, float cam_y1, float zoom) {
  if (src.vertices.empty() || src.indices.empty()) {
    return;
  }

  const size_t base_vertex = dst.vertices.size();
  const size_t vertex_count = src.vertices.size();
  const size_t index_count = src.indices.size();

  dst.vertices.resize(base_vertex + vertex_count);
  for (size_t i = 0; i < vertex_count; ++i) {
    SDL_Vertex vertex = src.vertices[i];
    vertex.position.x = (vertex.position.x - cam_x1) * zoom;
    vertex.position.y = (vertex.position.y - cam_y1) * zoom;
    dst.vertices[base_vertex + i] = vertex;
  }

  const size_t base_index = dst.indices.size();
  dst.indices.resize(base_index + index_count);
  for (size_t i = 0; i < index_count; ++i) {
    dst.indices[base_index + i] =
        static_cast<int>(base_vertex) + src.indices[i];
  }
}

void rebuild_static_chunk_mesh(Engine *engine, StaticChunk &chunk) {
  if (!engine) {
    return;
  }

  chunk.cached_batches.clear();
  chunk.vertices.clear();
  chunk.indices.clear();

  for (const EntityRef &ref : chunk.refs) {
    if (!engine->entityManager.isHandleValid(ref.index, static_cast<int>(ref.type))) {
      continue;
    }

    auto *container =
        engine->entityManager.containers[ref.type].get();
    auto *renderable = as_renderable(container);
    if (!renderable) {
      continue;
    }

    const uint32_t slot = renderable->getSlot(ref.index);
    if (slot == INVALID_SLOT ||
        slot >= static_cast<uint32_t>(renderable->count) ||
        !is_entity_renderable_visible(renderable, slot)) {
      continue;
    }

    add_world_quad(chunk.cached_batches, engine->atlas, renderable, slot);
  }

  chunk.dirty = false;
}

} // namespace

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
  return static_cast<int32_t>(std::floor(world_value / STATIC_CHUNK_SIZE));
}

void StaticChunkCache::rebuildRefs(Engine *engine) {
  PROFILE_FUNCTION();
  if (!engine) {
    clear();
    return;
  }

  chunk_index_by_key.clear();
  chunks.clear();
  visible_chunk_indices.clear();

  for (const int type_id : engine->entityManager.getStaticTypeIds()) {
    if (type_id < 0 ||
        type_id >= static_cast<int>(engine->entityManager.containers.size())) {
      continue;
    }

    auto *container = engine->entityManager.containers[type_id].get();
    if (!container) {
      continue;
    }

    for (uint32_t slot = 0; slot < static_cast<uint32_t>(container->count);
         ++slot) {
      const EntityHandle id = container->getStableId(slot);
      if (id == INVALID_ID) {
        continue;
      }

      const int32_t chunk_x = worldToChunk(container->x_positions[slot]);
      const int32_t chunk_y = worldToChunk(container->y_positions[slot]);
      const int64_t key = makeChunkKey(chunk_x, chunk_y);

      size_t chunk_index = 0;
      auto it = chunk_index_by_key.find(key);
      if (it == chunk_index_by_key.end()) {
        chunk_index = chunks.size();
        chunk_index_by_key[key] = chunk_index;
        chunks.emplace_back();
        chunks.back().chunk_x = chunk_x;
        chunks.back().chunk_y = chunk_y;
        chunks.back().dirty = true;
      } else {
        chunk_index = it->second;
      }

      chunks[chunk_index].refs.push_back(
          {static_cast<uint32_t>(type_id), id});
    }
  }

  needs_full_rebuild = false;
}

std::vector<size_t> &StaticChunkCache::queryVisible(float x1, float y1, float x2,
                                                    float y2) {
  visible_chunk_indices.clear();

  if (x2 < x1) {
    std::swap(x1, x2);
  }
  if (y2 < y1) {
    std::swap(y1, y2);
  }

  const int32_t min_chunk_x = worldToChunk(x1);
  const int32_t min_chunk_y = worldToChunk(y1);
  const int32_t max_chunk_x = worldToChunk(x2);
  const int32_t max_chunk_y = worldToChunk(y2);

  for (int32_t cy = min_chunk_y; cy <= max_chunk_y; ++cy) {
    for (int32_t cx = min_chunk_x; cx <= max_chunk_x; ++cx) {
      const auto it = chunk_index_by_key.find(makeChunkKey(cx, cy));
      if (it != chunk_index_by_key.end()) {
        visible_chunk_indices.push_back(it->second);
      }
    }
  }

  return visible_chunk_indices;
}

// SECTION: engine_update_api

void process_pending_removals(Engine *engine) {
  PROFILE_FUNCTION();
  if (!engine || engine->pending_removals.empty()) {
    return;
  }

  bool touched_static = false;
  for (const EntityRef &ref : engine->pending_removals) {
    const int type_id = static_cast<int>(ref.type);
    if (!engine->entityManager.isHandleValid(ref.index, type_id)) {
      continue;
    }

    if (engine->entityManager.getRuntimeKind(type_id) ==
        ObjectRuntimeKind::Static) {
      touched_static = true;
    }

    engine->entityManager.removeEntity(ref.index, type_id, &engine->grid);
  }

  if (touched_static) {
    engine->staticChunkCache.markNeedsFullRebuild();
  }

  engine->pending_removals.clear();
}

void engine_update(Engine *engine) {
  PROFILE_FUNCTION();
  if (!engine) {
    return;
  }

  const Uint64 now = SDL_GetPerformanceCounter();
  const Uint64 frequency = SDL_GetPerformanceFrequency();

  float delta_time = 0.0f;
  if (engine->last_frame_time != 0 && frequency != 0) {
    delta_time =
        static_cast<float>(now - engine->last_frame_time) /
        static_cast<float>(frequency);
  }
  delta_time = std::clamp(delta_time, 0.0f, 0.1f);

  engine->last_frame_time = now;
  engine->fps = delta_time > 0.0f ? (1.0f / delta_time) : 0.0f;

  process_pending_removals(engine);

  engine->entityManager.updateDynamic(delta_time);

  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  get_camera_world_rect(engine, x1, y1, x2, y2);

  engine->entityManager.updateHybrid(
      engine, delta_time, x1 - HYBRID_ACTIVATION_MARGIN,
      y1 - HYBRID_ACTIVATION_MARGIN, x2 + HYBRID_ACTIVATION_MARGIN,
      y2 + HYBRID_ACTIVATION_MARGIN);

  process_pending_removals(engine);
}

void engine_update_entity_types(Engine *engine, float delta_time) {
  (void)engine;
  (void)delta_time;
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
  if (type_id >= 0) {
    engine->staticChunkCache.markNeedsFullRebuild();
  }
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
    return INVALID_ID;
  }

  const EntityHandle entity_id = engine->entityManager.createEntity(type_id);
  if (entity_id == INVALID_ID ||
      !engine->entityManager.isHandleValid(entity_id, type_id)) {
    return INVALID_ID;
  }

  auto *container = engine->entityManager.containers[type_id].get();
  if (!container) {
    return INVALID_ID;
  }

  const uint32_t slot = container->getSlot(entity_id);
  if (slot == INVALID_SLOT || slot >= static_cast<uint32_t>(container->count)) {
    return INVALID_ID;
  }

  update_cell_coords(container, slot, container->x_positions[slot],
                     container->y_positions[slot]);

  switch (engine->entityManager.getRuntimeKind(type_id)) {
  case ObjectRuntimeKind::Static:
    engine->staticChunkCache.markNeedsFullRebuild();
    break;
  case ObjectRuntimeKind::Dynamic:
  case ObjectRuntimeKind::Hybrid:
    container->grid_node_indices[slot] = engine->grid.add(
        {static_cast<uint32_t>(type_id), entity_id}, container->x_positions[slot],
        container->y_positions[slot]);
    break;
  }

  return entity_id;
}

void engine_destroy_entity(Engine *engine, EntityHandle entity_id, int type_id) {
  if (!engine || !engine->entityManager.isHandleValid(entity_id, type_id)) {
    return;
  }

  const bool is_static = engine->entityManager.getRuntimeKind(type_id) ==
                         ObjectRuntimeKind::Static;
  engine->entityManager.removeEntity(entity_id, type_id, &engine->grid);
  if (is_static) {
    engine->staticChunkCache.markNeedsFullRebuild();
  }
}

bool engine_is_handle_valid(Engine *engine, EntityHandle entity_id, int type_id) {
  return engine && engine->entityManager.isHandleValid(entity_id, type_id);
}

bool engine_set_entity_position(Engine *engine, EntityHandle entity_id,
                                int type_id, float x, float y) {
  if (!engine || !engine->entityManager.isHandleValid(entity_id, type_id)) {
    return false;
  }

  auto *container = engine->entityManager.containers[type_id].get();
  if (!container) {
    return false;
  }

  const uint32_t slot = container->getSlot(entity_id);
  if (slot == INVALID_SLOT || slot >= static_cast<uint32_t>(container->count)) {
    return false;
  }

  container->x_positions[slot] = x;
  container->y_positions[slot] = y;
  update_cell_coords(container, slot, x, y);

  if (engine->entityManager.getRuntimeKind(type_id) ==
      ObjectRuntimeKind::Static) {
    engine->staticChunkCache.markNeedsFullRebuild();
    return true;
  }

  const int32_t node_index = container->grid_node_indices[slot];
  if (node_index == -1) {
    container->grid_node_indices[slot] = engine->grid.add(
        {static_cast<uint32_t>(type_id), entity_id}, x, y);
  } else {
    engine->grid.move(node_index, x, y);
  }

  return true;
}

bool engine_set_entity_visible(Engine *engine, EntityHandle entity_id,
                               int type_id, bool visible) {
  if (!engine || !engine->entityManager.isHandleValid(entity_id, type_id)) {
    return false;
  }

  auto *container = engine->entityManager.containers[type_id].get();
  if (!container) {
    return false;
  }

  const uint32_t slot = container->getSlot(entity_id);
  if (slot == INVALID_SLOT || slot >= static_cast<uint32_t>(container->count)) {
    return false;
  }

  if (visible) {
    container->flags[slot] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
  } else {
    container->flags[slot] &= ~static_cast<uint8_t>(EntityFlag::VISIBLE);
  }

  if (engine->entityManager.getRuntimeKind(type_id) ==
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

void engine_set_entity_z_index(Engine *engine, EntityHandle entity_id,
                               int type_id, uint8_t z_index) {
  if (!engine || !engine->entityManager.isHandleValid(entity_id, type_id)) {
    return;
  }

  auto *container = engine->entityManager.containers[type_id].get();
  auto *renderable = as_renderable(container);
  if (!renderable) {
    return;
  }

  const uint32_t slot = renderable->getSlot(entity_id);
  if (slot == INVALID_SLOT || slot >= static_cast<uint32_t>(renderable->count)) {
    return;
  }

  renderable->z_indices[slot] = z_index;
  if (engine->entityManager.getRuntimeKind(type_id) ==
      ObjectRuntimeKind::Static) {
    engine->staticChunkCache.markNeedsFullRebuild();
  }
}

void engine_present(Engine *engine) {
  if (!engine || !engine->renderer) {
    return;
  }
  SDL_RenderPresent(engine->renderer);
}

// SECTION: texture_atlas_engine

TextureAtlas::TextureAtlas(SDL_Renderer *renderer, int width, int height,
                           int initialCapacity)
    : textures(nullptr), texture_count(0), texture_capacity(0), regions(nullptr),
      region_count(0), region_capacity(0), renderer(renderer) {
  (void)width;
  (void)height;

  texture_capacity = std::max(initialCapacity, 1);
  region_capacity = std::max(initialCapacity, 1);

  textures = new SDL_Texture *[texture_capacity];
  regions = new SDL_FRect[region_capacity];

  std::fill(textures, textures + texture_capacity, nullptr);
  std::fill(regions, regions + region_capacity, SDL_FRect{0.0f, 0.0f, 1.0f, 1.0f});
}

TextureAtlas::~TextureAtlas() {
  if (textures) {
    for (int i = 0; i < texture_count; ++i) {
      if (textures[i]) {
        SDL_DestroyTexture(textures[i]);
      }
    }
    delete[] textures;
  }

  delete[] regions;
}

TextureAtlas::TextureAtlas(TextureAtlas &&other) noexcept
    : textures(other.textures), texture_count(other.texture_count),
      texture_capacity(other.texture_capacity), regions(other.regions),
      region_count(other.region_count), region_capacity(other.region_capacity),
      renderer(other.renderer) {
  other.textures = nullptr;
  other.texture_count = 0;
  other.texture_capacity = 0;
  other.regions = nullptr;
  other.region_count = 0;
  other.region_capacity = 0;
  other.renderer = nullptr;
}

TextureAtlas &TextureAtlas::operator=(TextureAtlas &&other) noexcept {
  if (this != &other) {
    for (int i = 0; i < texture_count; ++i) {
      if (textures && textures[i]) {
        SDL_DestroyTexture(textures[i]);
      }
    }

    delete[] textures;
    delete[] regions;

    textures = other.textures;
    texture_count = other.texture_count;
    texture_capacity = other.texture_capacity;
    regions = other.regions;
    region_count = other.region_count;
    region_capacity = other.region_capacity;
    renderer = other.renderer;

    other.textures = nullptr;
    other.texture_count = 0;
    other.texture_capacity = 0;
    other.regions = nullptr;
    other.region_count = 0;
    other.region_capacity = 0;
    other.renderer = nullptr;
  }

  return *this;
}

int TextureAtlas::registerTexture(SDL_Surface *surface, int x, int y, int width,
                                  int height) {
  PROFILE_FUNCTION();
  if (!renderer || !surface) {
    return -1;
  }

  SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
  if (!texture) {
    SDL_Log("Failed to create texture from surface: %s", SDL_GetError());
    return -1;
  }

  ensureTextureCapacity(texture_count + 1);
  ensureRegionCapacity(region_count + 1);

  const int surface_width = std::max(surface->w, 1);
  const int surface_height = std::max(surface->h, 1);
  const int clamped_x = std::clamp(x, 0, surface_width - 1);
  const int clamped_y = std::clamp(y, 0, surface_height - 1);
  const int region_width =
      std::clamp(width > 0 ? width : (surface_width - clamped_x), 1,
                 surface_width - clamped_x);
  const int region_height =
      std::clamp(height > 0 ? height : (surface_height - clamped_y), 1,
                 surface_height - clamped_y);

  textures[texture_count] = texture;
  regions[region_count] = {static_cast<float>(clamped_x) / surface_width,
                           static_cast<float>(clamped_y) / surface_height,
                           static_cast<float>(region_width) / surface_width,
                           static_cast<float>(region_height) / surface_height};

  const int texture_id = texture_count;
  texture_count++;
  region_count++;
  return texture_id;
}

SDL_FRect TextureAtlas::getRegion(int textureId) const {
  if (textureId < 0 || textureId >= region_count || !regions) {
    return SDL_FRect{0.0f, 0.0f, 1.0f, 1.0f};
  }
  return regions[textureId];
}

SDL_Texture *TextureAtlas::getTexture(int textureId) const {
  if (textureId < 0 || textureId >= texture_count || !textures) {
    return nullptr;
  }
  return textures[textureId];
}

void TextureAtlas::ensureTextureCapacity(int needed) {
  if (needed <= texture_capacity) {
    return;
  }

  const int new_capacity = std::max(needed, texture_capacity * 2);
  SDL_Texture **new_textures = new SDL_Texture *[new_capacity];
  std::fill(new_textures, new_textures + new_capacity, nullptr);

  for (int i = 0; i < texture_count; ++i) {
    new_textures[i] = textures[i];
  }

  delete[] textures;
  textures = new_textures;
  texture_capacity = new_capacity;
}

void TextureAtlas::ensureRegionCapacity(int needed) {
  if (needed <= region_capacity) {
    return;
  }

  const int new_capacity = std::max(needed, region_capacity * 2);
  SDL_FRect *new_regions = new SDL_FRect[new_capacity];
  std::fill(new_regions, new_regions + new_capacity,
            SDL_FRect{0.0f, 0.0f, 1.0f, 1.0f});

  for (int i = 0; i < region_count; ++i) {
    new_regions[i] = regions[i];
  }

  delete[] regions;
  regions = new_regions;
  region_capacity = new_capacity;
}

SDL_FRect get_texture_region(const TextureAtlas &atlas, int16_t texture_id) {
  return atlas.getRegion(texture_id);
}

Engine *engine_create(int window_width, int window_height, int world_width,
                      int world_height, int cell_size) {
  PROFILE_FUNCTION();
  (void)cell_size;

  if (!SDL_WasInit(SDL_INIT_VIDEO) && !SDL_Init(SDL_INIT_VIDEO)) {
    SDL_Log("Failed to initialize SDL video: %s", SDL_GetError());
    return nullptr;
  }

  SDL_Window *window = nullptr;
  SDL_Renderer *renderer = nullptr;
  if (!SDL_CreateWindowAndRenderer("Attome Engine", window_width, window_height,
                                   0, &window, &renderer)) {
    SDL_Log("Failed to create window/renderer: %s", SDL_GetError());
    if (renderer) {
      SDL_DestroyRenderer(renderer);
    }
    if (window) {
      SDL_DestroyWindow(window);
    }
    return nullptr;
  }

  TextureAtlas atlas(renderer, world_width, world_height);
  Camera camera = {window_width * 0.5f, window_height * 0.5f,
                   static_cast<float>(window_width),
                   static_cast<float>(window_height), 1.0f};

  Engine *engine = new Engine{
      window,
      renderer,
      RenderBatchManager(),
      SpatialGrid(),
      std::move(atlas),
      camera,
      SDL_FRect{0.0f, 0.0f, static_cast<float>(world_width),
                static_cast<float>(world_height)},
      SDL_GetPerformanceCounter(),
      0.0f,
      {},
      StaticChunkCache(),
      EntityManager()};

  engine->pending_removals.reserve(256);
  return engine;
}

void engine_destroy(Engine *engine) {
  PROFILE_FUNCTION();
  if (!engine) {
    return;
  }

  SDL_Renderer *renderer = engine->renderer;
  SDL_Window *window = engine->window;

  engine->renderer = nullptr;
  engine->window = nullptr;
  delete engine;

  if (renderer) {
    SDL_DestroyRenderer(renderer);
  }
  if (window) {
    SDL_DestroyWindow(window);
  }
}

int engine_register_texture(Engine *engine, SDL_Surface *surface, int x, int y,
                            int width, int height) {
  if (!engine) {
    return -1;
  }
  return engine->atlas.registerTexture(surface, x, y, width, height);
}

// SECTION: render_and_grid

void engine_render_scene(Engine *engine) {
  PROFILE_FUNCTION();
  if (!engine || !engine->renderer) {
    return;
  }

  engine->renderBatchManager.clear();

  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  get_camera_world_rect(engine, x1, y1, x2, y2);
  const float zoom = get_safe_zoom(engine);

  if (engine->staticChunkCache.needs_full_rebuild) {
    engine->staticChunkCache.rebuildRefs(engine);
  }

  std::vector<size_t> &visible_chunks =
      engine->staticChunkCache.queryVisible(x1, y1, x2, y2);
  std::vector<StaticChunk> &chunks = engine->staticChunkCache.getChunks();
  for (size_t chunk_index : visible_chunks) {
    if (chunk_index >= chunks.size()) {
      continue;
    }

    StaticChunk &chunk = chunks[chunk_index];
    if (chunk.dirty) {
      rebuild_static_chunk_mesh(engine, chunk);
    }

    for (const RenderBatch &src_batch : chunk.cached_batches.getBatches()) {
      if (src_batch.vertices.empty() || src_batch.indices.empty()) {
        continue;
      }
      RenderBatch &dst_batch = engine->renderBatchManager.getBatch(
          src_batch.texture_id, src_batch.z_index);
      append_transformed_batch(dst_batch, src_batch, x1, y1, zoom);
    }
  }

  std::vector<EntityRef> &visible_refs = engine->grid.queryRect(x1, y1, x2, y2);
  for (const EntityRef &ref : visible_refs) {
    const int type_id = static_cast<int>(ref.type);
    if (!engine->entityManager.isHandleValid(ref.index, type_id) ||
        engine->entityManager.getRuntimeKind(type_id) ==
            ObjectRuntimeKind::Static) {
      continue;
    }

    auto *container = engine->entityManager.containers[type_id].get();
    auto *renderable = as_renderable(container);
    if (!renderable) {
      continue;
    }

    const uint32_t slot = renderable->getSlot(ref.index);
    if (slot == INVALID_SLOT ||
        slot >= static_cast<uint32_t>(renderable->count) ||
        !is_entity_renderable_visible(renderable, slot) ||
        !intersects_rect(renderable, slot, x1, y1, x2, y2)) {
      continue;
    }

    add_screen_quad(engine->renderBatchManager, engine->atlas, renderable, slot,
                    x1, y1, zoom);
  }

  for (const RenderBatch &batch : engine->renderBatchManager.getBatches()) {
    if (batch.vertices.empty() || batch.indices.empty()) {
      continue;
    }

    SDL_Texture *texture = engine->atlas.getTexture(batch.texture_id);
    SDL_RenderGeometry(engine->renderer, texture, batch.vertices.data(),
                       static_cast<int>(batch.vertices.size()),
                       batch.indices.data(),
                       static_cast<int>(batch.indices.size()));
  }
}

std::vector<EntityRef> &SpatialGrid::queryRect(float x1, float y1, float x2,
                                               float y2) {
  PROFILE_FUNCTION();
  queryResult.clear();

  if (x2 < x1) {
    std::swap(x1, x2);
  }
  if (y2 < y1) {
    std::swap(y1, y2);
  }

  int32_t minX = static_cast<int32_t>(std::floor(x1 * INV_GRID_CELL_SIZE));
  int32_t minY = static_cast<int32_t>(std::floor(y1 * INV_GRID_CELL_SIZE));
  int32_t maxX = static_cast<int32_t>(std::floor(x2 * INV_GRID_CELL_SIZE));
  int32_t maxY = static_cast<int32_t>(std::floor(y2 * INV_GRID_CELL_SIZE));

  constexpr int32_t kQueryPadCells = 4;
  minX -= kQueryPadCells;
  minY -= kQueryPadCells;

  minX = std::clamp(minX, 0, static_cast<int32_t>(GRID_CELL_WIDTH) - 1);
  minY = std::clamp(minY, 0, static_cast<int32_t>(GRID_CELL_HEIGHT) - 1);
  maxX = std::clamp(maxX, 0, static_cast<int32_t>(GRID_CELL_WIDTH) - 1);
  maxY = std::clamp(maxY, 0, static_cast<int32_t>(GRID_CELL_HEIGHT) - 1);

  for (int32_t cy = minY; cy <= maxY; ++cy) {
    const int32_t row_base = cy * GRID_CELL_WIDTH;
    for (int32_t cx = minX; cx <= maxX; ++cx) {
      int32_t node_index = cell_heads[row_base + cx];
      while (node_index != -1) {
        const GridNode &node = nodes[node_index];
        queryResult.push_back(node.entity);
        node_index = node.next;
      }
    }
  }

  return queryResult;
}

void SpatialGrid::rebuild_grid(Engine *engine) {
  PROFILE_FUNCTION();
  clearAll();

  if (!engine) {
    return;
  }

  for (size_t type_index = 0;
       type_index < engine->entityManager.containers.size(); ++type_index) {
    auto *container = engine->entityManager.containers[type_index].get();
    if (!container) {
      continue;
    }

    if (engine->entityManager.getRuntimeKind(static_cast<int>(type_index)) ==
        ObjectRuntimeKind::Static) {
      for (uint32_t slot = 0; slot < static_cast<uint32_t>(container->count);
           ++slot) {
        container->grid_node_indices[slot] = -1;
        update_cell_coords(container, slot, container->x_positions[slot],
                           container->y_positions[slot]);
      }
      continue;
    }

    for (uint32_t slot = 0; slot < static_cast<uint32_t>(container->count);
         ++slot) {
      const EntityHandle id = container->getStableId(slot);
      if (id == INVALID_ID) {
        container->grid_node_indices[slot] = -1;
        continue;
      }

      container->grid_node_indices[slot] = add(
          {static_cast<uint32_t>(type_index), id}, container->x_positions[slot],
          container->y_positions[slot]);
      update_cell_coords(container, slot, container->x_positions[slot],
                         container->y_positions[slot]);
    }
  }
}
