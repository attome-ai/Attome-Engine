#ifndef ENGINE_H
#define ENGINE_H

#include "ATMDynamicArray.h"
#include "ATMProfiler.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdbool.h>
#include <unordered_map>
#include <vector>

static constexpr uint32_t WORLD_WIDTH = 50000;
static constexpr uint32_t WORLD_HEIGHT = 50000;
static constexpr uint32_t GRID_CELL_SIZE = 64;
static constexpr uint32_t GRID_CELL_WIDTH =
    (WORLD_WIDTH % GRID_CELL_SIZE) == 0 ? (WORLD_WIDTH / GRID_CELL_SIZE)
                                        : (WORLD_WIDTH / GRID_CELL_SIZE) + 1;
static constexpr uint32_t GRID_CELL_HEIGHT =
    (WORLD_HEIGHT % GRID_CELL_SIZE) == 0 ? (WORLD_HEIGHT / GRID_CELL_SIZE)
                                         : (WORLD_HEIGHT / GRID_CELL_SIZE) + 1;
static constexpr int MAX_ENTITIES_PER_CELL = 256;

#define CACHE_LINE_SIZE 256

static constexpr float INV_GRID_CELL_SIZE = (1.0f / GRID_CELL_SIZE);
static constexpr int MAX_LAYERS = 32;
static constexpr uint32_t INVALID_ID = 0xFFFFFFFF;
static constexpr uint32_t INVALID_SLOT = INVALID_ID;
static constexpr int STATIC_CHUNK_SIZE = 512;

typedef uint32_t EntityHandle;

struct Engine;
class SpatialGrid;

enum class EntityFlag : uint8_t {
  NONE = 0,
  VISIBLE = 1 << 0,
};

enum class ContainerFlag : uint8_t {
  NONE = 0,
  RENDERABLE = 1 << 0,
  UPDATEABLE = 1 << 1
};

enum class ObjectRuntimeKind : uint8_t {
  Dynamic = 0,
  Static = 1,
  Hybrid = 2,
};

class TextureAtlas {
private:
  SDL_Texture **textures;
  int texture_count;
  int texture_capacity;
  SDL_FRect *regions;
  int region_count;
  int region_capacity;
  SDL_Renderer *renderer;

public:
  TextureAtlas(SDL_Renderer *renderer, int width, int height,
               int initialCapacity = 8);
  ~TextureAtlas();

  TextureAtlas(const TextureAtlas &) = delete;
  TextureAtlas &operator=(const TextureAtlas &) = delete;
  TextureAtlas(TextureAtlas &&other) noexcept;
  TextureAtlas &operator=(TextureAtlas &&other) noexcept;

  int registerTexture(SDL_Surface *surface, int x, int y, int width = 0,
                      int height = 0);
  SDL_FRect getRegion(int textureId) const;
  SDL_Texture *getTexture(int textureId) const;
  int getRegionCount() const { return region_count; }

private:
  void ensureTextureCapacity(int needed);
  void ensureRegionCapacity(int needed);
};

class Camera {
public:
  float x, y;
  float width, height;
  float zoom;
};

class EntityContainer {
public:
  DynamicArray<uint8_t> flags;
  DynamicArray<uint32_t> entity_ids;
  DynamicArray<uint32_t> parent_ids;
  DynamicArray<uint32_t> first_child_ids;
  DynamicArray<uint32_t> next_sibling_ids;

  DynamicArray<float> x_positions;
  DynamicArray<float> y_positions;

  AlignedDynamicArray<uint16_t, CACHE_LINE_SIZE> cell_x;
  AlignedDynamicArray<uint16_t, CACHE_LINE_SIZE> cell_y;
  AlignedDynamicArray<int32_t, CACHE_LINE_SIZE> grid_node_indices;
  DynamicArray<EntityHandle> slot_to_id;
  std::vector<uint32_t> id_to_slot;
  std::vector<EntityHandle> free_ids;
  EntityHandle next_id;

  uint8_t containerFlag;
  int type_id;
  uint8_t default_layer;
  int capacity;
  int count;

  EntityContainer(int typeId, uint8_t defaultLayer, int initialCapacity);
  virtual ~EntityContainer();

  virtual void update(float delta_time) = 0;
  virtual void updateVisible(const std::vector<uint32_t> &active_slots,
                             float delta_time);
  virtual EntityHandle createEntity();
  virtual void removeEntity(EntityHandle id);
  virtual void swapSlots(uint32_t a, uint32_t b);

  EntityHandle getStableId(uint32_t slot) const;
  uint32_t getSlot(EntityHandle id) const;
  bool isAlive(EntityHandle id) const;

  int getTypeId() const { return type_id; }
  int getCount() const { return count; }
  uint8_t getDefaultLayer() const { return default_layer; }
  bool hasSpace() const { return count < capacity; }

protected:
  virtual void resizeArrays(int newCapacity);
};

class RenderableEntityContainer : public EntityContainer {
public:
  DynamicArray<int16_t> widths;
  DynamicArray<int16_t> heights;
  DynamicArray<int16_t> texture_ids;
  DynamicArray<uint8_t> z_indices;
  DynamicArray<float> rotations;

  RenderableEntityContainer(int typeId, uint8_t defaultLayer,
                            int initialCapacity);
  ~RenderableEntityContainer() override;

  EntityHandle createEntity() override;

protected:
  void swapSlots(uint32_t a, uint32_t b) override;
  void resizeArrays(int newCapacity) override;
};

class Layer {
private:
  int layer_id;
  bool is_active;
  std::vector<EntityContainer *> entity_containers;

public:
  Layer(int id);

  void update(float delta_time);
  void addEntityContainer(EntityContainer *container);

  bool isActive() const { return is_active; }
  void setActive(bool active) { is_active = active; }
  int getId() const { return layer_id; }
};

class EntityManager {
public:
  struct TypeRuntimeState {
    ObjectRuntimeKind runtime_kind{ObjectRuntimeKind::Dynamic};
  };

  std::vector<std::unique_ptr<Layer>> layers;
  std::vector<std::unique_ptr<EntityContainer>> containers;
  std::vector<TypeRuntimeState> type_states;
  std::vector<int> dynamic_type_ids;
  std::vector<int> static_type_ids;
  std::vector<int> hybrid_type_ids;
  uint32_t next_entity_id;

  EntityManager();

  int registerEntityType(EntityContainer *container, ObjectRuntimeKind kind);
  int registerEntityType(EntityContainer *container);
  int registerDynamicEntityType(EntityContainer *container);
  int registerStaticEntityType(EntityContainer *container);
  int registerHybridEntityType(EntityContainer *container);

  EntityHandle createEntity(int type_id);
  void removeEntity(EntityHandle id, int type_id, SpatialGrid *grid = nullptr);
  bool isHandleValid(EntityHandle id, int type_id) const;

  ObjectRuntimeKind getRuntimeKind(int type_id) const;
  const std::vector<int> &getDynamicTypeIds() const { return dynamic_type_ids; }
  const std::vector<int> &getStaticTypeIds() const { return static_type_ids; }
  const std::vector<int> &getHybridTypeIds() const { return hybrid_type_ids; }

  void updateDynamic(float delta_time);
  void updateHybrid(Engine *engine, float delta_time, float x1, float y1,
                    float x2, float y2);
  void update(float delta_time);
};

struct EntityRef {
  uint32_t type;
  EntityHandle index;
};

struct GridNode {
  EntityRef entity;
  int32_t next;
  int32_t prev;
  int32_t cell_index;
};

class SpatialGrid {
private:
  std::vector<int32_t> cell_heads;
  std::vector<GridNode> nodes;
  std::vector<EntityRef> queryResult;
  int32_t first_free_node;

public:
  SpatialGrid() : first_free_node(-1) {
    cell_heads.resize(GRID_CELL_WIDTH * GRID_CELL_HEIGHT, -1);
    nodes.reserve(3200000);
    queryResult.reserve(15000);
  }

  int32_t allocateNode(const EntityRef &entity) {
    int32_t idx;
    if (first_free_node != -1) {
      idx = first_free_node;
      first_free_node = nodes[idx].next;
      nodes[idx].entity = entity;
    } else {
      idx = static_cast<int32_t>(nodes.size());
      nodes.push_back({entity, -1, -1, -1});
    }
    return idx;
  }

  void freeNode(int32_t nodeIndex) {
    nodes[nodeIndex].next = first_free_node;
    nodes[nodeIndex].prev = -1;
    nodes[nodeIndex].cell_index = -1;
    first_free_node = nodeIndex;
  }

  int32_t add(const EntityRef &entity, float x, float y) {
    uint16_t cellX = static_cast<uint16_t>(x * INV_GRID_CELL_SIZE);
    uint16_t cellY = static_cast<uint16_t>(y * INV_GRID_CELL_SIZE);

    if (cellX >= GRID_CELL_WIDTH)
      cellX = GRID_CELL_WIDTH - 1;
    if (cellY >= GRID_CELL_HEIGHT)
      cellY = GRID_CELL_HEIGHT - 1;

    int32_t cellIdx = cellY * GRID_CELL_WIDTH + cellX;
    int32_t nodeIdx = allocateNode(entity);
    int32_t oldHead = cell_heads[cellIdx];

    nodes[nodeIdx].next = oldHead;
    nodes[nodeIdx].prev = -1;
    nodes[nodeIdx].cell_index = cellIdx;

    if (oldHead != -1) {
      nodes[oldHead].prev = nodeIdx;
    }

    cell_heads[cellIdx] = nodeIdx;
    return nodeIdx;
  }

  void remove(int32_t nodeIndex) {
    if (nodeIndex == -1 || nodeIndex >= static_cast<int32_t>(nodes.size()))
      return;

    GridNode &node = nodes[nodeIndex];
    int32_t cellIdx = node.cell_index;
    if (cellIdx == -1)
      return;

    if (node.prev != -1) {
      nodes[node.prev].next = node.next;
    } else {
      cell_heads[cellIdx] = node.next;
    }

    if (node.next != -1) {
      nodes[node.next].prev = node.prev;
    }

    freeNode(nodeIndex);
  }

  bool move(int32_t nodeIndex, float x, float y) {
    uint16_t newCellX = static_cast<uint16_t>(x * INV_GRID_CELL_SIZE);
    uint16_t newCellY = static_cast<uint16_t>(y * INV_GRID_CELL_SIZE);

    if (newCellX >= GRID_CELL_WIDTH)
      newCellX = GRID_CELL_WIDTH - 1;
    if (newCellY >= GRID_CELL_HEIGHT)
      newCellY = GRID_CELL_HEIGHT - 1;

    int32_t newCellIdx = newCellY * GRID_CELL_WIDTH + newCellX;
    int32_t oldCellIdx = nodes[nodeIndex].cell_index;
    if (newCellIdx == oldCellIdx)
      return false;

    GridNode &node = nodes[nodeIndex];
    if (node.prev != -1) {
      nodes[node.prev].next = node.next;
    } else {
      cell_heads[oldCellIdx] = node.next;
    }

    if (node.next != -1) {
      nodes[node.next].prev = node.prev;
    }

    int32_t oldHead = cell_heads[newCellIdx];
    node.next = oldHead;
    node.prev = -1;
    node.cell_index = newCellIdx;

    if (oldHead != -1) {
      nodes[oldHead].prev = nodeIndex;
    }

    cell_heads[newCellIdx] = nodeIndex;
    return true;
  }

  void clearAll() {
    std::fill(cell_heads.begin(), cell_heads.end(), -1);
    nodes.clear();
    first_free_node = -1;
  }

  std::vector<EntityRef> &queryRect(float x1, float y1, float x2, float y2);

  inline void getCellCoords(const float &x, const float &y, uint16_t &outCellX,
                            uint16_t &outCellY) const {
    outCellX = x * INV_GRID_CELL_SIZE;
    outCellY = y * INV_GRID_CELL_SIZE;
  }

  const std::vector<EntityRef> &queryCircle(float centerX, float centerY,
                                            float radius) {
    queryResult.clear();

    int32_t minX =
        static_cast<int32_t>((centerX - radius) * INV_GRID_CELL_SIZE);
    int32_t minY =
        static_cast<int32_t>((centerY - radius) * INV_GRID_CELL_SIZE);
    int32_t maxX =
        static_cast<int32_t>((centerX + radius) * INV_GRID_CELL_SIZE);
    int32_t maxY =
        static_cast<int32_t>((centerY + radius) * INV_GRID_CELL_SIZE);

    if (minX < 0)
      minX = 0;
    if (minY < 0)
      minY = 0;
    if (maxX >= static_cast<int32_t>(GRID_CELL_WIDTH))
      maxX = GRID_CELL_WIDTH - 1;
    if (maxY >= static_cast<int32_t>(GRID_CELL_HEIGHT))
      maxY = GRID_CELL_HEIGHT - 1;

    for (int32_t cy = minY; cy <= maxY; ++cy) {
      int32_t rowBase = cy * GRID_CELL_WIDTH;
      for (int32_t cx = minX; cx <= maxX; ++cx) {
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

  void rebuild_grid(Engine *engine);
};

class RenderBatch {
public:
  int texture_id;
  int z_index;
  std::vector<SDL_Vertex> vertices;
  std::vector<int> indices;

  RenderBatch(int textureId, int zIndex, int initialVertexCapacity = 4096);
  ~RenderBatch();

  RenderBatch(const RenderBatch &) = delete;
  RenderBatch &operator=(const RenderBatch &) = delete;
  RenderBatch(RenderBatch &&other) noexcept;
  RenderBatch &operator=(RenderBatch &&other) noexcept;

  void addQuad(float x, float y, float w, float h, SDL_FRect tex_region);
  void clear();
};

class RenderBatchManager {
private:
  using BatchKey = uint64_t;
  static inline BatchKey createKey(int textureId, int zIndex) {
    return (static_cast<uint64_t>(textureId) << 32) |
           static_cast<uint64_t>(zIndex);
  }

  std::vector<RenderBatch> batches;
  std::unordered_map<BatchKey, size_t> batchMap;
  bool needsSort;

public:
  RenderBatchManager(int initialBatchCount = 8);
  RenderBatchManager(const RenderBatchManager &) = delete;
  RenderBatchManager &operator=(const RenderBatchManager &) = delete;
  RenderBatchManager(RenderBatchManager &&other) noexcept = default;
  RenderBatchManager &operator=(RenderBatchManager &&other) noexcept = default;
  void addQuad(int textureId, int zIndex, float x, float y, float w, float h,
               SDL_FRect tex_region);
  RenderBatch &getBatch(int textureId, int zIndex);
  void clear();
  void sortIfNeeded();
  const std::vector<RenderBatch> &getBatches();
  size_t getBatchCount() const;
};

struct StaticChunk {
  int32_t chunk_x{0};
  int32_t chunk_y{0};
  bool dirty{true};
  std::vector<EntityRef> refs;
  RenderBatchManager cached_batches{4};
  std::vector<SDL_Vertex> vertices;
  std::vector<int> indices;
};

class StaticChunkCache {
private:
  std::unordered_map<int64_t, size_t> chunk_index_by_key;
  std::vector<StaticChunk> chunks;
  std::vector<size_t> visible_chunk_indices;

  static int64_t makeChunkKey(int32_t chunk_x, int32_t chunk_y);

public:
  bool needs_full_rebuild{true};

  void clear();
  void markAllDirty();
  void markNeedsFullRebuild() { needs_full_rebuild = true; }
  int32_t worldToChunk(float world_value) const;
  void rebuildRefs(Engine *engine);
  std::vector<size_t> &queryVisible(float x1, float y1, float x2, float y2);
  std::vector<StaticChunk> &getChunks() { return chunks; }
};

typedef struct Engine {
  SDL_Window *window;
  SDL_Renderer *renderer;
  RenderBatchManager renderBatchManager;
  SpatialGrid grid;
  TextureAtlas atlas;
  Camera camera;
  SDL_FRect world_bounds;
  Uint64 last_frame_time;
  float fps;
  std::vector<EntityRef> pending_removals;
  StaticChunkCache staticChunkCache;
  EntityManager entityManager;
} Engine;

Engine *engine_create(int window_width, int window_height, int world_width,
                      int world_height, int cell_size);
void engine_render_scene(Engine *engine);
void engine_destroy(Engine *engine);
int engine_register_texture(Engine *engine, SDL_Surface *surface, int x, int y,
                            int width, int height);
void engine_present(Engine *engine);

SDL_FRect get_texture_region(const TextureAtlas &atlas, int16_t texture_id);

int engine_register_dynamic_type(Engine *engine, EntityContainer *container);
int engine_register_static_type(Engine *engine, EntityContainer *container);
int engine_register_hybrid_type(Engine *engine, EntityContainer *container);

EntityHandle engine_create_entity(Engine *engine, int type_id);
void engine_destroy_entity(Engine *engine, EntityHandle entity_id, int type_id);
bool engine_is_handle_valid(Engine *engine, EntityHandle entity_id, int type_id);
bool engine_set_entity_position(Engine *engine, EntityHandle entity_id,
                                int type_id, float x, float y);
bool engine_set_entity_visible(Engine *engine, EntityHandle entity_id,
                               int type_id, bool visible);
void engine_mark_static_dirty(Engine *engine);

void engine_update_entity_types(Engine *engine, float delta_time);
void process_pending_removals(Engine *engine);
void engine_update(Engine *engine);
void engine_set_entity_z_index(Engine *engine, EntityHandle entity_idx,
                               int type_id, uint8_t z_index);
void engine_present(Engine *engine);

SDL_Surface *load_texture(const char *filename);
SDL_Surface *create_colored_surface(int width, int height, Uint8 r, Uint8 g,
                                    Uint8 b);

#endif // ENGINE_H
