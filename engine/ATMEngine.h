#ifndef ATM_ENGINE_H
#define ATM_ENGINE_H

#include "ATMConfig.h"
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

// Spatial grid dimensions.
//
// By default (ATM_RUNTIME_GRID=0) the grid uses the compile-time constants
// below, exactly like the original engine: cell math on every entity move
// folds to immediates. Build with -DATTOME_RUNTIME_GRID=ON (defines
// ATM_RUNTIME_GRID=1) to size the grid from EngineConfig/JSON instead; that
// costs ~5% on engine_set_entity_position (see engine/bench/atm_compare).
// Either way, read the live values through engine->grid (cellSize(),
// invCellSize(), cellsWide(), cellsHigh()) so code works in both modes.
#ifndef ATM_RUNTIME_GRID
#define ATM_RUNTIME_GRID 0
#endif

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

#define CACHE_LINE_SIZE 64

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
  UPDATEABLE = 1 << 1,
  // Renderer honours `rotations[]` (radians, around the sprite centre).
  // Containers without this flag skip rotation entirely.
  ROTATABLE = 1 << 2,
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
  std::vector<int> free_ids;

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
  // Destroys the texture and frees its id for reuse by a later
  // registerTexture(). Entities still using the id render untextured, the
  // same as any other unregistered id.
  void unregisterTexture(int textureId);
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

  // Opt in to rotated rendering for this container (see ContainerFlag).
  void enableRotation(bool enabled = true) {
    if (enabled)
      containerFlag |= static_cast<uint8_t>(ContainerFlag::ROTATABLE);
    else
      containerFlag &= ~static_cast<uint8_t>(ContainerFlag::ROTATABLE);
  }

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

#if ATM_RUNTIME_GRID
  float cell_size_ = static_cast<float>(GRID_CELL_SIZE);
  float inv_cell_size_ = INV_GRID_CELL_SIZE;
  int32_t cells_w_ = static_cast<int32_t>(GRID_CELL_WIDTH);
  int32_t cells_h_ = static_cast<int32_t>(GRID_CELL_HEIGHT);
#else
  static constexpr float cell_size_ = static_cast<float>(GRID_CELL_SIZE);
  static constexpr float inv_cell_size_ = INV_GRID_CELL_SIZE;
  static constexpr int32_t cells_w_ = static_cast<int32_t>(GRID_CELL_WIDTH);
  static constexpr int32_t cells_h_ = static_cast<int32_t>(GRID_CELL_HEIGHT);
#endif
  int32_t query_pad_cells_ = 4;

  int32_t cellIndexFor(float x, float y) const {
#if !ATM_RUNTIME_GRID
    // Exactly the original engine's computation (and cost).
    uint16_t cellX = static_cast<uint16_t>(x * INV_GRID_CELL_SIZE);
    uint16_t cellY = static_cast<uint16_t>(y * INV_GRID_CELL_SIZE);
    if (cellX >= GRID_CELL_WIDTH)
      cellX = GRID_CELL_WIDTH - 1;
    if (cellY >= GRID_CELL_HEIGHT)
      cellY = GRID_CELL_HEIGHT - 1;
    return cellY * GRID_CELL_WIDTH + cellX;
#endif
    // Hot path (every move). Casting through uint32_t turns negatives into
    // huge values, so one min() per axis clamps both ends — the same result
    // as the original uint16_t code (negatives land in the last cell), but
    // well-defined.
    const uint32_t cx = std::min(
        static_cast<uint32_t>(static_cast<int32_t>(x * inv_cell_size_)),
        static_cast<uint32_t>(cells_w_ - 1));
    const uint32_t cy = std::min(
        static_cast<uint32_t>(static_cast<int32_t>(y * inv_cell_size_)),
        static_cast<uint32_t>(cells_h_ - 1));
    return static_cast<int32_t>(cy * static_cast<uint32_t>(cells_w_) + cx);
  }

public:
  // Legacy layout: WORLD_WIDTH x WORLD_HEIGHT with GRID_CELL_SIZE cells.
  SpatialGrid()
      : SpatialGrid(WORLD_WIDTH, WORLD_HEIGHT, GRID_CELL_SIZE, 3200000, 4) {}

  SpatialGrid(int world_width, int world_height, int cell_size,
              int node_reserve, int query_pad_cells)
      : first_free_node(-1) {
#if ATM_RUNTIME_GRID
    cell_size = std::max(cell_size, 1);
    cell_size_ = static_cast<float>(cell_size);
    inv_cell_size_ = 1.0f / cell_size_;
    cells_w_ =
        std::max((std::max(world_width, 1) + cell_size - 1) / cell_size, 1);
    cells_h_ =
        std::max((std::max(world_height, 1) + cell_size - 1) / cell_size, 1);
#else
    (void)world_width;
    (void)world_height;
    (void)cell_size;
#endif
    query_pad_cells_ = std::max(query_pad_cells, 0);
    cell_heads.resize(static_cast<size_t>(cells_w_) * cells_h_, -1);
    nodes.reserve(static_cast<size_t>(std::max(node_reserve, 0)));
    queryResult.reserve(15000);
  }

  static constexpr bool kRuntimeSized = ATM_RUNTIME_GRID != 0;
  float cellSize() const { return cell_size_; }
  float invCellSize() const { return inv_cell_size_; }
  int32_t cellsWide() const { return cells_w_; }
  int32_t cellsHigh() const { return cells_h_; }
  int32_t queryPadCells() const { return query_pad_cells_; }

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
    const int32_t cellIdx = cellIndexFor(x, y);
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
    const int32_t newCellIdx = cellIndexFor(x, y);
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
    outCellX = static_cast<uint16_t>(x * inv_cell_size_);
    outCellY = static_cast<uint16_t>(y * inv_cell_size_);
  }

  const std::vector<EntityRef> &queryCircle(float centerX, float centerY,
                                            float radius) {
    queryResult.clear();

    int32_t minX = static_cast<int32_t>((centerX - radius) * inv_cell_size_);
    int32_t minY = static_cast<int32_t>((centerY - radius) * inv_cell_size_);
    int32_t maxX = static_cast<int32_t>((centerX + radius) * inv_cell_size_);
    int32_t maxY = static_cast<int32_t>((centerY + radius) * inv_cell_size_);

    if (minX < 0)
      minX = 0;
    if (minY < 0)
      minY = 0;
    if (maxX >= cells_w_)
      maxX = cells_w_ - 1;
    if (maxY >= cells_h_)
      maxY = cells_h_ - 1;

    for (int32_t cy = minY; cy <= maxY; ++cy) {
      int32_t rowBase = cy * cells_w_;
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
  // Quad rotated by `radians` around its centre.
  void addQuadRotated(float x, float y, float w, float h, float radians,
                      SDL_FRect tex_region);
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
  void addQuadRotated(int textureId, int zIndex, float x, float y, float w,
                      float h, float radians, SDL_FRect tex_region);
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
  float chunk_size_ = static_cast<float>(STATIC_CHUNK_SIZE);

public:
  bool needs_full_rebuild{true};

  StaticChunkCache() = default;
  explicit StaticChunkCache(int chunk_size)
      : chunk_size_(static_cast<float>(std::max(chunk_size, 1))) {}

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

  // Settings the engine was created with.
  EngineConfig config{};

  // Timing, refreshed by engine_update():
  float frame_dt = 0.0f; // real (clamped) time since the last frame
  float sim_dt = 0.0f;   // dt passed to entity updates this frame
  int sim_steps = 0;     // entity update steps run this frame
  float fixed_accumulator = 0.0f;
  // Fixed-timestep mode only: leftover fraction of a step (0..1). Use it to
  // interpolate rendering between the previous and current simulation state.
  float fixed_alpha = 0.0f;
} Engine;

// Legacy entry point: keeps the original fixed 50000x50000 / 64px grid and
// variable timestep so existing games are unaffected.
Engine *engine_create(int window_width, int window_height, int world_width,
                      int world_height, int cell_size);
// Everything (window, grid, timestep, ...) comes from `config`, typically
// loaded with engine_config_load("config/engine.json", cfg).
Engine *engine_create_with_config(const EngineConfig &config);
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
// Radians around the sprite centre. Only drawn rotated when the container
// has ContainerFlag::ROTATABLE (RenderableEntityContainer::enableRotation()).
bool engine_set_entity_rotation(Engine *engine, EntityHandle entity_id,
                                int type_id, float radians);
void engine_present(Engine *engine);

SDL_Surface *load_texture(const char *filename);
SDL_Surface *create_colored_surface(int width, int height, Uint8 r, Uint8 g,
                                    Uint8 b);

#endif // ATM_ENGINE_H
