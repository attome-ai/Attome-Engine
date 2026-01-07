#ifndef ATM_ENGINE_H
#define ATM_ENGINE_H

#include "ATMLog.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_rect.h>
#include <SDL3/SDL_render.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>


// Forward declarations
class Engine;
class EntityContainer;

// Constants
#define CACHE_LINE_SIZE 64
#define INVALID_ID 0xFFFFFFFF

#define WORLD_WIDTH 50000
#define WORLD_HEIGHT 50000
#define GRID_CELL_SIZE 32
#define INV_GRID_CELL_SIZE (1.0f / GRID_CELL_SIZE)

// Flags
enum class EntityFlag : uint8_t { NONE = 0, VISIBLE = 1, ACTIVE = 2 };

enum class ContainerFlag : uint8_t { NONE = 0, UPDATEABLE = 1, RENDERABLE = 2 };

// Structures
struct EntityRef {
  int type;       // Container index
  uint32_t index; // Entity index
};

// Grid Constants
#define GRID_CELL_WIDTH (WORLD_WIDTH / GRID_CELL_SIZE)
#define GRID_CELL_HEIGHT (WORLD_HEIGHT / GRID_CELL_SIZE)

class SpatialGrid {
public:
  struct GridNode {
    EntityRef entity;
    int32_t next;
    int32_t cellIdx; // Added to support unlinking
  };

  std::vector<GridNode> nodes;
  std::vector<int32_t> cell_heads;
  std::vector<EntityRef> queryResult; // Cache for query results

  SpatialGrid() {
    // Initialize grid (approximate size)
    cell_heads.resize(GRID_CELL_WIDTH * GRID_CELL_HEIGHT, -1);
    nodes.reserve(100000);
    queryResult.reserve(1024);
  }

  // Helper: Get cell coordinates
  inline void getCellCoords(float x, float y, uint16_t &cx, uint16_t &cy) {
    cx = static_cast<uint16_t>(std::max(0.0f, x * INV_GRID_CELL_SIZE));
    cy = static_cast<uint16_t>(std::max(0.0f, y * INV_GRID_CELL_SIZE));
    if (cx >= GRID_CELL_WIDTH)
      cx = GRID_CELL_WIDTH - 1;
    if (cy >= GRID_CELL_HEIGHT)
      cy = GRID_CELL_HEIGHT - 1;
  }

  // Clear grid
  void clearAll() {
    std::fill(cell_heads.begin(), cell_heads.end(), -1);
    nodes.clear();
  }

  // Add entity
  int32_t add(EntityRef entity, float x, float y) {
    uint16_t cx, cy;
    getCellCoords(x, y, cx, cy);
    int32_t cellIdx = cy * GRID_CELL_WIDTH + cx;

    int32_t nodeIdx = static_cast<int32_t>(nodes.size());
    nodes.push_back({entity, cell_heads[cellIdx], cellIdx});
    cell_heads[cellIdx] = nodeIdx;
    return nodeIdx;
  }

  // Move entity (if cell changed)
  // nodeIdx is the index in 'nodes' vector.
  void move(int32_t nodeIdx, float x, float y) {
    if (nodeIdx < 0 || nodeIdx >= static_cast<int32_t>(nodes.size()))
      return;

    uint16_t newCx, newCy;
    getCellCoords(x, y, newCx, newCy);
    int32_t newCellIdx = newCy * GRID_CELL_WIDTH + newCx;

    int32_t oldCellIdx = nodes[nodeIdx].cellIdx;
    if (newCellIdx == oldCellIdx)
      return;

    // Unlink from old list (O(N) in cell)
    int32_t *prev = &cell_heads[oldCellIdx];
    while (*prev != -1) {
      if (*prev == nodeIdx) {
        *prev = nodes[nodeIdx].next;
        break;
      }
      prev = &nodes[*prev].next;
    }

    // Link to new list
    nodes[nodeIdx].cellIdx = newCellIdx;
    nodes[nodeIdx].next = cell_heads[newCellIdx];
    cell_heads[newCellIdx] = nodeIdx;
  }

  // Implemented in ATMEngine.cpp
  std::vector<EntityRef> &queryRect(float x1, float y1, float x2, float y2);
  void rebuild_grid(Engine *engine);

  // Added to support main.cpp usage
  std::vector<EntityRef> &queryCircle(float x, float y, float radius) {
    return queryRect(x - radius, y - radius, x + radius, y + radius);
  }
};

// Entity Container Base
class EntityContainer {
public:
  int type_id;
  uint8_t default_layer;
  int capacity;
  int count;
  uint8_t containerFlag;

  // Arrays
  uint8_t *flags;
  uint32_t *entity_ids;
  uint32_t *parent_ids;
  uint32_t *first_child_ids;
  uint32_t *next_sibling_ids;
  float *x_positions;
  float *y_positions;

  // Cell tracking (aligned)
  uint16_t *cell_x;
  uint16_t *cell_y;
  int32_t *grid_node_indices;

  EntityContainer(int typeId, uint8_t defaultLayer, int initialCapacity);
  virtual ~EntityContainer();

  virtual uint32_t createEntity();
  virtual void removeEntity(size_t index);
  virtual void resizeArrays(int newCapacity);
  virtual void update(float delta_time) {}

  uint8_t getDefaultLayer() const { return default_layer; }
  int getCount() const { return count; }
};

// Renderable Container
class RenderableEntityContainer : public EntityContainer {
public:
  int16_t *widths;
  int16_t *heights;
  int16_t *texture_ids;
  uint8_t *z_indices;
  float *rotations;

  RenderableEntityContainer(int typeId, uint8_t defaultLayer,
                            int initialCapacity);
  virtual ~RenderableEntityContainer();

  uint32_t createEntity() override;
  void removeEntity(size_t index) override;
  void resizeArrays(int newCapacity) override;
};

// Layer
class Layer {
public:
  int layer_id;
  bool is_active;
  std::vector<EntityContainer *> entity_containers;

  Layer(int id);
  void update(float delta_time);
  void addEntityContainer(EntityContainer *container);
};

// Entity Manager
class EntityManager {
public:
  std::vector<std::unique_ptr<EntityContainer>> containers;
  std::vector<std::unique_ptr<Layer>> layers;
  uint32_t next_entity_id;

  EntityManager();
  int registerEntityType(EntityContainer *container);
  uint32_t createEntity(int type_id);
  void removeEntity(uint32_t index, int type_id);
  void update(float delta_time);
};

// Render Batch
struct RenderBatch {
  int texture_id;
  int z_index;
  std::vector<SDL_Vertex> vertices;
  std::vector<int> indices;

  RenderBatch(int textureId, int zIndex, int initialVertexCapacity = 1024);
  ~RenderBatch();
  RenderBatch(RenderBatch &&other) noexcept;
  RenderBatch &operator=(RenderBatch &&other) noexcept;

  // Delete copy to prevent accidental copies
  RenderBatch(const RenderBatch &) = delete;
  RenderBatch &operator=(const RenderBatch &) = delete;

  void addQuad(float x, float y, float w, float h, SDL_FRect tex_region);
  void clear();
};

// Render Batch Manager
class RenderBatchManager {
public:
  // Key for map: (z_index << 32) | texture_id
  using BatchKey = uint64_t;

  std::vector<RenderBatch> batches;
  std::map<BatchKey, size_t> batchMap;
  bool needsSort;

  RenderBatchManager(int initialBatchCount);
  void addQuad(int textureId, int zIndex, float x, float y, float w, float h,
               SDL_FRect tex_region);
  RenderBatch &getBatch(int textureId, int zIndex);
  void clear();
  const std::vector<RenderBatch> &getBatches();
  size_t getBatchCount() const;

private:
  void sortIfNeeded();
  static BatchKey createKey(int textureId, int zIndex) {
    return (static_cast<uint64_t>(zIndex) << 32) |
           static_cast<uint32_t>(textureId);
  }
};

// Texture Atlas
class TextureAtlas {
public:
  SDL_Renderer *renderer;
  SDL_Texture **textures;
  int texture_count;
  int texture_capacity;
  SDL_FRect *regions;
  int region_count;
  int region_capacity;

  TextureAtlas(SDL_Renderer *renderer, int width, int height,
               int initialCapacity);
  ~TextureAtlas();
  TextureAtlas(TextureAtlas &&other) noexcept;
  TextureAtlas &operator=(TextureAtlas &&other) noexcept;

  int registerTexture(SDL_Surface *surface, int x, int y, int width,
                      int height);
  SDL_FRect getRegion(int textureId) const;
  SDL_Texture *getTexture(int textureId) const;

private:
  void ensureTextureCapacity(int needed);
  void ensureRegionCapacity(int needed);
};

// Camera
struct Camera {
  float x = 0, y = 0;
  float width = 0, height = 0;
  float zoom = 1.0f;
};

// Engine
struct Engine {
  SDL_Window *window;
  SDL_Renderer *renderer;
  SpatialGrid grid;
  EntityManager entityManager;
  RenderBatchManager renderBatchManager;
  std::vector<EntityRef> pending_removals;
  Camera camera;
  TextureAtlas atlas;
  SDL_Rect world_bounds;

  Uint64 last_frame_time;
  float fps;
};

// Functions
Engine *engine_create(int window_width, int window_height, int world_width,
                      int world_height, int cell_size);
void engine_update(Engine *engine);
void engine_render_scene(Engine *engine); // Assuming exists in scene render
void engine_present(Engine *engine);
void engine_destroy(Engine *engine);
int engine_register_texture(Engine *engine, SDL_Surface *surface, int x, int y,
                            int w, int h);
void engine_set_entity_z_index(Engine *engine, uint32_t entity_idx, int type_id,
                               uint8_t z_index);

// Helper
inline uint64_t get_ticks() { return SDL_GetTicks(); }

// Profiling macro stub
#define PROFILE_FUNCTION()

#endif // ATM_ENGINE_H
