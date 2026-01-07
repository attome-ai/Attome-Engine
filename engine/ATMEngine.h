#ifndef ENGINE_H
#define ENGINE_H

#include "ATMDynamicArray.h"
#include "ATMProfiler.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <memory>
#include <stdbool.h>
#include <unordered_map>
#include <vector>

// static constexpr bool useVulkan = 0; // Removed Vulkan path
// Spatial grid implementation
static constexpr uint32_t WORLD_WIDTH = 50000;
static constexpr uint32_t WORLD_HEIGHT = 50000;
static constexpr uint32_t GRID_CELL_SIZE = 64;
static constexpr uint32_t GRID_CELL_WIDTH =
    (WORLD_WIDTH % GRID_CELL_SIZE) == 0 ? (WORLD_WIDTH / GRID_CELL_SIZE)
                                        : (WORLD_WIDTH / GRID_CELL_SIZE) + 1;
static constexpr uint32_t GRID_CELL_HEIGHT =
    (WORLD_HEIGHT % GRID_CELL_SIZE) == 0 ? (WORLD_HEIGHT / GRID_CELL_SIZE)
                                         : (WORLD_HEIGHT / GRID_CELL_SIZE) + 1;
static constexpr int MAX_ENTITIES_PER_CELL = 256; // Fixed capacity

// Alignment for memory
#define CACHE_LINE_SIZE 256

// Inverse cell size for faster calculation (multiplication instead of division)
static constexpr float INV_GRID_CELL_SIZE = (1.0f / GRID_CELL_SIZE);

// Common constants
static constexpr int MAX_LAYERS = 32;
static constexpr uint32_t INVALID_ID = 0xFFFFFFFF;

// Forward declarations
struct Engine;

// Entity flags
enum class EntityFlag : uint8_t {
  NONE = 0,
  VISIBLE = 1 << 0,
};

enum class ContainerFlag : uint8_t {
  NONE = 0,
  RENDERABLE = 1 << 0,
  UPDATEABLE = 1 << 1
};

/**
 * TextureAtlas class - Manages multiple textures and regions efficiently
 */
class TextureAtlas {
private:
  SDL_Texture **textures; // Array of textures
  int texture_count;      // Number of textures
  int texture_capacity;   // Capacity of textures array
  SDL_FRect *regions;     // UV regions for each subtexture
  int region_count;       // Number of regions
  int region_capacity;    // Capacity of regions array
  SDL_Renderer *renderer; // Reference to the renderer

public:
  // Constructor and destructor
  TextureAtlas(SDL_Renderer *renderer, int width, int height,
               int initialCapacity = 8);
  ~TextureAtlas();

  // Prevent copying
  TextureAtlas(const TextureAtlas &) = delete;
  TextureAtlas &operator=(const TextureAtlas &) = delete;

  // Allow moving
  TextureAtlas(TextureAtlas &&other) noexcept;
  TextureAtlas &operator=(TextureAtlas &&other) noexcept;

  // Register a texture with the atlas
  int registerTexture(SDL_Surface *surface, int x, int y, int width = 0,
                      int height = 0);

  // Get texture region by ID
  SDL_FRect getRegion(int textureId) const;

  // Get texture by ID (currently returns the first texture)
  SDL_Texture *getTexture(int textureId) const;

  // Get count of registered regions
  int getRegionCount() const { return region_count; }

private:
  // Ensure capacity for textures and regions
  void ensureTextureCapacity(int needed);
  void ensureRegionCapacity(int needed);
};

// Camera for culling
class Camera {
public:
  float x, y;
  float width, height;
  float zoom;
};

// Base Entity Container using SOA with RAII wrappers
class EntityContainer {
protected:
  // Base entity data
public:
  DynamicArray<uint8_t> flags;
  DynamicArray<uint32_t> entity_ids;
  DynamicArray<uint32_t> parent_ids;
  DynamicArray<uint32_t> first_child_ids;
  DynamicArray<uint32_t> next_sibling_ids;

  DynamicArray<float> x_positions;
  DynamicArray<float> y_positions;

  // Cell tracking for incremental grid updates (cache-aligned)
  AlignedDynamicArray<uint16_t, CACHE_LINE_SIZE> cell_x;
  AlignedDynamicArray<uint16_t, CACHE_LINE_SIZE> cell_y;
  AlignedDynamicArray<int32_t, CACHE_LINE_SIZE> grid_node_indices;

  uint8_t containerFlag;
  int type_id;
  uint8_t default_layer;
  int capacity;
  int count;

  EntityContainer(int typeId, uint8_t defaultLayer, int initialCapacity);
  virtual ~EntityContainer();

  virtual void update(float delta_time) = 0;
  virtual uint32_t createEntity();
  virtual void removeEntity(size_t index);

  int getTypeId() const { return type_id; }
  int getCount() const { return count; }
  uint8_t getDefaultLayer() const { return default_layer; }
  bool hasSpace() const { return count < capacity; }

protected:
  virtual void resizeArrays(int newCapacity);
};

// Renderable Entity Container
class RenderableEntityContainer : public EntityContainer {
public:
  // Renderable entity data (RAII managed)
  DynamicArray<int16_t> widths;
  DynamicArray<int16_t> heights;
  DynamicArray<int16_t> texture_ids;
  DynamicArray<uint8_t> z_indices;
  DynamicArray<float> rotations; // Rotation in radians

  RenderableEntityContainer(int typeId, uint8_t defaultLayer,
                            int initialCapacity);
  ~RenderableEntityContainer() override;

  uint32_t createEntity() override;
  void removeEntity(size_t index) override;

protected:
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
  std::vector<std::unique_ptr<Layer>> layers;
  std::vector<std::unique_ptr<EntityContainer>> containers;
  uint32_t next_entity_id;

  EntityManager();

  int registerEntityType(EntityContainer *container);
  uint32_t createEntity(int type_id);
  void removeEntity(uint32_t index, int type_id);
  void update(float delta_time);
};

struct EntityRef {
  uint32_t type : 8;
  uint32_t index : 24;
};

// Node for intrusive linked list spatial grid
struct GridNode {
  EntityRef entity;
  int32_t next;
  int32_t prev;
  int32_t cell_index; // To validate move/remove
};

class SpatialGrid {
private:
  // Grid of heads: stores index of first node in the cell
  // Flattened: index = y * GRID_CELL_WIDTH + x
  std::vector<int32_t> cell_heads;

  // Global pool of nodes.
  // Nodes are allocated once and reused?
  // Constraint: "adding entity is allowed, removing not allowed".
  // Actually, we can just append to 'nodes' if we need new ones,
  // but better to pre-allocate.
  std::vector<GridNode> nodes;
  std::vector<int32_t> free_node_indices;

  std::vector<EntityRef> queryResult;
  int32_t first_free_node;

public:
  SpatialGrid() : first_free_node(-1) {
    // Initialize grid heads to -1 (empty)
    cell_heads.resize(GRID_CELL_WIDTH * GRID_CELL_HEIGHT, -1);

    // Pre-allocate nodes (e.g., 1,000,000 entities max?)
    // Let's reserve a safe amount for high entity count
    nodes.reserve(3200000);
    queryResult.reserve(15000);
  }

  // Allocate a node from the pool
  int32_t allocateNode(const EntityRef &entity) {
    int32_t idx;
    if (first_free_node != -1) {
      idx = first_free_node;
      first_free_node = nodes[idx].next; // Pop from free stack
    } else {
      idx = static_cast<int32_t>(nodes.size());
      nodes.push_back({entity, -1, -1, -1});
    }
    return idx;
  }

  // Free a node to the pool
  void freeNode(int32_t nodeIndex) {
    nodes[nodeIndex].next = first_free_node;
    nodes[nodeIndex].prev = -1;
    nodes[nodeIndex].cell_index = -1;
    first_free_node = nodeIndex;
  }

  // Add entity to grid, returns node index (handle)
  int32_t add(const EntityRef &entity, float x, float y) {
    uint16_t cellX = static_cast<uint16_t>(x * INV_GRID_CELL_SIZE);
    uint16_t cellY = static_cast<uint16_t>(y * INV_GRID_CELL_SIZE);

    // Boundary check
    if (cellX >= GRID_CELL_WIDTH)
      cellX = GRID_CELL_WIDTH - 1;
    if (cellY >= GRID_CELL_HEIGHT)
      cellY = GRID_CELL_HEIGHT - 1;

    int32_t cellIdx = cellY * GRID_CELL_WIDTH + cellX;
    int32_t nodeIdx = allocateNode(entity);

    // Insert at head of list
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

  // Remove by node handle (O(1))
  void remove(int32_t nodeIndex) {
    if (nodeIndex == -1 || nodeIndex >= nodes.size())
      return;

    GridNode &node = nodes[nodeIndex];
    int32_t cellIdx = node.cell_index;

    if (cellIdx == -1)
      return; // Already removed?

    if (node.prev != -1) {
      nodes[node.prev].next = node.next;
    } else {
      // It was the head
      cell_heads[cellIdx] = node.next;
    }

    if (node.next != -1) {
      nodes[node.next].prev = node.prev;
    }

    freeNode(nodeIndex);
  }

  // Move: remove from old list, add to new list.
  // Optimization: reusing the SAME node, just relinking.
  // Returns true if cell changed.
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

    // Unlink from old list
    GridNode &node = nodes[nodeIndex];

    if (node.prev != -1) {
      nodes[node.prev].next = node.next;
    } else {
      cell_heads[oldCellIdx] = node.next;
    }

    if (node.next != -1) {
      nodes[node.next].prev = node.prev;
    }

    // Link to new list
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

  // Clear all grid data (retains node memory/capacity)
  void clearAll() {
    // Fast clear: just reset all cell heads to -1.

    // If full rebuild:
    std::fill(cell_heads.begin(), cell_heads.end(), -1);
    nodes.clear(); // Reset count to 0
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

    // Clamp to grid bounds
    if (minX < 0)
      minX = 0;
    if (minY < 0)
      minY = 0;
    if (maxX >= (int32_t)GRID_CELL_WIDTH)
      maxX = GRID_CELL_WIDTH - 1;
    if (maxY >= (int32_t)GRID_CELL_HEIGHT)
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

  // Declaration only, implemented in cpp

  // Declaration for rebuild_grid
  void rebuild_grid(Engine *engine);
};

/**
 * RenderBatch class - High-performance batch renderer
 */
class RenderBatch {
private:
public:
  int texture_id;
  int z_index;

  std::vector<SDL_Vertex> vertices;
  std::vector<int> indices;
  // Constructor with initial capacity
  RenderBatch(int textureId, int zIndex, int initialVertexCapacity = 4096);

  // Destructor
  ~RenderBatch();

  // Prevent copying
  RenderBatch(const RenderBatch &) = delete;
  RenderBatch &operator=(const RenderBatch &) = delete;

  // Allow moving
  RenderBatch(RenderBatch &&other) noexcept;
  RenderBatch &operator=(RenderBatch &&other) noexcept;

  // Add a quad to this batch
  void addQuad(float x, float y, float w, float h, SDL_FRect tex_region);

  // Reset batch for reuse without reallocating memory
  void clear();

private:
};

/**
 * RenderBatchManager - Efficient batch management with fast lookups
 */
class RenderBatchManager {
private:
  // Use a 64-bit key to combine texture_id and z_index for faster lookup
  using BatchKey = uint64_t;
  static inline BatchKey createKey(int textureId, int zIndex) {
    return (static_cast<uint64_t>(textureId) << 32) |
           static_cast<uint64_t>(zIndex);
  };

  std::vector<RenderBatch> batches;
  std::unordered_map<BatchKey, size_t> batchMap; // Maps key to batch index
  bool needsSort;

public:
  // Constructor with pre-allocated batches for common textures
  RenderBatchManager(int initialBatchCount = 8);

  // Add a quad to the appropriate batch
  void addQuad(int textureId, int zIndex, float x, float y, float w, float h,
               SDL_FRect tex_region);

  // Get or create a batch for the given texture and z-index
  RenderBatch &getBatch(int textureId, int zIndex);

  // Clear all batches for next frame
  void clear();

  // Sort batches by z-index for correct rendering order
  void sortIfNeeded();

  // Get all batches for rendering
  const std::vector<RenderBatch> &getBatches();

  // Get the number of active batches
  size_t getBatchCount() const;
};

// Main engine struct
typedef struct Engine {
  SDL_Window *window;
  SDL_Renderer *renderer; // SDL renderer
  RenderBatchManager renderBatchManager;
  SpatialGrid grid;
  TextureAtlas atlas;
  Camera camera;
  SDL_FRect world_bounds;
  Uint64 last_frame_time;
  float fps;
  std::vector<EntityRef> pending_removals;
  // Entity type system
  EntityManager entityManager;
} Engine;

// Modify engine_create to support Vulkan initialization
Engine *engine_create(int window_width, int window_height, int world_width,
                      int world_height, int cell_size);

// Modified engine_render_scene to use the appropriate renderer
void engine_render_scene(Engine *engine);

// Add a new function to destroy the engine with Vulkan support
void engine_destroy(Engine *engine);

// Modify engine_register_texture to handle both renderers
int engine_register_texture(Engine *engine, SDL_Surface *surface, int x, int y,
                            int width, int height);

// Add engine_present with Vulkan support
void engine_present(Engine *engine);

SDL_FRect get_texture_region(const TextureAtlas &atlas, int16_t texture_id);

// Entity management
void engine_update_entity_types(Engine *engine, float delta_time);

// Process entity removals
void process_pending_removals(Engine *engine);

// Update engine state
void engine_update(Engine *engine);

// Set entity z-index
void engine_set_entity_z_index(Engine *engine, uint32_t entity_idx, int type_id,
                               uint8_t z_index);

// Present renderer
void engine_present(Engine *engine);

// Texture loading functions
SDL_Surface* load_texture(const char* filename);
SDL_Surface* create_colored_surface(int width, int height, Uint8 r, Uint8 g, Uint8 b);

#endif // ENGINE_H