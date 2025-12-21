#ifndef ENGINE_H
#define ENGINE_H

#include <SDL3/SDL.h>
#include <stdbool.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cassert>
#include "ATMBufferPool.h"
#include <array>
#include <unordered_map>
#include "ATMProfiler.h"


// Spatial grid implementation
static constexpr uint32_t WORLD_WIDTH = 50000;
static constexpr uint32_t WORLD_HEIGHT = 50000;
static constexpr uint32_t GRID_CELL_SIZE = 2024;
static constexpr uint32_t GRID_CELL_WIDTH = (WORLD_WIDTH % GRID_CELL_SIZE) == 0 ? (WORLD_WIDTH / GRID_CELL_SIZE) : (WORLD_WIDTH / GRID_CELL_SIZE) + 1;
static constexpr uint32_t GRID_CELL_HEIGHT = (WORLD_HEIGHT % GRID_CELL_SIZE) == 0 ? (WORLD_HEIGHT / GRID_CELL_SIZE) : (WORLD_HEIGHT / GRID_CELL_SIZE) + 1;
// Alignment for memory
#define CACHE_LINE_SIZE 64

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
    SDL_Texture** textures;     // Array of textures
    int texture_count;          // Number of textures
    int texture_capacity;       // Capacity of textures array
    SDL_FRect* regions;         // UV regions for each subtexture
    int region_count;           // Number of regions
    int region_capacity;        // Capacity of regions array
    SDL_Renderer* renderer;     // Reference to the renderer

public:
    // Constructor and destructor
    TextureAtlas(SDL_Renderer* renderer, int width, int height, int initialCapacity = 8);
    ~TextureAtlas();

    // Prevent copying
    TextureAtlas(const TextureAtlas&) = delete;
    TextureAtlas& operator=(const TextureAtlas&) = delete;

    // Allow moving
    TextureAtlas(TextureAtlas&& other) noexcept;
    TextureAtlas& operator=(TextureAtlas&& other) noexcept;

    // Register a texture with the atlas
    int registerTexture(SDL_Surface* surface, int x, int y, int width = 0, int height = 0);

    // Get texture region by ID
    SDL_FRect getRegion(int textureId) const;

    // Get texture by ID (currently returns the first texture)
    SDL_Texture* getTexture(int textureId) const;

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

// Base Entity Container using SOA with raw pointers
class EntityContainer {
protected:
    // Base entity data
public:
    uint8_t* flags;
    uint32_t* entity_ids;
    uint32_t* parent_ids;
    uint32_t* first_child_ids;
    uint32_t* next_sibling_ids;

    float* x_positions;
    float* y_positions;
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
    // Renderable entity data

    int16_t* widths;
    int16_t* heights;
    int16_t* texture_ids;
    uint8_t* z_indices;

    RenderableEntityContainer(int typeId, uint8_t defaultLayer, int initialCapacity);
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
    std::vector<EntityContainer*> entity_containers;

public:
    Layer(int id);

    void update(float delta_time);
    void addEntityContainer(EntityContainer* container);

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

    int registerEntityType(EntityContainer* container);
    uint32_t createEntity(int type_id);
    void removeEntity(uint32_t index, int type_id);
    void update(float delta_time);
};




struct EntityRef {
    uint32_t type : 10;
    uint32_t index : 22;
};

class SpatialGrid {
private:
    struct Cell {
        static constexpr size_t MAX_ENTITIES_PER_CELL = 2024;  // Fixed capacity
        EntityRef entities[MAX_ENTITIES_PER_CELL];
        uint32_t count;  // Track number of entities in cell

        Cell() : count(0) {}
    };

    alignas(CACHE_LINE_SIZE) std::array<std::array<Cell, GRID_CELL_WIDTH>, GRID_CELL_HEIGHT> cells;
    std::vector<EntityRef> queryResult;

public:
    SpatialGrid() {
        queryResult.reserve(15000);
    }

    void add(const EntityRef& entity, const float& x, const float& y) {
        const uint16_t cellX = x * INV_GRID_CELL_SIZE;
        const uint16_t cellY = y * INV_GRID_CELL_SIZE;
        add2(entity, cellX, cellY);
    }

    void add2(const EntityRef& entity, const uint16_t x, const uint16_t y) {
        Cell& cell = cells[y][x];

        // Ensure we don't exceed cell capacity
        if (cell.count < Cell::MAX_ENTITIES_PER_CELL)_LIKELY{
            cell.entities[cell.count++] = entity;
        }
    }

    inline void getCellCoords(const float& x, const float& y, uint16_t& outCellX, uint16_t& outCellY) const {
        outCellX = x * INV_GRID_CELL_SIZE;
        outCellY = y * INV_GRID_CELL_SIZE;
    }

    const std::vector<EntityRef>& queryCircle(float centerX, float centerY, float radius) {
        queryResult.clear();

        float minX = centerX - radius;
        float minY = centerY - radius;
        float maxX = centerX + radius;
        float maxY = centerY + radius;

        uint16_t minCellX, minCellY, maxCellX, maxCellY;
        getCellCoords(minX, minY, minCellX, minCellY);
        getCellCoords(maxX, maxY, maxCellX, maxCellY);

        float radiusSq = radius * radius;

        for (uint16_t cy = minCellY; cy <= maxCellY; ++cy) {
            for (uint16_t cx = minCellX; cx <= maxCellX; ++cx) {
                float cellWorldX = cx * GRID_CELL_SIZE;
                float cellWorldY = cy * GRID_CELL_SIZE;

                Cell& cell = cells[cy][cx];
                uint32_t entCount = cell.count;

                for (uint32_t i = 0; i < entCount; ++i) {
                    float dx = cellWorldX - centerX;
                    float dy = cellWorldY - centerY;
                    if ((dx * dx + dy * dy) <= radiusSq) {
                        queryResult.push_back(cell.entities[i]);
                    }
                }
            }
        }
        return queryResult;
    }

    const std::vector<EntityRef>& queryRect(float minX, float minY, float maxX, float maxY) {
        queryResult.clear();

        uint16_t minCellX, minCellY, maxCellX, maxCellY;

        getCellCoords(minX, minY, minCellX, minCellY);
        getCellCoords(maxX, maxY, maxCellX, maxCellY);

        minCellX = std::max((uint16_t)0, minCellX);
        minCellY = std::max((uint16_t)0, minCellY);
        maxCellX = std::min((uint16_t)(GRID_CELL_WIDTH - 1), maxCellX);
        maxCellY = std::min((uint16_t)(GRID_CELL_HEIGHT - 1), maxCellY);

        for (uint16_t cy = minCellY; cy <= maxCellY; ++cy) {
            for (uint16_t cx = minCellX; cx <= maxCellX; ++cx) {
                Cell& cell = cells[cy][cx];
                uint32_t entCount = cell.count;

                for (uint32_t i = 0; i < entCount; ++i) {
                    queryResult.push_back(cell.entities[i]);
                }
            }
        }
        return queryResult;
    }

    void clearAll() {
        PROFILE_FUNCTION();
        for (auto& row : cells) {
            for (auto& cell : row) {
                cell.count = 0;
            }
        }
        queryResult.clear();
    }

    void rebuild_grid(Engine* engine);

    void printGrid() const;
    void printQueryCircle(float centerX, float centerY, float radius);
    void printQueryRect(float minX, float minY, float maxX, float maxY);
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
    RenderBatch(const RenderBatch&) = delete;
    RenderBatch& operator=(const RenderBatch&) = delete;

    // Allow moving
    RenderBatch(RenderBatch&& other) noexcept;
    RenderBatch& operator=(RenderBatch&& other) noexcept;

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
        PROFILE_FUNCTION();
        return (static_cast<uint64_t>(textureId) << 32) | static_cast<uint64_t>(zIndex);
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
    RenderBatch& getBatch(int textureId, int zIndex);

    // Clear all batches for next frame
    void clear();

    // Sort batches by z-index for correct rendering order
    void sortIfNeeded();

    // Get all batches for rendering
    const std::vector<RenderBatch>& getBatches();

    // Get the number of active batches
    size_t getBatchCount() const;
};

// Main engine struct
typedef struct Engine {
    SDL_Window* window;
    SDL_Renderer* renderer;
    RenderBatchManager renderBatchManager;
    SpatialGrid grid;
    TextureAtlas atlas;  // Changed from TextureAtlas struct to TextureAtlas class
    Camera camera;
    SDL_FRect world_bounds;
    Uint64 last_frame_time;
    float fps;
    std::vector<EntityRef> pending_removals;
    // Entity type system
    EntityManager entityManager;
} Engine;



SDL_FRect get_texture_region(const TextureAtlas& atlas, int16_t texture_id);

// Engine initialization
Engine* engine_create(int window_width, int window_height, int world_width, int world_height, int cell_size);

// Render all entities efficiently
void engine_render_scene(Engine* engine);

// Clean up the engine resources
void engine_destroy(Engine* engine);



// Texture management
int engine_register_texture(Engine* engine, SDL_Surface* surface, int x, int y, int width, int height);


// Entity management
void engine_update_entity_types(Engine* engine, float delta_time);






// Process entity removals
void process_pending_removals(Engine* engine);

// Update engine state
void engine_update(Engine* engine);

// Set entity z-index
void engine_set_entity_z_index(Engine* engine, uint32_t entity_idx, int type_id, uint8_t z_index);

// Present renderer
void engine_present(Engine* engine);



#endif // ENGINE_H