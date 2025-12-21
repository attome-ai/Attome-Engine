// Modified ATMEngine.h to use Vulkan renderer
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
#include <atomic>
#include "ATMProfiler.h"
#include "VulkanRenderer.h"

// Position constants for fixed-point math
static constexpr uint32_t POSITION_MANTISSA_BITS = 10; // 8 bits for mantissa
static constexpr uint32_t POSITION_MANTISSA_MASK = (1 << POSITION_MANTISSA_BITS) - 1;
static constexpr uint32_t POSITION_UNIT = (1 << POSITION_MANTISSA_BITS);

// Helper functions for position conversion
inline int32_t floatToFixedPoint(const float& value) {
    return static_cast<int32_t>(value * POSITION_UNIT);
}

inline float fixedPointToFloat(const int32_t& value) {
    return static_cast<float>(value) / POSITION_UNIT;
}

inline int32_t toGameUnit(const int32_t& value)
{
    return value * POSITION_UNIT;
}

// Spatial grid implementation
static constexpr int32_t WORLD_WIDTH = 50000 * POSITION_UNIT;
static constexpr int32_t WORLD_HEIGHT = 50000 * POSITION_UNIT;
static constexpr int32_t GRID_CELL_SIZE = 1024 * POSITION_UNIT;
static constexpr uint32_t GRID_CELL_WIDTH = (WORLD_WIDTH % GRID_CELL_SIZE) == 0 ? (WORLD_WIDTH / GRID_CELL_SIZE) : (WORLD_WIDTH / GRID_CELL_SIZE) + 1;
static constexpr uint32_t GRID_CELL_HEIGHT = (WORLD_HEIGHT % GRID_CELL_SIZE) == 0 ? (WORLD_HEIGHT / GRID_CELL_SIZE) : (WORLD_HEIGHT / GRID_CELL_SIZE) + 1;
static constexpr int MAX_ENTITIES_PER_CELL = 2213;  // Fixed capacity

// Alignment for memory
#define CACHE_LINE_SIZE 64

// Common constants
static constexpr int MAX_LAYERS = 32;
static constexpr uint32_t INVALID_ID = 0xFFFFFFFF;

// Forward declarations
class Engine;

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

// Camera for culling
class Camera {
public:
    int32_t x, y;  // Changed from uint32_t to int32_t (fixed-point)
    uint32_t width, height;
    uint32_t zoom;
};

// Base Entity Container using SOA with raw pointers
class EntityContainer {
protected:
    // Base entity data
public:
    uint8_t* flags;
    int32_t* x_positions;  // Changed from uint32_t to int32_t (fixed-point)
    int32_t* y_positions;  // Changed from uint32_t to int32_t (fixed-point)
    uint8_t containerFlag;
    int type_id;
    uint8_t default_layer;
    int capacity;
    int count;

    EntityContainer(int typeId, uint8_t defaultLayer, int initialCapacity);
    virtual ~EntityContainer();

    virtual void update(uint32_t delta_time) = 0;  // Changed from float to uint32_t
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

    void update(uint32_t delta_time);  // Changed from float to uint32_t
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
    void update(uint32_t delta_time);  // Changed from float to uint32_t
};

struct EntityRef {
    uint32_t type : 8;
    uint32_t index : 24;
};

class SpatialGrid {
private:
    struct Cell {
        EntityRef entities[MAX_ENTITIES_PER_CELL];
        std::atomic<int> count;  // Track number of entities in cell

        Cell() : count(0) {}
    };

    alignas(CACHE_LINE_SIZE) std::array<std::array<Cell, GRID_CELL_WIDTH>, GRID_CELL_HEIGHT> cells;
    std::vector<EntityRef> queryResult;

public:
    SpatialGrid() {
        queryResult.reserve(15000);
    }

    void add(const EntityRef& entity, const int32_t& x, const int32_t& y) {
        // Convert fixed-point position to cell coordinates
        const uint16_t cellX = x / GRID_CELL_SIZE;
        const uint16_t cellY = y / GRID_CELL_SIZE;
        add2(entity, cellX, cellY);
    }

    void add2(const EntityRef& entity, const uint16_t x, const uint16_t y) {
        Cell& cell = cells[y][x];

        // Atomically increment the counter and get the previous value
        uint32_t index = cell.count.fetch_add(1, std::memory_order_relaxed);

        // Ensure we don't exceed cell capacity
        if (index < MAX_ENTITIES_PER_CELL) {
            cell.entities[index] = entity;
        }
        else
        {
            std::cout << "overflow" << std::endl;
        }
    }

    inline void getCellCoords(const int32_t& x, const int32_t& y, uint16_t& outCellX, uint16_t& outCellY) const {
        outCellX = x / GRID_CELL_SIZE;
        outCellY = y / GRID_CELL_SIZE;
    }

    const std::vector<EntityRef>& queryCircle(int32_t centerX, int32_t centerY, float radius) {
        queryResult.clear();

        float centerXFloat = fixedPointToFloat(centerX);
        float centerYFloat = fixedPointToFloat(centerY);

        float minX = centerXFloat - radius;
        float minY = centerYFloat - radius;
        float maxX = centerXFloat + radius;
        float maxY = centerYFloat + radius;

        uint16_t minCellX, minCellY, maxCellX, maxCellY;
        int32_t minXFixed = floatToFixedPoint(minX);
        int32_t minYFixed = floatToFixedPoint(minY);
        int32_t maxXFixed = floatToFixedPoint(maxX);
        int32_t maxYFixed = floatToFixedPoint(maxY);

        getCellCoords(minXFixed, minYFixed, minCellX, minCellY);
        getCellCoords(maxXFixed, maxYFixed, maxCellX, maxCellY);

        float radiusSq = radius * radius;

        for (uint16_t cy = minCellY; cy <= maxCellY; ++cy) {
            for (uint16_t cx = minCellX; cx <= maxCellX; ++cx) {
                float cellWorldX = cx * GRID_CELL_SIZE;
                float cellWorldY = cy * GRID_CELL_SIZE;

                Cell& cell = cells[cy][cx];
                uint32_t entCount = cell.count;

                for (uint32_t i = 0; i < entCount; ++i) {
                    float dx = cellWorldX - centerXFloat;
                    float dy = cellWorldY - centerYFloat;
                    if ((dx * dx + dy * dy) <= radiusSq) {
                        queryResult.push_back(cell.entities[i]);
                    }
                }
            }
        }
        return queryResult;
    }

    std::vector<EntityRef>& queryRect(int32_t minX, int32_t minY, int32_t maxX, int32_t maxY) {
        queryResult.clear();

        uint16_t minCellX, minCellY, maxCellX, maxCellY;

        getCellCoords(std::max(0, minX), std::max(0, minY), minCellX, minCellY);
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
};

// Vulkan Renderer Texture Atlas
// Vulkan Renderer Texture Atlas
class TextureAtlas {
private:
    VulkanRenderer* vulkanRenderer;
    int atlasTextureId;
    std::vector<VulkanRenderer::TextureRegion> regions;

public:
    TextureAtlas(VulkanRenderer* renderer, int width, int height)
        : vulkanRenderer(renderer), atlasTextureId(-1)
    {
        PROFILE_FUNCTION();
        if (renderer)
            atlasTextureId = renderer->createTextureAtlas(width, height);
    }

    TextureAtlas() : vulkanRenderer(nullptr), atlasTextureId(-1) {}

    ~TextureAtlas() {
        PROFILE_FUNCTION();
        // Textures are cleaned up by the VulkanRenderer
    }

    int registerTexture(SDL_Surface* surface, int x, int y, int width = 0, int height = 0) {
        PROFILE_FUNCTION();
        if (!vulkanRenderer || atlasTextureId == -1) return -1;

        return vulkanRenderer->registerTexture(surface, atlasTextureId, x, y, width, height);
    }

    void getRegion(int textureId, uint16_t& x1, uint16_t& y1, uint16_t& x2, uint16_t& y2) const {
        PROFILE_FUNCTION();

        auto region = vulkanRenderer->getTextureRegion(textureId);
        x1 = region.x;
        y1 = region.y;
        x2 = x1+ region.width;  // This should be just region.width
        y2 = y1+ region.height; // This should be just region.height
    }

    // For compatibility with the existing code
    SDL_Texture* getTexture(int textureId) const {
        return nullptr; // We don't use SDL textures anymore
    }

    void setRenderer(VulkanRenderer* renderer) {
        vulkanRenderer = renderer;
        if (renderer && atlasTextureId == -1) {
            // Create the texture atlas now that we have a valid renderer
            atlasTextureId = renderer->createTextureAtlas(2048, 2048);
        }
    }
};
// Convert Engine from struct to class
class Engine {
public:
    SDL_Window* window;
    VulkanRenderer* vulkanRenderer; // Replaced SDL_Renderer with VulkanRenderer
    SpatialGrid grid;
    TextureAtlas atlas;
    Camera camera;
    SDL_Rect world_bounds;
    uint64_t last_frame_time;
    float fps;
    std::vector<EntityRef> pending_removals;
    EntityManager entityManager;

    // Constructor and destructor
    Engine(int window_width, int window_height, int world_width, int world_height);
    ~Engine();

    // Engine methods
    int registerTexture(SDL_Surface* surface, int x, int y, int width, int height);
    void renderScene();
    void processPendingRemovals();
    void update();
    void setEntityZIndex(uint32_t entity_idx, int type_id, uint8_t z_index);
    void present();
};

#endif // ENGINE_H