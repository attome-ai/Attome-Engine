#ifndef HIGH_PERFORMANCE_ENGINE_H
#define HIGH_PERFORMANCE_ENGINE_H

#include <ATMEngine/ATMLog.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
// SIMD width detection
#if defined(__AVX512F__)
#define SIMD_WIDTH 16
#define USE_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define SIMD_WIDTH 8
#define USE_AVX2 1
#include <immintrin.h>
#elif defined(__AVX__)
#define SIMD_WIDTH 8
#define USE_AVX 1
#include <immintrin.h>
#elif defined(__SSE__)
#define SIMD_WIDTH 4
#define USE_SSE 1
#include <xmmintrin.h>
#else
#define SIMD_WIDTH 1
#define USE_SCALAR 1
#endif

// Memory and cache alignment
#define CACHE_LINE_SIZE 64
#define PAGE_SIZE (4096) 
#define HUGE_PAGE_SIZE (2 * 1024 * 1024) // 2MB huge pages
#define ALIGNED_SIZE(size) (((size) + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1))
#define ALIGN_TO_CACHE alignas(CACHE_LINE_SIZE)
#define ALIGN_TO_PAGE alignas(PAGE_SIZE)
#define ALIGN_TO_HUGE_PAGE alignas(HUGE_PAGE_SIZE)

// Performance-optimized chunk sizes
#define ENTITY_CHUNK_SIZE 16384         // Increased for better memory locality
#define ENTITY_BATCH_SIZE 512           // Process in larger cache-friendly batches
#define SPATIAL_CELL_CAPACITY 256       // More entities per spatial cell
#define VERTEX_BATCH_SIZE 16384         // Larger vertex batches
#define INSTANCE_BATCH_SIZE 2048        // Larger instance batches
#define MAX_HIERARCHY_DEPTH 32          // Maximum hierarchy depth
#define MAX_PHYSICS_LAYERS 32           // Physics layers
#define VERTEX_CACHE_SIZE 1024          // Increased vertex cache
#define MAX_COMPONENTS 256              // Maximum component types
#define MORTON_GRID_SIZE 256            // Larger spatial grid

#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

// Type definitions for IDs
using EntityTypeID = uint32_t;
using EntityID = uint32_t;
using ChunkID = uint32_t;
using PipelineID = uint32_t;
using MaterialID = uint32_t;
using MeshID = uint32_t;
using TextureID = uint32_t;

// Maximum supported entity types and instances
constexpr uint32_t MAX_ENTITY_TYPES = 256 * 1024; // 256K
constexpr uint32_t MAX_ENTITY_INSTANCES = 4096 * 1024; // 4M
constexpr uint32_t MAX_DEPTH_LEVELS = 32;
constexpr uint32_t MAX_WINDOWS = 16;
constexpr uint32_t INVALID_ID = 0xFFFFFFFF;

constexpr EntityID NULL_ENTITY = UINT64_MAX;



// Memory management
// Memory Allocator for aligned allocations
class MemoryAllocator {
public:
    static void* Allocate(size_t size, size_t alignment = CACHE_LINE_SIZE) {
        return SDL_aligned_alloc(alignment, ALIGNED_SIZE(size));
    }

    static void Free(void* ptr) {
        SDL_aligned_free(ptr);
    }

    // Pool allocator for fixed-size allocations
    template<typename T, size_t BlockSize = 1024>
    class Pool {
    private:
        struct Block {
            uint8_t data[BlockSize * sizeof(T)];
            Block* next;
        };

        Block* currentBlock;
        T* freeList;
        size_t itemsPerBlock;
        size_t numAllocated;

    public:
        Pool() : currentBlock(nullptr), freeList(nullptr), itemsPerBlock(BlockSize), numAllocated(0) {
            // Allocate first block
            currentBlock = (Block*)MemoryAllocator::Allocate(sizeof(Block), CACHE_LINE_SIZE);
            currentBlock->next = nullptr;

            // Initialize free list
            freeList = reinterpret_cast<T*>(currentBlock->data);
            for (size_t i = 0; i < itemsPerBlock - 1; ++i) {
                T* current = freeList + i;
                *reinterpret_cast<T**>(current) = freeList + i + 1;
            }
            *reinterpret_cast<T**>(freeList + itemsPerBlock - 1) = nullptr;
        }

        ~Pool() {
            Block* block = currentBlock;
            while (block) {
                Block* next = block->next;
                MemoryAllocator::Free(block);
                block = next;
            }
        }

        T* Allocate() {
            if (!freeList) {
                // Allocate new block
                Block* newBlock = (Block*)MemoryAllocator::Allocate(sizeof(Block), CACHE_LINE_SIZE);
                newBlock->next = currentBlock;
                currentBlock = newBlock;

                // Initialize free list for new block
                freeList = reinterpret_cast<T*>(currentBlock->data);
                for (size_t i = 0; i < itemsPerBlock - 1; ++i) {
                    T* current = freeList + i;
                    *reinterpret_cast<T**>(current) = freeList + i + 1;
                }
                *reinterpret_cast<T**>(freeList + itemsPerBlock - 1) = nullptr;
            }

            // Get item from free list
            T* result = freeList;
            freeList = *reinterpret_cast<T**>(freeList);
            ++numAllocated;
            return result;
        }

        void Free(T* ptr) {
            if (!ptr) return;

            // Add to free list
            *reinterpret_cast<T**>(ptr) = freeList;
            freeList = ptr;
            --numAllocated;
        }

        size_t GetNumAllocated() const {
            return numAllocated;
        }
    };
};




// Helper function to create a shader
SDL_GPUShader* createShader(SDL_GPUDevice* device, const char* code, SDL_GPUShaderStage stage, const char* entrypoint, int uni);


// Function to create a basic pipeline - note: returning by value is unusual
// and likely problematic for SDL objects, but implementing as requested
SDL_GPUGraphicsPipeline* createBasicPipeline(SDL_GPUDevice* device);
// Create multiple pipelines with different configurations
std::vector<SDL_GPUGraphicsPipeline*> createPipelines(SDL_GPUDevice* device);
// Utility to cleanup pipelines
void releasePipelines(SDL_GPUDevice* device, std::vector<SDL_GPUGraphicsPipeline*>& pipelines);


// Mesh structure - stores references into global buffer
struct ALIGN_TO_CACHE Mesh {

    
    uint32_t VertexOffset;      // Offset into vertex buffer
    uint32_t VertexCount;       // Number of vertices
    uint32_t IndexOffset;       // Offset into index buffer
    uint32_t IndexCount;        // Number of indices
    uint32_t materialIndex;     // Index of material
};





class MeshLoadData
{
    
public:

    std::vector<glm::vec3> pos;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> colors;
    std::vector<glm::vec2> texCoords;
    std::vector<uint16_t> index;

};
class ALIGN_TO_CACHE MeshLoader {
public:

    // Member variables
    SDL_GPUDevice* m_Device;


    std::vector<MeshLoadData> meshes;


    void addMesh(MeshLoadData&& data)
    {
        meshes.push_back(std::move(data));
    }
    // Load all meshes during initialization
    bool LoadAllMeshes(std::vector<const char*> Paths) {
    }

};

class GPUResource
{
    SDL_GPUDevice* device;
    SDL_GPUBuffer* vertexBuffer = nullptr;
    SDL_GPUBuffer* indexBuffer = nullptr;
    SDL_GPUBuffer* indirectBuffer = nullptr;
    std::vector<Mesh> meshes;

    // Upload mesh data to GPU
    bool UploadToGPU(const std::vector<MeshLoadData>& dataMeshes) {
        // Calculate total size needed
        uint32_t totalVertexCount = 0;
        uint32_t totalIndexCount = 0;

        for (const auto& mesh : dataMeshes) {
            totalVertexCount += mesh.pos.size();
            totalIndexCount += mesh.index.size();
        }

        // Create vertex buffer
        SDL_GPUBufferCreateInfo vertexBufferCI = {};
        vertexBufferCI.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
        vertexBufferCI.size = totalVertexCount * (52);
        vertexBuffer = SDL_CreateGPUBuffer(device, &vertexBufferCI);
        ATMLOGC(!vertexBuffer, "error creating vertex buffer");


        // Create index buffer
        SDL_GPUBufferCreateInfo indexBufferCI = {};
        indexBufferCI.usage = SDL_GPU_BUFFERUSAGE_INDEX;
        indexBufferCI.size = totalIndexCount * sizeof(uint16_t);
        indexBuffer = SDL_CreateGPUBuffer(device, &indexBufferCI);

        ATMLOGC(!indexBuffer, "error creating index buffer");

        // Create indirect draw buffer
        SDL_GPUBufferCreateInfo indirectBufferCI = {};
        indirectBufferCI.usage = SDL_GPU_BUFFERUSAGE_INDIRECT;
        indirectBufferCI.size = dataMeshes.size() * sizeof(SDL_GPUIndexedIndirectDrawCommand);
        indirectBuffer = SDL_CreateGPUBuffer(device, &indirectBufferCI);
        ATMLOGC(!indirectBuffer, "error creating indirect buffer");

        // Create staging buffer for uploads
        SDL_GPUTransferBufferCreateInfo stagingBufferCI = {};
        stagingBufferCI.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        stagingBufferCI.size = vertexBufferCI.size + indexBufferCI.size +  indirectBufferCI.size;
        SDL_GPUTransferBuffer* stagingBuffer = SDL_CreateGPUTransferBuffer(device, &stagingBufferCI);
        ATMLOGC(!indirectBuffer, "error creating staging buffer");



        // Upload vertex data
        uint32_t vertexOffset = 0;
        uint32_t indexOffset = 0;
        float* stagingData = (float*)SDL_MapGPUTransferBuffer(device, stagingBuffer, false);

        for (const auto& mesh : dataMeshes) 
        {
            for (int i = 0; i < mesh.pos.size(); i++)
            {
                Mesh meshInfo;
                meshInfo.VertexCount = mesh.pos.size();
                meshInfo.VertexOffset = vertexOffset;

                stagingData[vertexOffset++] =  mesh.pos[i].x;
                stagingData[vertexOffset++] =  mesh.pos[i].y;
                stagingData[vertexOffset++] =  mesh.pos[i].z;

                if (!mesh.normals.empty())
                {
                    stagingData[vertexOffset++] = mesh.normals[i].x;
                    stagingData[vertexOffset++] = mesh.normals[i].y;
                    stagingData[vertexOffset++] = mesh.normals[i].z;
                }
                if (!mesh.texCoords.empty())
                {
                    stagingData[vertexOffset++] = mesh.texCoords[i].x;
                    stagingData[vertexOffset++] = mesh.texCoords[i].y;
                }
                if (!mesh.colors.empty())
                {
                    stagingData[vertexOffset++] = mesh.colors[i].x;
                    stagingData[vertexOffset++] = mesh.colors[i].y;
                    stagingData[vertexOffset++] = mesh.colors[i].z;
                }

                meshes.push_back(std::move(meshInfo));
            }
        }

        for (int i = 0; i < meshes.size(); i++)
        {
            uint16_t* stagingDatai = (uint16_t*)(&stagingData[vertexOffset]);
            std::memcpy(&stagingDatai[indexOffset], dataMeshes[i].index.data(), dataMeshes[i].index.size() * sizeof(uint16_t));
            
            meshes[i].IndexOffset = indexOffset;
            meshes[i].IndexCount = dataMeshes[i].index.size();
            indexOffset += dataMeshes[i].index.size();

        }
        SDL_UnmapGPUTransferBuffer(device, stagingBuffer);




            SDL_GPUCommandBuffer* cmdBuffer = SDL_AcquireGPUCommandBuffer(device);
            SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmdBuffer);

            SDL_GPUTransferBufferLocation srcLocation = {};
            srcLocation.transfer_buffer = stagingBuffer;
            srcLocation.offset = 0;

            SDL_GPUBufferRegion dstRegion = {};
            dstRegion.buffer = vertexBuffer;
            dstRegion.offset = 0;
            dstRegion.size = vertexOffset * sizeof(float);

            SDL_UploadToGPUBuffer(copyPass, &srcLocation, &dstRegion, false);

            srcLocation.transfer_buffer = stagingBuffer;
            srcLocation.offset = vertexOffset;

            SDL_GPUBufferRegion dstRegion = {};
            dstRegion.buffer = indexBuffer;
            dstRegion.offset = 0;
            dstRegion.size = indexOffset * sizeof(uint16_t);

            SDL_UploadToGPUBuffer(copyPass, &srcLocation, &dstRegion, false);

            // End copy pass
            SDL_EndGPUCopyPass(copyPass);

            // Submit command buffer
            ATMLOGE(SDL_SubmitGPUCommandBuffer(cmdBuffer),"error submitting command buffer");
        

        return true;
    }

};



// Base entity chunk - uses Structure of Arrays (SoA)
class ALIGN_TO_CACHE EntityChunk {

public:

    // Function pointers for type-specific operations
    using CreateFunc = void(*)(void*, uint32_t);
    using UpdateFunc = void(*)(void*, uint32_t, uint32_t);
    using RenderFunc = void(*)(void*, uint32_t, uint32_t, void*);

protected:
    EntityTypeID m_TypeID;
    uint32_t m_Capacity;
    uint32_t m_Count;

    // Structure of Arrays (SoA) layout for cache-friendly access
    bool* m_Active;
    EntityID* m_ParentIDs;
    uint8_t* m_DepthLevels;
    glm::vec3* m_Positions;
    glm::quat* m_Rotations;
    glm::vec3* m_Scales;
    uint32_t* m_WorldMatrixIndices;

    // Function pointers for entity type-specific behavior
    CreateFunc m_OnCreate = nullptr;
    UpdateFunc m_OnUpdate = nullptr;
    RenderFunc m_OnRender = nullptr;

public:
    EntityChunk(EntityTypeID typeId, uint32_t capacity = ENTITY_CHUNK_SIZE) :
        m_TypeID(typeId),
        m_Capacity(capacity),
        m_Count(0) {

        // Allocate arrays with alignment
        m_Active = static_cast<bool*>(MemoryAllocator::Allocate(sizeof(bool) * capacity));
        m_ParentIDs = static_cast<EntityID*>(MemoryAllocator::Allocate(sizeof(EntityID) * capacity));
        m_DepthLevels = static_cast<uint8_t*>(MemoryAllocator::Allocate(sizeof(uint8_t) * capacity));
        m_Positions = static_cast<glm::vec3*>(MemoryAllocator::Allocate(sizeof(glm::vec3) * capacity));
        m_Rotations = static_cast<glm::quat*>(MemoryAllocator::Allocate(sizeof(glm::quat) * capacity));
        m_Scales = static_cast<glm::vec3*>(MemoryAllocator::Allocate(sizeof(glm::vec3) * capacity));
        m_WorldMatrixIndices = static_cast<uint32_t*>(MemoryAllocator::Allocate(sizeof(uint32_t) * capacity));

        // Initialize arrays
        for (uint32_t i = 0; i < capacity; ++i) {
            m_Active[i] = false;
            m_ParentIDs[i] = UINT32_MAX; // No parent
            m_DepthLevels[i] = 0;
            m_Positions[i] = glm::vec3(0.0f);
            m_Rotations[i] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); // Identity quaternion
            m_Scales[i] = glm::vec3(1.0f);
            m_WorldMatrixIndices[i] = UINT32_MAX;
        }
    }

    virtual ~EntityChunk() {
        MemoryAllocator::Free(m_Active);
        MemoryAllocator::Free(m_ParentIDs);
        MemoryAllocator::Free(m_DepthLevels);
        MemoryAllocator::Free(m_Positions);
        MemoryAllocator::Free(m_Rotations);
        MemoryAllocator::Free(m_Scales);
        MemoryAllocator::Free(m_WorldMatrixIndices);
    }

    // Create entity in this chunk (initialization only)
    EntityID CreateEntity(EntityID parentID = UINT32_MAX) {
        if (m_Count >= m_Capacity) return UINT32_MAX; // Chunk is full

        uint32_t index = m_Count++;
        m_Active[index] = true;
        m_ParentIDs[index] = parentID;

        // If has parent, set depth = parent depth + 1
        if (parentID != UINT32_MAX) {
            // Parent depth will be resolved by EntityManager
            m_DepthLevels[index] = 0; // Temporary, updated later
        }

        return index;
    }


    // Set callback functions for this entity type
    void SetCallbacks(CreateFunc onCreate, UpdateFunc onUpdate, RenderFunc onRender) {
        m_OnCreate = onCreate;
        m_OnUpdate = onUpdate;
        m_OnRender = onRender;
    }

    // Called when entity is created
    void InvokeCreate(uint32_t index) {
        if (m_OnCreate) m_OnCreate(this, index);
    }

    // Update all entities in batch
    void Update(float deltaTime) {
        // Process entities in SIMD-friendly batches
        for (uint32_t i = 0; i < m_Count; i += ENTITY_BATCH_SIZE) {
            uint32_t batchSize = std::min(ENTITY_BATCH_SIZE, (int) (m_Count - i));
            if (m_OnUpdate) m_OnUpdate(this, i, batchSize);
        }
    }

    // Render all entities in batch
    void Render(void* renderContext) {
        // Process entities in SIMD-friendly batches
        for (uint32_t i = 0; i < m_Count; i += ENTITY_BATCH_SIZE) {
            uint32_t batchSize = std::min(ENTITY_BATCH_SIZE, (int)(m_Count - i));
            if (m_OnRender) m_OnRender(this, i, batchSize, renderContext);
        }
    }

    // Accessors
    EntityTypeID GetTypeID() const { return m_TypeID; }
    uint32_t GetCount() const { return m_Count; }
    uint32_t GetCapacity() const { return m_Capacity; }
    bool IsFull() const { return m_Count >= m_Capacity; }

    // Data accessors with bound checking
    bool IsActive(uint32_t index) const { return index < m_Capacity ? m_Active[index] : false; }
    void SetActive(uint32_t index, bool active) { if (index < m_Capacity) m_Active[index] = active; }

    EntityID GetParentID(uint32_t index) const { return index < m_Capacity ? m_ParentIDs[index] : UINT32_MAX; }
    void SetParentID(uint32_t index, EntityID parentID) { if (index < m_Capacity) m_ParentIDs[index] = parentID; }

    uint8_t GetDepthLevel(uint32_t index) const { return index < m_Capacity ? m_DepthLevels[index] : 0; }
    void SetDepthLevel(uint32_t index, uint8_t depth) { if (index < m_Capacity) m_DepthLevels[index] = depth; }

    const glm::vec3& GetPosition(uint32_t index) const { static glm::vec3 zero(0); return index < m_Capacity ? m_Positions[index] : zero; }
    void SetPosition(uint32_t index, const glm::vec3& position) { if (index < m_Capacity) m_Positions[index] = position; }

    const glm::vec4& GetRotation(uint32_t index) const { static glm::vec4 identity(0, 0, 0, 1); return index < m_Capacity ? m_Rotations[index] : identity; }
    void SetRotation(uint32_t index, const glm::vec4& rotation) { if (index < m_Capacity) m_Rotations[index] = rotation; }

    const glm::vec3& GetScale(uint32_t index) const { static glm::vec3 one(1); return index < m_Capacity ? m_Scales[index] : one; }
    void SetScale(uint32_t index, const glm::vec3& scale) { if (index < m_Capacity) m_Scales[index] = scale; }

    uint32_t GetWorldMatrixIndex(uint32_t index) const { return index < m_Capacity ? m_WorldMatrixIndices[index] : UINT32_MAX; }
    void SetWorldMatrixIndex(uint32_t index, uint32_t matrixIndex) { if (index < m_Capacity) m_WorldMatrixIndices[index] = matrixIndex; }

    // Direct array access (for SIMD processing)
    bool* GetActiveArray() { return m_Active; }
    EntityID* GetParentIDArray() { return m_ParentIDs; }
    uint8_t* GetDepthLevelArray() { return m_DepthLevels; }
    glm::vec3* GetPositionArray() { return m_Positions; }
    glm::vec4* GetRotationArray() { return m_Rotations; }
    glm::vec3* GetScaleArray() { return m_Scales; }
    uint32_t* GetWorldMatrixIndexArray() { return m_WorldMatrixIndices; }
};
// renderer

// scene

// engine
// Maximum supported entity types
constexpr uint32_t MAX_ENTITY_TYPES = 200 * 1024; // 100K
// Maximum supported entity instances
constexpr uint32_t MAX_ENTITY_INSTANCES = 2048 * 1024; // 1M
constexpr uint32_t MAX_DEPTH_LEVELS = 16;
constexpr uint32_t CHUNK_SIZE = 1024;



class EntityChunk
{

    void (*onCreate)(void* ctx);
    void (*onDestroy)(void* ctx);
    //update all instance of same type
    void (*onUpdate)(void* ctx, uint32_t start_idx, uint32_t count);
    std::vector<glm::vec3> pos;
    std::vector<bool> isActive;
};

// Example triangle entity definition
class ALIGN_TO_CACHE TriangleEntityChunk : public EntityChunk {
public:
    TriangleEntityChunk(EntityTypeID typeId, uint32_t capacity = CHUNK_SIZE) :
        EntityChunk(typeId, capacity),
        m_Colors(capacity, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)) {

        // Set create/destroy/update functions
        m_OnCreate = [](EntityChunk* chunk, uint32_t index) {
            auto* triangleChunk = static_cast<TriangleEntityChunk*>(chunk);
            // Initialize triangle entity
            triangleChunk->m_Colors[index] = glm::vec4(1.0f);
            };

        m_OnUpdate = [](EntityChunk* chunk, uint32_t startIdx, uint32_t count) {
            auto* triangleChunk = static_cast<TriangleEntityChunk*>(chunk);
            // Process triangles in SIMD-friendly batches
            for (uint32_t i = 0; i < count; i += SIMD_WIDTH) {
                uint32_t batchSize = std::min(SIMD_WIDTH, count - i);
                // SIMD batch update for triangles
                triangleChunk->UpdateBatch(startIdx + i, batchSize);
            }
            };
    }

    // Batch update for triangles (SIMD-friendly)
    void UpdateBatch(uint32_t startIdx, uint32_t count) {
        // Implement SIMD batch processing here
        // This would use AVX/SSE instructions when available
    }

    // Triangle-specific accessors
    const glm::vec4& GetColor(uint32_t index) const { return m_Colors[index]; }
    void SetColor(uint32_t index, const glm::vec4& color) { m_Colors[index] = color; }

private:
    std::vector<glm::vec4> m_Colors;
};

class ALIGN_TO_CACHE CircleEntityChunk{};

class ALIGN_TO_CACHE PlayerEntityChunk{};

class ALIGN_TO_CACHE PlanetEntityChunk{};

class EntityChunks
{
    std::vector<EntityChunk> chunks;
    

};


// Entity depth level - contains all chunks at the same depth level
class EntityDepth {
public:
    EntityDepth() {}
    ~EntityDepth() {
        // Cleanup is handled by Scene
    }

    // Add a chunk at this depth level
    void AddChunk(EntityChunk* chunk) {
        m_Chunks.push_back(chunk);
    }

    // Get all chunks at this depth level
    const std::vector<EntityChunk*>& GetChunks() const { return m_Chunks; }

private:
    std::vector<EntityChunk*> m_Chunks;
};


class Renderer
{

};
class ALIGN_TO_CACHE Camera {
public:
    Camera() :
        m_Position(0.0f, 0.0f, 5.0f),
        m_Target(0.0f, 0.0f, 0.0f),
        m_Up(0.0f, 1.0f, 0.0f),
        m_FieldOfView(60.0f),
        m_AspectRatio(16.0f / 9.0f),
        m_NearClip(0.1f),
        m_FarClip(1000.0f),
        m_ViewDirty(true),
        m_ProjDirty(true) {

        UpdateViewMatrix();
        UpdateProjectionMatrix();
    }

    void SetPosition(const glm::vec3& position) {
        m_Position = position;
        m_ViewDirty = true;
    }

    void SetTarget(const glm::vec3& target) {
        m_Target = target;
        m_ViewDirty = true;
    }

    void SetUp(const glm::vec3& up) {
        m_Up = up;
        m_ViewDirty = true;
    }

    void SetPerspective(float fov, float aspectRatio, float nearClip, float farClip) {
        m_FieldOfView = fov;
        m_AspectRatio = aspectRatio;
        m_NearClip = nearClip;
        m_FarClip = farClip;
        m_ProjDirty = true;
    }

    void UpdateViewMatrix() {
        if (!m_ViewDirty) return;

        m_ViewMatrix = glm::lookAt(m_Position, m_Target, m_Up);
        m_ViewDirty = false;
    }

    void UpdateProjectionMatrix() {
        if (!m_ProjDirty) return;

        m_ProjMatrix = glm::perspective(glm::radians(m_FieldOfView), m_AspectRatio, m_NearClip, m_FarClip);
        m_ProjDirty = false;
    }

    void Update() {
        UpdateViewMatrix();
        UpdateProjectionMatrix();

        // Combine matrices
        m_ViewProjMatrix = m_ProjMatrix * m_ViewMatrix;
    }

    const glm::mat4& GetViewMatrix() const { return m_ViewMatrix; }
    const glm::mat4& GetProjMatrix() const { return m_ProjMatrix; }
    const glm::mat4& GetViewProjMatrix() const { return m_ViewProjMatrix; }

private:
    glm::vec3 m_Position;
    glm::vec3 m_Target;
    glm::vec3 m_Up;

    float m_FieldOfView;
    float m_AspectRatio;
    float m_NearClip;
    float m_FarClip;

    glm::mat4 m_ViewMatrix;
    glm::mat4 m_ProjMatrix;
    glm::mat4 m_ViewProjMatrix;

    bool m_ViewDirty;
    bool m_ProjDirty;
};
class SceneWindow
{
    Camera camera;
    int x;
    int y;
    int width;
    int height;

};
class Scene
{
public:
    int worldMin;
    int worldMax;

    Scene()
    {
    
    }
    
};

class ATMWindow
{
public:

    Scene scene;
    SDL_GPUDevice* device;
    SDL_Window* window;

    ATMWindow(SDL_Window* window): window(window)
    {
        ATMLOGE(!SDL_ClaimWindowForGPUDevice(device, window), "Failed to claim window for GPU device: %s\n", SDL_GetError());
        // Claim the window for our GPU device
        ATMLOGE(!SDL_SetGPUSwapchainParameters(device, window,
            SDL_GPU_SWAPCHAINCOMPOSITION_SDR, SDL_GPU_PRESENTMODE_MAILBOX), "Failed to set GPU swapchain parameters: %s\n", SDL_GetError());

        ATMLOGE(!SDL_SetGPUAllowedFramesInFlight(device, 2), "Failed to set allowed frames in flight: %s\n", SDL_GetError());
    }
};
class ALIGN_TO_CACHE Timer
{

};
class ALIGN_TO_CACHE Engine
{
    std::vector<ATMWindow> windows;

    Engine()
    {
        ATMLOGE(SDL_Init(SDL_INIT_VIDEO) < 0, "Error initializing SDL: %s\n", SDL_GetError());
    }

    int createWindow(int width, int height)
    {
        SDL_Window *window =  SDL_CreateWindow("High Performance Engine", width, height, 0);
        if (window) 
        {
            windows.push_back(window);
            return windows.size()-1;
        }
        return -1;
    }
    SDL_Window* getWindow(int index) 
    {
        ATMLOGC(windows.size() >= index, "error window index out of range");

        return windows[index].window;
    }

};


// updating  ->   depth -> chunk -> type ->
// phyics -> parallel
// ai -> parallel

#endif