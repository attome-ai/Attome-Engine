#include <engine3D/ATMEngine.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <numeric>
// -----------------------------------------------
// Memory Management System Implementation
// -----------------------------------------------

MemoryArena::MemoryArena(size_t size) : capacity(size), used(0), peak_usage(0), owns_memory(true) {
    // Align to huge page size for better performance
    size = ALIGNED_SIZE(size);

    // Use aligned allocation for better cache behavior
#if defined(_WIN32)
    memory = static_cast<uint8_t*>(_aligned_malloc(size, CACHE_LINE_SIZE));
#else
    posix_memalign(reinterpret_cast<void**>(&memory), CACHE_LINE_SIZE, size);
#endif

    // Pre-touch pages to avoid page faults during critical sections
    for (size_t i = 0; i < size; i += PAGE_SIZE) {
        memory[i] = 0;
    }
}

MemoryArena::MemoryArena(void* external_memory, size_t size)
    : memory(static_cast<uint8_t*>(external_memory)), capacity(size), used(0), peak_usage(0), owns_memory(false) {
}

MemoryArena::~MemoryArena() {
    if (owns_memory && memory) {
#if defined(_WIN32)
        _aligned_free(memory);
#else
        free(memory);
#endif
    }
}

void* MemoryArena::allocate(size_t size, size_t alignment) {
    // Calculate aligned address
    size_t mask = alignment - 1;
    size_t misalignment = (reinterpret_cast<uintptr_t>(memory + used) & mask);
    size_t adjustment = (misalignment == 0) ? 0 : (alignment - misalignment);

    // Check if enough space
    if (used + adjustment + size > capacity) {
        return nullptr;
    }

    // Allocate
    void* ptr = memory + used + adjustment;
    used += adjustment + size;

    // Update statistics
    peak_usage = std::max(peak_usage, used);

    return ptr;
}

void MemoryArena::reset() {
    used = 0;
}

// Frame arena allocator implementation
FrameArenaAllocator::FrameArenaAllocator(uint32_t arena_count, size_t arena_size)
    : arena_count(arena_count), current_arena(0), total_capacity(arena_count* arena_size), total_used(0) {

    arenas = new MemoryArena * [arena_count];
    for (uint32_t i = 0; i < arena_count; i++) {
        arenas[i] = new MemoryArena(arena_size);
    }
}

FrameArenaAllocator::~FrameArenaAllocator() {
    for (uint32_t i = 0; i < arena_count; i++) {
        delete arenas[i];
    }
    delete[] arenas;
}

void* FrameArenaAllocator::allocate(size_t size, size_t alignment) {
    void* result = arenas[current_arena]->allocate(size, alignment);
    if (result) {
        total_used.fetch_add(size, std::memory_order_relaxed);
        return result;
    }
    return nullptr;
}

void FrameArenaAllocator::advanceFrame() {
    arenas[current_arena]->reset();
    current_arena = (current_arena + 1) % arena_count;
}

void FrameArenaAllocator::resetAll() {
    for (uint32_t i = 0; i < arena_count; i++) {
        arenas[i]->reset();
    }
    total_used.store(0, std::memory_order_relaxed);
}

// Memory pool implementation template
template<typename T, size_t BlockSize>
TypedMemoryPool<T, BlockSize>::TypedMemoryPool() : blocks(nullptr), total_capacity(0), total_allocated(0) {
    // Create initial block
    blocks = new Block();
    blocks->free_count = BlockSize;
    blocks->next = nullptr;

    // Initialize free list
    for (uint32_t i = 0; i < BlockSize; i++) {
        blocks->free_list[i] = i;
    }

    total_capacity = BlockSize;
}

template<typename T, size_t BlockSize>
TypedMemoryPool<T, BlockSize>::~TypedMemoryPool() {
    Block* current = blocks;
    while (current) {
        Block* next = current->next;
        delete current;
        current = next;
    }
}

template<typename T, size_t BlockSize>
T* TypedMemoryPool<T, BlockSize>::allocate() {
    // Find a block with free space
    Block* current = blocks;
    while (current) {
        if (current->free_count > 0) {
            // Get next free index
            uint32_t free_idx = current->free_list[--current->free_count];
            total_allocated++;

            // Construct object in-place
            T* obj = reinterpret_cast<T*>(&current->data[free_idx * sizeof(T)]);
            return new (obj) T();
        }

        // No space in this block, try next
        if (!current->next) {
            // Allocate new block if needed
            current->next = new Block();
            current->next->free_count = BlockSize;
            current->next->next = nullptr;

            // Initialize free list for new block
            for (uint32_t i = 0; i < BlockSize; i++) {
                current->next->free_list[i] = i;
            }

            total_capacity += BlockSize;
        }

        current = current->next;
    }

    // Should not reach here if new blocks are allocated
    return nullptr;
}

template<typename T, size_t BlockSize>
void TypedMemoryPool<T, BlockSize>::deallocate(T* ptr) {
    // Find block containing the pointer
    Block* current = blocks;
    uint8_t* ptr_byte = reinterpret_cast<uint8_t*>(ptr);

    while (current) {
        uint8_t* block_start = reinterpret_cast<uint8_t*>(&current->data[0]);
        uint8_t* block_end = block_start + (BlockSize * sizeof(T));

        if (ptr_byte >= block_start && ptr_byte < block_end) {
            // Calculate index within block
            uint32_t idx = static_cast<uint32_t>((ptr_byte - block_start) / sizeof(T));

            // Call destructor
            ptr->~T();

            // Add to free list
            current->free_list[current->free_count++] = idx;
            total_allocated--;
            return;
        }

        current = current->next;
    }
}

template<typename T, size_t BlockSize>
void TypedMemoryPool<T, BlockSize>::reset() {
    Block* current = blocks;
    while (current) {
        current->free_count = BlockSize;

        // Reset free list
        for (uint32_t i = 0; i < BlockSize; i++) {
            current->free_list[i] = i;
        }

        // Destroy all objects
        for (uint32_t i = 0; i < BlockSize; i++) {
            T* obj = reinterpret_cast<T*>(&current->data[i * sizeof(T)]);
            obj->~T();
        }

        current = current->next;
    }

    total_allocated = 0;
}


// -----------------------------------------------
// Entity Component System Implementation
// -----------------------------------------------

// Initialize static class members
ComponentTypeId ComponentRegistry::nextId = 0;

// Component mask implementation
void ComponentMask::set(ComponentTypeId id) {
    uint32_t block_idx = id / BITS_PER_BLOCK;
    uint32_t bit_idx = id % BITS_PER_BLOCK;
    blocks[block_idx] |= (1ULL << bit_idx);
}

void ComponentMask::clear(ComponentTypeId id) {
    uint32_t block_idx = id / BITS_PER_BLOCK;
    uint32_t bit_idx = id % BITS_PER_BLOCK;
    blocks[block_idx] &= ~(1ULL << bit_idx);
}

bool ComponentMask::test(ComponentTypeId id) const {
    uint32_t block_idx = id / BITS_PER_BLOCK;
    uint32_t bit_idx = id % BITS_PER_BLOCK;
    return (blocks[block_idx] & (1ULL << bit_idx)) != 0;
}

bool ComponentMask::contains(const ComponentMask& other) const {
    for (uint32_t i = 0; i < NUM_BLOCKS; i++) {
        if ((blocks[i] & other.blocks[i]) != other.blocks[i]) {
            return false;
        }
    }
    return true;
}

uint64_t ComponentMask::hash() const {
    // FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    for (uint32_t i = 0; i < NUM_BLOCKS; i++) {
        hash ^= blocks[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Entity type implementation
EntityType::EntityType() : id(0), instance_size(0), hot_data_size(0), component_count(0),
update_func(nullptr), type_hash(0) {
}

EntityType::~EntityType() {
}

void EntityType::addComponent(const ComponentSpec& component) {
    components.push_back(component);
    component_mask.set(component.typeId);
    component_count++;
}

void EntityType::calculateLayout() {
    // First, sort components by hot/cold and alignment requirements
    std::sort(components.begin(), components.end(), [](const ComponentSpec& a, const ComponentSpec& b) {
        // Hot data first
        if (a.hot_data != b.hot_data) return a.hot_data > b.hot_data;
        // Then by alignment (largest alignment first for better packing)
        return a.alignment > b.alignment;
        });

    // Calculate offsets
    uint32_t hot_offset = 0;
    uint32_t cold_offset = 0;

    // Process hot components first
    for (auto& comp : components) {
        if (comp.hot_data) {
            // Align offset
            uint32_t aligned_offset = (hot_offset + comp.alignment - 1) & ~(comp.alignment - 1);
            comp.offset = aligned_offset;
            hot_offset = aligned_offset + comp.size;
        }
    }

    // Set hot data size
    hot_data_size = hot_offset;

    // Process cold components
    for (auto& comp : components) {
        if (!comp.hot_data) {
            // Align offset
            uint32_t aligned_offset = (cold_offset + comp.alignment - 1) & ~(comp.alignment - 1);
            comp.offset = aligned_offset + hot_data_size; // Offset after hot data
            cold_offset = aligned_offset + comp.size;
        }
    }

    // Set total instance size
    instance_size = hot_data_size + cold_offset;

    // Calculate hash for fast comparison
    type_hash = component_mask.hash();
}

void EntityType::setUpdateFunction(void (*func)(EntityUpdateContext*, uint32_t, uint32_t)) {
    update_func = func;
}

// Entity type registry implementation
EntityTypeRegistry::EntityTypeRegistry(uint32_t max_types) : type_count(0), capacity(max_types) {
    types = new EntityType[max_types];
}

EntityTypeRegistry::~EntityTypeRegistry() {
    delete[] types;
}

uint32_t EntityTypeRegistry::registerType(EntityType&& type) {
    if (type_count >= capacity) {
        return 0xFFFFFFFF; // Error
    }

    // Check if type already exists
    for (uint32_t i = 0; i < type_count; i++) {
        if (types[i].type_hash == type.type_hash &&
            types[i].component_mask.contains(type.component_mask) &&
            type.component_mask.contains(types[i].component_mask)) {
            return types[i].id;
        }
    }

    // New type
    uint32_t type_id = type_count++;
    type.id = type_id;
    types[type_id] = std::move(type);
    return type_id;
}

 EntityType* EntityTypeRegistry::getType(uint32_t id) const {
    if (id >= type_count) {
        return nullptr;
    }
    return &types[id];
}

// Hierarchy data implementation
HierarchyData::HierarchyData(uint32_t max_entities) : max_entities(max_entities) {
    parent_indices = new int32_t[max_entities];
    first_child_indices = new int32_t[max_entities];
    next_sibling_indices = new int32_t[max_entities];
    depths = new int32_t[max_entities];
    depth_first_indices = new uint32_t[max_entities];
    depth_type_sorted_indices = new uint32_t[max_entities];

    // Initialize
    for (uint32_t i = 0; i < max_entities; i++) {
        parent_indices[i] = -1;
        first_child_indices[i] = -1;
        next_sibling_indices[i] = -1;
        depths[i] = 0;
        depth_first_indices[i] = i;
        depth_type_sorted_indices[i] = i;
    }
}

HierarchyData::~HierarchyData() {
    delete[] parent_indices;
    delete[] first_child_indices;
    delete[] next_sibling_indices;
    delete[] depths;
    delete[] depth_first_indices;
    delete[] depth_type_sorted_indices;
}

void HierarchyData::setParent(uint32_t entity_idx, uint32_t parent_idx) {
    if (entity_idx >= max_entities || parent_idx >= max_entities) {
        return;
    }

    // Remove from current parent
    if (parent_indices[entity_idx] != -1) {
        int32_t curr_parent = parent_indices[entity_idx];

        // If we are the first child
        if (first_child_indices[curr_parent] == static_cast<int32_t>(entity_idx)) {
            first_child_indices[curr_parent] = next_sibling_indices[entity_idx];
        }
        else {
            // Find previous sibling
            int32_t sibling = first_child_indices[curr_parent];
            while (sibling != -1 && next_sibling_indices[sibling] != static_cast<int32_t>(entity_idx)) {
                sibling = next_sibling_indices[sibling];
            }

            if (sibling != -1) {
                next_sibling_indices[sibling] = next_sibling_indices[entity_idx];
            }
        }
    }

    // Add to new parent
    parent_indices[entity_idx] = parent_idx;
    next_sibling_indices[entity_idx] = first_child_indices[parent_idx];
    first_child_indices[parent_idx] = entity_idx;
}

int32_t HierarchyData::getParent(uint32_t entity_idx) const {
    if (entity_idx >= max_entities) {
        return -1;
    }
    return parent_indices[entity_idx];
}

void HierarchyData::updateDepths() {
    // Reset depths
    for (uint32_t i = 0; i < max_entities; i++) {
        depths[i] = 0;
    }

    // Recursive helper function to set depths
    std::function<void(uint32_t, int32_t)> setDepth = [&](uint32_t entity_idx, int32_t depth) {
        depths[entity_idx] = depth;

        // Process children
        int32_t child = first_child_indices[entity_idx];
        while (child != -1) {
            setDepth(child, depth + 1);
            child = next_sibling_indices[child];
        }
        };

    // Start with root entities (those without parents)
    for (uint32_t i = 0; i < max_entities; i++) {
        if (parent_indices[i] == -1) {
            setDepth(i, 0);
        }
    }

    // Now sort indices by depth
    std::iota(depth_first_indices, depth_first_indices + max_entities, 0);
    std::sort(depth_first_indices, depth_first_indices + max_entities,
        [this](uint32_t a, uint32_t b) {
            return depths[a] < depths[b];
        });
}

void HierarchyData::sortByDepthAndType(const uint32_t* entity_type_ids, uint32_t entity_count) {
    // First sort by depth
    updateDepths();

    // Copy depth-sorted indices
    std::copy(depth_first_indices, depth_first_indices + entity_count, depth_type_sorted_indices);

    // Then stable sort by type within each depth
    std::vector<std::pair<int32_t, uint32_t>> depth_ranges;

    int32_t current_depth = -1;
    uint32_t start_idx = 0;

    for (uint32_t i = 0; i < entity_count; i++) {
        uint32_t entity_idx = depth_first_indices[i];
        if (depths[entity_idx] != current_depth) {
            if (current_depth != -1) {
                depth_ranges.push_back({ current_depth, start_idx });
            }
            current_depth = depths[entity_idx];
            start_idx = i;
        }
    }

    // Add last range
    if (current_depth != -1) {
        depth_ranges.push_back({ current_depth, start_idx });
    }

    // Sort each range by type
    for (size_t i = 0; i < depth_ranges.size(); i++) {
        uint32_t start = depth_ranges[i].second;
        uint32_t end = (i + 1 < depth_ranges.size()) ? depth_ranges[i + 1].second : entity_count;

        std::stable_sort(depth_type_sorted_indices + start, depth_type_sorted_indices + end,
            [entity_type_ids](uint32_t a, uint32_t b) {
                return entity_type_ids[a] < entity_type_ids[b];
            });
    }
}

// Transform data implementation
TransformData::TransformData(uint32_t max_entities) : max_entities(max_entities) {
    // Allocate memory for transform components with strict SoA layout
    size_t aligned_size = ALIGNED_SIZE(max_entities * sizeof(float));

    // Local transform components
    local_pos_x = new float[max_entities];
    local_pos_y = new float[max_entities];
    local_pos_z = new float[max_entities];
    local_rot_x = new float[max_entities];
    local_rot_y = new float[max_entities];
    local_rot_z = new float[max_entities];
    local_rot_w = new float[max_entities];
    local_scale_x = new float[max_entities];
    local_scale_y = new float[max_entities];
    local_scale_z = new float[max_entities];

    // World transform components
    world_pos_x = new float[max_entities];
    world_pos_y = new float[max_entities];
    world_pos_z = new float[max_entities];
    world_rot_x = new float[max_entities];
    world_rot_y = new float[max_entities];
    world_rot_z = new float[max_entities];
    world_rot_w = new float[max_entities];
    world_scale_x = new float[max_entities];
    world_scale_y = new float[max_entities];
    world_scale_z = new float[max_entities];

    // Cached matrices
    local_matrices = static_cast<glm::mat4*>(
        SDL_aligned_alloc(CACHE_LINE_SIZE, ALIGNED_SIZE(max_entities * sizeof(glm::mat4))));
    world_matrices = static_cast<glm::mat4*>(
        SDL_aligned_alloc(CACHE_LINE_SIZE, ALIGNED_SIZE(max_entities * sizeof(glm::mat4))));

    // Dirty flags
    dirty_flags = new bool[max_entities];

    // Initialize defaults
    for (uint32_t i = 0; i < max_entities; i++) {
        local_pos_x[i] = 0.0f;
        local_pos_y[i] = 0.0f;
        local_pos_z[i] = 0.0f;

        local_rot_x[i] = 0.0f;
        local_rot_y[i] = 0.0f;
        local_rot_z[i] = 0.0f;
        local_rot_w[i] = 1.0f;

        local_scale_x[i] = 1.0f;
        local_scale_y[i] = 1.0f;
        local_scale_z[i] = 1.0f;

        // World transforms initialized to match local
        world_pos_x[i] = local_pos_x[i];
        world_pos_y[i] = local_pos_y[i];
        world_pos_z[i] = local_pos_z[i];

        world_rot_x[i] = local_rot_x[i];
        world_rot_y[i] = local_rot_y[i];
        world_rot_z[i] = local_rot_z[i];
        world_rot_w[i] = local_rot_w[i];

        world_scale_x[i] = local_scale_x[i];
        world_scale_y[i] = local_scale_y[i];
        world_scale_z[i] = local_scale_z[i];

        // Identity matrices
        local_matrices[i] = glm::mat4(1.0f);
        world_matrices[i] = glm::mat4(1.0f);

        dirty_flags[i] = true;
    }
}

TransformData::~TransformData() {
    delete[] local_pos_x;
    delete[] local_pos_y;
    delete[] local_pos_z;
    delete[] local_rot_x;
    delete[] local_rot_y;
    delete[] local_rot_z;
    delete[] local_rot_w;
    delete[] local_scale_x;
    delete[] local_scale_y;
    delete[] local_scale_z;

    delete[] world_pos_x;
    delete[] world_pos_y;
    delete[] world_pos_z;
    delete[] world_rot_x;
    delete[] world_rot_y;
    delete[] world_rot_z;
    delete[] world_rot_w;
    delete[] world_scale_x;
    delete[] world_scale_y;
    delete[] world_scale_z;

    SDL_aligned_free(local_matrices);
    SDL_aligned_free(world_matrices);

    delete[] dirty_flags;
}

void TransformData::setLocalPosition(uint32_t entity_idx, const glm::vec3& position) {
    if (entity_idx >= max_entities) return;

    local_pos_x[entity_idx] = position.x;
    local_pos_y[entity_idx] = position.y;
    local_pos_z[entity_idx] = position.z;
    dirty_flags[entity_idx] = true;
}

void TransformData::setLocalRotation(uint32_t entity_idx, const glm::quat& rotation) {
    if (entity_idx >= max_entities) return;

    local_rot_x[entity_idx] = rotation.x;
    local_rot_y[entity_idx] = rotation.y;
    local_rot_z[entity_idx] = rotation.z;
    local_rot_w[entity_idx] = rotation.w;
    dirty_flags[entity_idx] = true;
}

void TransformData::setLocalScale(uint32_t entity_idx, const glm::vec3& scale) {
    if (entity_idx >= max_entities) return;

    local_scale_x[entity_idx] = scale.x;
    local_scale_y[entity_idx] = scale.y;
    local_scale_z[entity_idx] = scale.z;
    dirty_flags[entity_idx] = true;
}

glm::vec3 TransformData::getWorldPosition(uint32_t entity_idx) const {
    if (entity_idx >= max_entities) return glm::vec3(0.0f);

    return glm::vec3(
        world_pos_x[entity_idx],
        world_pos_y[entity_idx],
        world_pos_z[entity_idx]
    );
}

glm::quat TransformData::getWorldRotation(uint32_t entity_idx) const {
    if (entity_idx >= max_entities) return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

    return glm::quat(
        world_rot_w[entity_idx],
        world_rot_x[entity_idx],
        world_rot_y[entity_idx],
        world_rot_z[entity_idx]
    );
}

glm::vec3 TransformData::getWorldScale(uint32_t entity_idx) const {
    if (entity_idx >= max_entities) return glm::vec3(1.0f);

    return glm::vec3(
        world_scale_x[entity_idx],
        world_scale_y[entity_idx],
        world_scale_z[entity_idx]
    );
}

void TransformData::updateLocalMatrices(uint32_t start_idx, uint32_t count) {
    constexpr uint32_t SIMD_BATCH_SIZE = SIMD_WIDTH;
    uint32_t end_idx = start_idx + count;
    uint32_t aligned_end = start_idx + (count / SIMD_BATCH_SIZE) * SIMD_BATCH_SIZE;

    // Process in SIMD batches
    for (uint32_t i = start_idx; i < aligned_end; i += SIMD_BATCH_SIZE) {
        // Load transform components
        SimdVec3 positions = SimdVec3::load(
            &local_pos_x[i], &local_pos_y[i], &local_pos_z[i]);

        SimdQuat rotations = SimdQuat::load(
            &local_rot_x[i], &local_rot_y[i], &local_rot_z[i], &local_rot_w[i]);

        SimdVec3 scales = SimdVec3::load(
            &local_scale_x[i], &local_scale_y[i], &local_scale_z[i]);

        // Create transform matrices
        SimdMat4 matrices = SimdMat4::createTransform(positions, rotations, scales);

        // Store matrices
        matrices.store(&local_matrices[i], SIMD_BATCH_SIZE);

        // Mark as clean
        for (uint32_t j = 0; j < SIMD_BATCH_SIZE; j++) {
            dirty_flags[i + j] = false;
        }
    }

    // Handle remaining entities
    for (uint32_t i = aligned_end; i < end_idx; i++) {
        // Skip if not dirty
        if (!dirty_flags[i]) continue;

        glm::vec3 position(local_pos_x[i], local_pos_y[i], local_pos_z[i]);
        glm::quat rotation(local_rot_w[i], local_rot_x[i], local_rot_y[i], local_rot_z[i]);
        glm::vec3 scale(local_scale_x[i], local_scale_y[i], local_scale_z[i]);

        local_matrices[i] = glm::translate(glm::mat4(1.0f), position) *
            glm::mat4_cast(rotation) *
            glm::scale(glm::mat4(1.0f), scale);

        dirty_flags[i] = false;
    }
}

void TransformData::updateWorldMatricesHierarchical(
    const int32_t* parent_indices,
    const uint32_t* hierarchy_indices,
    uint32_t count,
    uint32_t start_depth,
    uint32_t max_depth) {

    // Process each depth level sequentially to ensure parent matrices are up-to-date
    for (uint32_t depth_level = start_depth; depth_level <= max_depth; depth_level++) {
        for (uint32_t i = 0; i < count; i++) {
            uint32_t entity_idx = hierarchy_indices[i];
            int32_t parent_idx = parent_indices[entity_idx];

            if (parent_idx == -1) {
                // Root entity, world == local
                world_matrices[entity_idx] = local_matrices[entity_idx];
                world_pos_x[entity_idx] = local_pos_x[entity_idx];
                world_pos_y[entity_idx] = local_pos_y[entity_idx];
                world_pos_z[entity_idx] = local_pos_z[entity_idx];
                world_rot_x[entity_idx] = local_rot_x[entity_idx];
                world_rot_y[entity_idx] = local_rot_y[entity_idx];
                world_rot_z[entity_idx] = local_rot_z[entity_idx];
                world_rot_w[entity_idx] = local_rot_w[entity_idx];
                world_scale_x[entity_idx] = local_scale_x[entity_idx];
                world_scale_y[entity_idx] = local_scale_y[entity_idx];
                world_scale_z[entity_idx] = local_scale_z[entity_idx];
            }
            else {
                // Child entity, world = parent_world * local
                world_matrices[entity_idx] = world_matrices[parent_idx] * local_matrices[entity_idx];

                // Extract position, rotation, scale from world matrix
                // This is more efficient than computing each separately 
                // for when we need actual values and not just matrix
                glm::vec3 position, scale;
                glm::quat rotation;

                // Decompose matrix
                glm::vec3 skew;
                glm::vec4 perspective;
                glm::decompose(world_matrices[entity_idx], scale, rotation, position, skew, perspective);

                // Store components
                world_pos_x[entity_idx] = position.x;
                world_pos_y[entity_idx] = position.y;
                world_pos_z[entity_idx] = position.z;

                world_rot_x[entity_idx] = rotation.x;
                world_rot_y[entity_idx] = rotation.y;
                world_rot_z[entity_idx] = rotation.z;
                world_rot_w[entity_idx] = rotation.w;

                world_scale_x[entity_idx] = scale.x;
                world_scale_y[entity_idx] = scale.y;
                world_scale_z[entity_idx] = scale.z;
            }
        }
    }
}

// Entity Storage implementation
EntityStorage::EntityStorage(uint32_t max_entities, uint32_t max_component_types)
    : max_entities(max_entities), entity_count(0), pool_count(0),
    hot_data_size(0), type_count(0) {

    // Entity identification
    entity_ids = new EntityId[max_entities];
    type_ids = new uint32_t[max_entities];
    active_flags = new bool[max_entities];

    // Component pools
    component_pools = new ComponentPool[max_component_types];

    // Type sorting data
    entities_by_type = new uint32_t[max_entities];
    type_start_indices = new uint32_t[MAX_ENTITY_TYPES];
    type_counts = new uint32_t[MAX_ENTITY_TYPES];

    // Initialize
    for (uint32_t i = 0; i < max_entities; i++) {
        entity_ids[i] = EntityId::invalid();
        type_ids[i] = 0xFFFFFFFF;
        active_flags[i] = false;
        entities_by_type[i] = i;
    }

    for (uint32_t i = 0; i < MAX_ENTITY_TYPES; i++) {
        type_start_indices[i] = 0;
        type_counts[i] = 0;
    }

    // Create memory arena for entity data
    memory_arena = new MemoryArena(max_entities * 1024); // Approximate size
}

EntityStorage::~EntityStorage() {
    delete[] entity_ids;
    delete[] type_ids;
    delete[] active_flags;

    // Free component pools
    for (uint32_t i = 0; i < pool_count; i++) {
        if (component_pools[i].data) {
            delete[] component_pools[i].data;
        }
    }
    delete[] component_pools;

    delete[] entities_by_type;
    delete[] type_start_indices;
    delete[] type_counts;

    delete memory_arena;
}

uint32_t EntityStorage::createEntity(uint32_t type_id) {
    // Find free entity slot
    uint32_t entity_idx = 0xFFFFFFFF;
    for (uint32_t i = 0; i < max_entities; i++) {
        if (!active_flags[i]) {
            entity_idx = i;
            break;
        }
    }

    if (entity_idx == 0xFFFFFFFF) {
        return 0xFFFFFFFF; // No free entities
    }

    // Set up entity data
    entity_ids[entity_idx].index = entity_idx;
    entity_ids[entity_idx].generation++;
    type_ids[entity_idx] = type_id;
    active_flags[entity_idx] = true;

    // Update counts
    entity_count++;
    type_counts[type_id]++;

    // Sort entities by type
    sortEntitiesByType();

    return entity_idx;
}

void EntityStorage::destroyEntity(uint32_t entity_idx) {
    if (entity_idx >= max_entities || !active_flags[entity_idx]) {
        return;
    }

    uint32_t type_id = type_ids[entity_idx];

    // Mark as inactive
    active_flags[entity_idx] = false;

    // Update counts
    entity_count--;
    type_counts[type_id]--;

    // Sort entities by type
    sortEntitiesByType();
}

void* EntityStorage::getComponentByTypeId(uint32_t entity_idx, ComponentTypeId comp_id) const {
    if (entity_idx >= max_entities || !active_flags[entity_idx]) {
        return nullptr;
    }

    for (uint32_t i = 0; i < pool_count; i++) {
        if (component_pools[i].component_id == comp_id) {
            return component_pools[i].data + entity_idx * component_pools[i].stride;
        }
    }

    return nullptr;
}

void EntityStorage::sortEntitiesByType() {
    // Reset type counts
    for (uint32_t i = 0; i < MAX_ENTITY_TYPES; i++) {
        type_counts[i] = 0;
    }

    // Count entities per type
    for (uint32_t i = 0; i < max_entities; i++) {
        if (active_flags[i]) {
            type_counts[type_ids[i]]++;
        }
    }

    // Calculate start indices
    uint32_t offset = 0;
    for (uint32_t i = 0; i < MAX_ENTITY_TYPES; i++) {
        type_start_indices[i] = offset;
        offset += type_counts[i];
    }

    // Place entities in sorted order
    std::vector<uint32_t> type_offsets(MAX_ENTITY_TYPES, 0);

    for (uint32_t i = 0; i < max_entities; i++) {
        if (active_flags[i]) {
            uint32_t type_id = type_ids[i];
            uint32_t dest_idx = type_start_indices[type_id] + type_offsets[type_id]++;
            entities_by_type[dest_idx] = i;
        }
    }

    // Count active types
    type_count = 0;
    for (uint32_t i = 0; i < MAX_ENTITY_TYPES; i++) {
        if (type_counts[i] > 0) {
            type_count++;
        }
    }
}

// Physics data implementation
PhysicsData::PhysicsData(uint32_t max_entities, uint32_t max_collisions)
    : max_entities(max_entities), collision_pair_count(0), max_collision_pairs(max_collisions) {

    // Motion data
    velocity_x = new float[max_entities];
    velocity_y = new float[max_entities];
    velocity_z = new float[max_entities];

    // Collision data
    bounds = new AABB[max_entities];
    collision_layers = new uint32_t[max_entities];
    collision_masks = new uint32_t[max_entities];

    // Collision results
    collision_pairs = new CollisionPair[max_collisions];

    // Initialize
    for (uint32_t i = 0; i < max_entities; i++) {
        velocity_x[i] = 0.0f;
        velocity_y[i] = 0.0f;
        velocity_z[i] = 0.0f;

        bounds[i] = AABB(glm::vec3(0.0f), glm::vec3(0.0f));
        collision_layers[i] = 0;
        collision_masks[i] = 0;
    }

    // Default collision matrix - all layers collide with all layers
    for (uint32_t i = 0; i < MAX_PHYSICS_LAYERS; i++) {
        collision_layer_matrix[i] = 0xFFFFFFFF;
    }
}

PhysicsData::~PhysicsData() {
    delete[] velocity_x;
    delete[] velocity_y;
    delete[] velocity_z;

    delete[] bounds;
    delete[] collision_layers;
    delete[] collision_masks;

    delete[] collision_pairs;
}

void PhysicsData::setVelocity(uint32_t entity_idx, const glm::vec3& velocity) {
    if (entity_idx >= max_entities) return;

    velocity_x[entity_idx] = velocity.x;
    velocity_y[entity_idx] = velocity.y;
    velocity_z[entity_idx] = velocity.z;
}

void PhysicsData::setCollisionLayer(uint32_t entity_idx, uint32_t layer) {
    if (entity_idx >= max_entities || layer >= MAX_PHYSICS_LAYERS) return;

    collision_layers[entity_idx] = (1 << layer);
}

void PhysicsData::setCollisionMask(uint32_t entity_idx, uint32_t mask) {
    if (entity_idx >= max_entities) return;

    collision_masks[entity_idx] = mask;
}

void PhysicsData::setBounds(uint32_t entity_idx, const AABB& aabb) {
    if (entity_idx >= max_entities) return;

    bounds[entity_idx] = aabb;
}

void PhysicsData::updatePositions(TransformData* transforms, float delta_time,
    uint32_t start_idx, uint32_t count) {
    constexpr uint32_t SIMD_BATCH_SIZE = SIMD_WIDTH;
    uint32_t end_idx = start_idx + count;
    uint32_t aligned_end = start_idx + (count / SIMD_BATCH_SIZE) * SIMD_BATCH_SIZE;

    // Process in SIMD batches
    for (uint32_t i = start_idx; i < aligned_end; i += SIMD_BATCH_SIZE) {
        // Load positions and velocities
        SimdVec3 positions = SimdVec3::load(
            &transforms->local_pos_x[i],
            &transforms->local_pos_y[i],
            &transforms->local_pos_z[i]);

        SimdVec3 velocities = SimdVec3::load(
            &velocity_x[i], &velocity_y[i], &velocity_z[i]);

        // Multiply velocity by delta time
        velocities = velocities.mul(delta_time);

        // Add to positions
        positions = positions.add(velocities);

        // Store updated positions
        positions.store(
            &transforms->local_pos_x[i],
            &transforms->local_pos_y[i],
            &transforms->local_pos_z[i]);

        // Mark transforms as dirty
        for (uint32_t j = 0; j < SIMD_BATCH_SIZE; j++) {
            transforms->dirty_flags[i + j] = true;
        }
    }

    // Handle remaining entities
    for (uint32_t i = aligned_end; i < end_idx; i++) {
        transforms->local_pos_x[i] += velocity_x[i] * delta_time;
        transforms->local_pos_y[i] += velocity_y[i] * delta_time;
        transforms->local_pos_z[i] += velocity_z[i] * delta_time;

        transforms->dirty_flags[i] = true;
    }
}
void PhysicsData::detectCollisions(const EntityId* entityIds, uint32_t count) {
    clearCollisions();

    // Only detect collisions between active entities
    for (uint32_t i = 0; i < count; i++) {
        uint32_t entity_a = entityIds[i].index;
        uint32_t layer_a = collision_layers[entity_a];
        if (layer_a == 0) continue; // No layer assigned

        for (uint32_t j = i + 1; j < count; j++) {
            uint32_t entity_b = entityIds[j].index;
            uint32_t layer_b = collision_layers[entity_b];
            if (layer_b == 0) continue; // No layer assigned

            // Check layer masks
            if ((layer_a & collision_masks[entity_b]) == 0 ||
                (layer_b & collision_masks[entity_a]) == 0) {
                continue; // Layers don't collide
            }

            // Check AABB collision
            if (bounds[entity_a].intersect(bounds[entity_b])) {
                // Add collision pair
                if (collision_pair_count < max_collision_pairs) {
                    collision_pairs[collision_pair_count].entity_a = entity_a;
                    collision_pairs[collision_pair_count].entity_b = entity_b;
                    collision_pair_count++;
                }
            }
        }
    }
}

void PhysicsData::detectCollisions(const uint32_t* entity_indices, uint32_t count) {
    clearCollisions();

    // Only detect collisions between active entities
    for (uint32_t i = 0; i < count; i++) {
        uint32_t entity_a = entity_indices[i];
        uint32_t layer_a = collision_layers[entity_a];
        if (layer_a == 0) continue; // No layer assigned

        for (uint32_t j = i + 1; j < count; j++) {
            uint32_t entity_b = entity_indices[j];
            uint32_t layer_b = collision_layers[entity_b];
            if (layer_b == 0) continue; // No layer assigned

            // Check layer masks
            if ((layer_a & collision_masks[entity_b]) == 0 ||
                (layer_b & collision_masks[entity_a]) == 0) {
                continue; // Layers don't collide
            }

            // Check AABB collision
            if (bounds[entity_a].intersect(bounds[entity_b])) {
                // Add collision pair
                if (collision_pair_count < max_collision_pairs) {
                    collision_pairs[collision_pair_count].entity_a = entity_a;
                    collision_pairs[collision_pair_count].entity_b = entity_b;
                    collision_pair_count++;
                }
            }
        }
    }
}

void PhysicsData::clearCollisions() {
    collision_pair_count = 0;
}

// Render data implementation
RenderData::RenderData(uint32_t max_entities) : max_entities(max_entities) {
    mesh_ids = new int32_t[max_entities];
    material_ids = new int32_t[max_entities];
    shader_ids = new int32_t[max_entities];
    visibility_flags = new uint8_t[max_entities];
    lod_levels = new uint8_t[max_entities];
    batch_indices = new int32_t[max_entities];

    // Initialize
    for (uint32_t i = 0; i < max_entities; i++) {
        mesh_ids[i] = -1;
        material_ids[i] = -1;
        shader_ids[i] = -1;
        visibility_flags[i] = 0;
        lod_levels[i] = 0;
        batch_indices[i] = -1;
    }
}

RenderData::~RenderData() {
    delete[] mesh_ids;
    delete[] material_ids;
    delete[] shader_ids;
    delete[] visibility_flags;
    delete[] lod_levels;
    delete[] batch_indices;
}

void RenderData::setMesh(uint32_t entity_idx, int32_t mesh_id) {
    if (entity_idx >= max_entities) return;

    mesh_ids[entity_idx] = mesh_id;
}

void RenderData::setMaterial(uint32_t entity_idx, int32_t material_id) {
    if (entity_idx >= max_entities) return;

    material_ids[entity_idx] = material_id;
}

void RenderData::setShader(uint32_t entity_idx, int32_t shader_id) {
    if (entity_idx >= max_entities) return;

    shader_ids[entity_idx] = shader_id;
}

void RenderData::setVisibility(uint32_t entity_idx, bool visible) {
    if (entity_idx >= max_entities) return;

    visibility_flags[entity_idx] = visible ? 1 : 0;
}

int32_t RenderData::getMesh(uint32_t entity_idx) const {
    if (entity_idx >= max_entities) return -1;

    return mesh_ids[entity_idx];
}

int32_t RenderData::getMaterial(uint32_t entity_idx) const {
    if (entity_idx >= max_entities) return -1;

    return material_ids[entity_idx];
}

int32_t RenderData::getShader(uint32_t entity_idx) const {
    if (entity_idx >= max_entities) return -1;

    return shader_ids[entity_idx];
}

bool RenderData::isVisible(uint32_t entity_idx) const {
    if (entity_idx >= max_entities) return false;

    return visibility_flags[entity_idx] != 0;
}


// (Additional MortonGrid methods would be implemented here)

// Render Batch implementation
RenderBatch::RenderBatch(uint32_t capacity)
    : shader_id(-1), mesh_id(-1), material_id(-1),
    instance_count(0), instance_capacity(capacity),
    instance_buffer_id(-1), buffer_dirty(true), sort_key(0) {

    // Allocate memory
    instance_data = static_cast<InstanceData*>(
        SDL_aligned_alloc(CACHE_LINE_SIZE, capacity * sizeof(InstanceData)));
    entity_indices = new uint32_t[capacity];

    // Initialize
    for (uint32_t i = 0; i < capacity; i++) {
        entity_indices[i] = 0xFFFFFFFF;
    }
}

RenderBatch::~RenderBatch() {
    SDL_aligned_free(instance_data);
    delete[] entity_indices;
}

void RenderBatch::addEntity(uint32_t entity_idx) {
    if (instance_count >= instance_capacity) {
        return; // Batch is full
    }

    entity_indices[instance_count] = entity_idx;
    instance_count++;
    buffer_dirty = true;
}

void RenderBatch::removeEntity(uint32_t entity_idx) {
    for (uint32_t i = 0; i < instance_count; i++) {
        if (entity_indices[i] == entity_idx) {
            // Remove by swapping with last
            entity_indices[i] = entity_indices[instance_count - 1];
            entity_indices[instance_count - 1] = 0xFFFFFFFF;
            instance_count--;
            buffer_dirty = true;
            return;
        }
    }
}

void RenderBatch::clear() {
    instance_count = 0;
    buffer_dirty = true;
}

void RenderBatch::updateInstanceData(const TransformData* transforms,
    uint32_t start_idx, uint32_t count) {
    if (count == 0) return;

    for (uint32_t i = start_idx; i < start_idx + count; i++) {
        if (i >= instance_count) break;

        uint32_t entity_idx = entity_indices[i];
        instance_data[i].transform = transforms->world_matrices[entity_idx];
        instance_data[i].color = glm::vec4(1.0f); // Default color
    }

    buffer_dirty = true;
}

void RenderBatch::uploadToGPU(GPUResources* resources) {
    if (!buffer_dirty) return;

    if (instance_buffer_id == -1) {
        // Create buffer if it doesn't exist
        instance_buffer_id = resources->createInstanceBuffer(
            instance_data, instance_capacity * sizeof(InstanceData));
    }
    else {
        // Update existing buffer
        resources->updateBuffer(instance_buffer_id, instance_data,
            instance_count * sizeof(InstanceData));
    }

    buffer_dirty = false;
}

// Material implementation
Material::Material()
    : diffuse_texture(-1), normal_texture(-1), specular_texture(-1), sampler_id(-1),
    diffuse_color(1.0f), specular_color(1.0f), shininess(32.0f),
    diffuse_texture_ptr(nullptr), normal_texture_ptr(nullptr),
    specular_texture_ptr(nullptr), sampler_ptr(nullptr) {
}

// Mesh implementation
Mesh::Mesh()
    : vertex_buffer_id(-1), index_buffer_id(-1), vertex_count(0), index_count(0),
    vertex_buffer_ptr(nullptr), index_buffer_ptr(nullptr),
    lod_meshes(nullptr), lod_count(0) {

    bounds = AABB(glm::vec3(-1.0f), glm::vec3(1.0f));
}

Mesh::~Mesh() {
    delete[] lod_meshes;
}

// -----------------------------------------------
// Camera Implementation
// -----------------------------------------------

Camera::Camera()
    : position(0.0f, 0.0f, 10.0f), target(0.0f, 0.0f, 0.0f), up(0.0f, 1.0f, 0.0f),
    fov(45.0f), aspect_ratio(16.0f / 9.0f), near_plane(0.1f), far_plane(1000.0f),
    matrices_dirty(true), frustum_dirty(true) {

    updateMatrices();
}

void Camera::setPosition(const glm::vec3& pos) {
    position = pos;
    matrices_dirty = true;
    frustum_dirty = true;
}

void Camera::setTarget(const glm::vec3& tgt) {
    target = tgt;
    matrices_dirty = true;
    frustum_dirty = true;
}

void Camera::setUpVector(const glm::vec3& up_vec) {
    up = up_vec;
    matrices_dirty = true;
    frustum_dirty = true;
}

void Camera::setProjection(float fov_deg, float aspect, float near_z, float far_z) {
    fov = fov_deg;
    aspect_ratio = aspect;
    near_plane = near_z;
    far_plane = far_z;
    matrices_dirty = true;
    frustum_dirty = true;
}

 glm::mat4& Camera::getViewMatrix() {
    if (matrices_dirty) {
        updateMatrices();
    }
    return view_matrix;
}

 glm::mat4& Camera::getProjectionMatrix() {
    if (matrices_dirty) {
        updateMatrices();
    }
    return projection_matrix;
}

 glm::mat4& Camera::getViewProjectionMatrix() {
    if (matrices_dirty) {
        updateMatrices();
    }
    return view_projection_matrix;
}

void Camera::updateMatrices() {
    if (!matrices_dirty) return;

    // Calculate view matrix
    view_matrix = glm::lookAt(position, target, up);

    // Calculate projection matrix
    projection_matrix = glm::perspective(
        glm::radians(fov), aspect_ratio, near_plane, far_plane);

    // Combined view-projection matrix
    view_projection_matrix = projection_matrix * view_matrix;

    matrices_dirty = false;
    frustum_dirty = true;
}

void Camera::updateFrustumPlanes() {
    if (!frustum_dirty) return;

    if (matrices_dirty) {
        updateMatrices();
    }

    // Extract frustum planes from view-projection matrix
    const glm::mat4& m = view_projection_matrix;

    // Left plane
    frustum_planes[0] = glm::vec4(
        m[0][3] + m[0][0],
        m[1][3] + m[1][0],
        m[2][3] + m[2][0],
        m[3][3] + m[3][0]
    );

    // Right plane
    frustum_planes[1] = glm::vec4(
        m[0][3] - m[0][0],
        m[1][3] - m[1][0],
        m[2][3] - m[2][0],
        m[3][3] - m[3][0]
    );

    // Bottom plane
    frustum_planes[2] = glm::vec4(
        m[0][3] + m[0][1],
        m[1][3] + m[1][1],
        m[2][3] + m[2][1],
        m[3][3] + m[3][1]
    );

    // Top plane
    frustum_planes[3] = glm::vec4(
        m[0][3] - m[0][1],
        m[1][3] - m[1][1],
        m[2][3] - m[2][1],
        m[3][3] - m[3][1]
    );

    // Near plane
    frustum_planes[4] = glm::vec4(
        m[0][3] + m[0][2],
        m[1][3] + m[1][2],
        m[2][3] + m[2][2],
        m[3][3] + m[3][2]
    );

    // Far plane
    frustum_planes[5] = glm::vec4(
        m[0][3] - m[0][2],
        m[1][3] - m[1][2],
        m[2][3] - m[2][2],
        m[3][3] - m[3][2]
    );

    // Normalize planes
    for (int i = 0; i < 6; i++) {
        float length = sqrtf(
            frustum_planes[i].x * frustum_planes[i].x +
            frustum_planes[i].y * frustum_planes[i].y +
            frustum_planes[i].z * frustum_planes[i].z
        );

        frustum_planes[i] /= length;
    }

    frustum_dirty = false;
}

bool Camera::isPointVisible(const glm::vec3& point) const {
    for (int i = 0; i < 6; i++) {
        if (frustum_planes[i].x * point.x +
            frustum_planes[i].y * point.y +
            frustum_planes[i].z * point.z +
            frustum_planes[i].w <= 0.0f) {
            return false;
        }
    }
    return true;
}

bool Camera::isSphereVisible(const glm::vec3& center, float radius) const {
    for (int i = 0; i < 6; i++) {
        float distance =
            frustum_planes[i].x * center.x +
            frustum_planes[i].y * center.y +
            frustum_planes[i].z * center.z +
            frustum_planes[i].w;

        if (distance <= -radius) {
            return false;
        }
    }
    return true;
}

bool Camera::isAABBVisible(const AABB& aabb) const {
    // Test all 8 corners against each plane
    for (int i = 0; i < 6; i++) {
        int out = 0;

        // Test each corner
        for (int x = 0; x <= 1; x++) {
            for (int y = 0; y <= 1; y++) {
                for (int z = 0; z <= 1; z++) {
                    glm::vec3 corner(
                        x ? aabb.max.x : aabb.min.x,
                        y ? aabb.max.y : aabb.min.y,
                        z ? aabb.max.z : aabb.min.z
                    );

                    float distance =
                        frustum_planes[i].x * corner.x +
                        frustum_planes[i].y * corner.y +
                        frustum_planes[i].z * corner.z +
                        frustum_planes[i].w;

                    if (distance <= 0.0f) {
                        out++;
                    }
                }
            }
        }

        // If all corners are outside this plane, the AABB is not visible
        if (out == 8) {
            return false;
        }
    }

    return true;
}

// SIMD optimized frustum culling for batches of points
// SIMD optimized AABB frustum culling
void Camera::cullAABBsBatch(const AABB* aabbs, bool* results, uint32_t count) const {
    if (frustum_dirty) {
        const_cast<Camera*>(this)->updateFrustumPlanes();
    }

    // Initialize all as potentially visible
    memset(results, 1, count * sizeof(bool));

    // Define aligned_count before any #if blocks
    uint32_t aligned_count = 0;

#if defined(USE_AVX512)
    constexpr uint32_t SIMD_BATCH_SIZE = 16;
    aligned_count = (count / SIMD_BATCH_SIZE) * SIMD_BATCH_SIZE;

    for (uint32_t i = 0; i < aligned_count; i += SIMD_BATCH_SIZE) {
        // Load 16 AABBs at once
        __m512 min_x = _mm512_set_ps(
            aabbs[i + 15].min.x, aabbs[i + 14].min.x, aabbs[i + 13].min.x, aabbs[i + 12].min.x,
            aabbs[i + 11].min.x, aabbs[i + 10].min.x, aabbs[i + 9].min.x, aabbs[i + 8].min.x,
            aabbs[i + 7].min.x, aabbs[i + 6].min.x, aabbs[i + 5].min.x, aabbs[i + 4].min.x,
            aabbs[i + 3].min.x, aabbs[i + 2].min.x, aabbs[i + 1].min.x, aabbs[i + 0].min.x);

        // ... similar for min_y, min_z, max_x, max_y, max_z

        __mmask16 visible_mask = _mm512_int2mask(0xFFFF); // All visible initially

        // Test against each plane
        for (int p = 0; p < 6; p++) {
            // ... AVX512 frustum test implementation

            // Early exit if all culled
            if (_mm512_mask2int(visible_mask) == 0) break;
        }

        // Store results - convert mask to results
        uint16_t mask = _mm512_mask2int(visible_mask);
        for (uint32_t j = 0; j < SIMD_BATCH_SIZE; j++) {
            results[i + j] = (mask & (1 << j)) != 0;
        }
    }

#elif defined(USE_AVX) || defined(USE_AVX2)
    constexpr uint32_t SIMD_BATCH_SIZE = 8;
    aligned_count = (count / SIMD_BATCH_SIZE) * SIMD_BATCH_SIZE;

    for (uint32_t i = 0; i < aligned_count; i += SIMD_BATCH_SIZE) {
        __m256 min_x = _mm256_set_ps(
            aabbs[i + 7].min.x, aabbs[i + 6].min.x, aabbs[i + 5].min.x, aabbs[i + 4].min.x,
            aabbs[i + 3].min.x, aabbs[i + 2].min.x, aabbs[i + 1].min.x, aabbs[i + 0].min.x);
        __m256 min_y = _mm256_set_ps(
            aabbs[i + 7].min.y, aabbs[i + 6].min.y, aabbs[i + 5].min.y, aabbs[i + 4].min.y,
            aabbs[i + 3].min.y, aabbs[i + 2].min.y, aabbs[i + 1].min.y, aabbs[i + 0].min.y);
        __m256 min_z = _mm256_set_ps(
            aabbs[i + 7].min.z, aabbs[i + 6].min.z, aabbs[i + 5].min.z, aabbs[i + 4].min.z,
            aabbs[i + 3].min.z, aabbs[i + 2].min.z, aabbs[i + 1].min.z, aabbs[i + 0].min.z);
        __m256 max_x = _mm256_set_ps(
            aabbs[i + 7].max.x, aabbs[i + 6].max.x, aabbs[i + 5].max.x, aabbs[i + 4].max.x,
            aabbs[i + 3].max.x, aabbs[i + 2].max.x, aabbs[i + 1].max.x, aabbs[i + 0].max.x);
        __m256 max_y = _mm256_set_ps(
            aabbs[i + 7].max.y, aabbs[i + 6].max.y, aabbs[i + 5].max.y, aabbs[i + 4].max.y,
            aabbs[i + 3].max.y, aabbs[i + 2].max.y, aabbs[i + 1].max.y, aabbs[i + 0].max.y);
        __m256 max_z = _mm256_set_ps(
            aabbs[i + 7].max.z, aabbs[i + 6].max.z, aabbs[i + 5].max.z, aabbs[i + 4].max.z,
            aabbs[i + 3].max.z, aabbs[i + 2].max.z, aabbs[i + 1].max.z, aabbs[i + 0].max.z);

        int visible_mask = 0xFF; // All 8 AABBs are visible initially

        // Test against each frustum plane
        for (int p = 0; p < 6; p++) {
            const glm::vec4& plane = frustum_planes[p];
            __m256 plane_x = _mm256_set1_ps(plane.x);
            __m256 plane_y = _mm256_set1_ps(plane.y);
            __m256 plane_z = _mm256_set1_ps(plane.z);
            __m256 plane_w = _mm256_set1_ps(plane.w);

            // Find support points along the normal
            __m256 p_x = _mm256_blendv_ps(min_x, max_x, _mm256_cmp_ps(plane_x, _mm256_setzero_ps(), _CMP_GE_OQ));
            __m256 p_y = _mm256_blendv_ps(min_y, max_y, _mm256_cmp_ps(plane_y, _mm256_setzero_ps(), _CMP_GE_OQ));
            __m256 p_z = _mm256_blendv_ps(min_z, max_z, _mm256_cmp_ps(plane_z, _mm256_setzero_ps(), _CMP_GE_OQ));

            // Calculate distance from AABB's support point to plane
            __m256 distances = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(plane_x, p_x),
                    _mm256_mul_ps(plane_y, p_y)
                ),
                _mm256_add_ps(
                    _mm256_mul_ps(plane_z, p_z),
                    plane_w
                )
            );

            // If distance < 0, box is outside this plane
            __m256 mask = _mm256_cmp_ps(distances, _mm256_setzero_ps(), _CMP_LT_OQ);
            int culled_mask = _mm256_movemask_ps(mask);
            visible_mask &= ~culled_mask;

            // Early exit if all boxes are culled
            if (visible_mask == 0) break;
        }

        // Store culling results
        for (uint32_t j = 0; j < SIMD_BATCH_SIZE; j++) {
            results[i + j] = (visible_mask & (1 << j)) != 0;
        }
    }

#elif defined(USE_SSE)
    constexpr uint32_t SIMD_BATCH_SIZE = 4;
    aligned_count = (count / SIMD_BATCH_SIZE) * SIMD_BATCH_SIZE;

    for (uint32_t i = 0; i < aligned_count; i += SIMD_BATCH_SIZE) {
        __m128 min_x = _mm_set_ps(
            aabbs[i + 3].min.x, aabbs[i + 2].min.x, aabbs[i + 1].min.x, aabbs[i + 0].min.x);
        __m128 min_y = _mm_set_ps(
            aabbs[i + 3].min.y, aabbs[i + 2].min.y, aabbs[i + 1].min.y, aabbs[i + 0].min.y);
        __m128 min_z = _mm_set_ps(
            aabbs[i + 3].min.z, aabbs[i + 2].min.z, aabbs[i + 1].min.z, aabbs[i + 0].min.z);
        __m128 max_x = _mm_set_ps(
            aabbs[i + 3].max.x, aabbs[i + 2].max.x, aabbs[i + 1].max.x, aabbs[i + 0].max.x);
        __m128 max_y = _mm_set_ps(
            aabbs[i + 3].max.y, aabbs[i + 2].max.y, aabbs[i + 1].max.y, aabbs[i + 0].max.y);
        __m128 max_z = _mm_set_ps(
            aabbs[i + 3].max.z, aabbs[i + 2].max.z, aabbs[i + 1].max.z, aabbs[i + 0].max.z);

        int visible_mask = 0xF; // All 4 AABBs are visible initially

        // Test against each frustum plane
        for (int p = 0; p < 6; p++) {
            const glm::vec4& plane = frustum_planes[p];
            __m128 plane_x = _mm_set1_ps(plane.x);
            __m128 plane_y = _mm_set1_ps(plane.y);
            __m128 plane_z = _mm_set1_ps(plane.z);
            __m128 plane_w = _mm_set1_ps(plane.w);

            // Find support points along the normal
            __m128 p_x = _mm_blendv_ps(min_x, max_x, _mm_cmpge_ps(plane_x, _mm_setzero_ps()));
            __m128 p_y = _mm_blendv_ps(min_y, max_y, _mm_cmpge_ps(plane_y, _mm_setzero_ps()));
            __m128 p_z = _mm_blendv_ps(min_z, max_z, _mm_cmpge_ps(plane_z, _mm_setzero_ps()));

            // Calculate distance from AABB's support point to plane
            __m128 distances = _mm_add_ps(
                _mm_add_ps(
                    _mm_mul_ps(plane_x, p_x),
                    _mm_mul_ps(plane_y, p_y)
                ),
                _mm_add_ps(
                    _mm_mul_ps(plane_z, p_z),
                    plane_w
                )
            );

            // If distance < 0, box is outside this plane
            __m128 mask = _mm_cmplt_ps(distances, _mm_setzero_ps());
            int culled_mask = _mm_movemask_ps(mask);
            visible_mask &= ~culled_mask;

            // Early exit if all boxes are culled
            if (visible_mask == 0) break;
        }

        // Store culling results
        for (uint32_t j = 0; j < SIMD_BATCH_SIZE; j++) {
            results[i + j] = (visible_mask & (1 << j)) != 0;
}
    }
#else
    // Scalar fallback implementation - process all AABBs
    // No need to calculate aligned_count for the scalar case, just process everything
    aligned_count = 0;  // Process all remaining AABBs in the loop below

    for (uint32_t i = 0; i < count; i++) {
        const AABB& aabb = aabbs[i];
        bool visible = true;

        // Test all 8 corners against each plane
        for (int p = 0; p < 6 && visible; p++) {
            const glm::vec4& plane = frustum_planes[p];

            // Find the support point in the direction of the plane normal
            glm::vec3 support;
            support.x = plane.x >= 0 ? aabb.max.x : aabb.min.x;
            support.y = plane.y >= 0 ? aabb.max.y : aabb.min.y;
            support.z = plane.z >= 0 ? aabb.max.z : aabb.min.z;

            // Test if support point is behind plane
            float distance = plane.x * support.x +
                plane.y * support.y +
                plane.z * support.z +
                plane.w;

            if (distance < 0) {
                visible = false;
                break;
            }
        }

        results[i] = visible;
    }
#endif

    // Process remaining AABBs
    for (uint32_t i = aligned_count; i < count; i++) {
        results[i] = isAABBVisible(aabbs[i]);
    }
}
// Entity Update Context implementation
EntityUpdateContext::EntityUpdateContext()
    : entity_type(nullptr), entities(nullptr), transforms(nullptr),
    physics(nullptr), render_data(nullptr), delta_time(0.0f),
    total_time(0.0f), thread_id(0), thread_data(nullptr) {
}

void EntityUpdateContext::setTransform(uint32_t entity_idx, const glm::vec3& position,
    const glm::quat& rotation, const glm::vec3& scale) {
    if (!transforms) return;

    transforms->setLocalPosition(entity_idx, position);
    transforms->setLocalRotation(entity_idx, rotation);
    transforms->setLocalScale(entity_idx, scale);
}

glm::mat4 EntityUpdateContext::getWorldMatrix(uint32_t entity_idx) const {
    if (!transforms) return glm::mat4(1.0f);

    return transforms->world_matrices[entity_idx];
}

// Task System implementation
thread_local TaskSystem::ThreadData TaskSystem::thread_data = { 0, nullptr };

TaskSystem::TaskSystem(uint32_t thread_count) : running(false), idle_thread_count(0) {
    initialize(thread_count);
}

TaskSystem::~TaskSystem() {
    shutdown();
}

void TaskSystem::initialize(uint32_t thread_count) {
    if (running) return;

    if (thread_count == 0) {
        thread_count = std::thread::hardware_concurrency();
    }

    queue.completed_count = 0;
    queue.total_count = 0;

    running = true;

    // Create worker threads
    threads.resize(thread_count);
    for (uint32_t i = 0; i < thread_count; i++) {
        threads[i] = std::thread(&TaskSystem::workerThreadFunc, this, i);
    }
}

void TaskSystem::shutdown() {
    if (!running) return;

    // Signal shutdown
    running = false;

    // Wake all threads
    queue.cv.notify_all();

    // Join threads
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    threads.clear();
}

void TaskSystem::submitTask(std::function<void(uint32_t)> task, uint32_t count) {
    if (!running || count == 0) return;

    // Add task to queue
    {
        std::lock_guard<std::mutex> lock(queue.mutex);
        queue.tasks.push_back(task);
        queue.total_count += count;
    }

    // Wake worker threads
    queue.cv.notify_all();
}

void TaskSystem::waitForAll() {
    if (!running) return;

    // Wait for all tasks to complete
    std::unique_lock<std::mutex> lock(queue.mutex);
    queue.cv.wait(lock, [this]() {
        return queue.completed_count == queue.total_count;
        });

    // Reset counters
    queue.completed_count = 0;
    queue.total_count = 0;
}

void TaskSystem::workerThreadFunc(uint32_t thread_id) {
    // Initialize thread data
    thread_data.thread_id = thread_id;
    thread_data.frame_allocator = new FrameArenaAllocator(2, 4 * 1024 * 1024);

    while (running) {
        std::function<void(uint32_t)> task;

        // Get a task from the queue
        {
            std::unique_lock<std::mutex> lock(queue.mutex);

            // Wait for work
            idle_thread_count++;
            queue.cv.wait(lock, [this]() {
                return !running || !queue.tasks.empty();
                });
            idle_thread_count--;

            if (!running && queue.tasks.empty()) {
                break;
            }

            if (!queue.tasks.empty()) {
                task = queue.tasks.back();
                queue.tasks.pop_back();
            }
        }

        // Execute task
        if (task) {
            task(thread_id);

            // Mark as completed
            queue.completed_count.fetch_add(1, std::memory_order_relaxed);

            // Notify waiting threads
            queue.cv.notify_all();
        }
    }

    // Clean up thread data
    delete thread_data.frame_allocator;
}

uint32_t TaskSystem::getCurrentThreadId() {
    return thread_data.thread_id;
}

FrameArenaAllocator* TaskSystem::getThreadFrameAllocator() {
    return thread_data.frame_allocator;
}

void TaskSystem::resetThreadFrameAllocators() {
    // Reset the allocator for the current thread
    if (thread_data.frame_allocator) {
        thread_data.frame_allocator->resetAll();
    }
}


// Required utility implementation
uint64_t RenderCommand::createSortKey(int32_t shader_id, int32_t mesh_id,
    int32_t material_id, bool transparent) {
    // Pack components into 64-bit sort key for optimal batch sorting
    // Format: [shader(16 bits)][transparent(1 bit)][mesh(15 bits)][material(32 bits)]
    uint64_t key = 0;
    key |= static_cast<uint64_t>(shader_id & 0xFFFF) << 48;
    key |= static_cast<uint64_t>(transparent ? 1 : 0) << 47;
    key |= static_cast<uint64_t>(mesh_id & 0x7FFF) << 32;
    key |= static_cast<uint64_t>(material_id & 0xFFFFFFFF);
    return key;
}

// AABB intersection implementation
bool AABB::intersect(const AABB& other) const {
    return (min.x <= other.max.x && max.x >= other.min.x) &&
        (min.y <= other.max.y && max.y >= other.min.y) &&
        (min.z <= other.max.z && max.z >= other.min.z);
}

AABB AABB::transform(const glm::mat4& matrix) const {
    // Transform AABB by matrix (not optimal but works)
    glm::vec4 corners[8];
    corners[0] = glm::vec4(min.x, min.y, min.z, 1.0f);
    corners[1] = glm::vec4(max.x, min.y, min.z, 1.0f);
    corners[2] = glm::vec4(min.x, max.y, min.z, 1.0f);
    corners[3] = glm::vec4(max.x, max.y, min.z, 1.0f);
    corners[4] = glm::vec4(min.x, min.y, max.z, 1.0f);
    corners[5] = glm::vec4(max.x, min.y, max.z, 1.0f);
    corners[6] = glm::vec4(min.x, max.y, max.z, 1.0f);
    corners[7] = glm::vec4(max.x, max.y, max.z, 1.0f);

    // Transform corners
    for (int i = 0; i < 8; i++) {
        corners[i] = matrix * corners[i];
    }

    // Find min/max
    glm::vec3 new_min(corners[0]);
    glm::vec3 new_max(corners[0]);

    for (int i = 1; i < 8; i++) {
        new_min = glm::min(new_min, glm::vec3(corners[i]));
        new_max = glm::max(new_max, glm::vec3(corners[i]));
    }

    return AABB(new_min, new_max);
}