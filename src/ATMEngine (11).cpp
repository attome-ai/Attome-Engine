// engine.cpp - Implementation of the engine components

#include "ATMEngine.h" // Assuming the header definition is in this file



// Data Structures
//---------------


// EntityStorage implementation
EntityStorage::EntityStorage(size_t max_entity_types, size_t max_entities) {
    type_data_stride = max_entity_types;
    type_data = new EntityData * [max_entity_types];
    for (size_t i = 0; i < max_entity_types; ++i) {
        type_data[i] = nullptr;
    }

    // Initialize active bitsets
    active.resize(max_entity_types / SIMD_WIDTH + 1);
    for (auto& bitset : active) {
        bitset.reset(); // All entities inactive initially
    }
}

EntityStorage::~EntityStorage() {
    for (size_t i = 0; i < type_data_stride; ++i) {
        if (type_data[i]) {
            delete type_data[i]->cold;
            delete type_data[i]->hierarchy;
            delete type_data[i];
        }
    }
    delete[] type_data;
}

size_t EntityStorage::allocateEntityType(size_t count) {
    // Find an available type ID
    size_t type_id = 0;
    while (type_id < type_data_stride && type_data[type_id] != nullptr) {
        type_id++;
    }

    if (type_id >= type_data_stride) {
        return -1; // No available type IDs
    }

    // Calculate number of SIMD blocks needed
    size_t num_blocks = (count + SIMD_WIDTH - 1) / SIMD_WIDTH;

    // Allocate main type data
    type_data[type_id] = new EntityData();

    // Allocate cold data
    type_data[type_id]->cold = new EntityData::ColdData();

    // Allocate hierarchy data
    type_data[type_id]->hierarchy = new EntityData::HierarchyData();

    // Initialize hierarchy data
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        type_data[type_id]->hierarchy->parent_idx[i] = -1;  // No parent
        type_data[type_id]->hierarchy->first_child[i] = -1; // No children
        type_data[type_id]->hierarchy->next_sibling[i] = -1; // No siblings
        type_data[type_id]->hierarchy->depth[i] = 0;        // Root level
    }

    return type_id;
}

void EntityStorage::deallocateEntityType(size_t type_id) {
    if (type_id < type_data_stride && type_data[type_id]) {
        delete type_data[type_id]->cold;
        delete type_data[type_id]->hierarchy;
        delete type_data[type_id];
        type_data[type_id] = nullptr;
    }
}

// HierarchyStream implementation
HierarchyStream::HierarchyStream(size_t max_entities) : max_depth(0) {
    // Initialize depth batches
    for (int i = 0; i < MAX_HIERARCHY_DEPTH; ++i) {
        depth_batches[i] = 0;
        depth_ranges[i][0] = -1; // Start index
        depth_ranges[i][1] = -1; // End index
    }

    // Allocate parent matrices and update groups
    parent_matrices = new glm::mat4[max_entities];
    update_groups = new int32_t[max_entities];

    // Initialize parent matrices to identity
    for (size_t i = 0; i < max_entities; ++i) {
        parent_matrices[i] = glm::mat4(1.0f);
        update_groups[i] = 0;
    }
}

HierarchyStream::~HierarchyStream() {
    delete[] parent_matrices;
    delete[] update_groups;
}

void HierarchyStream::sortByDepth(const int32_t* depths, size_t entity_count) {
    // Reset depth batches
    max_depth = 0;
    for (int i = 0; i < MAX_HIERARCHY_DEPTH; ++i) {
        depth_batches[i] = 0;
        depth_ranges[i][0] = -1; // Start index
        depth_ranges[i][1] = -1; // End index
    }

    // Count entities at each depth
    for (size_t i = 0; i < entity_count; ++i) {
        int depth = depths[i];
        if (depth >= 0 && depth < MAX_HIERARCHY_DEPTH) {
            if (depth_ranges[depth][0] == -1) {
                depth_ranges[depth][0] = i; // First entity at this depth
            }
            depth_ranges[depth][1] = i; // Last entity at this depth
            depth_batches[depth]++;
            max_depth = std::max(max_depth, depth);
        }
    }
}

void HierarchyStream::generateUpdateGroups(size_t target_batch_size) {
    int group_id = 0;

    // For each depth level
    for (int depth = 0; depth <= max_depth; ++depth) {
         int start_idx = depth_ranges[depth][0];
         int end_idx = depth_ranges[depth][1];

        if (start_idx != -1 && end_idx != -1) {
            int count = end_idx - start_idx + 1;
            int num_groups = (count + target_batch_size - 1) / target_batch_size;

            // Divide entities at this depth into update groups
            for (int i = 0; i < num_groups; ++i) {
                int group_start = start_idx + i * target_batch_size;
               
                int group_end = std::min( (int) (group_start + target_batch_size - 1), end_idx);

                // Assign group ID to entities in this range
                for (int idx = group_start; idx <= group_end; ++idx) {
                    update_groups[idx] = group_id;
                }

                group_id++;
            }
        }
    }
}


// HierarchyAwareTaskPartitioner implementation
HierarchyAwareTaskPartitioner::HierarchyAwareTaskPartitioner() {
    // No initialization needed
}

HierarchyAwareTaskPartitioner::~HierarchyAwareTaskPartitioner() {
    // No cleanup needed
}

void HierarchyAwareTaskPartitioner::Initialize(int max_depth) {
    // Clear existing batches
    depth_batches.clear();

    // Reserve space for batches
    depth_batches.reserve(max_depth + 1);
}

void HierarchyAwareTaskPartitioner::PartitionByDepth(const int* depths, int entity_count) {
    // Clear existing batches
    depth_batches.clear();

    // Count entities at each depth
    std::vector<int> counts_by_depth(MAX_HIERARCHY_DEPTH, 0);
    int max_depth = 0;

    for (int i = 0; i < entity_count; ++i) {
        int depth = depths[i];
        if (depth >= 0 && depth < MAX_HIERARCHY_DEPTH) {
            counts_by_depth[depth]++;
            max_depth = std::max(max_depth, depth);
        }
    }

    // Create a batch for each depth level
    int start_idx = 0;
    for (int depth = 0; depth <= max_depth; ++depth) {
        int count = counts_by_depth[depth];
        if (count > 0) {
            depth_batches.push_back({ depth, start_idx, count });
            start_idx += count;
        }
    }
}

void HierarchyAwareTaskPartitioner::OptimizePartitioning(int target_batch_size) {
    std::vector<DepthBatch> optimized_batches;

    for (const auto& batch : depth_batches) {
        int depth = batch.depth;
        int start = batch.start_idx;
        int count = batch.count;

        // Determine optimal number of tasks for this depth
        int num_tasks = GetOptimalTaskCount(depth, count, target_batch_size);

        // Split batch if needed
        if (num_tasks <= 1) {
            optimized_batches.push_back(batch);
        }
        else {
            int entities_per_task = (count + num_tasks - 1) / num_tasks;

            for (int task = 0; task < num_tasks; ++task) {
                int task_start = start + task * entities_per_task;
                int task_count = std::min(entities_per_task, count - task * entities_per_task);

                if (task_count > 0) {
                    optimized_batches.push_back({ depth, task_start, task_count });
                }
            }
        }
    }

    // Replace original batches with optimized ones
    depth_batches = std::move(optimized_batches);
}

int HierarchyAwareTaskPartitioner::GetOptimalTaskCount(int depth, int total_entities_at_depth, int target_per_task) const {
    // For lower depths (higher in hierarchy), use smaller batches
    // to maximize parallelism for higher depths
    float depth_factor = std::pow(1.5f, depth);
    int scaled_target = static_cast<int>(target_per_task / depth_factor);
    scaled_target = std::max(16, scaled_target); // Minimum batch size

    // Calculate number of tasks based on adjusted target size
    return (total_entities_at_depth + scaled_target - 1) / scaled_target;
}




// EntityChunk implementation
EntityChunk::EntityChunk(int type_id, int capacity, size_t type_data_size)
    : type_id(type_id), count(0), capacity(capacity) {

    // Allocate hot data
    hot.position = new glm::vec3[capacity];
    hot.velocity = new glm::vec3[capacity];
    hot.rotation = new glm::quat[capacity];
    hot.scale = new glm::vec3[capacity];
    hot.local_transform = new glm::mat4[capacity];
    hot.world_transform = new glm::mat4[capacity];
    hot.active = new bool[capacity];
    hot.collision_mask = new uint32_t[capacity];
    hot.physics_layer = new uint32_t[capacity];
    hot.lod_level = new uint8_t[capacity];

    // Allocate cold data
    cold.debug_id = new int[capacity];
    cold.names = new char* [capacity];
    cold.creation_time = new float[capacity];
    cold.last_accessed_time = new float[capacity];
    cold.flags = new int[capacity];

    // Allocate hierarchy data
    hierarchy.parent_id = new int[capacity];
    hierarchy.first_child_id = new int[capacity];
    hierarchy.next_sibling_id = new int[capacity];
    hierarchy.depth = new int[capacity];
    hierarchy.parent_transform_cache = new glm::mat4[capacity];

    // Allocate entity bookkeeping
    entity_id = new int[capacity];
    local_index = new int[capacity];

    // Allocate activity masks (64 bits per mask)
    int mask_count = (capacity + 63) / 64;
    activity_masks = new uint64_t[mask_count];

    // Calculate transform block count
    transform_block_count = (capacity + SIMD_WIDTH - 1) / SIMD_WIDTH;
    transform_blocks = new TransformBlock<SIMD_WIDTH>[transform_block_count];

    // Allocate type-specific data if needed
    if (type_data_size > 0) {
        type_data = new uint8_t[capacity * type_data_size];
        type_data_stride = type_data_size;
    }
    else {
        type_data = nullptr;
        type_data_stride = 0;
    }

    // Initialize default values
    for (int i = 0; i < capacity; ++i) {
        hot.position[i] = glm::vec3(0.0f);
        hot.velocity[i] = glm::vec3(0.0f);
        hot.rotation[i] = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Identity
        hot.scale[i] = glm::vec3(1.0f);
        hot.local_transform[i] = glm::mat4(1.0f); // Identity
        hot.world_transform[i] = glm::mat4(1.0f); // Identity
        hot.active[i] = false;
        hot.collision_mask[i] = 0;
        hot.physics_layer[i] = 0;
        hot.lod_level[i] = 0;

        hierarchy.parent_id[i] = -1; // No parent
        hierarchy.first_child_id[i] = -1; // No children
        hierarchy.next_sibling_id[i] = -1; // No siblings
        hierarchy.depth[i] = 0; // Root level
        hierarchy.parent_transform_cache[i] = glm::mat4(1.0f); // Identity

        entity_id[i] = -1; // Invalid
        local_index[i] = i;

        cold.names[i] = nullptr;
    }

    // Initialize activity masks
    for (int i = 0; i < mask_count; ++i) {
        activity_masks[i] = 0;
    }
}

EntityChunk::~EntityChunk() {
    // Free hot data
    delete[] hot.position;
    delete[] hot.velocity;
    delete[] hot.rotation;
    delete[] hot.scale;
    delete[] hot.local_transform;
    delete[] hot.world_transform;
    delete[] hot.active;
    delete[] hot.collision_mask;
    delete[] hot.physics_layer;
    delete[] hot.lod_level;

    // Free cold data
    delete[] cold.debug_id;
    // Free name strings
    for (int i = 0; i < capacity; ++i) {
        if (cold.names[i]) {
            delete[] cold.names[i];
        }
    }
    delete[] cold.names;
    delete[] cold.creation_time;
    delete[] cold.last_accessed_time;
    delete[] cold.flags;

    // Free hierarchy data
    delete[] hierarchy.parent_id;
    delete[] hierarchy.first_child_id;
    delete[] hierarchy.next_sibling_id;
    delete[] hierarchy.depth;
    delete[] hierarchy.parent_transform_cache;

    // Free entity bookkeeping
    delete[] entity_id;
    delete[] local_index;

    // Free activity masks
    delete[] activity_masks;

    // Free transform blocks
    delete[] transform_blocks;

    // Free type-specific data
    if (type_data) {
        delete[] static_cast<uint8_t*>(type_data);
    }
}

void EntityChunk::updateTransforms() {
    // Basic implementation - compute local and world transforms
    // A full implementation would use SIMD batching
    for (int i = 0; i < count; ++i) {
        if (!hot.active[i]) continue;

        // Compute local transform
        glm::mat4 translation = glm::translate(glm::mat4(1.0f), hot.position[i]);
        glm::mat4 rotation = glm::mat4_cast(hot.rotation[i]);
        glm::mat4 scale = glm::scale(glm::mat4(1.0f), hot.scale[i]);

        hot.local_transform[i] = translation * rotation * scale;

        // Update world transform based on hierarchy
        int parent_id = hierarchy.parent_id[i];
        if (parent_id >= 0) {
            // Find parent's world transform
            glm::mat4 parent_world;

            // Check if parent is in this chunk
            int parent_local_idx = -1;
            for (int j = 0; j < count; ++j) {
                if (entity_id[j] == parent_id) {
                    parent_local_idx = j;
                    break;
                }
            }

            if (parent_local_idx >= 0) {
                parent_world = hot.world_transform[parent_local_idx];
            }
            else {
                // Parent is in another chunk, use cached transform
                parent_world = hierarchy.parent_transform_cache[i];
            }

            hot.world_transform[i] = parent_world * hot.local_transform[i];
        }
        else {
            // No parent, world transform equals local transform
            hot.world_transform[i] = hot.local_transform[i];
        }
    }
}

void EntityChunk::updateTransformsHierarchical(const int* depths_sorted, const int* entity_indices, int count) {
    // Process entities in order of hierarchy depth
    for (int i = 0; i < count; ++i) {
        int entity_idx = entity_indices[i];

        // Compute local transform
        glm::mat4 translation = glm::translate(glm::mat4(1.0f), hot.position[entity_idx]);
        glm::mat4 rotation = glm::mat4_cast(hot.rotation[entity_idx]);
        glm::mat4 scale = glm::scale(glm::mat4(1.0f), hot.scale[entity_idx]);

        hot.local_transform[entity_idx] = translation * rotation * scale;

        // Update world transform based on hierarchy
        int parent_id = hierarchy.parent_id[entity_idx];
        if (parent_id >= 0) {
            // Find parent's world transform
            glm::mat4 parent_world;

            // Check if parent is in this chunk
            int parent_local_idx = -1;
            for (int j = 0; j < count; ++j) {
                if (entity_id[j] == parent_id) {
                    parent_local_idx = j;
                    break;
                }
            }

            if (parent_local_idx >= 0) {
                parent_world = hot.world_transform[parent_local_idx];
            }
            else {
                // Parent is in another chunk, use cached transform
                parent_world = hierarchy.parent_transform_cache[entity_idx];
            }

            hot.world_transform[entity_idx] = parent_world * hot.local_transform[entity_idx];
        }
        else {
            // No parent, world transform equals local transform
            hot.world_transform[entity_idx] = hot.local_transform[entity_idx];
        }
    }
}

void EntityChunk::processBatch(int start_idx, int batch_size, float delta_time) {
    // Process a batch of entities
    int end_idx = std::min(start_idx + batch_size, count);

    for (int i = start_idx; i < end_idx; ++i) {
        if (!hot.active[i]) continue;

        // Update position based on velocity
        hot.position[i] += hot.velocity[i] * delta_time;

        // Update transform
        glm::mat4 translation = glm::translate(glm::mat4(1.0f), hot.position[i]);
        glm::mat4 rotation = glm::mat4_cast(hot.rotation[i]);
        glm::mat4 scale = glm::scale(glm::mat4(1.0f), hot.scale[i]);

        hot.local_transform[i] = translation * rotation * scale;
    }
}

void EntityChunk::updateActiveEntitiesMasked(float delta_time) {
    // Process entities using activity masks for branch reduction
    int mask_count = (count + 63) / 64;

    for (int mask_idx = 0; mask_idx < mask_count; ++mask_idx) {
        uint64_t mask = activity_masks[mask_idx];
        if (mask == 0) continue; // Skip if no active entities in this mask

        int base_idx = mask_idx * 64;

        while (mask) {
            int bit_idx = BitMaskOps::findFirstSetBit(mask);
            int entity_idx = base_idx + bit_idx;

            if (entity_idx < count) {
                // Update position based on velocity
                hot.position[entity_idx] += hot.velocity[entity_idx] * delta_time;

                // Update transform - in a full implementation, this would be batched
                glm::mat4 translation = glm::translate(glm::mat4(1.0f), hot.position[entity_idx]);
                glm::mat4 rotation = glm::mat4_cast(hot.rotation[entity_idx]);
                glm::mat4 scale = glm::scale(glm::mat4(1.0f), hot.scale[entity_idx]);

                hot.local_transform[entity_idx] = translation * rotation * scale;
            }

            // Clear processed bit
            mask &= ~(1ULL << bit_idx);
        }
    }
}

uint64_t EntityChunk::generateActiveMask(int start_idx, int count) const {
    uint64_t mask = 0;

    for (int i = 0; i < count && i + start_idx < this->count; ++i) {
        if (hot.active[i + start_idx]) {
            mask |= (1ULL << i);
        }
    }

    return mask;
}

void EntityChunk::simdUpdatePositions(float delta_time, int start_idx, int count) {
    // Simple scalar implementation - a full SIMD implementation would use the SimdFloat class
    for (int i = 0; i < count && i + start_idx < this->count; ++i) {
        if (hot.active[i + start_idx]) {
            hot.position[i + start_idx] += hot.velocity[i + start_idx] * delta_time;
        }
    }
}

void EntityChunk::simdUpdateTransforms() {
    // Simple implementation - a full SIMD implementation would process transforms in batches
    updateTransforms();
}

void EntityChunk::sortEntitiesByDepth(int* depths_out, int* indices_out) const {
    // Copy depths and create indices
    for (int i = 0; i < count; ++i) {
        depths_out[i] = hierarchy.depth[i];
        indices_out[i] = i;
    }

    // Sort indices by depth
    std::sort(indices_out, indices_out + count, [&](int a, int b) {
        return depths_out[a] < depths_out[b];
        });
}

void EntityChunk::prefetchEntityData(int entity_idx) const {
    if (entity_idx >= 0 && entity_idx < count) {
        // Prefetch hot data
        PREFETCH(&hot.position[entity_idx]);
        PREFETCH(&hot.rotation[entity_idx]);
        PREFETCH(&hot.scale[entity_idx]);
        PREFETCH(&hot.local_transform[entity_idx]);
        PREFETCH(&hot.world_transform[entity_idx]);

        // Prefetch hierarchy data
        PREFETCH(&hierarchy.parent_id[entity_idx]);
        PREFETCH(&hierarchy.depth[entity_idx]);

        // Don't prefetch cold data unless explicitly needed
    }
}

void EntityChunk::optimizeDataLayout() {
    // Example implementation - reorder entities by hierarchy depth
    // for better cache coherence when updating transforms

    // First, allocate temporary buffers
    glm::vec3* temp_positions = new glm::vec3[count];
    glm::vec3* temp_velocities = new glm::vec3[count];
    glm::quat* temp_rotations = new glm::quat[count];
    glm::vec3* temp_scales = new glm::vec3[count];
    glm::mat4* temp_local_transforms = new glm::mat4[count];
    glm::mat4* temp_world_transforms = new glm::mat4[count];
    bool* temp_active = new bool[count];
    uint32_t* temp_collision_masks = new uint32_t[count];
    uint32_t* temp_physics_layers = new uint32_t[count];
    uint8_t* temp_lod_levels = new uint8_t[count];

    int* temp_parent_ids = new int[count];
    int* temp_first_child_ids = new int[count];
    int* temp_next_sibling_ids = new int[count];
    int* temp_depths = new int[count];
    glm::mat4* temp_parent_transform_caches = new glm::mat4[count];

    int* temp_entity_ids = new int[count];
    int* temp_local_indices = new int[count];

    // Get sort indices by depth
    int* depths = new int[count];
    int* indices = new int[count];
    sortEntitiesByDepth(depths, indices);

    // Copy data in sorted order
    for (int i = 0; i < count; ++i) {
        int src_idx = indices[i];

        temp_positions[i] = hot.position[src_idx];
        temp_velocities[i] = hot.velocity[src_idx];
        temp_rotations[i] = hot.rotation[src_idx];
        temp_scales[i] = hot.scale[src_idx];
        temp_local_transforms[i] = hot.local_transform[src_idx];
        temp_world_transforms[i] = hot.world_transform[src_idx];
        temp_active[i] = hot.active[src_idx];
        temp_collision_masks[i] = hot.collision_mask[src_idx];
        temp_physics_layers[i] = hot.physics_layer[src_idx];
        temp_lod_levels[i] = hot.lod_level[src_idx];

        temp_parent_ids[i] = hierarchy.parent_id[src_idx];
        temp_first_child_ids[i] = hierarchy.first_child_id[src_idx];
        temp_next_sibling_ids[i] = hierarchy.next_sibling_id[src_idx];
        temp_depths[i] = hierarchy.depth[src_idx];
        temp_parent_transform_caches[i] = hierarchy.parent_transform_cache[src_idx];

        temp_entity_ids[i] = entity_id[src_idx];
        temp_local_indices[i] = i; // Update local indices
    }

    // Swap buffers
    std::swap(hot.position, temp_positions);
    std::swap(hot.velocity, temp_velocities);
    std::swap(hot.rotation, temp_rotations);
    std::swap(hot.scale, temp_scales);
    std::swap(hot.local_transform, temp_local_transforms);
    std::swap(hot.world_transform, temp_world_transforms);
    std::swap(hot.active, temp_active);
    std::swap(hot.collision_mask, temp_collision_masks);
    std::swap(hot.physics_layer, temp_physics_layers);
    std::swap(hot.lod_level, temp_lod_levels);

    std::swap(hierarchy.parent_id, temp_parent_ids);
    std::swap(hierarchy.first_child_id, temp_first_child_ids);
    std::swap(hierarchy.next_sibling_id, temp_next_sibling_ids);
    std::swap(hierarchy.depth, temp_depths);
    std::swap(hierarchy.parent_transform_cache, temp_parent_transform_caches);

    std::swap(entity_id, temp_entity_ids);
    std::swap(local_index, temp_local_indices);

    // Update activity masks
    int mask_count = (count + 63) / 64;
    for (int i = 0; i < mask_count; ++i) {
        activity_masks[i] = 0;
    }

    for (int i = 0; i < count; ++i) {
        if (hot.active[i]) {
            int mask_idx = i / 64;
            int bit_idx = i % 64;
            activity_masks[mask_idx] |= (1ULL << bit_idx);
        }
    }

    // Cleanup temporary buffers
    delete[] temp_positions;
    delete[] temp_velocities;
    delete[] temp_rotations;
    delete[] temp_scales;
    delete[] temp_local_transforms;
    delete[] temp_world_transforms;
    delete[] temp_active;
    delete[] temp_collision_masks;
    delete[] temp_physics_layers;
    delete[] temp_lod_levels;

    delete[] temp_parent_ids;
    delete[] temp_first_child_ids;
    delete[] temp_next_sibling_ids;
    delete[] temp_depths;
    delete[] temp_parent_transform_caches;

    delete[] temp_entity_ids;
    delete[] temp_local_indices;

    delete[] depths;
    delete[] indices;
}

// SpatialGrid3D implementation
SpatialGrid3D::SpatialGrid3D(int width, int height, int depth, float cell_size)
    : width(width), height(height), depth(depth), cell_size(cell_size) {

    cell_count = width * height * depth;
    cells = new Cell[cell_count];

    // Initialize cells
    for (int i = 0; i < cell_count; ++i) {
        cells[i].entity_indices = nullptr;
        cells[i].count = 0;
        cells[i].capacity = 0;
    }

    // Allocate query buffer
    query_buffer_size = 1024; // Initial size
    query_buffer = new int[query_buffer_size];

    // Allocate optimized data structures
    cell_active_counts = new int[cell_count];
    entity_to_cell_map = new int[1000000]; // Arbitrary large value
    cell_occupancy = new uint32_t[(cell_count + 31) / 32];
    cell_morton_codes = new uint64_t[cell_count];

    // Initialize
    std::fill_n(cell_active_counts, cell_count, 0);
    std::fill_n(entity_to_cell_map, 1000000, -1);
    std::fill_n(cell_occupancy, (cell_count + 31) / 32, 0);

    // Calculate Morton codes for cells
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int cell_idx = x + y * width + z * width * height;

                // Convert to normalized coordinates
                float nx = static_cast<float>(x) / width;
                float ny = static_cast<float>(y) / height;
                float nz = static_cast<float>(z) / depth;

                // Calculate Morton code
                cell_morton_codes[cell_idx] = MortonSystem::encodeMorton(
                    glm::vec3(nx, ny, nz), 1.0f);
            }
        }
    }
}

SpatialGrid3D::~SpatialGrid3D() {
    // Free cell entity arrays
    for (int i = 0; i < cell_count; ++i) {
        delete[] cells[i].entity_indices;
    }

    delete[] cells;
    delete[] query_buffer;
    delete[] cell_active_counts;
    delete[] entity_to_cell_map;
    delete[] cell_occupancy;
    delete[] cell_morton_codes;
}

void SpatialGrid3D::insertEntity(int entity_idx, const AABB& bounds) {
    // Calculate cell range for the AABB
    glm::vec3 min_cell = bounds.min / cell_size;
    glm::vec3 max_cell = bounds.max / cell_size;

    int min_x = std::max(0, static_cast<int>(min_cell.x));
    int min_y = std::max(0, static_cast<int>(min_cell.y));
    int min_z = std::max(0, static_cast<int>(min_cell.z));

    int max_x = std::min(width - 1, static_cast<int>(max_cell.x));
    int max_y = std::min(height - 1, static_cast<int>(max_cell.y));
    int max_z = std::min(depth - 1, static_cast<int>(max_cell.z));

    // Insert entity into cells
    for (int z = min_z; z <= max_z; ++z) {
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                int cell_idx = x + y * width + z * width * height;

                // Ensure cell has capacity
                if (cells[cell_idx].count >= cells[cell_idx].capacity) {
                    int new_capacity = cells[cell_idx].capacity == 0 ? 16 : cells[cell_idx].capacity * 2;
                    int* new_indices = new int[new_capacity];

                    if (cells[cell_idx].entity_indices) {
                        std::copy_n(cells[cell_idx].entity_indices, cells[cell_idx].count, new_indices);
                        delete[] cells[cell_idx].entity_indices;
                    }

                    cells[cell_idx].entity_indices = new_indices;
                    cells[cell_idx].capacity = new_capacity;
                }

                // Add entity to cell
                cells[cell_idx].entity_indices[cells[cell_idx].count++] = entity_idx;

                // Update occupancy bitmask
                cell_occupancy[cell_idx / 32] |= (1 << (cell_idx % 32));

                // Update entity to cell mapping (use the first cell for simplicity)
                if (entity_to_cell_map[entity_idx] < 0) {
                    entity_to_cell_map[entity_idx] = cell_idx;
                }
            }
        }
    }
}

void SpatialGrid3D::removeEntity(int entity_idx, int cell_idx) {
    // If cell_idx is not provided, try to find it from the map
    if (cell_idx < 0) {
        cell_idx = entity_to_cell_map[entity_idx];
        if (cell_idx < 0) {
            return; // Entity not in grid
        }
    }

    // Remove entity from the cell
    Cell& cell = cells[cell_idx];
    for (int i = 0; i < cell.count; ++i) {
        if (cell.entity_indices[i] == entity_idx) {
            // Swap with last element to avoid shifting
            cell.entity_indices[i] = cell.entity_indices[--cell.count];

            // Update occupancy if cell becomes empty
            if (cell.count == 0) {
                cell_occupancy[cell_idx / 32] &= ~(1 << (cell_idx % 32));
            }

            break;
        }
    }

    // Clear entity to cell mapping
    entity_to_cell_map[entity_idx] = -1;
}

int SpatialGrid3D::queryBox(const AABB& box, int* result, int max_results) {
    // Calculate cell range for the AABB
    glm::vec3 min_cell = box.min / cell_size;
    glm::vec3 max_cell = box.max / cell_size;

    int min_x = std::max(0, static_cast<int>(min_cell.x));
    int min_y = std::max(0, static_cast<int>(min_cell.y));
    int min_z = std::max(0, static_cast<int>(min_cell.z));

    int max_x = std::min(width - 1, static_cast<int>(max_cell.x));
    int max_y = std::min(height - 1, static_cast<int>(max_cell.y));
    int max_z = std::min(depth - 1, static_cast<int>(max_cell.z));

    // Collect entities from cells
    int result_count = 0;

    for (int z = min_z; z <= max_z; ++z) {
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                int cell_idx = x + y * width + z * width * height;

                // Check if cell is occupied using bitmask
                if ((cell_occupancy[cell_idx / 32] & (1 << (cell_idx % 32))) == 0) {
                    continue; // Cell is empty
                }

                // Check all entities in the cell
                const Cell& cell = cells[cell_idx];
                for (int i = 0; i < cell.count; ++i) {
                    int entity_idx = cell.entity_indices[i];

                    // Add entity to result if not already added
                    bool already_added = false;
                    for (int j = 0; j < result_count; ++j) {
                        if (result[j] == entity_idx) {
                            already_added = true;
                            break;
                        }
                    }

                    if (!already_added && result_count < max_results) {
                        result[result_count++] = entity_idx;

                        if (result_count >= max_results) {
                            return result_count;
                        }
                    }
                }
            }
        }
    }

    return result_count;
}

void SpatialGrid3D::updateEntity(int entity_idx, int old_cell_idx, const AABB& bounds) {
    // Remove from old cell
    removeEntity(entity_idx, old_cell_idx);

    // Insert into new cells
    insertEntity(entity_idx, bounds);
}

int SpatialGrid3D::posToCell(const glm::vec3& pos) const {
    int x = static_cast<int>(pos.x / cell_size);
    int y = static_cast<int>(pos.y / cell_size);
    int z = static_cast<int>(pos.z / cell_size);

    if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) {
        return -1; // Out of bounds
    }

    return x + y * width + z * width * height;
}

float SpatialGrid3D::calculateOptimalCellSize() const {
    // This would analyze entity distribution and determine optimal cell size
    // For now, return current cell size
    return cell_size;
}

void SpatialGrid3D::reorderCellsForCacheCoherence() {
    // Sort cells by Morton code for better cache coherence
    // - Create a mapping from cell index to sorted index
    std::vector<std::pair<uint64_t, int>> cell_codes;
    cell_codes.reserve(cell_count);

    for (int i = 0; i < cell_count; ++i) {
        cell_codes.emplace_back(cell_morton_codes[i], i);
    }

    std::sort(cell_codes.begin(), cell_codes.end());

    // - Create new cell array in Morton order
    Cell* new_cells = new Cell[cell_count];

    for (int i = 0; i < cell_count; ++i) {
        int old_idx = cell_codes[i].second;

        // Move cell contents
        new_cells[i].entity_indices = cells[old_idx].entity_indices;
        new_cells[i].count = cells[old_idx].count;
        new_cells[i].capacity = cells[old_idx].capacity;

        // Clear old cell to avoid double deletion
        cells[old_idx].entity_indices = nullptr;
        cells[old_idx].count = 0;
        cells[old_idx].capacity = 0;
    }

    // - Replace old cells
    delete[] cells;
    cells = new_cells;

    // - Update entity to cell mapping
    // This is complex and would require updating all entity_to_cell_map entries
}

// LODSystem implementation
LODSystem::LODSystem(int max_entity_types) : entity_type_count(max_entity_types) {
    entity_lod_data = new EntityLODData[max_entity_types];

    // Initialize LOD data
    for (int i = 0; i < max_entity_types; ++i) {
        entity_lod_data[i].levels = nullptr;
        entity_lod_data[i].level_count = 0;
    }

    // Allocate SIMD distance blocks
    distance_blocks = new LODDistanceBlock[4096]; // Arbitrary capacity
}

LODSystem::~LODSystem() {
    // Free LOD levels
    for (int i = 0; i < entity_type_count; ++i) {
        delete[] entity_lod_data[i].levels;
    }

    delete[] entity_lod_data;
    delete[] distance_blocks;
}

void LODSystem::registerEntityTypeLOD(int type_id, const LODLevel* levels, int level_count) {
    if (type_id < 0 || type_id >= entity_type_count) {
        return;
    }

    // Free any existing levels
    delete[] entity_lod_data[type_id].levels;

    // Allocate and copy new levels
    entity_lod_data[type_id].levels = new LODLevel[level_count];
    entity_lod_data[type_id].level_count = level_count;

    std::copy_n(levels, level_count, entity_lod_data[type_id].levels);
}

void LODSystem::updateLOD(EntityChunk** chunks, int chunk_count, const glm::vec3& camera_pos) {
    // Update LOD for all chunks
    for (int i = 0; i < chunk_count; ++i) {
        EntityChunk* chunk = chunks[i];
        if (!chunk) continue;

        int type_id = chunk->type_id;
        if (type_id < 0 || type_id >= entity_type_count) continue;

        const EntityLODData& lod_data = entity_lod_data[type_id];
        if (lod_data.level_count == 0) continue;

        // Update LOD for entities in this chunk
        for (int j = 0; j < chunk->count; ++j) {
            if (!chunk->hot.active[j]) continue;

            // Calculate distance to camera
            float distance = glm::length(chunk->hot.position[j] - camera_pos);

            // Determine LOD level
            int lod_level = 0;
            for (int k = 0; k < lod_data.level_count; ++k) {
                if (distance >= lod_data.levels[k].distance_threshold) {
                    lod_level = k;
                }
            }

            // Update LOD level
            chunk->hot.lod_level[j] = lod_level;
        }
    }
}

void LODSystem::updateLODParallel(LockFreeTaskSystem& tasks, EntityChunk** chunks, int chunk_count, const glm::vec3& camera_pos) {
    // Create task for each chunk
    for (int i = 0; i < chunk_count; ++i) {
        EntityChunk* chunk = chunks[i];
        if (!chunk) continue;

        int type_id = chunk->type_id;
        if (type_id < 0 || type_id >= entity_type_count) continue;

        const EntityLODData& lod_data = entity_lod_data[type_id];
        if (lod_data.level_count == 0) continue;

        // Create task for this chunk
        auto task = tasks.CreateTask([=]() {
            // Update LOD for entities in this chunk
            for (int j = 0; j < chunk->count; ++j) {
                if (!chunk->hot.active[j]) continue;

                // Calculate distance to camera
                float distance = glm::length(chunk->hot.position[j] - camera_pos);

                // Determine LOD level
                int lod_level = 0;
                for (int k = 0; k < lod_data.level_count; ++k) {
                    if (distance >= lod_data.levels[k].distance_threshold) {
                        lod_level = k;
                    }
                }

                // Update LOD level
                chunk->hot.lod_level[j] = lod_level;
            }
            });

        tasks.ScheduleTask(task);
    }
}

void LODSystem::updateLODWithSimd(EntityChunk* chunk, const glm::vec3& camera_pos) {
    // A full implementation would use SIMD for distance calculations
    // This is a simplified version
    updateLOD(&chunk, 1, camera_pos);
}

void LODSystem::updateLODAndVisibility(EntityChunk** chunks, int chunk_count, const glm::vec3& camera_pos, const void* view_frustum) {
    // This would update LOD and visibility using frustum culling
    // For now, just update LOD
    updateLOD(chunks, chunk_count, camera_pos);
}

// Camera implementation
Camera3D::Camera3D()
    : position(0.0f), target(0.0f, 0.0f, 1.0f), up(0.0f, 1.0f, 0.0f),
    fov(glm::radians(60.0f)), aspect_ratio(16.0f / 9.0f),
    near_plane(0.1f), far_plane(1000.0f) {
}

glm::mat4 Camera3D::getViewMatrix() const {
    return glm::lookAt(position, target, up);
}

glm::mat4 Camera3D::getProjectionMatrix() const {
    return glm::perspective(fov, aspect_ratio, near_plane, far_plane);
}

glm::mat4 Camera3D::getViewProjectionMatrix() const {
    return getProjectionMatrix() * getViewMatrix();
}

void Camera3D::extractFrustumPlanes(glm::vec4* planes) const {
    glm::mat4 viewProj = getViewProjectionMatrix();

    // Left plane
    planes[0].x = viewProj[0][3] + viewProj[0][0];
    planes[0].y = viewProj[1][3] + viewProj[1][0];
    planes[0].z = viewProj[2][3] + viewProj[2][0];
    planes[0].w = viewProj[3][3] + viewProj[3][0];

    // Right plane
    planes[1].x = viewProj[0][3] - viewProj[0][0];
    planes[1].y = viewProj[1][3] - viewProj[1][0];
    planes[1].z = viewProj[2][3] - viewProj[2][0];
    planes[1].w = viewProj[3][3] - viewProj[3][0];

    // Bottom plane
    planes[2].x = viewProj[0][3] + viewProj[0][1];
    planes[2].y = viewProj[1][3] + viewProj[1][1];
    planes[2].z = viewProj[2][3] + viewProj[2][1];
    planes[2].w = viewProj[3][3] + viewProj[3][1];

    // Top plane
    planes[3].x = viewProj[0][3] - viewProj[0][1];
    planes[3].y = viewProj[1][3] - viewProj[1][1];
    planes[3].z = viewProj[2][3] - viewProj[2][1];
    planes[3].w = viewProj[3][3] - viewProj[3][1];

    // Near plane
    planes[4].x = viewProj[0][3] + viewProj[0][2];
    planes[4].y = viewProj[1][3] + viewProj[1][2];
    planes[4].z = viewProj[2][3] + viewProj[2][2];
    planes[4].w = viewProj[3][3] + viewProj[3][2];

    // Far plane
    planes[5].x = viewProj[0][3] - viewProj[0][2];
    planes[5].y = viewProj[1][3] - viewProj[1][2];
    planes[5].z = viewProj[2][3] - viewProj[2][2];
    planes[5].w = viewProj[3][3] - viewProj[3][2];

    // Normalize planes
    for (int i = 0; i < 6; ++i) {
        float length = glm::sqrt(planes[i].x * planes[i].x +
            planes[i].y * planes[i].y +
            planes[i].z * planes[i].z);
        planes[i] /= length;
    }
}

int Camera3D::cullAABBsSimd(const AABB* bounds, int count, bool* results) const {
    // Extract frustum planes
    glm::vec4 planes[6];
    extractFrustumPlanes(planes);

    int visible_count = 0;

    // Cull each AABB against the frustum
    for (int i = 0; i < count; ++i) {
        const AABB& aabb = bounds[i];

        // Check if AABB is outside any frustum plane
        bool inside = true;
        for (int j = 0; j < 6; ++j) {
            // Find the positive vertex (P-vertex) relative to the plane normal
            glm::vec3 p_vertex;
            p_vertex.x = planes[j].x > 0 ? aabb.max.x : aabb.min.x;
            p_vertex.y = planes[j].y > 0 ? aabb.max.y : aabb.min.y;
            p_vertex.z = planes[j].z > 0 ? aabb.max.z : aabb.min.z;

            // If the positive vertex is outside, the whole AABB is outside
            if (planes[j].x * p_vertex.x +
                planes[j].y * p_vertex.y +
                planes[j].z * p_vertex.z +
                planes[j].w < 0) {
                inside = false;
                break;
            }
        }

        results[i] = inside;
        if (inside) {
            visible_count++;
        }
    }

    return visible_count;
}

// EntityManager implementation
EntityManager::EntityManager(int total_entity_count)
    : total_entity_count(total_entity_count), chunk_count(0) {

    // Allocate storage for entities
    storage = new EntityStorage(MAX_ENTITY_TYPES, total_entity_count);

    // Allocate chunks array
    chunk_capacity = 1024; // Initial capacity
    chunks = new EntityChunk * [chunk_capacity];
    for (int i = 0; i < chunk_capacity; ++i) {
        chunks[i] = nullptr;
    }

    // Allocate hierarchy stream
    hierarchy_stream = new HierarchyStream(total_entity_count);

    // Allocate lookup tables
    entity_to_chunk = new int[total_entity_count];
    entity_to_local = new int[total_entity_count];

    // Initialize lookup tables
    for (int i = 0; i < total_entity_count; ++i) {
        entity_to_chunk[i] = -1;
        entity_to_local[i] = -1;
    }
}

EntityManager::~EntityManager() {
    // Free entity storage
    delete storage;

    // Free chunks
    for (int i = 0; i < chunk_capacity; ++i) {
        delete chunks[i];
    }
    delete[] chunks;

    // Free hierarchy stream
    delete hierarchy_stream;

    // Free lookup tables
    delete[] entity_to_chunk;
    delete[] entity_to_local;
}

int EntityManager::registerEntityType(EntityTypeHandler* type_handler) {
    // Add to type handlers
    int type_id = static_cast<int>(entity_type_handlers.size());
    entity_type_handlers.push_back(type_handler);

    // Set type ID in handler
    type_handler->type_id = type_id;

    return type_id;
}

int EntityManager::createEntity(int type_id) {
    // Check if type is valid
    if (type_id < 0 || type_id >= static_cast<int>(entity_type_handlers.size())) {
        return -1;
    }

    // Find or create a chunk for this type
    EntityChunk* chunk = nullptr;
    int chunk_idx = -1;

    for (int i = 0; i < chunk_count; ++i) {
        if (chunks[i] && chunks[i]->type_id == type_id && chunks[i]->count < chunks[i]->capacity) {
            chunk = chunks[i];
            chunk_idx = i;
            break;
        }
    }

    if (!chunk) {
        // Create a new chunk
        if (chunk_count >= chunk_capacity) {
            // Resize chunks array
            int new_capacity = chunk_capacity * 2;
            EntityChunk** new_chunks = new EntityChunk * [new_capacity];

            // Copy old chunks
            for (int i = 0; i < chunk_capacity; ++i) {
                new_chunks[i] = chunks[i];
            }

            // Initialize new chunks
            for (int i = chunk_capacity; i < new_capacity; ++i) {
                new_chunks[i] = nullptr;
            }

            // Replace old array
            delete[] chunks;
            chunks = new_chunks;
            chunk_capacity = new_capacity;
        }

        // Create new chunk
        chunk = new EntityChunk(type_id, ENTITY_CHUNK_SIZE, 0); // No type-specific data for now
        chunks[chunk_count] = chunk;
        chunk_idx = chunk_count++;
    }

    // Add entity to chunk
    int local_idx = chunk->count++;
    int entity_id =  local_idx; // Generate a unique ID

    // Set entity ID in chunk
    chunk->entity_id[local_idx] = entity_id;

    // Update lookup tables
    entity_to_chunk[entity_id] = chunk_idx;
    entity_to_local[entity_id] = local_idx;

    // Mark entity as active
    chunk->hot.active[local_idx] = true;

    // Update activity mask
    int mask_idx = local_idx / 64;
    int bit_idx = local_idx % 64;
    chunk->activity_masks[mask_idx] |= (1ULL << bit_idx);

    return entity_id;
}

void EntityManager::destroyEntity(int entity_idx) {
    // Get chunk and local index
    int chunk_idx = entity_to_chunk[entity_idx];
    int local_idx = entity_to_local[entity_idx];

    if (chunk_idx < 0 || local_idx < 0) {
        return; // Invalid entity
    }

    EntityChunk* chunk = chunks[chunk_idx];

    // Mark entity as inactive
    chunk->hot.active[local_idx] = false;

    // Update activity mask
    int mask_idx = local_idx / 64;
    int bit_idx = local_idx % 64;
    chunk->activity_masks[mask_idx] &= ~(1ULL << bit_idx);

    // Update lookup tables
    entity_to_chunk[entity_idx] = -1;
    entity_to_local[entity_idx] = -1;

    // Note: We don't actually remove the entity from the chunk
    // for efficiency. The entity slot can be reused later.
}

void EntityManager::getChunkIndices(int entity_idx, int* chunk_idx, int* local_idx) const {
    *chunk_idx = entity_to_chunk[entity_idx];
    *local_idx = entity_to_local[entity_idx];
}

bool EntityManager::isValidEntity(int entity_idx) const {
    if (entity_idx < 0 || entity_idx >= total_entity_count) {
        return false;
    }

    int chunk_idx = entity_to_chunk[entity_idx];
    int local_idx = entity_to_local[entity_idx];

    if (chunk_idx < 0 || local_idx < 0) {
        return false;
    }

    EntityChunk* chunk = chunks[chunk_idx];
    return chunk && local_idx < chunk->count && chunk->hot.active[local_idx];
}

void EntityManager::setParent(int entity_idx, int parent_idx) {
    // Check if entity is valid
    if (!isValidEntity(entity_idx)) {
        return;
    }

    // Check if parent is valid or -1 (no parent)
    if (parent_idx != -1 && !isValidEntity(parent_idx)) {
        return;
    }

    // Get entity chunk and local index
    int chunk_idx = entity_to_chunk[entity_idx];
    int local_idx = entity_to_local[entity_idx];

    EntityChunk* chunk = chunks[chunk_idx];

    // Update parent
    chunk->hierarchy.parent_id[local_idx] = parent_idx;

    // Update depth
    if (parent_idx == -1) {
        // Root entity, depth 0
        chunk->hierarchy.depth[local_idx] = 0;
    }
    else {
        // Get parent depth
        int parent_chunk_idx = entity_to_chunk[parent_idx];
        int parent_local_idx = entity_to_local[parent_idx];

        EntityChunk* parent_chunk = chunks[parent_chunk_idx];

        // Set depth to parent depth + 1
        chunk->hierarchy.depth[local_idx] = parent_chunk->hierarchy.depth[parent_local_idx] + 1;

        // Update parent's children list
        chunk->hierarchy.next_sibling_id[local_idx] = parent_chunk->hierarchy.first_child_id[parent_local_idx];
        parent_chunk->hierarchy.first_child_id[parent_local_idx] = entity_idx;
    }
}

void EntityManager::sortByHierarchyDepth() {
    // Collect depth information from all entities
    int* depths = new int[total_entity_count];
    int* indices = new int[total_entity_count];

    int count = 0;

    // Collect entity depths
    for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
        EntityChunk* chunk = chunks[chunk_idx];
        if (!chunk) continue;

        for (int local_idx = 0; local_idx < chunk->count; ++local_idx) {
            if (!chunk->hot.active[local_idx]) continue;

            int entity_idx = chunk->entity_id[local_idx];
            depths[count] = chunk->hierarchy.depth[local_idx];
            indices[count] = entity_idx;
            count++;
        }
    }

    // Sort entities by depth
    hierarchy_stream->sortByDepth(depths, count);

    // Generate update groups for parallel processing
    hierarchy_stream->generateUpdateGroups(ENTITY_BATCH_SIZE);

    delete[] depths;
    delete[] indices;
}

void EntityManager::updateEntityTransforms() {
    // Process entities one chunk at a time
    for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
        EntityChunk* chunk = chunks[chunk_idx];
        if (!chunk) continue;

        chunk->updateTransforms();
    }
}

void EntityManager::updateEntityTransformsHierarchical() {
    // Get depth-sorted entities
    sortByHierarchyDepth();

    // Process entities by depth
    for (int depth = 0; depth <= hierarchy_stream->max_depth; ++depth) {
        int start_idx = hierarchy_stream->depth_ranges[depth][0];
        int end_idx = hierarchy_stream->depth_ranges[depth][1];

        if (start_idx < 0 || end_idx < 0) continue;

        // Process entities at this depth
        for (int i = start_idx; i <= end_idx; ++i) {
            int entity_idx = i; // This would be the entity index in a real implementation

            int chunk_idx = entity_to_chunk[entity_idx];
            int local_idx = entity_to_local[entity_idx];

            if (chunk_idx < 0 || local_idx < 0) continue;

            EntityChunk* chunk = chunks[chunk_idx];

            // Compute local transform
            glm::mat4 translation = glm::translate(glm::mat4(1.0f), chunk->hot.position[local_idx]);
            glm::mat4 rotation = glm::mat4_cast(chunk->hot.rotation[local_idx]);
            glm::mat4 scale = glm::scale(glm::mat4(1.0f), chunk->hot.scale[local_idx]);

            chunk->hot.local_transform[local_idx] = translation * rotation * scale;

            // Update world transform based on hierarchy
            int parent_id = chunk->hierarchy.parent_id[local_idx];
            if (parent_id >= 0) {
                int parent_chunk_idx = entity_to_chunk[parent_id];
                int parent_local_idx = entity_to_local[parent_id];

                if (parent_chunk_idx >= 0 && parent_local_idx >= 0) {
                    EntityChunk* parent_chunk = chunks[parent_chunk_idx];

                    chunk->hot.world_transform[local_idx] =
                        parent_chunk->hot.world_transform[parent_local_idx] *
                        chunk->hot.local_transform[local_idx];
                }
                else {
                    // Parent not found, use local transform as world transform
                    chunk->hot.world_transform[local_idx] = chunk->hot.local_transform[local_idx];
                }
            }
            else {
                // No parent, world transform equals local transform
                chunk->hot.world_transform[local_idx] = chunk->hot.local_transform[local_idx];
            }
        }
    }
}

template<typename Func>
void EntityManager::processEntitiesInOptimalOrder(Func process_func) {
    // Collect and sort entities, then process them
    // This is a simplified implementation

    // Process all chunks
    for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
        EntityChunk* chunk = chunks[chunk_idx];
        if (!chunk) continue;

        // Process entities in this chunk
        for (int local_idx = 0; local_idx < chunk->count; ++local_idx) {
            if (!chunk->hot.active[local_idx]) continue;

            int entity_idx = chunk->entity_id[local_idx];
            process_func(entity_idx);
        }
    }
}

void* EntityManager::getEntityTypeData(int entity_idx) {
    int chunk_idx = entity_to_chunk[entity_idx];
    int local_idx = entity_to_local[entity_idx];

    if (chunk_idx < 0 || local_idx < 0) {
        return nullptr;
    }

    EntityChunk* chunk = chunks[chunk_idx];

    if (!chunk || !chunk->type_data) {
        return nullptr;
    }

    // Calculate offset into type-specific data
    uint8_t* data = static_cast<uint8_t*>(chunk->type_data);
    return data + local_idx * chunk->type_data_stride;
}

void EntityManager::setEntityActive(int entity_idx, bool active) {
    int chunk_idx = entity_to_chunk[entity_idx];
    int local_idx = entity_to_local[entity_idx];

    if (chunk_idx < 0 || local_idx < 0) {
        return;
    }

    EntityChunk* chunk = chunks[chunk_idx];

    // Update active flag
    chunk->hot.active[local_idx] = active;

    // Update activity mask
    int mask_idx = local_idx / 64;
    int bit_idx = local_idx % 64;

    if (active) {
        chunk->activity_masks[mask_idx] |= (1ULL << bit_idx);
    }
    else {
        chunk->activity_masks[mask_idx] &= ~(1ULL << bit_idx);
    }
}

glm::vec3 EntityManager::getEntityPosition(int entity_idx) {
    int chunk_idx = entity_to_chunk[entity_idx];
    int local_idx = entity_to_local[entity_idx];

    if (chunk_idx < 0 || local_idx < 0) {
        return glm::vec3(0.0f);
    }

    EntityChunk* chunk = chunks[chunk_idx];
    return chunk->hot.position[local_idx];
}

void EntityManager::setEntityPosition(int entity_idx, const glm::vec3& position) {
    int chunk_idx = entity_to_chunk[entity_idx];
    int local_idx = entity_to_local[entity_idx];

    if (chunk_idx < 0 || local_idx < 0) {
        return;
    }

    EntityChunk* chunk = chunks[chunk_idx];
    chunk->hot.position[local_idx] = position;
}

void EntityManager::setEntityLocalPosition(int entity_idx, const glm::vec3& local_position) {
    int chunk_idx = entity_to_chunk[entity_idx];
    int local_idx = entity_to_local[entity_idx];

    if (chunk_idx < 0 || local_idx < 0) {
        return;
    }

    EntityChunk* chunk = chunks[chunk_idx];

    // Store local position
    chunk->hot.position[local_idx] = local_position;

    // Mark transform as dirty (would be implemented in a full engine)
}

void EntityManager::prefetchEntityData(int entity_idx) {
    int chunk_idx = entity_to_chunk[entity_idx];
    int local_idx = entity_to_local[entity_idx];

    if (chunk_idx < 0 || local_idx < 0) {
        return;
    }

    EntityChunk* chunk = chunks[chunk_idx];
    chunk->prefetchEntityData(local_idx);
}

void EntityManager::prefetchChunkData(EntityChunk* chunk, int start_idx, int count) {
    if (!chunk) return;

    // Prefetch a range of entities in a chunk
    for (int i = 0; i < count && i + start_idx < chunk->count; ++i) {
        chunk->prefetchEntityData(start_idx + i);
    }
}

void EntityManager::prefetchEntityTransform(int entity_idx) {
    int chunk_idx = entity_to_chunk[entity_idx];
    int local_idx = entity_to_local[entity_idx];

    if (chunk_idx < 0 || local_idx < 0) {
        return;
    }

    EntityChunk* chunk = chunks[chunk_idx];

    // Prefetch transform data
    PREFETCH(&chunk->hot.position[local_idx]);
    PREFETCH(&chunk->hot.rotation[local_idx]);
    PREFETCH(&chunk->hot.scale[local_idx]);
    PREFETCH(&chunk->hot.local_transform[local_idx]);
    PREFETCH(&chunk->hot.world_transform[local_idx]);
}

void EntityManager::optimizeMemoryLayout() {
    // Optimize memory layout of entities
    for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
        EntityChunk* chunk = chunks[chunk_idx];
        if (!chunk) continue;

        chunk->optimizeDataLayout();
    }

    // Update entity lookup tables
    for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
        EntityChunk* chunk = chunks[chunk_idx];
        if (!chunk) continue;

        for (int local_idx = 0; local_idx < chunk->count; ++local_idx) {
            int entity_idx = chunk->entity_id[local_idx];

            if (entity_idx >= 0) {
                entity_to_chunk[entity_idx] = chunk_idx;
                entity_to_local[entity_idx] = local_idx;
            }
        }
    }
}

void EntityManager::reorderEntitiesByType() {
    // This is a complex operation that would reorder entities
    // to improve cache locality for entities of the same type

    // For now, just optimize each chunk
    optimizeMemoryLayout();
}

void EntityManager::reorderEntitiesByHierarchy() {
    // This is a complex operation that would reorder entities
    // to improve cache locality for hierarchy traversal

    // For now, just optimize each chunk
    optimizeMemoryLayout();
}

template<typename Func>
void EntityManager::processByType(int type_id, Func process_func, int batch_size) {
    // Process all entities of the given type
    for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
        EntityChunk* chunk = chunks[chunk_idx];
        if (!chunk || chunk->type_id != type_id) continue;

        // Process in batches
        for (int start_idx = 0; start_idx < chunk->count; start_idx += batch_size) {
            int end_idx = std::min(start_idx + batch_size, chunk->count);

            // Prefetch next batch
            if (end_idx < chunk->count) {
                prefetchChunkData(chunk, end_idx, std::min(batch_size, chunk->count - end_idx));
            }

            // Process this batch
            for (int local_idx = start_idx; local_idx < end_idx; ++local_idx) {
                if (!chunk->hot.active[local_idx]) continue;

                int entity_idx = chunk->entity_id[local_idx];
                process_func(entity_idx);
            }
        }
    }
}

// Engine implementation
Engine::Engine(int window_width, int window_height, float world_size_x, float world_size_y, float world_size_z, float cell_size, int total_entities)
    : entities(total_entities),
    grid3d(static_cast<int>(world_size_x / cell_size),
        static_cast<int>(world_size_y / cell_size),
        static_cast<int>(world_size_z / cell_size),
        cell_size),
    lod_system(MAX_ENTITY_TYPES),
    task_system(std::thread::hardware_concurrency()),
    gpu_resources(1024, 256, 1024, 256, 64) {

    // Set up world bounds
    world_min = glm::vec3(0.0f);
    world_max = glm::vec3(world_size_x, world_size_y, world_size_z);
    grid_cell_size = cell_size;

    // Set up thread count
    thread_count = std::thread::hardware_concurrency();

    // Debug settings
    debug_mode = false;
    profile_memory = false;

    // Setup SDL
    SDL_Init(SDL_INIT_VIDEO);

    // Create window
    window = SDL_CreateWindow("Engine",
        window_width, window_height,SDL_WINDOW_BORDERLESS);

    // Create renderer
    renderer = SDL_CreateRenderer(window, 0);

    // Initialize spatial partitioning
    morton_grid = new MortonOrderedGrid(MORTON_GRID_SIZE, 512);
    morton_system = new MortonSystem(total_entities);

    // Initialize rendering
    batch_count3d = 0;
    batches3d = new RenderBatch3D[1024]; // Arbitrary limit
    gpuRenderer = new GPUDrivenRenderer(total_entities);

    // Initialize asset manager
    asset_manager = new AssetManager();

    // Initialize work buffers
    entity_buffer_size = 10000;
    entity_buffer = new int[entity_buffer_size];

    // Initialize transform batches
    transform_batch_count = (total_entities + SIMD_WIDTH - 1) / SIMD_WIDTH;
    transform_batches = new TransformBatch[transform_batch_count];

    // Initialize timing
    last_frame_time = SDL_GetTicks();
    delta_time = 0.0f;
    fps = 0.0f;
}

Engine::~Engine() {
    // Clean up work buffers
    delete[] entity_buffer;

    // Clean up transform batches
    delete[] transform_batches;

    // Clean up asset manager
    delete asset_manager;

    // Clean up rendering
    delete[] batches3d;
    delete gpuRenderer;

    // Clean up spatial partitioning
    delete morton_grid;
    delete morton_system;

    // Clean up SDL
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void Engine::update() {
    // Update timing
    uint64_t current_time = SDL_GetTicks();
    delta_time = (current_time - last_frame_time) / 1000.0f;
    last_frame_time = current_time;

    // Update FPS (simple moving average)
    fps = fps * 0.95f + (1.0f / delta_time) * 0.05f;

    // Update entities
    updateByHierarchyDepth();

    // Update collisions
    updateCollisionsBatched();
}

void Engine::render() {
    // Clear screen
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    // Render 3D scene
    gpuRenderer->UpdateEntityTransforms(entities.chunks, entities.chunk_count);
    gpuRenderer->CullWithCompute(camera3d);
    gpuRenderer->RenderVisible(batches3d, batch_count3d);

    // Present
    SDL_RenderPresent(renderer);
}



void Engine::updateEntityChunksBatched(EntityChunk** chunks, int chunk_count) {
    // Process chunks in parallel
    for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
        EntityChunk* chunk = chunks[chunk_idx];
        if (!chunk) continue;

        // Create task for this chunk
        auto task = task_system.CreateTask([chunk, this]() {
            // Process in batches
            for (int start_idx = 0; start_idx < chunk->count; start_idx += ENTITY_BATCH_SIZE) {
                chunk->processBatch(start_idx, ENTITY_BATCH_SIZE, delta_time);
            }
            });

        task_system.ScheduleTask(task);
    }

    // Wait for all tasks to complete
    task_system.WaitAll();
}


void Engine::updateCollisionsBatched() {
    // Detect collisions in parallel using spatial grid
    collision_system.detectCollisionsParallel(&grid3d, entities.chunks, entities.chunk_count, task_system);
}

int Engine::    (int type_id, const glm::vec3& position, const glm::vec3& scale, int mesh_id, int material_id) {
    // Create entity
    int entity_id = entities.createEntity(type_id);
    if (entity_id < 0) {
        return -1;
    }

    // Set position and scale
    setEntityPosition(entity_id, position);
    setEntityScale(entity_id, scale);

    // Set mesh and material (would be handled by type-specific data in a full implementation)

    return entity_id;
}




void Engine::setEntityRotation(int entity_id, const glm::quat& rotation) {
    // Get chunk and local index
    int chunk_idx = entities.entity_to_chunk[entity_id];
    int local_idx = entities.entity_to_local[entity_id];

    if (chunk_idx < 0 || local_idx < 0) {
        return;
    }

    EntityChunk* chunk = entities.chunks[chunk_idx];
    chunk->hot.rotation[local_idx] = rotation;
}

void Engine::setEntityScale(int entity_id, const glm::vec3& scale) {
    // Get chunk and local index
    int chunk_idx = entities.entity_to_chunk[entity_id];
    int local_idx = entities.entity_to_local[entity_id];

    if (chunk_idx < 0 || local_idx < 0) {
        return;
    }

    EntityChunk* chunk = entities.chunks[chunk_idx];
    chunk->hot.scale[local_idx] = scale;

    // Update spatial grid with new bounds
    AABB bounds;
    glm::vec3 position = chunk->hot.position[local_idx];
    glm::vec3 half_scale = scale * 0.5f;
    bounds.min = position - half_scale;
    bounds.max = position + half_scale;

    // Update in spatial grid
    int cell_idx = grid3d.entity_to_cell_map[entity_id];
    grid3d.updateEntity(entity_id, cell_idx, bounds);

    // Update in Morton grid
    float radius = glm::length(half_scale);
    morton_system->updateEntity(entity_id, position, radius);
}


void Engine::setEntityActive(int entity_id, bool active) {
    entities.setEntityActive(entity_id, active);
}

void Engine::setEntityMesh(int entity_id, int mesh_id) {
    // This would be handled by type-specific data in a full implementation
}

void Engine::setEntityMaterial(int entity_id, int material_id) {
    // This would be handled by type-specific data in a full implementation
}

void Engine::setParent(int entity_id, int parent_id) {
    entities.setParent(entity_id, parent_id);
}

int Engine::getParent(int entity_id) {
    // Get chunk and local index
    int chunk_idx = entities.entity_to_chunk[entity_id];
    int local_idx = entities.entity_to_local[entity_id];

    if (chunk_idx < 0 || local_idx < 0) {
        return -1;
    }

    EntityChunk* chunk = entities.chunks[chunk_idx];
    return chunk->hierarchy.parent_id[local_idx];
}

void Engine::setEntityLocalPosition(int entity_id, const glm::vec3& position) {
    entities.setEntityLocalPosition(entity_id, position);
}

int Engine::queryBox(const AABB& box, int* result_buffer, int max_results) {
    return grid3d.queryBox(box, result_buffer, max_results);
}

int Engine::queryFrustum(const Camera3D& camera, int* result_buffer, int max_results) {
    // Extract frustum planes
    glm::vec4 planes[6];
    camera.extractFrustumPlanes(planes);

    // Query all entities
    int result_count = 0;

    for (int chunk_idx = 0; chunk_idx < entities.chunk_count; ++chunk_idx) {
        EntityChunk* chunk = entities.chunks[chunk_idx];
        if (!chunk) continue;

        for (int local_idx = 0; local_idx < chunk->count; ++local_idx) {
            if (!chunk->hot.active[local_idx]) continue;

            // Calculate AABB for entity
            glm::vec3 position = chunk->hot.position[local_idx];
            glm::vec3 half_scale = chunk->hot.scale[local_idx] * 0.5f;
            AABB bounds;
            bounds.min = position - half_scale;
            bounds.max = position + half_scale;

            // Check if AABB is inside frustum
            bool inside = true;
            for (int i = 0; i < 6; ++i) {
                // Find the positive vertex (P-vertex) relative to the plane normal
                glm::vec3 p_vertex;
                p_vertex.x = planes[i].x > 0 ? bounds.max.x : bounds.min.x;
                p_vertex.y = planes[i].y > 0 ? bounds.max.y : bounds.min.y;
                p_vertex.z = planes[i].z > 0 ? bounds.max.z : bounds.min.z;

                // If the positive vertex is outside, the whole AABB is outside
                if (planes[i].x * p_vertex.x +
                    planes[i].y * p_vertex.y +
                    planes[i].z * p_vertex.z +
                    planes[i].w < 0) {
                    inside = false;
                    break;
                }
            }

            if (inside && result_count < max_results) {
                result_buffer[result_count++] = chunk->entity_id[local_idx];
            }
        }
    }

    return result_count;
}

int Engine::queryMortonRegion(const glm::vec3& min, const glm::vec3& max, int* result_buffer, int max_results) {
    // Convert to uint32_t array
    uint32_t* uint_results = reinterpret_cast<uint32_t*>(result_buffer);

    // Query Morton system
    return static_cast<int>(morton_system->queryRange(min, max, uint_results, max_results));
}

void Engine::setCameraPosition(const glm::vec3& position) {
    camera3d.position = position;
}

void Engine::setCameraTarget(const glm::vec3& target) {
    camera3d.target = target;
}

void Engine::setCameraUp(const glm::vec3& up) {
    camera3d.up = up;
}

void Engine::setCameraFov(float fov_degrees) {
    camera3d.fov = glm::radians(fov_degrees);
}

int Engine::addTexture(SDL_Renderer * renderer,SDL_Surface* surface) {
    return gpu_resources.createTexture(renderer, surface);
}

int Engine::addMesh(const float* vertices, int vertex_count, const int* indices, int index_count) {
    return gpu_resources.createMesh(vertices, vertex_count, indices, index_count);
}

int Engine::addMaterial(int diffuse_texture, int normal_texture, int specular_texture, const glm::vec4& diffuse_color, const glm::vec4& specular_color, float shininess) {
    return gpu_resources.createMaterial(diffuse_texture, normal_texture, specular_texture, diffuse_color, specular_color, shininess);
}

int Engine::addShader(const char* vertex_source, const char* fragment_source) {
    return gpu_resources.createShader(vertex_source, fragment_source);
}

void Engine::registerPhysicsLayer(int layer_id, uint32_t collides_with) {
    collision_system.registerLayer(layer_id, collides_with);
}

void Engine::setEntityPhysicsLayer(int entity_id, int layer_id) {
    // Get chunk and local index
    int chunk_idx = entities.entity_to_chunk[entity_id];
    int local_idx = entities.entity_to_local[entity_id];

    if (chunk_idx < 0 || local_idx < 0) {
        return;
    }

    EntityChunk* chunk = entities.chunks[chunk_idx];
    chunk->hot.physics_layer[local_idx] = layer_id;
}

void Engine::setEntityVelocity(int entity_id, const glm::vec3& velocity) {
    // Get chunk and local index
    int chunk_idx = entities.entity_to_chunk[entity_id];
    int local_idx = entities.entity_to_local[entity_id];

    if (chunk_idx < 0 || local_idx < 0) {
        return;
    }

    EntityChunk* chunk = entities.chunks[chunk_idx];
    chunk->hot.velocity[local_idx] = velocity;
}

int Engine::queryCollisions(int entity_id, int* result_buffer, int max_results) {
    // This would use collision_system to query collisions for the entity
    return 0;
}

void Engine::registerEntityLOD(int type_id, const LODSystem::LODLevel* levels, int level_count) {
    lod_system.registerEntityTypeLOD(type_id, levels, level_count);
}

int Engine::queueTextureLoad(const char* filename) {
    return asset_manager->QueueTextureLoad(filename);
}

int Engine::queueMeshLoad(const char* filename) {
    return asset_manager->QueueMeshLoad(filename);
}

int Engine::queueShaderLoad(const char* vs_filename, const char* fs_filename) {
    return asset_manager->QueueShaderLoad(vs_filename, fs_filename);
}

bool Engine::areAllAssetsLoaded() {
    return asset_manager->AreAllAssetsLoaded();
}

void Engine::waitForAssetLoading() {
    asset_manager->WaitForAll();
}

void Engine::setDebugMode(bool enabled) {
    debug_mode = enabled;
}

float Engine::getFrameTime() const {
    return delta_time;
}

float Engine::getFps() const {
    return fps;
}

void Engine::getMemoryStats(size_t* total_allocated, size_t* temp_allocated) const {
    // This would report memory usage statistics
    *total_allocated = 0;
    *temp_allocated = 0;
}

void Engine::optimizeMemoryLayout() {
    entities.optimizeMemoryLayout();
}

void Engine::collectPerformanceStats() {
    // This would collect performance metrics
}


// Let's correct the undefined references in Engine::setEntityPosition
void Engine::setEntityPosition(int entity_id, const glm::vec3& position) {
    // Set position in entity manager
    entities.setEntityPosition(entity_id, position);

    // Update spatial grid
    int chunk_idx = entities.entity_to_chunk[entity_id];
    int local_idx = entities.entity_to_local[entity_id];

    if (chunk_idx >= 0 && local_idx >= 0) {
        EntityChunk* chunk = entities.chunks[chunk_idx];
        AABB bounds;
        // Calculate bounds based on position and scale
        glm::vec3 half_scale = chunk->hot.scale[local_idx] * 0.5f;
        bounds.min = position - half_scale;
        bounds.max = position + half_scale;

        // Update in spatial grid
        int cell_idx = grid3d.entity_to_cell_map[entity_id];
        grid3d.updateEntity(entity_id, cell_idx, bounds);

        // Update in Morton grid
        float radius = glm::length(half_scale);
        morton_system->updateEntity(entity_id, position, radius);
    }
}

// Fix Engine::updateByHierarchyDepth to properly reference hierarchy_stream
void Engine::updateByHierarchyDepth() {
    // Sort entities by hierarchy depth
    entities.sortByHierarchyDepth();

    // Optimize partitioning for parallel processing
    hierarchy_task_partitioner.Initialize(entities.hierarchy_stream->max_depth);

    // Create depth batches
    for (int depth = 0; depth <= entities.hierarchy_stream->max_depth; ++depth) {
        int start_idx = entities.hierarchy_stream->depth_ranges[depth][0];
        int end_idx = entities.hierarchy_stream->depth_ranges[depth][1];

        if (start_idx >= 0 && end_idx >= 0) {
            hierarchy_task_partitioner.depth_batches.push_back({
                depth, start_idx, end_idx - start_idx + 1
                });
        }
    }

    // Optimize partitioning
    hierarchy_task_partitioner.OptimizePartitioning(ENTITY_BATCH_SIZE);

    // Schedule hierarchical tasks
    task_system.ScheduleHierarchicalBatches(
        hierarchy_task_partitioner,
        [this](int depth, int start, int count) {
            // Process entities at this depth
            for (int i = 0; i < count; ++i) {
                int entity_idx = start + i; // This would be the entity index in a real implementation

                // Get chunk and local index
                int chunk_idx = entities.entity_to_chunk[entity_idx];
                int local_idx = entities.entity_to_local[entity_idx];

                if (chunk_idx < 0 || local_idx < 0) continue;

                EntityChunk* chunk = entities.chunks[chunk_idx];
                if (!chunk->hot.active[local_idx]) continue;

                // Update position based on velocity
                chunk->hot.position[local_idx] += chunk->hot.velocity[local_idx] * delta_time;

                // Update local transform
                glm::mat4 translation = glm::translate(glm::mat4(1.0f), chunk->hot.position[local_idx]);
                glm::mat4 rotation = glm::mat4_cast(chunk->hot.rotation[local_idx]);
                glm::mat4 scale = glm::scale(glm::mat4(1.0f), chunk->hot.scale[local_idx]);

                chunk->hot.local_transform[local_idx] = translation * rotation * scale;

                // Update world transform if at root level
                if (depth == 0) {
                    chunk->hot.world_transform[local_idx] = chunk->hot.local_transform[local_idx];
                }
                // For non-root entities, world transform will be updated after parent transform is ready
            }
        }
    );

    // Wait for all tasks to complete
    task_system.WaitAll();

    // Update world transforms
    updateTransformsHierarchical();
}

// Fix updateTransformsHierarchical to properly reference hierarchy_stream
void Engine::updateTransformsHierarchical() {
    // Process entities by depth
    for (int depth = 0; depth <= entities.hierarchy_stream->max_depth; ++depth) {
        int start_idx = entities.hierarchy_stream->depth_ranges[depth][0];
        int end_idx = entities.hierarchy_stream->depth_ranges[depth][1];

        if (start_idx < 0 || end_idx < 0) continue;

        // Create task for this depth level
        auto task = task_system.CreateTask([this, depth, start_idx, end_idx]() {
            // Process entities at this depth
            for (int i = start_idx; i <= end_idx; ++i) {
                int entity_idx = i; // This would be the entity index in a real implementation

                // Get chunk and local index
                int chunk_idx = entities.entity_to_chunk[entity_idx];
                int local_idx = entities.entity_to_local[entity_idx];

                if (chunk_idx < 0 || local_idx < 0) continue;

                EntityChunk* chunk = entities.chunks[chunk_idx];

                // Skip if already processed at depth 0
                if (depth == 0) continue;

                // Update world transform based on hierarchy
                int parent_id = chunk->hierarchy.parent_id[local_idx];
                if (parent_id >= 0) {
                    int parent_chunk_idx = entities.entity_to_chunk[parent_id];
                    int parent_local_idx = entities.entity_to_local[parent_id];

                    if (parent_chunk_idx >= 0 && parent_local_idx >= 0) {
                        EntityChunk* parent_chunk = entities.chunks[parent_chunk_idx];

                        chunk->hot.world_transform[local_idx] =
                            parent_chunk->hot.world_transform[parent_local_idx] *
                            chunk->hot.local_transform[local_idx];
                    }
                }
            }
            });

        task_system.ScheduleTask(task);

        // Wait for this depth level to complete before proceeding to the next
        task_system.WaitAll();
    }
}

// Let's add the missing WorkStealingTaskSystem implementation
WorkStealingTaskSystem::Task::Task(std::function<void()> f)
    : func(f), dependencies(0) {
}

WorkStealingTaskSystem::WorkStealingTaskSystem(size_t thread_count) {
    if (thread_count == 0) {
        thread_count = std::thread::hardware_concurrency();
    }

    // Reserve space for the pointers
    thread_queues.reserve(thread_count);

    // Create each queue with make_unique
    for (size_t i = 0; i < thread_count; ++i) {
        thread_queues.push_back(std::make_unique<ThreadLocalQueue>());
    }

    // Rest of your initialization...
    active_thread_count.store(0, std::memory_order_relaxed);
    running.store(true, std::memory_order_relaxed);
    active_tasks.store(0, std::memory_order_relaxed);

    // Create worker threads
    workers.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        workers.emplace_back(&WorkStealingTaskSystem::worker_main, this, i);
    }
}

WorkStealingTaskSystem::~WorkStealingTaskSystem() {
    // Signal threads to exit
    running.store(false, std::memory_order_release);

    // Notify all waiting threads
    task_cv.notify_all();

    // Wait for threads to finish
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void WorkStealingTaskSystem::scheduleTask(Task* task) {
    // If no dependencies, add to work queue
    if (task->dependencies.load(std::memory_order_relaxed) == 0) {
        std::lock_guard<std::mutex> lock(task_mutex);
        active_tasks.fetch_add(1, std::memory_order_relaxed);

        // Add to a random thread queue
        int thread_id = rand() % thread_queues.size();
        std::lock_guard<std::mutex> queue_lock(thread_queues[thread_id]->queue_mutex);
        thread_queues[thread_id]->local_tasks.push_back(task);

        // Notify waiting threads
        task_cv.notify_one();
    }
}

void WorkStealingTaskSystem::scheduleTaskForThread(Task* task, int thread_id) {
    if (task->dependencies.load(std::memory_order_relaxed) == 0) {
        std::lock_guard<std::mutex> lock(task_mutex);
        active_tasks.fetch_add(1, std::memory_order_relaxed);

        // Add to specified thread queue
        if (thread_id >= 0 && thread_id < static_cast<int>(thread_queues.size())) {
            std::lock_guard<std::mutex> queue_lock(thread_queues[thread_id]->queue_mutex);
            thread_queues[thread_id]->local_tasks.push_back(task);
        }
        else {
            // If invalid thread ID, add to a random thread queue
            int random_thread = rand() % thread_queues.size();
            std::lock_guard<std::mutex> queue_lock(thread_queues[random_thread]->queue_mutex);
            thread_queues[random_thread]->local_tasks.push_back(task);
        }

        // Notify waiting threads
        task_cv.notify_one();
    }
}

WorkStealingTaskSystem::Task* WorkStealingTaskSystem::stealTask(int thief_thread_id) {
    // Try to steal from other thread queues
    for (size_t victim_id = 0; victim_id < thread_queues.size(); ++victim_id) {
        if (victim_id != thief_thread_id) {
            ThreadLocalQueue& victim_queue = *thread_queues[victim_id];
            std::lock_guard<std::mutex> lock(victim_queue.queue_mutex);

            if (!victim_queue.local_tasks.empty()) {
                Task* stolen_task = victim_queue.local_tasks.back();
                victim_queue.local_tasks.pop_back();
                return stolen_task;
            }
        }
    }

    return nullptr;
}

void WorkStealingTaskSystem::waitAll() {
    // Process tasks until all are done
    while (active_tasks.load(std::memory_order_acquire) > 0) {
        // Try to find and execute a task
        Task* task = nullptr;

        // Try to get a task from any thread queue
        for (size_t i = 0; i < thread_queues.size(); ++i) {
            ThreadLocalQueue& queue = *thread_queues[i];
            std::lock_guard<std::mutex> lock(queue.queue_mutex);

            if (!queue.local_tasks.empty()) {
                task = queue.local_tasks.back();
                queue.local_tasks.pop_back();
                break;
            }
        }

        if (task) {
            // Execute the task
            task->func();

            // Process dependents
            for (Task* dependent : task->dependents) {
                if (dependent->dependencies.fetch_sub(1, std::memory_order_release) == 1) {
                    scheduleTask(dependent);
                }
            }

            active_tasks.fetch_sub(1, std::memory_order_release);
        }
        else {
            // No task found, wait for notification
            std::unique_lock<std::mutex> lock(task_mutex);
            if (active_tasks.load(std::memory_order_acquire) > 0) {
                task_cv.wait_for(lock, std::chrono::milliseconds(1));
            }
        }
    }
}

void WorkStealingTaskSystem::worker_main(int thread_id) {
    active_thread_count.fetch_add(1, std::memory_order_release);

    ThreadLocalQueue& local_queue = *thread_queues[thread_id];

    while (running.load(std::memory_order_acquire)) {
        Task* task = nullptr;

        // Try to get a task from local queue
        {
            std::lock_guard<std::mutex> lock(local_queue.queue_mutex);
            if (!local_queue.local_tasks.empty()) {
                task = local_queue.local_tasks.back();
                local_queue.local_tasks.pop_back();
            }
        }

        // If no local task, try to steal
        if (!task) {
            task = stealTask(thread_id);
        }

        if (task) {
            // Execute the task
            task->func();

            // Process dependents
            for (Task* dependent : task->dependents) {
                if (dependent->dependencies.fetch_sub(1, std::memory_order_release) == 1) {
                    scheduleTask(dependent);
                }
            }

            active_tasks.fetch_sub(1, std::memory_order_release);
        }
        else {
            // No task found, wait for notification
            std::unique_lock<std::mutex> lock(task_mutex);
            if (running.load(std::memory_order_acquire) && active_tasks.load(std::memory_order_acquire) == 0) {
                task_cv.wait_for(lock, std::chrono::milliseconds(10));
            }
            else if (running.load(std::memory_order_acquire)) {
                task_cv.wait_for(lock, std::chrono::milliseconds(1));
            }
        }
    }

    active_thread_count.fetch_sub(1, std::memory_order_release);
}

// Implementation of FiberTaskSystem
FiberTaskSystem::FiberTaskSystem(size_t fiber_count) : running(true), main_fiber(nullptr) {
    // Fiber implementation is platform-specific
    // This is a simplified implementation

    // Allocate fibers
    fiber_pool.resize(fiber_count, nullptr);

    // Create main fiber (would be platform-specific)
    // main_fiber = CreateFiber(...);
}

FiberTaskSystem::~FiberTaskSystem() {
    running = false;

    // Delete fibers (would be platform-specific)
    // for (void* fiber : fiber_pool) {
    //     DeleteFiber(fiber);
    // }

    // Delete main fiber
    // DeleteFiber(main_fiber);
}

FiberTaskSystem::FiberTask* FiberTaskSystem::createFiberTask(std::function<void()> func) {
    FiberTask* task = new FiberTask();
    task->func = func;
    task->completed = false;
    task->parent = nullptr;
    // task->fiber would be initialized when scheduled

    return task;
}

void FiberTaskSystem::scheduleFiberTask(FiberTask* task) {
    std::lock_guard<std::mutex> lock(task_mutex);
    ready_tasks.push_back(task);
}

void FiberTaskSystem::yieldToScheduler() {
    // This would switch to the scheduler fiber
    // SwitchToFiber(main_fiber);
}

void FiberTaskSystem::waitForTask(FiberTask* task) {
    // Record current task as parent
    FiberTask* current_task = nullptr; // Would be retrieved from TLS
    if (current_task) {
        task->parent = current_task;
    }

    // Wait until task is completed
    while (!task->completed) {
        yieldToScheduler();
    }
}

void FiberTaskSystem::runUntilComplete() {
    while (running) {
        // Get next task
        FiberTask* next_task = nullptr;

        {
            std::lock_guard<std::mutex> lock(task_mutex);
            if (!ready_tasks.empty()) {
                next_task = ready_tasks.front();
                ready_tasks.erase(ready_tasks.begin());
            }
        }

        if (next_task) {
            // Execute task
            next_task->func();

            // Mark as completed
            next_task->completed = true;

            // Resume parent if any
            if (next_task->parent) {
                scheduleFiberTask(next_task->parent);
            }
        }
        else {
            // No tasks, check if we're done
            std::lock_guard<std::mutex> lock(task_mutex);
            if (ready_tasks.empty()) {
                break;
            }
        }
    }
}

// Let's implement the AssetManager
AssetManager::AssetManager(int numThreads) : running(true) {
    // Start IO threads
    for (int i = 0; i < numThreads; ++i) {
        ioThreads.emplace_back(&AssetManager::IOThreadMain, this);
    }
}

AssetManager::~AssetManager() {
    // Signal threads to exit
    running.store(false, std::memory_order_release);

    // Wake up threads waiting for tasks
    loadQueue.cv.notify_all();

    // Wait for threads to finish
    for (auto& thread : ioThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void AssetManager::ConcurrentQueue::Enqueue(LoadTask task) {
    std::lock_guard<std::mutex> lock(mutex);
    tasks.push_back(task);
    cv.notify_one();
}

bool AssetManager::ConcurrentQueue::TryDequeue(LoadTask& result) {
    std::lock_guard<std::mutex> lock(mutex);
    if (tasks.empty()) {
        return false;
    }

    result = tasks.front();
    tasks.pop_front();
    return true;
}

bool AssetManager::ConcurrentQueue::Empty() {
    std::lock_guard<std::mutex> lock(mutex);
    return tasks.empty();
}

int AssetManager::QueueTextureLoad(const std::string& filename) {
    // Check if already loaded or queued
    {
        std::lock_guard<std::mutex> lock(assetsMutex);
        auto it = loadedAssets.find(filename);
        if (it != loadedAssets.end()) {
            return it->second;
        }
    }

    // Generate asset ID
    int assetId = static_cast<int>(loadedAssets.size());

    // Add to loaded assets
    {
        std::lock_guard<std::mutex> lock(assetsMutex);
        loadedAssets[filename] = assetId;
    }

    // Create load task
    LoadTask task;
    task.name = filename;
    task.assetId = assetId;

    // Load function
    task.Load = [this, filename]() {
        // Load texture from file
        // This would use SDL_LoadBMP, IMG_Load, etc.
        // For now, just simulate loading

        // Simulate loading time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        };

    // Upload function
    task.UploadToGPU = [this, filename, assetId]() {
        // Upload texture to GPU
        // This would use SDL_CreateTextureFromSurface
        // For now, just simulate uploading

        // Simulate upload time
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        };

    // Queue task
    loadQueue.Enqueue(task);

    return assetId;
}

int AssetManager::QueueMeshLoad(const std::string& filename) {
    // Similar implementation to QueueTextureLoad
    // Check if already loaded or queued
    {
        std::lock_guard<std::mutex> lock(assetsMutex);
        auto it = loadedAssets.find(filename);
        if (it != loadedAssets.end()) {
            return it->second;
        }
    }

    // Generate asset ID
    int assetId = static_cast<int>(loadedAssets.size());

    // Add to loaded assets
    {
        std::lock_guard<std::mutex> lock(assetsMutex);
        loadedAssets[filename] = assetId;
    }

    // Create load task
    LoadTask task;
    task.name = filename;
    task.assetId = assetId;

    // Load function
    task.Load = [this, filename]() {
        // Load mesh from file
        // This would parse OBJ, FBX, etc.
        // For now, just simulate loading

        // Simulate loading time
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        };

    // Upload function
    task.UploadToGPU = [this, filename, assetId]() {
        // Upload mesh to GPU
        // This would create VAO, VBO, IBO
        // For now, just simulate uploading

        // Simulate upload time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        };

    // Queue task
    loadQueue.Enqueue(task);

    return assetId;
}

int AssetManager::QueueShaderLoad(const std::string& vsFilename, const std::string& fsFilename) {
    // Check if already loaded or queued
    std::string combined = vsFilename + "+" + fsFilename;

    {
        std::lock_guard<std::mutex> lock(assetsMutex);
        auto it = loadedAssets.find(combined);
        if (it != loadedAssets.end()) {
            return it->second;
        }
    }

    // Generate asset ID
    int assetId = static_cast<int>(loadedAssets.size());

    // Add to loaded assets
    {
        std::lock_guard<std::mutex> lock(assetsMutex);
        loadedAssets[combined] = assetId;
    }

    // Create load task
    LoadTask task;
    task.name = combined;
    task.assetId = assetId;

    // Load function
    task.Load = [this, vsFilename, fsFilename]() {
        // Load shader source files
        // This would read from files
        // For now, just simulate loading

        // Simulate loading time
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        };

    // Compile and link function
    task.UploadToGPU = [this, combined, assetId]() {
        // Compile and link shaders
        // This would use glCreateShader, glShaderSource, etc.
        // For now, just simulate compilation

        // Simulate compilation time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        };

    // Queue task
    loadQueue.Enqueue(task);

    return assetId;
}

bool AssetManager::IsAssetLoaded(const std::string& name, int* assetId) {
    std::lock_guard<std::mutex> lock(assetsMutex);
    auto it = loadedAssets.find(name);
    if (it != loadedAssets.end()) {
        if (assetId) {
            *assetId = it->second;
        }
        return true;
    }
    return false;
}

bool AssetManager::AreAllAssetsLoaded() {
    return loadQueue.Empty();
}

void AssetManager::WaitForAll() {
    while (!loadQueue.Empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void AssetManager::IOThreadMain() {
    while (running.load(std::memory_order_acquire)) {
        LoadTask task;
        bool hasTask = loadQueue.TryDequeue(task);

        if (hasTask) {
            // Execute load task
            task.Load();

            // Execute upload task
            task.UploadToGPU();
        }
        else {
            // No task, wait for notification
            std::unique_lock<std::mutex> lock(loadQueue.mutex);
            if (running.load(std::memory_order_acquire)) {
                loadQueue.cv.wait_for(lock, std::chrono::milliseconds(10));
            }
        }
    }
}

// Let's implement the EntityLayoutOptimizer
EntityLayoutOptimizer::EntityLayoutOptimizer() {
    // No initialization needed
}

EntityLayoutOptimizer::~EntityLayoutOptimizer() {
    // No cleanup needed
}

void EntityLayoutOptimizer::recordAccess(const std::vector<int>& entities_accessed_together) {
    // Record entities accessed together
    for (int entity_id : entities_accessed_together) {
        // Find existing access pattern or create new one
        auto it = std::find_if(access_patterns.begin(), access_patterns.end(),
            [entity_id](const AccessPattern& pattern) {
                return pattern.entity_id == entity_id;
            });

        if (it != access_patterns.end()) {
            // Update existing pattern
            it->access_count++;

            // Add other entities to accessed_with
            for (int other_id : entities_accessed_together) {
                if (other_id != entity_id) {
                    // Check if already in accessed_with
                    auto other_it = std::find(it->accessed_with.begin(), it->accessed_with.end(), other_id);
                    if (other_it == it->accessed_with.end()) {
                        it->accessed_with.push_back(other_id);
                    }
                }
            }
        }
        else {
            // Create new pattern
            AccessPattern pattern;
            pattern.entity_id = entity_id;
            pattern.access_count = 1;

            // Add other entities to accessed_with
            for (int other_id : entities_accessed_together) {
                if (other_id != entity_id) {
                    pattern.accessed_with.push_back(other_id);
                }
            }

            access_patterns.push_back(pattern);
        }
    }
}

void EntityLayoutOptimizer::analyzePatterns() {
    // Sort patterns by access count
    std::sort(access_patterns.begin(), access_patterns.end(),
        [](const AccessPattern& a, const AccessPattern& b) {
            return a.access_count > b.access_count;
        });

    // Create optimized order
    optimized_order.clear();
    std::vector<bool> processed(access_patterns.size(), false);

    // Start with the most frequently accessed entity
    if (!access_patterns.empty()) {
        int current_entity = access_patterns[0].entity_id;
        optimized_order.push_back(current_entity);

        // Mark as processed
        for (size_t i = 0; i < access_patterns.size(); ++i) {
            if (access_patterns[i].entity_id == current_entity) {
                processed[i] = true;
                break;
            }
        }

        // Process remaining entities
        while (optimized_order.size() < access_patterns.size()) {
            int next_entity = -1;
            int max_score = 0;

            // Find entity with highest score
            for (size_t i = 0; i < access_patterns.size(); ++i) {
                if (processed[i]) continue;

                const AccessPattern& pattern = access_patterns[i];
                int score = 0;

                // Score based on how many entities in optimized_order are accessed together
                for (int related_id : pattern.accessed_with) {
                    if (std::find(optimized_order.begin(), optimized_order.end(), related_id) != optimized_order.end()) {
                        score++;
                    }
                }

                // Weight by access count
                score *= pattern.access_count;

                if (score > max_score) {
                    max_score = score;
                    next_entity = pattern.entity_id;
                }
            }

            if (next_entity == -1) {
                // No related entities found, pick the next most accessed entity
                for (size_t i = 0; i < access_patterns.size(); ++i) {
                    if (!processed[i]) {
                        next_entity = access_patterns[i].entity_id;
                        processed[i] = true;
                        break;
                    }
                }
            }
            else {
                // Mark as processed
                for (size_t i = 0; i < access_patterns.size(); ++i) {
                    if (access_patterns[i].entity_id == next_entity) {
                        processed[i] = true;
                        break;
                    }
                }
            }

            if (next_entity != -1) {
                optimized_order.push_back(next_entity);
            }
            else {
                break; // No more entities to process
            }
        }
    }
}

void EntityLayoutOptimizer::optimizeLayout(EntityChunk** chunks, int chunk_count) {
    // Only optimize if we have an optimized order
    if (optimized_order.empty()) {
        return;
    }

    // For each chunk
    for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
        EntityChunk* chunk = chunks[chunk_idx];
        if (!chunk) continue;

        // Map entity IDs to indices in this chunk
        std::unordered_map<int, int> entity_to_idx;
        for (int i = 0; i < chunk->count; ++i) {
            entity_to_idx[chunk->entity_id[i]] = i;
        }

        // Create new order for this chunk
        std::vector<int> chunk_order;

        // First add entities from optimized_order that are in this chunk
        for (int entity_id : optimized_order) {
            auto it = entity_to_idx.find(entity_id);
            if (it != entity_to_idx.end()) {
                chunk_order.push_back(it->second);
                entity_to_idx.erase(it);
            }
        }

        // Then add remaining entities
        for (const auto& pair : entity_to_idx) {
            chunk_order.push_back(pair.second);
        }

        // Reorder data in chunk
        // This would be a complex operation that swaps all data
        // For simplicity, we'll just reorder entity IDs

        // Allocate temporary buffers
        int* temp_entity_ids = new int[chunk->count];

        // Copy entity IDs in new order
        for (int i = 0; i < chunk->count; ++i) {
            if (i < static_cast<int>(chunk_order.size())) {
                temp_entity_ids[i] = chunk->entity_id[chunk_order[i]];
            }
            else {
                temp_entity_ids[i] = chunk->entity_id[i];
            }
        }

        // Swap buffers
        std::swap(chunk->entity_id, temp_entity_ids);

        // Cleanup
        delete[] temp_entity_ids;
    }
}

const std::vector<int>& EntityLayoutOptimizer::getOptimizedOrder() const {
    return optimized_order;
}

void EntityLayoutOptimizer::applyToSpatialGrid(SpatialGrid3D* grid) {
    // This would reorder entities in the spatial grid
    // For now, just rebuild the grid

    // Iterate through cells
    for (int i = 0; i < grid->cell_count; ++i) {
        SpatialGrid3D::Cell& cell = grid->cells[i];

        if (cell.count > 0 && cell.entity_indices) {
            // Sort entity indices based on optimized order
            std::sort(cell.entity_indices, cell.entity_indices + cell.count,
                [this](int a, int b) {
                    // Find positions in optimized order
                    auto pos_a = std::find(optimized_order.begin(), optimized_order.end(), a);
                    auto pos_b = std::find(optimized_order.begin(), optimized_order.end(), b);

                    if (pos_a != optimized_order.end() && pos_b != optimized_order.end()) {
                        return std::distance(optimized_order.begin(), pos_a) < std::distance(optimized_order.begin(), pos_b);
                    }
                    else if (pos_a != optimized_order.end()) {
                        return true;
                    }
                    else if (pos_b != optimized_order.end()) {
                        return false;
                    }
                    else {
                        return a < b;
                    }
                });
        }
    }
}

void EntityLayoutOptimizer::applyToMortonGrid(MortonOrderedGrid* grid) {
    // This would reorder entities in the Morton grid
    // For now, just rebuild the grid

    // Sort entity indices in each cell
    for (int i = 0; i < grid->cell_count; ++i) {
        MortonOrderedGrid::Cell& cell = grid->cells[i];

        if (cell.count > 0 && cell.entity_indices) {
            // Sort entity indices based on optimized order
            std::sort(cell.entity_indices, cell.entity_indices + cell.count,
                [this](int a, int b) {
                    // Find positions in optimized order
                    auto pos_a = std::find(optimized_order.begin(), optimized_order.end(), a);
                    auto pos_b = std::find(optimized_order.begin(), optimized_order.end(), b);

                    if (pos_a != optimized_order.end() && pos_b != optimized_order.end()) {
                        return std::distance(optimized_order.begin(), pos_a) < std::distance(optimized_order.begin(), pos_b);
                    }
                    else if (pos_a != optimized_order.end()) {
                        return true;
                    }
                    else if (pos_b != optimized_order.end()) {
                        return false;
                    }
                    else {
                        return a < b;
                    }
                });
        }
    }
}

// Let's add some additional implementation for GPUDrivenRenderer
GPUDrivenRenderer::GPUDrivenRenderer(int maxEntities) {
    // This would initialize OpenGL/Direct3D resources
    // For now, just initialize data structures

    // Create buffer for entity transforms
    entitiesBuffer = 0; // This would be a GL buffer ID

    // Create buffer for visible instance indices
    visibleSSBO = 0;

    // Create buffer for indirect draw commands
    indirectDrawBuffer = 0;

    // Create buffer for hierarchy depths
    hierarchyDepthBuffer = 0;

    // Create buffer for depth-sorted indices
    depthSortedIndices = 0;

    // Create compute shader for frustum culling
    frustumCullShader = 0;
}

GPUDrivenRenderer::~GPUDrivenRenderer() {
    // This would clean up OpenGL/Direct3D resources
}

void GPUDrivenRenderer::UpdateEntityTransforms(EntityChunk** chunks, int chunkCount) {
    // This would update the entity transforms buffer with latest transforms
    // For now, just count entities

    int entity_count = 0;
    for (int i = 0; i < chunkCount; ++i) {
        EntityChunk* chunk = chunks[i];
        if (chunk) {
            entity_count += chunk->count;
        }
    }

    // TODO: Upload transforms to GPU buffer
}

void GPUDrivenRenderer::CullWithCompute(const Camera3D& camera) {
    // This would dispatch a compute shader for frustum culling
    // For now, just simulate culling

    // TODO: Set compute shader uniforms
    // TODO: Dispatch compute shader
}

// GPUDrivenRenderer::RenderVisible implementation
void GPUDrivenRenderer::RenderVisible(RenderBatch3D* batches, int batchCount) {
    // This renders all visible entities using the indirect draw buffer

    // In a real implementation, we would:
    // 1. Bind the appropriate shader for GPU-driven rendering
    // 2. Bind the visible indices SSBO
    // 3. For each batch, draw using indirect commands

    for (int i = 0; i < batchCount; ++i) {
        RenderBatch3D* batch = &batches[i];

        // In SDL3 GPU API, we would:
        // 1. Begin a render pass
        // 2. Bind the appropriate graphics pipeline
        // 3. Bind instance data
        // 4. Draw with indirect buffer

        // Example (pseudo-code that would be filled with actual SDL3 GPU API calls):
        // SDL_GPUCommandBuffer* cmdBuffer = SDL_AcquireGPUCommandBuffer();
        // SDL_BeginGPURenderPass(cmdBuffer, ...);
        // SDL_BindGPUGraphicsPipeline(cmdBuffer, ...);
        // SDL_BindGPUVertexBuffers(cmdBuffer, ...);
        // SDL_DrawGPUPrimitivesIndirect(cmdBuffer, ...);
        // SDL_EndGPURenderPass(cmdBuffer);
        // SDL_SubmitGPUCommandBuffer(cmdBuffer);

        // For now, we'll call the batch's own render method
        batch->renderInstanced();
    }
}

void GPUDrivenRenderer::UpdateHierarchyDepths(EntityChunk** chunks, int chunkCount) {
    // This would update the hierarchy depths buffer
    // For now, just count entities

    int entity_count = 0;
    for (int i = 0; i < chunkCount; ++i) {
        EntityChunk* chunk = chunks[i];
        if (chunk) {
            entity_count += chunk->count;
        }
    }

    // TODO: Upload hierarchy depths to GPU buffer
}

void GPUDrivenRenderer::RenderVisibleHierarchical(RenderBatch3D* batches, int batchCount) {
    // This would render visible entities in hierarchical depth order
    // For now, just simulate rendering

    // TODO: Draw batches in hierarchical order
}

// RenderBatch3D.cpp
RenderBatch3D::RenderBatch3D()
    : material_id(0), shader_id(0), count(0), capacity(0),
    entity_indices(nullptr), typed_blocks(nullptr), typed_block_count(0),
    instance_buffer(0) {
    // Initialize your instance data here
}

RenderBatch3D::~RenderBatch3D() {
    delete[] entity_indices;
    delete[] typed_blocks;
    // Clean up any other resources
}

// MortonOrderedGrid.cpp
MortonOrderedGrid::MortonOrderedGrid(int grid_size, float cell_size)
    : cells(nullptr), cell_count(0), cell_size(cell_size),
    morton_codes(nullptr), cell_start_indices(nullptr),
    cell_entity_counts(nullptr), all_entities(nullptr),
    query_buffer(nullptr), query_buffer_size(0) {
    // Allocate memory and initialize the grid
    // Example implementation:
    cell_count = grid_size * grid_size * grid_size;
    cells = new Cell[cell_count]();
    morton_codes = new uint64_t[cell_count]();
    cell_start_indices = new int[cell_count]();
    cell_entity_counts = new int[cell_count]();
    query_buffer_size = 1024;  // Or some suitable size
    query_buffer = new int[query_buffer_size]();
}

MortonOrderedGrid::~MortonOrderedGrid() {
    // Clean up allocated memory
    delete[] cells;
    delete[] morton_codes;
    delete[] cell_start_indices;
    delete[] cell_entity_counts;
    delete[] all_entities;
    delete[] query_buffer;
}

void MortonOrderedGrid::optimizeCellSize() {
    // Implementation to optimize cell size based on entity distribution
    // Example:
    // 1. Analyze entity density
    // 2. Calculate optimal cell size
    // 3. Rebuild grid if necessary
}

// HierarchyDepthContainer.cpp
HierarchyDepthContainer::HierarchyDepthContainer()
    : entities_by_depth(nullptr), counts_by_depth(nullptr), max_depth(0) {
}

HierarchyDepthContainer::~HierarchyDepthContainer() {
    if (entities_by_depth) {
        for (int i = 0; i <= max_depth; i++) {
            delete[] entities_by_depth[i];
        }
        delete[] entities_by_depth;
    }
    delete[] counts_by_depth;
}

// GPUResources.cpp
GPUResources::GPUResources(int max_textures, int max_shaders, int max_meshes, int max_materials, int max_render_targets)
    : textures(nullptr), texture_count(0), texture_capacity(max_textures),
    shaders(nullptr), shader_count(0), shader_capacity(max_shaders),
    meshes(nullptr), mesh_count(0), mesh_capacity(max_meshes),
    materials(nullptr), material_count(0), material_capacity(max_materials),
    render_targets(nullptr), render_target_count(0), render_target_capacity(max_render_targets) {

    // Allocate memory for resources
    textures = new SDL_Texture * [max_textures]();
    shaders = new Shader[max_shaders]();
    meshes = new Mesh[max_meshes]();
    materials = new Material[max_materials]();
    render_targets = new SDL_Texture * [max_render_targets]();
}

GPUResources::~GPUResources() {
    // Clean up allocated resources
    delete[] textures;
    delete[] shaders;
    delete[] meshes;
    delete[] materials;
    delete[] render_targets;
}

int GPUResources::createTexture(SDL_Renderer * renderer, SDL_Surface* surface) {
    if (texture_count >= texture_capacity) {
        return -1;  // Error: out of texture slots
    }

    // Create SDL texture from surface
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    
    if (!texture) {
        return -1;  // Error creating texture
    }

    textures[texture_count] = texture;
    return texture_count++;
}

// Similar implementations for other GPUResources methods...

// EnhancedCollisionSystem.cpp
EnhancedCollisionSystem::EnhancedCollisionSystem()
    : layer_count(0), collision_pairs(nullptr), collision_pair_count(0),
    collision_pairs_capacity(0), collision_simd_data(nullptr) {
    // Initialize collision layers
    for (int i = 0; i < MAX_PHYSICS_LAYERS; i++) {
        layers[i].layer_mask = 0;
        layers[i].collides_with = 0;
        collisionMatrix[i] = 0;
    }

    // Allocate memory for collision pairs
    collision_pairs_capacity = 1024;  // Or some suitable size
    collision_pairs = new CollisionPair[collision_pairs_capacity]();

    // Initialize SIMD data
    collision_simd_data = new CollisionSIMDData[16](); // Some reasonable size
}

EnhancedCollisionSystem::~EnhancedCollisionSystem() {
    delete[] collision_pairs;
    delete[] collision_simd_data;
}
bool EnhancedCollisionSystem::canCollide(int entityIdA, int entityIdB) const {
    // Your collision filtering logic here
    // For example:

    // Prevent self-collision
    if (entityIdA == entityIdB) {
        return false;
    }

    // Add any additional filtering logic based on your game's needs
    // For example, checking collision layers, entity types, etc.

    // By default, allow collisions between different entities
    return true;
}
void EnhancedCollisionSystem::registerLayer(int layer_id, unsigned int collides_with) {
    if (layer_id < 0 || layer_id >= MAX_PHYSICS_LAYERS) {
        return; // Invalid layer
    }

    layers[layer_id].layer_mask = 1u << layer_id;
    layers[layer_id].collides_with = collides_with;

    // Update layer count if needed
    if (layer_id >= layer_count) {
        layer_count = layer_id + 1;
    }

    // Update collision matrix
    precomputeCollisionMatrix();
}

void EnhancedCollisionSystem::mergeCollisionResults(CollisionJob* jobs, int job_count) {
    // First, count total collisions to ensure we have enough space
    int total_collisions = 0;
    for (int i = 0; i < job_count; i++) {
        total_collisions += jobs[i].collision_count;
    }

    // Check if we need to resize our collision pairs buffer
    if (total_collisions > collision_pairs_capacity) {
        // Resize to at least double the required size for future growth
        int new_capacity = std::max(total_collisions * 2, collision_pairs_capacity * 2);

        // Allocate new buffer
        CollisionPair* new_buffer = new CollisionPair[new_capacity];

        // Copy existing data if any
        if (collision_pair_count > 0 && collision_pairs != nullptr) {
            memcpy(new_buffer, collision_pairs, collision_pair_count * sizeof(CollisionPair));
        }

        // Clean up old buffer
        delete[] collision_pairs;

        // Update member variables
        collision_pairs = new_buffer;
        collision_pairs_capacity = new_capacity;
    }

    // Merge all job results into the main collision pairs buffer
    for (int i = 0; i < job_count; i++) {
        // Ensure we don't overflow
        int pairs_to_copy = std::min(jobs[i].collision_count,
            collision_pairs_capacity - collision_pair_count);

        if (pairs_to_copy > 0) {
            // Copy collision pairs from this job's buffer to the main buffer
            memcpy(collision_pairs + collision_pair_count,
                jobs[i].local_collision_buffer,
                pairs_to_copy * sizeof(CollisionPair));

            // Update collision count
            collision_pair_count += pairs_to_copy;
        }

        // If we couldn't copy all pairs, log a warning (in a real system)
        if (pairs_to_copy < jobs[i].collision_count) {
            // Log warning: collision buffer overflow
        }
    }
}

// EnhancedCollisionSystem implementations
void EnhancedCollisionSystem::precomputeCollisionMatrix() {
    // Precompute the collision matrix for efficient layer-based collision detection
    for (int i = 0; i < MAX_PHYSICS_LAYERS; i++) {
        // Initialize collision matrix with zero
        collisionMatrix[i] = 0;

        // For each layer, compute what it can collide with
        if (i < layer_count) {
            for (int j = 0; j < layer_count; j++) {
                // Check if layer i collides with layer j
                if (layers[i].collides_with & layers[j].layer_mask) {
                    // If yes, set the bit in the collision matrix
                    collisionMatrix[i] |= (1 << j);
                }
            }
        }
    }
}

void EnhancedCollisionSystem::detectCollisionsParallel(SpatialGrid3D* grid, EntityChunk** chunks, int chunk_count, LockFreeTaskSystem& tasks) {
    // This would be a complex implementation in practice
    // Here's a simplified version:

    // Validate input parameters
    if (!grid || !grid->cells || grid->cell_count <= 0 || !chunks || chunk_count <= 0) {
        // Handle invalid input - early return or throw exception
        return;
    }

    // Reset collision count
    collision_pair_count = 0;

    // Create collision jobs based on grid cells
    const int cells_per_job = 64; // Process 64 cells per job
    int cell_count = grid->cell_count;
    int job_count = (cell_count + cells_per_job - 1) / cells_per_job;

    if (job_count <= 0) {
        // No jobs to process
        return;
    }

    // Allocate collision jobs - use a shared_ptr for safer memory management
    CollisionJob* jobs_raw = new CollisionJob[job_count];
    if (!jobs_raw) {
        // Memory allocation failed
        return;
    }

    // Use a shared pointer with custom deleter to manage the jobs array
    std::shared_ptr<CollisionJob[]> jobs_shared(jobs_raw, [](CollisionJob* ptr) {
        // Custom deleter ensures proper cleanup of all local buffers
        if (ptr) {
            delete[] ptr;
        }
        });

    CollisionJob* jobs = jobs_shared.get();

    // Initialize jobs
    for (int i = 0; i < job_count; i++) {
        jobs[i].start_cell = i * cells_per_job;
        jobs[i].end_cell = std::min((i + 1) * cells_per_job, cell_count);
        jobs[i].local_collision_buffer = new CollisionPair[1024]; // Local buffer for each job

        if (!jobs[i].local_collision_buffer) {
            // Memory allocation failed - the shared_ptr will handle cleanup
            return;
        }

        jobs[i].collision_count = 0;

        // Create task for this job
        // Store job data locally to avoid capturing the jobs array directly
        int job_start = jobs[i].start_cell;
        int job_end = jobs[i].end_cell;
        CollisionPair* local_buffer = jobs[i].local_collision_buffer;

        // Capture local variables by value instead of capturing 'jobs'
        auto* task = tasks.CreateTask([this, grid, chunks, chunk_count,
            local_buffer, job_start, job_end, &jobs_shared, i]() {
                // No need to validate job_idx now, we're using local variables

                // Process cells assigned to this job
                for (int cell_idx = job_start; cell_idx < job_end; cell_idx++) {
                    // Validate cell index
                    if (cell_idx < 0 || cell_idx >= grid->cell_count) {
                        continue;
                    }

                    // Safely access the cell
                    SpatialGrid3D::Cell* cell_ptr = &grid->cells[cell_idx];
                    if (!cell_ptr || !cell_ptr->entity_indices) {
                        continue;
                    }

                    int count = cell_ptr->count;

                    // Skip empty cells
                    if (count <= 0) continue;

                    // Check for collisions within this cell
                    for (int a = 0; a < count; a++) {
                        int entity_a = cell_ptr->entity_indices[a];

                        // Get entity A chunk and local index
                        int chunk_a = -1, local_a = -1;
                        // In a real implementation, we'd get these from the EntityManager

                        if (chunk_a < 0 || chunk_a >= chunk_count) continue;
                        if (!chunks[chunk_a] || !chunks[chunk_a]->hot.physics_layer) continue;

                        // Check against other entities in the same cell
                        for (int b = a + 1; b < count; b++) {
                            int entity_b = cell_ptr->entity_indices[b];

                            // Get entity B chunk and local index
                            int chunk_b = -1, local_b = -1;
                            // In a real implementation, we'd get these from the EntityManager

                            if (chunk_b < 0 || chunk_b >= chunk_count) continue;
                            if (!chunks[chunk_b] || !chunks[chunk_b]->hot.physics_layer) continue;

                            // Check if these layers can collide
                            int layer_a = chunks[chunk_a]->hot.physics_layer[local_a];
                            int layer_b = chunks[chunk_b]->hot.physics_layer[local_b];

                            if (!canCollide(layer_a, layer_b)) continue;

                            // Add collision pair to local buffer - with bounds checking
                            // Get a reference to our job's data to update collision count atomically
                            CollisionJob& job = jobs_shared.get()[i];

                            if (local_buffer && job.collision_count < 1024) {
                                // Thread-safe update to the collision buffer
                                int collision_idx = job.collision_count++;
                                if (collision_idx < 1024) {
                                    local_buffer[collision_idx].entity_a = entity_a;
                                    local_buffer[collision_idx].entity_b = entity_b;
                                }
                            }
                        }
                    }
                }
            });

        // Schedule the task
        tasks.ScheduleTask(task);
    }

    // Wait for all tasks to complete
    tasks.WaitAll();

    // Merge collision results - using raw pointer for compatibility with existing function
    mergeCollisionResults(jobs, job_count);

    // Clean up local buffers before shared_ptr releases the jobs array
    for (int i = 0; i < job_count; i++) {
        delete[] jobs[i].local_collision_buffer;
        jobs[i].local_collision_buffer = nullptr;
    }

    // shared_ptr will automatically clean up the jobs array
}
// Template implementation
template<typename T, size_t BlockSize>
FixedAllocator<T, BlockSize>::FixedAllocator()
    : current_block(nullptr), current_offset(BlockSize) {
    // We don't allocate any memory in the constructor
}

template<typename T, size_t BlockSize>
FixedAllocator<T, BlockSize>::~FixedAllocator() {
    // Clean up all allocated blocks
    for (Block* block : blocks) {
        delete block;
    }
}

// Material implementation
Material::Material() {
    diffuse_texture = -1;
    normal_texture = -1;
    specular_texture = -1;
    diffuse_color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    specular_color = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
    shininess = 32.0f;
}

// Mesh implementation
Mesh::Mesh() {
    vao = 0;
    vbo = 0;
    ibo = 0;
    vertex_count = 0;
    index_count = 0;
    bounds = AABB();
    lod_meshes = nullptr;
    lod_count = 0;
}

Mesh::~Mesh() {
    // Clean up any OpenGL resources
    if (vao != 0) {
        // In a real implementation, this would use OpenGL calls like:
        // glDeleteVertexArrays(1, &vao);
        // glDeleteBuffers(1, &vbo);
        // glDeleteBuffers(1, &ibo);
        vao = 0;
        vbo = 0;
        ibo = 0;
    }

    // Clean up LOD meshes
    if (lod_meshes) {
        for (int i = 0; i < lod_count; i++) {
            if (lod_meshes[i].vao != 0) {
                // glDeleteVertexArrays(1, &lod_meshes[i].vao);
                // glDeleteBuffers(1, &lod_meshes[i].vbo);
                // glDeleteBuffers(1, &lod_meshes[i].ibo);
            }
        }
        delete[] lod_meshes;
        lod_meshes = nullptr;
    }
}

// InstanceData implementation
InstanceData::InstanceData() {
    instance_count = 0;
    gpu_blocks = nullptr;
    // Initialize transform and color arrays
    for (int i = 0; i < INSTANCE_BATCH_SIZE; i++) {
        transforms[i] = glm::mat4(1.0f); // Identity matrix
        colors[i] = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f); // White
    }
}

InstanceData::~InstanceData() {
    // Clean up GPU blocks if allocated
    if (gpu_blocks) {
        delete[] gpu_blocks;
        gpu_blocks = nullptr;
    }
}


// GPUResources implementations
int GPUResources::createShader(const char* vertex_source, const char* fragment_source) {
    // In a real implementation, this would compile and link a shader program
    // For now, we'll return a placeholder shader ID

    if (!vertex_source || !fragment_source) {
        return -1; // Invalid input
    }

    // Check if we have space for a new shader
    if (shader_count >= shader_capacity) {
        return -1; // Out of space
    }

    // Create a new shader placeholder
    int shader_id = shader_count++;
    shaders[shader_id].program = shader_id + 1; // Just use a dummy ID for now

    // In a real implementation, this would:
    // 1. Compile vertex shader
    // 2. Compile fragment shader
    // 3. Link program
    // 4. Extract uniform locations

    return shader_id;
}

int GPUResources::createMesh(const float* vertices, int vertex_count, const int* indices, int index_count) {
    // Create a new mesh and upload vertex/index data to GPU

    if (!vertices || !indices || vertex_count <= 0 || index_count <= 0) {
        return -1; // Invalid input
    }

    // Check if we have space for a new mesh
    if (mesh_count >= mesh_capacity) {
        return -1; // Out of space
    }

    // Create a new mesh
    int mesh_id = mesh_count++;
    Mesh& mesh = meshes[mesh_id];

    // Set basic properties
    mesh.vertex_count = vertex_count;
    mesh.index_count = index_count;

    // In a real implementation, this would:
    // 1. Generate VAO, VBO, IBO
    // 2. Upload vertex data to VBO
    // 3. Upload index data to IBO
    // 4. Configure vertex attributes
    // 5. Calculate AABB from vertices

    // For now, just set dummy values
    mesh.vao = mesh_id + 1;
    mesh.vbo = mesh_id + 1000;
    mesh.ibo = mesh_id + 2000;

    // Calculate rough AABB from the first few vertices
    // In a real implementation, you'd iterate all vertices
    glm::vec3 min_pos(FLT_MAX);
    glm::vec3 max_pos(-FLT_MAX);

    for (int i = 0; i < std::min(vertex_count, 100); i++) {
        // Assuming each vertex has position at the start (x,y,z)
        glm::vec3 pos(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]);
        min_pos = glm::min(min_pos, pos);
        max_pos = glm::max(max_pos, pos);
    }

    mesh.bounds = AABB(min_pos, max_pos);

    return mesh_id;
}

int GPUResources::createMaterial(int diffuse_texture, int normal_texture, int specular_texture,
    const glm::vec4& diffuse_color, const glm::vec4& specular_color,
    float shininess) {
    // Create a new material with the given properties

    // Check if we have space for a new material
    if (material_count >= material_capacity) {
        return -1; // Out of space
    }

    // Create a new material
    int material_id = material_count++;
    Material& material = materials[material_id];

    // Set properties
    material.diffuse_texture = diffuse_texture;
    material.normal_texture = normal_texture;
    material.specular_texture = specular_texture;
    material.diffuse_color = diffuse_color;
    material.specular_color = specular_color;
    material.shininess = shininess;

    return material_id;
}



HugePageMemoryBlock::HugePageMemoryBlock(size_t size_bytes) {
    // Align size to page boundary
    size = (size_bytes + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);


    data = (uint8_t*)SDL_aligned_alloc(PAGE_SIZE, size);

    if (!data) {
        // Fallback to regular allocation if huge pages aren't available
        data = new uint8_t[size];
    }

    // Zero the memory
    memset(data, 0, size);
}

HugePageMemoryBlock::~HugePageMemoryBlock() {
    if (data) {

        SDL_aligned_free(data);
    }
}



template<typename T, size_t BlockSize>
T* FixedAllocator<T, BlockSize>::allocate() {
    // Check if we need a new block
    if (!current_block || current_offset + sizeof(T) > BlockSize) {
        Block* new_block = new Block;
        new_block->next = nullptr;
        if (current_block) {
            current_block->next = new_block;
        }
        current_block = new_block;
        blocks.push_back(new_block);
        current_offset = 0;
    }

    // Allocate from current block
    T* result = reinterpret_cast<T*>(current_block->data + current_offset);
    current_offset += sizeof(T);

    // Align offset to proper boundary for next allocation
    current_offset = (current_offset + alignof(T) - 1) & ~(alignof(T) - 1);

    // Construct the object
    new (result) T();

    return result;
}

template<typename T, size_t BlockSize>
void FixedAllocator<T, BlockSize>::reset() {
    // Destroy all allocated objects and free memory
    for (Block* block : blocks) {
        Block* next = nullptr;
        while (block) {
            next = block->next;
            delete block;
            block = next;
        }
    }
    blocks.clear();
    current_block = nullptr;
    current_offset = BlockSize;
}

template<typename T, size_t BlockSize>
size_t FixedAllocator<T, BlockSize>::get_allocated_blocks() const {
    return blocks.size();
}

// EntityAllocator implementation
template<typename T>
EntityAllocator<T>::EntityAllocator()
    : currentBlock(nullptr), currentIndex(ENTITIES_PER_BLOCK) {
}

template<typename T>
EntityAllocator<T>::~EntityAllocator() {
    Reset();
}

template<typename T>
T* EntityAllocator<T>::Allocate() {
    if (!currentBlock || currentIndex >= ENTITIES_PER_BLOCK) {
        // Allocate new block with huge page alignment
        HugePageMemoryBlock* hugeBlock = new HugePageMemoryBlock(sizeof(T) * ENTITIES_PER_BLOCK);
        currentBlock = reinterpret_cast<T*>(hugeBlock->data);
        blocks.push_back(currentBlock);
        currentIndex = 0;
    }

    return &currentBlock[currentIndex++];
}

template<typename T>
void EntityAllocator<T>::Reset() {
    for (auto block : blocks) {
        delete[] block;
    }
    blocks.clear();
    currentBlock = nullptr;
    currentIndex = ENTITIES_PER_BLOCK;
}

// FrameArenaAllocator implementation
FrameArenaAllocator::FrameArenaAllocator(size_t size) : capacity(size), current_offset(0) {
    memory_block = new uint8_t[size];
}

FrameArenaAllocator::~FrameArenaAllocator() {
    delete[] memory_block;
}

void* FrameArenaAllocator::allocate(size_t size, size_t alignment) {
    // Align the current offset to the requested alignment
    size_t aligned_offset = (current_offset + alignment - 1) & ~(alignment - 1);

    // Check if we have enough space
    if (aligned_offset + size > capacity) {
        return nullptr; // Out of memory
    }

    // Update current offset and return the allocated memory
    current_offset = aligned_offset + size;
    return memory_block + aligned_offset;
}

void FrameArenaAllocator::reset() {
    current_offset = 0;
}

size_t FrameArenaAllocator::getUsedMemory() const {
    return current_offset;
}

// BufferPool implementation
thread_local FrameArenaAllocator BufferPool::frame_allocator(64 * 1024); // 64KB default

BufferPool::BufferHandle::BufferHandle() noexcept
    : buffer_(nullptr), size_(0), pool_(nullptr) {
}

BufferPool::BufferHandle::BufferHandle(uint8_t* buffer, size_t size, BufferPool* pool) noexcept
    : buffer_(buffer), size_(size), pool_(pool) {
}

BufferPool::BufferHandle::BufferHandle(BufferHandle&& other) noexcept
    : buffer_(other.buffer_), size_(other.size_), pool_(other.pool_) {
    other.buffer_ = nullptr;
    other.size_ = 0;
    other.pool_ = nullptr;
}

BufferPool::BufferHandle& BufferPool::BufferHandle::operator=(BufferHandle&& other) noexcept {
    if (this != &other) {
        release();
        buffer_ = other.buffer_;
        size_ = other.size_;
        pool_ = other.pool_;
        other.buffer_ = nullptr;
        other.size_ = 0;
        other.pool_ = nullptr;
    }
    return *this;
}

BufferPool::BufferHandle::~BufferHandle() {
    release();
}

void BufferPool::BufferHandle::release() {
    if (buffer_ && pool_) {
        pool_->returnBuffer(buffer_, size_);
        buffer_ = nullptr;
        size_ = 0;
        pool_ = nullptr;
    }
}

template<typename T>
T* BufferPool::BufferHandle::as() const noexcept {
    return reinterpret_cast<T*>(buffer_);
}

uint8_t* BufferPool::BufferHandle::data() const noexcept {
    return buffer_;
}

size_t BufferPool::BufferHandle::size() const noexcept {
    return size_;
}

bool BufferPool::BufferHandle::valid() const noexcept {
    return buffer_ != nullptr;
}

BufferPool::BufferHandle::operator bool() const noexcept {
    return valid();
}

void BufferPool::BufferHandle::clear() {
    if (buffer_) {
        memset(buffer_, 0, size_);
    }
}

// Define the BufferBucket structure
struct BufferPool::BufferBucket {
    size_t buffer_size;
    std::vector<uint8_t*> free_buffers;

    BufferBucket(size_t size) : buffer_size(size) {}

    ~BufferBucket() {
        for (auto buffer : free_buffers) {
            delete[] buffer;
        }
    }
};

BufferPool::BufferPool() {
    // Create buckets for common sizes: 128B, 256B, 512B, 1KB, 2KB, 4KB, 8KB, 16KB, 32KB, 64KB
    const size_t sizes[] = { 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
    for (size_t size : sizes) {
        buckets.emplace_back(size);
    }
}

BufferPool::~BufferPool() {
    // Destructors of BufferBucket will clean up
}

BufferPool::BufferHandle BufferPool::getBuffer(size_t requested_size) {
    // Find the appropriate bucket
    for (auto& bucket : buckets) {
        if (bucket.buffer_size >= requested_size) {
            std::lock_guard<std::mutex> lock(mutex);

            uint8_t* buffer = nullptr;
            if (!bucket.free_buffers.empty()) {
                // Reuse a buffer from the pool
                buffer = bucket.free_buffers.back();
                bucket.free_buffers.pop_back();
            }
            else {
                // Allocate a new buffer
                buffer = new uint8_t[bucket.buffer_size];
            }

            return BufferHandle(buffer, bucket.buffer_size, this);
        }
    }

    // If size is larger than any bucket, allocate directly
    uint8_t* buffer = new uint8_t[requested_size];
    return BufferHandle(buffer, requested_size, this);
}

void BufferPool::returnBuffer(uint8_t* buffer, size_t size) {
    // Find the appropriate bucket
    for (auto& bucket : buckets) {
        if (bucket.buffer_size == size) {
            std::lock_guard<std::mutex> lock(mutex);
            bucket.free_buffers.push_back(buffer);
            return;
        }
    }

    // If size doesn't match any bucket, free directly
    delete[] buffer;
}

FrameArenaAllocator* BufferPool::getFrameArena() {
    return &frame_allocator;
}

void BufferPool::resetFrameArenas() {
    frame_allocator.reset();
}



// ComponentMask implementation
ComponentMask::ComponentMask() {
    // Initialize all blocks to 0
    for (auto& block : blocks) {
        block = 0;
    }
}

void ComponentMask::set(size_t index) {
    size_t block_idx = index / BITS_PER_BLOCK;
    size_t bit_idx = index % BITS_PER_BLOCK;

    if (block_idx < NUM_BLOCKS) {
        blocks[block_idx] |= (1ULL << bit_idx);
    }
}

void ComponentMask::clear(size_t index) {
    size_t block_idx = index / BITS_PER_BLOCK;
    size_t bit_idx = index % BITS_PER_BLOCK;

    if (block_idx < NUM_BLOCKS) {
        blocks[block_idx] &= ~(1ULL << bit_idx);
    }
}

bool ComponentMask::test(size_t index) const {
    size_t block_idx = index / BITS_PER_BLOCK;
    size_t bit_idx = index % BITS_PER_BLOCK;

    if (block_idx < NUM_BLOCKS) {
        return (blocks[block_idx] & (1ULL << bit_idx)) != 0;
    }
    return false;
}

bool ComponentMask::containsAll(const ComponentMask& other) const {
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        if ((blocks[i] & other.blocks[i]) != other.blocks[i]) {
            return false;
        }
    }
    return true;
}

bool ComponentMask::containsNone(const ComponentMask& other) const {
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        if ((blocks[i] & other.blocks[i]) != 0) {
            return false;
        }
    }
    return true;
}




// MortonSystem implementation
MortonSystem::MortonSystem(size_t max_entities) {
    // Allocate arrays
    codes = new uint64_t[max_entities];
    entity_indices = new int32_t[max_entities];
    cell_ranges = new int32_t[MORTON_CAPACITY * 2]; // Start/end for each cell

    // Initialize cell ranges
    for (size_t i = 0; i < MORTON_CAPACITY * 2; i += 2) {
        cell_ranges[i] = -1;     // Start index (none)
        cell_ranges[i + 1] = -1; // End index (none)
    }
}

MortonSystem::~MortonSystem() {
    delete[] codes;
    delete[] entity_indices;
    delete[] cell_ranges;
}

uint64_t MortonSystem::encodeMorton(const glm::vec3& position, float grid_size) {
    // Scale position to grid coordinates
    int x = static_cast<int>(position.x / grid_size) & 0x3FF; // 10 bits per coordinate
    int y = static_cast<int>(position.y / grid_size) & 0x3FF;
    int z = static_cast<int>(position.z / grid_size) & 0x3FF;

    // Interleave bits to create Morton code
    uint64_t code = 0;
    for (int i = 0; i < 10; ++i) {
        code |= ((x & (1 << i)) << (2 * i)) |
            ((y & (1 << i)) << (2 * i + 1)) |
            ((z & (1 << i)) << (2 * i + 2));
    }

    return code;
}

void MortonSystem::insertEntity(uint32_t entity_id, const glm::vec3& position, float radius) {
    // Compute Morton code for entity position
    uint64_t code = encodeMorton(position, radius);

    // TODO: Insert entity into sorted list based on Morton code
    // This would involve finding the insertion point, shifting existing entities,
    // and updating the cell ranges.
}

void MortonSystem::updateEntity(uint32_t entity_id, const glm::vec3& position, float radius) {
    // For simplicity, we'll just remove and reinsert
    removeEntity(entity_id);
    insertEntity(entity_id, position, radius);
}

void MortonSystem::removeEntity(uint32_t entity_id) {
    // TODO: Find entity in array and remove it
    // This would involve finding the entity, shifting other entities,
    // and updating the cell ranges.
}

size_t MortonSystem::queryRange(const glm::vec3& min, const glm::vec3& max, uint32_t* results, size_t max_results) {
    // TODO: Implement range query using Morton codes
    // This would involve determining the Morton code range for the query box,
    // then searching for entities within that range.

    return 0; // Return number of entities found
}



// LockFreeTaskSystem implementation
LockFreeTaskSystem::TaskQueue::TaskQueue() : head(0), tail(0) {
    // Initialize buffer - typical ring buffer size is power of 2 for bit masking
    const int QUEUE_SIZE = 4096; // Must be power of 2
    buffer = new Task * [QUEUE_SIZE];
    for (int i = 0; i < QUEUE_SIZE; ++i) {
        buffer[i] = nullptr;
    }
}

LockFreeTaskSystem::TaskQueue::~TaskQueue() {
    delete[] buffer;
}


LockFreeTaskSystem::Task* LockFreeTaskSystem::TaskQueue::Pop() {
    uint32_t current_head = head.load(std::memory_order_relaxed);
    uint32_t current_tail = tail.load(std::memory_order_acquire);

    if (current_head == current_tail) {
        return nullptr; // Queue is empty
    }

    Task* task = buffer[current_head & 4095]; // Mask with queue size - 1

    if (head.compare_exchange_strong(current_head, current_head + 1, std::memory_order_release)) {
        return task;
    }

    return nullptr; // Someone else popped the task
}

bool LockFreeTaskSystem::TaskQueue::Push(Task* task) {
    uint32_t current_tail = tail.load(std::memory_order_relaxed);
    uint32_t current_head = head.load(std::memory_order_acquire);

    if ((current_tail - current_head) >= 4096) {
        return false; // Queue is full
    }

    buffer[current_tail & 4095] = task; // Mask with queue size - 1

    tail.store(current_tail + 1, std::memory_order_release);
    return true;
}

LockFreeTaskSystem::LockFreeTaskSystem(size_t threadCount) : running(true), activeThreadCount(0) {
    // Initialize global queue

    // Determine number of worker threads
    if (threadCount == 0) {
        threadCount = std::thread::hardware_concurrency();
    }

    // Create thread local queues
    threadQueues.resize(threadCount);

    // Start worker threads
    workers.reserve(threadCount);
    for (size_t i = 0; i < threadCount; ++i) {
        workers.emplace_back(&LockFreeTaskSystem::WorkerMain, this, i);
    }
}

LockFreeTaskSystem::~LockFreeTaskSystem() {
    // Signal threads to exit
    running.store(false, std::memory_order_release);

    // Wait for threads to finish
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

LockFreeTaskSystem::Task* LockFreeTaskSystem::CreateTask(std::function<void()> func) {
    Task* task = taskAllocator.allocate();
    new (task) Task(func);
    return task;
}

void LockFreeTaskSystem::AddDependency(Task* dependent, Task* dependency) {
    dependency->dependents.push_back(dependent);
    dependent->dependencies.fetch_add(1, std::memory_order_relaxed);
}

void LockFreeTaskSystem::ScheduleTask(Task* task) {
    // If task has no dependencies, it can be executed immediately
    if (task->dependencies.load(std::memory_order_relaxed) == 0) {
        // Try to push to global queue
        if (!globalQueue.Push(task)) {
            // If global queue is full, execute immediately
            task->func();

            // Schedule dependent tasks
            for (Task* dependent : task->dependents) {
                if (dependent->dependencies.fetch_sub(1, std::memory_order_release) == 1) {
                    ScheduleTask(dependent);
                }
            }
        }
    }
}

void LockFreeTaskSystem::ScheduleTaskForThread(Task* task, int threadId) {
    if (threadId >= 0 && threadId < static_cast<int>(threadQueues.size())) {
        if (!threadQueues[threadId].Push(task)) {
            // If thread queue is full, push to global queue
            if (!globalQueue.Push(task)) {
                // If global queue is full, execute immediately
                task->func();

                // Schedule dependent tasks
                for (Task* dependent : task->dependents) {
                    if (dependent->dependencies.fetch_sub(1, std::memory_order_release) == 1) {
                        ScheduleTask(dependent);
                    }
                }
            }
        }
    }
    else {
        // Invalid thread ID, use global queue
        ScheduleTask(task);
    }
}

LockFreeTaskSystem::Task* LockFreeTaskSystem::StealTask(int thiefThreadId) {
    // Try to steal from other thread queues
    for (size_t victimId = 0; victimId < threadQueues.size(); ++victimId) {
        if (victimId != thiefThreadId) {
            Task* stolen = threadQueues[victimId].Pop();
            if (stolen) {
                return stolen;
            }
        }
    }

    // Try global queue
    return globalQueue.Pop();
}

void LockFreeTaskSystem::WaitAll() {
    // Process tasks until all are done and all threads are idle
    while (activeThreadCount.load(std::memory_order_acquire) > 0) {
        Task* task = globalQueue.Pop();
        if (task) {
            task->func();

            // Schedule dependent tasks
            for (Task* dependent : task->dependents) {
                if (dependent->dependencies.fetch_sub(1, std::memory_order_release) == 1) {
                    ScheduleTask(dependent);
                }
            }
        }
        else {
            // No tasks in global queue, yield to other threads
            std::this_thread::yield();
        }
    }
}

void LockFreeTaskSystem::ScheduleHierarchicalBatches(
    const HierarchyAwareTaskPartitioner& partitioner,
    std::function<void(int depth, int start, int count)> processFunc) {

    // Schedule tasks by depth, starting from shallowest (roots)
    for (const auto& batch : partitioner.depth_batches) {
        Task* task = CreateTask([=]() {
            processFunc(batch.depth, batch.start_idx, batch.count);
            });

        // Set dependencies based on depth
        // Each depth level depends on the previous level being complete
        // This will be handled through task scheduling

        ScheduleTask(task);
    }
}
#include <iostream>
void LockFreeTaskSystem::WorkerMain(int threadId) {
    activeThreadCount.fetch_add(1, std::memory_order_release);
    bool isActive = true;

    while (running.load(std::memory_order_acquire)) {
        // First try local queue
        Task* task = threadQueues[threadId].Pop();

        if (!task) {
            // Try stealing
            task = StealTask(threadId);
        }

        if (task) {
            // If thread was inactive, mark it as active
            if (!isActive) {
                activeThreadCount.fetch_add(1, std::memory_order_release);
                isActive = true;
            }

            // Safety check: Validate task pointer before executing
            if (task != nullptr && task->func) {
                try {
                    // Execute task with exception handling
                    task->func();
                }
                catch (const std::exception& e) {
                    // Log exception
                    // In a real implementation, you would use your logging system
                    std::cerr << "Exception in task execution: " << e.what() << std::endl;
                }
                catch (...) {
                    // Log unknown exception
                    std::cerr << "Unknown exception in task execution" << std::endl;
                }

                // Schedule dependent tasks
                for (Task* dependent : task->dependents) {
                    if (dependent && dependent->dependencies.fetch_sub(1, std::memory_order_release) == 1) {
                        ScheduleTask(dependent);
                    }
                }
            }
            else {
                // Log error about invalid task
                std::cerr << "Invalid task encountered in worker " << threadId << std::endl;
            }
        }
        else {
            // No work found - mark thread as inactive
            if (isActive) {
                activeThreadCount.fetch_sub(1, std::memory_order_release);
                isActive = false;
            }

            // No work, yield to other threads and sleep a bit to reduce CPU usage
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Make sure to decrement only if the thread is still marked as active
    if (isActive) {
        activeThreadCount.fetch_sub(1, std::memory_order_release);
    }
}


// SIMD optimization helpers
//--------------------------

// SimdFloat implementation
#if defined(USE_AVX512)

SimdFloat::SimdFloat() {
    v = _mm512_setzero_ps();
}

SimdFloat::SimdFloat(float val) {
    v = _mm512_set1_ps(val);
}

SimdFloat SimdFloat::load(const float* ptr) {
    SimdFloat result;
    result.v = _mm512_loadu_ps(ptr);
    return result;
}

void SimdFloat::store(float* ptr) const {
    _mm512_storeu_ps(ptr, v);
}

SimdFloat SimdFloat::operator+(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm512_add_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator-(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm512_sub_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator*(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm512_mul_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator/(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm512_div_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::sqrt() const {
    SimdFloat result;
    result.v = _mm512_sqrt_ps(v);
    return result;
}

SimdFloat SimdFloat::rsqrt() const {
    SimdFloat result;
    result.v = _mm512_rsqrt14_ps(v); // Approximate reciprocal square root
    return result;
}

SimdFloat SimdFloat::zero() {
    return SimdFloat(0.0f);
}

#elif defined(USE_AVX) || defined(USE_AVX2)

SimdFloat::SimdFloat() {
    v = _mm256_setzero_ps();
}

SimdFloat::SimdFloat(float val) {
    v = _mm256_set1_ps(val);
}

SimdFloat SimdFloat::load(const float* ptr) {
    SimdFloat result;
    result.v = _mm256_loadu_ps(ptr);
    return result;
}

void SimdFloat::store(float* ptr) const {
    _mm256_storeu_ps(ptr, v);
}

SimdFloat SimdFloat::operator+(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm256_add_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator-(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm256_sub_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator*(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm256_mul_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator/(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm256_div_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::sqrt() const {
    SimdFloat result;
    result.v = _mm256_sqrt_ps(v);
    return result;
}

SimdFloat SimdFloat::rsqrt() const {
    SimdFloat result;
    result.v = _mm256_rsqrt_ps(v);
    return result;
}

SimdFloat SimdFloat::zero() {
    return SimdFloat(0.0f);
}

#elif defined(USE_SSE)

SimdFloat::SimdFloat() {
    v = _mm_setzero_ps();
}

SimdFloat::SimdFloat(float val) {
    v = _mm_set1_ps(val);
}

SimdFloat SimdFloat::load(const float* ptr) {
    SimdFloat result;
    result.v = _mm_loadu_ps(ptr);
    return result;
}

void SimdFloat::store(float* ptr) const {
    _mm_storeu_ps(ptr, v);
}

SimdFloat SimdFloat::operator+(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm_add_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator-(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm_sub_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator*(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm_mul_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator/(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm_div_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::sqrt() const {
    SimdFloat result;
    result.v = _mm_sqrt_ps(v);
    return result;
}

SimdFloat SimdFloat::rsqrt() const {
    SimdFloat result;
    result.v = _mm_rsqrt_ps(v);
    return result;
}

SimdFloat SimdFloat::zero() {
    return SimdFloat(0.0f);
}

#else

SimdFloat::SimdFloat() {
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        v[i] = 0.0f;
    }
}

SimdFloat::SimdFloat(float val) {
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        v[i] = val;
    }
}

SimdFloat SimdFloat::load(const float* ptr) {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = ptr[i];
    }
    return result;
}

void SimdFloat::store(float* ptr) const {
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        ptr[i] = v[i];
    }
}

SimdFloat SimdFloat::operator+(const SimdFloat& rhs) const {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] + rhs.v[i];
    }
    return result;
}

SimdFloat SimdFloat::operator-(const SimdFloat& rhs) const {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] - rhs.v[i];
    }
    return result;
}

SimdFloat SimdFloat::operator*(const SimdFloat& rhs) const {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] * rhs.v[i];
    }
    return result;
}

SimdFloat SimdFloat::operator/(const SimdFloat& rhs) const {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] / rhs.v[i];
    }
    return result;
}

SimdFloat SimdFloat::sqrt() const {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = std::sqrt(v[i]);
    }
    return result;
}

SimdFloat SimdFloat::rsqrt() const {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = 1.0f / std::sqrt(v[i]);
    }
    return result;
}

SimdFloat SimdFloat::zero() {
    return SimdFloat(0.0f);
}

#endif

// SimdInt implementation
#if defined(USE_AVX512)

SimdInt::SimdInt() {
    v = _mm512_setzero_si512();
}

SimdInt::SimdInt(int val) {
    v = _mm512_set1_epi32(val);
}

SimdInt SimdInt::load(const int* ptr) {
    SimdInt result;
    result.v = _mm512_loadu_si512(ptr);
    return result;
}

void SimdInt::store(int* ptr) const {
    _mm512_storeu_si512(ptr, v);
}

SimdInt SimdInt::operator&(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm512_and_si512(v, rhs.v);
    return result;
}

SimdInt SimdInt::operator|(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm512_or_si512(v, rhs.v);
    return result;
}

SimdInt SimdInt::operator^(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm512_xor_si512(v, rhs.v);
    return result;
}

#elif defined(USE_AVX) || defined(USE_AVX2)

SimdInt::SimdInt() {
    v = _mm256_setzero_si256();
}

SimdInt::SimdInt(int val) {
    v = _mm256_set1_epi32(val);
}

SimdInt SimdInt::load(const int* ptr) {
    SimdInt result;
    result.v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    return result;
}

void SimdInt::store(int* ptr) const {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v);
}

SimdInt SimdInt::operator&(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm256_and_si256(v, rhs.v);
    return result;
}

SimdInt SimdInt::operator|(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm256_or_si256(v, rhs.v);
    return result;
}

SimdInt SimdInt::operator^(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm256_xor_si256(v, rhs.v);
    return result;
}

#elif defined(USE_SSE)

SimdInt::SimdInt() {
    v = _mm_setzero_si128();
}

SimdInt::SimdInt(int val) {
    v = _mm_set1_epi32(val);
}

SimdInt SimdInt::load(const int* ptr) {
    SimdInt result;
    result.v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    return result;
}

void SimdInt::store(int* ptr) const {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v);
}

SimdInt SimdInt::operator&(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm_and_si128(v, rhs.v);
    return result;
}

SimdInt SimdInt::operator|(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm_or_si128(v, rhs.v);
    return result;
}

SimdInt SimdInt::operator^(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm_xor_si128(v, rhs.v);
    return result;
}

#else

SimdInt::SimdInt() {
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        v[i] = 0;
    }
}

SimdInt::SimdInt(int val) {
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        v[i] = val;
    }
}

SimdInt SimdInt::load(const int* ptr) {
    SimdInt result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = ptr[i];
    }
    return result;
}

void SimdInt::store(int* ptr) const {
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        ptr[i] = v[i];
    }
}

SimdInt SimdInt::operator&(const SimdInt& rhs) const {
    SimdInt result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] & rhs.v[i];
    }
    return result;
}

SimdInt SimdInt::operator|(const SimdInt& rhs) const {
    SimdInt result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] | rhs.v[i];
    }
    return result;
}

SimdInt SimdInt::operator^(const SimdInt& rhs) const {
    SimdInt result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] ^ rhs.v[i];
    }
    return result;
}

#endif

// SimdMask implementation
#if defined(USE_AVX512)

SimdMask::SimdMask() {
    mask = 0;
}

SimdMask::SimdMask(uint32_t bit_mask) {
    mask = bit_mask & 0xFFFF; // 16 lanes for AVX-512
}

bool SimdMask::get(int index) const {
    return (mask & (1 << index)) != 0;
}

void SimdMask::set(int index, bool value) {
    if (value) {
        mask |= (1 << index);
    }
    else {
        mask &= ~(1 << index);
    }
}

SimdMask SimdMask::operator&(const SimdMask& other) const {
    SimdMask result;
    result.mask = mask & other.mask;
    return result;
}

SimdMask SimdMask::operator|(const SimdMask& other) const {
    SimdMask result;
    result.mask = mask | other.mask;
    return result;
}

SimdMask SimdMask::operator^(const SimdMask& other) const {
    SimdMask result;
    result.mask = mask ^ other.mask;
    return result;
}

SimdMask SimdMask::operator~() const {
    SimdMask result;
    result.mask = ~mask & 0xFFFF; // 16 lanes for AVX-512
    return result;
}

bool SimdMask::none() const {
    return mask == 0;
}

bool SimdMask::any() const {
    return mask != 0;
}

bool SimdMask::all() const {
    return mask == 0xFFFF; // 16 lanes for AVX-512
}

#elif defined(USE_AVX2) || defined(USE_AVX)

SimdMask::SimdMask() {
    mask = _mm256_setzero_si256();
}

SimdMask::SimdMask(uint32_t bit_mask) {
    // Convert bit mask to 8 integers (0 or -1) for AVX/AVX2
    int expanded_mask[8];
    for (int i = 0; i < 8; ++i) {
        expanded_mask[i] = (bit_mask & (1 << i)) ? -1 : 0;
    }
    mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(expanded_mask));
}

bool SimdMask::get(int index) const {
    alignas(32) int values[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(values), mask);
    return values[index] != 0;
}

void SimdMask::set(int index, bool value) {
    alignas(32) int values[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(values), mask);
    values[index] = value ? -1 : 0;
    mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(values));
}

SimdMask SimdMask::operator&(const SimdMask& other) const {
    SimdMask result;
    result.mask = _mm256_and_si256(mask, other.mask);
    return result;
}

SimdMask SimdMask::operator|(const SimdMask& other) const {
    SimdMask result;
    result.mask = _mm256_or_si256(mask, other.mask);
    return result;
}

SimdMask SimdMask::operator^(const SimdMask& other) const {
    SimdMask result;
    result.mask = _mm256_xor_si256(mask, other.mask);
    return result;
}

SimdMask SimdMask::operator~() const {
    // Create all ones
    __m256i all_ones = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
    all_ones = _mm256_xor_si256(all_ones, all_ones); // -1 in all lanes

    SimdMask result;
    result.mask = _mm256_xor_si256(mask, all_ones);
    return result;
}

bool SimdMask::none() const {
    return _mm256_testz_si256(mask, mask) != 0;
}

bool SimdMask::any() const {
    return _mm256_testz_si256(mask, mask) == 0;
}

bool SimdMask::all() const {
    __m256i all_ones = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
    all_ones = _mm256_xor_si256(all_ones, all_ones); // -1 in all lanes

    // XOR with all_ones and then check if all zeros
    __m256i temp = _mm256_xor_si256(mask, all_ones);
    return _mm256_testz_si256(temp, temp) != 0;
}

#elif defined(USE_SSE)

SimdMask::SimdMask() {
    mask = _mm_setzero_si128();
}

SimdMask::SimdMask(uint32_t bit_mask) {
    // Convert bit mask to 4 integers (0 or -1) for SSE
    int expanded_mask[4];
    for (int i = 0; i < 4; ++i) {
        expanded_mask[i] = (bit_mask & (1 << i)) ? -1 : 0;
    }
    mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(expanded_mask));
}

bool SimdMask::get(int index) const {
    alignas(16) int values[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(values), mask);
    return values[index] != 0;
}

void SimdMask::set(int index, bool value) {
    alignas(16) int values[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(values), mask);
    values[index] = value ? -1 : 0;
    mask = _mm_load_si128(reinterpret_cast<const __m128i*>(values));
}

SimdMask SimdMask::operator&(const SimdMask& other) const {
    SimdMask result;
    result.mask = _mm_and_si128(mask, other.mask);
    return result;
}

SimdMask SimdMask::operator|(const SimdMask& other) const {
    SimdMask result;
    result.mask = _mm_or_si128(mask, other.mask);
    return result;
}

SimdMask SimdMask::operator^(const SimdMask& other) const {
    SimdMask result;
    result.mask = _mm_xor_si128(mask, other.mask);
    return result;
}

SimdMask SimdMask::operator~() const {
    // Create all ones
    __m128i all_ones = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
    all_ones = _mm_xor_si128(all_ones, all_ones); // -1 in all lanes

    SimdMask result;
    result.mask = _mm_xor_si128(mask, all_ones);
    return result;
}

bool SimdMask::none() const {
#if defined(__SSE4_1__)
    return _mm_testz_si128(mask, mask) != 0;
#else
    // Fallback for SSE2/3
    alignas(16) int values[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(values), mask);
    return values[0] == 0 && values[1] == 0 && values[2] == 0 && values[3] == 0;
#endif
}

bool SimdMask::any() const {
#if defined(__SSE4_1__)
    return _mm_testz_si128(mask, mask) == 0;
#else
    // Fallback for SSE2/3
    alignas(16) int values[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(values), mask);
    return values[0] != 0 || values[1] != 0 || values[2] != 0 || values[3] != 0;
#endif
}

bool SimdMask::all() const {
    __m128i all_ones = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
    all_ones = _mm_xor_si128(all_ones, all_ones); // -1 in all lanes

#if defined(__SSE4_1__)
    // XOR with all_ones and then check if all zeros
    __m128i temp = _mm_xor_si128(mask, all_ones);
    return _mm_testz_si128(temp, temp) != 0;
#else
    // Fallback for SSE2/3
    alignas(16) int values[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(values), mask);
    return values[0] == -1 && values[1] == -1 && values[2] == -1 && values[3] == -1;
#endif
}

#else

SimdMask::SimdMask() {
    mask = 0;
}

SimdMask::SimdMask(uint32_t bit_mask) {
    mask = bit_mask & ((1u << SIMD_WIDTH) - 1);
}

bool SimdMask::get(int index) const {
    return (mask & (1u << index)) != 0;
}

void SimdMask::set(int index, bool value) {
    if (value) {
        mask |= (1u << index);
    }
    else {
        mask &= ~(1u << index);
    }
}

SimdMask SimdMask::operator&(const SimdMask& other) const {
    SimdMask result;
    result.mask = mask & other.mask;
    return result;
}

SimdMask SimdMask::operator|(const SimdMask& other) const {
    SimdMask result;
    result.mask = mask | other.mask;
    return result;
}

SimdMask SimdMask::operator^(const SimdMask& other) const {
    SimdMask result;
    result.mask = mask ^ other.mask;
    return result;
}

SimdMask SimdMask::operator~() const {
    SimdMask result;
    result.mask = ~mask & ((1u << SIMD_WIDTH) - 1);
    return result;
}

bool SimdMask::none() const {
    return mask == 0;
}

bool SimdMask::any() const {
    return mask != 0;
}

bool SimdMask::all() const {
    return mask == ((1u << SIMD_WIDTH) - 1);
}

#endif

// SimdAABB implementation
#if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_AVX2) || defined(USE_SSE) || !defined(USE_SIMD)

SimdAABB::SimdAABB() {
    min_x = SimdFloat(std::numeric_limits<float>::max());
    min_y = SimdFloat(std::numeric_limits<float>::max());
    min_z = SimdFloat(std::numeric_limits<float>::max());
    max_x = SimdFloat(-std::numeric_limits<float>::max());
    max_y = SimdFloat(-std::numeric_limits<float>::max());
    max_z = SimdFloat(-std::numeric_limits<float>::max());
}

SimdAABB SimdAABB::load(const AABB* boxes) {
    SimdAABB result;

    float min_x_arr[SIMD_WIDTH], min_y_arr[SIMD_WIDTH], min_z_arr[SIMD_WIDTH];
    float max_x_arr[SIMD_WIDTH], max_y_arr[SIMD_WIDTH], max_z_arr[SIMD_WIDTH];

    for (int i = 0; i < SIMD_WIDTH; ++i) {
        min_x_arr[i] = boxes[i].min.x;
        min_y_arr[i] = boxes[i].min.y;
        min_z_arr[i] = boxes[i].min.z;
        max_x_arr[i] = boxes[i].max.x;
        max_y_arr[i] = boxes[i].max.y;
        max_z_arr[i] = boxes[i].max.z;
    }

    result.min_x = SimdFloat::load(min_x_arr);
    result.min_y = SimdFloat::load(min_y_arr);
    result.min_z = SimdFloat::load(min_z_arr);
    result.max_x = SimdFloat::load(max_x_arr);
    result.max_y = SimdFloat::load(max_y_arr);
    result.max_z = SimdFloat::load(max_z_arr);

    return result;
}

void SimdAABB::store(AABB* boxes) const {
    float min_x_arr[SIMD_WIDTH], min_y_arr[SIMD_WIDTH], min_z_arr[SIMD_WIDTH];
    float max_x_arr[SIMD_WIDTH], max_y_arr[SIMD_WIDTH], max_z_arr[SIMD_WIDTH];

    min_x.store(min_x_arr);
    min_y.store(min_y_arr);
    min_z.store(min_z_arr);
    max_x.store(max_x_arr);
    max_y.store(max_y_arr);
    max_z.store(max_z_arr);

    for (int i = 0; i < SIMD_WIDTH; ++i) {
        boxes[i].min.x = min_x_arr[i];
        boxes[i].min.y = min_y_arr[i];
        boxes[i].min.z = min_z_arr[i];
        boxes[i].max.x = max_x_arr[i];
        boxes[i].max.y = max_y_arr[i];
        boxes[i].max.z = max_z_arr[i];
    }
}



SimdMask SimdAABB::overlaps(const SimdAABB& other) const {
#if defined(USE_AVX512)
    __mmask16 overlap_mask = 0;

    // Get SIMD vectors for comparison
    __m512 this_min_x = min_x.v;
    __m512 this_min_y = min_y.v;
    __m512 this_min_z = min_z.v;
    __m512 this_max_x = max_x.v;
    __m512 this_max_y = max_y.v;
    __m512 this_max_z = max_z.v;

    __m512 other_min_x = other.min_x.v;
    __m512 other_min_y = other.min_y.v;
    __m512 other_min_z = other.min_z.v;
    __m512 other_max_x = other.max_x.v;
    __m512 other_max_y = other.max_y.v;
    __m512 other_max_z = other.max_z.v;

    // Check overlap condition for each axis
    __mmask16 x_overlap = _mm512_cmp_ps_mask(this_max_x, other_min_x, _CMP_GE_OQ) &
        _mm512_cmp_ps_mask(this_min_x, other_max_x, _CMP_LE_OQ);

    __mmask16 y_overlap = _mm512_cmp_ps_mask(this_max_y, other_min_y, _CMP_GE_OQ) &
        _mm512_cmp_ps_mask(this_min_y, other_max_y, _CMP_LE_OQ);

    __mmask16 z_overlap = _mm512_cmp_ps_mask(this_max_z, other_min_z, _CMP_GE_OQ) &
        _mm512_cmp_ps_mask(this_min_z, other_max_z, _CMP_LE_OQ);

    // Boxes overlap if they overlap on all axes
    overlap_mask = x_overlap & y_overlap & z_overlap;

    SimdMask result;
    result.mask = overlap_mask;
    return result;

#elif defined(USE_AVX) || defined(USE_AVX2)
    // Using AVX/AVX2 implementation
    // Compare each dimension for overlap
    __m256 x_min_overlap = _mm256_cmp_ps(max_x.v, other.min_x.v, _CMP_GE_OQ);
    __m256 x_max_overlap = _mm256_cmp_ps(min_x.v, other.max_x.v, _CMP_LE_OQ);
    __m256 x_overlaps = _mm256_and_ps(x_min_overlap, x_max_overlap);

    __m256 y_min_overlap = _mm256_cmp_ps(max_y.v, other.min_y.v, _CMP_GE_OQ);
    __m256 y_max_overlap = _mm256_cmp_ps(min_y.v, other.max_y.v, _CMP_LE_OQ);
    __m256 y_overlaps = _mm256_and_ps(y_min_overlap, y_max_overlap);

    __m256 z_min_overlap = _mm256_cmp_ps(max_z.v, other.min_z.v, _CMP_GE_OQ);
    __m256 z_max_overlap = _mm256_cmp_ps(min_z.v, other.max_z.v, _CMP_LE_OQ);
    __m256 z_overlaps = _mm256_and_ps(z_min_overlap, z_max_overlap);

    // All dimensions must overlap
    __m256 all_overlaps = _mm256_and_ps(x_overlaps, _mm256_and_ps(y_overlaps, z_overlaps));

    SimdMask result;
    result.mask = _mm256_castps_si256(all_overlaps);
    return result;

#elif defined(USE_SSE)
    // Using SSE implementation
    // Compare each dimension for overlap
    __m128 x_min_overlap = _mm_cmpge_ps(max_x.v, other.min_x.v);
    __m128 x_max_overlap = _mm_cmple_ps(min_x.v, other.max_x.v);
    __m128 x_overlaps = _mm_and_ps(x_min_overlap, x_max_overlap);

    __m128 y_min_overlap = _mm_cmpge_ps(max_y.v, other.min_y.v);
    __m128 y_max_overlap = _mm_cmple_ps(min_y.v, other.max_y.v);
    __m128 y_overlaps = _mm_and_ps(y_min_overlap, y_max_overlap);

    __m128 z_min_overlap = _mm_cmpge_ps(max_z.v, other.min_z.v);
    __m128 z_max_overlap = _mm_cmple_ps(min_z.v, other.max_z.v);
    __m128 z_overlaps = _mm_and_ps(z_min_overlap, z_max_overlap);

    // All dimensions must overlap
    __m128 all_overlaps = _mm_and_ps(x_overlaps, _mm_and_ps(y_overlaps, z_overlaps));

    SimdMask result;
    result.mask = _mm_castps_si128(all_overlaps);
    return result;

#else
    // Scalar implementation
    SimdMask result;
    result.mask = 0;

    float min_x_a[SIMD_WIDTH], min_y_a[SIMD_WIDTH], min_z_a[SIMD_WIDTH];
    float max_x_a[SIMD_WIDTH], max_y_a[SIMD_WIDTH], max_z_a[SIMD_WIDTH];

    float min_x_b[SIMD_WIDTH], min_y_b[SIMD_WIDTH], min_z_b[SIMD_WIDTH];
    float max_x_b[SIMD_WIDTH], max_y_b[SIMD_WIDTH], max_z_b[SIMD_WIDTH];

    min_x.store(min_x_a); min_y.store(min_y_a); min_z.store(min_z_a);
    max_x.store(max_x_a); max_y.store(max_y_a); max_z.store(max_z_a);

    other.min_x.store(min_x_b); other.min_y.store(min_y_b); other.min_z.store(min_z_b);
    other.max_x.store(max_x_b); other.max_y.store(max_y_b); other.max_z.store(max_z_b);

    for (int i = 0; i < SIMD_WIDTH; ++i) {
        bool x_overlap = max_x_a[i] >= min_x_b[i] && min_x_a[i] <= max_x_b[i];
        bool y_overlap = max_y_a[i] >= min_y_b[i] && min_y_a[i] <= max_y_b[i];
        bool z_overlap = max_z_a[i] >= min_z_b[i] && min_z_a[i] <= max_z_b[i];

        if (x_overlap && y_overlap && z_overlap) {
            result.mask |= (1u << i);
        }
    }

    return result;
#endif
}

SimdMask SimdAABB::contains(const SimdAABB& other) const {
#if defined(USE_AVX512)
    __mmask16 contains_mask = 0;

    // Check containment condition for each axis
    __mmask16 x_contains = _mm512_cmp_ps_mask(min_x.v, other.min_x.v, _CMP_LE_OQ) &
        _mm512_cmp_ps_mask(max_x.v, other.max_x.v, _CMP_GE_OQ);

    __mmask16 y_contains = _mm512_cmp_ps_mask(min_y.v, other.min_y.v, _CMP_LE_OQ) &
        _mm512_cmp_ps_mask(max_y.v, other.max_y.v, _CMP_GE_OQ);

    __mmask16 z_contains = _mm512_cmp_ps_mask(min_z.v, other.min_z.v, _CMP_LE_OQ) &
        _mm512_cmp_ps_mask(max_z.v, other.max_z.v, _CMP_GE_OQ);

    // All axes must be contained
    contains_mask = x_contains & y_contains & z_contains;

    SimdMask result;
    result.mask = contains_mask;
    return result;

#elif defined(USE_AVX) || defined(USE_AVX2)
    // Using AVX/AVX2 implementation
    __m256 x_min_contains = _mm256_cmp_ps(min_x.v, other.min_x.v, _CMP_LE_OQ);
    __m256 x_max_contains = _mm256_cmp_ps(max_x.v, other.max_x.v, _CMP_GE_OQ);
    __m256 x_contains = _mm256_and_ps(x_min_contains, x_max_contains);

    __m256 y_min_contains = _mm256_cmp_ps(min_y.v, other.min_y.v, _CMP_LE_OQ);
    __m256 y_max_contains = _mm256_cmp_ps(max_y.v, other.max_y.v, _CMP_GE_OQ);
    __m256 y_contains = _mm256_and_ps(y_min_contains, y_max_contains);

    __m256 z_min_contains = _mm256_cmp_ps(min_z.v, other.min_z.v, _CMP_LE_OQ);
    __m256 z_max_contains = _mm256_cmp_ps(max_z.v, other.max_z.v, _CMP_GE_OQ);
    __m256 z_contains = _mm256_and_ps(z_min_contains, z_max_contains);

    // All dimensions must contain
    __m256 all_contains = _mm256_and_ps(x_contains, _mm256_and_ps(y_contains, z_contains));

    SimdMask result;
    result.mask = _mm256_castps_si256(all_contains);
    return result;

#elif defined(USE_SSE)
    // Using SSE implementation
    __m128 x_min_contains = _mm_cmple_ps(min_x.v, other.min_x.v);
    __m128 x_max_contains = _mm_cmpge_ps(max_x.v, other.max_x.v);
    __m128 x_contains = _mm_and_ps(x_min_contains, x_max_contains);

    __m128 y_min_contains = _mm_cmple_ps(min_y.v, other.min_y.v);
    __m128 y_max_contains = _mm_cmpge_ps(max_y.v, other.max_y.v);
    __m128 y_contains = _mm_and_ps(y_min_contains, y_max_contains);

    __m128 z_min_contains = _mm_cmple_ps(min_z.v, other.min_z.v);
    __m128 z_max_contains = _mm_cmpge_ps(max_z.v, other.max_z.v);
    __m128 z_contains = _mm_and_ps(z_min_contains, z_max_contains);

    // All dimensions must contain
    __m128 all_contains = _mm_and_ps(x_contains, _mm_and_ps(y_contains, z_contains));

    SimdMask result;
    result.mask = _mm_castps_si128(all_contains);
    return result;

#else
    // Scalar implementation
    SimdMask result;
    result.mask = 0;

    float min_x_a[SIMD_WIDTH], min_y_a[SIMD_WIDTH], min_z_a[SIMD_WIDTH];
    float max_x_a[SIMD_WIDTH], max_y_a[SIMD_WIDTH], max_z_a[SIMD_WIDTH];

    float min_x_b[SIMD_WIDTH], min_y_b[SIMD_WIDTH], min_z_b[SIMD_WIDTH];
    float max_x_b[SIMD_WIDTH], max_y_b[SIMD_WIDTH], max_z_b[SIMD_WIDTH];

    min_x.store(min_x_a); min_y.store(min_y_a); min_z.store(min_z_a);
    max_x.store(max_x_a); max_y.store(max_y_a); max_z.store(max_z_a);

    other.min_x.store(min_x_b); other.min_y.store(min_y_b); other.min_z.store(min_z_b);
    other.max_x.store(max_x_b); other.max_y.store(max_y_b); other.max_z.store(max_z_b);

    for (int i = 0; i < SIMD_WIDTH; ++i) {
        bool x_contains = min_x_a[i] <= min_x_b[i] && max_x_a[i] >= max_x_b[i];
        bool y_contains = min_y_a[i] <= min_y_b[i] && max_y_a[i] >= max_y_b[i];
        bool z_contains = min_z_a[i] <= min_z_b[i] && max_z_a[i] >= max_z_b[i];

        if (x_contains && y_contains && z_contains) {
            result.mask |= (1u << i);
        }
    }

    return result;
#endif
}

#endif

// Mat4SIMD implementation
#if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_AVX2) || defined(USE_SSE) || !defined(USE_SIMD)

Mat4SIMD::Mat4SIMD() {
    for (int i = 0; i < 16; ++i) {
        m[i] = SimdFloat(0.0f);
    }
}

Mat4SIMD::Mat4SIMD(const glm::mat4& mat) {
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float values[SIMD_WIDTH];
            for (int i = 0; i < SIMD_WIDTH; ++i) {
                values[i] = mat[col][row];
            }
            m[col * 4 + row] = SimdFloat::load(values);
        }
    }
}

Mat4SIMD Mat4SIMD::load(const glm::mat4* matrices) {
    Mat4SIMD result;

    // Transpose matrices for SIMD operations
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float values[SIMD_WIDTH];
            for (int i = 0; i < SIMD_WIDTH; ++i) {
                values[i] = matrices[i][col][row];
            }
            result.m[col * 4 + row] = SimdFloat::load(values);
        }
    }

    return result;
}

void Mat4SIMD::store(glm::mat4* matrices) const {
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float values[SIMD_WIDTH];
            m[col * 4 + row].store(values);

            for (int i = 0; i < SIMD_WIDTH; ++i) {
                matrices[i][col][row] = values[i];
            }
        }
    }
}

Mat4SIMD Mat4SIMD::operator*(const Mat4SIMD& rhs) const {
    Mat4SIMD result;

    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            // Compute dot product of row of lhs with column of rhs
            SimdFloat sum = SimdFloat::zero();

            for (int k = 0; k < 4; ++k) {
                SimdFloat a = m[k * 4 + row];
                SimdFloat b = rhs.m[col * 4 + k];
                sum = sum + (a * b);
            }

            result.m[col * 4 + row] = sum;
        }
    }

    return result;
}

Mat4SIMD Mat4SIMD::identity() {
    Mat4SIMD result;

    for (int i = 0; i < 4; ++i) {
        result.m[i * 4 + i] = SimdFloat(1.0f);
    }

    return result;
}

Mat4SIMD Mat4SIMD::translation(const SimdFloat& x, const SimdFloat& y, const SimdFloat& z) {
    Mat4SIMD result = identity();

    result.m[3 * 4 + 0] = x;  // m[12]
    result.m[3 * 4 + 1] = y;  // m[13]
    result.m[3 * 4 + 2] = z;  // m[14]

    return result;
}

Mat4SIMD Mat4SIMD::rotation(const SimdFloat& qx, const SimdFloat& qy, const SimdFloat& qz, const SimdFloat& qw) {
    Mat4SIMD result;

    // Calculate quaternion terms
    SimdFloat qx2 = qx * qx;
    SimdFloat qy2 = qy * qy;
    SimdFloat qz2 = qz * qz;
    SimdFloat qw2 = qw * qw;

    SimdFloat qxqy = qx * qy;
    SimdFloat qxqz = qx * qz;
    SimdFloat qxqw = qx * qw;
    SimdFloat qyqz = qy * qz;
    SimdFloat qyqw = qy * qw;
    SimdFloat qzqw = qz * qw;

    // First row
    result.m[0] = qw2 + qx2 - qy2 - qz2;
    result.m[1] = SimdFloat(2.0f) * (qxqy - qzqw);
    result.m[2] = SimdFloat(2.0f) * (qxqz + qyqw);
    result.m[3] = SimdFloat(0.0f);

    // Second row
    result.m[4] = SimdFloat(2.0f) * (qxqy + qzqw);
    result.m[5] = qw2 - qx2 + qy2 - qz2;
    result.m[6] = SimdFloat(2.0f) * (qyqz - qxqw);
    result.m[7] = SimdFloat(0.0f);

    // Third row
    result.m[8] = SimdFloat(2.0f) * (qxqz - qyqw);
    result.m[9] = SimdFloat(2.0f) * (qyqz + qxqw);
    result.m[10] = qw2 - qx2 - qy2 + qz2;
    result.m[11] = SimdFloat(0.0f);

    // Fourth row
    result.m[12] = SimdFloat(0.0f);
    result.m[13] = SimdFloat(0.0f);
    result.m[14] = SimdFloat(0.0f);
    result.m[15] = SimdFloat(1.0f);

    return result;
}

Mat4SIMD Mat4SIMD::scale(const SimdFloat& x, const SimdFloat& y, const SimdFloat& z) {
    Mat4SIMD result = identity();

    result.m[0] = x;  // m[0]
    result.m[5] = y;  // m[5]
    result.m[10] = z;  // m[10]

    return result;
}

#endif

// SimdMatrixOps namespace implementation
namespace SimdMatrixOps {
    // Matrix multiplication with SIMD
    void multiplyBatch(const glm::mat4* matrices_a, const glm::mat4* matrices_b,
        glm::mat4* results, int count) {

        for (int i = 0; i < count; ++i) {
            results[i] = matrices_a[i] * matrices_b[i];
        }

        // Note: A full SIMD implementation would use the SIMD types
        // defined earlier to process multiple matrices in parallel
    }

    // Transform vectors in batches
    void transformPoints(const glm::mat4* matrices, const glm::vec3* points,
        glm::vec3* results, int count) {

        for (int i = 0; i < count; ++i) {
            glm::vec4 homogeneous(points[i], 1.0f);
            homogeneous = matrices[i] * homogeneous;
            results[i] = glm::vec3(homogeneous) / homogeneous.w;
        }
    }

    // Batch transform updates by hierarchy depth
    void updateTransformsByDepth(const glm::mat4* local, const glm::mat4* parent,
        glm::mat4* world, const int* parent_indices,
        int count, int depth) {

        for (int i = 0; i < count; ++i) {
            int parent_idx = parent_indices[i];
            if (parent_idx >= 0) {
                world[i] = parent[parent_idx] * local[i];
            }
            else {
                world[i] = local[i];
            }
        }
    }
}

// BitMaskOps namespace implementation
namespace BitMaskOps {
    // Find first set bit
    int findFirstSetBit(uint64_t mask) {
        if (mask == 0) return -1;

#if defined(_MSC_VER) && defined(_WIN64)
        unsigned long index;
        _BitScanForward64(&index, mask);
        return static_cast<int>(index);
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_ctzll(mask);
#else
        // Fallback implementation
        for (int i = 0; i < 64; ++i) {
            if (mask & (1ULL << i)) {
                return i;
            }
        }
        return -1;
#endif
    }

    // Count set bits
    int countSetBits(uint64_t mask) {
#if defined(_MSC_VER) && defined(_WIN64)
        return static_cast<int>(__popcnt64(mask));
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_popcountll(mask);
#else
        // Fallback implementation
        int count = 0;
        while (mask) {
            count += mask & 1;
            mask >>= 1;
        }
        return count;
#endif
    }

    // Process entities using bitmasks to avoid branches
    void processEntityBitMasked(void* entity_data, uint64_t mask,
        void (*process_func)(void* data, int entity_idx), int base_idx) {

        while (mask) {
            int bit_idx = findFirstSetBit(mask);
            process_func(entity_data, base_idx + bit_idx);
            mask &= ~(1ULL << bit_idx);
        }
    }
}

// Core Entity System Components
//------------------------------

// AABB implementation
AABB::AABB() : min(0), max(0) {}

AABB::AABB(const glm::vec3& min, const glm::vec3& max) : min(min), max(max) {}

bool AABB::contains(const AABB& other) const {
    return
        min.x <= other.min.x && min.y <= other.min.y && min.z <= other.min.z &&
        max.x >= other.max.x && max.y >= other.max.y && max.z >= other.max.z;
}

bool AABB::overlaps(const AABB& other) const {
    return
        min.x <= other.max.x && max.x >= other.min.x &&
        min.y <= other.max.y && max.y >= other.min.y &&
        min.z <= other.max.z && max.z >= other.min.z;
}

glm::vec3 AABB::center() const {
    return (min + max) * 0.5f;
}

glm::vec3 AABB::extents() const {
    return (max - min) * 0.5f;
}

float AABB::volume() const {
    glm::vec3 diff = max - min;
    return diff.x * diff.y * diff.z;
}



// That completes the implementation of the core engine components