#include <engine3D/ATMEngine.h>
MortonGrid::MortonGrid(uint32_t max_entities, const glm::vec3& world_min,
    const glm::vec3& world_max, float cell_size)
    : max_entities(max_entities), world_min(world_min), world_max(world_max), cell_size(cell_size)
{
    // Calculate grid dimensions based on world bounds and cell size
    glm::vec3 world_size = world_max - world_min;
    uint32_t grid_size_x = static_cast<uint32_t>(std::ceil(world_size.x / cell_size));
    uint32_t grid_size_y = static_cast<uint32_t>(std::ceil(world_size.y / cell_size));
    uint32_t grid_size_z = static_cast<uint32_t>(std::ceil(world_size.z / cell_size));

    // Ensure we don't exceed the defined grid size limit
    grid_size_x = std::min(grid_size_x, static_cast<uint32_t>(MORTON_GRID_SIZE));
    grid_size_y = std::min(grid_size_y, static_cast<uint32_t>(MORTON_GRID_SIZE));
    grid_size_z = std::min(grid_size_z, static_cast<uint32_t>(MORTON_GRID_SIZE));

    // Calculate total cell count
    cell_count = grid_size_x * grid_size_y * grid_size_z;

    // Allocate memory for cells with cache-line alignment
    cells = static_cast<Cell*>(SDL_aligned_alloc(CACHE_LINE_SIZE, cell_count * sizeof(Cell)));

    // Initialize cells
    for (uint32_t i = 0; i < cell_count; ++i) {
        cells[i].entity_indices = static_cast<uint32_t*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
            SPATIAL_CELL_CAPACITY * sizeof(uint32_t)));
        cells[i].capacity = SPATIAL_CELL_CAPACITY;
        cells[i].count = 0;
    }

    // Allocate memory for morton codes and optimization arrays
    morton_codes = static_cast<uint64_t*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
        max_entities * sizeof(uint64_t)));
    cell_start_indices = static_cast<uint32_t*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
        cell_count * sizeof(uint32_t)));
    sorted_entity_indices = static_cast<uint32_t*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
        max_entities * sizeof(uint32_t)));

    // Initialize memory to zero
    std::memset(morton_codes, 0, max_entities * sizeof(uint64_t));
    std::memset(cell_start_indices, 0, cell_count * sizeof(uint32_t));
    std::memset(sorted_entity_indices, 0, max_entities * sizeof(uint32_t));
}

MortonGrid::~MortonGrid()
{
    // Free cell entity indices
    for (uint32_t i = 0; i < cell_count; ++i) {
        SDL_aligned_free(cells[i].entity_indices);
    }

    // Free main arrays
    SDL_aligned_free(cells);
    SDL_aligned_free(morton_codes);
    SDL_aligned_free(cell_start_indices);
    SDL_aligned_free(sorted_entity_indices);
}

// Helper functions for bit interleaving (Morton encoding)
inline uint64_t expandBits(uint32_t v) {
    // Spread bits to create space for interleaving
    uint64_t x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    return x;
}

uint64_t MortonGrid::calculateMortonCode(const glm::vec3& position) const
{
    // Normalize position within grid bounds
    glm::vec3 normalized = (position - world_min) / (world_max - world_min);

    // Clamp normalized position to [0,1]
    normalized = glm::clamp(normalized, glm::vec3(0.0f), glm::vec3(1.0f));

    // Scale to grid size
    uint32_t x = static_cast<uint32_t>(normalized.x * (MORTON_GRID_SIZE - 1));
    uint32_t y = static_cast<uint32_t>(normalized.y * (MORTON_GRID_SIZE - 1));
    uint32_t z = static_cast<uint32_t>(normalized.z * (MORTON_GRID_SIZE - 1));

    // Interleave bits using the expandBits function
    uint64_t xx = expandBits(x);
    uint64_t yy = expandBits(y);
    uint64_t zz = expandBits(z);

    // Combine interleaved bits to form the Morton code
    return xx | (yy << 1) | (zz << 2);
}

uint64_t MortonGrid::encodeMorton(const glm::vec3& position) const
{
    return calculateMortonCode(position);
}

void MortonGrid::insertEntity(uint32_t entity_idx, const glm::vec3& position)
{
    // Calculate morton code for the entity
    uint64_t code = encodeMorton(position);
    morton_codes[entity_idx] = code;

    // Calculate cell index from morton code
    uint32_t cell_idx = static_cast<uint32_t>(code % cell_count);

    // Add entity to cell if there's space
    if (cells[cell_idx].count < cells[cell_idx].capacity) {
        cells[cell_idx].entity_indices[cells[cell_idx].count++] = entity_idx;
    }
    else {
        // Double capacity if needed
        uint32_t new_capacity = cells[cell_idx].capacity * 2;
        uint32_t* new_indices = static_cast<uint32_t*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
            new_capacity * sizeof(uint32_t)));

        // Copy existing indices
        std::memcpy(new_indices, cells[cell_idx].entity_indices,
            cells[cell_idx].count * sizeof(uint32_t));

        // Add new entity
        new_indices[cells[cell_idx].count++] = entity_idx;

        // Free old indices and update pointers
        SDL_aligned_free(cells[cell_idx].entity_indices);
        cells[cell_idx].entity_indices = new_indices;
        cells[cell_idx].capacity = new_capacity;
    }
}

void MortonGrid::updateEntity(uint32_t entity_idx, const glm::vec3& position)
{
    // Calculate new morton code
    uint64_t new_code = encodeMorton(position);
    uint64_t old_code = morton_codes[entity_idx];

    // If code hasn't changed, no need to update cell assignment
    if (new_code == old_code) {
        return;
    }

    // Calculate old cell index
    uint32_t old_cell_idx = static_cast<uint32_t>(old_code % cell_count);

    // Remove entity from old cell
    for (uint32_t i = 0; i < cells[old_cell_idx].count; ++i) {
        if (cells[old_cell_idx].entity_indices[i] == entity_idx) {
            // Replace with last entity and decrement count for O(1) removal
            cells[old_cell_idx].entity_indices[i] =
                cells[old_cell_idx].entity_indices[--cells[old_cell_idx].count];
            break;
        }
    }

    // Update morton code
    morton_codes[entity_idx] = new_code;

    // Calculate new cell index
    uint32_t new_cell_idx = static_cast<uint32_t>(new_code % cell_count);

    // Add entity to new cell
    if (cells[new_cell_idx].count < cells[new_cell_idx].capacity) {
        cells[new_cell_idx].entity_indices[cells[new_cell_idx].count++] = entity_idx;
    }
    else {
        // Resize cell if needed
        uint32_t new_capacity = cells[new_cell_idx].capacity * 2;
        uint32_t* new_indices = static_cast<uint32_t*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
            new_capacity * sizeof(uint32_t)));

        std::memcpy(new_indices, cells[new_cell_idx].entity_indices,
            cells[new_cell_idx].count * sizeof(uint32_t));
        new_indices[cells[new_cell_idx].count++] = entity_idx;

        SDL_aligned_free(cells[new_cell_idx].entity_indices);
        cells[new_cell_idx].entity_indices = new_indices;
        cells[new_cell_idx].capacity = new_capacity;
    }
}

void MortonGrid::removeEntity(uint32_t entity_idx)
{
    // Calculate cell index
    uint32_t cell_idx = static_cast<uint32_t>(morton_codes[entity_idx] % cell_count);

    // Remove entity from cell
    for (uint32_t i = 0; i < cells[cell_idx].count; ++i) {
        if (cells[cell_idx].entity_indices[i] == entity_idx) {
            // Replace with last entity and decrement count
            cells[cell_idx].entity_indices[i] =
                cells[cell_idx].entity_indices[--cells[cell_idx].count];
            break;
        }
    }

    // Clear morton code
    morton_codes[entity_idx] = 0;
}

uint32_t MortonGrid::queryRadius(const glm::vec3& center, float radius,
    uint32_t* result_indices, uint32_t max_results)
{
    // Create an AABB that encompasses the sphere for initial broad phase
    AABB query_box;
    query_box.min = center - glm::vec3(radius);
    query_box.max = center + glm::vec3(radius);

    // Get potential candidates from the AABB query
    uint32_t* temp_results = static_cast<uint32_t*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
        max_results * sizeof(uint32_t)));
    uint32_t candidate_count = queryBox(query_box, temp_results, max_results);

    // Now perform the actual radius test (narrow phase)
    uint32_t result_count = 0;
    float radius_squared = radius * radius;

    for (uint32_t i = 0; i < candidate_count && result_count < max_results; ++i) {
        uint32_t entity_idx = temp_results[i];

        // In a real system, we would get the actual entity position from TransformData
        // Here we approximate using the cell center for demonstration

        // Decode the Morton code to get an approximate position
        uint64_t code = morton_codes[entity_idx];
        uint32_t x = 0, y = 0, z = 0;

        // Extract interleaved bits (de-interleaving the Morton code)
        for (uint32_t bit = 0; bit < 21; ++bit) {
            x |= ((code >> (3 * bit)) & 1) << bit;
            y |= ((code >> (3 * bit + 1)) & 1) << bit;
            z |= ((code >> (3 * bit + 2)) & 1) << bit;
        }

        // Convert to world space
        glm::vec3 pos = world_min + glm::vec3(
            static_cast<float>(x) / (MORTON_GRID_SIZE - 1),
            static_cast<float>(y) / (MORTON_GRID_SIZE - 1),
            static_cast<float>(z) / (MORTON_GRID_SIZE - 1)
        ) * (world_max - world_min);

        // Check distance
        float dist_squared = glm::distance(glm::vec2(center), glm::vec2(pos));
        if (dist_squared <= radius_squared) {
            result_indices[result_count++] = entity_idx;
        }
    }

    SDL_aligned_free(temp_results);
    return result_count;
}

uint32_t MortonGrid::queryBox(const AABB& box, uint32_t* result_indices, uint32_t max_results)
{
    uint32_t result_count = 0;

    // Convert box to normalized grid coordinates
    glm::vec3 min_normalized = (box.min - world_min) / (world_max - world_min);
    glm::vec3 max_normalized = (box.max - world_min) / (world_max - world_min);

    // Clamp to grid bounds
    min_normalized = glm::clamp(min_normalized, glm::vec3(0.0f), glm::vec3(1.0f));
    max_normalized = glm::clamp(max_normalized, glm::vec3(0.0f), glm::vec3(1.0f));

    // Calculate grid cell ranges
    uint32_t min_x = static_cast<uint32_t>(min_normalized.x * (MORTON_GRID_SIZE - 1));
    uint32_t min_y = static_cast<uint32_t>(min_normalized.y * (MORTON_GRID_SIZE - 1));
    uint32_t min_z = static_cast<uint32_t>(min_normalized.z * (MORTON_GRID_SIZE - 1));

    uint32_t max_x = static_cast<uint32_t>(max_normalized.x * (MORTON_GRID_SIZE - 1));
    uint32_t max_y = static_cast<uint32_t>(max_normalized.y * (MORTON_GRID_SIZE - 1));
    uint32_t max_z = static_cast<uint32_t>(max_normalized.z * (MORTON_GRID_SIZE - 1));

    // Check cells within the box range
    for (uint32_t z = min_z; z <= max_z; ++z) {
        for (uint32_t y = min_y; y <= max_y; ++y) {
            for (uint32_t x = min_x; x <= max_x; ++x) {
                // Calculate Morton code for this cell position
                uint64_t code = 0;

                uint64_t xx = expandBits(x);
                uint64_t yy = expandBits(y);
                uint64_t zz = expandBits(z);
                code = xx | (yy << 1) | (zz << 2);

                // Calculate cell index
                uint32_t cell_idx = static_cast<uint32_t>(code % cell_count);

                // Add all entities from this cell to results
                for (uint32_t i = 0; i < cells[cell_idx].count && result_count < max_results; ++i) {
                    result_indices[result_count++] = cells[cell_idx].entity_indices[i];
                }

                // Check if we've reached the maximum result count
                if (result_count >= max_results) {
                    return result_count;
                }
            }
        }
    }

    return result_count;
}

void MortonGrid::rebuild(const TransformData* transforms, uint32_t entity_count)
{
    // Reset all cells
    for (uint32_t i = 0; i < cell_count; ++i) {
        cells[i].count = 0;
    }

    // Temporary storage for sorting
    std::vector<std::pair<uint64_t, uint32_t>> sorted_pairs;
    sorted_pairs.reserve(entity_count);

    // Rebuild morton codes and cell assignments
    for (uint32_t i = 0; i < entity_count; ++i) {
        // Extract entity position from the transforms SoA data
        glm::vec3 position(
            transforms->world_pos_x[i],
            transforms->world_pos_y[i],
            transforms->world_pos_z[i]
        );

        // Calculate morton code
        uint64_t code = encodeMorton(position);
        morton_codes[i] = code;

        // Store for sorting
        sorted_pairs.push_back({ code, i });

        // Add to appropriate cell
        uint32_t cell_idx = static_cast<uint32_t>(code % cell_count);

        // Ensure cell has enough capacity
        if (cells[cell_idx].count >= cells[cell_idx].capacity) {
            uint32_t new_capacity = cells[cell_idx].capacity * 2;
            uint32_t* new_indices = static_cast<uint32_t*>(SDL_aligned_alloc(CACHE_LINE_SIZE,
                new_capacity * sizeof(uint32_t)));

            std::memcpy(new_indices, cells[cell_idx].entity_indices,
                cells[cell_idx].count * sizeof(uint32_t));

            SDL_aligned_free(cells[cell_idx].entity_indices);
            cells[cell_idx].entity_indices = new_indices;
            cells[cell_idx].capacity = new_capacity;
        }

        // Add entity to cell
        cells[cell_idx].entity_indices[cells[cell_idx].count++] = i;
    }

    // Sort entities by morton code for improved spatial locality during queries
    std::sort(sorted_pairs.begin(), sorted_pairs.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    // Fill sorted indices array
    for (uint32_t i = 0; i < entity_count; ++i) {
        sorted_entity_indices[i] = sorted_pairs[i].second;
    }

    // Calculate start indices for each cell to enable fast lookups
    uint32_t current_cell = 0;
    uint32_t current_code = 0;

    for (uint32_t i = 0; i < entity_count; ++i) {
        uint32_t cell_idx = static_cast<uint32_t>(sorted_pairs[i].first % cell_count);

        // If we've moved to a new cell, update all cells up to this point
        while (current_cell < cell_idx) {
            cell_start_indices[current_cell++] = i;
        }

        if (current_cell == cell_idx && current_code != sorted_pairs[i].first) {
            cell_start_indices[cell_idx] = i;
            current_code = sorted_pairs[i].first;
        }
    }

    // Fill in any remaining cells
    while (current_cell < cell_count) {
        cell_start_indices[current_cell++] = entity_count;
    }
}