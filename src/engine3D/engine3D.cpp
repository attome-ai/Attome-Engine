#include <engine3D/ATMEngine.h>
#include "engine3D/ATMPipeline.h"
#include <engine3D/ATMPipeline.h>
// Constructor
Engine::Engine(int window_width, int window_height,
    float world_size_x, float world_size_y, float world_size_z,
    float cell_size, uint32_t max_entities)
    : delta_time(0.0f), total_time(0.0f), last_frame_time(0), render_batch_count(0)
{
    // Initialize world bounds
    world_min = glm::vec3(-world_size_x / 2.0f, -world_size_y / 2.0f, -world_size_z / 2.0f);
    world_max = glm::vec3(world_size_x / 2.0f, world_size_y / 2.0f, world_size_z / 2.0f);
    world_cell_size = cell_size;

    // Initialize SDL and create window
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Error initializing SDL: %s\n", SDL_GetError());
        window = nullptr;
        device = nullptr;
        return;
    }

    window = SDL_CreateWindow("High Performance Engine",
        window_width, window_height, SDL_WINDOW_RESIZABLE);

    if (!window) {
        fprintf(stderr, "Error creating window: %s\n", SDL_GetError());
        device = nullptr;
        return;
    }

    // Create GPU device with debug mode in development
    SDL_PropertiesID deviceProps = SDL_CreateProperties();
    SDL_SetBooleanProperty(deviceProps, SDL_PROP_GPU_DEVICE_CREATE_DEBUGMODE_BOOLEAN, true);
    SDL_SetBooleanProperty(deviceProps, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_SPIRV_BOOLEAN, true);

    device = SDL_CreateGPUDeviceWithProperties(deviceProps);
    SDL_DestroyProperties(deviceProps);

    if (!device) {
        fprintf(stderr, "Error creating GPU device: %s\n", SDL_GetError());
        return;
    }

    // Create task system with hardware concurrency
    task_system = new TaskSystem();

    // Create engine subsystems with specified capacities
    entity_types = new EntityTypeRegistry(MAX_ENTITY_TYPES);
    entities = new EntityStorage(max_entities, MAX_COMPONENTS);
    transforms = new TransformData(max_entities);
    physics = new PhysicsData(max_entities);
    render_data = new RenderData(max_entities);
    hierarchy = new HierarchyData(max_entities);

    // Create spatial grid
    spatial_grid = new MortonGrid(max_entities, world_min, world_max, world_cell_size);

    // Create camera
    camera = new Camera();
    camera->setPosition(glm::vec3(0.0f, 0.0f, 10.0f));
    camera->setTarget(glm::vec3(0.0f, 0.0f, 0.0f));
    camera->setProjection(60.0f, (float)window_width / (float)window_height, 0.1f, 1000.0f);

    // Create GPU resources
    gpu_resources = new GPUResources(device, 1024, 2048, 128, 1024, 1024, 128);
    buffer_pools = new GPUBufferPools(device, 512);
    gpu_renderer = new GPURenderer(device, window);

    // Create frame allocator (16MB arenas)
    frame_allocator = new FrameArenaAllocator(2, 16 * 1024 * 1024);

    // Allocate render batches
    max_render_batches = MAX_RENDER_BATCHES;
    render_batches = static_cast<RenderBatch**>(
        SDL_aligned_alloc(CACHE_LINE_SIZE, max_render_batches * sizeof(RenderBatch*)));

    // Initialize stats
    memset(&stats, 0, sizeof(EngineStats));

    // Initialize update context
    update_context.entity_type = nullptr;
    update_context.entities = entities;
    update_context.transforms = transforms;
    update_context.physics = physics;
    update_context.render_data = render_data;
    update_context.delta_time = 0.0f;
    update_context.total_time = 0.0f;
    update_context.thread_id = 0;
    update_context.thread_data = nullptr;
}

// Destructor
Engine::~Engine() {
    shutdown();

    // Free resources in reverse order of allocation
    if (render_batches) {
        SDL_aligned_free(render_batches);
        render_batches = nullptr;
    }

    delete frame_allocator;
    delete gpu_renderer;
    delete buffer_pools;
    delete gpu_resources;
    delete camera;
    delete spatial_grid;
    delete hierarchy;
    delete render_data;
    delete physics;
    delete transforms;
    delete entities;
    delete entity_types;
    delete task_system;

    if (window) {
        SDL_DestroyWindow(window);
    }

    SDL_Quit();
}

// Initialize the engine
bool Engine::initialize() {
    if (!device || !window) {
        return false;
    }

    // Claim the window for our GPU device
    if (!SDL_ClaimWindowForGPUDevice(device, window)) {
        fprintf(stderr, "Failed to claim window for GPU device: %s\n", SDL_GetError());
        return false;
    }

    // Set up swapchain parameters
    if (!SDL_SetGPUSwapchainParameters(device, window,
        SDL_GPU_SWAPCHAINCOMPOSITION_SDR,SDL_GPU_PRESENTMODE_MAILBOX)) {
        fprintf(stderr, "Failed to set GPU swapchain parameters: %s\n", SDL_GetError());
        return false;
    }

    // Set frames in flight
    if (!SDL_SetGPUAllowedFramesInFlight(device, 2)) {
        fprintf(stderr, "Failed to set allowed frames in flight: %s\n", SDL_GetError());
        return false;
    }

    // Initialize renderer
    if (!gpu_renderer->initialize()) {
        fprintf(stderr, "Failed to initialize GPU renderer\n");
        return false;
    }
    
    std::vector<SDL_GPUGraphicsPipeline*> pips = createPipelines(device);
    for(int i = 0; i < pips.size(); i++)
    {
        this->gpu_resources->addGraphicsPipeline(pips[i]);
    }
    // Initialize task system with hardware concurrency
    task_system->initialize(std::thread::hardware_concurrency());

    // Initialize timing
    last_frame_time = SDL_GetTicks();
    delta_time = 0.016f; // Initial frame time estimate (60 FPS)
    total_time = 0.0f;

    return true;
}
void Engine::resetFrameAllocator() {
    if (frame_allocator) {
        frame_allocator->resetAll();
    }
}
// Update engine state
void Engine::update() {
    // Calculate delta time
    uint64_t current_time = SDL_GetTicks();
    delta_time = (current_time - last_frame_time) / 1000.0f;
    last_frame_time = current_time;
    total_time += delta_time;

    // Cap delta time to avoid large jumps
    if (delta_time > 0.1f) delta_time = 0.1f;

    // Update context time information
    update_context.delta_time = delta_time;
    update_context.total_time = total_time;

    // Reset frame allocator
    resetFrameAllocator();

    // Timing measurements
    uint64_t update_start = SDL_GetTicks();
    uint64_t phase_start = update_start;

    createRenderBatches();
    // Update transforms (local to world)
    updateTransforms();

    // Update physics
    phase_start = SDL_GetTicks();
    updatePhysics();
    stats.physics_ms = (SDL_GetTicks() - phase_start) / 1000.0f;

    // Update entity logic
    updateEntityLogic();

    // Update visibility (frustum culling)
    updateVisibility();

    // Update render batches
    updateBatches();

    // Update spatial grid
    spatial_grid->rebuild(transforms, entities->entity_count);

    // Calculate total update time
    stats.update_ms = (SDL_GetTicks() - update_start) / 1000.0f;

    // Update engine statistics
    stats.active_entities = entities->entity_count;
}

// Render the scene
void Engine::render() {
    uint64_t render_start = SDL_GetTicks();

    // Begin frame
    gpu_renderer->beginFrame();

    // Update global uniforms

    // Prepare render commands
    prepareRenderCommands();

    // Begin render pass
    gpu_renderer->beginRenderPass();

    // Execute render commands
    gpu_renderer->executeRenderCommands(gpu_resources);

    // End render pass
    gpu_renderer->endRenderPass();

    // End frame (present to screen)
    gpu_renderer->endFrame();

    // Update render stats
    stats.render_ms = (SDL_GetTicks() - render_start) / 1000.0f;
    stats.frame_ms = stats.update_ms + stats.render_ms;
}

// Shutdown the engine
void Engine::shutdown() {
    // Wait for GPU to idle before cleanup
    SDL_WaitForGPUIdle(device);

    // Stop task system first
    if (task_system) {
        task_system->shutdown();
    }

    // Clean up render batches
    for (uint32_t i = 0; i < render_batch_count; ++i) {
        if (render_batches[i]) {
            delete render_batches[i];
            render_batches[i] = nullptr;
        }
    }
    render_batch_count = 0;

    // Shutdown GPU renderer
    if (gpu_renderer) {
        gpu_renderer->shutdown();
    }

    // Clean up GPU resources
    if (gpu_resources) {
        gpu_resources->destroyResources();
    }

    if (buffer_pools) {
        buffer_pools->destroyPools();
    }

    // Release window from GPU device
    if (device && window) {
        SDL_ReleaseWindowFromGPUDevice(device, window);
    }

    // Destroy GPU device
    if (device) {
        SDL_DestroyGPUDevice(device);
        device = nullptr;
    }
}

// Create and register a new entity type
uint32_t Engine::createEntityType() {
    EntityType type;
    return entity_types->registerType(std::move(type));
}

// Finalize entity type system
void Engine::finalizeEntityTypes() {
    // Sort entities by type
    entities->sortEntitiesByType();
}

// Create an entity instance
uint32_t Engine::createEntityInstance(uint32_t type_id,
    const glm::vec3& position, const glm::quat& rotation, const glm::vec3& scale) {

    // Create entity
    uint32_t entity_idx = entities->createEntity(type_id);

    // Set transform
    transforms->setLocalPosition(entity_idx, position);
    transforms->setLocalRotation(entity_idx, rotation);
    transforms->setLocalScale(entity_idx, scale);

    // Insert into spatial grid
    spatial_grid->insertEntity(entity_idx, position);

    return entity_idx;
}

// Set parent relationship for an entity
void Engine::setParent(uint32_t entity_idx, uint32_t parent_idx) {
    hierarchy->setParent(entity_idx, parent_idx);
}

// Finalize the hierarchy
void Engine::finalizeHierarchy() {
    // Update hierarchy depths
    hierarchy->updateDepths();

    // Sort entities by depth and type for efficient updates
    hierarchy->sortByDepthAndType(entities->type_ids, entities->entity_count);
}

// Transform operations
void Engine::setPosition(uint32_t entity_idx, const glm::vec3& position) {
    transforms->setLocalPosition(entity_idx, position);

    // Update spatial grid
    spatial_grid->updateEntity(entity_idx, position);
}

void Engine::setRotation(uint32_t entity_idx, const glm::quat& rotation) {
    transforms->setLocalRotation(entity_idx, rotation);
}

void Engine::setScale(uint32_t entity_idx, const glm::vec3& scale) {
    transforms->setLocalScale(entity_idx, scale);
}

glm::vec3 Engine::getPosition(uint32_t entity_idx) const {
    return transforms->getWorldPosition(entity_idx);
}

glm::quat Engine::getRotation(uint32_t entity_idx) const {
    return transforms->getWorldRotation(entity_idx);
}

glm::vec3 Engine::getScale(uint32_t entity_idx) const {
    return transforms->getWorldScale(entity_idx);
}

// Rendering operations
void Engine::setEntityMesh(uint32_t entity_idx, int32_t mesh_id) {
    render_data->setMesh(entity_idx, mesh_id);
}

void Engine::setEntityMaterial(uint32_t entity_idx, int32_t material_id) {
    render_data->setMaterial(entity_idx, material_id);
}

void Engine::setEntityShader(uint32_t entity_idx, int32_t shader_id) {
    render_data->setShader(entity_idx, shader_id);
}

void Engine::setEntityVisibility(uint32_t entity_idx, bool visible) {
    render_data->setVisibility(entity_idx, visible);
}

// Physics operations
void Engine::setEntityVelocity(uint32_t entity_idx, const glm::vec3& velocity) {
    physics->setVelocity(entity_idx, velocity);
}

void Engine::setEntityCollisionLayer(uint32_t entity_idx, uint32_t layer) {
    physics->setCollisionLayer(entity_idx, layer);
}

void Engine::setEntityCollisionMask(uint32_t entity_idx, uint32_t mask) {
    physics->setCollisionMask(entity_idx, mask);
}

void Engine::setEntityBounds(uint32_t entity_idx, const AABB& bounds) {
    physics->setBounds(entity_idx, bounds);
}

// Camera control
void Engine::setCameraPosition(const glm::vec3& position) {
    camera->setPosition(position);
}

void Engine::setCameraTarget(const glm::vec3& target) {
    camera->setTarget(target);
}

void Engine::setCameraFov(float fov_degrees) {
    float aspect = camera->aspect_ratio;
    float near_z = camera->near_plane;
    float far_z = camera->far_plane;
    camera->setProjection(fov_degrees, aspect, near_z, far_z);
}

// Resource management
int32_t Engine::addTexture(SDL_Surface* surface) {
    return gpu_resources->createTexture(surface);
}

int32_t Engine::addMesh(const Vertex* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t index_count) {
    return gpu_resources->createMesh(vertices, vertex_count, indices, index_count);
}

int32_t Engine::addMaterial(int32_t diffuse_texture, int32_t normal_texture,
    int32_t specular_texture, int32_t sampler_id,
    const glm::vec4& diffuse_color, const glm::vec4& specular_color,
    float shininess) {
    return gpu_resources->createMaterial(diffuse_texture, normal_texture,
        specular_texture, sampler_id, diffuse_color, specular_color, shininess);
}

int32_t Engine::addGraphicsPipeline(SDL_GPUGraphicsPipeline* pipeline)
{
    return gpu_resources->addGraphicsPipeline(pipeline);

}




// Update transforms
void Engine::updateTransforms() {
    // Update local matrices for all entities
    transforms->updateLocalMatrices(0, entities->entity_count);

    // Update world matrices based on hierarchy
    transforms->updateWorldMatricesHierarchical(
        hierarchy->parent_indices,
        hierarchy->depth_first_indices,
        entities->entity_count,
        0,
        MAX_HIERARCHY_DEPTH
    );
}

// Update physics
void Engine::updatePhysics() {
    // Use work batching for physics updates
    const uint32_t BATCH_SIZE = 1024;
    uint32_t entity_count = entities->entity_count;
    uint32_t batch_count = (entity_count + BATCH_SIZE - 1) / BATCH_SIZE;

    // Submit physics update tasks
    for (uint32_t i = 0; i < batch_count; ++i) {
        uint32_t start_idx = i * BATCH_SIZE;
        uint32_t count = std::min(BATCH_SIZE, entity_count - start_idx);

        task_system->submitTask([this, start_idx, count](uint32_t thread_id) {
            physics->updatePositions(transforms, delta_time, start_idx, count);
            });
    }

    // Wait for physics updates to complete
    task_system->waitForAll();

    // Detect collisions
    physics->detectCollisions(entities->entity_ids, entity_count);
}

// Update entity logic
void Engine::updateEntityLogic() {
    // Process entities by type
    for (uint32_t type_idx = 0; type_idx < entity_types->type_count; ++type_idx) {
        EntityType* type = entity_types->getType(type_idx);
        if (!type || !type->update_func) continue;

        uint32_t start_idx = entities->type_start_indices[type_idx];
        uint32_t count = entities->type_counts[type_idx];

        if (count == 0) continue;

        // Set current entity type for context
        update_context.entity_type = type;

        // Batch size based on entity type
        uint32_t batch_size = std::min((uint32_t)ENTITY_BATCH_SIZE, count);
        uint32_t batch_count = (count + batch_size - 1) / batch_size;

        // Submit tasks for parallel processing
        for (uint32_t i = 0; i < batch_count; ++i) {
            uint32_t batch_start = start_idx + i * batch_size;
            uint32_t batch_count = std::min(batch_size, start_idx + count - batch_start);

            task_system->submitTask([this, type, batch_start, batch_count](uint32_t thread_id) {
                update_context.thread_id = thread_id;
                type->update_func(&update_context, batch_start, batch_count);
                });
        }
    }

    // Wait for all entity updates to complete
    task_system->waitForAll();
}

// Update entity visibility
void Engine::updateVisibility() {
    const uint32_t BATCH_SIZE = 1024;
    uint32_t entity_count = entities->entity_count;
    uint32_t batch_count = (entity_count + BATCH_SIZE - 1) / BATCH_SIZE;

    // Allocate visibility results
    bool* visibility_results = static_cast<bool*>(
        frameAlloc(entity_count * sizeof(bool), CACHE_LINE_SIZE));

    // Update camera frustum
    camera->updateMatrices();
    camera->updateFrustumPlanes();

    // Submit frustum culling tasks
    for (uint32_t i = 0; i < batch_count; ++i) {
        uint32_t start_idx = i * BATCH_SIZE;
        uint32_t count = std::min(BATCH_SIZE, entity_count - start_idx);

        task_system->submitTask([this, start_idx, count, visibility_results](uint32_t thread_id) {
            // Get entity AABBs for this batch
            AABB* batch_aabbs = static_cast<AABB*>(
                TaskSystem::getThreadFrameAllocator()->allocate(
                    count * sizeof(AABB), CACHE_LINE_SIZE));

            // Extract AABBs from entity world transforms and physics bounds
            for (uint32_t j = 0; j < count; ++j) {
                uint32_t entity_idx = start_idx + j;
                batch_aabbs[j] = physics->bounds[entity_idx].transform(
                    transforms->world_matrices[entity_idx]);
            }

            // Perform SIMD batch culling
            camera->cullAABBsBatch(batch_aabbs, &visibility_results[start_idx], count);
            });
    }

    // Wait for culling to complete
    task_system->waitForAll();

    // Apply visibility results
    for (uint32_t i = 0; i < entity_count; ++i) {
        if (render_data->isVisible(i)) {
            render_data->setVisibility(i, visibility_results[i]);
        }
    }
}

// Update render batches
void Engine::updateBatches() {
    // Process each batch
    for (uint32_t i = 0; i < render_batch_count; ++i) {
        RenderBatch* batch = render_batches[i];
        if (!batch || batch->instance_count == 0) continue;

        // Update instance data from transforms
        batch->updateInstanceData(transforms, 0, batch->instance_count);

        // Upload to GPU if dirty
        if (batch->buffer_dirty) {
            batch->uploadToGPU(gpu_resources);
        }
    }
}

// Prepare render commands
void Engine::prepareRenderCommands() {
    // Clear previous commands
    gpu_renderer->clearRenderCommands();

    // Add render commands for each batch
    for (uint32_t i = 0; i < render_batch_count; ++i) {
        RenderBatch* batch = render_batches[i];
        if (!batch || batch->instance_count == 0) continue;

        RenderCommand cmd;
        cmd.mesh_id = batch->mesh_id;
        cmd.material_id = batch->material_id;
        cmd.shader_id = batch->shader_id;
        cmd.first_instance = 0;
        cmd.instance_count = batch->instance_count;
        cmd.sort_key = batch->sort_key;
        gpu_renderer->addRenderCommand(cmd);
    }

    // Sort commands for efficient rendering
    gpu_renderer->sortRenderCommands();

    // Update stats
    stats.draw_calls = gpu_renderer->render_command_count;
    stats.render_batches = render_batch_count;

    // Calculate triangles rendered
    uint32_t triangle_count = 0;
    for (uint32_t i = 0; i < gpu_renderer->render_command_count; ++i) {
        const RenderCommand& cmd = gpu_renderer->render_commands[i];
        if (cmd.mesh_id >= 0 && cmd.mesh_id < (int32_t)gpu_resources->mesh_count) {
            const Mesh& mesh = gpu_resources->meshes[cmd.mesh_id];
            triangle_count += (mesh.index_count / 3) * cmd.instance_count;
        }
    }
    stats.triangles_rendered = triangle_count;
}

// Create render batches
void Engine::createRenderBatches() {
    // Clear existing batches
    for (uint32_t i = 0; i < render_batch_count; ++i) {
        if (render_batches[i]) {
            delete render_batches[i];
            render_batches[i] = nullptr;
        }
    }
    render_batch_count = 0;

    // Group entities by shader, mesh, and material
    std::unordered_map<uint64_t, std::vector<uint32_t>> batch_map;

    for (uint32_t i = 0; i < entities->entity_count; ++i) {
        if (!render_data->isVisible(i)) continue;

        int32_t mesh_id = render_data->getMesh(i);
        int32_t material_id = render_data->getMaterial(i);
        int32_t shader_id = render_data->getShader(i);

        if (mesh_id < 0 || material_id < 0 || shader_id < 0) continue;

        // Create sort key
        uint64_t sort_key = RenderCommand::createSortKey(shader_id, mesh_id, material_id, false);

        // Add to batch map
        batch_map[sort_key].push_back(i);
    }

    // Create batches from map
    for (const auto& entry : batch_map) {
        uint64_t sort_key = entry.first;
        const std::vector<uint32_t>& entities = entry.second;

        if (entities.empty()) continue;

        // Extract shader, mesh, material from sort key
        int32_t shader_id = (sort_key >> 48) & 0xFFFF;
        int32_t mesh_id = (sort_key >> 32) & 0xFFFF;
        int32_t material_id = (sort_key >> 16) & 0xFFFF;

        // Create batch with appropriate capacity
        RenderBatch* batch = new RenderBatch(entities.size());
        batch->shader_id = shader_id;
        batch->mesh_id = mesh_id;
        batch->material_id = material_id;
        batch->sort_key = sort_key;

        // Add entities to batch
        for (uint32_t entity_idx : entities) {
            batch->addEntity(entity_idx);
        }

        // Update instance data
        batch->updateInstanceData(transforms, 0, batch->instance_count);

        // Add to batch array
        if (render_batch_count < max_render_batches) {
            render_batches[render_batch_count++] = batch;
        }
        else {
            delete batch; // Too many batches, discard
        }
    }
}

// Optimize existing batches
void Engine::optimizeBatches() {
    // Re-create batches completely
    // This is more efficient than trying to incrementally update
    createRenderBatches();
}

// Update GPU resources
void Engine::updateGPUResources() {
    // Update cached pointers
    gpu_resources->updatePointerCaches();
}


// Frame allocator methods
void* Engine::frameAlloc(size_t size, size_t alignment) {
    return frame_allocator->allocate(size, alignment);
}




