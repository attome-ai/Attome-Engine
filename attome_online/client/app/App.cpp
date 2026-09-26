// App: client config, startup / shutdown, content upload, main loop.

#include "app/AppInternal.h"

namespace ao::client {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

bool loadClientConfig(const std::string &path, ClientConfig &cfg, std::string *error) {
  atm::Json root;
  if (!atm::Json::parseFile(path, root, error))
    return false;
  auto num = [&](const char *p, auto &out) {
    if (const atm::Json *v = root.findPath(p); v && v->isNumber())
      out = static_cast<std::remove_reference_t<decltype(out)>>(v->asNumber());
  };
  auto boolean = [&](const char *p, bool &out) {
    if (const atm::Json *v = root.findPath(p); v && v->isBool())
      out = v->asBool();
  };
  auto str = [&](const char *p, std::string &out) {
    if (const atm::Json *v = root.findPath(p); v && v->isString())
      out = v->asString();
  };
  str("client.name", cfg.name);
  str("client.host", cfg.host);
  num("client.port", cfg.port);
  boolean("client.local", cfg.local);
  str("client.serverConfig", cfg.serverConfigPath);
  num("client.mouseSensitivity", cfg.mouseSensitivity);
  boolean("client.invertY", cfg.invertY);
  num("window.width", cfg.windowWidth);
  num("window.height", cfg.windowHeight);
  boolean("window.fullscreen", cfg.fullscreen);
  boolean("render.vsync", cfg.render.vsync);
  num("render.msaa", cfg.render.msaaSamples);
  boolean("render.validation", cfg.render.validation);
  boolean("render.bloom", cfg.render.bloom);
  num("render.fov", cfg.render.fovYDegrees);
  num("render.viewDistanceChunks", cfg.viewRadiusChunks);
  num("audio.masterVolume", cfg.masterVolume);
  cfg.render.viewDistanceBlocks = float(cfg.viewRadiusChunks * atm::voxel::kChunkSize);
  return true;
}

// ---------------------------------------------------------------------------
// Lifetime
// ---------------------------------------------------------------------------

App::App() = default;
App::~App() { shutdown(); }

bool App::init(const ClientConfig &cfg, std::string *error) {
  cfg_ = cfg;
  if (cfg.profile) {
    showProfiler_ = true;
    atm::prof::Profiler::get().setEnabled(true);
  }

  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
    if (error)
      *error = std::string("SDL_Init failed: ") + SDL_GetError();
    return false;
  }
  sdlInit_ = true;

  SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
  if (cfg_.fullscreen)
    flags |= SDL_WINDOW_FULLSCREEN;
  window_ = SDL_CreateWindow("Attome Online", cfg_.windowWidth, cfg_.windowHeight, flags);
  if (!window_) {
    if (error)
      *error = std::string("SDL_CreateWindow failed: ") + SDL_GetError();
    return false;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext(); // before Renderer::init (its ImGui Vulkan backend needs it)
  imguiContext_ = true;
  ImGui::GetIO().IniFilename = nullptr;
  ImGui::StyleColorsDark();
  ui::init(); // fonts (assets/fonts) + theme
  look_.load(atm::resolve_path("config/graphics.json")); // saved graphics panel values, if any
  if (!cfg.render.vsync)
    look_.vsync = false; // --no-vsync / client.json win over the default

  // Content definitions (items, NPCs, map regions) from data/*.json.
  if (!loadGameData(atm::resolve_path("data"), error))
    return false;

  if (!renderer_.init(window_, cfg_.render, error))
    return false;
  if (!ImGui_ImplSDL3_InitForVulkan(window_)) {
    if (error)
      *error = "ImGui SDL3 backend init failed";
    return false;
  }
  imguiSdlInit_ = true;

  // Content
  blocks_.registerDefaults();
  anims_.buildDefaults();
  models_.buildDefaults();
  uploadMaterials();
  createModelMeshes();

  // Tunables (movement etc.) + live reload.
  std::string tunErr;
  if (!atm::Tunables::instance().loadFile(atm::resolve_path("config/game.json"), &tunErr))
    SDL_Log("[client] using default tunables (%s)", tunErr.c_str());

  // Audio
  if (audio_.init(32, cfg_.masterVolume)) {
    sfx_ = std::make_unique<SfxPlayer>();
    sfx_->init(audio_);
    sfx_->startAmbient();
  } else {
    SDL_Log("[client] audio unavailable, continuing silently");
  }

  // Input bindings (overridable from config/client.json "input" section).
  input_.bind("forward", "W");
  input_.bind("back", "S");
  input_.bind("left", "A");
  input_.bind("right", "D");
  input_.bind("jump", "Space");
  input_.bind("jump", "pad:a");
  input_.bind("dash", "Q");
  input_.bind("dash", "pad:b");
  input_.bind("sprint", "Left Ctrl");
  input_.bind("primary", "mouse:left");
  input_.bind("primary", "pad:rightshoulder");
  input_.bind("secondary", "mouse:right");
  input_.bind("secondary", "pad:leftshoulder");
  input_.bind("inventory", "Tab");
  input_.bind("inventory", "I");
  input_.bind("skills", "K");
  input_.bind("debug", "F3");
  input_.bind("debug", "F11");
  input_.bind("graphics", "F10");
  input_.bind("hitboxes", "F8");
  input_.bind("pickup", "E");
  input_.bind("map", "M");
  input_.bind("profiler", "F7");
  input_.bind("chat", "Return");
  input_.bind("release_mouse", "Escape");
  input_.bind("screenshot", "F2");
  input_.defineAxis("move_x", "left", "right", "leftx", 0.2f);
  input_.defineAxis("move_z", "back", "forward", "", 0.2f);
  {
    atm::Json root;
    if (atm::Json::parseFile(atm::resolve_path("config/client.json"), root, nullptr)) {
      if (const atm::Json *in = root.find("input"))
        input_.loadBindings(*in);
    }
  }

  // Networking (local server first when requested).
  if (cfg_.local)
    startLocalServer();
  std::string netErr;
  if (!net_.connect(cfg_.host, cfg_.port, &netErr)) {
    if (error)
      *error = "connect failed: " + netErr;
    return false;
  }
  status_ = "Connecting to " + cfg_.host + ":" + std::to_string(cfg_.port) + "...";

  SDL_SetWindowRelativeMouseMode(window_, true);
  mouseCaptured_ = true;
  running_ = true;
  lastCounter_ = SDL_GetPerformanceCounter();
  return true;
}

void App::shutdown() {
  // Safe to call twice (main + destructor) and after a failed init(): every
  // step checks what init() actually brought up.
  if (!sdlInit_)
    return;
  running_ = false;
  net_.disconnect();
  stopLocalServer();
  world_.reset();
  sfx_.reset();
  audio_.shutdown();
  clearStructures();
  for (auto id : partMeshes_)
    if (id != atm::render::kInvalidModelMesh)
      renderer_.destroyModelMesh(id);
  for (auto id : blockItemMeshes_)
    if (id != atm::render::kInvalidModelMesh)
      renderer_.destroyModelMesh(id);
  if (lootBeamMesh_ != atm::render::kInvalidModelMesh)
    renderer_.destroyModelMesh(lootBeamMesh_);
  lootBeamMesh_ = atm::render::kInvalidModelMesh;
  if (arrowMesh_ != atm::render::kInvalidModelMesh)
    renderer_.destroyModelMesh(arrowMesh_);
  decor_.shutdown(renderer_);
  partMeshes_.clear();
  blockItemMeshes_.clear();
  arrowMesh_ = atm::render::kInvalidModelMesh;
  if (imguiSdlInit_) {
    ImGui_ImplSDL3_Shutdown();
    imguiSdlInit_ = false;
  }
  renderer_.shutdown(); // also shuts down the ImGui Vulkan backend
  if (imguiContext_) {
    ImGui::DestroyContext();
    imguiContext_ = false;
  }
  if (window_) {
    SDL_DestroyWindow(window_);
    window_ = nullptr;
  }
  sdlInit_ = false;
  SDL_Quit();
}

// ---------------------------------------------------------------------------
// Content upload
// ---------------------------------------------------------------------------

void App::uploadMaterials() {
  std::vector<atm::render::Material> mats;
  mats.reserve(blocks_.size() + 1024);
  for (size_t i = 0; i < blocks_.size(); ++i) {
    const auto &b = blocks_.get(atm::voxel::BlockId(i));
    atm::render::Material m;
    m.top = b.colorTop;
    m.side = b.colorSide;
    m.bottom = b.colorBottom;
    m.emissive = float(b.emission) / 15.0f;
    m.alpha = b.render == atm::voxel::BlockRender::Translucent ? 0.6f : 1.0f;
    if (b.liquid)
      m.flags |= atm::render::kMaterialWater;
    if (b.name.find("leaves") != std::string::npos || b.name.find("flowers") != std::string::npos)
      m.flags |= atm::render::kMaterialFoliage;
    if (b.name == "grass")
      m.flags |= atm::render::kMaterialGrassTop;
    mats.push_back(m);
  }
  // Model palettes follow the block materials (ModelLibrary material bases
  // are relative to this offset — see createModelMeshes()).
  const std::vector<uint32_t> modelColors = models_.materialColors();
  const std::vector<uint8_t> modelGlow = models_.materialEmissive();
  for (size_t i = 0; i < modelColors.size(); ++i) {
    atm::render::Material m;
    m.top = m.side = m.bottom = modelColors[i];
    m.emissive = (i < modelGlow.size() && modelGlow[i]) ? 0.8f : 0.0f;
    mats.push_back(m);
  }
  // Ground decoration colours (Decor.cpp palette) after the model palettes.
  decorMaterialBase_ = uint32_t(mats.size());
  for (uint32_t c : Decor::palette()) {
    atm::render::Material m;
    m.top = m.side = m.bottom = c;
    mats.push_back(m);
  }
  // Fine-voxel structure palette (town buildings); glow entries are emissive.
  microMaterialBase_ = uint32_t(mats.size());
  const auto &micro = ao::world::microPalette();
  for (size_t k = 0; k < micro.size(); ++k) {
    atm::render::Material m;
    m.top = m.side = m.bottom = micro[k];
    m.emissive = k >= ao::world::kMicroGlowFrom ? 0.85f : 0.0f;
    mats.push_back(m);
  }
  renderer_.setMaterials(mats);
}

void App::createModelMeshes() {
  const uint16_t base = uint16_t(blocks_.size());
  const auto &parts = models_.parts();
  partMeshes_.assign(parts.size(), atm::render::kInvalidModelMesh);
  atm::voxel::ChunkMeshData mesh;
  for (size_t i = 0; i < parts.size(); ++i) {
    mesh.clear();
    atm::model::meshPart(parts[i], uint16_t(base + models_.partMaterialBase(int(i))), mesh);
    if (!mesh.empty())
      partMeshes_[i] = renderer_.createModelMesh(mesh);
  }

  // Dropped block items: a small cube of the block's own material.
  blockItemMeshes_.assign(blocks_.size(), atm::render::kInvalidModelMesh);
  for (size_t id = 1; id < blocks_.size(); ++id) {
    atm::model::VoxelPart cube;
    cube.name = "item_block";
    cube.sx = cube.sy = cube.sz = 4;
    cube.voxels.assign(64, 1);
    cube.palette = {0, 0};
    cube.pivot = {2.0f, 2.0f, 2.0f};
    mesh.clear();
    // palette index 1 -> material (id - 1) + 1 = id
    atm::model::meshPart(cube, uint16_t(id - 1), mesh);
    if (!mesh.empty())
      blockItemMeshes_[id] = renderer_.createModelMesh(mesh);
  }

  decor_.init(renderer_, decorMaterialBase_);

  // Loot beam: a thin, tall column of the lamp block's emissive material;
  // tinted per rarity at draw time (glows through bloom).
  {
    atm::model::VoxelPart beam;
    beam.name = "loot_beam";
    beam.sx = 1;
    beam.sy = 24;
    beam.sz = 1;
    beam.voxels.assign(24, 1);
    beam.palette = {0, 0};
    beam.pivot = {0.5f, 0.0f, 0.5f};
    mesh.clear();
    atm::model::meshPart(beam, uint16_t(atm::voxel::blocks::Lamp - 1), mesh);
    if (!mesh.empty())
      lootBeamMesh_ = renderer_.createModelMesh(mesh);
  }

  // Arrow: thin stick with a tip (uses wood + stone block materials).
  {
    atm::model::VoxelPart arrow;
    arrow.name = "arrow";
    arrow.sx = 1;
    arrow.sy = 1;
    arrow.sz = 12;
    arrow.voxels.assign(12, 1);
    arrow.voxels[0] = 2;
    arrow.palette = {0, 0, 0};
    arrow.pivot = {0.5f, 0.5f, 6.0f};
    mesh.clear();
    // palette 1 -> Planks, palette 2 -> Stone (material = base + index)
    atm::model::meshPart(arrow, uint16_t(atm::voxel::blocks::Planks - 1), mesh);
    for (auto &f : mesh.opaque) {
      auto ff = atm::voxel::unpackFace(f);
      if (ff.material == atm::voxel::blocks::Planks + 1)
        ff.material = atm::voxel::blocks::Stone;
      f = atm::voxel::packFace(ff);
    }
    if (!mesh.empty())
      arrowMesh_ = renderer_.createModelMesh(mesh);
  }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

int App::run() {
  const double freq = double(SDL_GetPerformanceFrequency());
  while (running_) {
    atm::prof::Profiler::get().beginFrame();
    const uint64_t now = SDL_GetPerformanceCounter();
    float dt = float(double(now - lastCounter_) / freq);
    lastCounter_ = now;
    dt = std::clamp(dt, 0.0f, 0.25f);
    fps_ = fps_ * 0.95f + (dt > 0 ? 1.0f / dt : 0.0f) * 0.05f;

    {
      ATM_PROFILE_SCOPE("Events + input");
      handleEvents();
    }
    {
      ATM_PROFILE_SCOPE("Network");
      net_.update(*this);
    }

    if (welcomed_ && world_) {
      accumulator_ += dt;
      int steps = 0;
      ATM_PROFILE_SCOPE("Fixed ticks (prediction)");
      while (accumulator_ >= kSimDt && steps < kMaxTicksPerFrame) {
        fixedTick();
        accumulator_ -= kSimDt;
        ++steps;
      }
      if (steps == kMaxTicksPerFrame)
        accumulator_ = std::fmod(accumulator_, double(kSimDt));
      // Remote interpolation clock: advance smoothly, nudge toward newest.
      serverTickEstimate_ += dt * float(kSimHz);
      const float target = float(newestSnapshotTick_);
      const float drift = target - serverTickEstimate_;
      if (std::abs(drift) > 15.0f)
        serverTickEstimate_ = target;
      else
        serverTickEstimate_ += drift * std::min(1.0f, dt * 2.0f);
    }
    const float alpha = float(accumulator_ / kSimDt);

    {
      ATM_PROFILE_SCOPE("World streaming");
      updateWorld();
    }
    {
      ATM_PROFILE_SCOPE("Camera + interaction");
      updateCamera(dt);
      updateInteraction(dt);
    }
    {
      ATM_PROFILE_SCOPE("Entities + effects");
      updateEntities(dt);
    }
    prediction_.decayCorrection(dt);
    {
      ATM_PROFILE_SCOPE("Config hot-reload check");
      if (atm::Tunables::instance().reloadIfChanged())
        addChatLine("[config] tunables reloaded");
    }

    {
      ATM_PROFILE_SCOPE("Render");
      render(alpha, dt);
    }
    atm::prof::Profiler::get().endFrame();
  }
  return 0;
}


} // namespace ao::client
