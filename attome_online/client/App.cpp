#include "App.h"

#include "SfxPlayer.h"

#include "server/ZoneServer.h" // complete type for App's unique_ptr<ZoneServer>
#include "shared/Movement.h"

#include "../../engine/ATMConfig.h"
#include "../../engine/ATMJson.h"

#include <SDL3/SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <type_traits>

namespace ao::client {

using atm::voxel::BlockRegistry;
using atm::voxel::ChunkCoord;
using atm::voxel::FaceDir;
using atm::voxel::kFaceNormal;

namespace {

constexpr float kReach = 6.0f;
constexpr int kMaxTicksPerFrame = 5;
constexpr float kInterpDelayTicks = 3.0f; // ~100 ms at 30 Hz

glm::vec3 lookDir(float yaw, float pitch) {
  return {-std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch)};
}

uint32_t rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
  return uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24);
}

namespace act = ao::action; // EntityState::actionAnim codes (Snapshot.h)

} // namespace

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
  ImGuiStyle &style = ImGui::GetStyle();
  style.WindowRounding = 6.0f;
  style.FrameRounding = 4.0f;
  style.Colors[ImGuiCol_WindowBg].w = 0.82f;

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
  input_.bind("dash", "Left Shift");
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
  for (auto id : partMeshes_)
    if (id != atm::render::kInvalidModelMesh)
      renderer_.destroyModelMesh(id);
  for (auto id : blockItemMeshes_)
    if (id != atm::render::kInvalidModelMesh)
      renderer_.destroyModelMesh(id);
  if (arrowMesh_ != atm::render::kInvalidModelMesh)
    renderer_.destroyModelMesh(arrowMesh_);
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
    const uint64_t now = SDL_GetPerformanceCounter();
    float dt = float(double(now - lastCounter_) / freq);
    lastCounter_ = now;
    dt = std::clamp(dt, 0.0f, 0.25f);
    fps_ = fps_ * 0.95f + (dt > 0 ? 1.0f / dt : 0.0f) * 0.05f;

    handleEvents();
    net_.update(*this);

    if (welcomed_ && world_) {
      accumulator_ += dt;
      int steps = 0;
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

    updateWorld();
    updateCamera(dt);
    updateInteraction(dt);
    updateEntities(dt);
    prediction_.decayCorrection(dt);
    if (atm::Tunables::instance().reloadIfChanged())
      addChatLine("[config] tunables reloaded");

    render(alpha, dt);
  }
  return 0;
}

void App::handleEvents() {
  input_.beginFrame();
  SDL_Event e;
  while (SDL_PollEvent(&e)) {
    ImGui_ImplSDL3_ProcessEvent(&e);
    const bool uiKeyboard = ImGui::GetIO().WantCaptureKeyboard && (chatOpen_ || !mouseCaptured_);
    switch (e.type) {
    case SDL_EVENT_QUIT:
      running_ = false;
      break;
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
      renderer_.onWindowResized();
      break;
    case SDL_EVENT_WINDOW_FOCUS_LOST:
      if (mouseCaptured_) {
        SDL_SetWindowRelativeMouseMode(window_, false);
        mouseCaptured_ = false;
      }
      break;
    case SDL_EVENT_MOUSE_MOTION:
      if (mouseCaptured_) {
        camYaw_ -= e.motion.xrel * cfg_.mouseSensitivity;
        camPitch_ -= e.motion.yrel * cfg_.mouseSensitivity * (cfg_.invertY ? -1.0f : 1.0f);
        camPitch_ = std::clamp(camPitch_, -1.45f, 1.3f);
        if (camYaw_ > 3.14159265f) camYaw_ -= 6.2831853f;
        if (camYaw_ < -3.14159265f) camYaw_ += 6.2831853f;
      }
      break;
    case SDL_EVENT_MOUSE_WHEEL:
      if (mouseCaptured_) {
        if (SDL_GetModState() & SDL_KMOD_CTRL) {
          camDistance_ = std::clamp(camDistance_ - e.wheel.y * 0.75f, 2.0f, 14.0f);
        } else {
          // e.wheel.y is a float (fractional on smooth-scrolling devices).
          const int notch = e.wheel.y > 0.0f ? 1 : (e.wheel.y < 0.0f ? -1 : 0);
          hotbar_ = (hotbar_ - notch + 9) % 9;
        }
      }
      break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
      if (!mouseCaptured_ && !ImGui::GetIO().WantCaptureMouse && !chatOpen_ && !showInventory_) {
        SDL_SetWindowRelativeMouseMode(window_, true);
        mouseCaptured_ = true;
        continue; // don't treat the capture click as an attack
      }
      break;
    case SDL_EVENT_KEY_DOWN:
      if (!uiKeyboard && !e.key.repeat && e.key.scancode >= SDL_SCANCODE_1 &&
          e.key.scancode <= SDL_SCANCODE_9)
        hotbar_ = int(e.key.scancode - SDL_SCANCODE_1);
      break;
    default:
      break;
    }
    if (!uiKeyboard || e.type == SDL_EVENT_MOUSE_BUTTON_UP || e.type == SDL_EVENT_KEY_UP)
      input_.handleEvent(e);
  }

  if (chatOpen_)
    return; // typing: gameplay keys ignored

  if (input_.pressed("release_mouse")) {
    if (showInventory_ || showSkills_) {
      showInventory_ = showSkills_ = false;
    } else {
      SDL_SetWindowRelativeMouseMode(window_, false);
      mouseCaptured_ = false;
    }
  }
  if (input_.pressed("inventory")) {
    showInventory_ = !showInventory_;
    SDL_SetWindowRelativeMouseMode(window_, !showInventory_);
    mouseCaptured_ = !showInventory_;
  }
  if (input_.pressed("skills"))
    showSkills_ = !showSkills_;
  if (input_.pressed("debug"))
    showDebug_ = !showDebug_;
  if (input_.pressed("screenshot"))
    renderer_.requestScreenshot("screenshot.bmp");
  // Open on release: the chat InputText is created this frame, and opening on
  // the key-down would let it see the same Enter press and submit at once.
  if (input_.released("chat") && welcomed_) {
    chatOpen_ = true;
    chatInput_[0] = 0;
    SDL_SetWindowRelativeMouseMode(window_, false);
    mouseCaptured_ = false;
  }
  if (input_.pressed("jump"))
    pendingButtons_ |= button::Jump;
  if (input_.pressed("dash"))
    pendingButtons_ |= button::Dash;
}

// ---------------------------------------------------------------------------
// Fixed-rate simulation (prediction)
// ---------------------------------------------------------------------------

MoveInput App::buildInput() {
  MoveInput in;
  in.tick = clientTick_;
  in.seq = nextInputSeq_++;
  in.yaw = camYaw_;
  in.pitch = camPitch_;
  if (mouseCaptured_ && !chatOpen_) {
    in.moveX = input_.axis("move_x");
    in.moveZ = input_.axis("move_z");
    const float len = std::sqrt(in.moveX * in.moveX + in.moveZ * in.moveZ);
    if (len > 1.0f) {
      in.moveX /= len;
      in.moveZ /= len;
    }
    uint16_t b = pendingButtons_;
    if (input_.held("jump")) b |= button::Jump;
    if (input_.held("dash")) b |= button::Dash;
    if (input_.held("sprint")) b |= button::Sprint;
    if (input_.held("primary")) b |= button::Primary;
    if (input_.held("secondary")) b |= button::Secondary;
    in.buttons = b;
  }
  pendingButtons_ = 0;
  return in;
}

void App::fixedTick() {
  const MoveState before = prediction_.current();
  const MoveInput in = buildInput();
  prediction_.applyLocal(in, *world_, blocks_);
  const MoveState &after = prediction_.current();

  proto::InputBatch batch;
  MoveInput recent[3];
  const int n = prediction_.recentInputs(recent, 3);
  batch.inputs.assign(recent, recent + n);
  batch.lastSnapshotTick = newestSnapshotTick_;
  net_.send(batch, atm::net2::Channel::Unreliable);
  ++clientTick_;

  // Local movement sounds.
  if (sfx_) {
    if (before.onGround && !after.onGround && after.vel.y > 1.0f)
      sfx_->play(GameSound::Jump);
    else if (!before.onGround && !after.onGround && after.vel.y > before.vel.y + 3.0f)
      sfx_->play(GameSound::Jump); // double jump
    if (!before.onGround && after.onGround && before.vel.y < -6.0f)
      sfx_->play(GameSound::Land);
    if (after.dashTimer > 0.0f && before.dashTimer <= 0.0f)
      sfx_->play(GameSound::Dash);
    sfx_->setGlideWind(after.gliding);
    if (after.onGround) {
      const glm::dvec3 d = after.pos - before.pos;
      footstepDistance_ += float(std::sqrt(d.x * d.x + d.z * d.z));
      if (footstepDistance_ > 2.1f) {
        footstepDistance_ = 0.0f;
        const auto below = world_->blockAt({int32_t(std::floor(after.pos.x)),
                                            int32_t(std::floor(after.pos.y - 0.05)),
                                            int32_t(std::floor(after.pos.z))});
        sfx_->playFootstep(below);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

void App::updateWorld() {
  if (!world_)
    return;
  const glm::dvec3 p = prediction_.current().pos;
  world_->clearFoci();
  world_->addFocus(p.x, p.y, p.z);
  world_->update();

  meshResults_.clear();
  world_->takeMeshResults(meshResults_);
  for (const auto &r : meshResults_)
    renderer_.setChunkMesh(r.coord, r.mesh);

  unloaded_.clear();
  world_->takeUnloaded(unloaded_);
  for (const auto &c : unloaded_)
    renderer_.removeChunk(c);
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

glm::dvec3 App::eyePosition(float alpha) const {
  glm::dvec3 p = prediction_.renderPosition(alpha);
  p.y += moveTuning().eyeHeight;
  return p;
}

glm::vec3 App::aimDirection() const { return lookDir(camYaw_, camPitch_); }

void App::updateCamera(float) {
  const float alpha = float(accumulator_ / kSimDt);
  const glm::dvec3 target = eyePosition(alpha) + glm::dvec3(0.0, 0.25, 0.0);
  const glm::vec3 fwd = aimDirection();
  const glm::vec3 right{std::cos(camYaw_), 0.0f, -std::sin(camYaw_)};

  // Over-the-shoulder offset, then pull in when blocks are in the way.
  const glm::dvec3 shoulder = target + glm::dvec3(right) * 0.55;
  float dist = camDistance_;
  if (world_) {
    const glm::dvec3 back = -glm::dvec3(fwd);
    const float step = 0.2f;
    for (float t = 0.3f; t <= camDistance_; t += step) {
      const glm::dvec3 q = shoulder + back * double(t);
      const atm::voxel::BlockPos bp{int32_t(std::floor(q.x)), int32_t(std::floor(q.y)),
                                    int32_t(std::floor(q.z))};
      if (blocks_.solid(world_->blockAt(bp))) {
        dist = std::max(0.3f, t - 0.35f);
        break;
      }
    }
  }
  camPos_ = shoulder - glm::dvec3(fwd) * double(dist);

  camera_.position = camPos_;
  camera_.yaw = camYaw_;
  camera_.pitch = camPitch_;
  camera_.fovYDegrees = cfg_.render.fovYDegrees;

  int w = 0, h = 0;
  SDL_GetWindowSizeInPixels(window_, &w, &h);
  if (w > 0 && h > 0)
    aspect_ = float(w) / float(h);
}

bool App::worldToScreen(const glm::dvec3 &p, float &sx, float &sy) const {
  const glm::vec3 rel = glm::vec3(p - camPos_);
  const glm::vec3 fwd = aimDirection();
  const glm::mat4 view = glm::lookAt(glm::vec3(0.0f), fwd, glm::vec3(0, 1, 0));
  const glm::mat4 proj = glm::perspective(glm::radians(camera_.fovYDegrees), aspect_, 0.05f, 1000.0f);
  const glm::vec4 clip = proj * view * glm::vec4(rel, 1.0f);
  if (clip.w <= 0.01f)
    return false;
  const ImVec2 size = ImGui::GetIO().DisplaySize;
  sx = (clip.x / clip.w * 0.5f + 0.5f) * size.x;
  sy = (1.0f - (clip.y / clip.w * 0.5f + 0.5f)) * size.y;
  return true;
}

// ---------------------------------------------------------------------------
// Interaction: mining, placing, attacking
// ---------------------------------------------------------------------------

ItemId App::equippedMainHand() const {
  return equipped_.size() > size_t(atm::model::EquipSlot::MainHand)
             ? ItemId(equipped_[size_t(atm::model::EquipSlot::MainHand)])
             : ItemId(0);
}

void App::updateInteraction(float dt) {
  attackCooldown_ = std::max(0.0f, attackCooldown_ - dt);
  placeCooldown_ = std::max(0.0f, placeCooldown_ - dt);
  hasTarget_ = false;
  if (!world_ || !welcomed_ || !mouseCaptured_ || chatOpen_) {
    mineProgress_ = 0.0f;
    return;
  }

  // Aim from the camera through the crosshair; accept hits within reach of
  // the player's eye.
  const glm::dvec3 eye = eyePosition(float(accumulator_ / kSimDt));
  BlockPos hit{};
  FaceDir face = FaceDir::PosY;
  if (raycastBlock(*world_, blocks_, camPos_, aimDirection(), camDistance_ + kReach + 1.0f, hit, face)) {
    const glm::dvec3 center{hit.x + 0.5, hit.y + 0.5, hit.z + 0.5};
    const glm::dvec3 d = center - eye;
    if (std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z) <= kReach + 0.5) {
      hasTarget_ = true;
      targetBlock_ = hit;
      targetFace_ = face;
    }
  }

  const ItemId mainItem = equippedMainHand();
  const ItemDef &mainDef = itemDef(mainItem);
  const bool holdingWeapon = mainDef.kind == ItemKind::Weapon;
  const bool primary = input_.held("primary");

  // Mining: primary on a block with a tool or bare hands (not a weapon).
  // Unbreakable (hardness < 0, e.g. bedrock) or above our Mining level: the
  // server would refuse it, so don't start mining at all.
  bool breakable = false;
  if (hasTarget_) {
    const auto &tdef = blocks_.get(world_->blockAt(targetBlock_));
    breakable = tdef.hardness >= 0.0f &&
                levelForXp(skillXp_[size_t(Skill::Mining)]) >= uint32_t(tdef.miningLevel);
  }
  if (primary && breakable && !holdingWeapon) {
    if (!(miningBlock_ == targetBlock_)) {
      miningBlock_ = targetBlock_;
      mineProgress_ = 0.0f;
    }
    const auto &def = blocks_.get(world_->blockAt(targetBlock_));
    const bool pick = mainDef.weapon == WeaponType::Pickaxe;
    const float breakTime = std::max(0.08f, def.hardness * (pick ? 0.3f : 1.0f));
    const float before = mineProgress_;
    mineProgress_ += dt / breakTime;
    // Swing animation + sound about every 0.35 s while mining.
    if (int(before * breakTime / 0.35f) != int(mineProgress_ * breakTime / 0.35f) || before == 0.0f) {
      playActionAnim(selfAnim_, act::Mine, mainDef.weapon);
      if (sfx_)
        sfx_->playBlockHit(world_->blockAt(targetBlock_));
    }
    if (mineProgress_ >= 1.0f) {
      proto::BlockAction a;
      a.action = 0;
      a.x = targetBlock_.x;
      a.y = targetBlock_.y;
      a.z = targetBlock_.z;
      a.face = uint8_t(targetFace_);
      a.hotbarSlot = uint8_t(hotbar_);
      net_.send(a, atm::net2::Channel::ReliableOrdered);
      mineProgress_ = 0.0f;
    }
  } else {
    mineProgress_ = 0.0f;
  }

  // Attacking: primary with a weapon (or no block targeted).
  if (primary && (holdingWeapon || !hasTarget_) && attackCooldown_ <= 0.0f) {
    proto::Attack a;
    a.tick = clientTick_;
    a.yaw = camYaw_;
    a.pitch = camPitch_;
    a.ability = 0;
    net_.send(a, atm::net2::Channel::ReliableOrdered);
    switch (mainDef.weapon) {
    case WeaponType::Bow:
      attackCooldown_ = 0.7f;
      playActionAnim(selfAnim_, act::BowShoot, mainDef.weapon);
      if (sfx_) sfx_->play(GameSound::BowShoot);
      break;
    case WeaponType::Staff:
      attackCooldown_ = 0.6f;
      playActionAnim(selfAnim_, act::Cast, mainDef.weapon);
      if (sfx_) sfx_->play(GameSound::SwordSwing);
      break;
    default:
      attackCooldown_ = 0.45f;
      playActionAnim(selfAnim_, act::Swing, mainDef.weapon);
      if (sfx_) sfx_->play(GameSound::SwordSwing);
      break;
    }
  }

  // Placing: secondary on a block face with a block item selected.
  if (input_.held("secondary") && hasTarget_ && placeCooldown_ <= 0.0f) {
    const ItemStack &stack = inventory_[size_t(hotbar_)];
    if (stack.item != 0 && stack.count > 0 && itemDef(stack.item).kind == ItemKind::Block) {
      proto::BlockAction a;
      a.action = 1;
      a.x = targetBlock_.x;
      a.y = targetBlock_.y;
      a.z = targetBlock_.z;
      a.face = uint8_t(targetFace_);
      a.hotbarSlot = uint8_t(hotbar_);
      net_.send(a, atm::net2::Channel::ReliableOrdered);
      placeCooldown_ = 0.2f;
      playActionAnim(selfAnim_, act::Place, mainDef.weapon);
    }
  }
}

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

void App::updateEntities(float dt) {
  // Local player animation.
  const MoveState &me = prediction_.current();
  selectLocomotion(selfAnim_, me.vel,
                   uint8_t((me.gliding ? 1 : 0) | (me.inWater ? 2 : 0) | (me.onGround ? 8 : 0)),
                   true);
  selfAnim_.update(anims_, dt);

  for (auto &[id, r] : remotes_) {
    const RemoteSample s = r.track.sample(serverTickEstimate_ - kInterpDelayTicks);
    const int creature = r.last.kind == EntityKind::Monster ? creatureForMonster(r.last.type) : -1;
    if (r.last.kind == EntityKind::Player || r.last.kind == EntityKind::Monster)
      selectLocomotion(r.animator, s.vel, s.flags, false, creature);
    r.animator.update(anims_, dt);
    r.hitFlash = std::max(0.0f, r.hitFlash - dt);
    if (sfx_ && r.last.kind == EntityKind::Player && (s.flags & 8)) {
      r.stepDistance += glm::length(glm::vec2(s.vel.x, s.vel.z)) * dt;
      if (r.stepDistance > 2.1f) {
        r.stepDistance = 0.0f;
        const glm::dvec3 d = s.pos - prediction_.current().pos;
        if (d.x * d.x + d.y * d.y + d.z * d.z < 24.0 * 24.0)
          sfx_->playAt(GameSound::Footstep, float(std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z)));
      }
    }
  }

  // Floating texts, XP drops, banners age out.
  for (auto &f : floating_) f.age += dt;
  floating_.erase(std::remove_if(floating_.begin(), floating_.end(),
                                 [](const FloatingText &f) { return f.age >= f.life; }),
                  floating_.end());
  for (auto &x : xpDrops_) x.age += dt;
  xpDrops_.erase(std::remove_if(xpDrops_.begin(), xpDrops_.end(),
                                [](const XpDrop &x) { return x.age >= 1.8f; }),
                 xpDrops_.end());
  for (auto &b : banners_) b.age += dt;
  banners_.erase(std::remove_if(banners_.begin(), banners_.end(),
                                [](const Banner &b) { return b.age >= 3.5f; }),
                 banners_.end());
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

void App::render(float alpha, float dt) {
  atm::render::Environment env;
  env.sunDirection = glm::normalize(glm::vec3(0.35f, 0.85f, 0.25f));

  if (!renderer_.beginFrame(camera_, env)) {
    SDL_Delay(10); // minimised: don't spin; ImGui::NewFrame is skipped too
    return;
  }
  ImGui_ImplSDL3_NewFrame();
  ImGui::NewFrame();

  if (welcomed_ && world_) {
    // Local player.
    const MoveState &me = prediction_.current();
    drawCharacter(selfAppearance_, selfAnim_, prediction_.renderPosition(alpha), me.yaw,
                  hp_ == 0 ? rgba(120, 120, 120) : 0xFFFFFFFFu);

    // Remote entities.
    const float renderTick = serverTickEstimate_ - kInterpDelayTicks;
    for (auto &[id, r] : remotes_) {
      if (r.track.empty())
        continue; // known only from an AppearanceMsg so far: no position yet
      const RemoteSample s = r.track.sample(renderTick);
      const uint32_t tint = r.hitFlash > 0.0f ? rgba(255, 120, 120) : 0xFFFFFFFFu;
      switch (r.last.kind) {
      case EntityKind::Player:
        drawCharacter(r.appearance, r.animator, s.pos, s.yaw, tint);
        break;
      case EntityKind::Monster:
        drawMonster(r.last.type, r.animator, s.pos, s.yaw, tint);
        break;
      case EntityKind::DroppedItem:
        drawItemEntity(r.last.item, s.pos, renderTick * 0.08f);
        break;
      case EntityKind::Projectile:
        drawProjectile(s.pos, s.vel);
        break;
      }
    }

    if (hasTarget_)
      renderer_.drawBlockHighlight(targetBlock_);
  }

  drawHud(dt);
  renderer_.endFrame();
}

void App::addFloatingText(const glm::dvec3 &at, std::string text, uint32_t color) {
  if (floating_.size() > 64)
    floating_.erase(floating_.begin());
  floating_.push_back({at, std::move(text), color, 0.0f, 1.2f});
}

void App::addChatLine(std::string line) {
  chat_.push_back(std::move(line));
  while (chat_.size() > 50)
    chat_.pop_front();
}

// ---------------------------------------------------------------------------
// Network handlers
// ---------------------------------------------------------------------------

void App::onConnected() {
  status_ = "Connected, joining...";
  proto::Hello h;
  h.schemaHash = proto::kSchemaHash;
  h.name = cfg_.name;
  h.appearance = selfAppearance_;
  net_.send(h, atm::net2::Channel::ReliableOrdered);
}

void App::onDisconnected(atm::net2::DisconnectReason reason) {
  static const char *names[] = {"none", "timeout", "requested", "rejected",
                                "protocol mismatch", "server full", "too many retries", "backlog"};
  const unsigned r = unsigned(reason);
  status_ = std::string("Disconnected: ") + (r < 8 ? names[r] : "unknown");
  addChatLine(status_);
  welcomed_ = false;
}

void App::onWelcome(const proto::Welcome &m) {
  if (m.schemaHash != proto::kSchemaHash) {
    status_ = "Server runs a different game version";
    net_.disconnect();
    return;
  }
  selfId_ = m.playerEntity;
  worldSeed_ = m.worldSeed;
  clientTick_ = m.serverTick;
  newestSnapshotTick_ = m.serverTick;
  serverTickEstimate_ = float(m.serverTick);

  atm::voxel::VoxelWorldConfig wc;
  wc.seed = m.worldSeed;
  wc.meshing = true;
  wc.viewRadiusChunks = cfg_.viewRadiusChunks;
  world_ = std::make_unique<atm::voxel::VoxelWorld>(wc, blocks_);

  MoveState s;
  s.pos = m.spawn;
  prediction_.reset(s);
  camYaw_ = 0.0f;
  welcomed_ = true;
  status_.clear();
  addChatLine("Welcome to Attome Online! WASD move, Space jump/glide, Shift dash, LMB attack/mine, RMB place, Tab inventory, K skills.");
}

void App::onChunkData(const proto::ChunkData &m) {
  if (!world_)
    return;
  auto chunk = std::make_shared<atm::voxel::Chunk>();
  if (!chunk->deserialize(m.data)) {
    SDL_Log("[client] bad chunk data for %d,%d,%d", m.cx, m.cy, m.cz);
    return;
  }
  world_->setChunkData(ChunkCoord{m.cx, m.cy, m.cz}, std::move(chunk));
}

void App::onEditedChunks(const proto::EditedChunks &m) {
  if (world_)
    world_->setExpectedEdited(m.keys);
}

void App::onBlockChanged(const proto::BlockChanged &m) {
  if (!world_)
    return;
  const BlockPos p{m.x, m.y, m.z};
  const auto old = world_->blockAt(p);
  world_->setBlock(p, m.block);
  if (sfx_) {
    const glm::dvec3 d = glm::dvec3(p.x + 0.5, p.y + 0.5, p.z + 0.5) - prediction_.current().pos;
    const float dist = float(std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z));
    if (dist < 32.0f) {
      if (m.block == atm::voxel::kAir)
        sfx_->playBlockBreak(old, dist);
      else
        sfx_->playBlockPlace(m.block, dist);
    }
  }
}

void App::onSnapshot(const proto::SnapshotMsg &m) {
  const Snapshot &s = m.snapshot;
  if (s.tick < newestSnapshotTick_)
    return; // out of order
  newestSnapshotTick_ = s.tick;

  if (world_)
    prediction_.reconcile(s.self.move, s.ackInputSeq, *world_, blocks_);
  if (s.self.hp < hp_ && hp_ > 0 && sfx_)
    sfx_->play(GameSound::PlayerHurt);
  hp_ = s.self.hp;
  maxHp_ = std::max<uint16_t>(1, s.self.maxHp);

  for (const EntityState &in : s.entities) {
    if (in.id == selfId_)
      continue;
    auto [it, inserted] = remotes_.try_emplace(in.id);
    RemoteEntity &r = it->second;
    r.id = in.id;
    EntityState merged = inserted ? EntityState{} : r.last;
    if (in.mask & field::Type) { merged.kind = in.kind; merged.type = in.type; merged.item = in.item; }
    if (in.mask & field::Pos) merged.pos = in.pos;
    if (in.mask & field::Vel) merged.vel = in.vel;
    if (in.mask & field::Yaw) merged.yaw = in.yaw;
    if (in.mask & field::Anim) {
      merged.locoAnim = in.locoAnim;
      merged.actionAnim = in.actionAnim;
      merged.actionSeq = in.actionSeq;
    }
    if (in.mask & field::Health) { merged.hp = in.hp; merged.maxHp = in.maxHp; }
    if (in.mask & field::Flags) merged.flags = in.flags;
    merged.id = in.id;

    if (!inserted && merged.actionSeq != r.lastActionSeq && merged.actionAnim != act::None) {
      WeaponType w = WeaponType::None;
      int creature = -1;
      if (merged.kind == EntityKind::Player) {
        const auto piece = r.appearance.pieces[size_t(atm::model::EquipSlot::MainHand)];
        w = WeaponType(models_.piece(piece).weaponType);
      } else if (merged.kind == EntityKind::Monster) {
        creature = creatureForMonster(merged.type);
      }
      playActionAnim(r.animator, merged.actionAnim, w, creature);
    }
    r.lastActionSeq = merged.actionSeq;
    r.last = merged;
    r.track.push({float(s.tick), merged.pos, merged.vel, merged.yaw, merged.flags});
  }
  for (EntityId gone : s.removed)
    remotes_.erase(gone);
}

void App::onAppearance(const proto::AppearanceMsg &m) {
  if (m.entity == selfId_) {
    selfAppearance_ = m.appearance;
    return;
  }
  RemoteEntity &r = remotes_[m.entity];
  r.id = m.entity;
  r.appearance = m.appearance;
  r.name = m.name;
}

void App::onDamage(const proto::DamageEvent &m) {
  glm::dvec3 at = prediction_.current().pos + glm::dvec3(0, 2.2, 0);
  if (m.target != selfId_) {
    auto it = remotes_.find(m.target);
    if (it == remotes_.end())
      return;
    at = it->second.last.pos + glm::dvec3(0, 2.0, 0);
    it->second.hitFlash = 0.18f;
    if (m.killed) {
      const int creature = it->second.last.kind == EntityKind::Monster
                               ? creatureForMonster(it->second.last.type)
                               : -1;
      playActionAnim(it->second.animator, act::Death, WeaponType::None, creature);
    }
  }
  const uint32_t color = m.target == selfId_ ? rgba(255, 80, 80)
                         : m.critical        ? rgba(255, 200, 40)
                                             : rgba(255, 255, 255);
  addFloatingText(at, (m.critical ? std::to_string(m.amount) + "!" : std::to_string(m.amount)), color);
  if (sfx_ && m.source == selfId_)
    sfx_->play(GameSound::MonsterHit);
}

void App::onInventory(const proto::InventoryMsg &m) {
  for (size_t i = 0; i < inventory_.size(); ++i)
    inventory_[i] = i < m.slots.size() ? m.slots[i] : ItemStack{};
  for (size_t i = 0; i < equipped_.size(); ++i)
    equipped_[i] = i < m.equipped.size() ? m.equipped[i] : 0;
  // Local appearance follows equipment immediately.
  for (size_t i = 0; i < equipped_.size() && i < selfAppearance_.pieces.size(); ++i) {
    const ItemDef &d = itemDef(ItemId(equipped_[i]));
    selfAppearance_.pieces[i] = d.piece ? models_.findPiece(d.piece) : atm::model::kNoPiece;
  }
}

void App::onXpGain(const proto::XpGain &m) {
  if (m.skill >= kSkillCount)
    return;
  const uint32_t before = levelForXp(skillXp_[m.skill]);
  skillXp_[m.skill] = m.totalXp;
  const uint32_t after = levelForXp(m.totalXp);
  xpDrops_.push_back({Skill(m.skill), m.amount, 0.0f});
  if (after > before) {
    banners_.push_back({"Congratulations! " + std::string(skillName(Skill(m.skill))) +
                            " level " + std::to_string(after),
                        0.0f});
    addChatLine("You advanced a " + std::string(skillName(Skill(m.skill))) + " level: " +
                std::to_string(after));
    if (sfx_)
      sfx_->play(GameSound::LevelUp);
  }
}

void App::onLoot(const proto::LootMsg &m) {
  const ItemDef &d = itemDef(m.item);
  addChatLine(std::string(m.rare ? "RARE DROP: " : "Loot: ") + std::string(d.name) + " x" +
              std::to_string(m.count));
  if (m.rare)
    banners_.push_back({"Rare drop: " + std::string(d.name) + "!", 0.0f});
  if (sfx_)
    sfx_->play(GameSound::Pickup);
}

void App::onChat(const proto::ChatMsg &m) { addChatLine(m.from + ": " + m.text); }

} // namespace ao::client
