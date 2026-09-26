// App: SDL events, key bindings, per-tick input and prediction.

#include "app/AppInternal.h"

namespace ao::client {

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
        // Trove-style: the wheel zooms the camera; Shift + wheel changes
        // the hotbar slot (1-9 also select slots).
        if (SDL_GetModState() & SDL_KMOD_SHIFT) {
          // e.wheel.y is a float (fractional on smooth-scrolling devices).
          const int notch = e.wheel.y > 0.0f ? 1 : (e.wheel.y < 0.0f ? -1 : 0);
          hotbar_ = (hotbar_ - notch + 9) % 9;
        } else {
          camDistance_ = std::clamp(camDistance_ - e.wheel.y * 0.75f, 2.0f, 14.0f);
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
    if (showInventory_ || showSkills_ || showLook_ || showWorldMap_) {
      showInventory_ = showSkills_ = false;
      if (showLook_ || showWorldMap_) {
        showLook_ = showWorldMap_ = false;
        SDL_SetWindowRelativeMouseMode(window_, true);
        mouseCaptured_ = true;
      }
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
  // M: world map (frees the mouse for panning / zooming).
  if (input_.pressed("map") && welcomed_) {
    showWorldMap_ = !showWorldMap_;
    worldMapPan_ = glm::dvec2(0.0);
    SDL_SetWindowRelativeMouseMode(window_, !showWorldMap_);
    mouseCaptured_ = !showWorldMap_;
  }
  // E: pick up ground items within reach (hold to keep collecting).
  if (input_.held("pickup") && welcomed_ && SDL_GetTicks() >= nextPickupMs_) {
    proto::Pickup pk;
    pk.entity = 0;
    net_.send(pk, atm::net2::Channel::ReliableOrdered);
    nextPickupMs_ = SDL_GetTicks() + 250;
  }
  if (input_.pressed("profiler")) {
    showProfiler_ = !showProfiler_;
    atm::prof::Profiler::get().setEnabled(showProfiler_);
  }
  if (input_.pressed("hitboxes"))
    showHitboxes_ = !showHitboxes_;
  if (input_.pressed("graphics")) { // F10: live look tuning (sliders need the mouse)
    showLook_ = !showLook_;
    SDL_SetWindowRelativeMouseMode(window_, !showLook_);
    mouseCaptured_ = !showLook_;
  }
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


} // namespace ao::client
