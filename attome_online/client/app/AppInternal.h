#pragma once

// Private helpers shared by the App*.cpp translation units (the App class is
// split by responsibility: App.cpp = startup / shutdown / main loop,
// AppInput = events and input ticks, AppWorld = world streaming + camera +
// entity updates, AppInteraction = mining / placing / attacking, AppRender =
// the frame, AppNet = server message handlers, AppCharacters = models,
// AppLocalServer = the in-process server). Not for use outside app/.

#include "app/App.h"
#include "audio/SfxPlayer.h"
#include "ui/UiTheme.h"

#include "server/ZoneServer.h" // complete type for App's unique_ptr<ZoneServer>
#include "shared/Movement.h"

#include "../../../engine/ATMConfig.h"
#include "../../../engine/ATMJson.h"
#include "../../../engine/ATMFrameProfiler.h"

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

inline constexpr float kReach = 6.0f;
inline constexpr int kMaxTicksPerFrame = 5;
inline constexpr float kInterpDelayTicks = 3.0f; // ~100 ms at 30 Hz

inline glm::vec3 lookDir(float yaw, float pitch) {
  return {-std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch)};
}

inline uint32_t rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
  return uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24);
}

namespace act = ao::action; // EntityState::actionAnim codes (Snapshot.h)

} // namespace ao::client
