#pragma once

// Local-player prediction and reconciliation (NETWORK_PLAN §6.6).
//
// Every fixed tick the client applies its own input immediately (so movement
// feels instant) and remembers it. When a snapshot arrives with the server's
// authoritative state and the newest input it applied, the client rewinds to
// that state and replays the inputs the server hasn't seen yet. Any visible
// difference is blended out over a few frames instead of snapping.

#include "shared/Movement.h"
#include "shared/Snapshot.h"

#include <array>
#include <cstdint>

namespace ao::client {

class Prediction {
public:
  void reset(const MoveState &state);

  // Applies one input locally (fixed step) and records it.
  void applyLocal(const MoveInput &input, const atm::voxel::IBlockAccess &world,
                  const atm::voxel::BlockRegistry &blocks);

  // Server correction: state as of `ackInputSeq`.
  void reconcile(const MoveState &server, uint32_t ackInputSeq,
                 const atm::voxel::IBlockAccess &world,
                 const atm::voxel::BlockRegistry &blocks);

  const MoveState &current() const { return current_; }
  const MoveState &previous() const { return previous_; }

  // Render position: interpolated between the last two ticks plus the
  // decaying correction offset.
  glm::dvec3 renderPosition(float alpha) const;
  void decayCorrection(float dt);

  // The last up-to-3 inputs, newest last (for InputBatch redundancy).
  int recentInputs(MoveInput *out, int max) const;

  uint32_t lastCorrectionMicroBlocks() const { return lastCorrection_; }

private:
  static constexpr int kHistory = 256; // > 8 s of inputs at 30 Hz
  std::array<MoveInput, kHistory> inputs_{};
  uint32_t oldestSeq_ = 1, nextSeq_ = 1; // [oldest, next) are unacknowledged
  MoveState current_{}, previous_{};
  glm::dvec3 correction_{0.0};
  uint32_t lastCorrection_ = 0;
};

// Snapshot interpolation for remote entities: holds a short history of
// server states and returns a smoothed state ~100 ms in the past.
struct RemoteSample {
  float tick = 0;           // server tick (float for interpolation)
  glm::dvec3 pos{0.0};
  glm::vec3 vel{0.0f};
  float yaw = 0;
  uint8_t flags = 0;
};

class RemoteTrack {
public:
  void push(const RemoteSample &s);
  // Interpolates at `renderTick`; extrapolates up to 250 ms when data is late.
  RemoteSample sample(float renderTick) const;
  bool empty() const { return count_ == 0; }
  float newestTick() const;

private:
  static constexpr int kCap = 16;
  std::array<RemoteSample, kCap> samples_{};
  int head_ = 0, count_ = 0; // ring, newest at head_-1
};

} // namespace ao::client
