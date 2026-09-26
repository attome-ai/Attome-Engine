#include "Prediction.h"

#include <algorithm>
#include <cmath>

namespace ao::client {

void Prediction::reset(const MoveState &state) {
  current_ = state;
  previous_ = state;
  correction_ = glm::dvec3(0.0);
  oldestSeq_ = nextSeq_;
}

void Prediction::applyLocal(const MoveInput &input, const atm::voxel::IBlockAccess &world,
                            const atm::voxel::BlockRegistry &blocks) {
  // If the history is full (server not acking for ~8 s), drop the oldest:
  // reconciliation will then snap, which is the right outcome after a stall.
  if (nextSeq_ - oldestSeq_ >= kHistory)
    ++oldestSeq_;
  inputs_[input.seq % kHistory] = input;
  nextSeq_ = input.seq + 1;

  previous_ = current_;
  stepMovement(current_, input, world, blocks, kSimDt);
}

void Prediction::reconcile(const MoveState &server, uint32_t ackInputSeq,
                           const atm::voxel::IBlockAccess &world,
                           const atm::voxel::BlockRegistry &blocks) {
  // Ignore acks for inputs we no longer have or that are older than the last
  // reconciliation (out-of-order snapshots).
  if (ackInputSeq + 1 < oldestSeq_ || ackInputSeq >= nextSeq_)
    return;
  oldestSeq_ = ackInputSeq + 1;

  const glm::dvec3 before = current_.pos;
  MoveState replay = server;
  for (uint32_t seq = oldestSeq_; seq < nextSeq_; ++seq)
    stepMovement(replay, inputs_[seq % kHistory], world, blocks, kSimDt);

  const glm::dvec3 error = before - replay.pos;
  const double errLen = std::sqrt(error.x * error.x + error.y * error.y + error.z * error.z);
  lastCorrection_ = uint32_t(std::min(errLen * 1e6, 4e9));

  // Small errors are blended out visually; large ones (teleport, respawn,
  // blocked by an edit we hadn't seen) snap immediately.
  if (errLen < 2.0)
    correction_ += error;
  else
    correction_ = glm::dvec3(0.0);

  const glm::dvec3 shift = replay.pos - current_.pos;
  current_ = replay;
  previous_.pos += shift; // keep the interpolation pair consistent
}

glm::dvec3 Prediction::renderPosition(float alpha) const {
  const glm::dvec3 p = previous_.pos + (current_.pos - previous_.pos) * double(alpha);
  return p + correction_;
}

void Prediction::decayCorrection(float dt) {
  // Exponential decay: ~90% of the error gone in 0.2 s.
  const double k = std::exp(-double(dt) * 11.5);
  correction_ *= k;
  if (std::abs(correction_.x) + std::abs(correction_.y) + std::abs(correction_.z) < 1e-4)
    correction_ = glm::dvec3(0.0);
}

int Prediction::recentInputs(MoveInput *out, int max) const {
  const uint32_t available = nextSeq_ > 1 ? std::min<uint32_t>(nextSeq_ - 1, uint32_t(kHistory)) : 0;
  const int n = int(std::min<uint32_t>(available, uint32_t(max)));
  for (int i = 0; i < n; ++i)
    out[i] = inputs_[(nextSeq_ - uint32_t(n) + uint32_t(i)) % kHistory];
  return n;
}

// ---------------------------------------------------------------------------

void RemoteTrack::push(const RemoteSample &s) {
  // Drop out-of-order samples.
  if (count_ > 0 && s.tick <= newestTick())
    return;
  samples_[size_t(head_)] = s;
  head_ = (head_ + 1) % kCap;
  count_ = std::min(count_ + 1, kCap);
}

float RemoteTrack::newestTick() const {
  if (count_ == 0)
    return 0.0f;
  return samples_[size_t((head_ + kCap - 1) % kCap)].tick;
}

RemoteSample RemoteTrack::sample(float renderTick) const {
  if (count_ == 0)
    return {};
  auto at = [&](int i) -> const RemoteSample & { // i = 0 oldest
    return samples_[size_t((head_ - count_ + i + kCap) % kCap)];
  };
  if (renderTick <= at(0).tick)
    return at(0);
  for (int i = 0; i + 1 < count_; ++i) {
    const RemoteSample &a = at(i);
    const RemoteSample &b = at(i + 1);
    if (renderTick >= a.tick && renderTick <= b.tick) {
      const float span = std::max(b.tick - a.tick, 1e-3f);
      const float t = (renderTick - a.tick) / span;
      RemoteSample out = b;
      out.pos = a.pos + (b.pos - a.pos) * double(t);
      out.vel = a.vel + (b.vel - a.vel) * t;
      float dy = b.yaw - a.yaw;
      while (dy > 3.14159265f) dy -= 6.2831853f;
      while (dy < -3.14159265f) dy += 6.2831853f;
      out.yaw = a.yaw + dy * t;
      out.tick = renderTick;
      return out;
    }
  }
  // Late data: extrapolate from the newest sample for at most 250 ms.
  const RemoteSample &n = at(count_ - 1);
  const float ahead = std::min(renderTick - n.tick, float(kSimHz) * 0.25f);
  RemoteSample out = n;
  out.pos += glm::dvec3(n.vel) * double(ahead / float(kSimHz));
  out.tick = renderTick;
  return out;
}

} // namespace ao::client
