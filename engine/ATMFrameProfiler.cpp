#include "ATMFrameProfiler.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <thread>

namespace atm::prof {

namespace {

uint64_t thisThread() { return uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id())); }

double toMs(uint64_t ticks) { return double(ticks) / 1.0e6; } // now() is in ns

} // namespace

Profiler &Profiler::get() {
  static Profiler p;
  return p;
}

uint64_t Profiler::now() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

void Profiler::setEnabled(bool on) {
  enabled_ = on;
  if (!on) {
    stack_.clear();
    inFrame_ = false;
  }
}

void Profiler::beginFrame() {
  if (!enabled_)
    return;
  threadId_ = thisThread();
  frameStart_ = now();
  stack_.clear();
  inFrame_ = true;
}

void Profiler::endFrame() {
  if (!enabled_ || !inFrame_)
    return;
  inFrame_ = false;
  ++frame_;
  const double frameMs = toMs(now() - frameStart_);
  frameAvgMs_ = frameAvgMs_ <= 0.0 ? frameMs : frameAvgMs_ * 0.95 + frameMs * 0.05;
  const double t = toMs(now());
  const bool newWindow = t - windowStart_ >= 1000.0;
  if (newWindow)
    windowStart_ = t;
  for (Zone &z : zones_) {
    z.avgMs = z.avgMs * 0.95 + z.frameMs * 0.05;
    z.callsPerFrame = z.callsPerFrame * 0.95f + float(z.frameCalls) * 0.05f;
    z.peakWindowMs = std::max(z.peakWindowMs, z.frameMs);
    if (newWindow) {
      z.peakMs = z.peakWindowMs;
      z.peakWindowMs = 0.0;
    }
    z.frameMs = 0.0;
    z.frameCalls = 0;
  }
}

int Profiler::enter(const char *name) {
  if (!inFrame_ || thisThread() != threadId_)
    return -1;
  const int parent = stack_.empty() ? -1 : stack_.back();
  int id = -1;
  for (size_t i = 0; i < zones_.size(); ++i)
    if (zones_[i].name == name && zones_[i].parent == parent) {
      id = int(i);
      break;
    }
  if (id < 0) {
    Zone z;
    z.name = name;
    z.parent = parent;
    zones_.push_back(z);
    id = int(zones_.size() - 1);
  }
  stack_.push_back(id);
  return id;
}

void Profiler::leave(int zone, uint64_t startTicks) {
  if (zone < 0 || size_t(zone) >= zones_.size())
    return;
  Zone &z = zones_[size_t(zone)];
  z.frameMs += toMs(now() - startTicks);
  ++z.frameCalls;
  z.lastSeenFrame = frame_;
  if (!stack_.empty() && stack_.back() == zone)
    stack_.pop_back();
}

} // namespace atm::prof
