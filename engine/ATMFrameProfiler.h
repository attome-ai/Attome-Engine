#ifndef ATM_FRAME_PROFILER_H
#define ATM_FRAME_PROFILER_H

// Lightweight hierarchical frame profiler (main thread).
//
//   ATM_PROFILE_SCOPE("Render");       // times the enclosing block
//   atm::prof::Profiler::get().beginFrame() / endFrame()   once per frame
//
// Compile-time switch: ATM_PROFILING (CMake option ATTOME_PROFILING, default
// ON). With ATM_PROFILING=0 every macro compiles to nothing.
// Runtime switch: Profiler::get().setEnabled(bool); when disabled a scope
// costs one branch. Zones form a tree by nesting; each zone keeps a smoothed
// average, a peak over the last second and calls per frame. Scopes on other
// threads are ignored (the tree belongs to the thread that calls beginFrame).
//
// Differs from the 2D engine's ATMProfiler.h (string map + mutex + console
// report): zones are keyed by (name pointer, parent) in a flat vector, no
// allocation after warm-up, and the data is meant for an in-game panel.

#ifndef ATM_PROFILING
#define ATM_PROFILING 1
#endif

#include <cstdint>
#include <vector>

namespace atm::prof {

struct Zone {
  const char *name = nullptr;   // string literal (compared by pointer)
  int parent = -1;
  double frameMs = 0.0;         // this frame (accumulated over calls)
  uint32_t frameCalls = 0;
  double avgMs = 0.0;           // exponentially smoothed
  double peakMs = 0.0;          // max over the current / last second
  double peakWindowMs = 0.0;
  float callsPerFrame = 0.0f;
  uint32_t lastSeenFrame = 0;
};

class Profiler {
public:
  static Profiler &get();

  void setEnabled(bool on);
  bool enabled() const { return enabled_; }

  void beginFrame();
  void endFrame();

  // Scope internals.
  int enter(const char *name);
  void leave(int zone, uint64_t startTicks);
  static uint64_t now();

  const std::vector<Zone> &zones() const { return zones_; }
  double frameAvgMs() const { return frameAvgMs_; }
  uint32_t frameIndex() const { return frame_; }

private:
  bool enabled_ = false;
  bool inFrame_ = false;
  uint64_t frameStart_ = 0;
  double frameAvgMs_ = 0.0;
  double windowStart_ = 0.0;
  uint32_t frame_ = 0;
  uint64_t threadId_ = 0;
  std::vector<Zone> zones_;
  std::vector<int> stack_;
};

class Scope {
public:
  explicit Scope(const char *name) {
    Profiler &p = Profiler::get();
    if (p.enabled()) {
      zone_ = p.enter(name);
      if (zone_ >= 0)
        start_ = Profiler::now();
    }
  }
  ~Scope() {
    if (zone_ >= 0)
      Profiler::get().leave(zone_, start_);
  }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;

private:
  int zone_ = -1;
  uint64_t start_ = 0;
};

} // namespace atm::prof

#define ATM_PROF_CAT2(a, b) a##b
#define ATM_PROF_CAT(a, b) ATM_PROF_CAT2(a, b)
#if ATM_PROFILING
#define ATM_PROFILE_SCOPE(name) ::atm::prof::Scope ATM_PROF_CAT(atmProfScope_, __LINE__)(name)
#else
#define ATM_PROFILE_SCOPE(name) ((void)0)
#endif

#endif // ATM_FRAME_PROFILER_H
