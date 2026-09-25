// AnimLibrary lookup + Animator runtime (two layers with cross-fades).
// The clips themselves are authored in AnimClips.cpp.

#include "Character.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace atm::model {

// ---------------------------------------------------------------------------
// AnimLibrary
// ---------------------------------------------------------------------------

const AnimClip *AnimLibrary::find(std::string_view name) const {
  const int i = indexOf(name);
  return i < 0 ? nullptr : &clips_[size_t(i)];
}

int AnimLibrary::indexOf(std::string_view name) const {
  for (size_t i = 0; i < clips_.size(); ++i)
    if (clips_[i].name == name)
      return int(i);
  return -1;
}

int AnimLibrary::add(AnimClip clip) {
  const int existing = indexOf(clip.name);
  if (existing >= 0) {
    clips_[size_t(existing)] = std::move(clip);
    return existing;
  }
  clips_.push_back(std::move(clip));
  return int(clips_.size() - 1);
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

namespace {

struct Pose {
  std::array<glm::quat, kBoneCount> rot;
  std::array<glm::vec3, kBoneCount> off;
  Pose() {
    rot.fill(glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    off.fill(glm::vec3(0.0f));
  }
};

bool validClip(const AnimLibrary &lib, int clip) {
  return clip >= 0 && size_t(clip) < lib.size();
}

// Smoothstep between keys: soft in/out, and with the snappy key timing the
// clips use it reads as Trove's "pose to pose" motion.
float ease(float t) { return t * t * (3.0f - 2.0f * t); }

void sampleTrack(const std::vector<BoneKey> &keys, float t, glm::quat &rot, glm::vec3 &off) {
  if (keys.empty())
    return;
  if (keys.size() == 1 || t <= keys.front().time) {
    rot = keys.front().rotation;
    off = keys.front().offset;
    return;
  }
  if (t >= keys.back().time) {
    rot = keys.back().rotation;
    off = keys.back().offset;
    return;
  }
  size_t i = 1;
  while (i < keys.size() && keys[i].time < t)
    ++i;
  const BoneKey &a = keys[i - 1];
  const BoneKey &b = keys[i];
  const float span = b.time - a.time;
  const float u = span > 1e-6f ? ease(std::clamp((t - a.time) / span, 0.0f, 1.0f)) : 1.0f;
  rot = glm::slerp(a.rotation, b.rotation, u);
  off = glm::mix(a.offset, b.offset, u);
}

void sampleClip(const AnimClip &clip, float t, Pose &pose) {
  for (int b = 0; b < kBoneCount; ++b)
    sampleTrack(clip.tracks[size_t(b)], t, pose.rot[size_t(b)], pose.off[size_t(b)]);
}

void blendInto(Pose &dst, const Pose &src, float w, uint32_t mask) {
  for (int b = 0; b < kBoneCount; ++b) {
    if (!(mask & (1u << b)))
      continue;
    dst.rot[size_t(b)] = glm::slerp(dst.rot[size_t(b)], src.rot[size_t(b)], w);
    dst.off[size_t(b)] = glm::mix(dst.off[size_t(b)], src.off[size_t(b)], w);
  }
}

// Collects events with time in (a, b] (or [a, b] when inclusiveStart).
uint32_t eventsIn(const AnimClip &clip, float a, float b, bool inclusiveStart) {
  uint32_t bits = 0;
  for (const auto &e : clip.events) {
    const bool afterStart = inclusiveStart ? e.first >= a : e.first > a;
    if (afterStart && e.first <= b && e.second < 32)
      bits |= 1u << e.second;
  }
  return bits;
}

// Advances a clip time; returns events crossed and whether a one-shot ended.
uint32_t advance(const AnimClip &clip, float &time, float dt, bool &ended) {
  ended = false;
  const float len = std::max(clip.length, 1e-3f);
  const float t0 = time;
  const bool first = t0 <= 0.0f;
  float t1 = t0 + dt;
  uint32_t bits = 0;
  if (clip.loop) {
    if (t1 >= len) {
      bits |= eventsIn(clip, t0, len, first);
      t1 = std::fmod(t1, len);
      bits |= eventsIn(clip, 0.0f, t1, true);
    } else {
      bits |= eventsIn(clip, t0, t1, first);
    }
  } else {
    if (t1 >= len) {
      t1 = len;
      ended = true;
    }
    bits |= eventsIn(clip, t0, t1, first);
  }
  time = t1;
  return bits;
}

} // namespace

// ---------------------------------------------------------------------------
// Animator
// ---------------------------------------------------------------------------

void Animator::setLocomotion(int clip, float fadeSeconds) {
  if (clip == loco_.clip)
    return;
  loco_.prevClip = loco_.clip;
  loco_.prevTime = loco_.time;
  loco_.clip = clip;
  loco_.time = 0.0f;
  loco_.fade = 0.0f;
  loco_.fadeLen = loco_.prevClip >= 0 ? std::max(fadeSeconds, 0.0f) : 0.0f;
}

void Animator::playAction(int clip, float fadeSeconds) {
  action_.prevClip = action_.clip;
  action_.prevTime = action_.time;
  action_.clip = clip;
  action_.time = 0.0f;
  action_.fade = 0.0f;
  action_.fadeLen = std::max(fadeSeconds, 0.0f);
}

void Animator::update(const AnimLibrary &lib, float dt) {
  if (!(dt > 0.0f))
    return;
  bool ended = false;

  // Locomotion layer.
  if (validClip(lib, loco_.clip))
    events_ |= advance(lib.at(loco_.clip), loco_.time, dt, ended);
  if (validClip(lib, loco_.prevClip)) {
    bool prevEnded = false;
    advance(lib.at(loco_.prevClip), loco_.prevTime, dt, prevEnded);
  }
  loco_.fade += dt;
  if (loco_.fade >= loco_.fadeLen)
    loco_.prevClip = -1;

  // Action layer (one-shots end by themselves).
  if (validClip(lib, action_.clip)) {
    events_ |= advance(lib.at(action_.clip), action_.time, dt, ended);
    if (ended) {
      action_.clip = -1;
      action_.prevClip = -1;
    }
  } else {
    action_.clip = -1;
  }
  if (validClip(lib, action_.prevClip)) {
    bool prevEnded = false;
    advance(lib.at(action_.prevClip), action_.prevTime, dt, prevEnded);
  }
  action_.fade += dt;
  if (action_.fade >= action_.fadeLen)
    action_.prevClip = -1;
}

void Animator::evaluate(const AnimLibrary &lib, const Rig &rig,
                        std::array<glm::mat4, kBoneCount> &out) const {
  Pose pose;
  if (validClip(lib, loco_.clip))
    sampleClip(lib.at(loco_.clip), loco_.time, pose);
  if (validClip(lib, loco_.prevClip) && loco_.fadeLen > 0.0f && loco_.fade < loco_.fadeLen) {
    Pose prev;
    sampleClip(lib.at(loco_.prevClip), loco_.prevTime, prev);
    // blend from prev toward current
    const float w = std::clamp(loco_.fade / loco_.fadeLen, 0.0f, 1.0f);
    Pose cur = pose;
    pose = prev;
    blendInto(pose, cur, w, 0xFFFFFFFFu);
  }

  if (validClip(lib, action_.clip)) {
    const AnimClip &clip = lib.at(action_.clip);
    Pose act;
    sampleClip(clip, action_.time, act);
    float w = 1.0f;
    if (action_.fadeLen > 0.0f)
      w = std::min(w, action_.fade / action_.fadeLen);
    Pose base = pose;
    if (validClip(lib, action_.prevClip) && action_.fadeLen > 0.0f &&
        action_.fade < action_.fadeLen) {
      // Action interrupted by another action: fade from the old action.
      Pose prev;
      sampleClip(lib.at(action_.prevClip), action_.prevTime, prev);
      blendInto(base, prev, 1.0f, lib.at(action_.prevClip).boneMask);
    }
    if (!clip.loop) {
      const float fadeOut = 0.1f;
      w = std::min(w, (clip.length - action_.time) / fadeOut);
    }
    w = std::clamp(w, 0.0f, 1.0f);
    pose = base;
    blendInto(pose, act, w, clip.boneMask);
  }

  // Model-space matrices, parents first (Bone enum order is parent-first).
  for (int b = 0; b < kBoneCount; ++b) {
    const RigBone &rb = rig.bones[size_t(b)];
    glm::mat4 local = glm::translate(glm::mat4(1.0f), rb.offset + pose.off[size_t(b)]) *
                      glm::mat4_cast(glm::normalize(pose.rot[size_t(b)]));
    const int parent = int(rb.parent);
    out[size_t(b)] = (b == 0 || parent >= b) ? local : out[size_t(parent)] * local;
  }
}

} // namespace atm::model
