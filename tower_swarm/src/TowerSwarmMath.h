#pragma once

#include <cmath>
#include <cstdint>

namespace tower_swarm {

struct Vec2 final {
  float x = 0.0f;
  float y = 0.0f;
};

inline constexpr Vec2 make_vec2(float x, float y) { return Vec2{x, y}; }

inline constexpr Vec2 operator+(Vec2 a, Vec2 b) {
  return Vec2{a.x + b.x, a.y + b.y};
}
inline constexpr Vec2 operator-(Vec2 a, Vec2 b) {
  return Vec2{a.x - b.x, a.y - b.y};
}
inline constexpr Vec2 operator*(Vec2 v, float s) { return Vec2{v.x * s, v.y * s}; }

inline constexpr float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
inline constexpr float length_sq(Vec2 v) { return dot(v, v); }
inline float length(Vec2 v) { return std::sqrt(length_sq(v)); }

inline Vec2 safe_normalize(Vec2 v, float eps = 1e-6f) {
  const float len_sq = length_sq(v);
  if (len_sq <= eps * eps) {
    return Vec2{0.0f, 0.0f};
  }
  const float inv = 1.0f / std::sqrt(len_sq);
  return Vec2{v.x * inv, v.y * inv};
}

inline constexpr float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

inline constexpr float lerpf(float a, float b, float t) {
  return a + (b - a) * t;
}

inline constexpr float minf(float a, float b) { return a < b ? a : b; }
inline constexpr float maxf(float a, float b) { return a > b ? a : b; }

} // namespace tower_swarm
