#include "IconRender.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace atm::model {

namespace {

struct Rgba {
  float r, g, b, a;
};

Rgba unpack(uint32_t c) {
  return {float(c & 0xFF) / 255.0f, float((c >> 8) & 0xFF) / 255.0f, float((c >> 16) & 0xFF) / 255.0f, 1.0f};
}

uint32_t pack(const Rgba &c) {
  auto ch = [](float v) { return uint32_t(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
  return ch(c.r) | (ch(c.g) << 8) | (ch(c.b) << 16) | (ch(c.a) << 24);
}

} // namespace

void renderVoxelIcon(const VoxelPart &part, int size, std::vector<uint32_t> &out) {
  out.assign(size_t(size) * size, 0u);
  // Bounds of the filled voxels.
  glm::ivec3 lo(1 << 20), hi(-1);
  for (int y = 0; y < part.sy; ++y)
    for (int z = 0; z < part.sz; ++z)
      for (int x = 0; x < part.sx; ++x)
        if (part.at(x, y, z)) lo = glm::min(lo, glm::ivec3(x, y, z)), hi = glm::max(hi, glm::ivec3(x, y, z));
  if (hi.x < 0) return;
  const glm::vec3 bmin(lo), bmax(glm::vec3(hi) + 1.0f), centre = (bmin + bmax) * 0.5f, ext = bmax - bmin;

  // Camera: three-quarter view from the front-right, above.
  const float yaw = glm::radians(35.0f), pitch = glm::radians(28.0f);
  const glm::vec3 eyeDir(std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch));
  const glm::vec3 fwd = -eyeDir;
  glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));
  glm::vec3 up = glm::cross(right, fwd);
  // Long, thin items: roll the view so their long axis runs corner to corner.
  const float longest = std::max({ext.x, ext.y, ext.z});
  const float others = ext.x + ext.y + ext.z - longest;
  if (longest > 1.4f * others * 0.5f + 2.0f) {
    const glm::vec3 axis = ext.x == longest ? glm::vec3(1, 0, 0) : ext.y == longest ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1);
    glm::vec2 p(glm::dot(axis, right), glm::dot(axis, up));
    if (glm::length(p) > 1e-3f) {
      p = glm::normalize(p);
      const float roll = std::atan2(p.y, p.x) - glm::radians(45.0f);
      const glm::vec3 r2 = right * std::cos(roll) + up * std::sin(roll);
      const glm::vec3 u2 = -right * std::sin(roll) + up * std::cos(roll);
      right = r2, up = u2;
    }
  }
  // Fit the projected bounding box.
  float ex = 0.0f, ey = 0.0f;
  for (int i = 0; i < 8; ++i) {
    const glm::vec3 c((i & 1) ? bmax.x : bmin.x, (i & 2) ? bmax.y : bmin.y, (i & 4) ? bmax.z : bmin.z);
    ex = std::max(ex, std::fabs(glm::dot(c - centre, right)));
    ey = std::max(ey, std::fabs(glm::dot(c - centre, up)));
  }
  const float half = std::max(ex, ey) / 0.86f; // world units from the centre to the image edge

  const glm::vec3 light = glm::normalize(glm::vec3(-0.35f, 0.85f, -0.4f));
  const int N = size * 2; // supersampled
  std::vector<Rgba> hi_(size_t(N) * N, Rgba{0, 0, 0, 0});
  const float far = glm::length(ext) + 4.0f;
  const glm::vec3 boxMin(0.0f), boxMax(float(part.sx), float(part.sy), float(part.sz));
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i) {
      const float u = ((i + 0.5f) / N * 2.0f - 1.0f) * half, v = (1.0f - (j + 0.5f) / N * 2.0f) * half;
      const glm::vec3 o = centre + right * u + up * v + eyeDir * far;
      const glm::vec3 d = fwd;
      // Slab test against the part's grid.
      float t0 = 0.0f, t1 = 1e9f;
      for (int a = 0; a < 3; ++a) {
        if (std::fabs(d[a]) < 1e-8f) {
          if (o[a] < boxMin[a] || o[a] > boxMax[a]) t0 = 1e9f;
          continue;
        }
        float ta = (boxMin[a] - o[a]) / d[a], tb = (boxMax[a] - o[a]) / d[a];
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(t0, ta), t1 = std::min(t1, tb);
      }
      if (t0 >= t1) continue;
      // DDA through the voxels.
      glm::vec3 p = o + d * (t0 + 1e-4f);
      glm::ivec3 cell = glm::clamp(glm::ivec3(glm::floor(p)), glm::ivec3(0), glm::ivec3(part.sx - 1, part.sy - 1, part.sz - 1));
      const glm::ivec3 step(d.x > 0 ? 1 : -1, d.y > 0 ? 1 : -1, d.z > 0 ? 1 : -1);
      glm::vec3 tMax, tDelta;
      for (int a = 0; a < 3; ++a) {
        tDelta[a] = std::fabs(d[a]) < 1e-8f ? 1e9f : std::fabs(1.0f / d[a]);
        const float next = step[a] > 0 ? float(cell[a] + 1) : float(cell[a]);
        tMax[a] = std::fabs(d[a]) < 1e-8f ? 1e9f : (next - p[a]) / d[a];
      }
      // Entry face: the slab we came through.
      int axis = 0;
      {
        const glm::vec3 fromMin = glm::abs(p - boxMin), fromMax = glm::abs(p - boxMax);
        float best = 1e9f;
        for (int a = 0; a < 3; ++a)
          if (std::min(fromMin[a], fromMax[a]) < best) best = std::min(fromMin[a], fromMax[a]), axis = a;
      }
      for (int guard = 0; guard < part.sx + part.sy + part.sz + 3; ++guard) {
        if (cell.x < 0 || cell.y < 0 || cell.z < 0 || cell.x >= part.sx || cell.y >= part.sy || cell.z >= part.sz) break;
        const uint8_t idx = part.at(cell.x, cell.y, cell.z);
        if (idx && idx < part.palette.size()) {
          glm::vec3 n(0.0f);
          n[axis] = -float(step[axis]);
          const float lambert = std::max(0.0f, glm::dot(n, light));
          float shade = 0.52f + 0.55f * lambert + (n.y > 0.5f ? 0.08f : 0.0f);
          const bool glow = idx >= part.emissiveFrom;
          if (glow) shade = 1.15f;
          Rgba c = unpack(part.palette[idx]);
          c.r *= shade, c.g *= shade, c.b *= shade;
          hi_[size_t(j) * N + i] = c;
          break;
        }
        if (tMax.x < tMax.y && tMax.x < tMax.z) cell.x += step.x, tMax.x += tDelta.x, axis = 0;
        else if (tMax.y < tMax.z) cell.y += step.y, tMax.y += tDelta.y, axis = 1;
        else cell.z += step.z, tMax.z += tDelta.z, axis = 2;
      }
    }
  // Downsample 2x2 (premultiplied).
  std::vector<Rgba> img(size_t(size) * size, Rgba{0, 0, 0, 0});
  for (int j = 0; j < size; ++j)
    for (int i = 0; i < size; ++i) {
      Rgba s{0, 0, 0, 0};
      for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
          const Rgba &h = hi_[size_t(j * 2 + dy) * N + (i * 2 + dx)];
          s.r += h.r * h.a, s.g += h.g * h.a, s.b += h.b * h.a, s.a += h.a;
        }
      if (s.a > 0.0f) img[size_t(j) * size + i] = {s.r / s.a, s.g / s.a, s.b / s.a, s.a / 4.0f};
    }
  // Dark outline (RuneScape-style) and a soft drop shadow, under the icon.
  auto alphaAt = [&](int x, int y) {
    return (x < 0 || y < 0 || x >= size || y >= size) ? 0.0f : img[size_t(y) * size + x].a;
  };
  for (int j = 0; j < size; ++j)
    for (int i = 0; i < size; ++i) {
      Rgba c = img[size_t(j) * size + i];
      float edge = 0.0f;
      for (auto [dx, dy] : {std::pair{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) edge = std::max(edge, alphaAt(i + dx, j + dy));
      const float shadow = alphaAt(i - 2, j - 2) * 0.35f;
      const float under = std::max(edge * 0.9f, shadow);
      // Composite: icon over outline / shadow (both near-black).
      const float a = c.a + under * (1.0f - c.a);
      if (a <= 0.0f) continue;
      const float k = 0.08f;
      Rgba o{(c.r * c.a + k * under * (1.0f - c.a)) / a, (c.g * c.a + k * 0.9f * under * (1.0f - c.a)) / a,
             (c.b * c.a + k * 0.8f * under * (1.0f - c.a)) / a, a};
      out[size_t(j) * size + i] = pack(o);
    }
}

} // namespace atm::model
