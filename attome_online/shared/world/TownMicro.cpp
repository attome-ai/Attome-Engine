#include "TownMicro.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace ao::world::townmicro {

namespace {

constexpr int K = kMicro;
constexpr float kPi = 3.14159265f;

uint32_t hash3(int x, int y, int z, uint32_t seed) {
  uint32_t h = seed ^ (uint32_t(x) * 73856093u) ^ (uint32_t(y) * 19349663u) ^ (uint32_t(z) * 83492791u);
  h ^= h >> 13;
  h *= 0x5bd1e995u;
  h ^= h >> 15;
  return h;
}
float rnd(int x, int y, int z, uint32_t seed) { return float(hash3(x, y, z, seed) & 0xFFFFu) / 65535.0f; }

// Draws into a MicroModel in town-relative fine coordinates (block * K).
struct Canvas {
  MicroModel &m;
  int ox, oy, oz;
  // localY: y coordinates are model-local (0 = the model's bottom) instead of town-relative.
  explicit Canvas(MicroModel &mm, bool localY = false)
      : m(mm), ox(mm.bx * K), oy(localY ? 0 : mm.by * K), oz(mm.bz * K) {}

  void set(int x, int y, int z, uint8_t c) { m.set(x - ox, y - oy, z - oz, c); }
  uint8_t at(int x, int y, int z) const { return m.at(x - ox, y - oy, z - oz); }
  void box(int x0, int y0, int z0, int x1, int y1, int z1, uint8_t c) {
    m.box(x0 - ox, y0 - oy, z0 - oz, x1 - ox, y1 - oy, z1 - oz, c);
  }
  void speckle(int x0, int y0, int z0, int x1, int y1, int z1, uint8_t c, float chance, uint32_t seed) {
    m.speckle(x0 - ox, y0 - oy, z0 - oz, x1 - ox, y1 - oy, z1 - oz, c, chance, seed);
  }
  void recolor(int x0, int y0, int z0, int x1, int y1, int z1, uint8_t from, uint8_t to) {
    m.recolor(x0 - ox, y0 - oy, z0 - oz, x1 - ox, y1 - oy, z1 - oz, from, to);
  }
  void cyl(float cx, float cz, float r, int y0, int y1, uint8_t c, float thick = 0.0f) {
    m.cylinder(cx - float(ox), cz - float(oz), r, y0 - oy, y1 - oy, c, thick);
  }

  // Ashlar stonework: mortar lines every 6 up, staggered joints every 12.
  void ashlar(int x0, int y0, int z0, int x1, int y1, int z1, uint8_t stone, uint8_t joint) {
    box(x0, y0, z0, x1, y1, z1, stone);
    for (int y = std::min(y0, y1); y <= std::max(y0, y1); ++y)
      for (int z = std::min(z0, z1); z <= std::max(z0, z1); ++z)
        for (int x = std::min(x0, x1); x <= std::max(x0, x1); ++x) {
          const int row = (y - 1000) / 6, along = x + z + (row & 1) * 6;
          if (((y % 6) + 6) % 6 == 0 || ((along % 12) + 12) % 12 == 0) set(x, y, z, joint);
        }
  }

  // Cone roof: shells with stripes, gold band at the eaves, spire and flag.
  void cone(float cx, float cz, float r, int y0, int h, bool flag, uint8_t c1 = mc::RoofBlue,
            uint8_t c2 = mc::RoofBlueDark, uint8_t c3 = mc::RoofBlueLight) {
    for (int k = 0; k < h; ++k) {
      const float rk = r * (1.0f - float(k) / float(h)) + 0.4f;
      const int band = k / 3;
      const uint8_t c = k < 2 ? mc::Gold : band % 4 == 3 ? c3 : band % 2 ? c2 : c1;
      cyl(cx, cz, rk, y0 + k, y0 + k, c, 2.6f);
      if (k == h - 1) cyl(cx, cz, rk, y0 + k, y0 + k, c);
    }
    const int ix = int(std::floor(cx)), iz = int(std::floor(cz));
    box(ix - 1, y0 + h, iz - 1, ix, y0 + h + 5, iz, mc::Gold); // spire
    box(ix - 2, y0 + h + 6, iz - 2, ix + 1, y0 + h + 8, iz + 1, mc::GlowGold);
    if (flag) {
      box(ix, y0 + h + 9, iz, ix, y0 + h + 18, iz, mc::Iron);
      for (int j = 0; j < 10; ++j) {
        const int wave = int(std::lround(std::sin(j * 0.8f)));
        box(ix + 1 + j, y0 + h + 13, iz + wave, ix + 1 + j, y0 + h + 18 - (j > 7 ? j - 7 : 0), iz + wave,
            j % 5 == 4 ? mc::Gold : mc::BannerBlue);
      }
    }
  }

  // Battlements along a straight wall top (merlons 5 wide, gaps 3).
  void merlons(int x0, int z0, int x1, int z1, int y, int h, uint8_t c, uint8_t cap) {
    const bool alongX = std::abs(x1 - x0) >= std::abs(z1 - z0);
    const int a0 = alongX ? std::min(x0, x1) : std::min(z0, z1), a1 = alongX ? std::max(x0, x1) : std::max(z0, z1);
    for (int a = a0; a <= a1; ++a) {
      if (((a - a0) % 8) >= 5) continue;
      if (alongX) box(a, y, z0, a, y + h - 1, z1, c), box(a, y + h, z0, a, y + h, z1, cap);
      else box(x0, y, a, x1, y + h - 1, a, c), box(x0, y + h, a, x1, y + h, a, cap);
    }
  }

  // A glowing window through a wall. The wall plane is at `face` (x for
  // walls along z, z for walls along x), outward sign `out`, thickness th.
  void window(bool wallAlongX, int face, int out, int thick, int a0, int y0, int w, int h, bool arched,
              uint8_t frame = mc::StoneTrim, uint8_t glass = mc::GlowWindow) {
    auto S = [&](int a, int y, int d, uint8_t c) {
      const int p = face - out * d; // d = 0 outer layer, < 0 outside
      if (wallAlongX) set(a, y, p, c);
      else set(p, y, a, c);
    };
    for (int j = 0; j < h; ++j)
      for (int i = 0; i < w; ++i) {
        if (arched && j >= h - 2 && (i == 0 || i == w - 1)) continue;
        if (arched && j == h - 1 && (i == 1 || i == w - 2) && w > 4) continue;
        for (int d = 0; d < thick; ++d) S(a0 + i, y0 + j, d, glass);
      }
    // Frame one voxel proud of the wall, sill below.
    for (int j = -1; j <= h; ++j)
      for (int i = -1; i <= w; ++i) {
        const bool border = i == -1 || i == w || j == -1 || j == h;
        if (!border) continue;
        if (arched && j == h && (i <= 0 || i >= w - 1)) continue;
        S(a0 + i, y0 + j, -1, frame);
      }
    for (int i = -2; i <= w + 1; ++i) S(a0 + i, y0 - 1, -2, frame);
    if (w >= 5) // mullion
      for (int j = 0; j < h; ++j) S(a0 + w / 2, y0 + j, 0, mc::Iron);
  }

  // Hanging banner in front of a wall: border, emblem, swallowtail.
  void banner(bool wallAlongX, int face, int out, int a0, int yTop, int w, int h) {
    auto S = [&](int a, int y, uint8_t c) {
      const int p = face + out;
      if (wallAlongX) set(a, y, p, c);
      else set(p, y, a, c);
    };
    for (int i = -1; i <= w; ++i) S(a0 + i, yTop + 1, mc::Gold); // rod
    for (int j = 0; j < h; ++j)
      for (int i = 0; i < w; ++i) {
        const int y = yTop - j;
        const int tail = j - (h - 4);
        if (tail > 0 && std::abs(i - w / 2) < tail) continue; // swallowtail
        const bool border = i == 0 || i == w - 1 || j == 0;
        uint8_t c = border ? mc::Gold : mc::BannerBlue;
        // Emblem: a gold diamond with a bright core.
        const int ey = h / 2 - 1, ex = w / 2;
        const int dd = std::abs(i - ex) + std::abs(j - ey);
        if (!border && dd <= w / 3) c = dd <= 1 ? mc::GlowGold : mc::Gold;
        if (!border && dd == w / 3 + 1) c = mc::BannerBlueDark;
        S(a0 + i, y, c);
      }
  }

  // Round tower with batter, windows, corbels, crenels and a cone roof.
  void tower(float cx, float cz, float r, int y0, int wallH, int roofH, bool flag) {
    const int top = y0 + wallH;
    cyl(cx, cz, r + 2.0f, y0, y0 + 5, mc::StoneTrim);                  // batter
    cyl(cx, cz, r, y0 + 6, top, mc::Stone, 5.0f);
    // Mortar courses.
    for (int y = y0 + 6; y <= top; y += 6) cyl(cx, cz, r, y, y, mc::StoneShade, 1.2f);
    cyl(cx, cz, r, top - 8, top - 7, mc::Gold, 1.2f);                   // gold band
    // Corbels and a machicolation ring.
    for (int k = 0; k < 24; ++k) {
      const float a = float(k) / 24.0f * 2.0f * kPi;
      const int x = int(std::floor(cx + std::cos(a) * (r + 1.2f))), z = int(std::floor(cz + std::sin(a) * (r + 1.2f)));
      box(x, top - 4, z, x, top - 1, z, mc::StoneTrim);
    }
    cyl(cx, cz, r + 3.0f, top, top + 2, mc::StoneTrim, 4.0f);
    // Crenels on the ring.
    for (int k = 0; k < 20; ++k) {
      if (k % 2) continue;
      const float a0 = float(k) / 20.0f * 2.0f * kPi;
      for (float a = a0; a < a0 + 2.0f * kPi / 20.0f; a += 0.03f)
        for (float rr = r + 0.5f; rr <= r + 3.0f; rr += 0.7f) {
          const int x = int(std::floor(cx + std::cos(a) * rr)), z = int(std::floor(cz + std::sin(a) * rr));
          box(x, top + 3, z, x, top + 6, z, mc::Stone);
        }
    }
    // Arched windows on four sides, every 18 up.
    const int ix = int(std::floor(cx)), iz = int(std::floor(cz)), ri = int(r);
    for (int y = y0 + 12; y < top - 14; y += 18) {
      window(true, iz + ri - 1, 1, 5, ix - 2, y, 4, 9, true);
      window(true, iz - ri, -1, 5, ix - 2, y, 4, 9, true);
      window(false, ix + ri - 1, 1, 5, iz - 2, y, 4, 9, true);
      window(false, ix - ri, -1, 5, iz - 2, y, 4, 9, true);
    }
    cone(cx, cz, r + 2.5f, top + 3, roofH, flag);
  }

  // Hipped roof over a fine rectangle, rising 5 per 4, striped, gold crest.
  void hipRoof(int x0, int z0, int x1, int z1, int y0, uint8_t c1, uint8_t c2, uint8_t c3) {
    for (int z = z0; z <= z1; ++z)
      for (int x = x0; x <= x1; ++x) {
        const int s = std::min({x - x0, x1 - x, z - z0, z1 - z});
        const int yS = y0 + (s * 5) / 4;
        const int band = s / 2;
        const uint8_t c = s == 0 ? mc::WoodDark : band % 5 == 4 ? c3 : band % 2 ? c2 : c1;
        box(x, yS - 2, z, x, yS, z, c);
      }
    // Ridge crest.
    const int smax = std::min(x1 - x0, z1 - z0) / 2;
    const int yR = y0 + (smax * 5) / 4 + 1;
    for (int z = z0 + smax; z <= z1 - smax; ++z)
      for (int x = x0 + smax; x <= x1 - smax; ++x) {
        set(x, yR, z, mc::Gold);
        if (((x + z) % 6) == 0) box(x, yR + 1, z, x, yR + 3, z, mc::Gold);
      }
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Houses: timber-framed cottages, fine detail everywhere
// ---------------------------------------------------------------------------

MicroModel house(const HouseSpec &h) {
  const int W = h.x1 - h.x0 + 1, D = h.z1 - h.z0 + 1;
  const bool ridgeX = W >= D;
  const int U = (ridgeX ? W : D) * K, V = (ridgeX ? D : W) * K;
  const int F = std::clamp(h.floors, 1, 3);
  const int o = 3;                       // eave overhang (fine)
  const int yT = 4 + F * 14;             // wall top
  const int smax = (V + 2 * o - 1) / 2;
  const int roofTop = yT + (smax * 5) / 4 + 6;
  const int H = roofTop / K + 3;
  MicroModel m(h.x0 - 1, 1, h.z0 - 1, W + 2, H, D + 2);
  const int mx = K, mz = K; // wall footprint offset inside the model (one block margin)
  auto P = [&](int u, int y, int v, uint8_t c) {
    if (ridgeX) m.set(mx + u, y, mz + v, c);
    else m.set(mx + v, y, mz + u, c);
  };
  auto PB = [&](int u0, int y0, int v0, int u1, int y1, int v1, uint8_t c) {
    for (int y = std::min(y0, y1); y <= std::max(y0, y1); ++y)
      for (int v = std::min(v0, v1); v <= std::max(v0, v1); ++v)
        for (int u = std::min(u0, u1); u <= std::max(u0, u1); ++u) P(u, y, v, c);
  };
  const uint8_t roof1 = h.redRoof ? mc::RoofRed : mc::RoofBlue, roof2 = h.redRoof ? mc::RoofRedDark : mc::RoofBlueDark,
                roof3 = h.redRoof ? mc::Orange : mc::RoofBlueLight;

  // Plinth (one block) with a proud top course, plank floor inside.
  PB(0, 0, 0, U - 1, 3, V - 1, mc::StoneTrim);
  m.speckle(0, 0, 0, m.sx - 1, 3, m.sz - 1, mc::StoneDark, 0.22f, h.seed);
  PB(-1, 3, -1, U, 3, V, mc::StoneShade);
  PB(2, 3, 2, U - 3, 3, V - 3, mc::WoodLight);

  // Walls: plaster between a dark timber frame.
  auto faceLen = [&](int f) { return f < 2 ? U : V; };
  auto FP = [&](int f, int a, int y, int d, uint8_t c) { // face f, along a, depth d (0 outer, <0 outside)
    switch (f) {
    case 0: P(a, y, d, c); break;
    case 1: P(a, y, V - 1 - d, c); break;
    case 2: P(d, y, a, c); break;
    default: P(U - 1 - d, y, a, c); break;
    }
  };
  for (int f = 0; f < 4; ++f) {
    const int L = faceLen(f);
    for (int y = 4; y < yT; ++y)
      for (int a = 0; a < L; ++a)
        for (int d = 0; d < 2; ++d) {
          const int fy = (y - 4) % 14;
          uint8_t c = rnd(a + f * 1000, y, d, h.seed) < 0.1f ? mc::PlasterShade : mc::Plaster;
          const bool corner = a < 3 || a > L - 4;
          const bool stud = (a % 10) < 2;
          if (corner || stud || fy >= 12 || y < 6) c = mc::WoodDark;
          FP(f, a, y, d, c);
        }
    // Proud floor beams.
    for (int fl = 1; fl <= F; ++fl)
      for (int a = -1; a <= L; ++a) FP(f, a, 4 + fl * 14 - 1, -1, mc::WoodDark);
  }
  // Upper floors: plank ceilings.
  for (int fl = 1; fl < F; ++fl) PB(2, 4 + fl * 14 - 2, 2, U - 3, 4 + fl * 14 - 1, V - 3, mc::WoodLight);

  // Door face (local): which face points to (doorDX, doorDZ)?
  int df;
  if (ridgeX) df = h.doorDZ < 0 ? 0 : h.doorDZ > 0 ? 1 : h.doorDX < 0 ? 2 : 3;
  else df = h.doorDX < 0 ? 0 : h.doorDX > 0 ? 1 : h.doorDZ < 0 ? 2 : 3;
  const int dL = faceLen(df), dc = dL / 2;

  // Windows (glowing or glass) with frames, sills, mullions and flower
  // boxes; X-braces in the upper-floor bays without windows.
  for (int f = 0; f < 4; ++f) {
    const int L = faceLen(f);
    for (int fl = 0; fl < F; ++fl) {
      const int yb = 4 + fl * 14;
      for (int b = 0; b * 10 + 9 < L; ++b) {
        const int a0 = b * 10 + 2;
        if (a0 < 3 || a0 + 7 > L - 3) continue;
        if (f == df && fl == 0 && a0 <= dc + 5 && a0 + 7 >= dc - 5) continue; // the door
        const bool brace = fl > 0 && (b % 2 == 1);
        if (brace) {
          for (int t = 0; t < 8; ++t) {
            const int y1 = yb + 1 + (t * 10) / 8, y2 = yb + 11 - (t * 10) / 8;
            FP(f, a0 + t, y1, 0, mc::WoodDark), FP(f, a0 + t, y1 + 1, 0, mc::WoodDark);
            FP(f, a0 + t, y2, 0, mc::WoodDark), FP(f, a0 + t, y2 - 1, 0, mc::WoodDark);
          }
          continue;
        }
        const bool lit = rnd(b, fl, f, h.seed + 7) < 0.55f;
        for (int j = 0; j < 7; ++j)
          for (int i = 1; i <= 6; ++i) {
            FP(f, a0 + i, yb + 3 + j, 1, lit ? mc::GlowWindow : mc::Glass);
            FP(f, a0 + i, yb + 3 + j, 0, mc::Empty); // recess the outer layer
          }
        for (int j = 0; j < 7; ++j) FP(f, a0 + 3, yb + 3 + j, 1, mc::WoodDark);   // mullion
        for (int i = 1; i <= 6; ++i) FP(f, a0 + i, yb + 6, 1, mc::WoodDark);      // transom
        for (int j = -1; j <= 7; ++j) FP(f, a0, yb + 3 + j, 0, mc::WoodDark), FP(f, a0 + 7, yb + 3 + j, 0, mc::WoodDark);
        for (int i = 0; i <= 7; ++i) FP(f, a0 + i, yb + 10, 0, mc::WoodDark);
        for (int i = -1; i <= 8; ++i) FP(f, a0 + i, yb + 2, -1, mc::StoneTrim);   // sill
        // Shutters.
        for (int j = 0; j < 7; ++j) {
          FP(f, a0 - 1, yb + 3 + j, -1, roof2);
          FP(f, a0 + 8, yb + 3 + j, -1, roof2);
        }
        if (rnd(b, fl, f, h.seed + 9) < 0.6f) { // flower box
          for (int i = 0; i <= 7; ++i) {
            FP(f, a0 + i, yb + 1, -2, mc::Wood), FP(f, a0 + i, yb + 1, -3, mc::Wood);
            const float r = rnd(a0 + i, fl, f, h.seed + 11);
            const uint8_t fc = r < 0.35f ? mc::FlowerRed : r < 0.6f ? mc::FlowerYellow : mc::Leaf;
            FP(f, a0 + i, yb + 2, -2, mc::Leaf);
            FP(f, a0 + i, yb + 2, -3, fc);
            if (r > 0.8f) FP(f, a0 + i, yb + 3, -3, mc::FlowerRed);
          }
        }
      }
    }
  }

  // Door: open doorway with a timber frame, keystone, step and an open leaf;
  // a lantern on an iron bracket beside it.
  for (int y = 4; y <= 14; ++y)
    for (int a = dc - 3; a <= dc + 3; ++a)
      for (int d = 0; d < 2; ++d) FP(df, a, y, d, 0);
  for (int y = 4; y <= 15; ++y) FP(df, dc - 4, y, -1, mc::WoodDark), FP(df, dc + 4, y, -1, mc::WoodDark);
  for (int a = dc - 4; a <= dc + 4; ++a) FP(df, a, 15, -1, mc::WoodDark);
  FP(df, dc, 16, -1, mc::Gold);
  FP(df, dc - 3, 14, -1, mc::WoodDark), FP(df, dc + 3, 14, -1, mc::WoodDark);
  for (int a = dc - 4; a <= dc + 4; ++a)
    for (int d = -3; d <= -1; ++d) FP(df, a, 3, d, mc::StoneShade), FP(df, a, 2, d, mc::StoneTrim);
  for (int y = 4; y <= 14; ++y)
    for (int t = 0; t < 7; ++t) FP(df, dc + 4 + t, y, 2, (t % 2) ? mc::Wood : mc::WoodLight);
  for (int d = -3; d <= -1; ++d) FP(df, dc + 7, 18, d, mc::Iron);
  for (int y = 13; y <= 17; ++y)
    for (int a = dc + 6; a <= dc + 8; ++a)
      for (int d = -4; d <= -2; ++d)
        FP(df, a, y, d, (y == 13 || y == 17) ? mc::Iron : (a == dc + 7 && d == -3) ? mc::GlowLantern : mc::GlowWindow);
  FP(df, dc + 7, 12, -3, mc::Iron);

  // Roof: 45-ish degree slopes (5 up per 4 across), shingle bands, fascia,
  // barge boards, a ridge beam with gold crest; plastered gables with a
  // round window.
  auto roofY = [&](int v) {
    const int s = std::min(v + o, V - 1 + o - v);
    return yT + (s * 5) / 4 - 2;
  };
  for (int v = -o; v <= V - 1 + o; ++v) {
    const int s = std::min(v + o, V - 1 + o - v);
    const int yS = roofY(v);
    const int band = s / 2;
    const uint8_t c = s == 0 ? mc::WoodDark : band % 5 == 4 ? roof3 : band % 2 ? roof2 : roof1;
    for (int u = -2; u <= U + 1; ++u) {
      const bool barge = u == -2 || u == U + 1;
      for (int y = yS - 2; y <= yS; ++y) P(u, y, v, barge ? mc::WoodDark : c);
      // Shingle edges: a darker lip every other row.
      if (!barge && s % 2 == 0 && s > 0) P(u, yS, v, roof2);
    }
  }
  const int yRidge = yT + (smax * 5) / 4 - 1;
  for (int u = -2; u <= U + 1; ++u) {
    for (int v = V / 2 - 1; v <= V / 2; ++v) P(u, yRidge, v, mc::WoodDark);
    if (u % 6 == 0) {
      P(u, yRidge + 1, V / 2 - 1, mc::Gold), P(u, yRidge + 1, V / 2, mc::Gold);
      P(u, yRidge + 2, V / 2 - 1, mc::Gold);
    }
  }
  for (int u : {0, 1, U - 2, U - 1})
    for (int v = 0; v < V; ++v)
      for (int y = yT; y <= roofY(v) - 3; ++y) {
        uint8_t c = mc::Plaster;
        if (std::abs(v - V / 2) <= 0 || y <= yT + 1) c = mc::WoodDark;
        const float dy = float(y - (yT + 7)), dv = float(v) - (float(V) - 1.0f) * 0.5f;
        if (dv * dv + dy * dy <= 7.0f) c = mc::GlowWindow;
        else if (dv * dv + dy * dy <= 12.0f) c = mc::WoodDark;
        P(u, y, v, c);
      }

  // Chimney through the slope.
  const int cu = U - 9, cv = 4;
  const int cTop = roofY(cv + 3) + 7;
  PB(cu, 4, cv, cu + 3, cTop, cv + 3, mc::Cobble);
  for (int y = 4; y <= cTop; ++y)
    for (int v = cv; v <= cv + 3; ++v)
      for (int u = cu; u <= cu + 3; ++u)
        if (rnd(u, y, v, h.seed + 3) < 0.25f) P(u, y, v, mc::StoneDark);
  PB(cu - 1, cTop + 1, cv - 1, cu + 4, cTop + 2, cv + 4, mc::StoneDark);
  PB(cu + 1, cTop + 2, cv + 1, cu + 2, cTop + 2, cv + 2, mc::Iron);
  return m;
}

// ---------------------------------------------------------------------------
// Plaza: tiled square with a great sun emblem, cobbled ring street
// ---------------------------------------------------------------------------

MicroModel plazaFloor() {
  constexpr int RB = 22; // blocks
  MicroModel m(-RB, 0, -RB, 2 * RB + 1, 1, 2 * RB + 1);
  Canvas c(m);
  const float cx = 2.0f, cz = 2.0f; // centre of block (0, 0) in fine units
  for (int z = -RB * K; z < (RB + 1) * K; ++z)
    for (int x = -RB * K; x < (RB + 1) * K; ++x) {
      const float dx = x + 0.5f - cx, dz = z + 0.5f - cz;
      const float r = std::sqrt(dx * dx + dz * dz);
      if (r > 21.5f * K) continue;
      uint8_t top;
      bool groove = false;
      if (r <= 16.5f * K) {
        // Square tiles (8 fine) with grout grooves, two warm stone tones.
        const int tx = (x + 1000) / 8, tz = (z + 1000) / 8;
        groove = ((x + 1000) % 8 == 0) || ((z + 1000) % 8 == 0);
        top = rnd(tx, 0, tz, 5) < 0.35f ? mc::StoneWarm : mc::Stone;
        // The sun: disc, ring and sixteen tapering rays.
        const float a = std::atan2(dz, dx);
        const float rayPhase = std::fmod(a / (2.0f * kPi) * 16.0f + 16.5f, 1.0f) - 0.5f; // -0.5..0.5
        const float rayHalf = 0.34f * std::max(0.0f, 1.0f - (r - 20.0f) / 40.0f);
        if (r <= 18.0f) top = r > 15.0f ? mc::SunGold : mc::SunYellow, groove = false;
        else if (r <= 60.0f && std::fabs(rayPhase) < rayHalf)
          top = std::fabs(rayPhase) > rayHalf - 0.05f ? mc::SunGold : mc::SunYellow, groove = false;
        if (r > 62.0f && r <= 66.0f) top = ((int(a * 20.0f) & 1) ? mc::Gold : mc::StoneTrim), groove = false;
        if (r <= 6.0f) top = mc::GlowGold;
      } else if (r <= 17.5f * K) {
        top = mc::StoneTrim;
      } else {
        // Cobbles: irregular stones from a jittered 5-fine grid.
        const int gx = (x + 1000) / 5, gz = (z + 1000) / 5;
        const int lx = (x + 1000) % 5, lz = (z + 1000) % 5;
        groove = (lx == 0 && rnd(gx, 1, gz, 6) < 0.8f) || (lz == 0 && rnd(gx, 2, gz, 6) < 0.8f);
        const float t = rnd(gx, 0, gz, 7);
        top = t < 0.15f ? mc::StoneDark : t < 0.2f ? mc::Moss : mc::Cobble;
      }
      c.box(x, 0, z, x, 2, z, mc::StoneShade);
      if (!groove) c.set(x, 3, z, top);
    }
  return m;
}

// ---------------------------------------------------------------------------
// Fountain: stone basin (water is world blocks), two bowls, crystal spire
// ---------------------------------------------------------------------------

MicroModel fountain() {
  MicroModel m(-7, 1, -7, 15, 16, 15);
  Canvas c(m, true); // y: 0 = the top of the square (block 1)
  const float cx = 2.0f, cz = 2.0f;
  // Basin rim with gold studs and a moulded top.
  c.cyl(cx, cz, 26.0f, 0, 5, mc::Stone, 3.0f);
  c.cyl(cx, cz, 27.0f, 6, 7, mc::StoneTrim, 5.0f);
  for (int k = 0; k < 16; ++k) {
    const float a = float(k) / 16.0f * 2.0f * kPi;
    const int x = int(std::floor(cx + std::cos(a) * 25.0f)), z = int(std::floor(cz + std::sin(a) * 25.0f));
    c.box(x, 8, z, x + 1, 8, z + 1, mc::Gold);
  }
  // Glow crystals on the basin floor (seen through the water).
  for (int k = 0; k < 12; ++k) {
    const float a = float(k) / 12.0f * 2.0f * kPi + 0.2f;
    const int x = int(std::floor(cx + std::cos(a) * 16.0f)), z = int(std::floor(cz + std::sin(a) * 16.0f));
    c.box(x, 0, z, x + 1, 2 + k % 2, z + 1, mc::Crystal);
    c.set(x, 3 + k % 2, z, mc::CrystalBright);
  }
  // Pedestal, lower bowl, column, upper bowl.
  c.cyl(cx, cz, 6.0f, 0, 12, mc::Stone);
  for (int y = 0; y <= 12; y += 3) c.cyl(cx, cz, 6.0f, y, y, mc::StoneShade, 1.0f);
  c.cyl(cx, cz, 13.0f, 13, 15, mc::Stone);
  c.cyl(cx, cz, 13.0f, 16, 16, mc::Gold, 1.5f);
  c.cyl(cx, cz, 11.5f, 16, 16, mc::Water);
  c.cyl(cx, cz, 4.0f, 17, 30, mc::StoneWarm);
  c.cyl(cx, cz, 8.0f, 31, 32, mc::Stone);
  c.cyl(cx, cz, 8.0f, 33, 33, mc::Gold, 1.5f);
  c.cyl(cx, cz, 6.5f, 33, 33, mc::Water);
  // Crystal spire: stacked tapering prisms, bright core, orbiting shards.
  for (int y = 34; y <= 58; ++y) {
    const float t = float(y - 34) / 24.0f;
    const float r = 4.5f * (1.0f - t) + 0.6f;
    c.cyl(cx, cz, r, y, y, mc::Crystal);
    if (r > 2.0f) c.cyl(cx, cz, r - 1.6f, y, y, mc::CrystalBright);
  }
  for (int k = 0; k < 4; ++k) {
    const float a = float(k) * kPi * 0.5f + kPi * 0.25f;
    const int x = int(std::floor(cx + std::cos(a) * 9.0f)), z = int(std::floor(cz + std::sin(a) * 9.0f));
    c.box(x, 42 + k * 2, z, x, 47 + k * 2, z, mc::Crystal);
    c.set(x, 48 + k * 2, z, mc::CrystalBright);
  }
  // Water spouts: arcs of bright water from the upper bowl.
  for (int k = 0; k < 8; ++k) {
    const float a = float(k) / 8.0f * 2.0f * kPi;
    for (int t = 0; t < 8; ++t) {
      const float rr = 8.0f + float(t) * 0.7f;
      const int y = 34 + 2 - (t * t) / 6;
      const int x = int(std::floor(cx + std::cos(a) * rr)), z = int(std::floor(cz + std::sin(a) * rr));
      c.set(x, y, z, mc::CrystalBright);
    }
  }
  return m;
}

// ---------------------------------------------------------------------------
// Castle: terrace with stairs, curtain wall, keep, towers
// ---------------------------------------------------------------------------

MicroModel castle(int T) {
  MicroModel m(-31, 1, -60, 63, 62, 44); // blocks x -31..31, y 1..62, z -60..-17
  Canvas c(m);
  const int y0 = (T + 1) * K; // terrace top (fine)

  // Terrace: ashlar facade, mossy base, coping, balustrade.
  c.ashlar(-26 * K, K, -54 * K, 27 * K - 1, y0 - 1, -22 * K - 1, mc::StoneTrim, mc::StoneDark);
  c.speckle(-26 * K, K, -54 * K, 27 * K - 1, K + 5, -22 * K - 1, mc::Moss, 0.35f, 41);
  c.box(-26 * K - 1, y0 - 2, -54 * K - 1, 27 * K, y0 - 1, -22 * K, mc::Stone);         // coping
  // Paving on top: big tiles.
  for (int z = -54 * K; z < -22 * K; ++z)
    for (int x = -26 * K; x < 27 * K; ++x) {
      const bool groove = ((x + 1000) % 12 == 0) || ((z + 1000) % 12 == 0);
      c.set(x, y0 - 1, z, groove ? mc::StoneShade : (rnd(x / 12, 3, z / 12, 43) < 0.4f ? mc::StoneWarm : mc::Stone));
    }
  // Balustrade: posts every 8, rail on top, gap for the stairs.
  auto balustrade = [&](int x0, int z0, int x1, int z1) {
    const bool alongX = z0 == z1;
    const int a0 = alongX ? x0 : z0, a1 = alongX ? x1 : z1;
    for (int a = a0; a <= a1; ++a) {
      const int x = alongX ? a : x0, z = alongX ? z0 : a;
      if (((a - a0) % 4) < 2) c.box(x, y0, z, x + (alongX ? 0 : 1), y0 + 4, z + (alongX ? 1 : 0), mc::Stone);
      c.box(x, y0 + 5, z, x + (alongX ? 0 : 1), y0 + 6, z + (alongX ? 1 : 0), mc::StoneTrim);
      if (((a - a0) % 16) == 0) c.box(x, y0, z, x + 1, y0 + 8, z + 1, mc::StoneTrim), c.set(x, y0 + 9, z, mc::Gold);
    }
  };
  balustrade(-26 * K, -22 * K - 2, -6 * K, -22 * K - 2);
  balustrade(6 * K + 3, -22 * K - 2, 27 * K - 1, -22 * K - 2);
  balustrade(-26 * K, -54 * K, -26 * K, -22 * K - 2);
  balustrade(27 * K - 2, -54 * K, 27 * K - 2, -22 * K - 2);

  // Grand stairs (one block per step), nosings, cheek walls with gold caps.
  for (int s = 0; s < T; ++s) {
    const int zb = (-17 - s) * K;
    const int top = (s + 2) * K - 1;
    c.box(-4 * K, K, zb, 5 * K - 1, top, zb + K - 1, mc::StoneShade);
    c.box(-4 * K, top, zb, 5 * K - 1, top, zb + K - 1, mc::Stone);
    c.box(-4 * K, top, zb + K, 5 * K - 1, top, zb + K, mc::StoneTrim); // nosing
    for (int x : {-5 * K, 5 * K}) {
      c.box(x, K, zb, x + K - 1, top + 3, zb + K - 1, mc::StoneTrim);
      c.box(x, top + 4, zb, x + K - 1, top + 4, zb + K - 1, mc::Gold);
    }
  }
  // Carpet runner up the stairs.
  for (int s = 0; s < T; ++s) {
    const int zb = (-17 - s) * K, top = (s + 2) * K - 1;
    c.box(-2 * K + 2, top, zb, 3 * K - 3, top, zb + K - 1, mc::BannerBlue);
    c.box(-2 * K + 2, top, zb, -2 * K + 2, top, zb + K - 1, mc::Gold);
    c.box(3 * K - 3, top, zb, 3 * K - 3, top, zb + K - 1, mc::Gold);
  }

  // Curtain wall: 6 thick, 9 blocks high, batter, gold band, merlons, slits.
  const int wallTop = y0 + 36;
  auto wallRun = [&](int x0, int z0, int x1, int z1) { // fine, inclusive
    c.ashlar(x0, y0, z0, x1, wallTop, z1, mc::Stone, mc::StoneShade);
    c.box(x0 - 1, y0, z0 - 1, x1 + 1, y0 + 3, z1 + 1, mc::StoneTrim);
    c.box(x0, wallTop - 6, z0, x1, wallTop - 5, z1, mc::Gold);
    c.box(x0 - 1, wallTop, z0 - 1, x1 + 1, wallTop + 1, z1 + 1, mc::StoneTrim);
    c.merlons(x0, z0, x1, z1, wallTop + 2, 5, mc::Stone, mc::StoneTrim);
  };
  const int wx0 = -22 * K - 1, wx1 = 23 * K, wzF = -28 * K + 2, wzB = -52 * K; // outer faces
  wallRun(wx0, wzF - 5, wx1, wzF);          // front
  wallRun(wx0, wzB, wx1, wzB + 5);          // back
  wallRun(wx0, wzB, wx0 + 5, wzF);          // west
  wallRun(wx1 - 5, wzB, wx1, wzF);          // east
  // Arrow slits.
  for (int x = wx0 + 10; x < wx1 - 10; x += 14)
    if (std::abs(x) > 14) c.box(x, y0 + 14, wzF, x, y0 + 21, wzF - 5, mc::Iron);
  // Gate: pointed arch through the front wall, crystal frame, portcullis.
  const int gw = 10, gh = 28;
  for (int y = y0; y < y0 + gh; ++y) {
    const int narrow = y > y0 + 20 ? (y - (y0 + 20)) : 0;
    c.box(-gw + narrow + 2, y, wzF - 6, gw - narrow + 1, y, wzF + 1, 0);
    c.box(-gw + narrow, y, wzF + 1, -gw + narrow + 1, y, wzF + 1, mc::Crystal);
    c.box(gw - narrow + 2, y, wzF + 1, gw - narrow + 3, y, wzF + 1, mc::Crystal);
  }
  c.box(-1, y0 + gh, wzF + 1, 3, y0 + gh + 2, wzF + 1, mc::GlowGold); // keystone
  for (int x = -gw + 4; x <= gw; x += 3) c.box(x, y0 + 21, wzF - 3, x, y0 + 26, wzF - 3, mc::Iron);
  // Banners on the front wall.
  for (int bx : {-58, -34, 26, 50}) c.banner(true, wzF, 1, bx, wallTop - 8, 9, 22);

  // Keep: pilasters, three rows of arched windows, frieze, cornice, hipped roof.
  const int kx0 = -12 * K, kx1 = 13 * K - 1, kz0 = -50 * K, kz1 = -35 * K - 1, kTop = y0 + 72;
  c.ashlar(kx0, y0, kz0, kx1, kTop, kz1, mc::Stone, mc::StoneShade);
  c.box(kx0 + 6, y0, kz0 + 6, kx1 - 6, kTop - 1, kz1 - 6, 0); // hollow
  for (int x = kx0; x <= kx1; x += 12) c.box(x, y0, kz1 + 1, x + 2, kTop - 6, kz1 + 1, mc::StoneTrim);
  for (int x = kx0; x <= kx1; x += 12) c.box(x, y0, kz0 - 1, x + 2, kTop - 6, kz0 - 1, mc::StoneTrim);
  for (int z = kz0; z <= kz1; z += 12) c.box(kx0 - 1, y0, z, kx0 - 1, kTop - 6, z + 2, mc::StoneTrim);
  for (int z = kz0; z <= kz1; z += 12) c.box(kx1 + 1, y0, z, kx1 + 1, kTop - 6, z + 2, mc::StoneTrim);
  for (int row = 0; row < 3; ++row) {
    const int wy = y0 + 14 + row * 20;
    for (int x = kx0 + 5; x + 5 < kx1; x += 12) {
      if (row == 0 && std::abs(x + 3) < 14) continue; // the door
      c.window(true, kz1, 1, 6, x, wy, 5, 11, true);
      c.window(true, kz0, -1, 6, x, wy, 5, 11, true);
    }
    for (int z = kz0 + 5; z + 5 < kz1; z += 12) {
      c.window(false, kx0, -1, 6, z, wy, 5, 11, true);
      c.window(false, kx1, 1, 6, z, wy, 5, 11, true);
    }
  }
  c.box(kx0 - 1, kTop - 8, kz0 - 1, kx1 + 1, kTop - 6, kz1 + 1, mc::Gold);        // frieze
  c.box(kx0 - 3, kTop - 2, kz0 - 3, kx1 + 3, kTop, kz1 + 3, mc::StoneTrim);       // cornice
  c.hipRoof(kx0 - 3, kz0 - 3, kx1 + 3, kz1 + 3, kTop + 1, mc::RoofBlue, mc::RoofBlueDark, mc::RoofBlueLight);
  // Glowing crystal door with a gold arch and steps.
  for (int y = y0; y < y0 + 26; ++y) {
    const int narrow = y > y0 + 19 ? (y - (y0 + 19)) : 0;
    for (int x = -8 + narrow; x <= 9 - narrow; ++x) {
      const bool edge = x <= -7 + narrow || x >= 8 - narrow || y >= y0 + 25;
      c.box(x, y, kz1 - 1, x, y, kz1 + 2, edge ? mc::Gold : ((x + y) % 5 == 0 ? mc::CrystalBright : mc::Crystal));
    }
  }
  // Towers: curtain corners, keep turrets (corbelled out), the great tower.
  for (int sx : {-1, 1}) {
    const float tx = sx < 0 ? float(wx0) + 3.0f : float(wx1) - 2.0f;
    c.tower(tx, float(wzF) - 2.0f, 16.0f, y0, 58, 44, true);
    c.tower(tx, float(wzB) + 3.0f, 16.0f, y0, 66, 48, true);
    const float kxT = sx < 0 ? float(kx0) : float(kx1) + 1.0f;
    for (float kzT : {float(kz0), float(kz1) + 1.0f}) {
      c.cyl(kxT, kzT, 9.0f, kTop - 20, kTop - 12, mc::StoneTrim);       // corbel
      c.tower(kxT, kzT, 9.0f, kTop - 12, 30, 26, false);
    }
  }
  c.tower(2.0f, -42.5f * K, 20.0f, y0, 132, 56, true);
  return m;
}

// ---------------------------------------------------------------------------
// Square furniture, market, walls
// ---------------------------------------------------------------------------

MicroModel bannerPole(int bx, int bz, int dirX, int dirZ) {
  MicroModel m(bx - 3, 1, bz - 3, 7, 12, 7);
  Canvas c(m, true);
  const int px = bx * K + 1, pz = bz * K + 1;
  c.box(px - 1, 0, pz - 1, px + 2, 2, pz + 2, mc::StoneTrim);          // footing
  c.box(px, 3, pz, px + 1, 40, pz + 1, mc::WoodDark);                 // pole
  c.box(px, 18, pz, px + 1, 18, pz + 1, mc::Gold);
  c.box(px - 1, 41, pz - 1, px + 2, 43, pz + 2, mc::GlowGold);        // finial
  // Crossbar and banner hanging to one side.
  const bool alongX = dirX != 0;
  const int s = alongX ? dirX : dirZ;
  for (int t = 1; t <= 12; ++t) {
    if (alongX) c.box(px + (s > 0 ? 1 + t : -t), 38, pz, px + (s > 0 ? 1 + t : -t), 39, pz, mc::WoodDark);
    else c.box(px, 38, pz + (s > 0 ? 1 + t : -t), px, 39, pz + (s > 0 ? 1 + t : -t), mc::WoodDark);
  }
  const int a0 = (alongX ? px : pz) + (s > 0 ? 3 : -12);
  c.banner(alongX, alongX ? pz : px, 0, a0, 37, 10, 24);
  return m;
}

MicroModel marketStall(int x0, int side, uint32_t seed) {
  // Blocks x0..x0+4 along the road, depth 3 behind the road edge.
  const int zf = side * 5, zb = side * 8;
  MicroModel m(x0 - 1, 1, side < 0 ? -10 : 3, 7, 7, 7);
  Canvas c(m, true);
  const int X0 = x0 * K, X1 = (x0 + 5) * K - 1;
  const int Zf = side < 0 ? zf * K + K - 1 : zf * K; // front (road-side) face
  (void)zb;
  auto zAt = [&](int d) { return Zf + side * d; }; // d = depth from the front face, away from the road
  // Posts.
  for (int x : {X0, X1 - 1})
    for (int d : {0, 10}) c.box(x, 0, zAt(d), x + 1, 22, zAt(d + 1), mc::WoodDark);
  // Counter: plank front, top, goods.
  c.box(X0 + 2, 0, zAt(0), X1 - 2, 6, zAt(3), mc::WoodLight);
  for (int x = X0 + 2; x <= X1 - 2; x += 3) c.box(x, 0, zAt(0), x, 6, zAt(0), mc::Wood);
  c.box(X0 + 1, 7, zAt(-1), X1 - 1, 7, zAt(4), mc::WoodDark);
  for (int x = X0 + 3; x <= X1 - 3; ++x)
    for (int d = 0; d <= 3; ++d) {
      const float r = rnd(x, d, side, seed);
      if (r < 0.5f) continue;
      const uint8_t g = r < 0.62f ? mc::FlowerRed : r < 0.74f ? mc::Orange : r < 0.84f ? mc::Leaf : r < 0.93f ? mc::WoodLight : mc::Gold;
      c.set(x, 8, zAt(d), g);
      if (r > 0.9f) c.set(x, 9, zAt(d), g);
    }
  // Crates at the back.
  for (int k = 0; k < 3; ++k) {
    const int cx = X0 + 2 + k * 7, cy = k == 2 ? 6 : 0;
    const int d0 = 7, d1 = 12;
    c.box(cx, cy, zAt(d0), cx + 5, cy + 5, zAt(d1), mc::WoodLight);
    c.box(cx, cy, zAt(d0), cx + 5, cy, zAt(d1), mc::WoodDark);
    c.box(cx, cy + 5, zAt(d0), cx + 5, cy + 5, zAt(d1), mc::WoodDark);
    c.box(cx, cy, zAt(d0), cx, cy + 5, zAt(d0), mc::WoodDark);
    c.box(cx + 5, cy, zAt(d0), cx + 5, cy + 5, zAt(d0), mc::WoodDark);
  }
  // Striped awning: low at the front (scalloped), rising to the back.
  for (int d = -3; d <= 12; ++d) {
    const int y = 22 + (d + 3) / 3;
    for (int x = X0 - 1; x <= X1 + 1; ++x) {
      const uint8_t stripe = ((x - X0 + 100) / 3) % 2 ? mc::AwningWhite : mc::AwningRed;
      c.set(x, y, zAt(d), stripe);
      if (d == -3) { // scallops
        c.set(x, y - 1, zAt(d), stripe);
        if (((x - X0 + 100) % 3) == 1) c.set(x, y - 2, zAt(d), stripe);
      }
    }
  }
  return m;
}

// One side of the town wall (blocks at |coord| = wall-1 .. wall), with
// buttresses, crenels, a walkway parapet and gates on the road axes.
MicroModel wallSide(int side, int wall) {
  const bool alongX = side <= 1;
  const int s = (side == 0 || side == 2) ? -1 : 1;
  MicroModel m = alongX ? MicroModel(-wall, 0, s < 0 ? -wall - 1 : wall - 2, 2 * wall + 1, 11, 4)
                        : MicroModel(s < 0 ? -wall - 1 : wall - 2, 0, -wall, 4, 11, 2 * wall + 1);
  Canvas c(m);
  const int inner = s < 0 ? (-wall + 2) * K - 1 : (wall - 1) * K;   // inner face (fine)
  const int outer = s < 0 ? -wall * K : (wall + 1) * K - 1;          // outer face
  const int a0 = -wall * K, a1 = (wall + 1) * K - 1;
  const int top = 8 * K - 1;
  auto B = [&](int aa0, int y0, int d0, int aa1, int y1, int d1, uint8_t col) { // d = depth coordinates
    if (alongX) c.box(aa0, y0, d0, aa1, y1, d1, col);
    else c.box(d0, y0, aa0, d1, y1, aa1, col);
  };
  const int lo = std::min(inner, outer), hi = std::max(inner, outer);
  if (alongX) c.ashlar(a0, K, lo, a1, top, hi, mc::Stone, mc::StoneShade);
  else c.ashlar(lo, K, a0, hi, top, a1, mc::Stone, mc::StoneShade);
  if (alongX) c.speckle(a0, K, lo, a1, K + 7, hi, mc::Moss, 0.3f, 51 + side);
  else c.speckle(lo, K, a0, hi, K + 7, a1, mc::Moss, 0.3f, 51 + side);
  B(a0, K, outer - s * 1, a1, K + 3, outer + s * 1, mc::StoneTrim);            // plinth
  B(a0, top - 1, lo - 1, a1, top, hi + 1, mc::StoneTrim);                      // coping
  // Crenels on the outer edge, a low parapet inside.
  for (int a = a0; a <= a1; ++a) {
    if (((a - a0) % 8) < 5) B(a, top + 1, outer, a, top + 6, outer - s * 2, mc::Stone), B(a, top + 7, outer, a, top + 7, outer - s * 2, mc::StoneTrim);
    B(a, top + 1, inner, a, top + 2, inner + s * 1, mc::StoneTrim);
  }
  // Buttresses every 24 on the outer face.
  for (int a = a0 + 12; a < a1 - 12; a += 24) B(a, K, outer + s * 1, a + 3, top - 8, outer + s * 3, mc::StoneTrim);
  // Gate (on the road axis, sides 1..3): pointed arch, gold keystone.
  if (side != 0) {
    for (int y = K; y < K + 22; ++y) {
      const int narrow = y > K + 15 ? (y - (K + 15)) : 0;
      B(-12 + narrow + 2, y, lo - 4, 13 - narrow, y, hi + 4, 0);
    }
    B(-1, K + 22, outer, 2, K + 24, outer + s, mc::GlowGold);
    c.banner(alongX, outer, s, -24, top - 2, 8, 16);
    c.banner(alongX, outer, s, 18, top - 2, 8, 16);
  }
  return m;
}

MicroModel gatehouse(int cx, int cz, int wallHeight) {
  MicroModel m(cx - 3, 1, cz - 3, 7, wallHeight + 9, 7);
  Canvas c(m);
  const int x0 = (cx - 2) * K, x1 = (cx + 3) * K - 1, z0 = (cz - 2) * K, z1 = (cz + 3) * K - 1;
  const int top = (wallHeight + 1) * K;
  c.ashlar(x0, K, z0, x1, top, z1, mc::Stone, mc::StoneShade);
  c.box(x0 - 1, K, z0 - 1, x1 + 1, K + 4, z1 + 1, mc::StoneTrim);
  c.box(x0 - 1, top - 8, z0 - 1, x1 + 1, top - 7, z1 + 1, mc::Gold);
  for (int y = K * 3; y < top - 12; y += 16) {
    c.window(true, z1, 1, 4, cx * K, y, 4, 9, true);
    c.window(true, z0, -1, 4, cx * K, y, 4, 9, true);
    c.window(false, x0, -1, 4, cz * K, y, 4, 9, true);
    c.window(false, x1, 1, 4, cz * K, y, 4, 9, true);
  }
  c.box(x0 - 2, top, z0 - 2, x1 + 2, top + 2, z1 + 2, mc::StoneTrim);
  c.hipRoof(x0 - 2, z0 - 2, x1 + 2, z1 + 2, top + 3, mc::RoofBlue, mc::RoofBlueDark, mc::RoofBlueLight);
  c.box(cx * K + 1, top + 22, cz * K + 1, cx * K + 2, top + 26, cz * K + 2, mc::Gold);
  return m;
}

MicroModel cornerTower(int cx, int cz) {
  MicroModel m(cx - 6, 0, cz - 6, 13, 28, 13);
  Canvas c(m);
  c.tower(cx * K + 2.0f, cz * K + 2.0f, 15.0f, K, 46, 36, true);
  return m;
}

} // namespace ao::world::townmicro
