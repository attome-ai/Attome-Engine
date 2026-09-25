// Binary greedy mesher (after cgerikj/binary-greedy-meshing), adapted to
// 32^3 chunks with a 1-block border (34^3 input) and exact per-quad AO/light.
//
// 1. One pass over the 34^3 input builds, for each normal axis, 34x34 64-bit
//    column masks (bit = padded coordinate along the axis) for opaque,
//    drawn-in-opaque-pass (opaque + cutout) and translucent cells.
// 2. Face culling per column with shifts/ANDs:
//      +dir: faces = draw & ~(opaque >> 1)      -dir: draw & ~(opaque << 1)
//    translucent: trans & ~neighbourOpaque & ~neighbourSameTranslucent.
//    The 32 inner bits are scattered into per-direction 32x32 slice grids
//    (one uint32 row per (layer, v), bit = u).
// 3. Greedy merge per slice: find runs with countr_zero, extend along u while
//    the key (material, AO, sky, block light) is equal, then extend along v
//    while the whole run mask is present (one AND) and keys are equal.
// 4. Cave-culling visibility graph: bit-parallel scanline flood fill over
//    32-bit rows of non-opaque cells.

#include "BlockRegistry.h"
#include "Chunk.h"

#include <algorithm>
#include <bit>
#include <cstring>

namespace atm::voxel {

namespace {

constexpr int P = MeshInput::kPad; // 34
constexpr int SX = 1, SZ = P, SY = P * P;
constexpr int kBase = SY + SZ + SX; // padded (1,1,1) = local (0,0,0)

constexpr uint8_t kClsAir = 0, kClsOpaque = 1, kClsCutout = 2, kClsTrans = 3;

// Per normal axis (0 = X, 1 = Y, 2 = Z): strides of normal, U and V axes.
// Tangents follow MeshTypes.h: ±X: U=Z, V=Y; ±Y: U=X, V=Z; ±Z: U=X, V=Y.
constexpr int kStrideN[3] = {SX, SY, SZ};
constexpr int kStrideU[3] = {SZ, SX, SX};
constexpr int kStrideV[3] = {SY, SZ, SY};

inline uint8_t aoValue(uint8_t side1, uint8_t side2, uint8_t corner) {
  return (side1 && side2) ? uint8_t(0) : uint8_t(3 - (side1 + side2 + corner));
}

inline uint32_t runMask(int u0, int w) {
  return (w >= 32 ? 0xFFFFFFFFu : ((1u << w) - 1u)) << u0;
}

struct Bounds {
  int minX = 32, minY = 32, minZ = 32, maxX = 0, maxY = 0, maxZ = 0;
  void add(int x0, int y0, int z0, int x1, int y1, int z1) {
    minX = std::min(minX, x0);
    minY = std::min(minY, y0);
    minZ = std::min(minZ, z0);
    maxX = std::max(maxX, x1);
    maxY = std::max(maxY, y1);
    maxZ = std::max(maxZ, z1);
  }
};

// Greedy-merges one 32x32 slice. `rows` is consumed (bits cleared).
// keys[v * 32 + u] valid where the bit is set.
template <typename Emit>
void greedySlice(uint32_t *rows, const uint32_t *keys, Emit &&emit) {
  for (int v = 0; v < 32; ++v) {
    while (rows[v]) {
      const uint32_t bits = rows[v];
      const int u0 = std::countr_zero(bits);
      const uint32_t k = keys[v * 32 + u0];
      int u1 = u0 + 1;
      while (u1 < 32 && ((bits >> u1) & 1u) && keys[v * 32 + u1] == k)
        ++u1;
      const int w = u1 - u0;
      const uint32_t m = runMask(u0, w);
      int h = 1;
      while (v + h < 32 && (rows[v + h] & m) == m) {
        const uint32_t *kr = keys + (v + h) * 32;
        bool same = true;
        for (int u = u0; u < u1; ++u)
          if (kr[u] != k) {
            same = false;
            break;
          }
        if (!same)
          break;
        ++h;
      }
      for (int i = 0; i < h; ++i)
        rows[v + i] &= ~m;
      emit(u0, v, w, h, k);
    }
  }
}

} // namespace

std::unique_ptr<ChunkMesher::Scratch> ChunkMesher::makeScratch() { return std::make_unique<Scratch>(); }

void ChunkMesher::mesh(const MeshInput &in, const BlockRegistry &blocks, Scratch &s, ChunkMeshData &out) {
  out.clear();
  if (in.blocks.size() < size_t(Scratch::kPadCells) || in.light.size() < size_t(Scratch::kPadCells))
    return; // malformed input

  // --- per-id class table ---------------------------------------------------
  const size_t nDefs = blocks.size();
  s.table.resize(nDefs);
  for (size_t i = 0; i < nDefs; ++i) {
    switch (blocks.renderMode(BlockId(i))) {
    case BlockRender::None: s.table[i] = kClsAir; break;
    case BlockRender::Opaque: s.table[i] = kClsOpaque; break;
    case BlockRender::Cutout: s.table[i] = kClsCutout; break;
    case BlockRender::Translucent: s.table[i] = kClsTrans; break;
    }
  }

  // --- 1. column masks --------------------------------------------------------
  for (int a = 0; a < 3; ++a) {
    std::fill(s.opaqueCols[a].begin(), s.opaqueCols[a].end(), 0ull);
    std::fill(s.drawCols[a].begin(), s.drawCols[a].end(), 0ull);
    std::fill(s.transCols[a].begin(), s.transCols[a].end(), 0ull);
  }
  uint64_t *opX = s.opaqueCols[0].data(), *opY = s.opaqueCols[1].data(), *opZ = s.opaqueCols[2].data();
  uint64_t *drX = s.drawCols[0].data(), *drY = s.drawCols[1].data(), *drZ = s.drawCols[2].data();
  uint64_t *trX = s.transCols[0].data(), *trY = s.transCols[1].data(), *trZ = s.transCols[2].data();
  const BlockId *ids = in.blocks.data();
  const uint8_t *light = in.light.data();
  const uint8_t *table = s.table.data();
  uint8_t *cls = s.cls.data();
  uint8_t *occ = s.occ.data();

  BlockId firstTrans = 0;
  bool anyTrans = false, multiTrans = false;
  int innerOpaque = 0;
  for (int py = 0; py < P; ++py)
    for (int pz = 0; pz < P; ++pz) {
      const int rowBase = (py * P + pz) * P;
      for (int px = 0; px < P; ++px) {
        const int idx = rowBase + px;
        const BlockId id = ids[idx];
        const uint8_t c = id < nDefs ? table[id] : kClsOpaque; // unknown ids: opaque
        cls[idx] = c;
        occ[idx] = uint8_t(c == kClsOpaque || c == kClsCutout);
        if (c == kClsAir)
          continue;
        const uint64_t bx = 1ull << px, by = 1ull << py, bz = 1ull << pz;
        const int iX = py * P + pz, iY = pz * P + px, iZ = py * P + px;
        if (c == kClsTrans) {
          trX[iX] |= bx;
          trY[iY] |= by;
          trZ[iZ] |= bz;
          if (!anyTrans) {
            anyTrans = true;
            firstTrans = id;
          } else if (id != firstTrans) {
            multiTrans = true;
          }
          continue;
        }
        drX[iX] |= bx;
        drY[iY] |= by;
        drZ[iZ] |= bz;
        if (c == kClsOpaque) {
          opX[iX] |= bx;
          opY[iY] |= by;
          opZ[iZ] |= bz;
          if (px >= 1 && px <= 32 && py >= 1 && py <= 32 && pz >= 1 && pz <= 32)
            ++innerOpaque;
        }
      }
    }

  Bounds bounds;
  uint32_t *gridO = s.gridOpaque.data();
  uint32_t *gridT = s.gridTrans.data();
  uint32_t *keys = s.keys.data();

  for (int d = 0; d < kFaceDirCount; ++d) {
    out.opaqueDirOffset[size_t(d)] = uint32_t(out.opaque.size());
    const int A = d >> 1;
    const bool positive = (d & 1) == 0;
    const int N = kStrideN[A], U = kStrideU[A], V = kStrideV[A];
    const int nOff = positive ? N : -N;
    const uint64_t *opc = s.opaqueCols[size_t(A)].data();
    const uint64_t *drc = s.drawCols[size_t(A)].data();
    const uint64_t *trc = s.transCols[size_t(A)].data();

    // --- 2. face culling + scatter into slice grids ---------------------------
    std::memset(gridO, 0, sizeof(uint32_t) * 1024);
    std::memset(gridT, 0, sizeof(uint32_t) * 1024);
    bool anyO = false, anyT = false;
    for (int pv = 1; pv <= 32; ++pv)
      for (int pu = 1; pu <= 32; ++pu) {
        const int ci = pv * P + pu;
        const uint64_t o = opc[ci];
        const uint64_t nOpaque = positive ? (o >> 1) : (o << 1);
        const int u = pu - 1, v = pv - 1;
        uint64_t vd = drc[ci] & ~nOpaque;
        vd = (vd >> 1) & 0xFFFFFFFFull;
        if (vd) {
          anyO = true;
          while (vd) {
            const int l = std::countr_zero(vd);
            vd &= vd - 1;
            gridO[l * 32 + v] |= 1u << u;
          }
        }
        const uint64_t t = trc[ci];
        if (t) {
          const uint64_t nTrans = positive ? (t >> 1) : (t << 1);
          uint64_t vt = t & ~nOpaque;
          if (!multiTrans) {
            vt &= ~nTrans; // single translucent type: same neighbour hides
          } else {
            uint64_t both = vt & nTrans & (0xFFFFFFFFull << 1); // bits 1..32
            while (both) {
              const int b = std::countr_zero(both);
              both &= both - 1;
              const int cell = kBase + (b - 1) * N + u * U + v * V;
              if (ids[cell] == ids[cell + nOff])
                vt &= ~(1ull << b);
            }
          }
          vt = (vt >> 1) & 0xFFFFFFFFull;
          if (vt) {
            anyT = true;
            while (vt) {
              const int l = std::countr_zero(vt);
              vt &= vt - 1;
              gridT[l * 32 + v] |= 1u << u;
            }
          }
        }
      }

    // --- 3. greedy merge per slice --------------------------------------------
    for (int pass = 0; pass < 2; ++pass) {
      const bool trans = pass == 1;
      if (trans ? !anyT : !anyO)
        continue;
      uint32_t *grid = trans ? gridT : gridO;
      std::vector<PackedFace> &dst = trans ? out.translucent : out.opaque;
      for (int l = 0; l < 32; ++l) {
        uint32_t *rows = grid + l * 32;
        uint32_t any = 0;
        for (int v = 0; v < 32; ++v)
          any |= rows[v];
        if (!any)
          continue;
        // Keys for every visible face of this slice.
        for (int v = 0; v < 32; ++v) {
          uint32_t bits = rows[v];
          while (bits) {
            const int u = std::countr_zero(bits);
            bits &= bits - 1;
            const int cell = kBase + l * N + u * U + v * V;
            const int q = cell + nOff;
            uint32_t ao = 0xFF;
            if (!trans) {
              const uint8_t um = occ[q - U], up = occ[q + U], vm = occ[q - V], vp = occ[q + V];
              const uint8_t a0 = aoValue(um, vm, occ[q - U - V]);
              const uint8_t a1 = aoValue(up, vm, occ[q + U - V]);
              const uint8_t a2 = aoValue(up, vp, occ[q + U + V]);
              const uint8_t a3 = aoValue(um, vp, occ[q - U + V]);
              ao = uint32_t(a0) | (uint32_t(a1) << 2) | (uint32_t(a2) << 4) | (uint32_t(a3) << 6);
            }
            keys[v * 32 + u] = uint32_t(ids[cell]) | (ao << 16) | (uint32_t(light[q]) << 24);
          }
        }
        const uint32_t nCoord = uint32_t(positive ? l + 1 : l);
        greedySlice(rows, keys, [&](int u0, int v0, int w, int h, uint32_t k) {
          FaceFields f{};
          f.w = uint32_t(w);
          f.h = uint32_t(h);
          f.dir = FaceDir(d);
          f.ao = uint8_t((k >> 16) & 0xFFu);
          f.material = uint16_t(k & 0xFFFFu);
          f.sky = uint8_t((k >> 28) & 15u);
          f.block = uint8_t((k >> 24) & 15u);
          switch (A) {
          case 0: // U = Z, V = Y
            f.x = nCoord;
            f.z = uint32_t(u0);
            f.y = uint32_t(v0);
            bounds.add(l, v0, u0, l + 1, v0 + h, u0 + w);
            break;
          case 1: // U = X, V = Z
            f.y = nCoord;
            f.x = uint32_t(u0);
            f.z = uint32_t(v0);
            bounds.add(u0, l, v0, u0 + w, l + 1, v0 + h);
            break;
          default: // U = X, V = Y
            f.z = nCoord;
            f.x = uint32_t(u0);
            f.y = uint32_t(v0);
            bounds.add(u0, v0, l, u0 + w, v0 + h, l + 1);
            break;
          }
          dst.push_back(packFace(f));
        });
      }
    }
  }
  out.opaqueDirOffset[kFaceDirCount] = uint32_t(out.opaque.size());

  if (!out.empty()) {
    out.minX = uint8_t(bounds.minX);
    out.minY = uint8_t(bounds.minY);
    out.minZ = uint8_t(bounds.minZ);
    out.maxX = uint8_t(bounds.maxX);
    out.maxY = uint8_t(bounds.maxY);
    out.maxZ = uint8_t(bounds.maxZ);
  }

  // --- 4. visibility graph (cave culling) -------------------------------------
  if (innerOpaque == 0) {
    out.faceConnectivity = ~0ull;
    return;
  }
  if (innerOpaque == kChunkVolume) {
    out.faceConnectivity = 0;
    return;
  }
  uint32_t *open = s.open.data();
  uint32_t *visited = s.visited.data();
  for (int y = 0; y < 32; ++y)
    for (int z = 0; z < 32; ++z) // bits 1..32 of the X column are the inner cells
      open[y * 32 + z] = ~uint32_t((s.opaqueCols[0][size_t((y + 1) * P + (z + 1))] >> 1) & 0xFFFFFFFFull);
  std::memset(visited, 0, sizeof(uint32_t) * 1024);

  // Expands `seed` to all cells of `avail` connected along the row.
  auto fillRow = [](uint32_t seed, uint32_t avail) {
    uint32_t cur = seed & avail;
    for (;;) {
      const uint32_t next = (cur | (cur << 1) | (cur >> 1)) & avail;
      if (next == cur)
        return cur;
      cur = next;
    }
  };

  uint64_t conn = 0;
  for (int r = 0; r < 1024; ++r) {
    for (;;) {
      const uint32_t avail = open[r] & ~visited[r];
      if (!avail)
        break;
      const uint32_t seedBits = fillRow(avail & (0u - avail), avail);
      visited[r] |= seedBits;
      s.stack.clear();
      s.stack.push_back((uint64_t(uint32_t(r)) << 32) | seedBits);
      uint32_t faces = 0;
      while (!s.stack.empty()) {
        const uint64_t item = s.stack.back();
        s.stack.pop_back();
        const int row = int(item >> 32);
        const uint32_t bits = uint32_t(item);
        const int y = row >> 5, z = row & 31;
        if (bits & 1u)
          faces |= 1u << int(FaceDir::NegX);
        if (bits & 0x80000000u)
          faces |= 1u << int(FaceDir::PosX);
        if (y == 0)
          faces |= 1u << int(FaceDir::NegY);
        if (y == 31)
          faces |= 1u << int(FaceDir::PosY);
        if (z == 0)
          faces |= 1u << int(FaceDir::NegZ);
        if (z == 31)
          faces |= 1u << int(FaceDir::PosZ);
        const int nbr[4] = {y > 0 ? row - 32 : -1, y < 31 ? row + 32 : -1, z > 0 ? row - 1 : -1,
                            z < 31 ? row + 1 : -1};
        for (int m : nbr) {
          if (m < 0)
            continue;
          const uint32_t availM = open[m] & ~visited[m];
          const uint32_t cand = bits & availM;
          if (!cand)
            continue;
          const uint32_t grown = fillRow(cand, availM);
          visited[m] |= grown;
          s.stack.push_back((uint64_t(uint32_t(m)) << 32) | grown);
        }
      }
      for (int a = 0; a < 6; ++a)
        if (faces & (1u << a))
          for (int b = a + 1; b < 6; ++b)
            if (faces & (1u << b))
              conn |= 1ull << ChunkMeshData::connectivityBit(a, b);
    }
  }
  // Canonical "fully open" value when every pair connects.
  uint64_t allPairs = 0;
  for (int a = 0; a < 6; ++a)
    for (int b = a + 1; b < 6; ++b)
      allPairs |= 1ull << ChunkMeshData::connectivityBit(a, b);
  out.faceConnectivity = (conn == allPairs) ? ~0ull : conn;
}

} // namespace atm::voxel
