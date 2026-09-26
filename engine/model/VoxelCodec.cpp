#include "VoxelCodec.h"

namespace atm::model {

namespace {

void putVar(std::vector<uint8_t> &out, uint64_t v) {
  while (v >= 0x80) {
    out.push_back(uint8_t(v | 0x80));
    v >>= 7;
  }
  out.push_back(uint8_t(v));
}

bool getVar(std::span<const uint8_t> in, size_t &pos, uint64_t &v) {
  v = 0;
  for (int shift = 0; shift < 64; shift += 7) {
    if (pos >= in.size()) return false;
    const uint8_t b = in[pos++];
    v |= uint64_t(b & 0x7F) << shift;
    if (!(b & 0x80)) return true;
  }
  return false;
}

bool fail(std::string *error, const char *what) {
  if (error) *error = what;
  return false;
}

} // namespace

std::vector<uint8_t> encodeVoxels(int sx, int sy, int sz, std::span<const uint8_t> voxels,
                                  std::span<const uint32_t> palette) {
  std::vector<uint8_t> out = {'A', 'V', 'X', '1'};
  putVar(out, uint64_t(sx));
  putVar(out, uint64_t(sy));
  putVar(out, uint64_t(sz));
  putVar(out, palette.size());
  for (uint32_t c : palette)
    for (int s = 0; s < 32; s += 8) out.push_back(uint8_t(c >> s));
  const size_t n = size_t(sx) * size_t(sy) * size_t(sz);
  size_t i = 0;
  while (i < n && i < voxels.size()) {
    const uint8_t v = voxels[i];
    size_t run = 1;
    while (i + run < n && voxels[i + run] == v) ++run;
    putVar(out, (uint64_t(run) << 8) | v);
    i += run;
  }
  return out;
}

bool decodeVoxels(std::span<const uint8_t> in, EncodedVoxels &out, std::string *error) {
  if (in.size() < 4 || in[0] != 'A' || in[1] != 'V' || in[2] != 'X' || in[3] != '1') return fail(error, "bad magic");
  size_t pos = 4;
  uint64_t sx, sy, sz, np;
  if (!getVar(in, pos, sx) || !getVar(in, pos, sy) || !getVar(in, pos, sz) || !getVar(in, pos, np))
    return fail(error, "truncated header");
  if (sx == 0 || sy == 0 || sz == 0 || sx > 4096 || sy > 4096 || sz > 4096 || sx * sy * sz > (1ull << 30))
    return fail(error, "bad size");
  if (np == 0 || np > 256 || pos + np * 4 > in.size()) return fail(error, "bad palette");
  out.sx = int(sx), out.sy = int(sy), out.sz = int(sz);
  out.palette.resize(size_t(np));
  for (size_t k = 0; k < np; ++k, pos += 4)
    out.palette[k] = uint32_t(in[pos]) | (uint32_t(in[pos + 1]) << 8) | (uint32_t(in[pos + 2]) << 16) |
                     (uint32_t(in[pos + 3]) << 24);
  const size_t n = size_t(sx * sy * sz);
  out.voxels.clear();
  out.voxels.reserve(n);
  while (out.voxels.size() < n) {
    uint64_t r;
    if (!getVar(in, pos, r)) return fail(error, "truncated runs");
    const uint8_t v = uint8_t(r & 0xFF);
    const uint64_t run = r >> 8;
    if (run == 0 || run > n - out.voxels.size()) return fail(error, "bad run");
    if (v >= np) return fail(error, "bad palette index");
    out.voxels.insert(out.voxels.end(), size_t(run), v);
  }
  if (pos != in.size()) return fail(error, "trailing data");
  return true;
}

} // namespace atm::model
