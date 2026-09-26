#pragma once

// Compact storage for voxel models (the "cache format" for authored assets),
// in the spirit of RuneScape's cache: small integers as varints, runs of
// equal values collapsed, one shared palette. A mostly-empty building of a
// few million voxels encodes to tens of kilobytes; generated content (the
// home town) isn't stored at all.
//
// Layout: "AVX1" | varint sx, sy, sz | varint paletteSize | palette RGBA8 x n
//       | runs: varint (length << 8 | index)... in (y * sz + z) * sx + x order.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace atm::model {

struct EncodedVoxels {
  int sx = 0, sy = 0, sz = 0;
  std::vector<uint32_t> palette;  // RGBA8, [0] unused (empty)
  std::vector<uint8_t> voxels;    // palette indices
};

std::vector<uint8_t> encodeVoxels(int sx, int sy, int sz, std::span<const uint8_t> voxels,
                                  std::span<const uint32_t> palette);
// False on malformed data (size limits, truncated runs, bad indices).
bool decodeVoxels(std::span<const uint8_t> data, EncodedVoxels &out, std::string *error = nullptr);

} // namespace atm::model
