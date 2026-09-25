#include "Chunk.h"

#include <algorithm>
#include <cstring>

namespace atm::voxel {

namespace {

constexpr uint8_t kSerialVersion = 1;
constexpr uint8_t kModeRaw = 0; // packed 64-bit words, little endian
constexpr uint8_t kModeRle = 1; // (varint run-1, varint index) pairs

// Smallest supported bit width holding `n` palette entries.
constexpr uint8_t bitsFor(size_t n) {
  return n <= 1 ? 0 : n <= 2 ? 1 : n <= 4 ? 2 : n <= 16 ? 4 : n <= 256 ? 8 : 16;
}
constexpr int log2Bits(int bits) {
  return bits == 1 ? 0 : bits == 2 ? 1 : bits == 4 ? 2 : bits == 8 ? 3 : 4;
}
constexpr size_t wordsFor(int bits) { return size_t(kChunkVolume) * size_t(bits) / 64; }
constexpr bool validBits(int b) { return b == 0 || b == 1 || b == 2 || b == 4 || b == 8 || b == 16; }

template <int B>
void decodeT(const uint64_t *data, const BlockId *pal, BlockId *out) {
  constexpr int E = 64 / B;
  constexpr uint64_t M = (uint64_t(1) << B) - 1;
  constexpr int W = kChunkVolume / E;
  for (int w = 0; w < W; ++w) {
    uint64_t v = data[w];
    BlockId *o = out + w * E;
    for (int e = 0; e < E; ++e) {
      o[e] = pal[v & M];
      v >>= B;
    }
  }
}

template <int B>
void decodeIndicesT(const uint64_t *data, uint16_t *out) {
  constexpr int E = 64 / B;
  constexpr uint64_t M = (uint64_t(1) << B) - 1;
  constexpr int W = kChunkVolume / E;
  for (int w = 0; w < W; ++w) {
    uint64_t v = data[w];
    uint16_t *o = out + w * E;
    for (int e = 0; e < E; ++e) {
      o[e] = uint16_t(v & M);
      v >>= B;
    }
  }
}

template <int B>
void encodeIndicesT(const uint16_t *in, uint64_t *data) {
  constexpr int E = 64 / B;
  constexpr int W = kChunkVolume / E;
  for (int w = 0; w < W; ++w) {
    uint64_t v = 0;
    const uint16_t *s = in + w * E;
    for (int e = 0; e < E; ++e)
      v |= uint64_t(s[e]) << (e * B);
    data[w] = v;
  }
}

void decodeIndices(int bits, const uint64_t *data, uint16_t *out) {
  switch (bits) {
  case 1: decodeIndicesT<1>(data, out); break;
  case 2: decodeIndicesT<2>(data, out); break;
  case 4: decodeIndicesT<4>(data, out); break;
  case 8: decodeIndicesT<8>(data, out); break;
  case 16: decodeIndicesT<16>(data, out); break;
  default: std::fill(out, out + kChunkVolume, uint16_t(0)); break;
  }
}

void encodeIndices(int bits, const uint16_t *in, uint64_t *data) {
  switch (bits) {
  case 1: encodeIndicesT<1>(in, data); break;
  case 2: encodeIndicesT<2>(in, data); break;
  case 4: encodeIndicesT<4>(in, data); break;
  case 8: encodeIndicesT<8>(in, data); break;
  case 16: encodeIndicesT<16>(in, data); break;
  default: break;
  }
}

// Per-thread scratch (no per-call heap allocation after first use).
struct CodecScratch {
  std::vector<uint16_t> indices = std::vector<uint16_t>(kChunkVolume);
  std::vector<BlockId> ids = std::vector<BlockId>(kChunkVolume);
  std::vector<int32_t> lookup = std::vector<int32_t>(65536, -1); // id -> palette index
};
CodecScratch &scratch() {
  thread_local CodecScratch s;
  return s;
}

void putVarint(std::vector<uint8_t> &out, uint32_t v) {
  while (v >= 0x80) {
    out.push_back(uint8_t(v | 0x80));
    v >>= 7;
  }
  out.push_back(uint8_t(v));
}
size_t varintSize(uint32_t v) {
  size_t n = 1;
  while (v >= 0x80) {
    v >>= 7;
    ++n;
  }
  return n;
}

struct Reader {
  const uint8_t *p;
  size_t n, pos = 0;
  bool ok = true;
  uint8_t u8() {
    if (pos >= n) {
      ok = false;
      return 0;
    }
    return p[pos++];
  }
  uint32_t varint() {
    uint32_t v = 0;
    for (int shift = 0; shift < 35; shift += 7) {
      if (pos >= n) {
        ok = false;
        return 0;
      }
      const uint8_t b = p[pos++];
      if (shift == 28 && (b & 0xF0)) { // would overflow 32 bits
        ok = false;
        return 0;
      }
      v |= uint32_t(b & 0x7F) << shift;
      if (!(b & 0x80))
        return v;
    }
    ok = false;
    return 0;
  }
};

} // namespace

Chunk::Chunk() : palette_{kAir} {}
Chunk::Chunk(BlockId fill) : palette_{fill} {}

uint32_t Chunk::readIndex(int i) const {
  const int lb = log2Bits(bits_);
  const int epwShift = 6 - lb;
  const uint64_t word = data_[size_t(i >> epwShift)];
  const int shift = (i & ((1 << epwShift) - 1)) << lb;
  return uint32_t((word >> shift) & ((uint64_t(1) << bits_) - 1));
}

void Chunk::writeIndex(int i, uint32_t v) {
  const int lb = log2Bits(bits_);
  const int epwShift = 6 - lb;
  uint64_t &word = data_[size_t(i >> epwShift)];
  const int shift = (i & ((1 << epwShift) - 1)) << lb;
  const uint64_t mask = ((uint64_t(1) << bits_) - 1) << shift;
  word = (word & ~mask) | ((uint64_t(v) << shift) & mask);
}

void Chunk::growBits(int newBits) {
  if (newBits <= bits_)
    return;
  CodecScratch &s = scratch();
  decodeIndices(bits_, data_.data(), s.indices.data());
  std::vector<uint64_t> fresh(wordsFor(newBits), 0);
  encodeIndices(newBits, s.indices.data(), fresh.data());
  data_.swap(fresh);
  bits_ = uint8_t(newBits);
}

BlockId Chunk::get(int x, int y, int z) const {
  // Out of range is always air (also for uniform chunks, as documented).
  if (unsigned(x) >= unsigned(kChunkSize) || unsigned(y) >= unsigned(kChunkSize) ||
      unsigned(z) >= unsigned(kChunkSize))
    return kAir;
  if (bits_ == 0)
    return palette_.empty() ? kAir : palette_[0];
  return palette_[readIndex(localIndex(x, y, z))];
}

void Chunk::set(int x, int y, int z, BlockId id) {
  if (unsigned(x) >= unsigned(kChunkSize) || unsigned(y) >= unsigned(kChunkSize) ||
      unsigned(z) >= unsigned(kChunkSize))
    return;
  if (palette_.empty())
    palette_.push_back(kAir);
  if (get(x, y, z) == id)
    return;
  size_t idx = 0;
  const size_t n = palette_.size();
  while (idx < n && palette_[idx] != id)
    ++idx;
  if (idx == n) {
    palette_.push_back(id);
    const uint8_t need = bitsFor(palette_.size());
    if (need > bits_)
      growBits(need);
  }
  writeIndex(localIndex(x, y, z), uint32_t(idx));
  ++revision_;
}

void Chunk::fill(BlockId id) {
  palette_.assign(1, id);
  data_.clear();
  data_.shrink_to_fit();
  bits_ = 0;
  ++revision_;
}

void Chunk::decodeAll(BlockId *out) const {
  const BlockId *pal = palette_.data();
  switch (bits_) {
  case 1: decodeT<1>(data_.data(), pal, out); break;
  case 2: decodeT<2>(data_.data(), pal, out); break;
  case 4: decodeT<4>(data_.data(), pal, out); break;
  case 8: decodeT<8>(data_.data(), pal, out); break;
  case 16: decodeT<16>(data_.data(), pal, out); break;
  default: std::fill(out, out + kChunkVolume, uniformBlock()); break;
  }
}

void Chunk::encodeAll(const BlockId *in) {
  CodecScratch &s = scratch();
  palette_.clear();
  for (int i = 0; i < kChunkVolume; ++i) {
    const BlockId id = in[i];
    int32_t &slot = s.lookup[id];
    if (slot < 0) {
      slot = int32_t(palette_.size());
      palette_.push_back(id);
    }
    s.indices[size_t(i)] = uint16_t(slot);
  }
  for (BlockId id : palette_) // reset only the touched lookup entries
    s.lookup[id] = -1;
  bits_ = bitsFor(palette_.size());
  if (bits_ == 0) {
    data_.clear();
    data_.shrink_to_fit();
  } else {
    data_.assign(wordsFor(bits_), 0);
    encodeIndices(bits_, s.indices.data(), data_.data());
  }
  ++revision_;
}

void Chunk::compact() {
  if (bits_ == 0) {
    if (palette_.size() > 1)
      palette_.resize(1);
    return;
  }
  CodecScratch &s = scratch();
  decodeAll(s.ids.data());
  const uint32_t rev = revision_;
  encodeAll(s.ids.data());
  palette_.shrink_to_fit();
  data_.shrink_to_fit();
  revision_ = rev; // contents unchanged
}

// Format v1:
//   u8 version, u8 bits, varint paletteSize, paletteSize x u16 LE,
//   if bits > 0: u8 mode, then
//     mode 0 (raw): 512*bits little-endian u64 words
//     mode 1 (rle): (varint runLength-1, varint paletteIndex)* covering 32768
//   The encoder picks whichever mode is smaller.
void Chunk::serialize(std::vector<uint8_t> &out) const {
  out.clear();
  out.push_back(kSerialVersion);
  out.push_back(bits_);
  const uint32_t palN = uint32_t(palette_.empty() ? 1 : palette_.size());
  putVarint(out, palN);
  for (uint32_t i = 0; i < palN; ++i) {
    const BlockId id = palette_.empty() ? kAir : palette_[i];
    out.push_back(uint8_t(id & 0xFF));
    out.push_back(uint8_t(id >> 8));
  }
  if (bits_ == 0)
    return;

  CodecScratch &s = scratch();
  uint16_t *idx = s.indices.data();
  decodeIndices(bits_, data_.data(), idx);
  size_t rleBytes = 0;
  for (int i = 0; i < kChunkVolume;) {
    int j = i + 1;
    while (j < kChunkVolume && idx[j] == idx[i])
      ++j;
    rleBytes += varintSize(uint32_t(j - i - 1)) + varintSize(idx[i]);
    i = j;
  }
  const size_t rawBytes = wordsFor(bits_) * 8;
  if (rleBytes < rawBytes) {
    out.reserve(out.size() + 1 + rleBytes);
    out.push_back(kModeRle);
    for (int i = 0; i < kChunkVolume;) {
      int j = i + 1;
      while (j < kChunkVolume && idx[j] == idx[i])
        ++j;
      putVarint(out, uint32_t(j - i - 1));
      putVarint(out, idx[i]);
      i = j;
    }
  } else {
    const size_t base = out.size() + 1;
    out.push_back(kModeRaw);
    out.resize(base + rawBytes);
    uint8_t *p = out.data() + base;
    for (uint64_t w : data_) {
      for (int b = 0; b < 8; ++b)
        *p++ = uint8_t(w >> (8 * b));
    }
  }
}

bool Chunk::deserialize(std::span<const uint8_t> in) {
  Reader r{in.data(), in.size()};
  if (r.u8() != kSerialVersion || !r.ok)
    return false;
  const uint8_t bits = r.u8();
  if (!r.ok || !validBits(bits))
    return false;
  const uint32_t palN = r.varint();
  if (!r.ok || palN == 0 || palN > 65536 || bitsFor(palN) != bits)
    return false;
  if (in.size() - r.pos < size_t(palN) * 2)
    return false;
  std::vector<BlockId> palette(palN);
  for (uint32_t i = 0; i < palN; ++i) {
    const uint8_t lo = r.u8();
    const uint8_t hi = r.u8();
    palette[i] = BlockId(lo | (hi << 8));
  }
  if (!r.ok)
    return false;

  std::vector<uint64_t> data;
  if (bits > 0) {
    const uint8_t mode = r.u8();
    if (!r.ok)
      return false;
    CodecScratch &s = scratch();
    uint16_t *idx = s.indices.data();
    if (mode == kModeRaw) {
      const size_t words = wordsFor(bits);
      if (in.size() - r.pos != words * 8)
        return false;
      data.resize(words);
      const uint8_t *p = in.data() + r.pos;
      for (size_t w = 0; w < words; ++w) {
        uint64_t v = 0;
        for (int b = 0; b < 8; ++b)
          v |= uint64_t(p[w * 8 + size_t(b)]) << (8 * b);
        data[w] = v;
      }
      r.pos += words * 8;
      decodeIndices(bits, data.data(), idx);
      for (int i = 0; i < kChunkVolume; ++i)
        if (idx[i] >= palN)
          return false;
    } else if (mode == kModeRle) {
      int filled = 0;
      while (filled < kChunkVolume) {
        const uint32_t run = r.varint();
        const uint32_t value = r.varint();
        if (!r.ok || value >= palN || run >= uint32_t(kChunkVolume - filled))
          return false;
        std::fill(idx + filled, idx + filled + int(run) + 1, uint16_t(value));
        filled += int(run) + 1;
      }
      if (r.pos != in.size())
        return false;
      data.assign(wordsFor(bits), 0);
      encodeIndices(bits, idx, data.data());
    } else {
      return false;
    }
  } else if (r.pos != in.size()) {
    return false;
  }

  palette_.swap(palette);
  data_.swap(data);
  bits_ = bits;
  ++revision_;
  return true;
}

size_t Chunk::memoryBytes() const {
  return sizeof(Chunk) + palette_.capacity() * sizeof(BlockId) +
         data_.capacity() * sizeof(uint64_t);
}

} // namespace atm::voxel
