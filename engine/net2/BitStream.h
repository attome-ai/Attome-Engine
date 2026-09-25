#pragma once

// Bit-packed serialisation for game messages (NETWORK_PLAN §6.3).
// Writers never fail loudly: overflow sets a flag and further writes are
// ignored. Readers never read out of bounds: underflow sets `ok() == false`
// and returns zeros. Callers check ok() once at the end.
//
// Bit order: least significant bit first, bytes filled from bit 0 upwards.
// A BitWriter appends to the vector it is given (it starts writing at
// out.size() bytes); clear() the vector first to reuse it for a new message.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace atm::net2 {

class BitWriter {
public:
  explicit BitWriter(std::vector<uint8_t> &out, size_t maxBytes = 1u << 20)
      : out_(out), maxBytes_(maxBytes), bitPos_(out.size() * 8) {}

  void bits(uint32_t value, int count) { // count 1..32
    if (overflow_ || count <= 0) return;
    if (count > 32) count = 32;
    if (count < 32) value &= (1u << count) - 1u;
    const size_t needBytes = (bitPos_ + size_t(count) + 7) >> 3;
    if (needBytes > out_.size()) {
      if (needBytes > maxBytes_) { overflow_ = true; return; }
      out_.resize(needBytes, 0);
    }
    while (count > 0) {
      const int bitOff = int(bitPos_ & 7);
      const int take = (8 - bitOff) < count ? (8 - bitOff) : count;
      const uint32_t part = value & ((1u << take) - 1u);
      out_[bitPos_ >> 3] = uint8_t(out_[bitPos_ >> 3] | uint8_t(part << bitOff));
      value = take < 32 ? (value >> take) : 0u;
      bitPos_ += size_t(take);
      count -= take;
    }
  }
  void boolean(bool v) { bits(v ? 1u : 0u, 1); }
  void u8(uint8_t v) { bits(v, 8); }
  void u16(uint16_t v) { bits(v, 16); }
  void u32(uint32_t v) { bits(v, 32); }
  void u64(uint64_t v) { bits(uint32_t(v), 32); bits(uint32_t(v >> 32), 32); }
  void i32(int32_t v) { bits(uint32_t(v), 32); }
  void f32(float v) { uint32_t u; std::memcpy(&u, &v, 4); bits(u, 32); }
  void f64(double v) { uint64_t u; std::memcpy(&u, &v, 8); u64(u); }
  // Variable-length unsigned (7 bits per group): small numbers stay small.
  void varu(uint64_t v) {
    do { bits(uint32_t(v & 0x7F) | (v > 0x7F ? 0x80u : 0u), 8); v >>= 7; } while (v);
  }
  // Quantised float in [min, max] with the given number of bits (1..31).
  void quant(float v, float min, float max, int count) {
    if (count < 1) count = 1;
    if (count > 31) count = 31;
    float t = (max > min) ? (v - min) / (max - min) : 0.0f;
    if (!(t >= 0.0f)) t = 0.0f; // also catches NaN
    if (t > 1.0f) t = 1.0f;
    bits(uint32_t(std::lround(double(t) * double((1u << count) - 1u))), count);
  }
  void string(std::string_view s, size_t maxLen = 255) {
    const size_t n = s.size() < maxLen ? s.size() : maxLen;
    varu(n);
    raw(reinterpret_cast<const uint8_t *>(s.data()), n);
  }
  void bytes(std::span<const uint8_t> b) { varu(b.size()); raw(b.data(), b.size()); }
  // Raw bytes without a length prefix (memcpy when byte aligned).
  void raw(const uint8_t *data, size_t n) {
    if (n == 0 || overflow_) return;
    if ((bitPos_ & 7) == 0) {
      const size_t start = bitPos_ >> 3;
      if (start + n > maxBytes_) { overflow_ = true; return; }
      if (start + n > out_.size()) out_.resize(start + n, 0);
      std::memcpy(out_.data() + start, data, n);
      bitPos_ += n * 8;
      return;
    }
    for (size_t i = 0; i < n; ++i) u8(data[i]);
  }

  size_t bitsWritten() const { return bitPos_; }
  size_t bytesWritten() const { return (bitPos_ + 7) >> 3; }
  bool overflow() const { return overflow_; }

private:
  std::vector<uint8_t> &out_;
  size_t maxBytes_;
  size_t bitPos_ = 0;
  bool overflow_ = false;
};

class BitReader {
public:
  explicit BitReader(std::span<const uint8_t> in) : in_(in) {}

  uint32_t bits(int count) {
    if (count <= 0) return 0;
    if (count > 32) count = 32;
    if (!ok_ || bitPos_ + size_t(count) > in_.size() * 8) {
      ok_ = false;
      bitPos_ = in_.size() * 8;
      return 0;
    }
    uint32_t v = 0;
    int got = 0;
    while (got < count) {
      const int bitOff = int(bitPos_ & 7);
      const int take = (8 - bitOff) < (count - got) ? (8 - bitOff) : (count - got);
      const uint32_t part = (uint32_t(in_[bitPos_ >> 3]) >> bitOff) & ((1u << take) - 1u);
      v |= part << got;
      got += take;
      bitPos_ += size_t(take);
    }
    return v;
  }
  bool boolean() { return bits(1) != 0; }
  uint8_t u8() { return uint8_t(bits(8)); }
  uint16_t u16() { return uint16_t(bits(16)); }
  uint32_t u32() { return bits(32); }
  uint64_t u64() { const uint64_t lo = bits(32); return lo | (uint64_t(bits(32)) << 32); }
  int32_t i32() { return int32_t(bits(32)); }
  float f32() { const uint32_t u = bits(32); float f; std::memcpy(&f, &u, 4); return f; }
  double f64() { const uint64_t u = u64(); double d; std::memcpy(&d, &u, 8); return d; }
  uint64_t varu() {
    uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      const uint32_t b = bits(8);
      if (!ok_) return 0;
      v |= uint64_t(b & 0x7F) << shift;
      if (!(b & 0x80)) return v;
    }
    ok_ = false;
    return 0;
  }
  float quant(float min, float max, int count) {
    if (count < 1) count = 1;
    if (count > 31) count = 31;
    const uint32_t q = bits(count);
    return min + (max - min) * float(double(q) / double((1u << count) - 1u));
  }
  std::string string(size_t maxLen = 255) {
    const uint64_t n = varu();
    if (!ok_ || n > maxLen || n > remainingBytes()) { ok_ = false; return {}; }
    std::string s(size_t(n), '\0');
    if (n) raw(reinterpret_cast<uint8_t *>(s.data()), size_t(n));
    return s;
  }
  // Returns a copy (sizes are bounded by maxLen to stop memory abuse).
  std::vector<uint8_t> bytes(size_t maxLen) {
    std::vector<uint8_t> b;
    bytesInto(b, maxLen);
    return b;
  }
  // Same as bytes() but reuses `out`'s storage.
  bool bytesInto(std::vector<uint8_t> &out, size_t maxLen) {
    out.clear();
    const uint64_t n = varu();
    if (!ok_ || n > maxLen || n > remainingBytes()) { ok_ = false; return false; }
    out.resize(size_t(n));
    if (n) raw(out.data(), size_t(n));
    return ok_;
  }
  // Raw bytes without a length prefix (memcpy when byte aligned).
  bool raw(uint8_t *dst, size_t n) {
    if (n == 0) return ok_;
    if (!ok_ || n > remainingBytes()) { ok_ = false; bitPos_ = in_.size() * 8; std::memset(dst, 0, n); return false; }
    if ((bitPos_ & 7) == 0) {
      std::memcpy(dst, in_.data() + (bitPos_ >> 3), n);
      bitPos_ += n * 8;
      return true;
    }
    for (size_t i = 0; i < n; ++i) dst[i] = u8();
    return ok_;
  }

  bool ok() const { return ok_; }
  // Bytes not yet touched (a partly read byte counts as used).
  size_t remainingBytes() const {
    const size_t used = (bitPos_ + 7) >> 3;
    return used >= in_.size() ? 0 : in_.size() - used;
  }
  size_t remainingBits() const {
    const size_t total = in_.size() * 8;
    return bitPos_ >= total ? 0 : total - bitPos_;
  }
  size_t bitsRead() const { return bitPos_; }

private:
  std::span<const uint8_t> in_;
  size_t bitPos_ = 0;
  bool ok_ = true;
};

} // namespace atm::net2
