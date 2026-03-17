#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

namespace attome::net {

struct MessageReader {
  std::span<const uint8_t> data{};
  size_t pos{0};

  bool ok() const { return pos <= data.size(); }

  std::span<const uint8_t> remaining() const {
    if (!ok()) {
      return {};
    }
    return data.subspan(pos);
  }

  int8_t read_i8() { return static_cast<int8_t>(read_u8()); }
  uint8_t read_u8() { return read_pod<uint8_t>(); }

  int16_t read_i16() { return static_cast<int16_t>(read_u16()); }
  uint16_t read_u16() { return read_u16_le(); }

  int32_t read_i32() { return static_cast<int32_t>(read_u32()); }
  uint32_t read_u32() { return read_u32_le(); }

  float read_f32() {
    static_assert(sizeof(float) == 4, "Expected 32-bit float.");
    uint32_t bits = read_u32_le();
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
  }

  std::string_view read_str() {
    const uint16_t len = read_u16_le();
    if (!ok()) {
      return {};
    }
    if (remaining().size() < len) {
      pos = data.size() + 1;
      return {};
    }

    const auto *p = reinterpret_cast<const char *>(data.data() + pos);
    pos += len;
    return std::string_view{p, len};
  }

private:
  template <typename T> T read_pod() {
    if (!ok() || remaining().size() < sizeof(T)) {
      pos = data.size() + 1;
      return T{};
    }
    const T v = *reinterpret_cast<const T *>(data.data() + pos);
    pos += sizeof(T);
    return v;
  }

  uint16_t read_u16_le() {
    if (!ok() || remaining().size() < 2) {
      pos = data.size() + 1;
      return 0;
    }
    const uint8_t b0 = data[pos + 0];
    const uint8_t b1 = data[pos + 1];
    pos += 2;
    return static_cast<uint16_t>(static_cast<uint16_t>(b0) |
                                 (static_cast<uint16_t>(b1) << 8));
  }

  uint32_t read_u32_le() {
    if (!ok() || remaining().size() < 4) {
      pos = data.size() + 1;
      return 0;
    }
    const uint32_t b0 = data[pos + 0];
    const uint32_t b1 = data[pos + 1];
    const uint32_t b2 = data[pos + 2];
    const uint32_t b3 = data[pos + 3];
    pos += 4;
    return (b0) | (b1 << 8) | (b2 << 16) | (b3 << 24);
  }
};

} // namespace attome::net
