#pragma once

#include "../NetConfig.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace attome::net {

class ActionBuilder {
public:
  ActionBuilder() { reset(); }

  void reset();

  ActionBuilder &write_i8(int8_t v);
  ActionBuilder &write_u8(uint8_t v);
  ActionBuilder &write_i16(int16_t v);
  ActionBuilder &write_u16(uint16_t v);
  ActionBuilder &write_i32(int32_t v);
  ActionBuilder &write_u32(uint32_t v);
  ActionBuilder &write_f32(float v);
  ActionBuilder &write_str(std::string_view v);

  std::span<const uint8_t> build() const;

  bool overflowed() const { return overflow_; }

private:
  uint8_t staging_[NET_MTU];
  uint16_t pos_{0};
  bool overflow_{false};

  bool can_write(uint16_t bytes) const;
  void write_bytes(const void *src, uint16_t bytes);
  void write_u16_le(uint16_t v);
  void write_u32_le(uint32_t v);
};

} // namespace attome::net
