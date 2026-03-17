#include "ActionBuilder.h"

#include <cstring>

namespace attome::net {

void ActionBuilder::reset() {
  pos_ = 0;
  overflow_ = false;
}

bool ActionBuilder::can_write(uint16_t bytes) const {
  if (overflow_) {
    return false;
  }
  if (bytes > NET_MTU) {
    return false;
  }
  return (pos_ <= NET_MTU) && (static_cast<uint32_t>(pos_) + bytes <= NET_MTU);
}

void ActionBuilder::write_bytes(const void *src, uint16_t bytes) {
  if (!can_write(bytes)) {
    overflow_ = true;
    return;
  }
  std::memcpy(&staging_[pos_], src, bytes);
  pos_ = static_cast<uint16_t>(pos_ + bytes);
}

void ActionBuilder::write_u16_le(uint16_t v) {
  const uint8_t b[2] = {static_cast<uint8_t>(v & 0xFFu),
                        static_cast<uint8_t>((v >> 8) & 0xFFu)};
  write_bytes(b, 2);
}

void ActionBuilder::write_u32_le(uint32_t v) {
  const uint8_t b[4] = {static_cast<uint8_t>(v & 0xFFu),
                        static_cast<uint8_t>((v >> 8) & 0xFFu),
                        static_cast<uint8_t>((v >> 16) & 0xFFu),
                        static_cast<uint8_t>((v >> 24) & 0xFFu)};
  write_bytes(b, 4);
}

ActionBuilder &ActionBuilder::write_i8(int8_t v) {
  const uint8_t b = static_cast<uint8_t>(v);
  write_bytes(&b, 1);
  return *this;
}

ActionBuilder &ActionBuilder::write_u8(uint8_t v) {
  write_bytes(&v, 1);
  return *this;
}

ActionBuilder &ActionBuilder::write_i16(int16_t v) {
  write_u16_le(static_cast<uint16_t>(v));
  return *this;
}

ActionBuilder &ActionBuilder::write_u16(uint16_t v) {
  write_u16_le(v);
  return *this;
}

ActionBuilder &ActionBuilder::write_i32(int32_t v) {
  write_u32_le(static_cast<uint32_t>(v));
  return *this;
}

ActionBuilder &ActionBuilder::write_u32(uint32_t v) {
  write_u32_le(v);
  return *this;
}

ActionBuilder &ActionBuilder::write_f32(float v) {
  static_assert(sizeof(float) == 4, "Expected 32-bit float.");
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  write_u32_le(bits);
  return *this;
}

ActionBuilder &ActionBuilder::write_str(std::string_view v) {
  if (v.size() > UINT16_MAX) {
    overflow_ = true;
    return *this;
  }

  write_u16_le(static_cast<uint16_t>(v.size()));
  if (overflow_) {
    return *this;
  }

  write_bytes(v.data(), static_cast<uint16_t>(v.size()));
  return *this;
}

std::span<const uint8_t> ActionBuilder::build() const {
  if (overflow_) {
    return {};
  }
  return {staging_, pos_};
}

} // namespace attome::net

