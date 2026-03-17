#pragma once

#include "NetTypes.h"

#include <cstddef>
#include <cstdint>

namespace attome::net {

template <typename Enum, size_t N = 65536> struct OpcodeRegistry {
  OpcodeInfo table[N]{};

  inline const OpcodeInfo &get(Enum op) const {
    return table[static_cast<uint16_t>(op)];
  }

  inline bool is_dynamic(Enum op) const {
    return get(op).payload_size == NET_PAYLOAD_DYNAMIC;
  }

  inline int16_t fixed_size(Enum op) const { return get(op).payload_size; }
};

} // namespace attome::net
