#pragma once

// X-macro message schema (NETWORK_PLAN §6.9).
//
//   #define MY_MSG(F) F(uint8_t, slot, Bits<3>) F(std::string, text, Str<200>)
//   ATM_DEFINE_MESSAGE(MyMsg, 42, MY_MSG)
//
// generates, in the current namespace:
//   struct MyMsg { static constexpr uint16_t kId = 42; uint8_t slot{}; std::string text{};
//                  static constexpr uint64_t messageSchemaHash(); };
//   (named so that a field called `schemaHash` - Hello/Welcome - cannot clash
//   with it: a data member and a member function may not share a name)
//   inline void encode(atm::net2::BitWriter&, const MyMsg&);
//   inline bool decode(atm::net2::BitReader&, MyMsg&);   // false on malformed input
// encode/decode are free functions next to the struct, so they are found by
// argument-dependent lookup (encodeMessage / decodeMessage below use that).
//
// Wire form of a whole message: varu(kId) followed by the fields in order.
//
// The codec argument of F(...) may contain commas (List<Custom<X>, 3>): it is
// taken as __VA_ARGS__. Codec names are looked up with
// `using namespace atm::net2::schema` in effect, plus the enclosing namespace
// of the message (for Custom<T> codec types).

#include "BitStream.h"

#include <glm/glm.hpp>

#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace atm::net2::schema {

// ---- compile-time hashing ---------------------------------------------------
inline constexpr uint64_t kFnvBasis = 1469598103934665603ull;
inline constexpr uint64_t kFnvPrime = 1099511628211ull;

constexpr uint64_t fnv1a(const char *s, uint64_t h = kFnvBasis) {
  while (*s) {
    h ^= uint64_t(uint8_t(*s++));
    h *= kFnvPrime;
  }
  return h;
}
constexpr uint64_t mix(uint64_t h, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    h ^= (v >> (i * 8)) & 0xFFu;
    h *= kFnvPrime;
  }
  return h;
}

// ---- codecs -----------------------------------------------------------------
// Each codec: static encode(BitWriter&, const T&), static bool decode(BitReader&, T&),
// static constexpr uint64_t kHash.

struct U8 {
  static constexpr uint64_t kHash = fnv1a("U8");
  template <class T> static void encode(BitWriter &w, const T &v) { w.u8(uint8_t(v)); }
  template <class T> static bool decode(BitReader &r, T &v) { v = T(r.u8()); return r.ok(); }
};
struct U16 {
  static constexpr uint64_t kHash = fnv1a("U16");
  template <class T> static void encode(BitWriter &w, const T &v) { w.u16(uint16_t(v)); }
  template <class T> static bool decode(BitReader &r, T &v) { v = T(r.u16()); return r.ok(); }
};
struct U32 {
  static constexpr uint64_t kHash = fnv1a("U32");
  template <class T> static void encode(BitWriter &w, const T &v) { w.u32(uint32_t(v)); }
  template <class T> static bool decode(BitReader &r, T &v) { v = T(r.u32()); return r.ok(); }
};
struct U64 {
  static constexpr uint64_t kHash = fnv1a("U64");
  template <class T> static void encode(BitWriter &w, const T &v) { w.u64(uint64_t(v)); }
  template <class T> static bool decode(BitReader &r, T &v) { v = T(r.u64()); return r.ok(); }
};
struct I32 {
  static constexpr uint64_t kHash = fnv1a("I32");
  template <class T> static void encode(BitWriter &w, const T &v) { w.i32(int32_t(v)); }
  template <class T> static bool decode(BitReader &r, T &v) { v = T(r.i32()); return r.ok(); }
};
struct F32 {
  static constexpr uint64_t kHash = fnv1a("F32");
  static void encode(BitWriter &w, float v) { w.f32(v); }
  static bool decode(BitReader &r, float &v) {
    v = r.f32();
    if (!std::isfinite(v)) v = 0.0f; // never let NaN/inf into the simulation
    return r.ok();
  }
};
struct Bool {
  static constexpr uint64_t kHash = fnv1a("Bool");
  static void encode(BitWriter &w, bool v) { w.boolean(v); }
  static bool decode(BitReader &r, bool &v) { v = r.boolean(); return r.ok(); }
};
struct VarU {
  static constexpr uint64_t kHash = fnv1a("VarU");
  template <class T> static void encode(BitWriter &w, const T &v) { w.varu(uint64_t(v)); }
  template <class T> static bool decode(BitReader &r, T &v) { v = T(r.varu()); return r.ok(); }
};

template <int N> struct Bits {
  static_assert(N >= 1 && N <= 32, "Bits<N>: 1..32");
  static constexpr uint64_t kHash = mix(fnv1a("Bits"), uint64_t(N));
  template <class T> static void encode(BitWriter &w, const T &v) { w.bits(uint32_t(v), N); }
  template <class T> static bool decode(BitReader &r, T &v) { v = T(r.bits(N)); return r.ok(); }
};

template <size_t MaxLen> struct Str {
  static constexpr uint64_t kHash = mix(fnv1a("Str"), uint64_t(MaxLen));
  static void encode(BitWriter &w, const std::string &s) { w.string(s, MaxLen); }
  static bool decode(BitReader &r, std::string &s) { s = r.string(MaxLen); return r.ok(); }
};

template <size_t MaxLen> struct Blob {
  static constexpr uint64_t kHash = mix(fnv1a("Blob"), uint64_t(MaxLen));
  static void encode(BitWriter &w, const std::vector<uint8_t> &b) {
    const size_t n = b.size() < MaxLen ? b.size() : MaxLen;
    w.varu(n);
    w.raw(b.data(), n);
  }
  static bool decode(BitReader &r, std::vector<uint8_t> &b) { return r.bytesInto(b, MaxLen); }
};

inline constexpr float kPi = 3.14159265358979323846f;
inline float wrapAngle(float a) { // -> [-pi, pi)
  if (!std::isfinite(a)) return 0.0f;
  const float twoPi = 2.0f * kPi;
  a = std::fmod(a + kPi, twoPi);
  if (a < 0.0f) a += twoPi;
  return a - kPi;
}

template <int NBits> struct Angle {
  static_assert(NBits >= 2 && NBits <= 16, "Angle<Bits>: 2..16");
  static constexpr uint64_t kHash = mix(fnv1a("Angle"), uint64_t(NBits));
  static void encode(BitWriter &w, float a) {
    const float t = (wrapAngle(a) + kPi) / (2.0f * kPi); // [0,1)
    const uint32_t q = uint32_t(std::lround(double(t) * double(1u << NBits))) & ((1u << NBits) - 1u);
    w.bits(q, NBits);
  }
  static bool decode(BitReader &r, float &a) {
    const uint32_t q = r.bits(NBits);
    a = float(double(q) / double(1u << NBits)) * 2.0f * kPi - kPi;
    return r.ok();
  }
};

// Float in [MinMilli/1000, MaxMilli/1000] with NBits bits.
template <int32_t MinMilli, int32_t MaxMilli, int NBits> struct QuantF {
  static_assert(MaxMilli > MinMilli, "QuantF: Max > Min");
  static_assert(NBits >= 1 && NBits <= 31, "QuantF: 1..31 bits");
  static constexpr uint64_t kHash =
      mix(mix(mix(fnv1a("QuantF"), uint64_t(uint32_t(MinMilli))), uint64_t(uint32_t(MaxMilli))), uint64_t(NBits));
  static void encode(BitWriter &w, float v) {
    w.quant(v, float(MinMilli) / 1000.0f, float(MaxMilli) / 1000.0f, NBits);
  }
  static bool decode(BitReader &r, float &v) {
    v = r.quant(float(MinMilli) / 1000.0f, float(MaxMilli) / 1000.0f, NBits);
    return r.ok();
  }
};

// World position: 1/64 block. x/z as int32 (±33M blocks), y as u16 (0..1023.98).
struct Pos {
  static constexpr uint64_t kHash = fnv1a("Pos.64.i32.u16");
  static int32_t qxz(double v) {
    if (!std::isfinite(v)) return 0;
    const double q = std::round(v * 64.0);
    if (q > 2147483647.0) return 2147483647;
    if (q < -2147483648.0) return int32_t(-2147483647 - 1);
    return int32_t(q);
  }
  static uint16_t qy(double v) {
    if (!std::isfinite(v)) return 0;
    const double q = std::round(v * 64.0);
    if (q < 0.0) return 0;
    if (q > 65535.0) return 65535;
    return uint16_t(q);
  }
  static void encode(BitWriter &w, const glm::dvec3 &p) {
    w.i32(qxz(p.x));
    w.u16(qy(p.y));
    w.i32(qxz(p.z));
  }
  static bool decode(BitReader &r, glm::dvec3 &p) {
    p.x = double(r.i32()) / 64.0;
    p.y = double(r.u16()) / 64.0;
    p.z = double(r.i32()) / 64.0;
    return r.ok();
  }
};

// Velocity: 1/32 block/s, signed 16 bits per axis (±1024 b/s).
struct Vel {
  static constexpr uint64_t kHash = fnv1a("Vel.32.i16");
  static uint16_t q(float v) {
    if (!std::isfinite(v)) return 0;
    float s = std::round(v * 32.0f);
    if (s > 32767.0f) s = 32767.0f;
    if (s < -32768.0f) s = -32768.0f;
    return uint16_t(int16_t(s));
  }
  static float dq(uint32_t bitsv) { return float(int16_t(uint16_t(bitsv))) / 32.0f; }
  static void encode(BitWriter &w, const glm::vec3 &v) {
    w.u16(q(v.x));
    w.u16(q(v.y));
    w.u16(q(v.z));
  }
  static bool decode(BitReader &r, glm::vec3 &v) {
    v.x = dq(r.u16());
    v.y = dq(r.u16());
    v.z = dq(r.u16());
    return r.ok();
  }
};

template <class Codec, size_t MaxN> struct List {
  static constexpr uint64_t kHash = mix(mix(fnv1a("List"), Codec::kHash), uint64_t(MaxN));
  template <class T> static void encode(BitWriter &w, const std::vector<T> &v) {
    const size_t n = v.size() < MaxN ? v.size() : MaxN;
    w.varu(n);
    for (size_t i = 0; i < n; ++i) Codec::encode(w, v[i]);
  }
  template <class T> static bool decode(BitReader &r, std::vector<T> &v) {
    const uint64_t n = r.varu();
    // Each element takes at least one bit, so a count larger than the bits
    // left is malformed (stops huge allocations from tiny packets).
    if (!r.ok() || n > MaxN || n > r.remainingBits()) return false;
    v.clear();
    v.resize(size_t(n));
    for (size_t i = 0; i < size_t(n); ++i)
      if (!Codec::decode(r, v[i])) return false;
    return r.ok();
  }
};

template <class T> struct Custom {
  static constexpr uint64_t kHash = mix(fnv1a("Custom"), fnv1a(T::kName));
  template <class V> static void encode(BitWriter &w, const V &v) { T::encode(w, v); }
  template <class V> static bool decode(BitReader &r, V &v) { return T::decode(r, v) && r.ok(); }
};

} // namespace atm::net2::schema

// ---- message generation ---------------------------------------------------------

#define ATM_SCHEMA_FIELD_DECL(type, name, ...) type name{};
#define ATM_SCHEMA_FIELD_HASH(type, name, ...)                                      \
  h = ::atm::net2::schema::fnv1a(#type, h);                                         \
  h = ::atm::net2::schema::fnv1a(#name, h);                                         \
  h = ::atm::net2::schema::mix(h, __VA_ARGS__::kHash);
#define ATM_SCHEMA_FIELD_ENC(type, name, ...) __VA_ARGS__::encode(w, m.name);
#define ATM_SCHEMA_FIELD_DEC(type, name, ...)                                       \
  if (!__VA_ARGS__::decode(r, m.name)) return false;

#define ATM_DEFINE_MESSAGE(Name, Id, FIELDS)                                        \
  struct Name {                                                                     \
    static constexpr uint16_t kId = Id;                                             \
    static constexpr const char *kName = #Name;                                     \
    FIELDS(ATM_SCHEMA_FIELD_DECL)                                                   \
    static constexpr uint64_t messageSchemaHash();                                  \
  };                                                                                \
  constexpr uint64_t Name::messageSchemaHash() {                                    \
    using namespace ::atm::net2::schema;                                            \
    uint64_t h = ::atm::net2::schema::fnv1a(#Name);                                 \
    h = ::atm::net2::schema::mix(h, uint64_t(Id));                                  \
    FIELDS(ATM_SCHEMA_FIELD_HASH)                                                   \
    return h;                                                                       \
  }                                                                                 \
  inline void encode(::atm::net2::BitWriter &w, const Name &m) {                    \
    using namespace ::atm::net2::schema;                                            \
    (void)w;                                                                        \
    (void)m;                                                                        \
    FIELDS(ATM_SCHEMA_FIELD_ENC)                                                    \
  }                                                                                 \
  inline bool decode(::atm::net2::BitReader &r, Name &m) {                          \
    using namespace ::atm::net2::schema;                                            \
    (void)m;                                                                        \
    FIELDS(ATM_SCHEMA_FIELD_DEC)                                                    \
    return r.ok();                                                                  \
  }

namespace atm::net2 {

// Combined schema hash of a set of messages (order-sensitive).
template <class... Msgs> constexpr uint64_t schemaHash() {
  uint64_t h = schema::fnv1a("atm.schema.v1");
  ((h = schema::mix(h, Msgs::messageSchemaHash())), ...);
  return h;
}

// Encodes varu(M::kId) + fields into `buf` (cleared first; its capacity is
// reused). On overflow the buffer is returned empty.
template <class M>
std::vector<uint8_t> &encodeMessage(std::vector<uint8_t> &buf, const M &m, size_t maxBytes = 1u << 20) {
  buf.clear();
  BitWriter w(buf, maxBytes);
  w.varu(M::kId);
  encode(w, m); // ADL: the generated free function next to M
  if (w.overflow()) buf.clear();
  return buf;
}

// Reads the message id at the start of `data`; `rest` is positioned at the
// first field. Dispatch on `id`, then call decodeMessage(rest, msg).
inline bool peekMessageId(std::span<const uint8_t> data, uint16_t &id, BitReader &rest) {
  rest = BitReader(data);
  const uint64_t v = rest.varu();
  if (!rest.ok() || v > 0xFFFFu) return false;
  id = uint16_t(v);
  return true;
}

template <class M> bool decodeMessage(BitReader &r, M &m) { return decode(r, m) && r.ok(); }

// Decodes a whole message including its id; false if the id doesn't match.
template <class M> bool decodeMessage(std::span<const uint8_t> data, M &m) {
  BitReader r(data);
  uint16_t id = 0;
  if (!peekMessageId(data, id, r) || id != M::kId) return false;
  return decode(r, m) && r.ok();
}

} // namespace atm::net2
