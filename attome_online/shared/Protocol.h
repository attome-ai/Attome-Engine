#pragma once

// Game protocol: every message is declared once as an X-macro table
// (NETWORK_PLAN §6.9). engine/net2/Schema.h turns each table into:
//   struct <Name> { fields... };
//   void encode(BitWriter&, const <Name>&);  bool decode(BitReader&, <Name>&);
//   constexpr uint16_t <Name>::kId;  and a constexpr schema hash over all
//   message names, field names and codecs (exchanged in Hello/Welcome).
//
// Field line:  F(cpp_type, field_name, Codec)
// Codecs (engine/net2/Schema.h): U8 U16 U32 U64 I32 F32 Bool VarU
//   Bits<N>                   unsigned in N bits
//   Str<MaxLen>               UTF-8 string, length-checked on decode
//   Blob<MaxLen>              std::vector<uint8_t>
//   Angle<Bits>               radians wrapped to [-pi, pi)
//   QuantF<MinMilli, MaxMilli, Bits>  float quantised in [Min/1000, Max/1000]
//                             (integer NTTPs: float NTTPs are not portable yet)
//   Pos                       glm::dvec3 world position, 1/64 block precision (i32 x/z, u16 y)
//   Vel                       glm::vec3, 1/32 block/s precision, 16 bits per axis
//   List<Codec, MaxN>         std::vector of elements, count checked on decode
//   Custom<T>                 T provides static encode/decode (for snapshots)
//
// Channels: see the comment on each message.

#include "GameTypes.h"
#include "Snapshot.h"

#include "../../engine/net2/Schema.h"

#include <string>
#include <vector>

namespace ao::proto {

inline constexpr uint16_t kProtocolVersion = 1;

// ------------------------------- client -> server ---------------------------

// Reliable. First message after the transport connects.
#define AO_MSG_Hello(F)                                                        \
  F(uint64_t, schemaHash, U64)                                                 \
  F(std::string, name, Str<24>)                                                \
  F(atm::model::Appearance, appearance, Custom<AppearanceCodec>)
ATM_DEFINE_MESSAGE(Hello, 1, AO_MSG_Hello)

// Unreliable, every sim tick. Carries the last 3 inputs so one lost packet
// never loses an input (NETWORK_PLAN §6.6).
#define AO_MSG_InputBatch(F)                                                   \
  F(std::vector<MoveInput>, inputs, List<Custom<MoveInputCodec>, 3>)           \
  F(uint32_t, lastSnapshotTick, U32)
ATM_DEFINE_MESSAGE(InputBatch, 2, AO_MSG_InputBatch)

// Reliable. Break or place a block; the server validates reach, permissions,
// tool and inventory, then broadcasts BlockChanged.
#define AO_MSG_BlockAction(F)                                                  \
  F(uint8_t, action, Bits<2>) /* 0 = break, 1 = place */                       \
  F(int32_t, x, I32) F(int32_t, y, I32) F(int32_t, z, I32)                     \
  F(uint8_t, face, Bits<3>)                                                    \
  F(uint8_t, hotbarSlot, Bits<5>)
ATM_DEFINE_MESSAGE(BlockAction, 3, AO_MSG_BlockAction)

// Reliable. Attack in the aim direction with the equipped weapon. `tick` is
// the server tick the client was displaying (interpolated view), used by the
// server to rewind monster positions (lag compensation).
#define AO_MSG_Attack(F)                                                       \
  F(uint32_t, tick, U32)                                                       \
  F(float, yaw, Angle<12>) F(float, pitch, Angle<12>)                          \
  F(uint8_t, ability, Bits<3>) /* 0 primary, 1 secondary, 2 Q, 3 E, 4 R */
ATM_DEFINE_MESSAGE(Attack, 4, AO_MSG_Attack)

// Reliable. Equip an item from the inventory into its slot (or unequip).
#define AO_MSG_Equip(F)                                                        \
  F(uint8_t, inventorySlot, Bits<5>)                                           \
  F(uint8_t, equipSlot, Bits<4>)                                               \
  F(bool, unequip, Bool)
ATM_DEFINE_MESSAGE(Equip, 5, AO_MSG_Equip)

// Reliable.
#define AO_MSG_ChatSend(F) F(std::string, text, Str<200>)
ATM_DEFINE_MESSAGE(ChatSend, 6, AO_MSG_ChatSend)

// ------------------------------- server -> client ---------------------------

// Reliable. Reply to Hello.
#define AO_MSG_Welcome(F)                                                      \
  F(uint64_t, schemaHash, U64)                                                 \
  F(uint32_t, playerEntity, U32)                                               \
  F(uint32_t, serverTick, U32)                                                 \
  F(uint64_t, worldSeed, U64)                                                  \
  F(glm::dvec3, spawn, Pos)
ATM_DEFINE_MESSAGE(Welcome, 64, AO_MSG_Welcome)

// Bulk. One chunk's full contents (Chunk::serialize bytes). Only chunks that
// differ from the generated world are sent; the client generates the rest
// itself from worldSeed.
#define AO_MSG_ChunkData(F)                                                    \
  F(int32_t, cx, I32) F(int32_t, cy, I32) F(int32_t, cz, I32)                  \
  F(std::vector<uint8_t>, data, Blob<70000>)
ATM_DEFINE_MESSAGE(ChunkData, 65, AO_MSG_ChunkData)

// Reliable. Tells the client which chunks near it have edits (so it knows to
// wait for ChunkData instead of using its own generated copy).
#define AO_MSG_EditedChunks(F)                                                 \
  F(std::vector<uint64_t>, keys, List<U64, 4096>)
ATM_DEFINE_MESSAGE(EditedChunks, 66, AO_MSG_EditedChunks)

// Reliable. A single block changed (edit by any player).
#define AO_MSG_BlockChanged(F)                                                 \
  F(int32_t, x, I32) F(int32_t, y, I32) F(int32_t, z, I32)                     \
  F(uint16_t, block, U16)
ATM_DEFINE_MESSAGE(BlockChanged, 67, AO_MSG_BlockChanged)

// Unreliable, kSnapshotHz. See Snapshot.h for the delta/priority encoding.
#define AO_MSG_SnapshotMsg(F) F(Snapshot, snapshot, Custom<SnapshotCodec>)
ATM_DEFINE_MESSAGE(SnapshotMsg, 68, AO_MSG_SnapshotMsg)

// Reliable. Appearance of an entity changed (gear swap) or first seen.
#define AO_MSG_AppearanceMsg(F)                                                \
  F(uint32_t, entity, U32)                                                     \
  F(std::string, name, Str<24>)                                                \
  F(atm::model::Appearance, appearance, Custom<AppearanceCodec>)
ATM_DEFINE_MESSAGE(AppearanceMsg, 69, AO_MSG_AppearanceMsg)

// Reliable. Combat feedback (floating numbers, hit flashes, death).
#define AO_MSG_DamageEvent(F)                                                  \
  F(uint32_t, source, U32) F(uint32_t, target, U32)                            \
  F(uint16_t, amount, U16) F(bool, critical, Bool) F(bool, killed, Bool)
ATM_DEFINE_MESSAGE(DamageEvent, 70, AO_MSG_DamageEvent)

// Reliable. Full inventory + equipment (small; sent on change).
#define AO_MSG_InventoryMsg(F)                                                 \
  F(std::vector<ItemStack>, slots, List<Custom<ItemStackCodec>, 28>)           \
  F(std::vector<uint16_t>, equipped, List<U16, 8>)
ATM_DEFINE_MESSAGE(InventoryMsg, 71, AO_MSG_InventoryMsg)

// Reliable. XP gained (drives XP drops and level-up effects).
#define AO_MSG_XpGain(F)                                                       \
  F(uint8_t, skill, Bits<5>) F(uint32_t, amount, U32) F(uint64_t, totalXp, U64)
ATM_DEFINE_MESSAGE(XpGain, 72, AO_MSG_XpGain)

// Reliable. Personal loot notification (GAME_DESIGN §11: each player sees
// only their own drops).
#define AO_MSG_LootMsg(F)                                                      \
  F(uint32_t, fromEntity, U32) F(uint16_t, item, U16) F(uint16_t, count, U16)  \
  F(bool, rare, Bool)
ATM_DEFINE_MESSAGE(LootMsg, 73, AO_MSG_LootMsg)

// Reliable.
#define AO_MSG_ChatMsg(F) F(std::string, from, Str<24>) F(std::string, text, Str<200>)
ATM_DEFINE_MESSAGE(ChatMsg, 74, AO_MSG_ChatMsg)

// Schema hash over every message above (compile time).
inline constexpr uint64_t kSchemaHash = atm::net2::schemaHash<
    Hello, InputBatch, BlockAction, Attack, Equip, ChatSend, Welcome, ChunkData,
    EditedChunks, BlockChanged, SnapshotMsg, AppearanceMsg, DamageEvent,
    InventoryMsg, XpGain, LootMsg, ChatMsg>();

// Transport-level protocol id (net2::HostConfig::protocolId): peers built
// from a different schema are rejected during the handshake.
inline constexpr uint32_t kTransportProtocolId = uint32_t(kSchemaHash ^ (kSchemaHash >> 32));

// Every message on the wire is: varu(message id) + encoded fields.
// encode()/decode() are free functions in ao::proto (found by ADL).
// Dispatch helpers (encodeMessage / peekMessageId / decodeMessage) are in
// Schema.h.

} // namespace ao::proto
