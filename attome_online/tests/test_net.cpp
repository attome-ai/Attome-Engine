// Network module tests: bit streams, codecs, X-macro schema, transport.

#include "atm_test.h"

#include "../shared/Protocol.h"
#include "../shared/Snapshot.h"

#include "../../engine/net2/BitStream.h"
#include "../../engine/net2/Net.h"
#include "../../engine/net2/Schema.h"
#include "../../engine/net2/Socket.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <thread>
#include <vector>

namespace {

namespace net = atm::net2;
namespace schema = atm::net2::schema;
using namespace ao;
using namespace ao::proto;

// ---------------------------------------------------------------------------
// BitStream
// ---------------------------------------------------------------------------

ATM_TEST(net_bitstream_roundtrip) {
  std::vector<uint8_t> buf;
  net::BitWriter w(buf);
  w.bits(5, 3);
  w.boolean(true);
  w.u8(200);
  w.bits(0x1ABCD, 17);
  w.u16(65535);
  w.u32(0xDEADBEEFu);
  w.u64(0x0123456789ABCDEFull);
  w.i32(-123456);
  w.f32(3.25f);
  w.f64(-1.0e300);
  w.varu(0);
  w.varu(127);
  w.varu(128);
  w.varu(0xFFFFFFFFFFFFFFFFull);
  w.string("hello");
  const uint8_t raw[5] = {1, 2, 3, 4, 5};
  w.bytes(std::span<const uint8_t>(raw, 5));
  w.bits(0xFFFFFFFFu, 32);
  ATM_CHECK(!w.overflow());

  net::BitReader r(buf);
  ATM_CHECK_EQ(r.bits(3), 5u);
  ATM_CHECK(r.boolean());
  ATM_CHECK_EQ(int(r.u8()), 200);
  ATM_CHECK_EQ(r.bits(17), 0x1ABCDu);
  ATM_CHECK_EQ(int(r.u16()), 65535);
  ATM_CHECK_EQ(r.u32(), 0xDEADBEEFu);
  ATM_CHECK_EQ(r.u64(), 0x0123456789ABCDEFull);
  ATM_CHECK_EQ(r.i32(), -123456);
  ATM_CHECK_EQ(r.f32(), 3.25f);
  ATM_CHECK_EQ(r.f64(), -1.0e300);
  ATM_CHECK_EQ(r.varu(), 0ull);
  ATM_CHECK_EQ(r.varu(), 127ull);
  ATM_CHECK_EQ(r.varu(), 128ull);
  ATM_CHECK_EQ(r.varu(), 0xFFFFFFFFFFFFFFFFull);
  ATM_CHECK_EQ(r.string(), std::string("hello"));
  const std::vector<uint8_t> b = r.bytes(16);
  ATM_REQUIRE(b.size() == 5);
  ATM_CHECK_EQ(int(b[4]), 5);
  ATM_CHECK_EQ(r.bits(32), 0xFFFFFFFFu);
  ATM_CHECK(r.ok());
}

ATM_TEST(net_bitstream_underflow_is_safe) {
  const uint8_t one[1] = {0xFF};
  net::BitReader r(std::span<const uint8_t>(one, 1));
  ATM_CHECK_EQ(r.bits(8), 0xFFu);
  ATM_CHECK(r.ok());
  ATM_CHECK_EQ(r.u32(), 0u); // past the end
  ATM_CHECK(!r.ok());
  ATM_CHECK_EQ(r.u64(), 0ull);
  ATM_CHECK(r.string().empty());
  ATM_CHECK(r.bytes(100).empty());
  ATM_CHECK(!r.ok());

  // Length prefixes larger than the data are rejected without allocating.
  std::vector<uint8_t> buf;
  net::BitWriter w(buf);
  w.varu(1u << 30);
  net::BitReader r2(buf);
  ATM_CHECK(r2.bytes(size_t(1) << 31).empty());
  ATM_CHECK(!r2.ok());

  // Unterminated varu.
  const uint8_t bad[12] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
  net::BitReader r3(std::span<const uint8_t>(bad, sizeof(bad)));
  (void)r3.varu();
  ATM_CHECK(!r3.ok());

  // Writer cap.
  std::vector<uint8_t> small;
  net::BitWriter w2(small, 4);
  w2.u32(1);
  ATM_CHECK(!w2.overflow());
  w2.u8(1);
  ATM_CHECK(w2.overflow());
  ATM_CHECK_EQ(small.size(), size_t(4));
}

// ---------------------------------------------------------------------------
// Codecs
// ---------------------------------------------------------------------------

template <class Codec, class T> bool roundTrip(const T &in, T &out) {
  std::vector<uint8_t> buf;
  net::BitWriter w(buf);
  Codec::encode(w, in);
  net::BitReader r(buf);
  return Codec::decode(r, out) && r.ok();
}

ATM_TEST(net_codec_scalars) {
  uint32_t u = 0;
  ATM_CHECK(roundTrip<schema::Bits<5>>(uint32_t(17), u));
  ATM_CHECK_EQ(u, 17u);
  uint64_t v = 0;
  ATM_CHECK(roundTrip<schema::VarU>(uint64_t(300000), v));
  ATM_CHECK_EQ(v, 300000ull);
  bool b = false;
  ATM_CHECK(roundTrip<schema::Bool>(true, b));
  ATM_CHECK(b);
  int32_t i = 0;
  ATM_CHECK(roundTrip<schema::I32>(int32_t(-7), i));
  ATM_CHECK_EQ(i, -7);
  float f = 0;
  ATM_CHECK(roundTrip<schema::F32>(1.5f, f));
  ATM_CHECK_EQ(f, 1.5f);

  float a = 0;
  ATM_CHECK(roundTrip<schema::Angle<12>>(2.0f, a));
  ATM_CHECK_NEAR(a, 2.0f, 0.002);
  ATM_CHECK(roundTrip<schema::Angle<12>>(2.0f + 6.2831853f, a)); // wraps
  ATM_CHECK_NEAR(a, 2.0f, 0.002);
  ATM_CHECK(roundTrip<schema::Angle<12>>(-3.0f, a));
  ATM_CHECK_NEAR(a, -3.0f, 0.002);

  float q = 0;
  ATM_CHECK((roundTrip<schema::QuantF<-1000, 1000, 10>>(0.5f, q)));
  ATM_CHECK_NEAR(q, 0.5f, 0.002);
  ATM_CHECK((roundTrip<schema::QuantF<-1000, 1000, 10>>(7.0f, q))); // clamps
  ATM_CHECK_NEAR(q, 1.0f, 1e-6);

  glm::dvec3 p{0.0};
  ATM_CHECK(roundTrip<schema::Pos>(glm::dvec3(-12345.5, 70.25, 99999.015625), p));
  ATM_CHECK_NEAR(p.x, -12345.5, 1.0 / 128.0);
  ATM_CHECK_NEAR(p.y, 70.25, 1.0 / 128.0);
  ATM_CHECK_NEAR(p.z, 99999.015625, 1.0 / 128.0);
  ATM_CHECK(roundTrip<schema::Pos>(glm::dvec3(0.0, -5.0, 0.0), p)); // y clamps at 0
  ATM_CHECK_EQ(p.y, 0.0);

  glm::vec3 vel{0.0f};
  ATM_CHECK(roundTrip<schema::Vel>(glm::vec3(-3.3f, 20.0f, 0.1f), vel));
  ATM_CHECK_NEAR(vel.x, -3.3f, 1.0 / 64.0);
  ATM_CHECK_NEAR(vel.y, 20.0f, 1.0 / 64.0);
  ATM_CHECK_NEAR(vel.z, 0.1f, 1.0 / 64.0);

  std::string s;
  ATM_CHECK(roundTrip<schema::Str<4>>(std::string("abcdefgh"), s)); // truncated on encode
  ATM_CHECK_EQ(s, std::string("abcd"));

  std::vector<uint8_t> blob = {9, 8, 7}, blobOut;
  ATM_CHECK(roundTrip<schema::Blob<16>>(blob, blobOut));
  ATM_CHECK(blobOut == blob);

  std::vector<uint64_t> list = {1, 2, 3}, listOut;
  ATM_CHECK((roundTrip<schema::List<schema::U64, 8>>(list, listOut)));
  ATM_CHECK(listOut == list);

  // A list count above MaxN is rejected.
  std::vector<uint8_t> buf;
  net::BitWriter w(buf);
  w.varu(9);
  for (int k = 0; k < 9; ++k) w.u64(uint64_t(k));
  net::BitReader r(buf);
  std::vector<uint64_t> tooMany;
  ATM_CHECK(!(schema::List<schema::U64, 8>::decode(r, tooMany)));
}

ATM_TEST(net_codec_custom) {
  MoveInput in;
  in.tick = 77;
  in.seq = 123456;
  in.moveX = -0.5f;
  in.moveZ = 1.0f;
  in.yaw = 1.25f;
  in.pitch = -0.3f;
  in.buttons = button::Jump | button::Sprint;
  MoveInput out;
  ATM_CHECK(roundTrip<MoveInputCodec>(in, out));
  ATM_CHECK_EQ(out.tick, 77u);
  ATM_CHECK_EQ(out.seq, 123456u);
  ATM_CHECK_EQ(out.moveX, -0.5f); // exact: prediction must match the server
  ATM_CHECK_EQ(out.moveZ, 1.0f);
  ATM_CHECK_EQ(out.yaw, 1.25f);
  ATM_CHECK_EQ(out.pitch, -0.3f);
  ATM_CHECK_EQ(out.buttons, in.buttons);

  atm::model::Appearance ap, apOut;
  ap.pieces[0] = 5;
  ap.pieces[6] = 300;
  ap.skinTone = 2;
  ap.dyes[3] = 9;
  ATM_CHECK(roundTrip<AppearanceCodec>(ap, apOut));
  ATM_CHECK_EQ(apOut.pieces[6], ap.pieces[6]);
  ATM_CHECK_EQ(int(apOut.skinTone), 2);
  ATM_CHECK_EQ(int(apOut.dyes[3]), 9);

  ItemStack st{items::Dirt, 64}, stOut;
  ATM_CHECK(roundTrip<ItemStackCodec>(st, stOut));
  ATM_CHECK_EQ(stOut.item, st.item);
  ATM_CHECK_EQ(stOut.count, st.count);

  Snapshot s;
  s.tick = 1000;
  s.ackInputSeq = 55;
  s.self.move.pos = glm::dvec3(1.234567, 80.5, -9.87654321);
  s.self.move.vel = glm::vec3(1, 2, 3);
  s.self.move.onGround = true;
  s.self.move.jumpsLeft = 1;
  s.self.hp = 90;
  s.self.maxHp = 100;
  EntityState e;
  e.id = 42;
  e.mask = field::All;
  e.kind = EntityKind::DroppedItem;
  e.type = 7;
  e.item = 300;
  e.pos = glm::dvec3(10.5, 64.0, -3.25);
  e.vel = glm::vec3(0.5f, 0, 0);
  e.yaw = 1.0f;
  e.actionAnim = action::Swing;
  e.actionSeq = 3;
  e.hp = 5;
  e.maxHp = 10;
  e.flags = eflag::OnGround;
  s.entities.push_back(e);
  EntityState partial;
  partial.id = 43;
  partial.mask = field::Pos;
  partial.pos = glm::dvec3(1, 2, 3);
  s.entities.push_back(partial);
  s.removed = {7, 8, 9};
  Snapshot so;
  ATM_REQUIRE(roundTrip<SnapshotCodec>(s, so));
  ATM_CHECK_EQ(so.tick, 1000u);
  ATM_CHECK_EQ(so.ackInputSeq, 55u);
  ATM_CHECK_EQ(so.self.move.pos.x, 1.234567); // full precision for reconciliation
  ATM_CHECK_EQ(so.self.move.pos.z, -9.87654321);
  ATM_CHECK(so.self.move.onGround);
  ATM_CHECK_EQ(so.self.hp, 90);
  ATM_REQUIRE(so.entities.size() == 2);
  ATM_CHECK_EQ(so.entities[0].id, 42u);
  ATM_CHECK(so.entities[0].kind == EntityKind::DroppedItem);
  ATM_CHECK_EQ(so.entities[0].item, 300);
  ATM_CHECK_NEAR(so.entities[0].pos.z, -3.25, 1e-9);
  ATM_CHECK_NEAR(so.entities[0].yaw, 1.0f, 0.005);
  ATM_CHECK_EQ(int(so.entities[0].actionAnim), int(action::Swing));
  ATM_CHECK_EQ(int(so.entities[0].flags), int(eflag::OnGround));
  ATM_CHECK_EQ(so.entities[1].mask, field::Pos);
  ATM_CHECK_NEAR(so.entities[1].pos.y, 2.0, 1e-9);
  ATM_CHECK(so.removed == s.removed);
}

// ---------------------------------------------------------------------------
// Schema (every Protocol message)
// ---------------------------------------------------------------------------

template <class M> bool schemaRoundTrip(const M &in, M &out, std::vector<uint8_t> &wire) {
  net::encodeMessage(wire, in);
  if (wire.empty()) return false;
  net::BitReader r(wire);
  uint16_t id = 0;
  if (!net::peekMessageId(wire, id, r) || id != M::kId) return false;
  return net::decodeMessage(r, out);
}

// Every strict prefix of a valid message must be rejected, and random bytes
// must never crash the decoder.
template <class M> void checkRejects(const std::vector<uint8_t> &wire) {
  for (size_t n = 0; n < wire.size(); ++n) {
    M m;
    const bool ok = net::decodeMessage(std::span<const uint8_t>(wire.data(), n), m);
    ATM_CHECK(!ok);
  }
  std::mt19937 rng(1234);
  std::vector<uint8_t> garbage;
  for (int iter = 0; iter < 200; ++iter) {
    garbage.resize(size_t(rng() % 64));
    for (auto &b : garbage) b = uint8_t(rng());
    M m;
    (void)net::decodeMessage(std::span<const uint8_t>(garbage.data(), garbage.size()), m);
    // Also with the right id in front.
    std::vector<uint8_t> withId;
    net::BitWriter w(withId);
    w.varu(M::kId);
    w.raw(garbage.data(), garbage.size());
    (void)net::decodeMessage(std::span<const uint8_t>(withId.data(), withId.size()), m);
  }
}

ATM_TEST(net_schema_messages) {
  std::vector<uint8_t> wire;
  {
    Hello a, b;
    a.schemaHash = kSchemaHash;
    a.name = "Alice";
    a.appearance.hairColor = 3;
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_EQ(b.schemaHash, kSchemaHash);
    ATM_CHECK_EQ(b.name, std::string("Alice"));
    ATM_CHECK_EQ(int(b.appearance.hairColor), 3);
    checkRejects<Hello>(wire);
  }
  {
    InputBatch a, b;
    for (uint32_t i = 0; i < 3; ++i) {
      MoveInput in;
      in.seq = 10 + i;
      in.moveZ = 1.0f;
      a.inputs.push_back(in);
    }
    a.lastSnapshotTick = 999;
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_REQUIRE(b.inputs.size() == 3);
    ATM_CHECK_EQ(b.inputs[2].seq, 12u);
    ATM_CHECK_EQ(b.lastSnapshotTick, 999u);
    checkRejects<InputBatch>(wire);
  }
  {
    BlockAction a, b;
    a.action = 1;
    a.x = -5;
    a.y = 70;
    a.z = 123456;
    a.face = 2;
    a.hotbarSlot = 8;
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_EQ(int(b.action), 1);
    ATM_CHECK_EQ(b.x, -5);
    ATM_CHECK_EQ(b.z, 123456);
    ATM_CHECK_EQ(int(b.face), 2);
    ATM_CHECK_EQ(int(b.hotbarSlot), 8);
    checkRejects<BlockAction>(wire);
  }
  {
    Attack a, b;
    a.tick = 5;
    a.yaw = -1.0f;
    a.pitch = 0.5f;
    a.ability = 4;
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_NEAR(b.yaw, -1.0f, 0.002);
    ATM_CHECK_NEAR(b.pitch, 0.5f, 0.002);
    ATM_CHECK_EQ(int(b.ability), 4);
    checkRejects<Attack>(wire);
  }
  {
    Equip a, b;
    a.inventorySlot = 27;
    a.equipSlot = 7;
    a.unequip = true;
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_EQ(int(b.inventorySlot), 27);
    ATM_CHECK(b.unequip);
    checkRejects<Equip>(wire);
  }
  {
    ChatSend a, b;
    a.text = "hello world";
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_EQ(b.text, a.text);
    checkRejects<ChatSend>(wire);
  }
  {
    Welcome a, b;
    a.schemaHash = 1;
    a.playerEntity = 77;
    a.serverTick = 88;
    a.worldSeed = 0xABCDEF;
    a.spawn = glm::dvec3(0.5, 70.0, -0.5);
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_EQ(b.playerEntity, 77u);
    ATM_CHECK_EQ(b.worldSeed, 0xABCDEFull);
    ATM_CHECK_NEAR(b.spawn.y, 70.0, 1e-9);
    checkRejects<Welcome>(wire);
  }
  {
    ChunkData a, b;
    a.cx = 1;
    a.cy = 2;
    a.cz = -3;
    a.data.assign(5000, 0x5A);
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_EQ(b.cz, -3);
    ATM_CHECK(b.data == a.data);
  }
  {
    EditedChunks a, b;
    a.keys = {atm::voxel::ChunkCoord{1, 2, 3}.key(), atm::voxel::ChunkCoord{-1, 0, -7}.key()};
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK(b.keys == a.keys);
    checkRejects<EditedChunks>(wire);
  }
  {
    BlockChanged a, b;
    a.x = 1;
    a.y = 2;
    a.z = 3;
    a.block = 17;
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_EQ(b.block, 17);
    checkRejects<BlockChanged>(wire);
  }
  {
    SnapshotMsg a, b;
    a.snapshot.tick = 3;
    EntityState e;
    e.id = 9;
    a.snapshot.entities.push_back(e);
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_EQ(b.snapshot.tick, 3u);
    ATM_CHECK_EQ(b.snapshot.entities.size(), size_t(1));
    checkRejects<SnapshotMsg>(wire);
  }
  {
    AppearanceMsg a, b;
    a.entity = 5;
    a.name = "Bob";
    a.appearance.pieces[1] = 12;
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_EQ(b.name, std::string("Bob"));
    ATM_CHECK_EQ(b.appearance.pieces[1], 12);
    checkRejects<AppearanceMsg>(wire);
  }
  {
    DamageEvent a, b;
    a.source = 1;
    a.target = 2;
    a.amount = 300;
    a.critical = true;
    a.killed = true;
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_EQ(b.amount, 300);
    ATM_CHECK(b.critical && b.killed);
    checkRejects<DamageEvent>(wire);
  }
  {
    InventoryMsg a, b;
    a.slots.resize(kInventorySlots);
    a.slots[3] = ItemStack{items::Planks, 64};
    a.equipped = {items::WoodenSword, 0, 0, 0, 0, 0, 0, 0};
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_REQUIRE(b.slots.size() == size_t(kInventorySlots));
    ATM_CHECK_EQ(b.slots[3].count, 64);
    ATM_CHECK(b.equipped == a.equipped);
    checkRejects<InventoryMsg>(wire);
  }
  {
    XpGain a, b;
    a.skill = 20;
    a.amount = 40;
    a.totalXp = 13034431;
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_EQ(int(b.skill), 20);
    ATM_CHECK_EQ(b.totalXp, 13034431ull);
    checkRejects<XpGain>(wire);
  }
  {
    LootMsg a, b;
    a.fromEntity = 4;
    a.item = items::RedCape;
    a.count = 1;
    a.rare = true;
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_EQ(b.item, items::RedCape);
    ATM_CHECK(b.rare);
    checkRejects<LootMsg>(wire);
  }
  {
    ChatMsg a, b;
    a.from = "Server";
    a.text = "hi";
    ATM_CHECK(schemaRoundTrip(a, b, wire));
    ATM_CHECK_EQ(b.from, std::string("Server"));
    checkRejects<ChatMsg>(wire);
  }
  // The schema hash is a compile-time constant and differs per message set.
  static_assert(kSchemaHash != 0, "schema hash");
  static_assert(net::schemaHash<Hello>() != net::schemaHash<Welcome>(), "per-message hashes differ");
  static_assert(kTransportProtocolId == uint32_t(kSchemaHash ^ (kSchemaHash >> 32)), "transport id");
}

// ---------------------------------------------------------------------------
// Transport (loopback)
// ---------------------------------------------------------------------------

struct Received {
  net::EventType type;
  net::PeerId peer;
  net::Channel channel;
  net::DisconnectReason reason;
  std::vector<uint8_t> data;
};

void drain(net::Host &h, std::vector<Received> &out) {
  net::Event ev;
  while (h.poll(ev))
    out.push_back(Received{ev.type, ev.peer, ev.channel, ev.reason,
                           std::vector<uint8_t>(ev.data.begin(), ev.data.end())});
}

// Updates the hosts (1 ms apart) until `done` or the timeout.
bool pumpUntil(std::vector<net::Host *> hosts, std::vector<std::vector<Received> *> sinks,
               const std::function<bool()> &done, uint64_t timeoutMs) {
  const uint64_t end = net::nowMs() + timeoutMs;
  while (net::nowMs() < end) {
    for (size_t i = 0; i < hosts.size(); ++i) {
      hosts[i]->update(net::nowMs());
      drain(*hosts[i], *sinks[i]);
    }
    if (done()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return done();
}

size_t countType(const std::vector<Received> &v, net::EventType t) {
  size_t n = 0;
  for (const auto &e : v) n += e.type == t ? 1 : 0;
  return n;
}

net::HostConfig testConfig() {
  net::HostConfig c;
  c.maxPeers = 4;
  c.protocolId = 0x1234ABCDu;
  c.timeoutMs = 1500;
  c.bandwidthBytesPerSec = 4u << 20; // fast loopback
  return c;
}

ATM_TEST(net_host_loopback) {
  net::Host server(testConfig());
  std::string err;
  ATM_REQUIRE(server.listen(0, &err));
  const uint16_t port = server.localPort();
  ATM_REQUIRE(port != 0);

  net::HostConfig cc = testConfig();
  cc.maxPeers = 1;
  net::Host client(cc);
  ATM_REQUIRE(client.connect("127.0.0.1", port, &err));

  std::vector<Received> se, ce;
  ATM_REQUIRE(pumpUntil({&server, &client}, {&se, &ce}, [&] {
    return countType(se, net::EventType::Connected) == 1 && client.serverPeer().valid();
  }, 3000));
  const net::PeerId clientOnServer = se[0].peer;
  const net::PeerId serverOnClient = client.serverPeer();
  ATM_CHECK(server.isConnected(clientOnServer));
  ATM_CHECK_EQ(server.peerCount(), size_t(1));

  // Reliable ordered: 300 small messages arrive once, in order.
  for (uint32_t i = 0; i < 300; ++i) {
    uint8_t msg[4];
    std::memcpy(msg, &i, 4);
    ATM_CHECK(client.send(serverOnClient, net::Channel::ReliableOrdered, std::span<const uint8_t>(msg, 4)));
  }
  // A 100 KB bulk message (fragmented) and a 10 KB reliable one.
  std::vector<uint8_t> big(100 * 1024);
  for (size_t i = 0; i < big.size(); ++i) big[i] = uint8_t(i * 31 + 7);
  ATM_CHECK(server.send(clientOnServer, net::Channel::Bulk, big));
  std::vector<uint8_t> mid(10 * 1024, 0xAB);
  ATM_CHECK(server.send(clientOnServer, net::Channel::ReliableOrdered, mid));
  // Unreliable: small ones go, oversized ones are refused.
  const uint8_t u[3] = {1, 2, 3};
  for (int i = 0; i < 10; ++i)
    ATM_CHECK(server.send(clientOnServer, net::Channel::Unreliable, std::span<const uint8_t>(u, 3)));
  std::vector<uint8_t> tooBig(4000, 1);
  ATM_CHECK(!server.send(clientOnServer, net::Channel::Unreliable, tooBig));

  auto countOn = [](const std::vector<Received> &v, net::Channel ch) {
    size_t n = 0;
    for (const auto &e : v) n += (e.type == net::EventType::Message && e.channel == ch) ? 1 : 0;
    return n;
  };
  ATM_REQUIRE(pumpUntil({&server, &client}, {&se, &ce}, [&] {
    return countOn(se, net::Channel::ReliableOrdered) == 300 && countOn(ce, net::Channel::Bulk) == 1 &&
           countOn(ce, net::Channel::ReliableOrdered) == 1;
  }, 5000));

  uint32_t expect = 0;
  for (const auto &e : se) {
    if (e.type != net::EventType::Message || e.channel != net::Channel::ReliableOrdered) continue;
    ATM_REQUIRE(e.data.size() == 4);
    uint32_t v = 0;
    std::memcpy(&v, e.data.data(), 4);
    ATM_CHECK_EQ(v, expect);
    ++expect;
  }
  for (const auto &e : ce) {
    if (e.type != net::EventType::Message) continue;
    if (e.channel == net::Channel::Bulk) ATM_CHECK(e.data == big);
    if (e.channel == net::Channel::ReliableOrdered) ATM_CHECK(e.data == mid);
  }
  ATM_CHECK(countOn(ce, net::Channel::Unreliable) >= 1); // loopback: normally all 10

  // Acks drain the send queues.
  ATM_CHECK(pumpUntil({&server, &client}, {&se, &ce}, [&] {
    const net::PeerStats s = server.stats(clientOnServer);
    return s.inFlight == 0 && s.bulkQueuedBytes == 0 && s.reliableQueued == 0;
  }, 3000));
  ATM_CHECK(server.stats(clientOnServer).rttMs >= 0.0f);

  // Clean disconnect reaches the server.
  client.disconnect(serverOnClient);
  ATM_CHECK(!client.serverPeer().valid());
  ATM_CHECK(pumpUntil({&server, &client}, {&se, &ce}, [&] {
    return countType(se, net::EventType::Disconnected) == 1;
  }, 2000));
  ATM_CHECK_EQ(server.peerCount(), size_t(0));
  ATM_CHECK(!server.isConnected(clientOnServer));
  ATM_CHECK(!server.send(clientOnServer, net::Channel::ReliableOrdered, mid)); // stale id
}

ATM_TEST(net_host_timeout_and_generation) {
  net::HostConfig sc = testConfig();
  sc.maxPeers = 1; // forces slot reuse
  net::Host server(sc);
  std::string err;
  ATM_REQUIRE(server.listen(0, &err));
  const uint16_t port = server.localPort();

  net::HostConfig cc = testConfig();
  cc.maxPeers = 1;
  std::vector<Received> se, ce1, ce2;
  net::PeerId first;
  {
    net::Host c1(cc);
    ATM_REQUIRE(c1.connect("127.0.0.1", port, &err));
    ATM_REQUIRE(pumpUntil({&server, &c1}, {&se, &ce1}, [&] {
      return countType(se, net::EventType::Connected) == 1 && c1.serverPeer().valid();
    }, 3000));
    first = se[0].peer;
    // Stop updating c1: the server must time it out.
    ATM_REQUIRE(pumpUntil({&server}, {&se}, [&] { return countType(se, net::EventType::Disconnected) == 1; },
                          4000));
    const auto &d = se.back();
    ATM_CHECK(d.reason == net::DisconnectReason::Timeout);
    ATM_CHECK(d.peer == first);
  } // c1 destroyed (its disconnect packets hit a dead connection id: ignored)

  se.clear();
  net::Host c2(cc);
  ATM_REQUIRE(c2.connect("127.0.0.1", port, &err));
  ATM_REQUIRE(pumpUntil({&server, &c2}, {&se, &ce2}, [&] {
    for (const auto &e : se)
      if (e.type == net::EventType::Connected) return true;
    return false;
  }, 3000));
  net::PeerId second;
  for (const auto &e : se)
    if (e.type == net::EventType::Connected) second = e.peer;
  ATM_CHECK_EQ(second.index(), first.index()); // same slot...
  ATM_CHECK(second.generation() != first.generation()); // ...new generation
  ATM_CHECK(!server.isConnected(first));
  ATM_CHECK(server.isConnected(second));
  const uint8_t m[1] = {1};
  ATM_CHECK(!server.send(first, net::Channel::ReliableOrdered, std::span<const uint8_t>(m, 1)));
  ATM_CHECK(server.send(second, net::Channel::ReliableOrdered, std::span<const uint8_t>(m, 1)));
}

ATM_TEST(net_host_protocol_mismatch) {
  net::Host server(testConfig());
  std::string err;
  ATM_REQUIRE(server.listen(0, &err));
  net::HostConfig cc = testConfig();
  cc.maxPeers = 1;
  cc.protocolId = 0x0BADF00Du;
  net::Host client(cc);
  ATM_REQUIRE(client.connect("127.0.0.1", server.localPort(), &err));
  std::vector<Received> se, ce;
  ATM_REQUIRE(pumpUntil({&server, &client}, {&se, &ce}, [&] {
    return countType(ce, net::EventType::Disconnected) == 1;
  }, 3000));
  ATM_CHECK(ce.back().reason == net::DisconnectReason::ProtocolMismatch);
  ATM_CHECK_EQ(server.peerCount(), size_t(0));
  ATM_CHECK_EQ(countType(se, net::EventType::Connected), size_t(0));
}

ATM_TEST(net_host_malformed_datagrams) {
  net::Host server(testConfig());
  std::string err;
  ATM_REQUIRE(server.listen(0, &err));
  net::detail::UdpSocket raw;
  ATM_REQUIRE(raw.open(0, &err));
  net::detail::Address to;
  ATM_REQUIRE(net::detail::UdpSocket::resolve("127.0.0.1", server.localPort(), to));

  std::mt19937 rng(42);
  std::vector<uint8_t> pkt;
  std::vector<Received> se;
  for (int i = 0; i < 3000; ++i) {
    pkt.resize(1 + rng() % 200);
    for (auto &b : pkt) b = uint8_t(rng());
    // Bias towards real packet types (handshake, data, disconnect) with bad contents.
    if (i % 2 == 0) pkt[0] = uint8_t(1 + rng() % 7);
    if (i % 7 == 0 && pkt.size() > 5) { // data packet aimed at slot 0 with a guessed id
      pkt[0] = 6;
      pkt[1] = 0;
      pkt[2] = 0;
    }
    raw.send(to, pkt.data(), int(pkt.size()));
    if (i % 100 == 0) {
      server.update(net::nowMs());
      drain(server, se);
    }
  }
  for (int i = 0; i < 20; ++i) {
    server.update(net::nowMs());
    drain(server, se);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ATM_CHECK_EQ(server.peerCount(), size_t(0)); // no state without a valid cookie
  ATM_CHECK_EQ(countType(se, net::EventType::Connected), size_t(0));
  ATM_CHECK(server.hostStats().malformedPackets > 0);

  // The server still accepts a real client afterwards.
  net::HostConfig cc = testConfig();
  cc.maxPeers = 1;
  net::Host client(cc);
  ATM_REQUIRE(client.connect("127.0.0.1", server.localPort(), &err));
  std::vector<Received> ce;
  ATM_CHECK(pumpUntil({&server, &client}, {&se, &ce}, [&] { return client.serverPeer().valid(); }, 3000));
}

} // namespace
