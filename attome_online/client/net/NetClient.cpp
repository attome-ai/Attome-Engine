#include "NetClient.h"

#include <cstdio>

namespace ao::client {

using atm::net2::Channel;
using atm::net2::EventType;

NetClient::NetClient() { scratch_.reserve(2048); }

NetClient::~NetClient() { disconnect(); }

bool NetClient::connect(const std::string &host, uint16_t port, std::string *error) {
  atm::net2::HostConfig cfg;
  cfg.maxPeers = 1;
  cfg.protocolId = proto::kTransportProtocolId;
  cfg.bandwidthBytesPerSec = 64u * 1024; // client upstream is small
  host_ = std::make_unique<atm::net2::Host>(cfg);
  if (!host_->connect(host, port, error)) {
    host_.reset();
    return false;
  }
  return true;
}

void NetClient::disconnect() {
  if (host_ && server_.valid()) {
    host_->disconnect(server_);
    // Give the disconnect a chance to leave before the socket closes.
    for (int i = 0; i < 3; ++i)
      host_->update(atm::net2::nowMs());
  }
  host_.reset();
  connected_ = false;
  server_ = {};
}

atm::net2::PeerStats NetClient::stats() const {
  if (!host_ || !server_.valid())
    return {};
  return host_->stats(server_);
}

uint64_t NetClient::bytesOut() const { return stats().bytesSent; }

void NetClient::update(NetHandler &handler) {
  if (!host_)
    return;
  host_->update(atm::net2::nowMs());

  atm::net2::Event ev;
  // A handler may call disconnect() (e.g. schema mismatch in onWelcome),
  // which destroys host_: re-check it before every poll.
  while (host_ && host_->poll(ev)) {
    switch (ev.type) {
    case EventType::Connected:
      server_ = ev.peer;
      connected_ = true;
      handler.onConnected();
      break;
    case EventType::Disconnected:
      connected_ = false;
      server_ = {};
      handler.onDisconnected(ev.reason);
      break;
    case EventType::Message:
      bytesIn_ += ev.data.size();
      dispatch(ev.data, handler);
      break;
    }
  }
}

namespace {

template <class M, class Fn>
void decodeAndCall(atm::net2::BitReader &r, Fn &&fn) {
  M msg{};
  if (decode(r, msg) && r.ok())
    fn(msg);
  else
    std::fprintf(stderr, "[net] dropped malformed message id %u\n", unsigned(M::kId));
}

} // namespace

void NetClient::dispatch(std::span<const uint8_t> data, NetHandler &h) {
  atm::net2::BitReader r(data);
  const uint64_t id = r.varu();
  if (!r.ok())
    return;

  using namespace proto;
  switch (id) {
  case Welcome::kId: decodeAndCall<Welcome>(r, [&](auto &m) { h.onWelcome(m); }); break;
  case ChunkData::kId: decodeAndCall<ChunkData>(r, [&](auto &m) { h.onChunkData(m); }); break;
  case EditedChunks::kId: decodeAndCall<EditedChunks>(r, [&](auto &m) { h.onEditedChunks(m); }); break;
  case BlockChanged::kId: decodeAndCall<BlockChanged>(r, [&](auto &m) { h.onBlockChanged(m); }); break;
  case SnapshotMsg::kId: decodeAndCall<SnapshotMsg>(r, [&](auto &m) { h.onSnapshot(m); }); break;
  case AppearanceMsg::kId: decodeAndCall<AppearanceMsg>(r, [&](auto &m) { h.onAppearance(m); }); break;
  case DamageEvent::kId: decodeAndCall<DamageEvent>(r, [&](auto &m) { h.onDamage(m); }); break;
  case InventoryMsg::kId: decodeAndCall<InventoryMsg>(r, [&](auto &m) { h.onInventory(m); }); break;
  case XpGain::kId: decodeAndCall<XpGain>(r, [&](auto &m) { h.onXpGain(m); }); break;
  case LootMsg::kId: decodeAndCall<LootMsg>(r, [&](auto &m) { h.onLoot(m); }); break;
  case ChatMsg::kId: decodeAndCall<ChatMsg>(r, [&](auto &m) { h.onChat(m); }); break;
  default:
    std::fprintf(stderr, "[net] unknown message id %llu\n", static_cast<unsigned long long>(id));
    break;
  }
}

} // namespace ao::client
