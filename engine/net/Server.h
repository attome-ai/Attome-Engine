#pragma once

#include "ConnectionSlots.h"
#include "MessageBatcher.h"
#include "MessageReader.h"
#include "NetSocket.h"
#include "PacketPool.h"
#include "SpscQueue.h"

#if NET_ENCRYPTION
#include "CryptoLayer.h"
#endif

#if NET_SIM
#include "SimLayer.h"
#endif

#include <atomic>
#include <bit>
#include <cstdint>
#include <functional>
#include <span>
#include <thread>

namespace attome::net {

class Server {
public:
  using Handler = std::function<void(ConnId, MessageReader &)>;

  Server(std::span<const OpcodeInfo> c2s_table,
         std::span<const OpcodeInfo> s2c_table);
  ~Server();

  bool bind(uint16_t port);
  uint16_t bound_port() const { return bound_port_; }
  void stop();

  void on(uint16_t opcode, Handler handler);

  template <typename Enum> void on(Enum opcode, Handler handler) {
    on(static_cast<uint16_t>(opcode), std::move(handler));
  }

  void send(ConnId conn, uint8_t ch, uint16_t opcode,
            std::span<const uint8_t> payload);

  template <typename Enum>
  void send(ConnId conn, uint8_t ch, Enum opcode,
            std::span<const uint8_t> payload) {
    send(conn, ch, static_cast<uint16_t>(opcode), payload);
  }

  void broadcast(uint8_t ch, uint16_t opcode, std::span<const uint8_t> payload);

  template <typename Enum>
  void broadcast(uint8_t ch, Enum opcode, std::span<const uint8_t> payload) {
    broadcast(ch, static_cast<uint16_t>(opcode), payload);
  }

  // Called by the game thread each frame.
  void tick(uint32_t delta_ms);

private:
  struct EndpointMap {
    void init();
    ConnId find(const NetEndpoint &ep) const;
    bool insert(const NetEndpoint &ep, ConnId conn);
    void erase(const NetEndpoint &ep);

  private:
    static constexpr uint32_t kCap =
        std::bit_ceil(static_cast<uint32_t>(NET_MAX_CONNECTIONS) * 4u);
    static constexpr uint32_t kMask = kCap - 1;
    static_assert((kCap & (kCap - 1)) == 0, "EndpointMap cap must be pow2.");

    // state: 0 empty, 1 occupied, 2 tombstone
    NetEndpoint keys_[kCap]{};
    ConnId vals_[kCap]{};
    uint8_t state_[kCap]{};

    static uint32_t hash_(const NetEndpoint &ep);
    static bool eq_(const NetEndpoint &a, const NetEndpoint &b);
  };

  asio::awaitable<void> recv_loop_();
  asio::awaitable<void> send_flush_loop_();

  void handle_datagram_(PoolIdx buf_idx, uint16_t len, const NetEndpoint &ep,
                        uint32_t now_ms);
  void free_conn_(ConnId conn);
  void send_welcome_(ConnId conn, const NetEndpoint &ep, uint32_t now_ms);

  static uint32_t now_ms_();
  static uint16_t read_u16_le_(const uint8_t *p);
  static uint32_t read_u32_le_(const uint8_t *p);
  static void write_u16_le_(uint8_t *dst, uint16_t v);
  static void write_u32_le_(uint8_t *dst, uint32_t v);

  static bool update_ack_window_(uint16_t &ack, uint32_t &ack_bits,
                                 uint16_t seq);

  std::span<const OpcodeInfo> c2s_table_;
  std::span<const OpcodeInfo> s2c_table_;

  PacketPool pool_{};
  ConnectionSlots slots_{};
  MessageBatcher batcher_{};

  asio::io_context ioc_{1};
  NetSocket socket_{ioc_};
  uint16_t bound_port_{0};
  std::thread net_thread_{};
  std::atomic<bool> running_{false};

  SpscQueue<RawRecv, 8192> raw_recv_{};
  SpscQueue<InboundMsg, 8192> net_to_game_{};
  SpscQueue<OutboundMsg, 8192> game_to_net_{};

  static constexpr uint32_t kSendListCap = 65536;
  SendItem send_list_[kSendListCap]{};

  Handler c2s_handlers_[65536]{};
  EndpointMap endpoint_map_{};

  // Per-connection ACK resend hints (reliable channels only). Set on recv when we
  // see reliable traffic; cleared when we send a fresh packet on that channel.
  uint8_t ack_pending_[NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]{};

#if NET_SIM
  SimLayer sim_{};
#endif

#if NET_ENCRYPTION
  CryptoLayer crypto_{};
#endif
};

} // namespace attome::net
