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
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <thread>

namespace attome::net {

class Client {
public:
  using Handler = std::function<void(MessageReader &)>;

  Client(std::span<const OpcodeInfo> c2s_table,
         std::span<const OpcodeInfo> s2c_table);
  ~Client();

  bool connect(const char *host, uint16_t port, uint32_t timeout_ms = 3000);
  void disconnect();

  void on(uint16_t opcode, Handler handler);

  template <typename Enum> void on(Enum opcode, Handler handler) {
    on(static_cast<uint16_t>(opcode), std::move(handler));
  }

  void send(uint8_t ch, uint16_t opcode, std::span<const uint8_t> payload);

  template <typename Enum>
  void send(uint8_t ch, Enum opcode, std::span<const uint8_t> payload) {
    send(ch, static_cast<uint16_t>(opcode), payload);
  }

  void tick(uint32_t delta_ms);

private:
  asio::awaitable<void> recv_loop_();
  asio::awaitable<void> send_flush_loop_();

  void handle_datagram_(PoolIdx buf_idx, uint16_t len, const NetEndpoint &ep,
                        uint32_t now_ms);
  void send_connect_(uint32_t now_ms);

  static uint32_t now_ms_();
  static uint16_t read_u16_le_(const uint8_t *p);
  static uint32_t read_u32_le_(const uint8_t *p);
  static void write_u16_le_(uint8_t *dst, uint16_t v);
  static void write_u32_le_(uint8_t *dst, uint32_t v);
  static bool update_ack_window_(uint16_t &ack, uint32_t &ack_bits,
                                 uint16_t seq);

  bool endpoint_eq_(const NetEndpoint &a, const NetEndpoint &b) const;

  std::span<const OpcodeInfo> c2s_table_;
  std::span<const OpcodeInfo> s2c_table_;

  PacketPool pool_{};
  ConnectionSlots slots_{};
  MessageBatcher batcher_{};

  asio::io_context ioc_{1};
  NetSocket socket_{ioc_};
  std::thread net_thread_{};
  std::atomic<bool> running_{false};

  SpscQueue<RawRecv, 8192> raw_recv_{};
  SpscQueue<InboundMsg, 8192> net_to_game_{};
  SpscQueue<OutboundMsg, 8192> game_to_net_{};

  static constexpr uint32_t kSendListCap = 65536;
  SendItem send_list_[kSendListCap]{};

  Handler s2c_handlers_[65536]{};

  // ACK resend hints (reliable channels only). Set on recv; cleared when we send
  // a fresh packet on that channel.
  uint8_t ack_pending_[NET_CHANNEL_COUNT]{};

  ConnId conn_{kInvalidConnId};
  NetEndpoint server_ep_{};
  uint32_t last_connect_sent_ms_{0};

#if NET_ENCRYPTION && NET_ENCRYPTION_KEY_MODE == 1
  uint8_t client_pk_[32]{};
  uint8_t client_sk_[32]{};
#endif

#if NET_SIM
  SimLayer sim_{};
#endif

#if NET_ENCRYPTION
  CryptoLayer crypto_{};
#endif

  std::mutex connect_mu_{};
  std::condition_variable connect_cv_{};
  std::atomic<bool> connected_{false};
};

} // namespace attome::net
