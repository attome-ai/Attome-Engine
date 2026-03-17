#include "Client.h"

#include "ReorderBuffer.h"
#include "ReliabilityLayer.h"

#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/ip/udp.hpp>
#include <asio/use_awaitable.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

namespace attome::net {

static constexpr uint16_t kC2S_CONNECT = 0x0001;
static constexpr uint16_t kC2S_DISCONNECT = 0x0002;
static constexpr uint16_t kS2C_WELCOME = 0x0001;

Client::Client(std::span<const OpcodeInfo> c2s_table,
               std::span<const OpcodeInfo> s2c_table)
    : c2s_table_(c2s_table), s2c_table_(s2c_table) {}

Client::~Client() { disconnect(); }

uint32_t Client::now_ms_() {
  using namespace std::chrono;
  return static_cast<uint32_t>(
      duration_cast<milliseconds>(steady_clock::now().time_since_epoch())
          .count());
}

uint16_t Client::read_u16_le_(const uint8_t *p) {
  return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) |
                               (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t Client::read_u32_le_(const uint8_t *p) {
  return (static_cast<uint32_t>(p[0]) |
          (static_cast<uint32_t>(p[1]) << 8) |
          (static_cast<uint32_t>(p[2]) << 16) |
          (static_cast<uint32_t>(p[3]) << 24));
}

void Client::write_u16_le_(uint8_t *dst, uint16_t v) {
  dst[0] = static_cast<uint8_t>(v & 0xFFu);
  dst[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
}

void Client::write_u32_le_(uint8_t *dst, uint32_t v) {
  dst[0] = static_cast<uint8_t>(v & 0xFFu);
  dst[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
  dst[2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
  dst[3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
}

bool Client::update_ack_window_(uint16_t &ack, uint32_t &ack_bits,
                                uint16_t seq) {
  // 0xFFFF is used as "none received yet" initial state.
  if (ack == 0xFFFFu) {
    ack = seq;
    ack_bits = 0;
    return true;
  }

  if (reliability::seq_acked(seq, ack, ack_bits)) {
    return false;
  }

  const uint16_t forward = static_cast<uint16_t>(seq - ack);
  if (forward < 32768) {
    if (forward > NET_ACK_BITS_COUNT) {
      ack_bits = 0;
    } else {
      const uint64_t shifted = (static_cast<uint64_t>(ack_bits) << forward);
      ack_bits =
          static_cast<uint32_t>(shifted) | (1u << (forward - 1));
    }
    ack = seq;
    return true;
  }

  const uint16_t back = static_cast<uint16_t>(ack - seq);
  if (back == 0 || back > NET_ACK_BITS_COUNT) {
    return true;
  }
  ack_bits |= (1u << (back - 1));
  return true;
}

bool Client::endpoint_eq_(const NetEndpoint &a, const NetEndpoint &b) const {
  if (a.port != b.port) {
    return false;
  }
  if (a.is_v6 != b.is_v6) {
    return false;
  }
  if (a.is_v6) {
    if (a.v6_scope_id != b.v6_scope_id) {
      return false;
    }
    return std::memcmp(a.addr.v6, b.addr.v6, 16) == 0;
  }
  return a.addr.v4 == b.addr.v4;
}

void Client::on(uint16_t opcode, Handler handler) {
  s2c_handlers_[opcode] = std::move(handler);
}

bool Client::connect(const char *host, uint16_t port, uint32_t timeout_ms) {
  if (host == nullptr) {
    return false;
  }
  if (running_.exchange(true)) {
    return false;
  }

  pool_.init();
  slots_.init();
  batcher_.init();
  std::memset(ack_pending_, 0, sizeof(ack_pending_));
  ioc_.restart();

#if NET_SIM
  sim_.init();
#endif

#if NET_ENCRYPTION
  if (!crypto_.init()) {
    running_.store(false);
    return false;
  }
#if NET_ENCRYPTION_KEY_MODE == 1
  if (!crypto_.generate_keypair(client_pk_, client_sk_)) {
    running_.store(false);
    return false;
  }
#endif
#endif

  // Resolve endpoint (sync).
  asio::ip::udp::resolver resolver(ioc_);
  asio::error_code ec;
  auto results = resolver.resolve(asio::ip::udp::v4(), host,
                                  std::to_string(port), ec);
  if (ec || results.empty()) {
    std::fprintf(stderr, "Client::connect resolve failed: %s\n",
                 ec.message().c_str());
    running_.store(false);
    return false;
  }

  server_ep_ = NetSocket::to_net_endpoint(results.begin()->endpoint());

  if (!socket_.bind(0)) {
    running_.store(false);
    return false;
  }

  conn_ = slots_.alloc_slot(server_ep_, now_ms_());
  if (conn_ == kInvalidConnId) {
    running_.store(false);
    return false;
  }

#if NET_ENCRYPTION
  slots_.encryption_ready[conn_] = 0;
  crypto_.set_psk(slots_, conn_);
#endif

  connected_.store(false, std::memory_order_release);
  last_connect_sent_ms_ = 0;

  asio::co_spawn(ioc_, socket_.recv_loop(pool_, raw_recv_), asio::detached);
  asio::co_spawn(ioc_, recv_loop_(), asio::detached);
  asio::co_spawn(ioc_, send_flush_loop_(), asio::detached);

  net_thread_ = std::thread([this]() { ioc_.run(); });

  asio::post(ioc_, [this]() { send_connect_(now_ms_()); });

  // Wait for WELCOME.
  std::unique_lock<std::mutex> lk(connect_mu_);
  const bool ok = connect_cv_.wait_for(
      lk, std::chrono::milliseconds(timeout_ms),
      [this]() { return connected_.load(std::memory_order_acquire); });
  if (!ok) {
    lk.unlock();
    disconnect();
    return false;
  }
  return true;
}

void Client::disconnect() {
  if (!running_.exchange(false)) {
    return;
  }

  // Best-effort plaintext DISCONNECT (may be dropped).
  asio::post(ioc_, [this]() {
    const uint32_t now_ms = now_ms_();
    uint8_t buf[NET_MTU]{};

    auto *hdr = reinterpret_cast<PacketHeader *>(&buf[0]);
    hdr->channel = NET_CHANNEL_UNRELIABLE;
    hdr->seq = reliability::assign_seq(slots_, conn_, NET_CHANNEL_UNRELIABLE);
    hdr->ack = slots_.recv_ack[conn_][NET_CHANNEL_UNRELIABLE];
    hdr->ack_bits = slots_.recv_ack_bits[conn_][NET_CHANNEL_UNRELIABLE];

    uint8_t *w = buf + NET_HEADER_SIZE;
    write_u16_le_(w, kC2S_DISCONNECT);
    w += NET_OPCODE_SIZE;

    const uint16_t total = static_cast<uint16_t>(w - buf);
    socket_.send_one(buf, total, server_ep_);
    (void)now_ms;
  });

  asio::post(ioc_, [this]() {
    socket_.close();
    ioc_.stop();
  });

  if (net_thread_.joinable()) {
    net_thread_.join();
  }

  if (conn_ != kInvalidConnId) {
#if NET_SIM
    sim_.purge_endpoint(pool_, server_ep_);
#endif
    batcher_.reset_conn(pool_, conn_);
    reliability::clear_all(slots_, pool_, conn_);
    reorder::clear_conn(slots_, pool_, conn_);
    slots_.free_slot(conn_);
    conn_ = kInvalidConnId;
  }

  std::memset(ack_pending_, 0, sizeof(ack_pending_));
}

void Client::send(uint8_t ch, uint16_t opcode, std::span<const uint8_t> payload) {
  if (conn_ == kInvalidConnId || !slots_.is_alive(conn_)) {
    return;
  }
  if (ch >= NET_CHANNEL_COUNT) {
    return;
  }

  if (payload.size() > sizeof(OutboundMsg::staging)) {
    std::fprintf(stderr, "Client::send payload too large (%zu)\n",
                 payload.size());
    return;
  }

  OutboundMsg msg{};
  msg.conn = conn_;
  msg.channel = ch;
  msg.opcode = opcode;
  msg.len = static_cast<uint16_t>(payload.size());
  if (!payload.empty()) {
    std::memcpy(msg.staging, payload.data(), payload.size());
  }

  if (!game_to_net_.push(msg)) {
    if (ch == NET_CHANNEL_UNRELIABLE) {
      std::fprintf(stderr, "Client::send outbound queue full (drop)\n");
      return;
    }

    // Reliable channels must not drop due to local queue pressure; apply
    // backpressure instead.
    while (!game_to_net_.push(msg)) {
      if (conn_ == kInvalidConnId || !slots_.is_alive(conn_) ||
          !running_.load(std::memory_order_acquire)) {
        return;
      }
      std::this_thread::yield();
    }
  }
}

void Client::send_connect_(uint32_t now_ms) {
  if (conn_ == kInvalidConnId || !slots_.is_alive(conn_)) {
    return;
  }

  uint8_t buf[NET_MTU]{};
  auto *hdr = reinterpret_cast<PacketHeader *>(&buf[0]);
  hdr->channel = NET_CHANNEL_UNRELIABLE;
  hdr->seq = reliability::assign_seq(slots_, conn_, NET_CHANNEL_UNRELIABLE);
  hdr->ack = slots_.recv_ack[conn_][NET_CHANNEL_UNRELIABLE];
  hdr->ack_bits = slots_.recv_ack_bits[conn_][NET_CHANNEL_UNRELIABLE];

  uint8_t *w = buf + NET_HEADER_SIZE;
  write_u16_le_(w, kC2S_CONNECT);
  w += NET_OPCODE_SIZE;

  uint8_t payload[64]{};
  uint16_t plen = 0;

#if NET_ENCRYPTION && NET_ENCRYPTION_KEY_MODE == 1
  std::memcpy(payload, client_pk_, 32);
  plen = 32;
#endif

  write_u16_le_(w, plen);
  w += NET_DYNLEN_SIZE;
  if (plen > 0) {
    std::memcpy(w, payload, plen);
    w += plen;
  }

  const uint16_t total = static_cast<uint16_t>(w - buf);
  socket_.send_one(buf, total, server_ep_);
  last_connect_sent_ms_ = now_ms;
}

asio::awaitable<void> Client::recv_loop_() {
  asio::steady_timer idle(ioc_);

  while (running_.load(std::memory_order_acquire)) {
    RawRecv r{};
    if (!raw_recv_.pop(r)) {
      idle.expires_after(std::chrono::milliseconds(1));
      co_await idle.async_wait(asio::use_awaitable);
      continue;
    }

    const uint32_t now_ms = now_ms_();

#if NET_SIM
    if (sim_.enqueue_recv(pool_, r.buf_idx, r.len, r.endpoint, now_ms)) {
      continue;
    }
#endif

    handle_datagram_(r.buf_idx, r.len, r.endpoint, now_ms);
  }
}

void Client::handle_datagram_(PoolIdx buf_idx, uint16_t len,
                              const NetEndpoint &ep, uint32_t now_ms) {
  if (buf_idx == kInvalidPoolIdx) {
    return;
  }
  if (conn_ == kInvalidConnId || !slots_.is_alive(conn_)) {
    pool_.release(buf_idx);
    return;
  }
  if (!endpoint_eq_(ep, server_ep_)) {
    pool_.release(buf_idx);
    return;
  }
  if (len < NET_HEADER_SIZE) {
    pool_.release(buf_idx);
    return;
  }

  uint8_t *buf = pool_.ptr(buf_idx);
  auto *hdr = reinterpret_cast<PacketHeader *>(buf);
  if (hdr->channel >= NET_CHANNEL_COUNT) {
    pool_.release(buf_idx);
    return;
  }

  uint8_t *payload = buf + NET_HEADER_SIZE;
  uint16_t payload_len = static_cast<uint16_t>(len - NET_HEADER_SIZE);

#if NET_ENCRYPTION
  if (slots_.encryption_ready[conn_] != 0) {
    const uint16_t rx_len = len;
    if (rx_len < (NET_HEADER_SIZE + NET_ENCRYPTION_TAG_SIZE)) {
      pool_.release(buf_idx);
      return;
    }
    if (!crypto_.decrypt_in_place(buf, len, conn_, slots_)) {
      static std::atomic<uint32_t> s_decrypt_fail_logs{0};
      const uint32_t n =
          s_decrypt_fail_logs.fetch_add(1, std::memory_order_relaxed);
      if (n < 8) {
        std::fprintf(stderr,
                     "Client decrypt failed conn_id=%u ch=%u seq=%u rx_len=%u\n",
                     static_cast<unsigned>(slots_.encryption_conn_id[conn_]),
                     static_cast<unsigned>(hdr->channel),
                     static_cast<unsigned>(hdr->seq),
                     static_cast<unsigned>(rx_len));
      }
      pool_.release(buf_idx);
      return;
    }
    payload_len = static_cast<uint16_t>(len - NET_HEADER_SIZE);
    reliability::process_ack(slots_, pool_, conn_, hdr->channel, hdr->ack,
                             hdr->ack_bits);
  }
#else
  reliability::process_ack(slots_, pool_, conn_, hdr->channel, hdr->ack,
                           hdr->ack_bits);
#endif

  if (payload_len < NET_OPCODE_SIZE) {
    pool_.release(buf_idx);
    return;
  }

  const uint16_t first_opcode = read_u16_le_(payload);

  if (!connected_.load(std::memory_order_acquire) &&
      first_opcode != kS2C_WELCOME) {
    pool_.release(buf_idx);
    return;
  }

  if (first_opcode == kS2C_WELCOME) {
    // Welcome is dynamic: [opcode][len][payload]
    if (payload_len < (NET_OPCODE_SIZE + NET_DYNLEN_SIZE + 4)) {
      pool_.release(buf_idx);
      return;
    }
    const uint16_t dyn = read_u16_le_(payload + NET_OPCODE_SIZE);
    if (dyn < 4 || payload_len < (NET_OPCODE_SIZE + NET_DYNLEN_SIZE + dyn)) {
      pool_.release(buf_idx);
      return;
    }

    const uint8_t *p = payload + NET_OPCODE_SIZE + NET_DYNLEN_SIZE;
    const uint32_t conn_id = read_u32_le_(p);

#if NET_ENCRYPTION
    slots_.encryption_conn_id[conn_] = conn_id;

#if NET_ENCRYPTION_KEY_MODE == 1
    if (dyn < (4 + 32)) {
      pool_.release(buf_idx);
      return;
    }
    const uint8_t *server_pk = p + 4;
    if (!crypto_.client_derive_key(slots_.encryption_key[conn_], client_sk_,
                                   server_pk, conn_id)) {
      pool_.release(buf_idx);
      return;
    }
#endif

    slots_.encryption_ready[conn_] = 1;
#else
    (void)conn_id;
#endif

    connected_.store(true, std::memory_order_release);
    connect_cv_.notify_all();
  }

  // Duplicate check: CH0 always delivered; reliable channels drop duplicates.
  const uint8_t ch = hdr->channel;
  if (ch == NET_CHANNEL_RELIABLE_UNORD || ch == NET_CHANNEL_RELIABLE_ORD) {
    ack_pending_[ch] = 1;
    if (ch == NET_CHANNEL_RELIABLE_UNORD) {
      if (!reliability::update_recv_window_ext(
              slots_.recv_ack[conn_][ch], slots_.recv_ack_bits[conn_][ch],
              std::span<uint32_t>(slots_.recv_rel_unord_bits[conn_],
                                  ConnectionSlots::kReliableUnordRecvWindowWords),
              hdr->seq)) {
        pool_.release(buf_idx);
        return;
      }
    } else {
      // For ordered reliable, duplicate suppression is handled by the reorder
      // buffer (window + occupied checks). We update ACK state after a packet is
      // accepted into the reorder buffer.
    }
  } else {
    (void)update_ack_window_(slots_.recv_ack[conn_][ch],
                             slots_.recv_ack_bits[conn_][ch], hdr->seq);
  }

  if (ch == NET_CHANNEL_RELIABLE_ORD) {
    const bool inserted =
        reorder::insert(slots_, pool_, conn_, hdr->seq, buf_idx, len,
                        net_to_game_);
    if (inserted) {
      (void)update_ack_window_(slots_.recv_ack[conn_][ch],
                               slots_.recv_ack_bits[conn_][ch], hdr->seq);
    }
    return;
  }

  if (!net_to_game_.push(InboundMsg{
          .conn = conn_,
          .channel = ch,
          .buf_idx = buf_idx,
          .payload_offset = NET_HEADER_SIZE,
          .payload_len = payload_len,
      })) {
    pool_.release(buf_idx);
  }

  (void)now_ms;
}

asio::awaitable<void> Client::send_flush_loop_() {
  asio::steady_timer timer(ioc_);

#if NET_SIM
  SimSend sim_send[NET_SIM_DELAY_QUEUE_SIZE];
  RawRecv sim_recv[NET_SIM_DELAY_QUEUE_SIZE];
#endif

  while (running_.load(std::memory_order_acquire)) {
    timer.expires_after(std::chrono::milliseconds(NET_FLUSH_INTERVAL_MS));
    co_await timer.async_wait(asio::use_awaitable);

    if (!running_.load(std::memory_order_acquire)) {
      break;
    }

    const uint32_t now_ms = now_ms_();
    uint32_t send_count = 0;

    if (!connected_.load(std::memory_order_acquire)) {
      constexpr uint32_t kConnectRetryMs = 250;
      if (last_connect_sent_ms_ == 0 ||
          (now_ms - last_connect_sent_ms_) >= kConnectRetryMs) {
        send_connect_(now_ms);
      }
    }

#if NET_SIM
    uint32_t sim_send_count = 0;
    uint32_t sim_recv_count = 0;
    sim_.drain_ready(pool_, now_ms, std::span<SimSend>(sim_send),
                     sim_send_count, std::span<RawRecv>(sim_recv),
                     sim_recv_count);

    for (uint32_t i = 0; i < sim_recv_count; ++i) {
      handle_datagram_(sim_recv[i].buf_idx, sim_recv[i].len, sim_recv[i].endpoint,
                       now_ms);
    }

    if (sim_send_count > 0) {
      SendItem sim_items[NET_SIM_DELAY_QUEUE_SIZE];
      for (uint32_t i = 0; i < sim_send_count; ++i) {
        sim_items[i] = SendItem{pool_.ptr(sim_send[i].buf_idx), sim_send[i].len,
                                &sim_send[i].endpoint};
      }
      socket_.send_batch(sim_items, static_cast<int>(sim_send_count));

      for (uint32_t i = 0; i < sim_send_count; ++i) {
        const PoolIdx buf_idx = sim_send[i].buf_idx;
        if (buf_idx == kInvalidPoolIdx) {
          continue;
        }
        const uint8_t ch = pool_.ptr(buf_idx)[0];
        const bool is_ack_only =
            (sim_send[i].len == NET_HEADER_SIZE)
#if NET_ENCRYPTION
            || (sim_send[i].len == (NET_HEADER_SIZE + NET_ENCRYPTION_TAG_SIZE))
#endif
            ;
        pool_.sim_send_ref_dec(buf_idx);
        if (ch == NET_CHANNEL_UNRELIABLE || is_ack_only) {
          pool_.release(buf_idx);
        }
      }
    }
#endif

    // Reorder buffer draining is opportunistic on packet insert, but if the
    // net->game queue was full at the moment the expected packet arrived, the
    // packet is buffered and needs periodic drain attempts even if no further
    // ordered packets arrive.
    if (conn_ != kInvalidConnId && slots_.is_alive(conn_)) {
      reorder::try_drain(slots_, pool_, conn_, net_to_game_);
    }

    OutboundMsg msg{};
    const CryptoLayer *crypto =
#if NET_ENCRYPTION
        &crypto_;
#else
        nullptr;
#endif

    while (game_to_net_.pop(msg)) {
      if (!slots_.is_alive(msg.conn)) {
        continue;
      }
      if (msg.opcode >= c2s_table_.size()) {
        continue;
      }

      const OpcodeInfo &info = c2s_table_[msg.opcode];
      (void)batcher_.write(pool_, slots_, msg.conn, msg.channel, msg.opcode,
                           info, msg.staging, msg.len, now_ms,
                           std::span<SendItem>(send_list_), send_count, crypto);
    }

    batcher_.flush_all(pool_, slots_, now_ms, std::span<SendItem>(send_list_),
                       send_count, crypto);

    // Clear ACK-only hints when we already have a fresh packet scheduled on
    // that reliable channel (it already carries recv_ack + recv_ack_bits).
    const uint32_t fresh_count = send_count;
    for (uint32_t i = 0; i < fresh_count; ++i) {
      const uint8_t out_ch = send_list_[i].data[0];
      if (out_ch == NET_CHANNEL_RELIABLE_UNORD || out_ch == NET_CHANNEL_RELIABLE_ORD) {
        ack_pending_[out_ch] = 0;
      }
    }

    auto enqueue_ack_only = [&](uint8_t ch) -> bool {
      if (conn_ == kInvalidConnId || !slots_.is_alive(conn_)) {
        return false;
      }
      if (send_count >= kSendListCap) {
        return false;
      }

      const PoolIdx buf_idx = pool_.acquire();
      if (buf_idx == kInvalidPoolIdx) {
        return false;
      }

      uint8_t *buf = pool_.ptr(buf_idx);
      std::memset(buf, 0, NET_HEADER_SIZE);

      uint16_t pkt_len = NET_HEADER_SIZE;
      auto *hdr = reinterpret_cast<PacketHeader *>(buf);
      hdr->channel = ch;
      // ACK-only packets are dropped on the recv side before duplicate/ordering
      // checks. Do not advance the channel sequence number here, otherwise CH2
      // (ordered) will see gaps that stall the reorder buffer.
      hdr->seq = slots_.send_seq[conn_][ch];
      hdr->ack = slots_.recv_ack[conn_][ch];
      hdr->ack_bits = slots_.recv_ack_bits[conn_][ch];

#if NET_ENCRYPTION
      if (crypto != nullptr && slots_.encryption_ready[conn_] != 0) {
        if (!crypto->encrypt_in_place(buf, pkt_len, conn_, slots_)) {
          pool_.release(buf_idx);
          return false;
        }
      }
#endif

      send_list_[send_count++] = SendItem{buf, pkt_len, &slots_.endpoints[conn_]};
      return true;
    };

    // Standalone ACK-only packets: needed when the client is only receiving
    // reliable traffic and has no outbound payloads to piggyback ACKs onto.
    if (ack_pending_[NET_CHANNEL_RELIABLE_UNORD] != 0) {
      if (enqueue_ack_only(NET_CHANNEL_RELIABLE_UNORD)) {
        ack_pending_[NET_CHANNEL_RELIABLE_UNORD] = 0;
      }
    }
    if (ack_pending_[NET_CHANNEL_RELIABLE_ORD] != 0) {
      if (enqueue_ack_only(NET_CHANNEL_RELIABLE_ORD)) {
        ack_pending_[NET_CHANNEL_RELIABLE_ORD] = 0;
      }
    }

    reliability::retransmit_pass(slots_, pool_, conn_, NET_CHANNEL_RELIABLE_UNORD,
                                 now_ms, std::span<SendItem>(send_list_),
                                 send_count);
    reliability::retransmit_pass(slots_, pool_, conn_, NET_CHANNEL_RELIABLE_ORD,
                                 now_ms, std::span<SendItem>(send_list_),
                                 send_count);

#if NET_SIM
    uint32_t imm_count = 0;
    for (uint32_t i = 0; i < send_count; ++i) {
      const SendItem &it = send_list_[i];
      const PoolIdx idx = pool_.idx_from_ptr(it.data);
      if (idx == kInvalidPoolIdx || it.endpoint == nullptr) {
        continue;
      }
      const NetEndpoint epv = *it.endpoint;
      if (sim_.enqueue_send(pool_, idx, it.len, epv, now_ms)) {
        continue;
      }
      send_list_[imm_count++] = it;
    }
    send_count = imm_count;
#endif

    if (send_count > 0) {
      socket_.send_batch(send_list_, static_cast<int>(send_count));

      for (uint32_t i = 0; i < send_count; ++i) {
        const SendItem &it = send_list_[i];
        const PoolIdx idx = pool_.idx_from_ptr(it.data);
        if (idx == kInvalidPoolIdx) {
          continue;
        }
        const uint8_t ch = it.data[0];
        const bool is_ack_only =
            (it.len == NET_HEADER_SIZE)
#if NET_ENCRYPTION
            || (it.len == (NET_HEADER_SIZE + NET_ENCRYPTION_TAG_SIZE))
#endif
            ;
        if (ch == NET_CHANNEL_UNRELIABLE || is_ack_only) {
          pool_.release(idx);
        }
      }
    }
  }
}

void Client::tick(uint32_t delta_ms) {
  (void)delta_ms;

  InboundMsg msg{};
  while (net_to_game_.pop(msg)) {
    if (msg.buf_idx == kInvalidPoolIdx) {
      continue;
    }

    const uint8_t *base = pool_.ptr(msg.buf_idx);
    const uint8_t *cur = base + msg.payload_offset;
    const uint8_t *end = cur + msg.payload_len;

    while ((end - cur) >= NET_OPCODE_SIZE) {
      const uint16_t opcode = read_u16_le_(cur);
      cur += NET_OPCODE_SIZE;

      if (opcode >= s2c_table_.size()) {
        break;
      }

      const OpcodeInfo &info = s2c_table_[opcode];
      uint16_t plen = 0;

      if (info.payload_size == NET_PAYLOAD_DYNAMIC) {
        if ((end - cur) < NET_DYNLEN_SIZE) {
          break;
        }
        plen = read_u16_le_(cur);
        cur += NET_DYNLEN_SIZE;
      } else {
        plen = static_cast<uint16_t>(info.payload_size);
      }

      if (static_cast<size_t>(end - cur) < plen) {
        break;
      }

      auto &handler = s2c_handlers_[opcode];
      if (handler) {
        MessageReader r{std::span<const uint8_t>{cur, plen}, 0};
        handler(r);
      }

      cur += plen;
    }

    pool_.release(msg.buf_idx);
  }
}

} // namespace attome::net
