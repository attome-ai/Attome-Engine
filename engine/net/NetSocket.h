#pragma once

#include "NetTypes.h"
#include "PacketPool.h"
#include "SpscQueue.h"

#include <asio/awaitable.hpp>
#include <asio/io_context.hpp>
#include <asio/ip/udp.hpp>

#include <cstdint>

namespace attome::net {

class NetSocket {
public:
  explicit NetSocket(asio::io_context &ioc);

  bool bind(uint16_t port);
  void close();

  asio::ip::udp::socket &socket() { return socket_; }

  asio::awaitable<void> recv_loop(PacketPool &pool,
                                  SpscQueue<RawRecv, 8192> &raw_recv);

  void send_batch(const SendItem *items, int count);
  void send_one(const uint8_t *data, uint16_t len, const NetEndpoint &endpoint);

  static NetEndpoint to_net_endpoint(const asio::ip::udp::endpoint &ep);
  static asio::ip::udp::endpoint to_asio_endpoint(const NetEndpoint &ep);

private:
  asio::ip::udp::socket socket_;
  asio::io_context &ioc_;
};

} // namespace attome::net
