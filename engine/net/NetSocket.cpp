#include "NetSocket.h"

#include <asio/buffer.hpp>
#include <asio/redirect_error.hpp>
#include <asio/steady_timer.hpp>
#include <asio/use_awaitable.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

#if defined(__linux__)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace attome::net {

NetSocket::NetSocket(asio::io_context &ioc) : socket_(ioc), ioc_(ioc) {}

bool NetSocket::bind(uint16_t port) {
  asio::error_code ec;

  socket_.open(asio::ip::udp::v4(), ec);
  if (ec) {
    std::fprintf(stderr, "NetSocket::bind open failed: %s\n",
                 ec.message().c_str());
    return false;
  }

  socket_.set_option(asio::socket_base::reuse_address(true), ec);
  (void)ec;

  socket_.bind(asio::ip::udp::endpoint(asio::ip::udp::v4(), port), ec);
  if (ec) {
    std::fprintf(stderr, "NetSocket::bind bind failed: %s\n",
                 ec.message().c_str());
    return false;
  }

  return true;
}

void NetSocket::close() {
  asio::error_code ec;
  socket_.close(ec);
}

asio::awaitable<void> NetSocket::recv_loop(PacketPool &pool,
                                           SpscQueue<RawRecv, 8192> &raw_recv) {
  asio::steady_timer backoff(ioc_);

  while (true) {
    if (!socket_.is_open()) {
      co_return;
    }

    PoolIdx p = pool.acquire();
    if (p == kInvalidPoolIdx) {
      backoff.expires_after(std::chrono::milliseconds(1));
      co_await backoff.async_wait(asio::use_awaitable);
      continue;
    }

    asio::ip::udp::endpoint ep{};
    asio::error_code ec;
    const std::size_t n = co_await socket_.async_receive_from(
        asio::buffer(pool.ptr(p), NET_MTU), ep,
        asio::redirect_error(asio::use_awaitable, ec));

    if (ec) {
      pool.release(p);
      if (ec == asio::error::operation_aborted ||
          ec == asio::error::bad_descriptor) {
        co_return;
      }
      continue;
    }

    RawRecv r{};
    r.buf_idx = p;
    r.len = static_cast<uint16_t>(n);
    r.endpoint = to_net_endpoint(ep);

    if (!raw_recv.push(r)) {
      pool.release(p);
    }
  }
}

void NetSocket::send_batch(const SendItem *items, int count) {
  if (items == nullptr || count <= 0) {
    return;
  }

#if defined(__linux__)
  // Best-effort sendmmsg batching. Falls back to per-item sends for the
  // remainder if the syscall fails or partially sends.
  constexpr int kMaxBatch = 128;
  const int fd = socket_.native_handle();

  auto fill_addr = [](const NetEndpoint &ep, sockaddr_storage &dst,
                      socklen_t &dst_len) {
    std::memset(&dst, 0, sizeof(dst));
    if (ep.is_v6 == 0) {
      auto *a = reinterpret_cast<sockaddr_in *>(&dst);
      a->sin_family = AF_INET;
      a->sin_port = htons(ep.port);
      a->sin_addr.s_addr = htonl(ep.addr.v4);
      dst_len = sizeof(sockaddr_in);
      return;
    }

    auto *a = reinterpret_cast<sockaddr_in6 *>(&dst);
    a->sin6_family = AF_INET6;
    a->sin6_port = htons(ep.port);
    std::memcpy(&a->sin6_addr, ep.addr.v6, 16);
    a->sin6_scope_id = ep.v6_scope_id;
    dst_len = sizeof(sockaddr_in6);
  };

  int fallback_start = count;

  for (int base = 0; base < count; base += kMaxBatch) {
    const int batch = std::min(kMaxBatch, count - base);

    mmsghdr msgs[kMaxBatch]{};
    iovec iov[kMaxBatch]{};
    sockaddr_storage addrs[kMaxBatch]{};
    socklen_t addr_lens[kMaxBatch]{};
    int idx_map[kMaxBatch]{};

    int actual = 0;
    for (int i = 0; i < batch; ++i) {
      const SendItem &it = items[base + i];
      if (it.data == nullptr || it.len == 0 || it.endpoint == nullptr) {
        continue;
      }

      fill_addr(*it.endpoint, addrs[actual], addr_lens[actual]);
      iov[actual].iov_base = const_cast<uint8_t *>(it.data);
      iov[actual].iov_len = it.len;

      msghdr &hdr = msgs[actual].msg_hdr;
      hdr.msg_name = &addrs[actual];
      hdr.msg_namelen = addr_lens[actual];
      hdr.msg_iov = &iov[actual];
      hdr.msg_iovlen = 1;

      idx_map[actual] = base + i;
      actual += 1;
    }

    if (actual == 0) {
      continue;
    }

    int sent = ::sendmmsg(fd, msgs, actual, 0);
    if (sent < 0) {
      fallback_start = base;
      break;
    }
    if (sent < actual) {
      fallback_start = idx_map[sent];
      break;
    }
  }
#endif

  int start = 0;
#if defined(__linux__)
  if (fallback_start >= count) {
    return;
  }
  start = fallback_start;
#endif

  for (int i = start; i < count; ++i) {
    const SendItem &it = items[i];
    if (it.data == nullptr || it.len == 0 || it.endpoint == nullptr) {
      continue;
    }

    asio::error_code ec;
    const auto ep = to_asio_endpoint(*it.endpoint);
    socket_.send_to(asio::buffer(it.data, it.len), ep, 0, ec);
    if (ec) {
      std::fprintf(stderr, "NetSocket::send_batch send_to failed: %s\n",
                   ec.message().c_str());
    }
  }
}

void NetSocket::send_one(const uint8_t *data, uint16_t len,
                         const NetEndpoint &endpoint) {
  if (data == nullptr || len == 0) {
    return;
  }
  const SendItem item{data, len, &endpoint};
  send_batch(&item, 1);
}

NetEndpoint NetSocket::to_net_endpoint(const asio::ip::udp::endpoint &ep) {
  NetEndpoint out{};
  out.port = ep.port();

  const auto addr = ep.address();
  if (addr.is_v4()) {
    out.is_v6 = 0;
    out.addr.v4 = addr.to_v4().to_uint();
    out.v6_scope_id = 0;
    return out;
  }

  out.is_v6 = 1;
  const auto v6 = addr.to_v6();
  out.v6_scope_id = v6.scope_id();
  const auto bytes = v6.to_bytes();
  std::memcpy(out.addr.v6, bytes.data(), 16);
  return out;
}

asio::ip::udp::endpoint NetSocket::to_asio_endpoint(const NetEndpoint &ep) {
  if (ep.is_v6 == 0) {
    return asio::ip::udp::endpoint(
        asio::ip::address_v4(ep.addr.v4), ep.port);
  }

  asio::ip::address_v6::bytes_type bytes{};
  std::memcpy(bytes.data(), ep.addr.v6, 16);
  return asio::ip::udp::endpoint(
      asio::ip::address_v6(bytes, ep.v6_scope_id), ep.port);
}

} // namespace attome::net
