#pragma once

// Minimal non-blocking UDP socket (IPv4) behind a tiny platform layer:
// Winsock2 on Windows, BSD sockets elsewhere. Internal to AttomeNet2.
// TODO(N1): IPv6, recvmmsg/sendmmsg batching, SO_REUSEPORT sharding.

#include <cstdint>
#include <string>

namespace atm::net2::detail {

struct Address {
  uint32_t ip = 0;   // host byte order
  uint16_t port = 0; // host byte order
  uint64_t key() const { return (uint64_t(ip) << 16) | port; }
  friend bool operator==(const Address &a, const Address &b) { return a.ip == b.ip && a.port == b.port; }
};

class UdpSocket {
public:
  UdpSocket();
  ~UdpSocket();
  UdpSocket(const UdpSocket &) = delete;
  UdpSocket &operator=(const UdpSocket &) = delete;

  // Binds to 0.0.0.0:port (0 = ephemeral). Exclusive bind (D18).
  bool open(uint16_t port, std::string *error);
  void close();
  bool isOpen() const;

  // >0: bytes received; 0: nothing pending; <0: fatal socket error.
  // Oversized datagrams and ICMP "port unreachable" resets are skipped.
  int receive(uint8_t *buf, int capacity, Address &from);
  bool send(const Address &to, const uint8_t *data, int len);
  uint16_t localPort() const { return localPort_; }

  static bool resolve(const std::string &host, uint16_t port, Address &out);

private:
  intptr_t handle_;
  uint16_t localPort_ = 0;
};

} // namespace atm::net2::detail
