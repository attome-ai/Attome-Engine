#include "Socket.h"

#include <cstring>
#include <mutex>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mstcpip.h>
#ifndef SIO_UDP_CONNRESET
#define SIO_UDP_CONNRESET _WSAIOW(IOC_VENDOR, 12)
#endif
using SockLen = int;
static constexpr intptr_t kInvalid = intptr_t(INVALID_SOCKET);
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
using SockLen = socklen_t;
static constexpr intptr_t kInvalid = -1;
#endif

namespace atm::net2::detail {

namespace {

#ifdef _WIN32
std::mutex g_wsaMutex;
int g_wsaRefs = 0;
bool wsaAcquire() {
  std::lock_guard<std::mutex> lock(g_wsaMutex);
  if (g_wsaRefs == 0) {
    WSADATA data;
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return false;
  }
  ++g_wsaRefs;
  return true;
}
void wsaRelease() {
  std::lock_guard<std::mutex> lock(g_wsaMutex);
  if (g_wsaRefs > 0 && --g_wsaRefs == 0) WSACleanup();
}
SOCKET sock(intptr_t h) { return SOCKET(h); }
int lastError() { return WSAGetLastError(); }
#else
bool wsaAcquire() { return true; }
void wsaRelease() {}
int sock(intptr_t h) { return int(h); }
int lastError() { return errno; }
#endif

void setError(std::string *error, const char *what) {
  if (error) *error = std::string(what) + " (error " + std::to_string(lastError()) + ")";
}

} // namespace

UdpSocket::UdpSocket() : handle_(kInvalid) {}
UdpSocket::~UdpSocket() { close(); }

bool UdpSocket::isOpen() const { return handle_ != kInvalid; }

bool UdpSocket::open(uint16_t port, std::string *error) {
  close();
  if (!wsaAcquire()) {
    if (error) *error = "WSAStartup failed";
    return false;
  }
#ifdef _WIN32
  SOCKET s = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (s == INVALID_SOCKET) {
    setError(error, "socket() failed");
    wsaRelease();
    return false;
  }
  handle_ = intptr_t(s);
  BOOL exclusive = TRUE; // no port sharing/hijack (D18)
  ::setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char *>(&exclusive), sizeof(exclusive));
  // Ignore ICMP port-unreachable resets (otherwise recvfrom fails with
  // WSAECONNRESET after sending to a closed client port).
  BOOL newBehavior = FALSE;
  DWORD bytesReturned = 0;
  ::WSAIoctl(s, SIO_UDP_CONNRESET, &newBehavior, sizeof(newBehavior), nullptr, 0, &bytesReturned, nullptr, nullptr);
  u_long nonBlocking = 1;
  if (::ioctlsocket(s, FIONBIO, &nonBlocking) != 0) {
    setError(error, "ioctlsocket(FIONBIO) failed");
    close();
    return false;
  }
#else
  int s = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (s < 0) {
    setError(error, "socket() failed");
    return false;
  }
  handle_ = intptr_t(s);
  const int flags = ::fcntl(s, F_GETFL, 0);
  if (flags < 0 || ::fcntl(s, F_SETFL, flags | O_NONBLOCK) != 0) {
    setError(error, "fcntl(O_NONBLOCK) failed");
    close();
    return false;
  }
#endif
  int bufSize = 4 * 1024 * 1024;
  ::setsockopt(sock(handle_), SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char *>(&bufSize), sizeof(bufSize));
  ::setsockopt(sock(handle_), SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char *>(&bufSize), sizeof(bufSize));

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  if (::bind(sock(handle_), reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) != 0) {
    setError(error, "bind() failed");
    close();
    return false;
  }
  sockaddr_in bound;
  std::memset(&bound, 0, sizeof(bound));
  SockLen len = sizeof(bound);
  if (::getsockname(sock(handle_), reinterpret_cast<sockaddr *>(&bound), &len) == 0)
    localPort_ = ntohs(bound.sin_port);
  else
    localPort_ = port;
  return true;
}

void UdpSocket::close() {
  if (handle_ == kInvalid) return;
#ifdef _WIN32
  ::closesocket(sock(handle_));
#else
  ::close(sock(handle_));
#endif
  handle_ = kInvalid;
  localPort_ = 0;
  wsaRelease();
}

int UdpSocket::receive(uint8_t *buf, int capacity, Address &from) {
  if (handle_ == kInvalid) return -1;
  for (int attempts = 0; attempts < 64; ++attempts) {
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    SockLen len = sizeof(addr);
#ifdef _WIN32
    const int n = ::recvfrom(sock(handle_), reinterpret_cast<char *>(buf), capacity, 0,
                             reinterpret_cast<sockaddr *>(&addr), &len);
    if (n == SOCKET_ERROR) {
      const int e = WSAGetLastError();
      if (e == WSAEWOULDBLOCK) return 0;
      if (e == WSAECONNRESET || e == WSAEMSGSIZE || e == WSAENETRESET || e == WSAEINTR) continue;
      return -1;
    }
#else
    const ssize_t n = ::recvfrom(sock(handle_), buf, size_t(capacity), MSG_TRUNC,
                                 reinterpret_cast<sockaddr *>(&addr), &len);
    if (n < 0) {
      const int e = errno;
      if (e == EAGAIN || e == EWOULDBLOCK) return 0;
      if (e == EINTR || e == ECONNREFUSED) continue;
      return -1;
    }
    if (n > capacity) continue; // truncated oversized datagram: drop
#endif
    if (addr.sin_family != AF_INET) continue;
    from.ip = ntohl(addr.sin_addr.s_addr);
    from.port = ntohs(addr.sin_port);
    if (n == 0) continue;
    return int(n);
  }
  return 0;
}

bool UdpSocket::send(const Address &to, const uint8_t *data, int len) {
  if (handle_ == kInvalid || len <= 0) return false;
  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(to.ip);
  addr.sin_port = htons(to.port);
#ifdef _WIN32
  const int n = ::sendto(sock(handle_), reinterpret_cast<const char *>(data), len, 0,
                         reinterpret_cast<const sockaddr *>(&addr), sizeof(addr));
#else
  const ssize_t n = ::sendto(sock(handle_), data, size_t(len), 0,
                             reinterpret_cast<const sockaddr *>(&addr), sizeof(addr));
#endif
  return n == len;
}

bool UdpSocket::resolve(const std::string &host, uint16_t port, Address &out) {
  if (!wsaAcquire()) return false;
  bool ok = false;
  in_addr a;
  if (::inet_pton(AF_INET, host.c_str(), &a) == 1) {
    out.ip = ntohl(a.s_addr);
    out.port = port;
    ok = true;
  } else {
    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo *res = nullptr;
    if (::getaddrinfo(host.c_str(), nullptr, &hints, &res) == 0 && res) {
      for (addrinfo *p = res; p; p = p->ai_next) {
        if (p->ai_family == AF_INET && p->ai_addr) {
          const sockaddr_in *sin = reinterpret_cast<const sockaddr_in *>(p->ai_addr);
          out.ip = ntohl(sin->sin_addr.s_addr);
          out.port = port;
          ok = true;
          break;
        }
      }
      ::freeaddrinfo(res);
    }
  }
  wsaRelease();
  return ok;
}

} // namespace atm::net2::detail
