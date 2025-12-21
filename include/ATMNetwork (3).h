#ifndef ATM_NETWORK_H
#define ATM_NETWORK_H

#include <vector>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>
#include <random>
#include <cstring>
#include <algorithm>

// Cross-platform socket handling
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET SocketHandle;
#define INVALID_SOCKET_HANDLE INVALID_SOCKET
#define SOCKET_ERROR_RETURN SOCKET_ERROR
#define CLOSE_SOCKET(s) closesocket(s)
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
typedef int SocketHandle;
#define INVALID_SOCKET_HANDLE -1
#define SOCKET_ERROR_RETURN -1
#define CLOSE_SOCKET(s) close(s)
#endif

// Constants
constexpr uint32_t MAX_ACTIVE_USERS = 1024;
constexpr uint32_t MAX_PACKET_SIZE = 1500;
constexpr uint32_t PACKET_HEADER_SIZE = 16;
constexpr uint32_t MAX_BUFFER_COUNT = 1024;
constexpr uint32_t MAX_RETRANSMISSION_TIME = 1000; // ms

// Forward declarations
class SocketAddress;
class BufferHandle;
class UDPPacket;
struct UserData;

// Header structures
struct UnreliableHeader {
    uint16_t userID;
};

struct ReliableUnorderedHeader {
    uint16_t userID;
    uint16_t packetID;
    uint8_t AckSize;
};

struct ReliableOrderedHeader {
    uint16_t userID;
    uint16_t packetID;
    uint8_t AckSize;
};

// Socket Address wrapper
class SocketAddress {
private:


public:
    SocketAddress() {
        memset(&addr, 0, sizeof(addr));
        addrLen = sizeof(addr);
    }

    void setIPv4(const std::string& ipAddress, uint16_t port) {
        sockaddr_in* addrv4 = reinterpret_cast<sockaddr_in*>(&addr);
        addrv4->sin_family = AF_INET;
        addrv4->sin_port = htons(port);
        inet_pton(AF_INET, ipAddress.c_str(), &addrv4->sin_addr);
        addrLen = sizeof(sockaddr_in);
    }

    void setIPv6(const std::string& ipAddress, uint16_t port) {
        sockaddr_in6* addrv6 = reinterpret_cast<sockaddr_in6*>(&addr);
        addrv6->sin6_family = AF_INET6;
        addrv6->sin6_port = htons(port);
        inet_pton(AF_INET6, ipAddress.c_str(), &addrv6->sin6_addr);
        addrLen = sizeof(sockaddr_in6);
    }

    uint16_t getPort() const {
        if (addr.ss_family == AF_INET) {
            return ntohs(reinterpret_cast<const sockaddr_in*>(&addr)->sin_port);
        }
        else if (addr.ss_family == AF_INET6) {
            return ntohs(reinterpret_cast<const sockaddr_in6*>(&addr)->sin6_port);
        }
        return 0;
    }

    std::string getIPString() const {
        char buffer[INET6_ADDRSTRLEN];
        if (addr.ss_family == AF_INET) {
            inet_ntop(AF_INET, &(reinterpret_cast<const sockaddr_in*>(&addr)->sin_addr), buffer, INET_ADDRSTRLEN);
        }
        else if (addr.ss_family == AF_INET6) {
            inet_ntop(AF_INET6, &(reinterpret_cast<const sockaddr_in6*>(&addr)->sin6_addr), buffer, INET6_ADDRSTRLEN);
        }
        else {
            return "";
        }
        return std::string(buffer);
    }

    const sockaddr* asSockAddr() const {
        return reinterpret_cast<const sockaddr*>(&addr);
    }

    sockaddr* asSockAddr() {
        return reinterpret_cast<sockaddr*>(&addr);
    }

    socklen_t getSize() const {
        return addrLen;
    }

    bool operator==(const SocketAddress& other) const {
        if (addr.ss_family != other.addr.ss_family) return false;

        if (addr.ss_family == AF_INET) {
            const sockaddr_in* a = reinterpret_cast<const sockaddr_in*>(&addr);
            const sockaddr_in* b = reinterpret_cast<const sockaddr_in*>(&other.addr);
            return (a->sin_port == b->sin_port && a->sin_addr.s_addr == b->sin_addr.s_addr);
        }
        else if (addr.ss_family == AF_INET6) {
            const sockaddr_in6* a = reinterpret_cast<const sockaddr_in6*>(&addr);
            const sockaddr_in6* b = reinterpret_cast<const sockaddr_in6*>(&other.addr);
            return (a->sin6_port == b->sin6_port &&
                memcmp(&a->sin6_addr, &b->sin6_addr, sizeof(in6_addr)) == 0);
        }

        return false;
    }

    bool operator!=(const SocketAddress& other) const {
        return !(*this == other);
    }
};

// Buffer pool handle
class BufferHandle {
private:
    uint8_t* buffer;
    std::function<void(uint8_t*)> releaseFunction;

public:
    BufferHandle() : buffer(nullptr), releaseFunction(nullptr) {}

    BufferHandle(uint8_t* buf, std::function<void(uint8_t*)> relFunc)
        : buffer(buf), releaseFunction(relFunc) {}

    BufferHandle(BufferHandle&& other) noexcept
        : buffer(other.buffer), releaseFunction(std::move(other.releaseFunction)) {
        other.buffer = nullptr;
    }

    BufferHandle& operator=(BufferHandle&& other) noexcept {
        if (this != &other) {
            if (buffer && releaseFunction) {
                releaseFunction(buffer);
            }
            buffer = other.buffer;
            releaseFunction = std::move(other.releaseFunction);
            other.buffer = nullptr;
        }
        return *this;
    }

    ~BufferHandle() {
        if (buffer && releaseFunction) {
            releaseFunction(buffer);
        }
    }

    // Disallow copying
    BufferHandle(const BufferHandle&) = delete;
    BufferHandle& operator=(const BufferHandle&) = delete;

    uint8_t* data() const { return buffer; }

    explicit operator bool() const { return buffer != nullptr; }
};

// UDP Packet
class UDPPacket {
public:
    BufferHandle bufferHandle;
    SocketAddress senderAddress;
    size_t dataLength;

    UDPPacket() : dataLength(0) {}

    UDPPacket(SocketAddress* sender, BufferHandle&& handle)
        : bufferHandle(std::move(handle)), dataLength(0) {
        if (sender) {
            senderAddress = *sender;
        }
    }

    UDPPacket(UDPPacket&& other) noexcept
        : bufferHandle(std::move(other.bufferHandle)),
        senderAddress(other.senderAddress),
        dataLength(other.dataLength) {}

    UDPPacket& operator=(UDPPacket&& other) noexcept {
        if (this != &other) {
            bufferHandle = std::move(other.bufferHandle);
            senderAddress = other.senderAddress;
            dataLength = other.dataLength;
        }
        return *this;
    }

    // Disallow copying
    UDPPacket(const UDPPacket&) = delete;
    UDPPacket& operator=(const UDPPacket&) = delete;
};

// User data structure
struct UserData {
    uint16_t userID;
    std::array<uint8_t, 32> aesKey;
    SocketAddress lastAddress;
    uint16_t lastReliableUnorderedPacketID;
    uint16_t lastReliableOrderedPacketID;
    uint16_t nextReliableUnorderedPacketID;
    uint16_t nextReliableOrderedPacketID;
    std::vector<uint16_t> pendingAcksUnordered;
    std::vector<uint16_t> pendingAcksOrdered;

    UserData()
    {
    }
    UserData(uint16_t id) : userID(id),
        lastReliableUnorderedPacketID(0),
        lastReliableOrderedPacketID(0),
        nextReliableUnorderedPacketID(1),
        nextReliableOrderedPacketID(1) {}
};

// Base UDP class with common functionality
class UDPBase {
protected:
    SocketHandle socketHandle;
    SocketAddress boundAddress;
    std::vector<std::array<uint8_t, MAX_PACKET_SIZE>> bufferPool;
    std::vector<bool> bufferInUse;
    int packetLossPercentage;
    std::mt19937 rng;

    std::unordered_map<uint16_t, UserData> users;

public:
    UDPBase() : socketHandle(INVALID_SOCKET_HANDLE), packetLossPercentage(0),
        rng(std::random_device{}()) {
        // Initialize buffer pool
        bufferPool.resize(MAX_BUFFER_COUNT);
        bufferInUse.resize(MAX_BUFFER_COUNT, false);

        // Initialize network on Windows
#ifdef _WIN32
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
    }

    virtual ~UDPBase() {
        shutdown();

#ifdef _WIN32
        WSACleanup();
#endif
    }

    // Initialize socket
    bool init(const std::string& address = "0.0.0.0", uint16_t port = 0) {
        // Create socket
        socketHandle = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socketHandle == INVALID_SOCKET_HANDLE) {
            return false;
        }

        // Set non-blocking mode
#ifdef _WIN32
        u_long mode = 1;
        ioctlsocket(socketHandle, FIONBIO, &mode);
#else
        int flags = fcntl(socketHandle, F_GETFL, 0);
        fcntl(socketHandle, F_SETFL, flags | O_NONBLOCK);
#endif

        // Bind to address and port
        boundAddress.setIPv4(address, port);
        if (bind(socketHandle, boundAddress.asSockAddr(), boundAddress.getSize()) == SOCKET_ERROR_RETURN) {
            shutdown();
            return false;
        }

        // Get assigned port if we used port 0 (system-assigned)
        if (port == 0) {
            sockaddr_in sin;
            socklen_t len = sizeof(sin);
            getsockname(socketHandle, (sockaddr*)&sin, &len);
            boundAddress.setIPv4(address, ntohs(sin.sin_port));
        }

        return true;
    }

    // Close socket
    void shutdown() {
        if (socketHandle != INVALID_SOCKET_HANDLE) {
            CLOSE_SOCKET(socketHandle);
            socketHandle = INVALID_SOCKET_HANDLE;
        }
    }

    // Check if socket is valid
    bool isValid() const {
        return socketHandle != INVALID_SOCKET_HANDLE;
    }

    // Get bound port
    uint16_t getPort() const {
        return boundAddress.getPort();
    }

    // Get bound address string
    std::string getAddress() const {
        return boundAddress.getIPString();
    }

    // Get buffer from pool
    BufferHandle getBuffer() {
        for (size_t i = 0; i < bufferInUse.size(); i++) {
            if (!bufferInUse[i]) {
                bufferInUse[i] = true;

                return BufferHandle(bufferPool[i].data(), [this, i](uint8_t*) {
                    bufferInUse[i] = false;
                    });
            }
        }

        // No buffers available
        return BufferHandle();
    }

    // Receive a packet from the socket
    UDPPacket receive() {
        if (!isValid()) {
            return UDPPacket();
        }

        auto buffer = getBuffer();
        if (!buffer) {
            return UDPPacket();
        }

        SocketAddress sender;
        socklen_t addrLen = sender.getSize();

        int bytesReceived = recvfrom(socketHandle,
            reinterpret_cast<char*>(buffer.data()),
            MAX_PACKET_SIZE,
            0,
            sender.asSockAddr(),
            &addrLen);

        if (bytesReceived <= 0) {
            return UDPPacket();
        }

        UDPPacket packet(&sender, std::move(buffer));
        packet.dataLength = bytesReceived;

        return packet;
    }

    // Wait for data with timeout
    int waitForData(int timeoutMs) {
        if (!isValid()) {
            return 0;
        }

        fd_set readSet;
        FD_ZERO(&readSet);
        FD_SET(socketHandle, &readSet);

        timeval timeout;
        timeout.tv_sec = timeoutMs / 1000;
        timeout.tv_usec = (timeoutMs % 1000) * 1000;

        return select(socketHandle + 1, &readSet, nullptr, nullptr, &timeout);
    }

    // Add a user
    bool addUser(uint16_t userID, const std::array<uint8_t, 32>& aesKey) {
        if (users.find(userID) != users.end()) {
            return false;
        }

        UserData userData(userID);
        userData.aesKey = aesKey;
        users[userID] = userData;

        return true;
    }

    // Remove a user
    bool removeUser(uint16_t userID) {
        auto it = users.find(userID);
        if (it == users.end()) {
            return false;
        }

        users.erase(it);
        return true;
    }

    // Check if a user is active
    bool hasActiveUser(uint16_t userID) const {
        return users.find(userID) != users.end();
    }

    // Get user data
    bool getUser(uint16_t userID, UserData*& userData) {
        auto it = users.find(userID);
        if (it == users.end()) {
            return false;
        }

        userData = &(it->second);
        return true;
    }

    // Get user's AES key
    std::array<uint8_t, 32> getUserAESKey(uint16_t userID) {
        auto it = users.find(userID);
        if (it == users.end()) {
            std::array<uint8_t, 32> emptyKey = {};
            return emptyKey;
        }

        return it->second.aesKey;
    }

    // Simulate packet loss (for testing)
    void simulatePacketLoss(int percentage) {
        packetLossPercentage = max(0, min(100, percentage));
    }

    // Check if packet should be dropped based on configured loss rate
    bool shouldDropPacket() {
        if (packetLossPercentage <= 0) return false;
        if (packetLossPercentage >= 100) return true;

        std::uniform_int_distribution<> dis(1, 100);
        return dis(rng) <= packetLossPercentage;
    }
};

// Default UDP class - basic UDP functionality with no headers
class DefaultUDP : public UDPBase {
public:
    DefaultUDP() : UDPBase() {}

    // Send a packet
    int sendPacket(UDPPacket&& packet, const SocketAddress& destination) {
        if (!isValid() || !packet.bufferHandle || shouldDropPacket()) {
            return 0;
        }

        return sendto(socketHandle,
            reinterpret_cast<const char*>(packet.bufferHandle.data()),
            packet.dataLength,
            0,
            destination.asSockAddr(),
            destination.getSize());
    }
};

// Unreliable UDP class - UDP with userID
class UnreliableUDP : public UDPBase {
public:
    UnreliableUDP() : UDPBase() {}

    // Prepare unreliable packet
    UDPPacket preparePacket(uint16_t userID) {
        if (!hasActiveUser(userID)) {
            return UDPPacket();
        }

        auto buffer = getBuffer();
        if (!buffer) {
            return UDPPacket();
        }

        // Set up header
        UnreliableHeader* header = reinterpret_cast<UnreliableHeader*>(buffer.data());
        header->userID = userID;

        UserData* userData;
        if (!getUser(userID, userData)) {
            return UDPPacket();
        }

        return UDPPacket(&userData->lastAddress, std::move(buffer));
    }

    // Send an unreliable packet
    int sendPacket(UDPPacket&& packet) {
        if (!isValid() || !packet.bufferHandle || shouldDropPacket()) {
            return 0;
        }

        // Get the userID from the packet
        UnreliableHeader* header = reinterpret_cast<UnreliableHeader*>(packet.bufferHandle.data());
        UserData* userData;

        if (!getUser(header->userID, userData)) {
            return 0;
        }

        return sendto(socketHandle,
            reinterpret_cast<const char*>(packet.bufferHandle.data()),
            packet.dataLength,
            0,
            userData->lastAddress.asSockAddr(),
            userData->lastAddress.getSize());
    }

    // Process an unreliable packet (when received)
    bool processPacket(const UDPPacket& packet) {
        if (!packet.bufferHandle || packet.dataLength < sizeof(UnreliableHeader)) {
            return false;
        }

        UnreliableHeader* header = reinterpret_cast<UnreliableHeader*>(packet.bufferHandle.data());

        // Update user's last address
        UserData* userData;
        if (getUser(header->userID, userData)) {
            userData->lastAddress = packet.senderAddress;
            return true;
        }

        return false;
    }
};

// Reliable Unordered UDP class
class ReliableUnorderedUDP : public UDPBase {
public:
    ReliableUnorderedUDP() : UDPBase() {}

    // Prepare reliable unordered packet
    UDPPacket preparePacket(uint16_t userID) {
        if (!hasActiveUser(userID)) {
            return UDPPacket();
        }

        auto buffer = getBuffer();
        if (!buffer) {
            return UDPPacket();
        }

        UserData* userData;
        if (!getUser(userID, userData)) {
            return UDPPacket();
        }

        // Set up header
        ReliableUnorderedHeader* header = reinterpret_cast<ReliableUnorderedHeader*>(buffer.data());
        header->userID = userID;
        header->packetID = userData->nextReliableUnorderedPacketID++;
        header->AckSize = static_cast<uint8_t>(min(userData->pendingAcksUnordered.size(), size_t(255)));

        // Add pending ACKs after the header
        uint16_t* ackPtr = reinterpret_cast<uint16_t*>(buffer.data() + sizeof(ReliableUnorderedHeader));
        for (size_t i = 0; i < header->AckSize; i++) {
            ackPtr[i] = userData->pendingAcksUnordered[i];
        }

        // Clear pending ACKs now that they are included
        userData->pendingAcksUnordered.clear();

        return UDPPacket(&userData->lastAddress, std::move(buffer));
    }

    // Send a reliable unordered packet
    int sendPacket(UDPPacket&& packet) {
        if (!isValid() || !packet.bufferHandle || shouldDropPacket()) {
            return 0;
        }

        // Get the userID from the packet
        ReliableUnorderedHeader* header = reinterpret_cast<ReliableUnorderedHeader*>(packet.bufferHandle.data());
        UserData* userData;

        if (!getUser(header->userID, userData)) {
            return 0;
        }

        return sendto(socketHandle,
            reinterpret_cast<const char*>(packet.bufferHandle.data()),
            packet.dataLength,
            0,
            userData->lastAddress.asSockAddr(),
            userData->lastAddress.getSize());
    }

    // Process a reliable unordered packet (when received)
    bool processPacket(const UDPPacket& packet) {
        if (!packet.bufferHandle || packet.dataLength < sizeof(ReliableUnorderedHeader)) {
            return false;
        }

        ReliableUnorderedHeader* header = reinterpret_cast<ReliableUnorderedHeader*>(packet.bufferHandle.data());

        // Update user's last address
        UserData* userData;
        if (!getUser(header->userID, userData)) {
            return false;
        }

        userData->lastAddress = packet.senderAddress;

        // Process ACKs in the packet
        const uint16_t* acks = reinterpret_cast<const uint16_t*>(
            packet.bufferHandle.data() + sizeof(ReliableUnorderedHeader));

        for (size_t i = 0; i < header->AckSize; i++) {
            // Process each ACK (implementation would handle retransmission tracking)
        }

        // Add this packet's ID to pending ACKs to acknowledge receipt
        userData->pendingAcksUnordered.push_back(header->packetID);

        return true;
    }

    // Execute periodic tasks (retransmission, etc.)
    void execute() {
        // For each user, handle retransmissions as needed
        for (auto& userPair : users) {
            UserData& userData = userPair.second;

            // Retransmission logic would go here
            // For packets that haven't been ACKed within timeout
        }
    }
};

// Reliable Ordered UDP class
class ReliableOrderedUDP : public UDPBase {
public:
    ReliableOrderedUDP() : UDPBase() {}

    // Prepare reliable ordered packet
    UDPPacket preparePacket(uint16_t userID) {
        if (!hasActiveUser(userID)) {
            return UDPPacket();
        }

        auto buffer = getBuffer();
        if (!buffer) {
            return UDPPacket();
        }

        UserData* userData;
        if (!getUser(userID, userData)) {
            return UDPPacket();
        }

        // Set up header
        ReliableOrderedHeader* header = reinterpret_cast<ReliableOrderedHeader*>(buffer.data());
        header->userID = userID;
        header->packetID = userData->nextReliableOrderedPacketID++;
        header->AckSize = static_cast<uint8_t>(min(userData->pendingAcksOrdered.size(), size_t(255)));

        // Add pending ACKs after the header
        uint16_t* ackPtr = reinterpret_cast<uint16_t*>(buffer.data() + sizeof(ReliableOrderedHeader));
        for (size_t i = 0; i < header->AckSize; i++) {
            ackPtr[i] = userData->pendingAcksOrdered[i];
        }

        // Clear pending ACKs now that they are included
        userData->pendingAcksOrdered.clear();

        return UDPPacket(&userData->lastAddress, std::move(buffer));
    }

    // Send a reliable ordered packet
    int sendPacket(UDPPacket&& packet) {
        if (!isValid() || !packet.bufferHandle || shouldDropPacket()) {
            return 0;
        }

        // Get the userID from the packet
        ReliableOrderedHeader* header = reinterpret_cast<ReliableOrderedHeader*>(packet.bufferHandle.data());
        UserData* userData;

        if (!getUser(header->userID, userData)) {
            return 0;
        }

        return sendto(socketHandle,
            reinterpret_cast<const char*>(packet.bufferHandle.data()),
            packet.dataLength,
            0,
            userData->lastAddress.asSockAddr(),
            userData->lastAddress.getSize());
    }

    // Process a reliable ordered packet (when received)
    bool processPacket(const UDPPacket& packet) {
        if (!packet.bufferHandle || packet.dataLength < sizeof(ReliableOrderedHeader)) {
            return false;
        }

        ReliableOrderedHeader* header = reinterpret_cast<ReliableOrderedHeader*>(packet.bufferHandle.data());

        // Update user's last address
        UserData* userData;
        if (!getUser(header->userID, userData)) {
            return false;
        }

        userData->lastAddress = packet.senderAddress;

        // Process ACKs in the packet
        const uint16_t* acks = reinterpret_cast<const uint16_t*>(
            packet.bufferHandle.data() + sizeof(ReliableOrderedHeader));

        for (size_t i = 0; i < header->AckSize; i++) {
            // Process each ACK (implementation would handle retransmission tracking)
        }

        // For ordered packets, only accept if it's the next expected or a future packet
        if (header->packetID < userData->lastReliableOrderedPacketID + 1) {
            // Packet is old, but still ACK it
            userData->pendingAcksOrdered.push_back(header->packetID);
            return false;
        }

        // Update last received packet ID if this is the next one
        if (header->packetID == userData->lastReliableOrderedPacketID + 1) {
            userData->lastReliableOrderedPacketID = header->packetID;
        }

        // Add this packet's ID to pending ACKs
        userData->pendingAcksOrdered.push_back(header->packetID);

        return true;
    }

    // Execute periodic tasks (retransmission, etc.)
    void execute() {
        // For each user, handle retransmissions as needed
        for (auto& userPair : users) {
            UserData& userData = userPair.second;

            // Retransmission logic would go here
            // For packets that haven't been ACKed within timeout
        }
    }
};


#endif // ATM_NETWORK_H