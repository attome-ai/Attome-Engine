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
#include "ATMBufferPool.h"
#include "ATMByteBuffer.h"
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
#undef max
#undef min

// Constants
constexpr uint32_t MAX_ACTIVE_USERS = 100;
constexpr uint32_t MAX_ACTIVE_PACKET = 128;
constexpr uint32_t MAX_PACKET_BUFFER_SIZE = 1500;
constexpr uint32_t MAX_BUFFER_COUNT = 1024;
constexpr uint32_t MAX_RETRANSMISSION_TIME = 1000; // ms

// Forward declarations
class SocketAddress;
class UDPPacket;
struct UserData;


#include <cassert>
#include <bitset>

// Socket Address wrapper
class SocketAddress {
private:
    sockaddr_storage addr;
    socklen_t addrLen;

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

    uint16_t getPortV4() const
    {
        assert(addr.ss_family == AF_INET && "Attempting to get IPv4 port from non-IPv4 address");
        return ntohs(reinterpret_cast<const sockaddr_in*>(&addr)->sin_port);
    }

    uint16_t getPortV6() const
    {
        assert(addr.ss_family == AF_INET6 && "Attempting to get IPv6 port from non-IPv6 address");
        return ntohs(reinterpret_cast<const sockaddr_in6*>(&addr)->sin6_port);
    }

    std::string getIPStringV4() const
    {
        assert(addr.ss_family == AF_INET && "Attempting to get IPv4 string from non-IPv4 address");
        char buffer[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(reinterpret_cast<const sockaddr_in*>(&addr)->sin_addr), buffer, INET_ADDRSTRLEN);
        return std::string(buffer);
    }

    std::string getIPStringV6() const
    {
        assert(addr.ss_family == AF_INET6 && "Attempting to get IPv6 string from non-IPv6 address");
        char buffer[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, &(reinterpret_cast<const sockaddr_in6*>(&addr)->sin6_addr), buffer, INET6_ADDRSTRLEN);
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

class UDPPacket {
public:
    FixedBufferPool::BufferHandle bufferHandle;
    uint16_t userID;
    uint16_t dataLength;

    UDPPacket() : dataLength(-1), userID(-1) {}

    UDPPacket(FixedBufferPool::BufferHandle&& handle)
        : bufferHandle(std::move(handle)), dataLength(0), userID(-1) {

    }
    size_t getAvailable() const {
        return MAX_PACKET_BUFFER_SIZE - dataLength;
    }

    UDPPacket(UDPPacket&& other) noexcept
        : bufferHandle(std::move(other.bufferHandle)),
        userID(other.userID),
        dataLength(other.dataLength) {}

    void setUserID(uint16_t id)
    {
        this->userID = id;
    }
    uint16_t getUserID()
    {
        assert(userID < MAX_ACTIVE_USERS && "userID cant be more than active users");

        return this->userID;
    }
    UDPPacket& operator=(UDPPacket&& other) noexcept {
        if (this != &other) {
            bufferHandle = std::move(other.bufferHandle);
            userID = other.userID;
            dataLength = other.dataLength;
        }
        return *this;
    }

    // Method to check validity of the packet
    bool isValid() const {
        return static_cast<bool>(bufferHandle) && dataLength > 0;
    }

    // Add assertion when accessing data
    const uint8_t* data() const {
        assert(dataLength != -1 && "Attempting to access data from invalid buffer handle");
        return bufferHandle ? bufferHandle.data() : nullptr;
    }

    uint8_t* data() {
        assert(dataLength != -1 && "Attempting to access data from invalid buffer handle");
        return bufferHandle ? bufferHandle.data() : nullptr;
    }

    // Disallow copying
    UDPPacket(const UDPPacket&) = delete;
    UDPPacket& operator=(const UDPPacket&) = delete;
};


enum class UDPPacketProcessFlag
{
    PACKET_DUPLICATE = -2,
    PACKET_OUT_OF_RANGE = -1,
    PACKET_PROCESS = 0
};



// For GCC, Clang, and compatible compilers
#if defined(__GNUC__) || defined(__clang__)
#define PACKED [[gnu::packed]]
// For MSVC
#elif defined(_MSC_VER)
#pragma pack(push, 1)
#define PACKED
#define NEEDS_PACK_POP
#else
    // For other compilers - fallback
#define PACKED
#endif

struct defaultHeader {
    uint8_t type : 2;
    uint8_t padding : 6;
} PACKED;
struct UnreliableHeader {
    uint16_t type : 2;
    uint16_t userID : 14;
} PACKED;

struct ReliableUnorderedHeader {
    uint16_t type : 2;
    uint16_t userID : 14;
    uint16_t packetID;
    uint8_t AckSize;
} PACKED;

struct ReliableOrderedHeader {
    uint16_t type : 2;
    uint16_t userID : 14;
    uint16_t packetID;
    uint8_t AckSize;
}PACKED;


#if defined(NEEDS_PACK_POP)
#pragma pack(pop)
#undef NEEDS_PACK_POP
#endif




class UserData {
private:
    uint16_t userID;
    std::array<uint8_t, 32> aesKey;
    SocketAddress lastAddress;

    // Receiving: Track last/next packet IDs for packets we've received
    uint16_t reliableUnorderedPacketIDRecv;

    uint16_t reliableUnorderedPacketIDSendBegin;
    uint16_t reliableUnorderedPacketIDSendEnd;


    uint16_t reliableOrderedPacketIDRecv;

    uint16_t reliableOrderedPacketIDSendBegin;
    uint16_t reliableOrderedPacketIDSendEnd;

    // Bitmaps to track received packets and prevent duplicates
    std::bitset<MAX_ACTIVE_PACKET> receivedUnorderedPackets;
    std::bitset<MAX_ACTIVE_PACKET> receivedOrderedPackets;

    std::array<uint16_t, MAX_ACTIVE_PACKET> pendingUnorderedAcks;
    std::array<uint16_t, MAX_ACTIVE_PACKET> pendingOrderedAcks;

    std::array<UDPPacket, MAX_ACTIVE_PACKET> reciviedUnorderedPacket;
    std::array<UDPPacket, MAX_ACTIVE_PACKET> reciviedOrderedPacket;


    // Helper method to check if a is newer than b (handling wraparound)
    bool isNewer(uint16_t a, uint16_t b) const {
        // Half the range of uint16_t
        constexpr uint16_t HALF_RANGE = 32768;

        // Standard sequence number comparison with wraparound handling
        return ((a > b) && (a - b <= HALF_RANGE)) ||
            ((a < b) && (b - a > HALF_RANGE));
    }

    // Helper method to calculate distance between sequence numbers (handling wraparound)
    uint16_t sequenceDistance(uint16_t a, uint16_t b) const {
        // Assert that we're not calculating distance to itself
        assert(a != b || a == 0 && "Attempting to calculate distance to the same sequence number");

        // If a >= b, straightforward calculation
        if (a >= b) {
            return a - b;
        }
        // If a < b, we need to account for wraparound
        else {
            return (65535 - b) + a + 1;
        }
    }

    bool isValidPacket(uint16_t packetID, uint16_t packet2ID) const
    {
        return isNewer(packetID, packet2ID) && sequenceDistance(packetID, packet2ID) < MAX_ACTIVE_PACKET;
    }
    // Get bitmap index for a packet ID relative to the base
    uint16_t getBitmapIndex(uint16_t packetID, uint16_t basePacketID) const {
        // Calculate how far ahead this packet is from the base
        uint16_t distance = sequenceDistance(packetID, basePacketID);

        // Assert that the distance is within the packet window
        assert(distance < MAX_ACTIVE_PACKET &&
            "Packet ID is too far ahead of base ID for bitmap tracking");

        return distance % MAX_ACTIVE_PACKET;
    }

    //when recieve push
    void setReliableUnorderedPacketIDRecv(uint16_t packetID)
    {
        assert(isNewer(packetID, reliableUnorderedPacketIDRecv) && " packet ID not newer");
        assert((sequenceDistance(packetID, reliableUnorderedPacketIDRecv) < MAX_ACTIVE_PACKET) && "Reliable Unordered Packet ID has overflown the active packet range");
        reliableUnorderedPacketIDRecv = packetID;

    }



    void setReliableOrderedPacketIDRecv(uint16_t packetID)
    {
        assert(isNewer(packetID, reliableOrderedPacketIDRecv) && " packet ID not newer");
        assert((sequenceDistance(packetID, reliableOrderedPacketIDRecv) < MAX_ACTIVE_PACKET) && "Reliable Ordered Packet ID has overflown active packet range");
        reliableOrderedPacketIDRecv = packetID;

    }

    //-------------------------------------------------------------------------
// RECEIVE SIDE METHODS (for packets we receive from the network)
//-------------------------------------------------------------------------
    uint16_t getReliableUnorderedPacketIDRecv() const
    {
        return reliableUnorderedPacketIDRecv;
    }

    uint16_t getReliableUnorderedPacketIDSendBegin() const
    {
        return reliableUnorderedPacketIDSendBegin;
    }
    uint16_t getReliableUnorderedPacketIDSendEnd() const
    {
        return reliableUnorderedPacketIDSendEnd;
    }


    uint16_t getReliableOrderedPacketIDRecv() const
    {
        return reliableOrderedPacketIDRecv;
    }

    uint16_t getReliableOrderedPacketIDSendBegin() const
    {
        return reliableOrderedPacketIDSendBegin;
    }
    uint16_t getReliableOrderedPacketIDSendEnd() const
    {
        return reliableOrderedPacketIDSendEnd;
    }



public:
    UserData() {}

    UserData(uint16_t id) : userID(id),
        // Initialize receiving trackers
        reliableUnorderedPacketIDRecv(0),
        reliableOrderedPacketIDRecv(0),
        // Initialize sending trackers
        reliableUnorderedPacketIDSendEnd(0),
        reliableOrderedPacketIDSendEnd(0),
        reliableUnorderedPacketIDSendBegin(0),
        reliableOrderedPacketIDSendBegin(0) {
        // No need to assert on id >= 0 since uint16_t is always >= 0
        assert(id < MAX_ACTIVE_USERS && "User ID exceeds maximum allowed active users");

        // Initialize bitmaps to 0 (no packets received)
        receivedUnorderedPackets.reset();
        receivedOrderedPackets.reset();
    }

    // User ID access
    uint16_t getUserID() const {
        return userID;
    }

    // AES key access
    const std::array<uint8_t, 32>& getAESKey() const {
        return aesKey;
    }

    void setAESKey(const std::array<uint8_t, 32>& key) {
        // Assert that the key contains valid data (non-zero)
        assert(std::any_of(key.begin(), key.end(), [](uint8_t b) { return b != 0; }) &&
            "AES key should contain at least some non-zero bytes");
        aesKey = key;
    }

    // Last address access
    const SocketAddress& getLastAddress() const {
        // Assert that the address is valid before returning
        assert((lastAddress.asSockAddr()->sa_family == AF_INET ||
            lastAddress.asSockAddr()->sa_family == AF_INET6) &&
            "Returning potentially invalid socket address");
        return lastAddress;
    }

    void setLastAddress(const SocketAddress& address)
    {
        // Assert that the address has a valid family (either IPv4 or IPv6)
        assert((address.asSockAddr()->sa_family == AF_INET ||
            address.asSockAddr()->sa_family == AF_INET6) &&
            "Invalid socket address family");

#ifndef NDEBUG
        // Assert that the address has a valid port based on its family
        if (address.asSockAddr()->sa_family == AF_INET) {
            assert(address.getPortV4() > 0 && "IPv4 socket address must have a valid port");
        }
        else if (address.asSockAddr()->sa_family == AF_INET6) {
            assert(address.getPortV6() > 0 && "IPv6 socket address must have a valid port");
        }
#endif
        lastAddress = address;
    }


    UDPPacketProcessFlag shouldProcessUnordered(uint16_t packetID)
    {

        //test duplicate
        if (receivedUnorderedPackets.test(packetID % MAX_ACTIVE_PACKET))
        {
            return UDPPacketProcessFlag::PACKET_DUPLICATE;
        }
        // test not out of range
        if (!isValidPacket(packetID, reliableUnorderedPacketIDRecv))
        {
            return UDPPacketProcessFlag::PACKET_OUT_OF_RANGE;
        }
        return UDPPacketProcessFlag::PACKET_PROCESS;

    }
    UDPPacketProcessFlag shouldProcessOrdered(uint16_t packetID)
    {

        //test duplicate
        if (receivedOrderedPackets.test(packetID % MAX_ACTIVE_PACKET))
        {
            return UDPPacketProcessFlag::PACKET_DUPLICATE;
        }
        // test not out of range
        if (!isValidPacket(packetID, reliableOrderedPacketIDRecv))
        {
            return UDPPacketProcessFlag::PACKET_OUT_OF_RANGE;
        }
        return UDPPacketProcessFlag::PACKET_PROCESS;

    }

    void setRecivedOrderedBit(uint16_t packetID)
    {
        assert(!receivedOrderedPackets.test(packetID % MAX_ACTIVE_PACKET) && " packetID Already Set");
        receivedOrderedPackets.set(packetID % MAX_ACTIVE_PACKET);
    }
    void setRecivedUnorderedBit(uint16_t packetID)
    {
        assert(!receivedUnorderedPackets.test(packetID % MAX_ACTIVE_PACKET) && "packetID Already Set");
        receivedUnorderedPackets.set(packetID % MAX_ACTIVE_PACKET);
    }

    void resetRecivedOrderedBit(uint16_t packetID)
    {
        assert(receivedOrderedPackets.test(packetID % MAX_ACTIVE_PACKET) && " packetID Not Set");
        receivedOrderedPackets.reset(packetID % MAX_ACTIVE_PACKET);
    }
    void resetRecivedUnorderedBit(uint16_t packetID)
    {
        assert(receivedUnorderedPackets.test(packetID % MAX_ACTIVE_PACKET) && " packetID Not Set");
        receivedUnorderedPackets.reset(packetID % MAX_ACTIVE_PACKET);
    }





    uint16_t getLastReliableOrderedPacketIDRecv() const {
        return reliableOrderedPacketIDRecv;
    }

    void setReciviedUnorderedPacket(UDPPacket&& packet)
    {

        assert(packet.getUserID() != -1 && packet.getUserID() < MAX_ACTIVE_USERS && "user is invalid");
        const uint16_t& temp = packet.getUserID() % MAX_ACTIVE_PACKET;
        reciviedUnorderedPacket[temp] = std::move(packet);
    }
    void setReciviedOrderedPacket(UDPPacket&& packet)
    {
        assert(packet.getUserID() != -1 && packet.getUserID() < MAX_ACTIVE_USERS && "user is invalid");
        const uint16_t& temp = packet.getUserID() % MAX_ACTIVE_PACKET;
        reciviedOrderedPacket[temp] = std::move(packet);
    }
    void setRecievedUnorderedPacket(UDPPacket&& packet)
    {
        assert(packet.getUserID() != -1 && packet.getUserID() < MAX_ACTIVE_USERS && "user is invalid");
        const uint16_t& temp = packet.getUserID() % MAX_ACTIVE_PACKET;
        reciviedUnorderedPacket[temp] = std::move(packet);
    }

    bool isOrderedPacketReadyForProcess(uint16_t packetID)
    {
        if (reliableOrderedPacketIDRecv == packetID)
        {
            resetRecivedUnorderedBit(packetID);
            reliableOrderedPacketIDRecv++;
            return true;
        }
        return false;
    }

};



// UDP communication class - unified class for all UDP types
class UDPNode {
public:



    SocketAddress boundAddress;

private:
    SocketHandle socketHandle;

    FixedBufferPool bufferPool;
    int packetLossPercentage;
    std::mt19937 rng;

    std::array<UserData, MAX_ACTIVE_USERS> users;
    std::bitset<MAX_ACTIVE_USERS> activeUsers;
public:
    UDPNode() : socketHandle(INVALID_SOCKET_HANDLE), packetLossPercentage(0), bufferPool(MAX_PACKET_BUFFER_SIZE),
        rng(std::random_device{}()) {
        // Initialize buffer pool

        // Initialize network on Windows
#ifdef _WIN32
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
    }

    ~UDPNode() {
        shutdown();

#ifdef _WIN32
        WSACleanup();
#endif
    }

    // Initialize socket
    bool initIpv4(const std::string& address = "0.0.0.0", uint16_t port = 0) {
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

    bool initIpv6(const std::string& address = "::", uint16_t port = 0) {
        // Create socket
        socketHandle = socket(AF_INET6, SOCK_DGRAM, IPPROTO_UDP);
        if (socketHandle == INVALID_SOCKET_HANDLE) {
            return false;
        }

        // Set up the socket address structure
        sockaddr_in6 serverAddress{};
        serverAddress.sin6_family = AF_INET6;
        serverAddress.sin6_port = htons(port);

        // Convert the address string to a binary IPv6 address
        if (inet_pton(AF_INET6, address.c_str(), &serverAddress.sin6_addr) <= 0) {
            closesocket(socketHandle); // Clean up the socket if conversion fails
            return false;
        }


        // Bind the socket to the address and port
        if (bind(socketHandle, reinterpret_cast<sockaddr*>(&serverAddress), sizeof(serverAddress)) == SOCKET_ERROR) {
            closesocket(socketHandle); // Clean up the socket if binding fails
            return false;
        }
        boundAddress.setIPv6(address, port);


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
    inline bool isValid() const {
        return socketHandle != INVALID_SOCKET_HANDLE;
    }



    // ProcessPacket after reciving
    int PreProcessPacket(UDPPacket& packet)
    {
        if (packet.dataLength < sizeof(defaultHeader))
        {
            return false;
        }

        const defaultHeader& baseHeader = *(defaultHeader*)packet.data();

        switch (baseHeader.type)
        {
        case 0:
            return 0;
        case 1:
            return packet.dataLength >= sizeof(UnreliableHeader);
            return 1;
        case 2:

            if (packet.dataLength >= sizeof(ReliableUnorderedHeader))
            {
                const ReliableUnorderedHeader& header = *(ReliableUnorderedHeader*)packet.data();

                UserData& user = users[header.userID];
                if (user.shouldProcessUnordered(header.packetID) == UDPPacketProcessFlag::PACKET_PROCESS)
                {
                    packet.setUserID(user.getUserID());

                    user.setRecivedUnorderedBit(header.packetID);

                    user.setRecievedUnorderedPacket(std::move(packet));
                    return 2;
                }
            }
        case 3:

            if (packet.dataLength >= sizeof(ReliableOrderedHeader))
            {
                const ReliableOrderedHeader& header = *(ReliableOrderedHeader*)packet.data();


                UserData& user = users[header.userID];


                if (user.shouldProcessOrdered(header.packetID) == UDPPacketProcessFlag::PACKET_PROCESS)
                {
                    // set bit test
                    user.setRecivedOrderedBit(header.packetID);
                    packet.setUserID(user.getUserID());

                    user.setReciviedOrderedPacket(std::move(packet));

                    //
                    return 3;
                }
            }
        }
        return -1;
    }


    // Receive a packet from the socket
    int receive(UDPPacket& packet) {


        auto buffer = bufferPool.getBuffer();



        SocketAddress sender;
        socklen_t addrLen = sender.getSize();

        int bytesReceived = recvfrom(socketHandle,
            reinterpret_cast<char*>(buffer.data()),
            MAX_PACKET_BUFFER_SIZE,
            0,
            sender.asSockAddr(),
            &addrLen);

        if (bytesReceived > 0) {
            packet = std::move(buffer);
            packet.dataLength = bytesReceived;
            int type = PreProcessPacket(packet);

            if (type == 3)
            {
                UserData& user = this->getUser(packet.userID);
                if (user.isOrderedPacketReadyForProcess(packet.getUserID()))
                {
                    return bytesReceived;
                }

            }
            else if (type == -1)
            {
                return 0;
            }
        }

        return bytesReceived;
    }

    // Wait for data with timeout
    int waitForData(int timeoutMs) {



        fd_set readSet;
        FD_ZERO(&readSet);
        FD_SET(socketHandle, &readSet);

        timeval timeout;
        timeout.tv_sec = timeoutMs / 1000;
        timeout.tv_usec = (timeoutMs % 1000) * 1000;

        return select(socketHandle + 1, &readSet, nullptr, nullptr, &timeout);
    }

    // Add a user
    void addUser(uint16_t userID, SocketAddress address, const std::array<uint8_t, 32>& aesKey) {


        UserData userData(userID);
        userData.setLastAddress(address);
        userData.setAESKey(aesKey);
        users[userID] = std::move(userData);
        assert(!activeUsers.test(userID) && userID < MAX_ACTIVE_USERS && " OUT OF RANGE");
        activeUsers.set(userID);
    }
    // Add a user
    void addUser(const uint16_t userID, SocketAddress address) {

        UserData userData(userID);
        users[userID] = std::move(userData);
        userData.setLastAddress(address);
        assert(!activeUsers.test(userID) && userID < MAX_ACTIVE_USERS && " OUT OF RANGE");
        activeUsers.set(userID);

    }

    // Remove a user
    void removeUser(const uint16_t userID) {

        assert(activeUsers.test(userID) && userID < MAX_ACTIVE_USERS && " OUT OF RANGE");
        activeUsers.reset(userID);
    }

    // Check if a user is active
    bool hasActiveUser(uint16_t userID) const {
        assert(userID < MAX_ACTIVE_USERS && " OUT OF RANGE");

        return activeUsers.test(userID);
    }

    // Get user data
    UserData& getUser(const uint16_t userID) {

        return users.at(userID);
    }
    // Get user data
    const UserData& getUser(const uint16_t userID) const {


        return users.at(userID);
    }

    bool receivePacket(UDPPacket& packet)
    {
        int bytes = receive(packet);


    }

    void execute()
    {
        // send pending packet


    }


    void prepareDefault()
    {
    }


    // Get user's AES key
    std::array<uint8_t, 32>& getUserAESKey(uint16_t userID) {

    }

    // Simulate packet loss (for testing)
    void simulatePacketLoss(int percentage) {
        packetLossPercentage = std::max(0, std::min(100, percentage));
    }

    // Check if packet should be dropped based on configured loss rate
    bool shouldDropPacket() {
        if (packetLossPercentage <= 0) return false;
        if (packetLossPercentage >= 100) return true;

        std::uniform_int_distribution<> dis(1, 100);
        return dis(rng) <= packetLossPercentage;
    }




    // === UNRELIABLE UDP FUNCTIONALITY ===

    // Prepare unreliable packet
    int prepareUnreliablePacket(uint16_t userID, UDPPacket&& packet) {


    }

    // Send an unreliable packet
    int sendUnreliablePacket(UDPPacket&& packet) {

    }

    // Process an unreliable packet (when received)
    bool processUnreliablePacket(const UDPPacket& packet) {

    }

    // === RELIABLE UNORDERED UDP FUNCTIONALITY ===

    // Prepare reliable unordered packet
    UDPPacket prepareReliableUnorderedPacket(uint16_t userID) {

    }

    // Send a reliable unordered packet
    int sendReliableUnorderedPacket(UDPPacket&& packet) {

    }

    // Process a reliable unordered packet (when received)
    bool processReliableUnorderedPacket(const UDPPacket& packet) {

        return true;
    }

    // === RELIABLE ORDERED UDP FUNCTIONALITY ===

    // Prepare reliable ordered packet
    UDPPacket prepareReliableOrderedPacket(uint16_t userID) {

    }

    // Send a reliable ordered packet
    int sendReliableOrderedPacket(UDPPacket&& packet) {

    }

    // Process a reliable ordered packet (when received)
    bool processReliableOrderedPacket(const UDPPacket& packet) {


        return true;
    }
    // Receive a packet from the socket
    bool receivePacket(UDPPacket& packet, SocketAddress& sender) {
        auto buffer = bufferPool.getBuffer();
        if (!buffer) {
            return false;  // No buffer available
        }

        socklen_t addrLen = sender.getSize();
        int bytesReceived = recvfrom(socketHandle,
            reinterpret_cast<char*>(buffer.data()),
            MAX_PACKET_BUFFER_SIZE,
            0,
            sender.asSockAddr(),
            &addrLen);

        if (bytesReceived > 0) {
            packet = UDPPacket(std::move(buffer));
            packet.dataLength = bytesReceived;
            return true;
        }

        return false;
    }
    // Prepare a default UDP packet with data
    UDPPacket prepareDefaultPacket() {
        auto buffer = bufferPool.getBuffer();

        UDPPacket packet;

        if (buffer) {
            defaultHeader& header = *(defaultHeader*)buffer.data();
            header = {};
            // Return an invalid packet if no buffer is available
            packet = std::move(buffer);
            packet.dataLength = 1;
        }

        return packet;
    }
    // Send a packet directly to a destination
    int sendDefaultPacket(const UDPPacket& packet, const SocketAddress& destAddr) {
        if (!packet.isValid() || shouldDropPacket()) {
            return 0;
        }

        return sendto(socketHandle,
            reinterpret_cast<const char*>(packet.data()),
            packet.dataLength,
            0,
            destAddr.asSockAddr(),
            destAddr.getSize());
    }
    // Send packet to a known user
    int sendToUser(uint16_t userID, const UDPPacket& packet) {

        assert(hasActiveUser(userID) && "dont have active user");

        const UserData& user = getUser(userID);
        return sendDefaultPacket(packet, user.getLastAddress());
    }
};

#endif // ATM_NETWORK_H
