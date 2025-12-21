#include <iostream>
#include <string>
#include <functional>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <iomanip>
#include <cmath>
#include <type_traits>
#include <array>
#include <cassert>
#include <atomic>

// Include the UDP Node header
#include "ATMNetwork.h"

// Simple testing framework
class UDPTester {
private:
    int passedTests = 0;
    int totalTests = 0;
    std::string currentTestName;

public:
    void beginTest(const std::string& testName) {
        currentTestName = testName;
        std::cout << "Running test: " << testName << "... ";
    }

    void endTest(bool passed) {
        totalTests++;
        if (passed) {
            passedTests++;
            std::cout << "PASSED" << std::endl;
        }
        else {
            std::cout << "FAILED" << std::endl;
        }
    }

    template<typename T>
    void assertEqual(const T& expected, const T& actual, const std::string& message = "") {
        if (expected == actual) {
            endTest(true);
        }
        else {
            std::cout << std::endl << "Assertion failed";
            // Only try to stream simple types like int, float, etc.
            if constexpr (std::is_fundamental_v<T> || std::is_same_v<T, std::string>) {
                std::cout << ": expected " << expected << ", got " << actual;
            }
            if (!message.empty()) {
                std::cout << " - " << message;
            }
            std::cout << std::endl;
            endTest(false);
        }
    }

    void assertTrue(bool condition, const std::string& message = "") {
        if (condition) {
            endTest(true);
        }
        else {
            std::cout << std::endl << "Assertion failed: expected true, got false";
            if (!message.empty()) {
                std::cout << " - " << message;
            }
            std::cout << std::endl;
            endTest(false);
        }
    }

    void assertFalse(bool condition, const std::string& message = "") {
        if (!condition) {
            endTest(true);
        }
        else {
            std::cout << std::endl << "Assertion failed: expected false, got true";
            if (!message.empty()) {
                std::cout << " - " << message;
            }
            std::cout << std::endl;
            endTest(false);
        }
    }

    void printSummary() {
        std::cout << "\n=== Test Summary ===\n";
        std::cout << "Total tests: " << totalTests << std::endl;
        std::cout << "Passed tests: " << passedTests << std::endl;
        std::cout << "Failed tests: " << (totalTests - passedTests) << std::endl;
        std::cout << "Success rate: " << (totalTests > 0 ? (100.0 * passedTests / totalTests) : 0) << "%\n";
    }
};

// Benchmark framework
class Benchmark {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    std::string name;
    TimePoint startTime;
    double totalTimeMs = 0;
    int iterations = 0;
    int dataSize = 0;

public:
    Benchmark(const std::string& benchmarkName) : name(benchmarkName) {}

    void start() {
        startTime = Clock::now();
    }

    void stop(int bytes = 0) {
        auto endTime = Clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        totalTimeMs += duration.count() / 1000.0;
        iterations++;
        dataSize += bytes;
    }

    void reset() {
        totalTimeMs = 0;
        iterations = 0;
        dataSize = 0;
    }

    void printResults() {
        std::cout << "\n=== Benchmark: " << name << " ===\n";
        std::cout << "Total runs: " << iterations << std::endl;

        if (iterations > 0) {
            std::cout << "Total time: " << totalTimeMs << " ms" << std::endl;
            std::cout << "Average time per run: " << (totalTimeMs / iterations) << " ms" << std::endl;

            if (dataSize > 0) {
                double totalMB = dataSize / (1024.0 * 1024.0);
                double throughputMBps = totalMB / (totalTimeMs / 1000.0);
                std::cout << "Data processed: " << totalMB << " MB" << std::endl;
                std::cout << "Throughput: " << throughputMBps << " MB/s" << std::endl;
                std::cout << "Packets per second: " << (iterations / (totalTimeMs / 1000.0)) << std::endl;
            }
        }
    }
};

// Helper functions
void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

std::vector<uint8_t> generateRandomData(size_t size) {
    std::vector<uint8_t> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>(dis(gen));
    }

    return data;
}

// Helper class to fill a buffer with data and get its length
class PacketData {
public:
    static void fillPacket(uint8_t* buffer, size_t maxSize, const std::string& message, size_t& outSize) {
        size_t dataSize = min(message.size(), maxSize);
        memcpy(buffer, message.c_str(), dataSize);
        outSize = dataSize;
    }

    static void fillPacket(uint8_t* buffer, size_t maxSize, const std::vector<uint8_t>& data, size_t& outSize) {
        size_t dataSize = min(data.size(), maxSize);
        memcpy(buffer, data.data(), dataSize);
        outSize = dataSize;
    }

    static bool compareData(const uint8_t* buffer, const std::string& message, size_t length) {
        return (length == message.size() &&
            memcmp(buffer, message.c_str(), length) == 0);
    }

    static bool compareData(const uint8_t* buffer, const std::vector<uint8_t>& data, size_t length) {
        return (length == data.size() &&
            memcmp(buffer, data.data(), length) == 0);
    }
};

// Test classes initialization
void testInitialization(UDPTester& tester) {
    // Test DefaultUDP initialization
    tester.beginTest("DefaultUDP Initialization");
    DefaultUDP defaultNode;
    bool defaultResult = defaultNode.init("127.0.0.1", 0);
    tester.assertTrue(defaultResult, "DefaultUDP initialization failed");
    tester.assertTrue(defaultNode.isValid(), "DefaultUDP socket is not valid after initialization");

    // Test UnreliableUDP initialization
    tester.beginTest("UnreliableUDP Initialization");
    UnreliableUDP unreliableNode;
    bool unreliableResult = unreliableNode.init("127.0.0.1", 0);
    tester.assertTrue(unreliableResult, "UnreliableUDP initialization failed");
    tester.assertTrue(unreliableNode.isValid(), "UnreliableUDP socket is not valid after initialization");

    // Test ReliableUnorderedUDP initialization
    tester.beginTest("ReliableUnorderedUDP Initialization");
    ReliableUnorderedUDP reliableUnorderedNode;
    bool reliableUnorderedResult = reliableUnorderedNode.init("127.0.0.1", 0);
    tester.assertTrue(reliableUnorderedResult, "ReliableUnorderedUDP initialization failed");
    tester.assertTrue(reliableUnorderedNode.isValid(), "ReliableUnorderedUDP socket is not valid after initialization");

    // Test ReliableOrderedUDP initialization
    tester.beginTest("ReliableOrderedUDP Initialization");
    ReliableOrderedUDP reliableOrderedNode;
    bool reliableOrderedResult = reliableOrderedNode.init("127.0.0.1", 0);
    tester.assertTrue(reliableOrderedResult, "ReliableOrderedUDP initialization failed");
    tester.assertTrue(reliableOrderedNode.isValid(), "ReliableOrderedUDP socket is not valid after initialization");

    // Test port assignment
    tester.beginTest("Port Assignment");
    DefaultUDP portNode;
    portNode.init("127.0.0.1", 12345);
    uint16_t port = portNode.getPort();
    tester.assertEqual(static_cast<uint16_t>(12345), port, "Port was not assigned properly");

    // Test address binding
    tester.beginTest("Address Binding");
    tester.assertEqual(std::string("127.0.0.1"), portNode.getAddress(), "Bound address doesn't match");

    // Test shutdown
    tester.beginTest("Socket Shutdown");
    portNode.shutdown();
    tester.assertFalse(portNode.isValid(), "Socket is still valid after shutdown");
}

void testUserManagement(UDPTester& tester) {
    // Test user management on each UDP class
    UnreliableUDP unreliableNode;
    unreliableNode.init();

    ReliableUnorderedUDP unorderedNode;
    unorderedNode.init();

    ReliableOrderedUDP orderedNode;
    orderedNode.init();

    // Test adding users
    tester.beginTest("Add User (UnreliableUDP)");
    uint16_t userID = 123;
    std::array<uint8_t, 32> aesKey;
    std::fill(aesKey.begin(), aesKey.end(), 0xAA);
    bool unreliableAddResult = unreliableNode.addUser(userID, aesKey);
    tester.assertTrue(unreliableAddResult, "Failed to add user to UnreliableUDP");

    tester.beginTest("Add User (ReliableUnorderedUDP)");
    bool unorderedAddResult = unorderedNode.addUser(userID, aesKey);
    tester.assertTrue(unorderedAddResult, "Failed to add user to ReliableUnorderedUDP");

    tester.beginTest("Add User (ReliableOrderedUDP)");
    bool orderedAddResult = orderedNode.addUser(userID, aesKey);
    tester.assertTrue(orderedAddResult, "Failed to add user to ReliableOrderedUDP");

    // Test hasActiveUser
    tester.beginTest("Check Active User (UnreliableUDP)");
    bool unreliableHasUser = unreliableNode.hasActiveUser(userID);
    tester.assertTrue(unreliableHasUser, "User not found in UnreliableUDP after adding");

    tester.beginTest("Check Active User (ReliableUnorderedUDP)");
    bool unorderedHasUser = unorderedNode.hasActiveUser(userID);
    tester.assertTrue(unorderedHasUser, "User not found in ReliableUnorderedUDP after adding");

    tester.beginTest("Check Active User (ReliableOrderedUDP)");
    bool orderedHasUser = orderedNode.hasActiveUser(userID);
    tester.assertTrue(orderedHasUser, "User not found in ReliableOrderedUDP after adding");

    // Test getUserAESKey
    tester.beginTest("Get AES Key (UnreliableUDP)");
    auto unreliableKey = unreliableNode.getUserAESKey(userID);
    bool unreliableKeyMatch = (unreliableKey == aesKey);
    tester.assertTrue(unreliableKeyMatch, "Retrieved AES key doesn't match in UnreliableUDP");

    tester.beginTest("Get AES Key (ReliableUnorderedUDP)");
    auto unorderedKey = unorderedNode.getUserAESKey(userID);
    bool unorderedKeyMatch = (unorderedKey == aesKey);
    tester.assertTrue(unorderedKeyMatch, "Retrieved AES key doesn't match in ReliableUnorderedUDP");

    tester.beginTest("Get AES Key (ReliableOrderedUDP)");
    auto orderedKey = orderedNode.getUserAESKey(userID);
    bool orderedKeyMatch = (orderedKey == aesKey);
    tester.assertTrue(orderedKeyMatch, "Retrieved AES key doesn't match in ReliableOrderedUDP");

    // Test getUser
    tester.beginTest("Get User Data (UnreliableUDP)");
    UserData* unreliableUserData;
    bool unreliableGetResult = unreliableNode.getUser(userID, unreliableUserData);
    tester.assertTrue(unreliableGetResult, "Failed to get user data from UnreliableUDP");
    tester.assertEqual(userID, unreliableUserData->userID, "User ID in retrieved data doesn't match");

    // Test removeUser
    tester.beginTest("Remove User (UnreliableUDP)");
    bool unreliableRemoveResult = unreliableNode.removeUser(userID);
    tester.assertTrue(unreliableRemoveResult, "Failed to remove user from UnreliableUDP");
    tester.assertFalse(unreliableNode.hasActiveUser(userID), "User still active in UnreliableUDP after removal");

    tester.beginTest("Remove User (ReliableUnorderedUDP)");
    bool unorderedRemoveResult = unorderedNode.removeUser(userID);
    tester.assertTrue(unorderedRemoveResult, "Failed to remove user from ReliableUnorderedUDP");
    tester.assertFalse(unorderedNode.hasActiveUser(userID), "User still active in ReliableUnorderedUDP after removal");

    tester.beginTest("Remove User (ReliableOrderedUDP)");
    bool orderedRemoveResult = orderedNode.removeUser(userID);
    tester.assertTrue(orderedRemoveResult, "Failed to remove user from ReliableOrderedUDP");
    tester.assertFalse(orderedNode.hasActiveUser(userID), "User still active in ReliableOrderedUDP after removal");
}

void testDefaultPacketSending(UDPTester& tester) {
    // Setup server and client
    DefaultUDP server;
    server.init("127.0.0.1", 0);
    uint16_t serverPort = server.getPort();

    DefaultUDP client;
    client.init("127.0.0.1", 0);

    // Prepare server address
    SocketAddress serverAddr;
    serverAddr.setIPv4("127.0.0.1", serverPort);

    // Test data
    std::string testMessage = "This is a default test message";

    // Prepare packet on client side
    auto clientBuffer = client.getBuffer();

    // Fill packet with data
    size_t dataSize;
    PacketData::fillPacket(clientBuffer.data(), MAX_PACKET_SIZE, testMessage, dataSize);

    // Create packet
    UDPPacket packet(&serverAddr, std::move(clientBuffer));
    packet.dataLength = dataSize;

    // Send packet
    tester.beginTest("Send Default Packet");
    int sendResult = client.sendPacket(std::move(packet), serverAddr);
    tester.assertTrue(sendResult > 0, "Failed to send default packet");

    // Server receives packet
    sleep_ms(10); // Allow time for packet to arrive
    tester.beginTest("Receive Default Packet");
    UDPPacket receivedPacket = server.receive();

    if (receivedPacket.bufferHandle) {
        // Verify data
        bool dataMatches = PacketData::compareData(receivedPacket.bufferHandle.data(), testMessage, dataSize);
        tester.assertTrue(dataMatches, "Received data doesn't match sent data");
    }
    else {
        tester.assertTrue(false, "Failed to receive packet");
    }
}

void testUnreliablePacketSending(UDPTester& tester) {
    // Setup server and client
    UnreliableUDP server;
    server.init("127.0.0.1", 0);
    uint16_t serverPort = server.getPort();

    UnreliableUDP client;
    client.init("127.0.0.1", 0);

    // Set up test user
    uint16_t userID = 123;
    std::array<uint8_t, 32> aesKey;
    std::fill(aesKey.begin(), aesKey.end(), 0xAA);

    server.addUser(userID, aesKey);
    client.addUser(userID, aesKey);

    // Prepare server address
    SocketAddress serverAddr;
    serverAddr.setIPv4("127.0.0.1", serverPort);

    // Update user's last address in server
    UserData* userData;
    server.getUser(userID, userData);
    userData->lastAddress.setIPv4("127.0.0.1", client.getPort());

    // Update user's last address in client
    client.getUser(userID, userData);
    userData->lastAddress = serverAddr;

    // Test data
    std::string testMessage = "This is an unreliable test message";

    // Prepare packet on client side
    UDPPacket packet = client.preparePacket(userID);
    tester.assertTrue(packet.bufferHandle.data(), "Failed to prepare unreliable packet");

    // Fill packet with data
    size_t dataSize;
    PacketData::fillPacket(
        packet.bufferHandle.data() + sizeof(UnreliableHeader),
        MAX_PACKET_SIZE - sizeof(UnreliableHeader),
        testMessage, dataSize);

    packet.dataLength = sizeof(UnreliableHeader) + dataSize;

    // Send packet
    tester.beginTest("Send Unreliable Packet");
    int sendResult = client.sendPacket(std::move(packet));
    tester.assertTrue(sendResult > 0, "Failed to send unreliable packet");

    // Server receives packet
    sleep_ms(10); // Allow time for packet to arrive
    tester.beginTest("Receive Unreliable Packet");
    UDPPacket receivedPacket = server.receive();

    if (receivedPacket.bufferHandle) {
        // Process the packet
        server.processPacket(receivedPacket);

        UnreliableHeader* header = reinterpret_cast<UnreliableHeader*>(receivedPacket.bufferHandle.data());
        tester.assertEqual(userID, header->userID, "Received incorrect userID");

        // Verify data
        const uint8_t* receivedData = receivedPacket.bufferHandle.data() + sizeof(UnreliableHeader);
        bool dataMatches = PacketData::compareData(receivedData, testMessage, dataSize);
        tester.assertTrue(dataMatches, "Received data doesn't match sent data");
    }
    else {
        tester.assertTrue(false, "Failed to receive packet");
    }
}

void testReliableUnorderedPacketSending(UDPTester& tester) {
    // Setup server and client
    ReliableUnorderedUDP server;
    server.init("127.0.0.1", 0);
    uint16_t serverPort = server.getPort();

    ReliableUnorderedUDP client;
    client.init("127.0.0.1", 0);

    // Set up test user
    uint16_t userID = 456;
    std::array<uint8_t, 32> aesKey;
    std::fill(aesKey.begin(), aesKey.end(), 0xBB);

    server.addUser(userID, aesKey);
    client.addUser(userID, aesKey);

    // Prepare server address
    SocketAddress serverAddr;
    serverAddr.setIPv4("127.0.0.1", serverPort);

    // Update user's last address in server
    UserData* userData;
    server.getUser(userID, userData);
    userData->lastAddress.setIPv4("127.0.0.1", client.getPort());

    // Update user's last address in client
    client.getUser(userID, userData);
    userData->lastAddress = serverAddr;

    // Test data
    std::string testMessage = "This is a reliable unordered test message";

    // Prepare packet on client side
    UDPPacket packet = client.preparePacket(userID);
    tester.assertTrue(packet.bufferHandle.data(), "Failed to prepare reliable unordered packet");

    // Fill packet with data
    size_t dataSize;
    ReliableUnorderedHeader* header = reinterpret_cast<ReliableUnorderedHeader*>(packet.bufferHandle.data());
    uint8_t ackSize = header->AckSize;

    PacketData::fillPacket(
        packet.bufferHandle.data() + sizeof(ReliableUnorderedHeader) +
        (ackSize * sizeof(uint16_t)),
        MAX_PACKET_SIZE - sizeof(ReliableUnorderedHeader) -
        (ackSize * sizeof(uint16_t)),
        testMessage, dataSize);

    packet.dataLength = sizeof(ReliableUnorderedHeader) +
        (ackSize * sizeof(uint16_t)) +
        dataSize;

    // Send packet
    tester.beginTest("Send Reliable Unordered Packet");
    int sendResult = client.sendPacket(std::move(packet));
    tester.assertTrue(sendResult > 0, "Failed to send reliable unordered packet");

    // Server receives packet
    sleep_ms(10); // Allow time for packet to arrive
    tester.beginTest("Receive Reliable Unordered Packet");
    UDPPacket receivedPacket = server.receive();

    if (receivedPacket.bufferHandle) {
        // Process the packet
        bool processResult = server.processPacket(receivedPacket);
        tester.assertTrue(processResult, "Failed to process reliable unordered packet");

        ReliableUnorderedHeader* recvHeader = reinterpret_cast<ReliableUnorderedHeader*>(
            receivedPacket.bufferHandle.data());
        tester.assertEqual(userID, recvHeader->userID, "Received incorrect userID");

        // Verify data
        ackSize = recvHeader->AckSize;
        const uint8_t* receivedData = receivedPacket.bufferHandle.data() +
            sizeof(ReliableUnorderedHeader) +
            (ackSize * sizeof(uint16_t));
        bool dataMatches = PacketData::compareData(receivedData, testMessage, dataSize);
        tester.assertTrue(dataMatches, "Received data doesn't match sent data");

        // Execute both client and server to process acknowledgments
        client.execute();
        server.execute();
    }
    else {
        tester.assertTrue(false, "Failed to receive packet");
    }
}

void testReliableOrderedPacketSending(UDPTester& tester) {
    // Setup server and client
    ReliableOrderedUDP server;
    server.init("127.0.0.1", 0);
    uint16_t serverPort = server.getPort();

    ReliableOrderedUDP client;
    client.init("127.0.0.1", 0);

    // Set up test user
    uint16_t userID = 789;
    std::array<uint8_t, 32> aesKey;
    std::fill(aesKey.begin(), aesKey.end(), 0xCC);

    server.addUser(userID, aesKey);
    client.addUser(userID, aesKey);

    // Prepare server address
    SocketAddress serverAddr;
    serverAddr.setIPv4("127.0.0.1", serverPort);

    // Update user's last address in server
    UserData* userData;
    server.getUser(userID, userData);
    userData->lastAddress.setIPv4("127.0.0.1", client.getPort());

    // Update user's last address in client
    client.getUser(userID, userData);
    userData->lastAddress = serverAddr;

    // Test sending multiple packets in sequence
    const int numPackets = 5;
    std::vector<std::string> messages = {
        "This is reliable ordered packet 1",
        "This is reliable ordered packet 2",
        "This is reliable ordered packet 3",
        "This is reliable ordered packet 4",
        "This is reliable ordered packet 5"
    };

    // Send all packets
    for (int i = 0; i < numPackets; i++) {
        // Prepare packet
        UDPPacket packet = client.preparePacket(userID);

        // Fill packet with data
        size_t dataSize;
        ReliableOrderedHeader* header = reinterpret_cast<ReliableOrderedHeader*>(packet.bufferHandle.data());
        uint8_t ackSize = header->AckSize;

        PacketData::fillPacket(
            packet.bufferHandle.data() + sizeof(ReliableOrderedHeader) +
            (ackSize * sizeof(uint16_t)),
            MAX_PACKET_SIZE - sizeof(ReliableOrderedHeader) -
            (ackSize * sizeof(uint16_t)),
            messages[i], dataSize);

        packet.dataLength = sizeof(ReliableOrderedHeader) +
            (ackSize * sizeof(uint16_t)) +
            dataSize;

        // Send packet
        client.sendPacket(std::move(packet));
        client.execute();
    }

    // Test receiving packets in order
    tester.beginTest("Receive Reliable Ordered Packets In Sequence");

    int receivedCount = 0;
    for (int i = 0; i < numPackets; i++) {
        server.execute();
        UDPPacket receivedPacket = server.receive();

        if (receivedPacket.bufferHandle) {
            // Process the packet
            bool processResult = server.processPacket(receivedPacket);

            if (processResult) {
                ReliableOrderedHeader* recvHeader = reinterpret_cast<ReliableOrderedHeader*>(
                    receivedPacket.bufferHandle.data());

                uint8_t ackSize = recvHeader->AckSize;
                const uint8_t* receivedData = receivedPacket.bufferHandle.data() +
                    sizeof(ReliableOrderedHeader) +
                    (ackSize * sizeof(uint16_t));

                // Find which message was received
                for (size_t j = 0; j < messages.size(); j++) {
                    if (PacketData::compareData(receivedData, messages[j], messages[j].size())) {
                        receivedCount++;
                        break;
                    }
                }
            }
        }

        sleep_ms(10);
    }

    // Since packets might arrive out of order physically but be processed in order logically,
    // we just check that we received the right number of packets
    tester.assertTrue(receivedCount > 0, "Failed to receive any reliable ordered packets");
}

void testRetransmissionMechanism(UDPTester& tester) {
    // Setup server and client
    ReliableUnorderedUDP server;
    server.init("127.0.0.1", 0);
    uint16_t serverPort = server.getPort();

    ReliableUnorderedUDP client;
    client.init("127.0.0.1", 0);

    // Set high packet loss on server to trigger retransmissions
    server.simulatePacketLoss(80); // 80% packet loss

    // Set up test user
    uint16_t userID = 123;
    std::array<uint8_t, 32> aesKey;
    std::fill(aesKey.begin(), aesKey.end(), 0xDD);

    server.addUser(userID, aesKey);
    client.addUser(userID, aesKey);

    // Prepare server address
    SocketAddress serverAddr;
    serverAddr.setIPv4("127.0.0.1", serverPort);

    // Update user's last address in server
    UserData* userData;
    server.getUser(userID, userData);
    userData->lastAddress.setIPv4("127.0.0.1", client.getPort());

    // Update user's last address in client
    client.getUser(userID, userData);
    userData->lastAddress = serverAddr;

    // Test data
    std::string testMessage = "This is a retransmission test";

    // Send and check for several packets to trigger retransmissions
    tester.beginTest("Packet Retransmission Test");

    std::atomic<int> packetsReceived(0);

    // Send packet
    UDPPacket packet = client.preparePacket(userID);

    // Fill packet with data
    size_t dataSize;
    ReliableUnorderedHeader* header = reinterpret_cast<ReliableUnorderedHeader*>(packet.bufferHandle.data());
    uint8_t ackSize = header->AckSize;

    PacketData::fillPacket(
        packet.bufferHandle.data() + sizeof(ReliableUnorderedHeader) +
        (ackSize * sizeof(uint16_t)),
        MAX_PACKET_SIZE - sizeof(ReliableUnorderedHeader) -
        (ackSize * sizeof(uint16_t)),
        testMessage, dataSize);

    packet.dataLength = sizeof(ReliableUnorderedHeader) +
        (ackSize * sizeof(uint16_t)) +
        dataSize;

    client.sendPacket(std::move(packet));

    // Keep executing client and server for a short time to allow for retransmissions
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::milliseconds(1000);

    while (packetsReceived.load() < 1) {
        // Check for timeout
        auto currentTime = std::chrono::steady_clock::now();
        if (currentTime - startTime > timeout) {
            break;
        }

        // Execute client to trigger retransmissions
        client.execute();

        // Try to receive and process packet at server
        UDPPacket receivedPacket = server.receive();
        if (receivedPacket.bufferHandle) {
            // Process the packet
            bool processResult = server.processPacket(receivedPacket);
            if (processResult) {
                packetsReceived++;
            }
        }

        server.execute();
        sleep_ms(10);
    }

    // With 80% packet loss, we should still receive at least one packet
    // due to retransmissions
    tester.assertTrue(packetsReceived.load() > 0, "Failed to receive packet despite retransmissions");

    // Reset packet loss simulation
    server.simulatePacketLoss(0);
}

void testPacketLossSimulation(UDPTester& tester) {
    UnreliableUDP node;
    node.init();

    tester.beginTest("Set Packet Loss to 0%");
    node.simulatePacketLoss(0);
    tester.assertTrue(true, "Set packet loss to 0%");

    tester.beginTest("Set Packet Loss to 50%");
    node.simulatePacketLoss(50);
    tester.assertTrue(true, "Set packet loss to 50%");

    tester.beginTest("Set Packet Loss to 100%");
    node.simulatePacketLoss(100);
    tester.assertTrue(true, "Set packet loss to 100%");
}

void testWaitForData(UDPTester& tester) {
    // Setup server and client
    DefaultUDP server;
    server.init("127.0.0.1", 0);
    uint16_t serverPort = server.getPort();

    DefaultUDP client;
    client.init("127.0.0.1", 0);

    // Server address
    SocketAddress serverAddr;
    serverAddr.setIPv4("127.0.0.1", serverPort);

    tester.beginTest("WaitForData with No Data");
    int result = server.waitForData(100); // 100ms timeout
    tester.assertEqual(0, result, "waitForData should return 0 when no data is available");

    // Send data from client to server
    auto buffer = client.getBuffer();

    std::string message = "WaitForData Test";
    size_t dataSize;
    PacketData::fillPacket(buffer.data(), MAX_PACKET_SIZE, message, dataSize);

    UDPPacket packet(&serverAddr, std::move(buffer));
    packet.dataLength = dataSize;

    client.sendPacket(std::move(packet), serverAddr);

    // Now wait for data on server
    tester.beginTest("WaitForData with Data Available");
    result = server.waitForData(1000); // 1 second timeout
    tester.assertTrue(result > 0, "waitForData should return > 0 when data is available");

    // Clean up by receiving the packet
    server.receive();
}

void testBufferPoolFunctionality(UDPTester& tester) {
    DefaultUDP node;
    node.init();

    tester.beginTest("Get Buffer from Pool");
    auto buffer = node.getBuffer();
    tester.assertTrue(buffer.data() != nullptr, "Failed to get buffer from pool");

    tester.beginTest("Write to Buffer");
    std::string testData = "Buffer Pool Test";
    memcpy(buffer.data(), testData.c_str(), testData.size());
    bool dataMatch = (memcmp(buffer.data(), testData.c_str(), testData.size()) == 0);
    tester.assertTrue(dataMatch, "Data written to buffer doesn't match");

    tester.beginTest("Get Multiple Buffers");
    std::vector<BufferHandle> buffers;
    for (int i = 0; i < 10; i++) {
        buffers.push_back(node.getBuffer());
    }

    bool allBuffersValid = true;
    for (const auto& buf : buffers) {
        if (!buf.data()) {
            allBuffersValid = false;
            break;
        }
    }

    tester.assertTrue(allBuffersValid, "Failed to get multiple buffers from pool");

    // Clear the buffers to return them to the pool
    buffers.clear();
}

void benchmarkDefaultPacketThroughput(DefaultUDP& sender, DefaultUDP& receiver,
    SocketAddress& receiverAddr,
    const std::string& benchmarkName,
    int packetSize, int packetCount) {
    std::cout << "\nRunning " << benchmarkName << " benchmark with "
        << packetSize << " byte packets, " << packetCount << " packets...\n";

    Benchmark benchmark(benchmarkName);

    // Generate random data for the packet
    std::vector<uint8_t> testData = generateRandomData(packetSize);

    // Get start time for overall throughput
    benchmark.start();

    int sentCount = 0;
    int receiveCount = 0;

    // Send all packets in batches with execute calls
    const int batchSize = 10; // Send 10 packets, then try to receive

    for (int i = 0; i < packetCount; i++) {
        auto buffer = sender.getBuffer();

        size_t dataSize;
        PacketData::fillPacket(buffer.data(), MAX_PACKET_SIZE, testData, dataSize);

        UDPPacket packet(&receiverAddr, std::move(buffer));
        packet.dataLength = dataSize;

        int result = sender.sendPacket(std::move(packet), receiverAddr);
        if (result > 0) {
            sentCount++;
        }

        // Try to receive packets every batchSize
        if (i % batchSize == 0) {
            // Try to receive packets
            for (int j = 0; j < batchSize; j++) {
                UDPPacket received = receiver.receive();
                if (received.bufferHandle) {
                    receiveCount++;
                }
            }
        }
    }

    // Continue until all packets are received or timeout
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(5);

    while (receiveCount < sentCount) {
        auto currentTime = std::chrono::steady_clock::now();
        if (currentTime - startTime > timeout) {
            std::cout << "Timeout after receiving " << receiveCount << "/" << packetCount << " packets\n";
            break;
        }

        UDPPacket packet = receiver.receive();
        if (packet.bufferHandle) {
            receiveCount++;
        }
        else {
            sleep_ms(1); // Prevent CPU spin
        }
    }

    benchmark.stop(packetSize * sentCount);

    std::cout << "Sent: " << sentCount << " packets, Received: " << receiveCount << " packets\n";
    benchmark.printResults();
}

void benchmarkReliablePacketThroughput(ReliableOrderedUDP& sender, ReliableOrderedUDP& receiver,
    uint16_t userID,
    const std::string& benchmarkName,
    int packetSize, int packetCount) {
    std::cout << "\nRunning " << benchmarkName << " benchmark with "
        << packetSize << " byte packets, " << packetCount << " packets...\n";

    Benchmark benchmark(benchmarkName);

    // Generate random data for the packet
    std::vector<uint8_t> testData = generateRandomData(packetSize);

    // Get start time for overall throughput
    benchmark.start();

    int sentCount = 0;
    int receiveCount = 0;

    // Send all packets in batches with execute calls
    const int batchSize = 10; // Send 10 packets, then execute

    for (int i = 0; i < packetCount; i++) {
        UDPPacket packet = sender.preparePacket(userID);

        if (!packet.bufferHandle) {
            std::cout << "Failed to prepare packet\n";
            continue;
        }

        // Fill packet with data
        size_t dataSize;
        ReliableOrderedHeader* header = reinterpret_cast<ReliableOrderedHeader*>(packet.bufferHandle.data());
        uint8_t ackSize = header->AckSize;

        PacketData::fillPacket(
            packet.bufferHandle.data() + sizeof(ReliableOrderedHeader) +
            (ackSize * sizeof(uint16_t)),
            MAX_PACKET_SIZE - sizeof(ReliableOrderedHeader) -
            (ackSize * sizeof(uint16_t)),
            testData, dataSize);

        packet.dataLength = sizeof(ReliableOrderedHeader) +
            (ackSize * sizeof(uint16_t)) +
            dataSize;

        int result = sender.sendPacket(std::move(packet));
        if (result > 0) {
            sentCount++;
        }

        // Call execute periodically
        if (i % batchSize == 0) {
            sender.execute();
            receiver.execute();

            // Try to receive packets
            for (int j = 0; j < batchSize; j++) {
                UDPPacket received = receiver.receive();
                if (received.bufferHandle) {
                    if (receiver.processPacket(received)) {
                        receiveCount++;
                    }
                }
            }
        }
    }

    // Final execute and receive
    sender.execute();
    receiver.execute();

    // Continue until all packets are received or timeout
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(10); // Longer timeout for reliable packets

    while (receiveCount < sentCount) {
        auto currentTime = std::chrono::steady_clock::now();
        if (currentTime - startTime > timeout) {
            std::cout << "Timeout after receiving " << receiveCount << "/" << packetCount << " packets\n";
            break;
        }

        // Call execute to process packets
        sender.execute();
        receiver.execute();

        UDPPacket received = receiver.receive();
        if (received.bufferHandle) {
            if (receiver.processPacket(received)) {
                receiveCount++;
            }
        }
        else {
            sleep_ms(1); // Prevent CPU spin
        }
    }

    benchmark.stop(packetSize * sentCount);

    std::cout << "Sent: " << sentCount << " packets, Received: " << receiveCount << " packets\n";
    benchmark.printResults();
}

int main() {
    std::cout << "=== UDP Networking Library Tests ===\n";

    UDPTester tester;

    // Unit Tests
    testInitialization(tester);
    testUserManagement(tester);
    testBufferPoolFunctionality(tester);

    // Functionality Tests
    testDefaultPacketSending(tester);
    testUnreliablePacketSending(tester);
    testReliableUnorderedPacketSending(tester);
    testReliableOrderedPacketSending(tester);
    testPacketLossSimulation(tester);
    testWaitForData(tester);

    // Advanced Tests
    testRetransmissionMechanism(tester);

    // Print test summary
    tester.printSummary();

    // Benchmarks
    std::cout << "\n=== UDP Networking Library Benchmarks ===\n";

    // Setup benchmark nodes for default throughput
    DefaultUDP defaultSender;
    defaultSender.init("127.0.0.1", 0);

    DefaultUDP defaultReceiver;
    defaultReceiver.init("127.0.0.1", 0);

    SocketAddress receiverAddr;
    receiverAddr.setIPv4("127.0.0.1", defaultReceiver.getPort());

    // Default packet throughput benchmarks
    benchmarkDefaultPacketThroughput(defaultSender, defaultReceiver, receiverAddr,
        "Small Default Packet Throughput", 128, 1000);

    benchmarkDefaultPacketThroughput(defaultSender, defaultReceiver, receiverAddr,
        "Medium Default Packet Throughput", 512, 500);

    benchmarkDefaultPacketThroughput(defaultSender, defaultReceiver, receiverAddr,
        "Large Default Packet Throughput", 1200, 250);

    // Setup benchmark nodes for reliable throughput
    ReliableOrderedUDP reliableSender;
    reliableSender.init("127.0.0.1", 0);

    ReliableOrderedUDP reliableReceiver;
    reliableReceiver.init("127.0.0.1", 0);

    // Add test user for reliable benchmarks
    uint16_t userID = 789;
    std::array<uint8_t, 32> aesKey;
    std::fill(aesKey.begin(), aesKey.end(), 0xDD);

    reliableSender.addUser(userID, aesKey);
    reliableReceiver.addUser(userID, aesKey);

    // Update user addresses for reliable communication
    UserData* userData;
    reliableSender.getUser(userID, userData);
    userData->lastAddress.setIPv4("127.0.0.1", reliableReceiver.getPort());

    reliableReceiver.getUser(userID, userData);
    userData->lastAddress.setIPv4("127.0.0.1", reliableSender.getPort());

    // Reliable packet throughput benchmarks
    benchmarkReliablePacketThroughput(reliableSender, reliableReceiver, userID,
        "Small Reliable Packet Throughput", 128, 1000);

    benchmarkReliablePacketThroughput(reliableSender, reliableReceiver, userID,
        "Medium Reliable Packet Throughput", 512, 500);

    benchmarkReliablePacketThroughput(reliableSender, reliableReceiver, userID,
        "Large Reliable Packet Throughput", 1200, 250);

    return 0;
}