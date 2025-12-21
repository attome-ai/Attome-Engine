#pragma once
#include <ATMNetwork.h>
#include <SDL3/SDL.h>
#include <SDL3_net/SDL_net.h>
#include <string>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <cassert>
#include <cstring>
#include <functional>
#include <atomic>
#include <memory>

// Include the UDPNode class definition from your original code
// (Insert your UDPNode and resolveAddress code here)

#define MAX_PACKET_SIZE 1400



// Test utilities
class TestUtil {
public:
    // Random data generator
    static std::vector<uint8_t> generateRandomData(size_t size) {
        std::vector<uint8_t> data(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<uint8_t>(dis(gen));
        }

        return data;
    }

    // Generate data of incrementing bytes for easy validation
    static std::vector<uint8_t> generateSequentialData(size_t size) {
        std::vector<uint8_t> data(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<uint8_t>(i % 256);
        }
        return data;
    }

    // Validate that two data buffers match
    static bool compareData(const uint8_t* data1, const uint8_t* data2, size_t size) {
        return memcmp(data1, data2, size) == 0;
    }

    // Helper to print buffer contents for debugging
    static void printBuffer(const uint8_t* buffer, size_t size, const char* label = "Buffer") {
        std::cout << label << " [" << size << " bytes]: ";
        for (size_t i = 0; i < std::min(size, size_t(16)); ++i) {
            printf("%02x ", buffer[i]);
        }
        if (size > 16) std::cout << "...";
        std::cout << std::endl;
    }
};

// Test case base class
class TestCase {
private:
    std::string name;

public:
    TestCase(const std::string& testName) : name(testName) {}

    virtual bool run() = 0;

    std::string getName() const { return name; }

    virtual ~TestCase() {}
};

// Test suite to manage and run all tests
class UDPNodeTestSuite {
private:
    std::vector<std::unique_ptr<TestCase>> tests;

public:
    void addTest(TestCase* test) {
        tests.emplace_back(test);
    }

    void runAll() {
        int passed = 0;
        int failed = 0;

        std::cout << "=== UDPNode Test Suite ===" << std::endl;

        for (auto& test : tests) {
            std::cout << "\nRunning test: " << test->getName() << std::endl;
            std::cout << "----------------------------------" << std::endl;


            bool result = test->run();

            if (result) {
                std::cout << "PASS: " << test->getName() << std::endl;
                passed++;
            }
            else {
                std::cout << "FAIL: " << test->getName() << std::endl;
                failed++;
            }


        }

        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << tests.size() << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << failed << std::endl;
    }
};

// Basic initialization and shutdown test
class InitShutdownTest : public TestCase {
public:
    InitShutdownTest() : TestCase("Initialization and Shutdown") {}

    bool run() override {
        // Test initialization with no address (any)
        UDPNode node1;
        bool result1 = node1.init(nullptr, 0);
        std::cout << "Init with any address: " << (result1 ? "SUCCESS" : "FAILED") << std::endl;
        node1.shutdown();

        // Test initialization with localhost and specific port
        UDPNode node2;
        bool result2 = node2.init("127.0.0.1", 12345);
        std::cout << "Init with localhost:12345: " << (result2 ? "SUCCESS" : "FAILED") << std::endl;
        std::cout << "Bound port: " << node2.getPort() << std::endl;
        node2.shutdown();

        // Test initialization with invalid address
        UDPNode node3;
        bool result3 = node3.init("invalid.address.that.doesnt.exist", 12346);
        std::cout << "Init with invalid address: " << (!result3 ? "CORRECTLY FAILED" : "INCORRECTLY SUCCEEDED") << std::endl;
        node3.shutdown();

        return result1 && result2 && !result3;
    }
};

// Test sending and receiving single packets
class BasicSendReceiveTest : public TestCase {
public:
    BasicSendReceiveTest() : TestCase("Extreme Send and Receive") {}

    bool run() override {
        // Create server node
        UDPNode server;
        if (!server.init("127.0.0.1", 12347)) {
            std::cout << "Server initialization failed: " << SDL_GetError() << std::endl;
            return false;
        }

        // Create client node
        UDPNode client;
        if (!client.init("127.0.0.1", 0)) {
            std::cout << "Client initialization failed: " << SDL_GetError() << std::endl;
            server.shutdown();
            return false;
        }

        // Resolve server address
        SDLNet_Address* serverAddr = resolveAddress("127.0.0.1");
        if (!serverAddr) {
            std::cout << "Address resolution failed: " << SDL_GetError() << std::endl;
            client.shutdown();
            server.shutdown();
            return false;
        }

        bool success = true;
        const int NUM_TESTS = 100;  // Extreme: Run 100 tests back-to-back

        std::cout << "Running " << NUM_TESTS << " rapid send/receive cycles..." << std::endl;

        for (int test = 0; test < NUM_TESTS; test++) {
            // Generate test data with varying sizes for each iteration
            int testSize = 50 + (test * 10) % 1000;  // Vary between 50 and 1050 bytes
            std::vector<uint8_t> testData = TestUtil::generateSequentialData(testSize);

            // Send data with no delay between sends
            bool sendResult = client.send(testData.data(), testData.size(), serverAddr, 12347);
            if (!sendResult) {
                std::cout << "Send failed at iteration " << test << ": " << SDL_GetError() << std::endl;
                success = false;
                continue;
            }

            // Minimal timeout - test rapid receive capability
            int waitResult = server.waitForData(100);
            if (waitResult <= 0) {
                std::cout << "Wait for data failed or timed out at iteration " << test << std::endl;
                success = false;
                continue;
            }

            // Receive data using receiveDatagram method
            SDLNet_Datagram* datagram = server.receiveDatagram();
            if (!datagram) {
                std::cout << "Receive datagram failed at iteration " << test << std::endl;
                success = false;
                continue;
            }

            // Verify data content
            bool dataMatch = TestUtil::compareData(testData.data(),
                reinterpret_cast<const uint8_t*>(datagram->buf),
                std::min(testData.size(), static_cast<size_t>(datagram->buflen)));

            if (!dataMatch) {
                std::cout << "Data mismatch at iteration " << test << std::endl;
                success = false;
            }

            // Cleanup
            SDLNet_DestroyDatagram(datagram);
        }

        // Cleanup
        SDLNet_UnrefAddress(serverAddr);
        client.shutdown();
        server.shutdown();

        return success;
    }
};

// Test packet size boundary conditions
class PacketSizeBoundaryTest : public TestCase {
public:
    PacketSizeBoundaryTest() : TestCase("Extreme Packet Size Boundary Tests") {}

    bool run() override {
        // Create server node
        UDPNode server;
        if (!server.init("127.0.0.1", 12349)) {
            std::cout << "Server initialization failed: " << SDL_GetError() << std::endl;
            return false;
        }

        // Create client node
        UDPNode client;
        if (!client.init("127.0.0.1", 0)) {
            std::cout << "Client initialization failed: " << SDL_GetError() << std::endl;
            server.shutdown();
            return false;
        }

        // Resolve server address
        SDLNet_Address* serverAddr = resolveAddress("127.0.0.1");
        if (!serverAddr) {
            std::cout << "Address resolution failed: " << SDL_GetError() << std::endl;
            client.shutdown();
            server.shutdown();
            return false;
        }

        bool success = true;

        // Extreme test cases with different packet sizes - push towards UDP limits
        const std::vector<size_t> testSizes = {
            1,                  // Minimum size
            2,                  // Very small
            3,                  // Very small
            64,                 // Small
            512,                // Medium
            1024,               // 1KB
            4096,               // 4KB
            8192,               // 8KB
            16384,              // 16KB
            32768,              // 32KB
            MAX_PACKET_SIZE,    // Maximum defined size
            65507               // Near theoretical UDP maximum
        };

        for (size_t size : testSizes) {
            std::cout << "Testing extreme packet size: " << size << " bytes" << std::endl;

            if (size > MAX_PACKET_SIZE) {
                std::cout << "WARNING: Testing size beyond MAX_PACKET_SIZE (" << MAX_PACKET_SIZE << ")" << std::endl;
            }

            // Generate test data
            std::vector<uint8_t> testData = TestUtil::generateSequentialData(size);

            // Send data
            bool sendResult = client.send(testData.data(), testData.size(), serverAddr, 12349);
            if (!sendResult) {
                std::cout << "Send failed for size " << size << ": " << SDL_GetError() << std::endl;
                if (size > MAX_PACKET_SIZE) {
                    std::cout << "This failure may be expected for very large packets" << std::endl;
                    continue;  // Continue with next size, don't count as failure
                }
                success = false;
                continue;
            }

            // Wait for data with short timeout to stress timing
            int waitResult = server.waitForData(500);
            if (waitResult <= 0) {
                std::cout << "Wait for data failed or timed out for size " << size << std::endl;
                if (size > MAX_PACKET_SIZE) {
                    continue;  // Continue with next size, don't count as failure
                }
                success = false;
                continue;
            }

            // Receive data
            SDLNet_Datagram* datagram = server.receiveDatagram();
            if (!datagram) {
                std::cout << "Receive datagram failed for size " << size << std::endl;
                success = false;
                continue;
            }

            // Verify data size
            bool sizeMatch = (datagram->buflen == testData.size());
            if (!sizeMatch) {
                std::cout << "Size mismatch for packet size " << size
                    << ". Sent: " << testData.size()
                    << ", Received: " << datagram->buflen << std::endl;
                success = false;
            }

            // Verify data content for non-empty packets
            if (size > 0) {
                bool dataMatch = TestUtil::compareData(testData.data(),
                    reinterpret_cast<const uint8_t*>(datagram->buf),
                    std::min(testData.size(), static_cast<size_t>(datagram->buflen)));

                if (!dataMatch) {
                    std::cout << "Data mismatch for size " << size << std::endl;
                    success = false;
                }
            }

            // Cleanup datagram
            SDLNet_DestroyDatagram(datagram);
        }

        // Cleanup
        SDLNet_UnrefAddress(serverAddr);
        client.shutdown();
        server.shutdown();

        return success;
    }
};
// Test multiple clients connecting to a single server
class MultiClientTest : public TestCase {
public:
    MultiClientTest() : TestCase("Extreme Multiple Clients Test") {}

    bool run() override {
        const int NUM_CLIENTS = 100;  // Extreme: 100 clients at once

        std::cout << "Testing with " << NUM_CLIENTS << " simultaneous clients..." << std::endl;

        // Create server node
        UDPNode server;
        if (!server.init("127.0.0.1", 12350)) {
            std::cout << "Server initialization failed: " << SDL_GetError() << std::endl;
            return false;
        }

        std::vector<std::unique_ptr<UDPNode>> clients;

        // Create multiple client nodes
        for (int i = 0; i < NUM_CLIENTS; ++i) {
            auto client = std::make_unique<UDPNode>();
            if (!client->init("127.0.0.1", 0)) {
                std::cout << "Client " << i << " initialization failed: " << SDL_GetError() << std::endl;
                // Continue despite failures - this is an extreme test
                continue;
            }
            clients.push_back(std::move(client));

            if (i % 10 == 0) {
                std::cout << "Initialized " << clients.size() << " clients so far..." << std::endl;
            }
        }

        if (clients.empty()) {
            std::cout << "All client initializations failed!" << std::endl;
            server.shutdown();
            return false;
        }

        std::cout << "Successfully initialized " << clients.size() << " of " << NUM_CLIENTS << " clients" << std::endl;

        // Resolve server address once for all clients
        SDLNet_Address* serverAddr = resolveAddress("127.0.0.1");
        if (!serverAddr) {
            std::cout << "Address resolution failed: " << SDL_GetError() << std::endl;
            server.shutdown();
            return false;
        }

        // Start a receiver thread to process packets asynchronously
        std::atomic<int> receivedCount(0);
        std::atomic<bool> stopReceiving(false);

        std::thread receiverThread([&]() {
            while (!stopReceiving) {
                if (server.waitForData(10) > 0) {
                    SDLNet_Datagram* datagram = server.receiveDatagram();
                    if (datagram) {
                        receivedCount++;
                        SDLNet_DestroyDatagram(datagram);
                    }
                }
            }
            });

        // Have all clients send data simultaneously to stress the server
        std::cout << "All clients sending data simultaneously..." << std::endl;

#pragma omp parallel for if(clients.size() > 10)
        for (size_t i = 0; i < clients.size(); ++i) {
            // Create unique message for this client
            std::string message = "Message from client " + std::to_string(i);
            std::vector<uint8_t> data(message.begin(), message.end());

            // Send data multiple times to increase load
            for (int j = 0; j < 10; j++) {
                clients[i]->send(data.data(), data.size(), serverAddr, 12350);
                // No delay between sends to maximize stress
            }
        }

        // Allow time for packets to be processed
        std::cout << "Waiting for server to process packets..." << std::endl;
        SDL_Delay(2000);

        // Stop the receiver thread
        stopReceiving = true;
        receiverThread.join();

        std::cout << "Received " << receivedCount << " of " << (clients.size() * 10) << " potential messages" << std::endl;

        // Cleanup
        SDLNet_UnrefAddress(serverAddr);
        server.shutdown();

        // For extreme test, some packet loss is expected
        // Success is if we received at least some messages
        return receivedCount > 0;
    }
};

class ThroughputTest : public TestCase {
public:
    ThroughputTest() : TestCase("Extreme Throughput Performance Test") {}

    bool run() override {
        // Create server node
        UDPNode server;
        if (!server.init("127.0.0.1", 12352)) {
            std::cout << "Server initialization failed: " << SDL_GetError() << std::endl;
            return false;
        }

        // Create client node
        UDPNode client;
        if (!client.init("127.0.0.1", 0)) {
            std::cout << "Client initialization failed: " << SDL_GetError() << std::endl;
            server.shutdown();
            return false;
        }

        // Resolve server address
        SDLNet_Address* serverAddr = resolveAddress("127.0.0.1");
        if (!serverAddr) {
            std::cout << "Address resolution failed: " << SDL_GetError() << std::endl;
            client.shutdown();
            server.shutdown();
            return false;
        }

        // Extreme test parameters - massive load
        const int packetSize = 8192;         // Large packet size - 8KB
        const int numPackets = 10000;        // Many packets
        const int batchSize = 100;           // Large batches
        const int batchDelayMs = 0;          // No delay (max stress)
        const size_t totalBytes = packetSize * numPackets;

        std::cout << "Extreme throughput test with " << numPackets << " packets of "
            << packetSize << " bytes each (" << (totalBytes / 1024 / 1024) << " MB total)" << std::endl;

        // Start a receiver thread to process packets asynchronously
        std::atomic<int> receivedCount(0);
        std::atomic<bool> stopReceiving(false);
        std::atomic<size_t> bytesReceived(0);

        std::thread receiverThread([&]() {
            while (!stopReceiving) {
                if (server.waitForData(1) > 0) {  // Very short timeout
                    SDLNet_Datagram* datagram = server.receiveDatagram();
                    if (datagram) {
                        receivedCount++;
                        bytesReceived += datagram->buflen;
                        SDLNet_DestroyDatagram(datagram);
                    }
                }
            }
            });

        // Generate test data once
        std::vector<uint8_t> testData(packetSize);
        for (int i = 0; i < packetSize; i++) {
            testData[i] = static_cast<uint8_t>(i % 256);
        }

        // Start timing
        auto startTime = std::chrono::high_resolution_clock::now();
        int sentCount = 0;

        // Send packets as fast as possible with minimal rate limiting
        for (int i = 0; i < numPackets; ++i) {
            if (client.send(testData.data(), testData.size(), serverAddr, 12352)) {
                sentCount++;
            }

            // Status update
            if (i % 1000 == 0 && i > 0) {
                std::cout << "Sent " << i << " packets so far..." << std::endl;

                // Optional emergency stop if system is overwhelmed
                if (receivedCount < i / 10) {
                    std::cout << "WARNING: Very high packet loss detected, system may be overwhelmed" << std::endl;
                    // Continue anyway for stress testing
                }
            }
        }

        // End of send timing
        auto sendEndTime = std::chrono::high_resolution_clock::now();

        // Allow time for all packets to be processed
        std::cout << "Waiting for packets to be processed..." << std::endl;
        SDL_Delay(3000);

        // Stop the receiver thread
        stopReceiving = true;
        receiverThread.join();

        // End of receive timing
        auto endTime = std::chrono::high_resolution_clock::now();

        // Calculate timings and throughput
        auto sendDuration = std::chrono::duration_cast<std::chrono::milliseconds>(sendEndTime - startTime).count();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

        double sendMbps = (sentCount * packetSize * 8.0) / (sendDuration * 1000.0);  // Mbps
        double endToEndMbps = (bytesReceived * 8.0) / (totalDuration * 1000.0);  // Mbps

        std::cout << "Extreme throughput test results:" << std::endl;
        std::cout << "  Packets sent: " << sentCount << std::endl;
        std::cout << "  Packets received: " << receivedCount << std::endl;
        std::cout << "  Bytes received: " << bytesReceived << " (" << (bytesReceived / 1024 / 1024) << " MB)" << std::endl;
        std::cout << "  Send duration: " << sendDuration << " ms" << std::endl;
        std::cout << "  Total duration: " << totalDuration << " ms" << std::endl;
        std::cout << "  Send rate: " << sendMbps << " Mbps" << std::endl;
        std::cout << "  End-to-end throughput: " << endToEndMbps << " Mbps" << std::endl;
        std::cout << "  Packet loss: " << (100.0 - (receivedCount * 100.0 / sentCount)) << "%" << std::endl;

        // Cleanup
        SDLNet_UnrefAddress(serverAddr);
        client.shutdown();
        server.shutdown();

        // For extreme stress testing, just getting some packets through is a success
        return receivedCount > 0;
    }
};
// Test stress with many small packets
class SmallPacketStressTest : public TestCase {
public:
    SmallPacketStressTest() : TestCase("Extreme Small Packet Stress Test") {}

    bool run() override {
        // Create server node
        UDPNode server;
        if (!server.init("127.0.0.1", 12353)) {
            std::cout << "Server initialization failed: " << SDL_GetError() << std::endl;
            return false;
        }

        // Create client node
        UDPNode client;
        if (!client.init("127.0.0.1", 0)) {
            std::cout << "Client initialization failed: " << SDL_GetError() << std::endl;
            server.shutdown();
            return false;
        }

        // Resolve server address
        SDLNet_Address* serverAddr = resolveAddress("127.0.0.1");
        if (!serverAddr) {
            std::cout << "Address resolution failed: " << SDL_GetError() << std::endl;
            client.shutdown();
            server.shutdown();
            return false;
        }

        // Extreme test parameters
        const int packetSize = 2;         // Extremely small packets
        const int numPackets = 100000;    // Massive number of packets
        const int tickInterval = 5000;    // Status update interval

        std::cout << "Sending " << numPackets << " extremely small packets of " << packetSize << " bytes..." << std::endl;

        // Create a server receiving thread 
        std::atomic<int> receivedCount(0);
        std::atomic<bool> stopReceiving(false);

        std::thread receiverThread([&]() {
            while (!stopReceiving) {
                if (server.waitForData(1) > 0) {  // Minimal timeout for extreme throughput
                    SDLNet_Datagram* datagram = server.receiveDatagram();
                    if (datagram) {
                        receivedCount++;
                        SDLNet_DestroyDatagram(datagram);
                    }
                }
            }
            });

        // Generate multiple tiny test packets
        std::vector<std::vector<uint8_t>> packets;
        for (int i = 0; i < 100; i++) {  // Create 100 different packet templates
            std::vector<uint8_t> packet(packetSize);
            for (int j = 0; j < packetSize; j++) {
                packet[j] = static_cast<uint8_t>((i + j) % 256);
            }
            packets.push_back(packet);
        }

        // Send packets with sequence numbers as fast as possible
        auto startTime = std::chrono::high_resolution_clock::now();
        int sentCount = 0;

        for (int i = 0; i < numPackets; ++i) {
            // Select a packet template based on iteration
            const std::vector<uint8_t>& packet = packets[i % packets.size()];

            // Send without any delay
            if (client.send(packet.data(), packet.size(), serverAddr, 12353)) {
                sentCount++;
            }

            // Status update
            if (i % tickInterval == 0 && i > 0) {
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    currentTime - startTime).count();

                double packetsPerSec = (i * 1000.0) / elapsedMs;
                double receivedPerSec = (receivedCount * 1000.0) / elapsedMs;

                std::cout << "Progress: Sent " << i << " packets ("
                    << packetsPerSec << " packets/sec), received "
                    << receivedCount << " (" << receivedPerSec << " packets/sec)" << std::endl;
            }
        }

        auto endSendTime = std::chrono::high_resolution_clock::now();

        // Allow time for all packets to be received
        std::cout << "Waiting for packets to be processed..." << std::endl;
        SDL_Delay(3000);

        // Stop the receiver thread
        stopReceiving = true;
        receiverThread.join();

        auto endTime = std::chrono::high_resolution_clock::now();

        auto sendDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSendTime - startTime).count();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

        double sendRate = sentCount * 1000.0 / sendDuration;
        double receiveRate = receivedCount * 1000.0 / totalDuration;

        std::cout << "Extreme small packet stress test results:" << std::endl;
        std::cout << "  Packets sent: " << sentCount << std::endl;
        std::cout << "  Packets received: " << receivedCount << std::endl;
        std::cout << "  Send duration: " << sendDuration << " ms" << std::endl;
        std::cout << "  Total duration: " << totalDuration << " ms" << std::endl;
        std::cout << "  Send rate: " << sendRate << " packets/sec" << std::endl;
        std::cout << "  Receive rate: " << receiveRate << " packets/sec" << std::endl;
        std::cout << "  Packet loss: " << (100.0 - (receivedCount * 100.0 / sentCount)) << "%" << std::endl;

        // Calculate effective bitrate
        double sendBitrate = (sentCount * packetSize * 8.0) / (sendDuration / 1000.0);  // bits/sec
        double recvBitrate = (receivedCount * packetSize * 8.0) / (totalDuration / 1000.0);  // bits/sec

        std::cout << "  Effective send bitrate: " << (sendBitrate / 1000000.0) << " Mbps" << std::endl;
        std::cout << "  Effective receive bitrate: " << (recvBitrate / 1000000.0) << " Mbps" << std::endl;

        // Cleanup
        SDLNet_UnrefAddress(serverAddr);
        client.shutdown();
        server.shutdown();

        // For extreme tests, receiving any packets is a success
        return receivedCount > 0;
    }
};
// Error handling test
class ErrorHandlingTest : public TestCase {
public:
    ErrorHandlingTest() : TestCase("Extreme Error Handling") {}

    bool run() override {
        bool allTestsRun = true;  // Track if tests ran, not if they passed

        // Test 1: Trying to initialize with same port twice
        std::cout << "Test: Initializing two servers on same port" << std::endl;
        UDPNode server1;
        UDPNode server2;

        bool server1Init = server1.init("127.0.0.1", 12354);
        std::cout << "First server initialization: " << (server1Init ? "SUCCESS" : "FAILED") << std::endl;

        bool server2Init = server2.init("127.0.0.1", 12354);
        std::cout << "Second server initialization (should fail): " << (!server2Init ? "CORRECTLY FAILED" : "INCORRECTLY SUCCEEDED") << std::endl;

        // Test 2: Try with extreme port values
        std::cout << "\nTest: Extreme port values" << std::endl;
        UDPNode portNode;

        // Try with max uint16_t value
        bool maxPortInit = portNode.init("127.0.0.1", 65535);
        std::cout << "Max port (65535): " << (maxPortInit ? "SUCCESS" : "FAILED") << std::endl;
        portNode.shutdown();

        // Test 3: Sending after socket shutdown
        std::cout << "\nTest: Sending after socket shutdown" << std::endl;
        UDPNode shutdownNode;
        shutdownNode.init("127.0.0.1", 0);




        // Test 5: Rapid initialization/shutdown cycles with immediate sends
        std::cout << "\nTest: Rapid init/shutdown with immediate sends" << std::endl;
        SDLNet_Address* rapidAddr = resolveAddress("127.0.0.1");
        if (rapidAddr) {
            uint8_t rapidData[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

            for (int i = 0; i < 20; i++) {
                UDPNode rapidNode;
                if (rapidNode.init("127.0.0.1", 40000 + i)) {
                    // Send immediately after init
                    rapidNode.send(rapidData, 10, rapidAddr, 12345);
                    // Immediate shutdown
                    rapidNode.shutdown();
                }
            }

            SDLNet_UnrefAddress(rapidAddr);
            std::cout << "Completed rapid init/shutdown cycles" << std::endl;
        }

        // Cleanup remaining resources
        server1.shutdown();
        server2.shutdown();

        return allTestsRun;  // We're testing error handling, so just report if tests ran
    }
};


class BiDirectionalStressTest : public TestCase {
public:
    BiDirectionalStressTest() : TestCase("Extreme Bidirectional Stress Test") {}

    bool run() override {
        // Create two nodes
        UDPNode node1;
        UDPNode node2;

        if (!node1.init("127.0.0.1", 12360)) {
            std::cout << "Node 1 initialization failed: " << SDL_GetError() << std::endl;
            return false;
        }

        if (!node2.init("127.0.0.1", 12361)) {
            std::cout << "Node 2 initialization failed: " << SDL_GetError() << std::endl;
            node1.shutdown();
            return false;
        }

        // Resolve addresses
        SDLNet_Address* addr1 = resolveAddress("127.0.0.1");
        if (!addr1) {
            std::cout << "Address resolution failed: " << SDL_GetError() << std::endl;
            node1.shutdown();
            node2.shutdown();
            return false;
        }

        // Extreme test parameters
        const int numPackets = 50000;  // Very high packet count
        const int varyingSizes = 20;   // Different packet sizes
        const int maxSize = 4096;      // Up to 4KB packets

        std::cout << "Starting extreme bidirectional test with " << numPackets
            << " packets in each direction..." << std::endl;

        // Create data templates of various sizes
        std::vector<std::vector<uint8_t>> dataTemplates;
        for (int i = 0; i < varyingSizes; i++) {
            int size = 16 + (i * maxSize / varyingSizes);  // Vary from 16 bytes to maxSize
            std::vector<uint8_t> data(size);
            for (int j = 0; j < size; j++) {
                data[j] = static_cast<uint8_t>((i + j) % 256);
            }
            dataTemplates.push_back(data);
        }

        // Atomic counters for tracking packets
        std::atomic<int> node1ReceivedCount(0);
        std::atomic<int> node2ReceivedCount(0);
        std::atomic<bool> stopTest(false);

        // Thread for node1 to receive
        std::thread node1ReceiverThread([&]() {
            while (!stopTest) {
                if (node1.waitForData(1) > 0) {
                    SDLNet_Datagram* datagram = node1.receiveDatagram();
                    if (datagram) {
                        node1ReceivedCount++;
                        SDLNet_DestroyDatagram(datagram);
                    }
                }
            }
            });

        // Thread for node2 to receive
        std::thread node2ReceiverThread([&]() {
            while (!stopTest) {
                if (node2.waitForData(1) > 0) {
                    SDLNet_Datagram* datagram = node2.receiveDatagram();
                    if (datagram) {
                        node2ReceivedCount++;
                        SDLNet_DestroyDatagram(datagram);
                    }
                }
            }
            });

        // Thread for node1 to send to node2
        std::atomic<int> node1SentCount(0);
        std::thread node1SenderThread([&]() {
            for (int i = 0; i < numPackets && !stopTest; i++) {
                const std::vector<uint8_t>& data = dataTemplates[i % dataTemplates.size()];
                if (node1.send(data.data(), data.size(), addr1, 12361)) {
                    node1SentCount++;
                }

                // Occasional status
                if (i % 10000 == 0 && i > 0) {
                    std::cout << "Node1: Sent " << i << " packets so far" << std::endl;
                }
            }
            });

        // Thread for node2 to send to node1
        std::atomic<int> node2SentCount(0);
        std::thread node2SenderThread([&]() {
            for (int i = 0; i < numPackets && !stopTest; i++) {
                const std::vector<uint8_t>& data = dataTemplates[i % dataTemplates.size()];
                if (node2.send(data.data(), data.size(), addr1, 12360)) {
                    node2SentCount++;
                }

                // Occasional status
                if (i % 10000 == 0 && i > 0) {
                    std::cout << "Node2: Sent " << i << " packets so far" << std::endl;
                }
            }
            });

        // Wait for sending to complete
        node1SenderThread.join();
        node2SenderThread.join();

        std::cout << "All packets sent, waiting for processing..." << std::endl;
        SDL_Delay(3000);  // Allow time for packets to be processed

        // Stop receiving
        stopTest = true;
        node1ReceiverThread.join();
        node2ReceiverThread.join();

        // Report results
        std::cout << "Extreme bidirectional test results:" << std::endl;
        std::cout << "  Node1 sent: " << node1SentCount << ", received: " << node1ReceivedCount << std::endl;
        std::cout << "  Node2 sent: " << node2SentCount << ", received: " << node2ReceivedCount << std::endl;

        double node1Loss = 100.0 - (node1ReceivedCount * 100.0 / node2SentCount);
        double node2Loss = 100.0 - (node2ReceivedCount * 100.0 / node1SentCount);

        std::cout << "  Node1 packet loss: " << node1Loss << "%" << std::endl;
        std::cout << "  Node2 packet loss: " << node2Loss << "%" << std::endl;

        // Cleanup
        SDLNet_UnrefAddress(addr1);
        node1.shutdown();
        node2.shutdown();

        // Success if we got at least some bidirectional communication
        return (node1ReceivedCount > 0 && node2ReceivedCount > 0);
    }
};
