    #include <ATMNetwork.h>

#include <iostream>
#include <ATMEngine/ATMLog.h>
#include <unordered_map>
#include <array>
#include <vector>
#include <deque>

UDPNode::UDPNode()
    : socket(nullptr),
    boundPort(0),
    bufferPool(MAX_PACKET_SIZE, 16) // Initialize with suitable size and count
{
}

UDPNode::~UDPNode() {
    shutdown();
}




bool UDPNode::init(const char* address, Uint16 port) {

    SDLNet_Address* bindAddr = nullptr;
    if (address != nullptr) {
        bindAddr = SDLNet_ResolveHostname(address);
        if (!bindAddr) {
            SDL_SetError("Failed to resolve hostname: %s", SDL_GetError());
            return false;
        }

        // Wait for address to resolve
        int result = SDLNet_WaitUntilResolved(bindAddr, 5000);  // 5 second timeout
        if (result <= 0) {  // Failed or timed out
            SDLNet_UnrefAddress(bindAddr);
            SDL_SetError("Failed to resolve address: %s", SDL_GetError());
            return false;
        }
    }


    // Create the datagram socket
    socket = SDLNet_CreateDatagramSocket(bindAddr, port);
    
    // Cleanup address if we created it
    if (bindAddr) {
        SDLNet_UnrefAddress(bindAddr);
        bindAddr = nullptr;
    }

    if (!socket) {
        SDL_SetError("Failed to create datagram socket: %s", SDL_GetError());
        return false;
    }

    // Store the bound address and port for diagnostics
    boundPort = port;
    boundAddress = address ? address : "any";

    return true;
}

bool UDPNode::sendRaw(const uint8_t* buffer, int length, SDLNet_Address* address, Uint16 port) {



    bool result = SDLNet_SendDatagram(socket, address, port, buffer, length);
    ATMLOGC(result, "Failed to send datagram : %s", SDL_GetError());

    return result;
}
//must check if datagram is not nullptr;
SDLNet_Datagram* UDPNode::receiveDatagram() {
    SDLNet_Datagram* datagram = nullptr;
    bool result = SDLNet_ReceiveDatagram(socket, &datagram);

    if (!result) {
        return nullptr;
    }
    return datagram;
}

int UDPNode::waitForData(int timeout) {
    return SDLNet_WaitUntilInputAvailable((void**)&socket, 1, timeout);
}



void UDPNode::shutdown() {
    if (socket) {
        SDLNet_DestroyDatagramSocket(socket);
        socket = nullptr;
    }
}

void UDPNode::simulatePacketLoss(int percentLoss)
{
    SDLNet_SimulateDatagramPacketLoss(socket, percentLoss);
}

FixedBufferPool::BufferHandle UDPNode::getBuffer() {
    return bufferPool.getBuffer();
}

Uint16 UDPNode::getPort() const {
    return boundPort;
}

const std::string& UDPNode::getAddress() const {
    return boundAddress;
}



// Utility function implementation
SDLNet_Address* resolveAddress(const char* hostname) {
    // Create the address
    SDLNet_Address* address = SDLNet_ResolveHostname(hostname);
    if (!address) {
        return nullptr;
    }

    // Wait for resolution (with timeout)
    int result = SDLNet_WaitUntilResolved(address, 5000);
    if (result <= 0) {
        SDLNet_UnrefAddress(address);
        return nullptr;
    }

    return address;
}


UDPUnreliable::UDPUnreliable(UDPNode& udpNode) : node(udpNode) {
}

FixedBufferPool::BufferHandle UDPUnreliable::prepareBuffer() {
    // Get buffer from pool
    auto bufferHandle = node.getBuffer();
    uint8_t* buffer = bufferHandle.data();

    if (!buffer) {
        return FixedBufferPool::BufferHandle();
    }

    // Set first byte: 2 bits for type (00 for Type 0), 6 bits for padding
    buffer[0] = 0x00; // Type 0 with zero padding

    return bufferHandle;
}

bool UDPUnreliable::send(const uint8_t* data, int length, SDLNet_Address* address, Uint16 port) {
    // Check for zero-length packets
    if (length <= 0 || data == nullptr) {
        SDL_SetError("Cannot send empty packet or null buffer");
        return false;
    }

    if (!address) {
        SDL_SetError("Cannot send to null address");
        return false;
    }

    // Increase reference count since we're using this address
    SDLNet_RefAddress(address);

    // Prepare the buffer with header
    auto bufferHandle = prepareBuffer();
    if (!bufferHandle.valid()) {
        SDL_SetError("Failed to prepare buffer");
        SDLNet_UnrefAddress(address);
        return false;
    }

    // Copy data after the header
    uint8_t* buffer = bufferHandle.data();
    memcpy(buffer + 1, data, length);

    // Send using the raw send function
    // Add 1 to length to account for the header byte
    bool result = node.sendRaw(buffer, length + 1, address, port);

    // Decrease reference count after use
    SDLNet_UnrefAddress(address);

    return result;
}

SocketType UDPUnreliable::extractSocketType(const uint8_t* data) {
    if (data == nullptr) {
        return SocketType::UNRELIABLE;
    }

    // Extract the first 2 bits from the first byte
    uint8_t typeBits = (data[0] >> 6) & 0x03;
    return static_cast<SocketType>(typeBits);
}

const uint8_t* UDPUnreliable::extractData(const uint8_t* srcData) {
    if (srcData == nullptr) {
        return nullptr;
    }

    // Return pointer to data after the 1-byte header
    return srcData + 1;
}


// UDPReliableUser implementation
UDPReliableUser::UDPReliableUser(UDPNode& udpNode) : node(udpNode) {
}

FixedBufferPool::BufferHandle UDPReliableUser::prepareBuffer(uint32_t userID) {
    if (!validateUserID(userID)) {
        return FixedBufferPool::BufferHandle();
    }

    // Get buffer from pool
    auto bufferHandle = node.getBuffer();
    uint8_t* buffer = bufferHandle.data();

    if (!buffer) {
        return FixedBufferPool::BufferHandle();
    }

    // Set first 4 bytes: 2 bits type (01 for Type 1), 30 bits userID
    // Type bits: 01 (shifted to most significant position)
    buffer[0] = 0x40 | ((userID >> 24) & 0x3F); // Top 6 bits of userID
    buffer[1] = (userID >> 16) & 0xFF;          // Next 8 bits of userID
    buffer[2] = (userID >> 8) & 0xFF;           // Next 8 bits of userID
    buffer[3] = userID & 0xFF;                  // Bottom 8 bits of userID

    return bufferHandle;
}

bool UDPReliableUser::send(uint32_t userID, const uint8_t* data, int length, SDLNet_Address* address, Uint16 port) {
    if (!validateUserID(userID) || data == nullptr || length <= 0) {
        SDL_SetError("Invalid userID or empty data");
        return false;
    }

    if (!address) {
        SDL_SetError("Cannot send to null address");
        return false;
    }

    if (!hasUser(userID)) {
        SDL_SetError("User ID not registered");
        return false;
    }

    // Increase reference count since we're using this address
    SDLNet_RefAddress(address);

    auto bufferHandle = prepareBuffer(userID);
    if (!bufferHandle.valid()) {
        SDL_SetError("Failed to prepare buffer");
        SDLNet_UnrefAddress(address);
        return false;
    }

    // Copy data after the header
    uint8_t* buffer = bufferHandle.data();
    memcpy(buffer + 4, data, length);

    // Update send counter for the user
    auto& userData = users[userID];
    userData.unorderedCounter++;

    // Store the address with proper reference counting
    userData.updateAddress(address);
    userData.lastPort = port;

    // Add 4 to length to account for the 4-byte header
    bool result = node.sendRaw(buffer, length + 4, address, port);

    // Decrease reference count after use
    SDLNet_UnrefAddress(address);

    return result;
}

uint32_t UDPReliableUser::extractUserID(const uint8_t* data) {
    if (data == nullptr) {
        return 0;
    }

    // Combine bytes to form userID (30 bits)
    uint32_t userID = ((data[0] & 0x3F) << 24) |
        (data[1] << 16) |
        (data[2] << 8) |
        data[3];

    return userID;
}

const uint8_t* UDPReliableUser::extractData(const uint8_t* srcData) {
    if (srcData == nullptr) {
        return nullptr;
    }

    // Return pointer to data after the 4-byte header
    return srcData + 4;
}

bool UDPReliableUser::validateUserID(uint32_t userID) {
    // Ensure userID fits in 30 bits (max value 0x3FFFFFFF)
    return (userID <= 0x3FFFFFFF);
}

bool UDPReliableUser::addUser(uint32_t userID, const std::array<uint8_t, 32>& aesKey) {
    if (!validateUserID(userID)) {
        return false;
    }

    // Add user with AES key to the map
    users[userID] = UserData(userID, aesKey);
    return true;
}

bool UDPReliableUser::removeUser(uint32_t userID) {
    if (!validateUserID(userID)) {
        return false;
    }

    // Remove user from the map (returns number of elements removed, 0 or 1)
    return users.erase(userID) > 0;
}

bool UDPReliableUser::hasUser(uint32_t userID) const {
    if (!validateUserID(userID)) {
        return false;
    }

    return users.find(userID) != users.end();
}

const std::array<uint8_t, 32>* UDPReliableUser::getUserAESKey(uint32_t userID) const {
    if (!validateUserID(userID)) {
        return nullptr;
    }

    auto it = users.find(userID);
    if (it != users.end()) {
        return &(it->second.aesKey);
    }

    return nullptr;
}

UserData* UDPReliableUser::getUserData(uint32_t userID) {
    if (!validateUserID(userID)) {
        return nullptr;
    }

    auto it = users.find(userID);
    if (it != users.end()) {
        return &(it->second);
    }

    return nullptr;
}


// Type 2 UDP implementation - Reliable Unordered with integrated acknowledgments
UDPType2::UDPType2(UDPNode& udpNode) : node(udpNode) {
}

UDPType2::~UDPType2() {
    // Clear each pending packet carefully to avoid issues with the buffer handles
    pendingPackets.clear();

    // The UserData destructor will handle unreferencing addresses
    users.clear();
}

FixedBufferPool::BufferHandle UDPType2::prepareBuffer(uint32_t userID, int& ackCount) {
    if (!validateUserID(userID)) {
        ackCount = 0;
        return FixedBufferPool::BufferHandle();
    }

    // Get buffer from pool
    auto bufferHandle = node.getBuffer();
    uint8_t* buffer = bufferHandle.data();

    if (!buffer) {
        ackCount = 0;
        return FixedBufferPool::BufferHandle();
    }

    // Find the user data
    auto userData = getUserData(userID);
    if (!userData) {
        ackCount = 0;
        return FixedBufferPool::BufferHandle();
    }

    // Use current value and then increment
    uint16_t packetCounter = userData->sendCounter2;

    // Increment the counter with proper overflow handling
    userData->sendCounter2 = (userData->sendCounter2 + 1) & 0xFFFF;

    // Set header bytes: 2 bits type (10 for Type 2), 30 bits userID
    // Type bits: 10 (0x80) (shifted to most significant position)
    buffer[0] = 0x80 | ((userID >> 24) & 0x3F); // Top 6 bits of userID
    buffer[1] = (userID >> 16) & 0xFF;          // Next 8 bits of userID
    buffer[2] = (userID >> 8) & 0xFF;           // Next 8 bits of userID
    buffer[3] = userID & 0xFF;                  // Bottom 8 bits of userID

    // Add 16-bit packet counter (2 bytes)
    buffer[4] = (packetCounter >> 8) & 0xFF;   // High byte of packet counter
    buffer[5] = packetCounter & 0xFF;          // Low byte of packet counter

    // Flags byte: bit 0 = is acknowledgment-only packet (1 = true, 0 = false)
    // Other bits reserved for future use
    buffer[6] = 0;

    // Add acknowledgment count (1 byte)
    buffer[7] = 0;

    // Add acknowledgments if user has any pending
    if (!userData->pendingAcks.empty()) {
        uint8_t* ackPtr = buffer + 8;
        ackCount = 0;

        // Add up to MAX_ACKS_PER_PACKET acknowledgments
        for (size_t i = 0; i < userData->pendingAcks.size() && ackCount < MAX_ACKS_PER_PACKET; i++) {
            uint16_t ackId = userData->pendingAcks[i];

            // Write each 16-bit ack packet ID (2 bytes each)
            *ackPtr++ = (ackId >> 8) & 0xFF; // High byte
            *ackPtr++ = ackId & 0xFF;       // Low byte
            ackCount++;
        }

        // Update acknowledgment count in the header
        buffer[7] = static_cast<uint8_t>(ackCount);

        // Remove the acknowledgments that were sent
        if (ackCount > 0) {
            if (ackCount >= userData->pendingAcks.size()) {
                userData->pendingAcks.clear();
            }
            else {
                userData->pendingAcks.erase(userData->pendingAcks.begin(),
                    userData->pendingAcks.begin() + ackCount);
            }
        }
    }

    return bufferHandle;
}

bool UDPType2::send(uint32_t userID, const uint8_t* data, int length, SDLNet_Address* address, Uint16 port) {
    if (!validateUserID(userID) || (length > 0 && data == nullptr)) {
        SDL_SetError("Invalid userID or invalid data");
        return false;
    }

    if (!address) {
        SDL_SetError("Cannot send to null address");
        return false;
    }

    auto userData = getUserData(userID);
    if (!userData) {
        SDL_SetError("User ID not registered");
        return false;
    }

    // Increase reference count since we're using this address
    SDLNet_RefAddress(address);

    // Store the address and port for future acknowledgments with proper reference counting
    userData->updateAddress(address);
    userData->lastPort = port;

    // Prepare the buffer with header and automatic acknowledgments
    int ackCount = 0;
    auto bufferHandle = prepareBuffer(userID, ackCount);
    if (!bufferHandle.valid()) {
        SDL_SetError("Failed to prepare buffer");
        SDLNet_UnrefAddress(address);
        return false;
    }

    // Get the packet counter from the buffer
    uint8_t* buffer = bufferHandle.data();
    uint16_t packetCounter = (buffer[4] << 8) | buffer[5];

    // Calculate header size: 8 bytes base + 2 bytes per acknowledgment
    int headerSize = 8 + (ackCount * 2);

    // Copy data after the header if there is any
    if (length > 0 && data != nullptr) {
        memcpy(buffer + headerSize, data, length);
    }
    else {
        // If no data, mark this as an acknowledgment-only packet
        buffer[6] |= 0x01;
    }

    // Only add to pending packets if this is NOT an acknowledgment-only packet
    if (!(buffer[6] & 0x01) && length > 0) {
        // Store in pending packets for potential retransmission
        pendingPackets.emplace_back(
            userID, packetCounter, data, length, address, port,
            std::move(bufferHandle), headerSize
        );

        // Limit pendingPackets size to prevent memory growth
        while (pendingPackets.size() > MAX_PENDING_PACKETS) {
            pendingPackets.pop_front();
        }
    }

    // Send using the raw send function
    // Total packet length: header size + data length
    bool result = node.sendRaw(buffer, headerSize + length, address, port);

    // Decrease reference count after use
    SDLNet_UnrefAddress(address);

    return result;
}

// Convenience method for sending just acknowledgments (no data)
bool UDPType2::sendAcknowledgmentOnly(uint32_t userID, SDLNet_Address* address, Uint16 port) {
    auto userData = getUserData(userID);
    if (!userData || userData->pendingAcks.empty()) {
        return false; // Nothing to acknowledge
    }

    if (!address) {
        return false; // Invalid address
    }

    // Send an empty data packet (will automatically be marked as ack-only)
    return send(userID, nullptr, 0, address, port);
}

uint32_t UDPType2::extractUserID(const uint8_t* data) {
    if (data == nullptr) {
        return 0;
    }

    // Combine bytes to form userID (30 bits)
    uint32_t userID = ((data[0] & 0x3F) << 24) |
        (data[1] << 16) |
        (data[2] << 8) |
        data[3];

    return userID;
}

UserData* UDPType2::getUserData(uint32_t userID) {
    if (!validateUserID(userID)) {
        return nullptr;
    }

    auto it = users.find(userID);
    if (it != users.end()) {
        return &(it->second);
    }

    return nullptr;
}

uint16_t UDPType2::extractPacketCounter(const uint8_t* data) {
    if (data == nullptr) {
        return 0;
    }

    // Combine bytes to form packet counter (16 bits)
    uint16_t packetCounter = (data[4] << 8) | data[5];

    return packetCounter;
}

bool UDPType2::isAckOnlyPacket(const uint8_t* data) {
    if (data == nullptr) {
        return false;
    }

    // Check the ack-only flag (bit 0 of flags byte)
    return (data[6] & 0x01) != 0;
}

uint8_t UDPType2::extractAckCount(const uint8_t* data) {
    if (data == nullptr) {
        return 0;
    }

    // Get acknowledgment count
    return data[7];
}

std::vector<uint16_t> UDPType2::extractAckPackets(const uint8_t* data) {
    std::vector<uint16_t> result;

    if (data == nullptr) {
        return result;
    }

    uint8_t ackCount = data[7];
    if (ackCount == 0) {
        return result;
    }

    // Reserve space for efficiency
    result.reserve(ackCount);

    // Extract each acknowledgment
    const uint8_t* ackData = data + 8;
    for (uint8_t i = 0; i < ackCount; i++) {
        uint16_t ackId = (ackData[0] << 8) | ackData[1];
        result.push_back(ackId);
        ackData += 2; // Move to next acknowledgment
    }

    return result;
}

const uint8_t* UDPType2::extractData(const uint8_t* srcData) {
    if (srcData == nullptr) {
        return nullptr;
    }

    // Get acknowledgment count to calculate header size
    uint8_t ackCount = srcData[7];

    // Calculate header size: 8 bytes base + 2 bytes per acknowledgment
    int headerSize = 8 + (ackCount * 2);

    // Return pointer to data after the header
    return srcData + headerSize;
}

bool UDPType2::validateUserID(uint32_t userID) {
    // Ensure userID fits in 30 bits (max value 0x3FFFFFFF)
    return (userID <= 0x3FFFFFFF);
}

bool UDPType2::validatePacketCounter(uint16_t packetCounter) {
    // All 16-bit values are valid
    return true;
}

bool UDPType2::addUser(uint32_t userID, const std::array<uint8_t, 32>& aesKey) {
    if (!validateUserID(userID)) {
        return false;
    }

    // Add user with AES key to the map
    UserData userData(userID, aesKey);
    userData.sendCounter2 = 0;    // Initialize packet counter for sending
    userData.receiveCounter2 = 0; // Initialize packet counter for receiving
    userData.baseAckNumber = 0;   // Initialize base of sliding window
    users[userID] = userData;

    return true;
}

bool UDPType2::removeUser(uint32_t userID) {
    if (!validateUserID(userID)) {
        return false;
    }

    // Remove pending packets for this user
    auto it = pendingPackets.begin();
    while (it != pendingPackets.end()) {
        if (it->userID == userID) {
            it = pendingPackets.erase(it);
        }
        else {
            ++it;
        }
    }

    // Remove user from the map (returns number of elements removed, 0 or 1)
    return users.erase(userID) > 0;
}

// Helper function to check if a sequence number is within the valid window,
// accounting for 16-bit overflow
bool isSequenceInWindow(uint16_t sequence, uint16_t base, uint16_t windowSize) {
    // Calculate distance considering overflow
    uint16_t distance = (sequence - base) & 0xFFFF;
    return distance < windowSize;
}

bool UDPType2::processReceivedPacket(const uint8_t* data, int length, SDLNet_Address* address, Uint16 port) {
    if (data == nullptr || length < 8) {
        return false;
    }

    if (!address) {
        return false;
    }

    // Increase reference count since we're using this address
    SDLNet_RefAddress(address);

    // Extract basic header information
    uint32_t userID = extractUserID(data);
    uint16_t packetCounter = extractPacketCounter(data);
    bool isAckOnly = isAckOnlyPacket(data);

    // Get user data
    auto userData = getUserData(userID);
    if (!userData) {
        SDLNet_UnrefAddress(address);
        return false; // Unknown user
    }

    // Store the address and port for future acknowledgments with proper reference counting
    userData->updateAddress(address);
    userData->lastPort = port;

    // Process any acknowledgments in this packet
    auto acks = extractAckPackets(data);
    for (uint16_t ackId : acks) {
        handleAcknowledgment(userID, ackId);
    }

    // If this is a data packet (not just ACKs), we need to acknowledge it
    if (!isAckOnly) {
        // Check if the packet is within our sliding window
        if (isSequenceInWindow(packetCounter, userData->baseAckNumber, MAX_PENDING_PACKETS)) {
            // Check if we already have this acknowledgment
            bool alreadyHave = false;
            for (uint16_t ack : userData->pendingAcks) {
                if (ack == packetCounter) {
                    alreadyHave = true;
                    break;
                }
            }

            // Add this packet to the list of packets to acknowledge if not already there
            if (!alreadyHave) {
                userData->pendingAcks.push_back(packetCounter);
            }

            // Update receive counter and possibly advance the window base
            // If this is the exact next packet we expected
            if (packetCounter == userData->receiveCounter2) {
                // Advance receive counter with overflow handling
                userData->receiveCounter2 = (userData->receiveCounter2 + 1) & 0xFFFF;

                // If we've filled a significant portion of our window, advance the base
                if (userData->pendingAcks.size() > MAX_PENDING_PACKETS / 2) {
                    userData->baseAckNumber = (userData->baseAckNumber + MAX_PENDING_PACKETS / 4) & 0xFFFF;
                }
            }
        }
    }

    // Decrease reference count after use
    SDLNet_UnrefAddress(address);

    return true;
}

bool UDPType2::handleAcknowledgment(uint32_t userID, uint16_t packetCounter) {
    // Find and remove the pending packet
    for (auto it = pendingPackets.begin(); it != pendingPackets.end(); ++it) {
        if (it->userID == userID && it->packetCounter == packetCounter) {
            pendingPackets.erase(it);
            return true;
        }
    }

    return false; // No matching packet found
}

void UDPType2::execute() {
    // Get current time
    uint32_t currentTime = SDL_GetTicks();

    // Process any users with pending acknowledgments
    for (auto& [userId, userData] : users) {
        if (!userData.pendingAcks.empty() && userData.lastAddress != nullptr) {
            // If we have pending acknowledgments and know where to send them
            sendAcknowledgmentOnly(userId, userData.lastAddress, userData.lastPort);
        }
    }

    // Check for packets that need retransmission
    for (auto it = pendingPackets.begin(); it != pendingPackets.end(); /* increment inside loop */) {
        // Calculate elapsed time since last transmission
        uint32_t elapsedTime = currentTime - it->lastSentTime;

        // If cooldown period has passed, attempt retransmission
        if (elapsedTime >= RETRANSMIT_COOLDOWN) {
            // Check if maximum retry count reached
            if (it->retryCount >= MAX_TRANSMIT_RETRY) {
                // Get user data to potentially advance the window
                auto userData = getUserData(it->userID);
                if (userData) {
                    // Check if this was the oldest packet in the window
                    if (it->packetCounter == userData->baseAckNumber) {
                        // Advance the base of the window since we're giving up on this packet
                        userData->baseAckNumber = (userData->baseAckNumber + 1) & 0xFFFF;
                    }
                }

                // Remove failed packet and advance iterator
                it = pendingPackets.erase(it);
                continue;
            }

            // Ensure we have a valid address before attempting retransmission
            if (!it->address) {
                // No valid address, remove the packet
                it = pendingPackets.erase(it);
                continue;
            }

            // Increase reference count for the address we're about to use
            SDLNet_RefAddress(it->address);

            // Reuse the original buffer handle for retransmission
            if (it->bufferHandle.valid()) {
                // Get the buffer data
                uint8_t* buffer = it->bufferHandle.data();

                // Update acknowledgment info in the header if there are any pending acks
                uint8_t oldAckCount = buffer[7];

                auto userData = getUserData(it->userID);
                if (userData && !userData->pendingAcks.empty()) {
                    // Calculate space available after the original header
                    uint8_t* ackPtr = buffer + 8 + (oldAckCount * 2);
                    int newAckCount = 0;

                    // Add up to MAX_ACKS_PER_PACKET acknowledgments
                    for (size_t i = 0; i < userData->pendingAcks.size() &&
                        newAckCount + oldAckCount < MAX_ACKS_PER_PACKET; i++) {

                        uint16_t ackId = userData->pendingAcks[i];

                        // Write each 16-bit ack packet ID (2 bytes each)
                        *ackPtr++ = (ackId >> 8) & 0xFF; // High byte
                        *ackPtr++ = ackId & 0xFF;       // Low byte
                        newAckCount++;
                    }

                    // Update total acknowledgment count in the header
                    buffer[7] = static_cast<uint8_t>(oldAckCount + newAckCount);

                    // Remove the acknowledgments that were sent
                    if (newAckCount > 0) {
                        if (newAckCount >= userData->pendingAcks.size()) {
                            userData->pendingAcks.clear();
                        }
                        else {
                            userData->pendingAcks.erase(
                                userData->pendingAcks.begin(),
                                userData->pendingAcks.begin() + newAckCount
                            );
                        }
                    }

                    // Update the header size for this retransmission
                    it->headerSize = 8 + ((oldAckCount + newAckCount) * 2);
                }

                // If we have data in our backup buffer, use it for retransmission
                if (!it->dataBackup.empty()) {
                    // Copy backup data after the header
                    memcpy(buffer + it->headerSize, it->dataBackup.data(), it->dataLength);

                    // Send the packet
                    node.sendRaw(buffer, it->headerSize + it->dataLength, it->address, it->port);
                }
                else {
                    // No data to send (should be rare)
                    node.sendRaw(buffer, it->headerSize, it->address, it->port);
                }

                // Update retry count and timestamp
                it->retryCount++;
                it->lastSentTime = currentTime;
            }

            // Decrease reference count after use
            SDLNet_UnrefAddress(it->address);

            // Advance iterator
            ++it;
        }
        else {
            // No action needed, advance iterator
            ++it;
        }
    }
}