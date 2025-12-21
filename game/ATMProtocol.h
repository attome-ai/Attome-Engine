// ATMProtocol.h
#ifndef ATM_PROTOCOL_H
#define ATM_PROTOCOL_H

#include <cstdint>
#include <string>

// Message types
enum MessageType : uint8_t {
    MSG_LOGIN_REQUEST,
    MSG_REGISTER_REQUEST,
    MSG_GUEST_LOGIN_REQUEST,
    MSG_LOGIN_RESPONSE,
    MSG_REGISTER_RESPONSE,
    MSG_HEARTBEAT,
    MSG_DISCONNECT
};

// Response status
enum ResponseStatus : uint8_t {
    STATUS_SUCCESS = 0,
    STATUS_INVALID_CREDENTIALS,
    STATUS_USER_ALREADY_EXISTS,
    STATUS_SERVER_ERROR,
    STATUS_USER_BANNED
};

// Maximum sizes
constexpr size_t MAX_USERNAME_LENGTH = 32;
constexpr size_t MAX_PASSWORD_LENGTH = 32;
constexpr size_t MAX_ERROR_LENGTH = 128;

// Message structures
#pragma pack(push, 1)  // Ensure consistent packing across platforms

// Base message header
struct MessageHeader {
    MessageType type;
    uint16_t length;
};

// Login request message
struct LoginRequestMsg {
    MessageHeader header;
    char username[MAX_USERNAME_LENGTH];
    char password[MAX_PASSWORD_LENGTH];
};

// Register request message
struct RegisterRequestMsg {
    MessageHeader header;
    char username[MAX_USERNAME_LENGTH];
    char password[MAX_PASSWORD_LENGTH];
};

// Guest login request
struct GuestLoginRequestMsg {
    MessageHeader header;
};

// Response message
struct LoginResponseMsg {
    MessageHeader header;
    ResponseStatus status;
    uint16_t userId;
    char errorMsg[MAX_ERROR_LENGTH];
};

// Heartbeat message
struct HeartbeatMsg {
    MessageHeader header;
    uint16_t userId;
};

#pragma pack(pop)

#endif // ATM_PROTOCOL_H