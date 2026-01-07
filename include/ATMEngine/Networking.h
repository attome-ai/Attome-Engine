#pragma once

#include "ATMEngine/GameState.h"

// Network initialization and management
void initializeNetwork(GameState* state);
void sendLoginRequest(GameState* state);
void sendRegisterRequest(GameState* state);
void sendGuestLoginRequest(GameState* state);
void handleAuthenticationMessages(GameState* state);
void sendHeartbeatIfNeeded(GameState* state);
void sendDisconnectMessage(GameState* state);
