// Single-player (--local): the authoritative zone server runs in-process on
// its own thread, and the client connects to it over loopback UDP exactly as
// it would to a remote server (same code path, same protocol).

#include "App.h"

#include "server/ZoneServer.h"

#include "../../engine/ATMConfig.h"

#include <SDL3/SDL.h>

namespace ao::client {

void App::startLocalServer() {
  std::string err;
  server::ServerConfig sc = server::ServerConfig::load(atm::resolve_path(cfg_.serverConfigPath), &err);
  if (!err.empty())
    SDL_Log("[client] local server config: %s (using defaults)", err.c_str());
  sc.port = 0;                                  // any free port
  sc.viewRadiusChunks = cfg_.viewRadiusChunks;  // sync edits across the whole view
  sc.verbose = false;

  localServer_ = std::make_unique<server::ZoneServer>();
  if (!localServer_->start(sc, &err)) {
    SDL_Log("[client] local server failed to start: %s", err.c_str());
    localServer_.reset();
    return;
  }
  cfg_.host = "127.0.0.1";
  cfg_.port = localServer_->port();
  localServerStop_ = false;
  localServerThread_ = std::thread([this] { localServer_->run(localServerStop_); });
  SDL_Log("[client] local server running on port %u", unsigned(cfg_.port));
}

void App::stopLocalServer() {
  if (!localServer_)
    return;
  localServerStop_ = true;
  if (localServerThread_.joinable())
    localServerThread_.join();
  localServer_->stop();
  localServer_.reset();
}

} // namespace ao::client
