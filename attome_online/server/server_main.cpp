// ao_server: headless authoritative zone server.
//
//   ao_server [--config config/server.json] [--port 27015]
//
// Runs until Ctrl+C; prints stats every 10 s.

#include "ZoneServer.h"

#include "../shared/GameTypes.h"

#include "../../engine/ATMConfig.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {

std::atomic<bool> g_stop{false};

void onSignal(int) { g_stop.store(true); }

} // namespace

int main(int argc, char **argv) {
  std::string configPath = "config/server.json";
  int portOverride = -1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      configPath = argv[++i];
    } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      portOverride = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::printf("usage: ao_server [--config path] [--port n]\n");
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }

  std::string err;
  ao::server::ServerConfig cfg = ao::server::ServerConfig::load(atm::resolve_path(configPath), &err);
  if (!err.empty()) std::fprintf(stderr, "[server] config: %s (using defaults)\n", err.c_str());
  if (portOverride >= 0 && portOverride <= 65535) cfg.port = uint16_t(portOverride);

  // Shared gameplay tunables (movement etc.): must match the client's values
  // or prediction and the server simulation diverge. The in-process --local
  // server shares the client's already-loaded tunables, so this lives here and
  // not in ZoneServer.
  {
    std::string terr;
    if (!atm::Tunables::instance().loadFile(atm::resolve_path("config/game.json"), &terr))
      std::fprintf(stderr, "[server] config/game.json: %s (using compiled-in tunables)\n", terr.c_str());
  }

  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  ao::server::ZoneServer server;
  if (!server.start(cfg, &err)) {
    std::fprintf(stderr, "[server] failed to start: %s\n", err.c_str());
    return 1;
  }

  using clock = std::chrono::steady_clock;
  const auto dt = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / ao::kSimHz));
  auto next = clock::now();
  auto lastStats = clock::now();
  ao::server::ServerStats prev = server.stats();

  while (!g_stop.load() && server.running()) {
    server.tick();
    atm::Tunables::instance().reloadIfChanged(1000); // throttled: one timestamp check per second
    next += dt;
    const auto now = clock::now();
    if (now < next) std::this_thread::sleep_until(next);
    else if (now - next > dt * 10) next = now;

    if (now - lastStats >= std::chrono::seconds(10)) {
      const double secs = std::chrono::duration<double>(now - lastStats).count();
      const ao::server::ServerStats st = server.stats();
      std::printf("[server] tick %u | players %u | entities %u (monsters %u) | tick ms avg %.2f p99 %.2f max %.2f | "
                  "out %.1f KB/s in %.1f KB/s | chunks %zu (edited %zu) | malformed %llu\n",
                  st.tick, st.players, st.entities, st.monsters, st.tickMsAvg, st.tickMsP99, st.tickMsMax,
                  double(st.bytesSent - prev.bytesSent) / 1024.0 / secs,
                  double(st.bytesReceived - prev.bytesReceived) / 1024.0 / secs, st.loadedChunks,
                  st.editedChunks, (unsigned long long)st.malformedPackets);
      std::fflush(stdout);
      prev = st;
      lastStats = now;
    }
  }
  server.stop();
  return 0;
}
