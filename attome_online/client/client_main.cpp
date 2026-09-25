// Attome Online client.
//   ao_client                      connect using config/client.json
//   ao_client --local              single player: starts a server in-process
//   ao_client --host H --port P --name N

#include "App.h"

#include "../../engine/ATMConfig.h"

#include <SDL3/SDL.h> // SDL_ShowSimpleMessageBox
#include <SDL3/SDL_main.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char **argv) {
  ao::client::ClientConfig cfg;
  std::string configPath = atm::resolve_path("config/client.json");
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc)
      configPath = argv[++i];

  std::string err;
  if (!ao::client::loadClientConfig(configPath, cfg, &err))
    std::fprintf(stderr, "[client] %s (using defaults)\n", err.c_str());

  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (std::strcmp(a, "--local") == 0) cfg.local = true;
    else if (std::strcmp(a, "--host") == 0) cfg.host = next();
    else if (std::strcmp(a, "--port") == 0) cfg.port = uint16_t(std::atoi(next()));
    else if (std::strcmp(a, "--name") == 0) cfg.name = next();
    else if (std::strcmp(a, "--validation") == 0) cfg.render.validation = true;
    else if (std::strcmp(a, "--no-vsync") == 0) cfg.render.vsync = false;
    else if (std::strcmp(a, "--config") == 0) ++i;
  }

  ao::client::App app;
  if (!app.init(cfg, &err)) {
    std::fprintf(stderr, "[client] startup failed: %s\n", err.c_str());
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Attome Online", err.c_str(), nullptr);
    return 1;
  }
  const int code = app.run();
  app.shutdown();
  return code;
}
