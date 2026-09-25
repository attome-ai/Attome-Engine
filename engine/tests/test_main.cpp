#include "atm_test.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

int main(int argc, char **argv) {
  // Headless: tests need a renderer and an audio device but no real window
  // or sound card.
  SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "dummy");
  SDL_SetHint(SDL_HINT_RENDER_DRIVER, "software");
  SDL_SetHint(SDL_HINT_AUDIO_DRIVER, "dummy");
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    std::printf("SDL_Init failed: %s\n", SDL_GetError());
    return 1;
  }
  const int result = atm_test::run_all(argc, argv);
  SDL_Quit();
  return result;
}
