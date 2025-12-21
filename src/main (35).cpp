
#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <iostream>
#include <ATMEngine.h>



// SDL callback functions for SDL_MAIN_USE_CALLBACKS
SDL_AppResult SDL_AppInit(void** appstate, int argc, char* argv[]) {
    std::cout << "Initializing Vulkan application..." << std::endl;

    if (!g_app.init()) {
        return SDL_APP_FAILURE;
    }

    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult result) {
    std::cout << "Shutting down Vulkan application..." << std::endl;
    g_app.cleanup();
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* eve) {
    if (eve->type == SDL_EVENT_WINDOW_RESIZED) {
        g_app.setFramebufferResized();
    }
    else if (eve->type == SDL_EVENT_QUIT) {
        return SDL_APP_FAILURE;
    }
    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appstate) {
    // FPS tracking variables
    static Uint64 frameCount = 0;
    static Uint64 lastTime = SDL_GetTicks();
    static double fps = 0.0;

    // Draw the frame
    g_app.drawFrame();
    frameCount++;

    // Calculate FPS every second
    Uint64 currentTime = SDL_GetTicks();
    if (currentTime - lastTime >= 1000) {
        fps = frameCount * 1000.0 / (currentTime - lastTime);
        std::cout << "FPS: " << fps << std::endl;
        frameCount = 0;
        lastTime = currentTime;
    }

    return SDL_APP_CONTINUE;
}