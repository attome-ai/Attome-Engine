#pragma once

#include <SDL3/SDL.h>

// Texture loading functions
SDL_Surface* load_texture(const char* filename);
SDL_Surface* create_colored_surface(int width, int height, Uint8 r, Uint8 g, Uint8 b);
