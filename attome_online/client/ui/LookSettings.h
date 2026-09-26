#pragma once

// Tunable look of the game (graphics panel, F10). Saved to / loaded from
// config/graphics.json next to the executable. One field table drives the
// defaults, the JSON file and the panel, so they never disagree.

#include "../../../engine/render/Renderer.h"

#include <string>
#include <vector>

namespace ao::client {

struct LookSettings {
  atm::render::Environment env; // look fields (exposure, sun, haze, ...)
  float sunAzimuth = 118.6f;    // degrees clockwise from north (-Z)
  float sunElevation = 44.6f;   // degrees above the horizon
  float decorDensity = 1.0f;    // grass tufts / flowers / pebbles multiplier
  float fov = 70.0f;            // vertical field of view (degrees)
  bool vsync = true;            // cap at the display refresh (off = uncapped FPS)

  LookSettings(); // the shipped look

  struct Field {
    const char *key;   // JSON key
    const char *label; // panel label
    const char *group; // panel section
    float *value;
    float min, max;
    const char *help;  // tooltip
  };
  std::vector<Field> fields(); // pointers into *this

  bool load(const std::string &path, std::string *error = nullptr);
  bool save(const std::string &path) const;
};

} // namespace ao::client
