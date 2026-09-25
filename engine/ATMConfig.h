#ifndef ATM_CONFIG_H
#define ATM_CONFIG_H

// Runtime configuration.
//
// 1. EngineConfig — window, world, spatial grid, timestep and audio settings.
//    Load it from JSON and pass it to engine_create_with_config().
//
// 2. Tunables — game values that live in a JSON file instead of being
//    hard-coded. Declare them in a header:
//
//      namespace my_game {
//      ATM_TUNABLE_SECTION("player");
//      ATM_TUNABLE(float, kSpeed, 540.0f);
//      ATM_TUNABLE(int, kLives, 3);
//      }
//
//    then at startup:
//
//      atm::Tunables::instance().loadFile(atm::resolve_path("config/game.json"));
//
//    The JSON mirrors the sections:  { "player": { "kSpeed": 600, "kLives": 5 } }
//    Missing keys keep their compiled-in default. Call reloadIfChanged() once
//    per frame to pick up edits while the game runs (it only checks the file
//    timestamp twice a second).

#include "ATMJson.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

// SECTION: engine_config

struct EngineConfig {
  // window
  std::string window_title = "Attome Engine";
  int window_width = 1280;
  int window_height = 720;
  bool window_resizable = false;
  bool vsync = false;

  // world / spatial grid
  int world_width = 50000;
  int world_height = 50000;
  int grid_cell_size = 64;
  int grid_node_reserve = 3200000;
  int grid_query_pad_cells = 4;
  int static_chunk_size = 512;
  float hybrid_activation_margin = 150.0f;

  // time
  float max_frame_dt = 0.1f;
  // 0 = variable timestep (one update per frame with the real frame time).
  // > 0 = fixed-rate simulation; engine_update() runs 0..N steps per frame.
  float fixed_timestep_hz = 0.0f;
  int max_fixed_steps_per_frame = 4;

  // audio
  bool audio_enabled = true;
  int audio_max_voices = 32;
  float audio_master_volume = 1.0f;
};

// Reads known fields from `json` into `config` (unset fields are untouched).
// Unknown keys are reported through SDL_Log so typos don't go unnoticed.
bool engine_config_from_json(const atm::Json &json, EngineConfig &config,
                             std::string *error = nullptr);
bool engine_config_load(const std::string &path, EngineConfig &config,
                        std::string *error = nullptr);
atm::Json engine_config_to_json(const EngineConfig &config);

namespace atm {

// Resolves a relative path against the executable directory when it does not
// exist relative to the working directory. Absolute paths are returned as-is.
std::string resolve_path(const std::string &relative_path);

// SECTION: tunable_traits

template <typename T, typename Enable = void> struct TunableTraits;

template <> struct TunableTraits<bool> {
  static bool fromJson(const Json &j, bool &out) {
    if (!j.isBool())
      return false;
    out = j.asBool();
    return true;
  }
  static Json toJson(bool v) { return Json(v); }
};

template <typename T>
struct TunableTraits<T, std::enable_if_t<std::is_integral_v<T> &&
                                         !std::is_same_v<T, bool>>> {
  static bool fromJson(const Json &j, T &out) {
    if (!j.isNumber())
      return false;
    const double v = j.asNumber();
    if (v != static_cast<double>(static_cast<int64_t>(v)) ||
        v < static_cast<double>(std::numeric_limits<T>::lowest()) ||
        v > static_cast<double>(std::numeric_limits<T>::max()))
      return false;
    out = static_cast<T>(v);
    return true;
  }
  static Json toJson(T v) { return Json(static_cast<double>(v)); }
};

template <typename T>
struct TunableTraits<T, std::enable_if_t<std::is_floating_point_v<T>>> {
  static bool fromJson(const Json &j, T &out) {
    if (!j.isNumber())
      return false;
    out = static_cast<T>(j.asNumber());
    return true;
  }
  static Json toJson(T v) { return Json(static_cast<double>(v)); }
};

template <typename T>
struct TunableTraits<T, std::enable_if_t<std::is_enum_v<T>>> {
  static bool fromJson(const Json &j, T &out) {
    using U = std::underlying_type_t<T>;
    U raw{};
    if (!TunableTraits<U>::fromJson(j, raw))
      return false;
    out = static_cast<T>(raw);
    return true;
  }
  static Json toJson(T v) {
    return TunableTraits<std::underlying_type_t<T>>::toJson(
        static_cast<std::underlying_type_t<T>>(v));
  }
};

template <> struct TunableTraits<std::string> {
  static bool fromJson(const Json &j, std::string &out) {
    if (!j.isString())
      return false;
    out = j.asString();
    return true;
  }
  static Json toJson(const std::string &v) { return Json(v); }
};

// SECTION: tunables_registry

class Tunables {
public:
  static Tunables &instance();

  template <typename T>
  bool add(const char *section, const char *name, T *value) {
    Entry entry;
    entry.key = (section && section[0]) ? std::string(section) + "." + name
                                        : std::string(name);
    entry.ptr = value;
    entry.from_json = [](const Json &j, void *p) {
      return TunableTraits<T>::fromJson(j, *static_cast<T *>(p));
    };
    entry.to_json = [](const void *p) {
      return TunableTraits<T>::toJson(*static_cast<const T *>(p));
    };
    return addEntry(std::move(entry));
  }

  // Loads and applies a JSON file and remembers it for reloadIfChanged().
  bool loadFile(const std::string &path, std::string *error = nullptr);
  // Applies values from an already-parsed document.
  bool apply(const Json &root, std::string *error = nullptr);
  // Re-reads the loaded file if its timestamp changed. Throttled to one
  // timestamp check every `check_interval_ms`. Returns true when reloaded.
  bool reloadIfChanged(uint64_t check_interval_ms = 500);

  // Current values as a nested JSON document (use it to write a default file).
  Json toJson() const;
  bool writeFile(const std::string &path) const;

  // Called after every successful load/reload, e.g. to recompute values
  // derived from tunables.
  void addReloadListener(std::function<void()> listener);

  size_t count() const { return entries_.size(); }
  const std::string &loadedPath() const { return path_; }

private:
  struct Entry {
    std::string key;
    void *ptr = nullptr;
    bool (*from_json)(const Json &, void *) = nullptr;
    Json (*to_json)(const void *) = nullptr;
  };

  bool addEntry(Entry entry);
  void reportUnknownKeys(const Json &node, const std::string &prefix) const;

  std::vector<Entry> entries_;
  std::vector<std::function<void()>> listeners_;
  std::string path_;
  int64_t loaded_mtime_ = 0;
  uint64_t last_check_ms_ = 0;
};

} // namespace atm

// Default section for tunables declared outside any ATM_TUNABLE_SECTION.
inline constexpr const char *atm_tunable_section = "";

// Sets the JSON section for the ATM_TUNABLEs in the enclosing namespace.
// Nested namespaces inherit the parent section unless they declare their own.
#define ATM_TUNABLE_SECTION(section_name)                                      \
  inline constexpr const char *atm_tunable_section = section_name

// Declares a mutable global with a compiled-in default and registers it under
// "<section>.<name>". Brace initialisers work: ATM_TUNABLE(Rgba8, kC, {1,2,3,4})
#define ATM_TUNABLE(type, name, ...)                                           \
  inline type name = __VA_ARGS__;                                              \
  inline const bool atm_tunable_registered_##name =                            \
      ::atm::Tunables::instance().add<type>(atm_tunable_section, #name, &name)

#endif // ATM_CONFIG_H
