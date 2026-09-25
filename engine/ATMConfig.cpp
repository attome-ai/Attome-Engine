#include "ATMConfig.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <set>
#include <variant>

// SECTION: engine_config

namespace {

using FieldMember =
    std::variant<int EngineConfig::*, float EngineConfig::*,
                 bool EngineConfig::*, std::string EngineConfig::*>;

struct Field {
  const char *path; // dotted path inside the JSON document
  FieldMember member;
};

#define ATM_FIELD(path, member) Field{path, &EngineConfig::member}

const Field kEngineFields[] = {
    ATM_FIELD("window.title", window_title),
    ATM_FIELD("window.width", window_width),
    ATM_FIELD("window.height", window_height),
    ATM_FIELD("window.resizable", window_resizable),
    ATM_FIELD("window.vsync", vsync),
    ATM_FIELD("world.width", world_width),
    ATM_FIELD("world.height", world_height),
    ATM_FIELD("grid.cell_size", grid_cell_size),
    ATM_FIELD("grid.node_reserve", grid_node_reserve),
    ATM_FIELD("grid.query_pad_cells", grid_query_pad_cells),
    ATM_FIELD("grid.static_chunk_size", static_chunk_size),
    ATM_FIELD("grid.hybrid_activation_margin", hybrid_activation_margin),
    ATM_FIELD("time.max_frame_dt", max_frame_dt),
    ATM_FIELD("time.fixed_timestep_hz", fixed_timestep_hz),
    ATM_FIELD("time.max_fixed_steps_per_frame", max_fixed_steps_per_frame),
    ATM_FIELD("audio.enabled", audio_enabled),
    ATM_FIELD("audio.max_voices", audio_max_voices),
    ATM_FIELD("audio.master_volume", audio_master_volume),
};

#undef ATM_FIELD

// Top-level sections owned by other systems that may share the same file.
const char *const kForeignSections[] = {"input", "game"};

bool read_field(const atm::Json &value, const Field &field, EngineConfig &cfg) {
  return std::visit(
      [&](auto member) {
        using T = std::remove_reference_t<decltype(cfg.*member)>;
        return atm::TunableTraits<T>::fromJson(value, cfg.*member);
      },
      field.member);
}

atm::Json write_field(const Field &field, const EngineConfig &cfg) {
  return std::visit(
      [&](auto member) {
        using T = std::remove_cv_t<std::remove_reference_t<decltype(cfg.*member)>>;
        return atm::TunableTraits<T>::toJson(cfg.*member);
      },
      field.member);
}

void set_path(atm::Json &root, const std::string &dotted, atm::Json value) {
  atm::Json *node = &root;
  size_t start = 0;
  while (true) {
    const size_t dot = dotted.find('.', start);
    const std::string part = dotted.substr(start, dot - start);
    if (dot == std::string::npos) {
      (*node)[part] = std::move(value);
      return;
    }
    node = &(*node)[part];
    start = dot + 1;
  }
}

void clamp_config(EngineConfig &c) {
  c.window_width = std::max(c.window_width, 1);
  c.window_height = std::max(c.window_height, 1);
  c.world_width = std::max(c.world_width, 1);
  c.world_height = std::max(c.world_height, 1);
  c.grid_cell_size = std::max(c.grid_cell_size, 1);
  c.grid_node_reserve = std::max(c.grid_node_reserve, 0);
  c.grid_query_pad_cells = std::max(c.grid_query_pad_cells, 0);
  c.static_chunk_size = std::max(c.static_chunk_size, 1);
  c.max_frame_dt = std::max(c.max_frame_dt, 0.0f);
  c.fixed_timestep_hz = std::max(c.fixed_timestep_hz, 0.0f);
  c.max_fixed_steps_per_frame = std::max(c.max_fixed_steps_per_frame, 1);
  c.audio_max_voices = std::clamp(c.audio_max_voices, 1, 256);
  c.audio_master_volume = std::clamp(c.audio_master_volume, 0.0f, 4.0f);
}

} // namespace

bool engine_config_from_json(const atm::Json &json, EngineConfig &config,
                             std::string *error) {
  if (!json.isObject()) {
    if (error)
      *error = "engine config root must be an object";
    return false;
  }

  bool ok = true;
  std::set<std::string> known;
  for (const Field &field : kEngineFields) {
    known.insert(field.path);
    const atm::Json *value = json.findPath(field.path);
    if (!value)
      continue;
    if (!read_field(*value, field, config)) {
      SDL_Log("[config] '%s' has the wrong type; keeping %s", field.path,
              write_field(field, config).dump(0).c_str());
      if (error && ok)
        *error = std::string("wrong type for '") + field.path + "'";
      ok = false;
    }
  }

  for (const auto &[section, node] : json.asObject()) {
    bool foreign = false;
    for (const char *name : kForeignSections)
      foreign = foreign || section == name;
    if (foreign)
      continue;
    if (!node.isObject()) {
      SDL_Log("[config] unknown engine key '%s'", section.c_str());
      continue;
    }
    for (const auto &[key, unused] : node.asObject()) {
      (void)unused;
      if (!known.count(section + "." + key))
        SDL_Log("[config] unknown engine key '%s.%s'", section.c_str(),
                key.c_str());
    }
  }

  clamp_config(config);
  return ok;
}

bool engine_config_load(const std::string &path, EngineConfig &config,
                        std::string *error) {
  atm::Json json;
  if (!atm::Json::parseFile(path, json, error))
    return false;
  // A shared file may keep the engine settings under "engine".
  if (const atm::Json *engine = json.find("engine"))
    return engine_config_from_json(*engine, config, error);
  return engine_config_from_json(json, config, error);
}

atm::Json engine_config_to_json(const EngineConfig &config) {
  atm::Json root = atm::Json::object();
  for (const Field &field : kEngineFields)
    set_path(root, field.path, write_field(field, config));
  return root;
}

namespace atm {

std::string resolve_path(const std::string &relative_path) {
  if (relative_path.empty())
    return relative_path;
  const bool absolute = relative_path[0] == '/' || relative_path[0] == '\\' ||
                        (relative_path.size() > 1 && relative_path[1] == ':');
  if (absolute || SDL_GetPathInfo(relative_path.c_str(), nullptr))
    return relative_path;
  const char *base = SDL_GetBasePath();
  if (!base)
    return relative_path;
  const std::string candidate = std::string(base) + relative_path;
  if (SDL_GetPathInfo(candidate.c_str(), nullptr))
    return candidate;
  return relative_path;
}

// SECTION: tunables_registry

Tunables &Tunables::instance() {
  static Tunables tunables;
  return tunables;
}

bool Tunables::addEntry(Entry entry) {
  for (const Entry &existing : entries_) {
    if (existing.key == entry.key) {
      SDL_Log("[tunables] duplicate key '%s' (give one of them its own "
              "ATM_TUNABLE_SECTION)",
              entry.key.c_str());
      return false;
    }
  }
  entries_.push_back(std::move(entry));
  return true;
}

bool Tunables::apply(const Json &root, std::string *error) {
  if (!root.isObject()) {
    if (error)
      *error = "tunables root must be an object";
    return false;
  }

  bool ok = true;
  for (const Entry &entry : entries_) {
    const Json *value = root.findPath(entry.key);
    if (!value)
      continue;
    if (!entry.from_json(*value, entry.ptr)) {
      SDL_Log("[tunables] '%s' has the wrong type or is out of range; keeping "
              "%s",
              entry.key.c_str(), entry.to_json(entry.ptr).dump(0).c_str());
      if (error && ok)
        *error = "wrong type for '" + entry.key + "'";
      ok = false;
    }
  }
  reportUnknownKeys(root, "");

  for (const auto &listener : listeners_)
    listener();
  return ok;
}

void Tunables::reportUnknownKeys(const Json &node,
                                 const std::string &prefix) const {
  for (const auto &[key, child] : node.asObject()) {
    const std::string path = prefix.empty() ? key : prefix + "." + key;
    bool registered = false;
    bool is_prefix = false;
    for (const Entry &entry : entries_) {
      if (entry.key == path) {
        registered = true;
        break;
      }
      if (entry.key.size() > path.size() &&
          entry.key.compare(0, path.size(), path) == 0 &&
          entry.key[path.size()] == '.')
        is_prefix = true;
    }
    if (registered)
      continue;
    if (is_prefix && child.isObject()) {
      reportUnknownKeys(child, path);
      continue;
    }
    if (path == "engine" || path == "input")
      continue; // sections owned by EngineConfig / InputMap
    SDL_Log("[tunables] unknown key '%s' (typo?)", path.c_str());
  }
}

static int64_t file_mtime(const std::string &path) {
  SDL_PathInfo info;
  if (!SDL_GetPathInfo(path.c_str(), &info))
    return 0;
  return static_cast<int64_t>(info.modify_time);
}

bool Tunables::loadFile(const std::string &path, std::string *error) {
  Json root;
  if (!Json::parseFile(path, root, error))
    return false;
  path_ = path;
  loaded_mtime_ = file_mtime(path);
  last_check_ms_ = SDL_GetTicks();
  return apply(root, error);
}

bool Tunables::reloadIfChanged(uint64_t check_interval_ms) {
  if (path_.empty())
    return false;
  const uint64_t now = SDL_GetTicks();
  if (now - last_check_ms_ < check_interval_ms)
    return false;
  last_check_ms_ = now;

  const int64_t mtime = file_mtime(path_);
  if (mtime == 0 || mtime == loaded_mtime_)
    return false;

  Json root;
  std::string error;
  if (!Json::parseFile(path_, root, &error)) {
    // Keep the old values; the file is probably mid-save or has a typo.
    SDL_Log("[tunables] reload failed: %s", error.c_str());
    loaded_mtime_ = mtime;
    return false;
  }
  loaded_mtime_ = mtime;
  apply(root, nullptr);
  SDL_Log("[tunables] reloaded %s", path_.c_str());
  return true;
}

Json Tunables::toJson() const {
  Json root = Json::object();
  for (const Entry &entry : entries_)
    set_path(root, entry.key, entry.to_json(entry.ptr));
  return root;
}

bool Tunables::writeFile(const std::string &path) const {
  return write_text_file(path, toJson().dump(2));
}

void Tunables::addReloadListener(std::function<void()> listener) {
  listeners_.push_back(std::move(listener));
}

} // namespace atm
