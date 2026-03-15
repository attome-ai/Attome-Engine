#include "levels/SaveState.h"

#include "Constants.h"
#include "characters/CharacterId.h"
#include "shop/RelicSystem.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#else
#include <filesystem>
#include <fstream>
#include <sstream>
#endif

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace tower_swarm {
namespace {

constexpr int kSaveVersion = 1;

#if defined(__EMSCRIPTEN__)
EM_JS(void, ts_local_storage_set, (const char *key, const char *value), {
  try {
    localStorage.setItem(UTF8ToString(key), UTF8ToString(value));
  } catch (e) {
  }
});

EM_JS(int, ts_local_storage_get_len, (const char *key), {
  try {
    var v = localStorage.getItem(UTF8ToString(key));
    if (!v)
      return 0;
    return lengthBytesUTF8(v) + 1;
  } catch (e) {
    return 0;
  }
});

EM_JS(void, ts_local_storage_get, (const char *key, char *out, int out_len), {
  try {
    var v = localStorage.getItem(UTF8ToString(key));
    if (!v) {
      if (out_len > 0)
        HEAPU8[out] = 0;
      return;
    }
    stringToUTF8(v, out, out_len);
  } catch (e) {
    if (out_len > 0)
      HEAPU8[out] = 0;
  }
});

std::string platform_load() {
  const int len = ts_local_storage_get_len(SaveState::kStorageKey);
  if (len <= 0) {
    return {};
  }
  std::string data(static_cast<std::size_t>(len), '\0');
  ts_local_storage_get(SaveState::kStorageKey, data.data(), len);
  if (!data.empty() && data.back() == '\0') {
    data.pop_back();
  }
  return data;
}

bool platform_save(std::string_view data) {
  const std::string tmp(data);
  ts_local_storage_set(SaveState::kStorageKey, tmp.c_str());
  return true;
}
#else
std::filesystem::path platform_path() {
  return std::filesystem::path("tower_swarm_save.json");
}

std::string platform_load() {
  std::ifstream f(platform_path(), std::ios::in | std::ios::binary);
  if (!f) {
    return {};
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool platform_save(std::string_view data) {
  std::ofstream f(platform_path(),
                  std::ios::out | std::ios::binary | std::ios::trunc);
  if (!f) {
    return false;
  }
  f.write(data.data(), static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(f);
}
#endif

void append_escaped(std::string &out, std::string_view s) {
  out.push_back('"');
  for (const char ch : s) {
    switch (ch) {
    case '\\':
      out += "\\\\\\\\";
      break;
    case '"':
      out += "\\\\\"";
      break;
    case '\n':
      out += "\\\\n";
      break;
    case '\r':
      out += "\\\\r";
      break;
    case '\t':
      out += "\\\\t";
      break;
    default:
      out.push_back(ch);
      break;
    }
  }
  out.push_back('"');
}

struct JsonCursor final {
  std::string_view s{};
  std::size_t i{0};

  bool eof() const { return i >= s.size(); }
  char peek() const { return eof() ? '\0' : s[i]; }

  void skip_ws() {
    while (!eof() && static_cast<unsigned char>(s[i]) <= 0x20) {
      i++;
    }
  }

  bool consume(char c) {
    skip_ws();
    if (peek() != c) {
      return false;
    }
    i++;
    return true;
  }

  bool parse_string(std::string &out) {
    skip_ws();
    if (peek() != '"') {
      return false;
    }
    i++;
    out.clear();
    while (!eof()) {
      const char ch = s[i++];
      if (ch == '"') {
        return true;
      }
      if (ch == '\\') {
        if (eof()) {
          return false;
        }
        const char esc = s[i++];
        switch (esc) {
        case '"':
        case '\\':
        case '/':
          out.push_back(esc);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          // Minimal unicode escape support: skip 4 hex digits and emit '?'
          // since our save strings are ASCII identifiers.
          for (int k = 0; k < 4; ++k) {
            if (eof() || !std::isxdigit(static_cast<unsigned char>(s[i]))) {
              return false;
            }
            i++;
          }
          out.push_back('?');
          break;
        }
        default:
          return false;
        }
        continue;
      }
      out.push_back(ch);
    }
    return false;
  }

  bool parse_int(std::int32_t &out) {
    skip_ws();
    const std::size_t start = i;
    if (peek() == '-' || peek() == '+') {
      i++;
    }
    bool any = false;
    while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
      any = true;
      i++;
    }
    if (!any) {
      i = start;
      return false;
    }
    const std::string_view token = s.substr(start, i - start);
    std::int32_t v = 0;
    const auto res = std::from_chars(token.data(), token.data() + token.size(), v);
    if (res.ec != std::errc()) {
      return false;
    }
    out = v;
    return true;
  }

  bool skip_value() {
    skip_ws();
    const char ch = peek();
    if (ch == '{') {
      i++;
      skip_ws();
      if (consume('}')) {
        return true;
      }
      std::string key;
      while (!eof()) {
        if (!parse_string(key)) {
          return false;
        }
        if (!consume(':')) {
          return false;
        }
        if (!skip_value()) {
          return false;
        }
        skip_ws();
        if (consume(',')) {
          continue;
        }
        if (consume('}')) {
          return true;
        }
        return false;
      }
      return false;
    }
    if (ch == '[') {
      i++;
      skip_ws();
      if (consume(']')) {
        return true;
      }
      while (!eof()) {
        if (!skip_value()) {
          return false;
        }
        skip_ws();
        if (consume(',')) {
          continue;
        }
        if (consume(']')) {
          return true;
        }
        return false;
      }
      return false;
    }
    if (ch == '"') {
      std::string tmp;
      return parse_string(tmp);
    }
    if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-' || ch == '+') {
      std::int32_t tmp = 0;
      return parse_int(tmp);
    }
    if (s.substr(i, 4) == "true") {
      i += 4;
      return true;
    }
    if (s.substr(i, 5) == "false") {
      i += 5;
      return true;
    }
    if (s.substr(i, 4) == "null") {
      i += 4;
      return true;
    }
    return false;
  }
};

} // namespace

PersistentSnapshot SaveState::snapshotPersistent(const GameState &state) {
  PersistentSnapshot snap{};
  snap.max_level_reached = std::max(1, state.max_level_reached);
  snap.stars_per_level = state.stars_per_level;
  snap.essence = std::max(0, state.essence);
  snap.base_hp = std::max(0, state.base_hp);
  snap.next_level_base_hp_target = state.next_level_base_hp_target;
  snap.shards = std::max(0, state.shards);
  snap.player_level = std::max(1, state.player_level);
  snap.player_xp = std::max(0, state.player_xp);

  snap.lifetime_levels_completed = std::max(0, state.lifetime_levels_completed);
  snap.lifetime_stars_earned = std::max(0, state.lifetime_stars_earned);
  snap.lifetime_bosses_killed = std::max(0, state.lifetime_bosses_killed);
  snap.lifetime_merges = std::max(0, state.lifetime_merges);

  snap.unlocked_characters = state.unlocked_characters;
  snap.mastery_ranks = state.mastery_ranks;

  snap.roster = state.roster;
  snap.relic_unlocked = state.relic_unlocked;
  snap.equipped_relics = state.equipped_relics;
  return snap;
}

void SaveState::restorePersistent(GameState &io_state,
                                  const PersistentSnapshot &snapshot) {
  io_state.max_level_reached = std::max(1, snapshot.max_level_reached);
  io_state.stars_per_level = snapshot.stars_per_level;
  io_state.essence = std::max(0, snapshot.essence);
  io_state.base_hp = std::max(0, snapshot.base_hp);
  io_state.next_level_base_hp_target = snapshot.next_level_base_hp_target;
  io_state.shards = std::max(0, snapshot.shards);
  io_state.player_level = std::max(1, snapshot.player_level);
  io_state.player_xp = std::max(0, snapshot.player_xp);

  io_state.lifetime_levels_completed =
      std::max(0, snapshot.lifetime_levels_completed);
  io_state.lifetime_stars_earned = std::max(0, snapshot.lifetime_stars_earned);
  io_state.lifetime_bosses_killed =
      std::max(0, snapshot.lifetime_bosses_killed);
  io_state.lifetime_merges = std::max(0, snapshot.lifetime_merges);

  io_state.unlocked_characters = snapshot.unlocked_characters;
  io_state.mastery_ranks = snapshot.mastery_ranks;

  io_state.roster = snapshot.roster;
  io_state.relic_unlocked = snapshot.relic_unlocked;
  io_state.equipped_relics = snapshot.equipped_relics;
  if (io_state.roster.empty()) {
    io_state.resetToNewProfile();
  }
  io_state.sanitizeCharacterUnlocks();
  io_state.sanitizeMasteries();
  io_state.sanitizeMetaProgression();
  RelicSystem::sanitizePersistent(io_state);
}

std::string SaveState::toJson(const GameState &state) {
  std::string out;
  out.reserve(2048);

  out += "{";
  out += "\"version\":";
  out += std::to_string(kSaveVersion);

  out += ",\"max_level_reached\":";
  out += std::to_string(std::max(1, state.max_level_reached));

  out += ",\"stars_per_level\":[";
  for (std::size_t i = 0; i < state.stars_per_level.size(); ++i) {
    if (i != 0) {
      out.push_back(',');
    }
    out += std::to_string(static_cast<int>(state.stars_per_level[i]));
  }
  out.push_back(']');

  out += ",\"essence\":";
  out += std::to_string(std::max(0, state.essence));

  out += ",\"base_hp\":";
  out += std::to_string(std::max(0, state.base_hp));

  out += ",\"next_level_base_hp_target\":";
  out += std::to_string(state.next_level_base_hp_target);

  out += ",\"shards\":";
  out += std::to_string(std::max(0, state.shards));

  out += ",\"player_level\":";
  out += std::to_string(std::max(1, state.player_level));

  out += ",\"player_xp\":";
  out += std::to_string(std::max(0, state.player_xp));

  out += ",\"lifetime_levels_completed\":";
  out += std::to_string(std::max(0, state.lifetime_levels_completed));
  out += ",\"lifetime_stars_earned\":";
  out += std::to_string(std::max(0, state.lifetime_stars_earned));
  out += ",\"lifetime_bosses_killed\":";
  out += std::to_string(std::max(0, state.lifetime_bosses_killed));
  out += ",\"lifetime_merges\":";
  out += std::to_string(std::max(0, state.lifetime_merges));

  out += ",\"unlocked_characters\":[";
  bool first_char = true;
  for (std::size_t i = 0; i < state.unlocked_characters.size(); ++i) {
    if (state.unlocked_characters[i] == 0) {
      continue;
    }
    const CharacterId cid = static_cast<CharacterId>(static_cast<std::uint8_t>(i));
    if (cid == CharacterId::Count) {
      continue;
    }
    if (!first_char) {
      out.push_back(',');
    }
    first_char = false;
    append_escaped(out, to_string(cid));
  }
  out.push_back(']');

  out += ",\"mastery_ranks\":[";
  for (std::size_t i = 0; i < state.mastery_ranks.size(); ++i) {
    if (i != 0) {
      out.push_back(',');
    }
    out += std::to_string(static_cast<int>(state.mastery_ranks[i]));
  }
  out.push_back(']');

  out += ",\"unlocked_relics\":[";
  bool first_relic = true;
  for (std::size_t i = 0; i < state.relic_unlocked.size(); ++i) {
    if (state.relic_unlocked[i] == 0) {
      continue;
    }
    const RelicId rid = static_cast<RelicId>(static_cast<std::uint8_t>(i));
    if (rid == RelicId::None || rid == RelicId::Count) {
      continue;
    }
    if (!first_relic) {
      out.push_back(',');
    }
    first_relic = false;
    append_escaped(out, to_string(rid));
  }
  out.push_back(']');

  out += ",\"equipped_relics\":[";
  for (std::size_t i = 0; i < state.equipped_relics.size(); ++i) {
    if (i != 0) {
      out.push_back(',');
    }
    const RelicId rid = state.equipped_relics[i];
    if (rid == RelicId::None || rid == RelicId::Count) {
      out += "null";
    } else {
      append_escaped(out, to_string(rid));
    }
  }
  out.push_back(']');

  out += ",\"roster\":[";
  for (std::size_t i = 0; i < state.roster.size(); ++i) {
    if (i != 0) {
      out.push_back(',');
    }
    const RosterEntry &re = state.roster[i];
    out.push_back('{');
    out += "\"character\":";
    append_escaped(out, to_string(re.character));
    out += ",\"tier\":";
    out += std::to_string(std::max(1, re.tier));
    out += ",\"kills\":";
    out += std::to_string(std::max(0, re.kills));
    out += ",\"seed_cost_essence\":";
    out += std::to_string(std::max(0, re.seed_cost_essence));
    out += ",\"upgrades\":[";
    for (std::size_t u = 0;
         u < static_cast<std::size_t>(UpgradeNode::Count); ++u) {
      if (u != 0) {
        out.push_back(',');
      }
      out += std::to_string(static_cast<int>(re.upgrades[u]));
    }
    out.push_back(']');
    out.push_back('}');
  }
  out.push_back(']');

  out.push_back('}');
  return out;
}

bool SaveState::fromJson(std::string_view json, GameState &io_state) {
  JsonCursor c{};
  c.s = json;
  c.i = 0;

  c.skip_ws();
  if (!c.consume('{')) {
    return false;
  }

  std::string key;
  std::string str;

  while (!c.eof()) {
    c.skip_ws();
    if (c.consume('}')) {
      break;
    }

    if (!c.parse_string(key)) {
      return false;
    }
    if (!c.consume(':')) {
      return false;
    }

    if (key == "max_level_reached") {
      std::int32_t v = 1;
      if (!c.parse_int(v)) {
        return false;
      }
      io_state.max_level_reached = std::max(1, v);
    } else if (key == "essence") {
      std::int32_t v = 0;
      if (!c.parse_int(v)) {
        return false;
      }
      io_state.essence = std::max(0, v);
    } else if (key == "base_hp") {
      std::int32_t v = 0;
      if (!c.parse_int(v)) {
        return false;
      }
      io_state.base_hp = std::max(0, v);
    } else if (key == "next_level_base_hp_target") {
      std::int32_t v = 0;
      if (!c.parse_int(v)) {
        return false;
      }
      io_state.next_level_base_hp_target = v;
    } else if (key == "shards") {
      std::int32_t v = 0;
      if (!c.parse_int(v)) {
        return false;
      }
      io_state.shards = std::max(0, v);
    } else if (key == "player_level") {
      std::int32_t v = 1;
      if (!c.parse_int(v)) {
        return false;
      }
      io_state.player_level = std::max(1, v);
    } else if (key == "player_xp") {
      std::int32_t v = 0;
      if (!c.parse_int(v)) {
        return false;
      }
      io_state.player_xp = std::max(0, v);
    } else if (key == "lifetime_levels_completed") {
      std::int32_t v = 0;
      if (!c.parse_int(v)) {
        return false;
      }
      io_state.lifetime_levels_completed = std::max(0, v);
    } else if (key == "lifetime_stars_earned") {
      std::int32_t v = 0;
      if (!c.parse_int(v)) {
        return false;
      }
      io_state.lifetime_stars_earned = std::max(0, v);
    } else if (key == "lifetime_bosses_killed") {
      std::int32_t v = 0;
      if (!c.parse_int(v)) {
        return false;
      }
      io_state.lifetime_bosses_killed = std::max(0, v);
    } else if (key == "lifetime_merges") {
      std::int32_t v = 0;
      if (!c.parse_int(v)) {
        return false;
      }
      io_state.lifetime_merges = std::max(0, v);
    } else if (key == "unlocked_characters") {
      if (!c.consume('[')) {
        return false;
      }
      io_state.unlocked_characters.fill(0);
      c.skip_ws();
      if (!c.consume(']')) {
        while (!c.eof()) {
          if (!c.parse_string(str)) {
            return false;
          }
          CharacterId cid{};
          if (from_string(str, cid)) {
            const std::size_t idx = static_cast<std::size_t>(cid);
            if (cid != CharacterId::Count &&
                idx < io_state.unlocked_characters.size()) {
              io_state.unlocked_characters[idx] = 1;
            }
          }
          c.skip_ws();
          if (c.consume(',')) {
            continue;
          }
          if (c.consume(']')) {
            break;
          }
          return false;
        }
      }
    } else if (key == "mastery_ranks") {
      if (!c.consume('[')) {
        return false;
      }
      io_state.mastery_ranks.fill(0);
      int slot = 0;
      c.skip_ws();
      if (!c.consume(']')) {
        while (!c.eof()) {
          if (slot >= static_cast<int>(io_state.mastery_ranks.size())) {
            if (!c.skip_value()) {
              return false;
            }
          } else {
            std::int32_t v = 0;
            if (!c.parse_int(v)) {
              return false;
            }
            io_state.mastery_ranks[static_cast<std::size_t>(slot)] =
                static_cast<std::uint8_t>(std::clamp<std::int32_t>(v, 0, 255));
          }
          slot++;
          c.skip_ws();
          if (c.consume(',')) {
            continue;
          }
          if (c.consume(']')) {
            break;
          }
          return false;
        }
      }
    } else if (key == "unlocked_relics") {
      if (!c.consume('[')) {
        return false;
      }
      io_state.relic_unlocked.fill(0);
      c.skip_ws();
      if (!c.consume(']')) {
        while (!c.eof()) {
          if (!c.parse_string(str)) {
            return false;
          }
          RelicId rid = RelicId::None;
          if (from_string(str, rid)) {
            const std::size_t idx = static_cast<std::size_t>(rid);
            if (rid != RelicId::None && rid != RelicId::Count &&
                idx < io_state.relic_unlocked.size()) {
              io_state.relic_unlocked[idx] = 1;
            }
          }
          c.skip_ws();
          if (c.consume(',')) {
            continue;
          }
          if (c.consume(']')) {
            break;
          }
          return false;
        }
      }
    } else if (key == "equipped_relics") {
      if (!c.consume('[')) {
        return false;
      }
      io_state.equipped_relics.fill(RelicId::None);
      int slot = 0;
      c.skip_ws();
      if (!c.consume(']')) {
        while (!c.eof()) {
          if (slot >= static_cast<int>(io_state.equipped_relics.size())) {
            if (!c.skip_value()) {
              return false;
            }
          } else {
            c.skip_ws();
            if (c.peek() == '"') {
              if (!c.parse_string(str)) {
                return false;
              }
              RelicId rid = RelicId::None;
              if (from_string(str, rid)) {
                io_state.equipped_relics[static_cast<std::size_t>(slot)] = rid;
              }
            } else if (c.s.substr(c.i, 4) == "null") {
              c.i += 4;
              io_state.equipped_relics[static_cast<std::size_t>(slot)] = RelicId::None;
            } else {
              if (!c.skip_value()) {
                return false;
              }
            }
          }
          slot++;
          c.skip_ws();
          if (c.consume(',')) {
            continue;
          }
          if (c.consume(']')) {
            break;
          }
          return false;
        }
      }
    } else if (key == "stars_per_level") {
      if (!c.consume('[')) {
        return false;
      }
      io_state.stars_per_level.clear();
      c.skip_ws();
      if (!c.consume(']')) {
        while (!c.eof()) {
          std::int32_t v = 0;
          if (!c.parse_int(v)) {
            return false;
          }
          io_state.stars_per_level.push_back(static_cast<std::uint8_t>(
              std::clamp<std::int32_t>(v, 0, 3)));
          c.skip_ws();
          if (c.consume(',')) {
            continue;
          }
          if (c.consume(']')) {
            break;
          }
          return false;
        }
      }
    } else if (key == "roster") {
      if (!c.consume('[')) {
        return false;
      }
      io_state.roster.clear();
      c.skip_ws();
      if (!c.consume(']')) {
        while (!c.eof()) {
          if (!c.consume('{')) {
            return false;
          }

          RosterEntry re{};
          re.character = CharacterId::Brix;
          re.tier = 1;
          re.kills = 0;
          re.seed_cost_essence = 0;

          while (!c.eof()) {
            c.skip_ws();
            if (c.consume('}')) {
              break;
            }
            if (!c.parse_string(key)) {
              return false;
            }
            if (!c.consume(':')) {
              return false;
            }

            if (key == "character") {
              if (!c.parse_string(str)) {
                return false;
              }
              CharacterId cid{};
              if (from_string(str, cid)) {
                re.character = cid;
              }
            } else if (key == "tier") {
              std::int32_t v = 1;
              if (!c.parse_int(v)) {
                return false;
              }
              re.tier = std::max(1, v);
            } else if (key == "kills") {
              std::int32_t v = 0;
              if (!c.parse_int(v)) {
                return false;
              }
              re.kills = std::max(0, v);
            } else if (key == "seed_cost_essence") {
              std::int32_t v = 0;
              if (!c.parse_int(v)) {
                return false;
              }
              re.seed_cost_essence = std::max(0, v);
            } else if (key == "upgrades") {
              if (!c.consume('[')) {
                return false;
              }
              for (std::size_t u = 0;
                   u < static_cast<std::size_t>(UpgradeNode::Count); ++u) {
                c.skip_ws();
                if (c.peek() == ']') {
                  break;
                }
                std::int32_t v = 0;
                if (!c.parse_int(v)) {
                  return false;
                }
                re.upgrades[u] = static_cast<std::uint8_t>(
                    std::clamp<std::int32_t>(v, 0, 255));
                c.skip_ws();
                if (c.consume(',')) {
                  continue;
                }
                if (c.peek() == ']') {
                  break;
                }
              }
              if (!c.consume(']')) {
                return false;
              }
            } else {
              if (!c.skip_value()) {
                return false;
              }
            }

            c.skip_ws();
            if (c.consume(',')) {
              continue;
            }
            if (c.peek() == '}') {
              continue;
            }
            return false;
          }

          io_state.roster.push_back(re);

          c.skip_ws();
          if (c.consume(',')) {
            continue;
          }
          if (c.consume(']')) {
            break;
          }
          return false;
        }
      }
    } else {
      if (!c.skip_value()) {
        return false;
      }
    }

    c.skip_ws();
    if (c.consume(',')) {
      continue;
    }
    if (c.peek() == '}') {
      continue;
    }
    return false;
  }

  if (io_state.roster.empty()) {
    io_state.resetToNewProfile();
  }
  io_state.sanitizeCharacterUnlocks();
  io_state.sanitizeMasteries();
  io_state.sanitizeMetaProgression();
  RelicSystem::sanitizePersistent(io_state);

  return true;
}

bool SaveState::load(GameState &io_state) {
  const std::string data = platform_load();
  if (data.empty()) {
    return false;
  }

  GameState loaded{};
  loaded.resetToNewProfile();
  if (!fromJson(data, loaded)) {
    return false;
  }

  restorePersistent(io_state, snapshotPersistent(loaded));
  return true;
}

bool SaveState::save(const GameState &state) {
  const std::string json = toJson(state);
  return platform_save(json);
}

} // namespace tower_swarm
