#include "DefsInternal.h"

namespace ao {

namespace defs_detail {

std::vector<MonsterDef> &npcTable() {
  static std::vector<MonsterDef> t;
  return t;
}

} // namespace defs_detail

const MonsterDef &monsterDef(uint8_t type) {
  static const MonsterDef kNone{};
  const auto &t = defs_detail::npcTable();
  return type < t.size() ? t[type] : kNone;
}

uint8_t monsterTypeCount() { return uint8_t(defs_detail::npcTable().size()); }

int findNpc(std::string_view name) {
  const auto &t = defs_detail::npcTable();
  for (size_t i = 0; i < t.size(); ++i)
    if (t[i].name == name)
      return int(i);
  return -1;
}

} // namespace ao
