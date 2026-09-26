#include "DefsInternal.h"

#include <deque>

namespace ao {

namespace defs_detail {

const std::string &intern(std::string s) {
  static std::deque<std::string> storage; // deque: element addresses never move
  storage.push_back(std::move(s));
  return storage.back();
}

std::vector<ItemDef> &itemTable() {
  static std::vector<ItemDef> t(1); // id 0 = the empty item
  return t;
}

std::unordered_map<BlockId, ItemId> &blockDropTable() {
  static std::unordered_map<BlockId, ItemId> t;
  return t;
}

} // namespace defs_detail

const ItemDef &itemDef(ItemId id) {
  const auto &t = defs_detail::itemTable();
  return t[id < t.size() ? id : 0];
}

ItemId findItem(std::string_view name) {
  const auto &t = defs_detail::itemTable();
  for (size_t i = 1; i < t.size(); ++i)
    if (t[i].name == name)
      return ItemId(i);
  return 0;
}

uint16_t itemCount() { return uint16_t(defs_detail::itemTable().size()); }

ItemId blockDropItem(BlockId block) {
  const auto &t = defs_detail::blockDropTable();
  const auto it = t.find(block);
  return it == t.end() ? ItemId(0) : it->second;
}

} // namespace ao
