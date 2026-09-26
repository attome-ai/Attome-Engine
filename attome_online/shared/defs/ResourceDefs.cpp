#include "DefsInternal.h"

namespace ao {

namespace defs_detail {

std::vector<ResourceDef> &resourceTable() {
  static std::vector<ResourceDef> t;
  return t;
}

} // namespace defs_detail

const std::vector<ResourceDef> &resourceDefs() { return defs_detail::resourceTable(); }

const ResourceDef *resourceForBlock(BlockId block) {
  if (block == 0)
    return nullptr;
  for (const ResourceDef &r : defs_detail::resourceTable())
    if (r.block == block)
      return &r;
  return nullptr;
}

} // namespace ao
