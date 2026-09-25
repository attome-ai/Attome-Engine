#include "FaceAllocator.h"

#include <cassert>
#include <iterator>

namespace atm::render {

void FaceAllocator::reset(uint32_t capacityUnits) {
  capacity_ = capacityUnits;
  used_ = 0;
  byOffset_.clear();
  bySize_.clear();
  if (capacityUnits > 0) insertFree(0, capacityUnits);
}

void FaceAllocator::insertFree(uint32_t offset, uint32_t units) {
  byOffset_.emplace(offset, units);
  bySize_.emplace(units, offset);
}

void FaceAllocator::eraseBySize(uint32_t offset, uint32_t units) {
  auto [lo, hi] = bySize_.equal_range(units);
  for (auto it = lo; it != hi; ++it) {
    if (it->second == offset) {
      bySize_.erase(it);
      return;
    }
  }
  assert(false && "FaceAllocator: free range index out of sync");
}

uint32_t FaceAllocator::allocate(uint32_t units) {
  if (units == 0) return kInvalid;
  auto it = bySize_.lower_bound(units); // smallest range that fits
  if (it == bySize_.end()) return kInvalid;
  const uint32_t size = it->first, offset = it->second;
  bySize_.erase(it);
  byOffset_.erase(offset);
  if (size > units) insertFree(offset + units, size - units);
  used_ += units;
  return offset;
}

void FaceAllocator::free(uint32_t offset, uint32_t units) {
  if (units == 0 || offset == kInvalid) return;
  assert(offset + units <= capacity_);
  used_ -= units;
  uint32_t start = offset, size = units;
  // Merge with the following range.
  auto next = byOffset_.lower_bound(offset);
  if (next != byOffset_.end() && next->first == offset + units) {
    size += next->second;
    eraseBySize(next->first, next->second);
    next = byOffset_.erase(next);
  }
  // Merge with the preceding range.
  if (next != byOffset_.begin()) {
    auto prev = std::prev(next);
    if (prev->first + prev->second == offset) {
      start = prev->first;
      size += prev->second;
      eraseBySize(prev->first, prev->second);
      byOffset_.erase(prev);
    }
  }
  insertFree(start, size);
}

uint32_t FaceAllocator::largestFree() const {
  return bySize_.empty() ? 0u : bySize_.rbegin()->first;
}

float FaceAllocator::fragmentation() const {
  const uint32_t freeUnits = capacity_ - used_;
  if (freeUnits == 0) return 0.0f;
  return 1.0f - float(largestFree()) / float(freeUnits);
}

} // namespace atm::render
