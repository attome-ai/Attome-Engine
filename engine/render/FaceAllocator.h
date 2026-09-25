#pragma once

// CPU-side range allocator for the GPU face arenas. Units are faces
// (8 bytes = one PackedFace). Best-fit over free ranges indexed both by
// offset (for O(log n) coalescing on free) and by size (for O(log n) fits).
// No defragmentation: fragmentation() reports how bad it is.
// Pure CPU code, no Vulkan: unit-testable.

#include <cstdint>
#include <map>

namespace atm::render {

class FaceAllocator {
public:
  static constexpr uint32_t kInvalid = 0xFFFFFFFFu;

  void reset(uint32_t capacityUnits);
  // Returns the first unit of the range or kInvalid when no range fits.
  uint32_t allocate(uint32_t units);
  void free(uint32_t offset, uint32_t units);

  uint32_t capacity() const { return capacity_; }
  uint32_t used() const { return used_; }
  uint32_t largestFree() const;
  // 0 = all free space is one range, -> 1 = free space is scattered.
  float fragmentation() const;
  size_t freeRangeCount() const { return byOffset_.size(); }

private:
  void insertFree(uint32_t offset, uint32_t units);
  void eraseBySize(uint32_t offset, uint32_t units);

  uint32_t capacity_ = 0, used_ = 0;
  std::map<uint32_t, uint32_t> byOffset_;     // offset -> size
  std::multimap<uint32_t, uint32_t> bySize_;  // size -> offset
};

} // namespace atm::render
