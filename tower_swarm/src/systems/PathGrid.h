#pragma once

#include <cstdint>
#include <vector>

namespace tower_swarm {

class PathGrid final {
public:
  struct Cell final {
    std::int16_t col{0};
    std::int16_t row{0};
  };

  using ExtraBlockedFn = bool (*)(int col, int row, void *user);

  void reset(int cols, int rows);

  int cols() const { return cols_; }
  int rows() const { return rows_; }
  int cellCount() const { return cols_ > 0 && rows_ > 0 ? cols_ * rows_ : 0; }

  bool inBounds(int col, int row) const;
  bool isWalkable(int col, int row) const;
  void setWalkable(int col, int row, bool walkable);
  void setAllWalkable(bool walkable);

  // Returns a path containing {start .. goal} inclusive.
  bool findPath(Cell start, Cell goal, std::vector<Cell> &out_path,
                ExtraBlockedFn extra_blocked = nullptr,
                void *user = nullptr) const;

private:
  int cols_{0};
  int rows_{0};
  std::vector<std::uint8_t> walkable_{};

  // Scratch buffers (avoids per-call allocations).
  mutable std::vector<float> g_score_{};
  mutable std::vector<float> f_score_{};
  mutable std::vector<int> came_from_{};
  mutable std::vector<std::uint8_t> state_{};
};

} // namespace tower_swarm

