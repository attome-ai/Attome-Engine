#include "systems/PathGrid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace tower_swarm {
namespace {

struct HeapEntry final {
  float f{0.0f};
  int idx{0};
};

struct HeapGreater final {
  bool operator()(const HeapEntry &a, const HeapEntry &b) const {
    return a.f > b.f;
  }
};

int index_for(int col, int row, int cols) { return row * cols + col; }

PathGrid::Cell cell_for_index(int idx, int cols) {
  PathGrid::Cell c{};
  c.col = static_cast<std::int16_t>(idx % cols);
  c.row = static_cast<std::int16_t>(idx / cols);
  return c;
}

float heuristic_manhattan(int a_idx, int b_idx, int cols) {
  const int ax = a_idx % cols;
  const int ay = a_idx / cols;
  const int bx = b_idx % cols;
  const int by = b_idx / cols;
  return static_cast<float>(std::abs(ax - bx) + std::abs(ay - by));
}

} // namespace

void PathGrid::reset(int cols, int rows) {
  cols_ = std::max(0, cols);
  rows_ = std::max(0, rows);
  const int n = cellCount();
  walkable_.assign(static_cast<std::size_t>(n), 1u);

  g_score_.assign(static_cast<std::size_t>(n), std::numeric_limits<float>::infinity());
  f_score_.assign(static_cast<std::size_t>(n), std::numeric_limits<float>::infinity());
  came_from_.assign(static_cast<std::size_t>(n), -1);
  state_.assign(static_cast<std::size_t>(n), 0u);
}

bool PathGrid::inBounds(int col, int row) const {
  return col >= 0 && row >= 0 && col < cols_ && row < rows_;
}

bool PathGrid::isWalkable(int col, int row) const {
  if (!inBounds(col, row)) {
    return false;
  }
  const std::size_t idx =
      static_cast<std::size_t>(index_for(col, row, cols_));
  return idx < walkable_.size() && walkable_[idx] != 0u;
}

void PathGrid::setWalkable(int col, int row, bool walkable) {
  if (!inBounds(col, row)) {
    return;
  }
  const std::size_t idx =
      static_cast<std::size_t>(index_for(col, row, cols_));
  if (idx >= walkable_.size()) {
    return;
  }
  walkable_[idx] = walkable ? 1u : 0u;
}

void PathGrid::setAllWalkable(bool walkable) {
  std::fill(walkable_.begin(), walkable_.end(), walkable ? 1u : 0u);
}

bool PathGrid::findPath(Cell start, Cell goal, std::vector<Cell> &out_path,
                        ExtraBlockedFn extra_blocked, void *user) const {
  out_path.clear();
  if (cols_ <= 0 || rows_ <= 0) {
    return false;
  }
  if (!inBounds(start.col, start.row) || !inBounds(goal.col, goal.row)) {
    return false;
  }
  if (!isWalkable(start.col, start.row) || !isWalkable(goal.col, goal.row)) {
    return false;
  }

  const int start_idx = index_for(start.col, start.row, cols_);
  const int goal_idx = index_for(goal.col, goal.row, cols_);
  const int n = cellCount();
  if (start_idx < 0 || start_idx >= n || goal_idx < 0 || goal_idx >= n) {
    return false;
  }

  if (start_idx == goal_idx) {
    out_path.push_back(start);
    return true;
  }

  const std::size_t sz = static_cast<std::size_t>(n);
  if (g_score_.size() != sz) {
    g_score_.assign(sz, std::numeric_limits<float>::infinity());
  } else {
    std::fill(g_score_.begin(), g_score_.end(),
              std::numeric_limits<float>::infinity());
  }
  if (f_score_.size() != sz) {
    f_score_.assign(sz, std::numeric_limits<float>::infinity());
  } else {
    std::fill(f_score_.begin(), f_score_.end(),
              std::numeric_limits<float>::infinity());
  }
  if (came_from_.size() != sz) {
    came_from_.assign(sz, -1);
  } else {
    std::fill(came_from_.begin(), came_from_.end(), -1);
  }
  if (state_.size() != sz) {
    state_.assign(sz, 0u);
  } else {
    std::fill(state_.begin(), state_.end(), 0u);
  }

  std::vector<HeapEntry> open{};
  open.reserve(128);

  g_score_[static_cast<std::size_t>(start_idx)] = 0.0f;
  f_score_[static_cast<std::size_t>(start_idx)] =
      heuristic_manhattan(start_idx, goal_idx, cols_);
  open.push_back(HeapEntry{f_score_[static_cast<std::size_t>(start_idx)],
                           start_idx});
  std::push_heap(open.begin(), open.end(), HeapGreater{});
  state_[static_cast<std::size_t>(start_idx)] = 1u;

  constexpr std::array<std::pair<int, int>, 4> kDirs = {
      std::pair<int, int>{1, 0}, std::pair<int, int>{-1, 0},
      std::pair<int, int>{0, 1}, std::pair<int, int>{0, -1}};

  while (!open.empty()) {
    std::pop_heap(open.begin(), open.end(), HeapGreater{});
    const HeapEntry cur = open.back();
    open.pop_back();

    if (cur.idx < 0 || cur.idx >= n) {
      continue;
    }

    const float best_f = f_score_[static_cast<std::size_t>(cur.idx)];
    if (!std::isfinite(best_f) || cur.f > best_f) {
      continue; // stale entry
    }

    if (cur.idx == goal_idx) {
      // Reconstruct.
      std::vector<int> rev{};
      rev.reserve(64);
      int walk = goal_idx;
      while (walk != -1) {
        rev.push_back(walk);
        if (walk == start_idx) {
          break;
        }
        walk = came_from_[static_cast<std::size_t>(walk)];
      }
      if (rev.empty() || rev.back() != start_idx) {
        return false;
      }
      out_path.reserve(rev.size());
      for (auto it = rev.rbegin(); it != rev.rend(); ++it) {
        out_path.push_back(cell_for_index(*it, cols_));
      }
      return !out_path.empty();
    }

    state_[static_cast<std::size_t>(cur.idx)] = 2u;

    const int cx = cur.idx % cols_;
    const int cy = cur.idx / cols_;
    const float g_here = g_score_[static_cast<std::size_t>(cur.idx)];
    if (!std::isfinite(g_here)) {
      continue;
    }

    for (const auto [dx, dy] : kDirs) {
      const int nx = cx + dx;
      const int ny = cy + dy;
      if (!inBounds(nx, ny)) {
        continue;
      }
      if (!isWalkable(nx, ny)) {
        continue;
      }

      if (extra_blocked) {
        const bool blocked = extra_blocked(nx, ny, user);
        const int nidx = index_for(nx, ny, cols_);
        if (blocked && nidx != goal_idx) {
          continue;
        }
      }

      const int nidx = index_for(nx, ny, cols_);
      if (nidx < 0 || nidx >= n) {
        continue;
      }
      if (state_[static_cast<std::size_t>(nidx)] == 2u) {
        continue;
      }

      const float tentative_g = g_here + 1.0f;
      if (tentative_g >= g_score_[static_cast<std::size_t>(nidx)]) {
        continue;
      }

      came_from_[static_cast<std::size_t>(nidx)] = cur.idx;
      g_score_[static_cast<std::size_t>(nidx)] = tentative_g;
      const float f = tentative_g +
                      heuristic_manhattan(nidx, goal_idx, cols_);
      f_score_[static_cast<std::size_t>(nidx)] = f;

      open.push_back(HeapEntry{f, nidx});
      std::push_heap(open.begin(), open.end(), HeapGreater{});
      state_[static_cast<std::size_t>(nidx)] = 1u;
    }
  }

  return false;
}

} // namespace tower_swarm

