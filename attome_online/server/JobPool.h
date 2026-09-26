#pragma once

// Fork-join worker pool for the zone server's data-parallel phases (snapshot
// building, per-shard network pumping). run() blocks until every item is
// done; the calling thread works too, as worker 0. Items are handed out in
// small chunks from an atomic counter, so uneven items balance themselves.
// Not reentrant: call run() from one thread at a time.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ao::server {

class JobPool {
public:
  // threads = total workers including the caller (1 = run inline).
  explicit JobPool(int threads = 1) { resize(threads); }
  ~JobPool() { shutdown(); }
  JobPool(const JobPool &) = delete;
  JobPool &operator=(const JobPool &) = delete;

  void resize(int threads) {
    shutdown();
    count_ = std::max(1, threads);
    quit_ = false;
    for (int i = 1; i < count_; ++i) workers_.emplace_back([this, i] { workerLoop(size_t(i)); });
  }
  int threads() const { return count_; }

  // Calls fn(item, worker) for item in [0, items); worker < threads().
  void run(size_t items, const std::function<void(size_t, size_t)> &fn, size_t chunk = 1) {
    if (items == 0) return;
    if (count_ == 1 || items == 1) {
      for (size_t i = 0; i < items; ++i) fn(i, 0);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fn_ = &fn;
      items_ = items;
      chunk_ = std::max<size_t>(1, chunk);
      next_.store(0, std::memory_order_relaxed);
      busy_ = count_ - 1;
      ++generation_;
    }
    wake_.notify_all();
    work(0);
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return busy_ == 0; });
    fn_ = nullptr;
  }

private:
  void work(size_t worker) {
    for (;;) {
      const size_t begin = next_.fetch_add(chunk_, std::memory_order_relaxed);
      if (begin >= items_) return;
      const size_t end = std::min(items_, begin + chunk_);
      for (size_t i = begin; i < end; ++i) (*fn_)(i, worker);
    }
  }

  void workerLoop(size_t worker) {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [&] { return quit_ || generation_ != seen; });
        if (quit_) return;
        seen = generation_;
      }
      work(worker);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--busy_ == 0) done_.notify_one();
      }
    }
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      quit_ = true;
    }
    wake_.notify_all();
    for (auto &t : workers_) t.join();
    workers_.clear();
  }

  int count_ = 1;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable wake_, done_;
  const std::function<void(size_t, size_t)> *fn_ = nullptr;
  size_t items_ = 0, chunk_ = 1;
  std::atomic<size_t> next_{0};
  int busy_ = 0;
  uint64_t generation_ = 0;
  bool quit_ = false;
};

} // namespace ao::server
