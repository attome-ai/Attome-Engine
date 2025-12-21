# Engine Optimization Log

## 🏆 Successful Optimizations

| Optimization | FPS Impact | Why it Worked |
|--------------|------------|---------------|
| **Incremental Grid Updates** | **240 → 658 FPS** (+174%) | Eliminated the massive per-frame cost of `rebuild_grid()` by only updating entities that actually moved cells. |
| **O(1) Slot-Based Removal** | **658 → 671 FPS** (+2%) | Replaced linear O(N) searching for entity removal with constant time O(1) by tracking `cell_slot[i]`. |
| **Parallel Scalar Updates** | **Baseline** | Full utilization of 32 hardware threads via `std::execution::par` proved superior to single-threaded SIMD. |

---

## ❌ Failed Optimizations

| Optimization | FPS Impact | Why it Failed |
|--------------|------------|---------------|
| **Batch Grid Updates** | **671 → 530 FPS** (-21%) | Collecting cell changes in thread-local buffers and merging them caused mutex contention that outweighed the cost of simple atomic operations. |
| **SIMD (AVX2)** | **671 → 188 FPS** (-72%) | **Reason 1**: Ran on a single thread, losing 32x parallelism.<br>**Reason 2**: Separating movement (SIMD) from grid logic (Scalar) doubled the iteration overhead and cache pressure. |
| **Parallel Render Batching** | **671 → 605 FPS** (-10%) | The cost of merging thread-local instruction buffers (`memcpy` + offset math) was higher than the extremely fast serial pointer increments. |
| **Memory Prefetching** | **671 → 586 FPS** (-13%) | **Reason**: `_mm_prefetch` instructions added instruction overhead without hiding enough latency. The CPU's hardware prefetcher was likely already doing a good job on the sequential streams, and the "gather" prefetch was too late or interfered with the pipeline. |

---

## 🧠 Key Takeaways

1.  **Parallelism > SIMD**: On a high-core-count system (32 threads), saturation via `std::execution::par` beats single-threaded vectorization.
2.  **Atomics are Cheap**: For dispersed updates, atomic operations are often faster than the locking or merging overhead of batching.
3.  **Contiguous Memory is King**: The serial rendering loop is effectively O(N) and memory-bandwidth bound; complex parallel chunking schemes just add CPU overhead.
