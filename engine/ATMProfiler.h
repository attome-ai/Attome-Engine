#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#undef max
#undef min

// SIMD detection macros
#if defined(__EMSCRIPTEN__)
// Emscripten SIMD detection
#if defined(__wasm_simd128__)
#define HAS_WASM_SIMD 1
#include <wasm_simd128.h>
#else
#define HAS_WASM_SIMD 0
#endif
#define HAS_SSE 0
#elif defined(__SSE__) && !defined(__EMSCRIPTEN__)
// x86/x64 SSE detection
#define HAS_SSE 1
#include <xmmintrin.h> // SSE
#if defined(__SSE2__)
#include <emmintrin.h> // SSE2
#endif
#else
#define HAS_SSE 0
#endif

// Structure to hold profiling data for a single function
struct ProfileData {
  uint64_t callCount = 0;
  double totalTimeMs = 0.0;
  double minTimeMs = std::numeric_limits<double>::max();
  double maxTimeMs = 0.0;
  std::chrono::steady_clock::time_point lastReportTime =
      std::chrono::steady_clock::now();
};

// Global profiling data
class Profiler {
private:
  std::unordered_map<std::string, ProfileData> stats;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
      startTimes;
  std::mutex mutex;
  std::chrono::steady_clock::time_point lastReportTime =
      std::chrono::steady_clock::now();
  bool reportEnabled = true;
  double reportIntervalSeconds = 1.0;

public:
  static Profiler &getInstance() {
    static Profiler instance;
    return instance;
  }

  void begin(const std::string &functionName) {
    std::lock_guard<std::mutex> lock(mutex);
    startTimes[functionName] = std::chrono::steady_clock::now();
  }

  void end(const std::string &functionName) {
    std::lock_guard<std::mutex> lock(mutex);

    auto endTime = std::chrono::steady_clock::now();
    auto it = startTimes.find(functionName);

    if (it != startTimes.end()) {
      auto startTime = it->second;
      double elapsedMs =
          std::chrono::duration<double, std::milli>(endTime - startTime)
              .count();

      auto &data = stats[functionName];
      data.callCount++;
      data.totalTimeMs += elapsedMs;
      data.minTimeMs = std::min(data.minTimeMs, elapsedMs);
      data.maxTimeMs = std::max(data.maxTimeMs, elapsedMs);

      startTimes.erase(it);

      // Check if we should generate a report
      if (reportEnabled) {
        auto now = std::chrono::steady_clock::now();
        double secondsSinceLastReport =
            std::chrono::duration<double>(now - lastReportTime).count();

        if (secondsSinceLastReport >= reportIntervalSeconds) {
          generateReport();
          lastReportTime = now;
        }
      }
    }
  }

  void setReportInterval(double seconds) {
    std::lock_guard<std::mutex> lock(mutex);
    reportIntervalSeconds = seconds;
  }

  void enableReporting(bool enable) {
    std::lock_guard<std::mutex> lock(mutex);
    reportEnabled = enable;
  }

  void resetStats() {
    std::lock_guard<std::mutex> lock(mutex);
    stats.clear();
    startTimes.clear();
  }

  void generateReport() {
    std::cout << "\n=== Profiling Report ===\n";
    std::cout << std::left << std::setw(30) << "Function" << std::right
              << std::setw(10) << "Calls" << std::setw(15) << "Total (ms)"
              << std::setw(15) << "Avg (ms)" << std::setw(15) << "Min (ms)"
              << std::setw(15) << "Max (ms)" << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    for (const auto &entry : stats) {
      const auto &name = entry.first;
      const auto &data = entry.second;
      double avgTime =
          data.callCount > 0 ? data.totalTimeMs / data.callCount : 0;

      std::cout << std::left << std::setw(30) << name << std::right
                << std::setw(10) << data.callCount << std::setw(15)
                << std::fixed << std::setprecision(3) << data.totalTimeMs
                << std::setw(15) << std::fixed << std::setprecision(3)
                << avgTime << std::setw(15) << std::fixed
                << std::setprecision(3)
                << (data.callCount > 0 ? data.minTimeMs : 0) << std::setw(15)
                << std::fixed << std::setprecision(3) << data.maxTimeMs
                << std::endl;
    }

    std::cout << "========================\n";
  }
};

// Easy-to-use profiling macros
#define PROFILE_BEGIN(name) Profiler::getInstance().begin(name)
#define PROFILE_END(name) Profiler::getInstance().end(name)

// Automatic profiling using RAII pattern
class ScopedProfiler {
private:
  std::string functionName;

public:
  ScopedProfiler(const std::string &name) : functionName(name) {
    PROFILE_BEGIN(functionName);
  }

  ~ScopedProfiler() { PROFILE_END(functionName); }
};

#if 0

void print_active_entities_grid(Engine* engine) {

    // Static variable to track when we last printed
    static Uint64 last_print_time = 0;

    // Get current time
    Uint64 current_time = SDL_GetTicks();

    // Only print once per second
    if (current_time - last_print_time < 1000) {
        return; // Skip until at least 1 second has passed
    }

    // Update the last print time
    last_print_time = current_time;

    SpatialGrid* grid = &engine->grid;

    // Create a temporary 2D array to store active entity counts
    int** active_counts = new int* [grid->height];
    for (int y = 0; y < grid->height; y++) {
        active_counts[y] = new int[grid->width]();  // Initialize to zero
    }

    // Count all active entities by iterating through all chunks
    for (int chunk_idx = 0; chunk_idx < engine->entities.chunk_count; chunk_idx++) {
        EntityChunk* chunk = engine->entities.chunks[chunk_idx];

        // Skip empty chunks
        if (!chunk || chunk->count == 0) continue;

        for (int local_idx = 0; local_idx < chunk->count; local_idx++) {

            // Calculate which grid cell this entity belongs to
            float entity_x = chunk->x[local_idx];
            float entity_y = chunk->y[local_idx];

            // Skip if outside grid bounds
            if (entity_x < 0 || entity_y < 0 ||
                entity_x >= engine->world_bounds.w ||
                entity_y >= engine->world_bounds.h) {
                continue;
            }

            int grid_x = (int)(entity_x / grid->cell_size);
            int grid_y = (int)(entity_y / grid->cell_size);

            // Clamp to grid bounds
            grid_x = std::max(0, std::min(grid_x, grid->width - 1));
            grid_y = std::max(0, std::min(grid_y, grid->height - 1));

            // Increment active count for this cell
            active_counts[grid_y][grid_x]++;
        }
    }

    // Calculate statistics
    int total_active = 0;
    int active_cells = 0;
    int max_active = 0;
    int max_x = 0, max_y = 0;

    for (int y = 0; y < grid->height; y++) {
        for (int x = 0; x < grid->width; x++) {
            int active = active_counts[y][x];

            if (active > 0) {
                active_cells++;
                total_active += active;
            }

            if (active > max_active) {
                max_active = active;
                max_x = x;
                max_y = y;
            }
        }
    }

    // Print summary
    printf("\n--- Active Entities Grid (entities with active=true) ---\n");
    printf("Total active entities: %d in %d cells (%.1f%% of grid)\n",
        total_active, active_cells,
        (active_cells * 100.0f) / grid->total_cells);
    printf("Most populated cell: [%d,%d] with %d active entities\n",
        max_x, max_y, max_active);

    // Print grid visualization
    // Define grid display area - limit to reasonable output size
    int display_width = std::min(grid->width, 20);
    int display_height = std::min(grid->height, 20);

    printf("\nActive Entities Grid (showing up to %dx%d cells):\n",
        display_width, display_height);

    // Print column headers
    printf("     ");
    for (int x = 0; x < display_width; x++) {
        printf("%2d ", x);
    }
    printf("\n");

    // Print separator
    printf("   +");
    for (int x = 0; x < display_width; x++) {
        printf("---");
    }
    printf("\n");

    // Print rows
    for (int y = 0; y < display_height; y++) {
        printf("%2d |", y);
        for (int x = 0; x < display_width; x++) {
            int count = active_counts[y][x];
            // Use different symbols based on entity count
            if (count == 0) {
                printf(" . ");
            }
            else if (count < 10) {
                printf(" %d ", count);
            }
            else {
                printf("%2d ", count);
            }
        }
        printf("\n");
    }

    // If grid is larger than display area, indicate there's more
    if (grid->width > display_width || grid->height > display_height) {
        printf("(Grid continues beyond display area)\n");
    }

    printf("--- End of Active Entities Grid Report ---\n\n");

    // Clean up
    for (int y = 0; y < grid->height; y++) {
        delete[] active_counts[y];
    }
    delete[] active_counts;
}

void print_visible_entities_grid(Engine* engine) {

    // Static variable to track when we last printed
    static Uint64 last_print_time = 0;

    // Get current time
    Uint64 current_time = SDL_GetTicks();

    // Only print once per second
    if (current_time - last_print_time < 1000) {
        return; // Skip until at least 1 second has passed
    }

    // Update the last print time
    last_print_time = current_time;

    SpatialGrid* grid = &engine->grid;

    // Create a temporary 2D array to store visible entity counts
    int** visible_counts = new int* [grid->height];
    for (int y = 0; y < grid->height; y++) {
        visible_counts[y] = new int[grid->width]();  // Initialize to zero
    }

    // Count all visible entities by iterating through all chunks
    for (int chunk_idx = 0; chunk_idx < engine->entities.chunk_count; chunk_idx++) {
        EntityChunk* chunk = engine->entities.chunks[chunk_idx];

        // Skip empty chunks
        if (!chunk || chunk->count == 0) continue;

        for (int local_idx = 0; local_idx < chunk->count; local_idx++) {
            // Only count visible entities
            if (!chunk->visible[local_idx]) continue;

            // Calculate which grid cell this entity belongs to
            float entity_x = chunk->x[local_idx];
            float entity_y = chunk->y[local_idx];

            // Skip if outside grid bounds
            if (entity_x < 0 || entity_y < 0 ||
                entity_x >= engine->world_bounds.w ||
                entity_y >= engine->world_bounds.h) {
                continue;
            }

            int grid_x = (int)(entity_x / grid->cell_size);
            int grid_y = (int)(entity_y / grid->cell_size);

            // Clamp to grid bounds
            grid_x = std::max(0, std::min(grid_x, grid->width - 1));
            grid_y = std::max(0, std::min(grid_y, grid->height - 1));

            // Increment visible count for this cell
            visible_counts[grid_y][grid_x]++;
        }
    }

    // Calculate statistics
    int total_visible = 0;
    int visible_cells = 0;
    int max_visible = 0;
    int max_x = 0, max_y = 0;

    for (int y = 0; y < grid->height; y++) {
        for (int x = 0; x < grid->width; x++) {
            int visible = visible_counts[y][x];

            if (visible > 0) {
                visible_cells++;
                total_visible += visible;
            }

            if (visible > max_visible) {
                max_visible = visible;
                max_x = x;
                max_y = y;
            }
        }
    }

    // Print summary
    printf("\n--- Visible Entities Grid (entities with visible=true) ---\n");
    printf("Total visible entities: %d in %d cells (%.1f%% of grid)\n",
        total_visible, visible_cells,
        (visible_cells * 100.0f) / grid->total_cells);
    printf("Most populated cell: [%d,%d] with %d visible entities\n",
        max_x, max_y, max_visible);

    // Print grid visualization
    // Define grid display area - limit to reasonable output size
    int display_width = std::min(grid->width, 20);
    int display_height = std::min(grid->height, 20);

    printf("\nVisible Entities Grid (showing up to %dx%d cells):\n",
        display_width, display_height);

    // Print column headers
    printf("     ");
    for (int x = 0; x < display_width; x++) {
        printf("%2d ", x);
    }
    printf("\n");

    // Print separator
    printf("   +");
    for (int x = 0; x < display_width; x++) {
        printf("---");
    }
    printf("\n");

    // Print rows
    for (int y = 0; y < display_height; y++) {
        printf("%2d |", y);
        for (int x = 0; x < display_width; x++) {
            int count = visible_counts[y][x];
            // Use different symbols based on entity count
            if (count == 0) {
                printf(" . ");
            }
            else if (count < 10) {
                printf(" %d ", count);
            }
            else {
                printf("%2d ", count);
            }
        }
        printf("\n");
    }

    // If grid is larger than display area, indicate there's more
    if (grid->width > display_width || grid->height > display_height) {
        printf("(Grid continues beyond display area)\n");
    }

    printf("--- End of Visible Entities Grid Report ---\n\n");

    // Clean up
    for (int y = 0; y < grid->height; y++) {
        delete[] visible_counts[y];
    }
    delete[] visible_counts;
}

#else
#define print_visible_entities_grid(x)
#define print_active_entities_grid(y)
#endif

#if 0
// Easy macro for automatic function profiling
#define PROFILE_FUNCTION() ScopedProfiler profiler(__FUNCTION__)
#define PROFILE_SCOPE(name) ScopedProfiler profiler(name)

#else

#define PROFILE_FUNCTION()
#define PROFILE_SCOPE(name)

#endif // 1
