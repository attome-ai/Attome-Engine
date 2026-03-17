#pragma once

#include <cstdint>

namespace attome::net {

// -----------------------------------------------------------------------------
// Self-test build/run configuration (compile-time)
//
// Toggle these by editing this header (no CMake cache knobs required).
// -----------------------------------------------------------------------------

#ifndef NET_SELF_TEST_AUTORUN
#define NET_SELF_TEST_AUTORUN 1
#endif

// Returns true on full pass.
bool run_self_tests();

} // namespace attome::net
