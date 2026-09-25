#pragma once

// Minimal test runner (no external dependency, works under Emscripten too).
//
//   ATM_TEST(name) { ATM_CHECK(1 + 1 == 2); ATM_CHECK_EQ(a, b); }
//
// Run the binary with a substring to filter: atm_tests grid

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace atm_test {

struct Case {
  const char *name;
  void (*fn)();
};

inline std::vector<Case> &registry() {
  static std::vector<Case> cases;
  return cases;
}

inline int &failures() {
  static int count = 0;
  return count;
}

struct Registrar {
  Registrar(const char *name, void (*fn)()) { registry().push_back({name, fn}); }
};

template <typename T> std::string show(const T &v) {
  std::ostringstream os;
  if constexpr (requires { os << v; })
    os << v;
  else
    os << "<value>";
  return os.str();
}

inline void report(const char *file, int line, const std::string &what) {
  std::printf("    %s:%d: %s\n", file, line, what.c_str());
  ++failures();
}

inline int run_all(int argc, char **argv) {
  const char *filter = argc > 1 ? argv[1] : nullptr;
  int ran = 0;
  int failed_cases = 0;
  for (const Case &c : registry()) {
    if (filter && !std::strstr(c.name, filter))
      continue;
    const int before = failures();
    std::printf("[ RUN  ] %s\n", c.name);
    c.fn();
    const bool ok = failures() == before;
    std::printf("[ %s ] %s\n", ok ? " OK " : "FAIL", c.name);
    ++ran;
    failed_cases += ok ? 0 : 1;
  }
  std::printf("\n%d test(s), %d failed\n", ran, failed_cases);
  return failed_cases == 0 && ran > 0 ? 0 : 1;
}

} // namespace atm_test

#define ATM_TEST_CONCAT2(a, b) a##b
#define ATM_TEST_CONCAT(a, b) ATM_TEST_CONCAT2(a, b)

#define ATM_TEST(name)                                                         \
  static void ATM_TEST_CONCAT(atm_test_fn_, name)();                           \
  static ::atm_test::Registrar ATM_TEST_CONCAT(atm_test_reg_, name)(           \
      #name, &ATM_TEST_CONCAT(atm_test_fn_, name));                            \
  static void ATM_TEST_CONCAT(atm_test_fn_, name)()

#define ATM_CHECK(expr)                                                        \
  do {                                                                         \
    if (!(expr))                                                               \
      ::atm_test::report(__FILE__, __LINE__, "CHECK(" #expr ") failed");       \
  } while (0)

#define ATM_CHECK_EQ(a, b)                                                     \
  do {                                                                         \
    const auto &atm_a_ = (a);                                                  \
    const auto &atm_b_ = (b);                                                  \
    if (!(atm_a_ == atm_b_))                                                   \
      ::atm_test::report(__FILE__, __LINE__,                                   \
                         "CHECK_EQ(" #a ", " #b ") failed: " +                 \
                             ::atm_test::show(atm_a_) + " vs " +               \
                             ::atm_test::show(atm_b_));                        \
  } while (0)

#define ATM_CHECK_NEAR(a, b, eps)                                              \
  do {                                                                         \
    const double atm_a_ = static_cast<double>(a);                              \
    const double atm_b_ = static_cast<double>(b);                              \
    if (std::fabs(atm_a_ - atm_b_) > (eps))                                    \
      ::atm_test::report(__FILE__, __LINE__,                                   \
                         "CHECK_NEAR(" #a ", " #b ") failed: " +               \
                             ::atm_test::show(atm_a_) + " vs " +               \
                             ::atm_test::show(atm_b_));                        \
  } while (0)

// Stops the current test (use when later checks would crash).
#define ATM_REQUIRE(expr)                                                      \
  do {                                                                         \
    if (!(expr)) {                                                             \
      ::atm_test::report(__FILE__, __LINE__, "REQUIRE(" #expr ") failed");     \
      return;                                                                  \
    }                                                                          \
  } while (0)
