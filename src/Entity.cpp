#include "ATMEngine/Entity.h"
#include "ATMEngine/Config.h"
#include <stdlib.h>

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

