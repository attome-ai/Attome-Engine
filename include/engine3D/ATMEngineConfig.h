

// SIMD width detection
#if defined(__AVX512F__)
#define SIMD_WIDTH 16
#define USE_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define SIMD_WIDTH 8
#define USE_AVX2 1
#include <immintrin.h>
#elif defined(__AVX__)
#define SIMD_WIDTH 8
#define USE_AVX 1
#include <immintrin.h>
#elif defined(__SSE__)
#define SIMD_WIDTH 4
#define USE_SSE 1
#include <xmmintrin.h>
#else
#define SIMD_WIDTH 1
#define USE_SCALAR 1
#endif
