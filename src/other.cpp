
#include "engine3D/ATMEngine.h"
// SIMD optimization helpers
//--------------------------

// SimdFloat implementation
#if defined(USE_AVX512)

SimdFloat::SimdFloat() {
    v = _mm512_setzero_ps();
}

SimdFloat::SimdFloat(float val) {
    v = _mm512_set1_ps(val);
}

SimdFloat SimdFloat::load(const float* ptr) {
    SimdFloat result;
    result.v = _mm512_loadu_ps(ptr);
    return result;
}

void SimdFloat::store(float* ptr) const {
    _mm512_storeu_ps(ptr, v);
}

SimdFloat SimdFloat::operator+(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm512_add_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator-(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm512_sub_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator*(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm512_mul_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator/(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm512_div_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::sqrt() const {
    SimdFloat result;
    result.v = _mm512_sqrt_ps(v);
    return result;
}

SimdFloat SimdFloat::rsqrt() const {
    SimdFloat result;
    result.v = _mm512_rsqrt14_ps(v); // Approximate reciprocal square root
    return result;
}

SimdFloat SimdFloat::zero() {
    return SimdFloat(0.0f);
}

#elif defined(USE_AVX) || defined(USE_AVX2)

SimdFloat::SimdFloat() {
    v = _mm256_setzero_ps();
}

SimdFloat::SimdFloat(float val) {
    v = _mm256_set1_ps(val);
}

SimdFloat SimdFloat::load(const float* ptr) {
    SimdFloat result;
    result.v = _mm256_loadu_ps(ptr);
    return result;
}

void SimdFloat::store(float* ptr) const {
    _mm256_storeu_ps(ptr, v);
}

SimdFloat SimdFloat::operator+(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm256_add_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator-(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm256_sub_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator*(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm256_mul_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator/(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm256_div_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::sqrt() const {
    SimdFloat result;
    result.v = _mm256_sqrt_ps(v);
    return result;
}

SimdFloat SimdFloat::rsqrt() const {
    SimdFloat result;
    result.v = _mm256_rsqrt_ps(v);
    return result;
}

SimdFloat SimdFloat::zero() {
    return SimdFloat(0.0f);
}

#elif defined(USE_SSE)

SimdFloat::SimdFloat() {
    v = _mm_setzero_ps();
}

SimdFloat::SimdFloat(float val) {
    v = _mm_set1_ps(val);
}

SimdFloat SimdFloat::load(const float* ptr) {
    SimdFloat result;
    result.v = _mm_loadu_ps(ptr);
    return result;
}

void SimdFloat::store(float* ptr) const {
    _mm_storeu_ps(ptr, v);
}

SimdFloat SimdFloat::operator+(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm_add_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator-(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm_sub_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator*(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm_mul_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::operator/(const SimdFloat& rhs) const {
    SimdFloat result;
    result.v = _mm_div_ps(v, rhs.v);
    return result;
}

SimdFloat SimdFloat::sqrt() const {
    SimdFloat result;
    result.v = _mm_sqrt_ps(v);
    return result;
}

SimdFloat SimdFloat::rsqrt() const {
    SimdFloat result;
    result.v = _mm_rsqrt_ps(v);
    return result;
}

SimdFloat SimdFloat::zero() {
    return SimdFloat(0.0f);
}

#else

SimdFloat::SimdFloat() {
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        v[i] = 0.0f;
    }
}

SimdFloat::SimdFloat(float val) {
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        v[i] = val;
    }
}

SimdFloat SimdFloat::load(const float* ptr) {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = ptr[i];
    }
    return result;
}

void SimdFloat::store(float* ptr) const {
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        ptr[i] = v[i];
    }
}

SimdFloat SimdFloat::operator+(const SimdFloat& rhs) const {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] + rhs.v[i];
    }
    return result;
}

SimdFloat SimdFloat::operator-(const SimdFloat& rhs) const {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] - rhs.v[i];
    }
    return result;
}

SimdFloat SimdFloat::operator*(const SimdFloat& rhs) const {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] * rhs.v[i];
    }
    return result;
}

SimdFloat SimdFloat::operator/(const SimdFloat& rhs) const {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] / rhs.v[i];
    }
    return result;
}

SimdFloat SimdFloat::sqrt() const {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = std::sqrt(v[i]);
    }
    return result;
}

SimdFloat SimdFloat::rsqrt() const {
    SimdFloat result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = 1.0f / std::sqrt(v[i]);
    }
    return result;
}

SimdFloat SimdFloat::zero() {
    return SimdFloat(0.0f);
}

#endif

// SimdInt implementation
#if defined(USE_AVX512)

SimdInt::SimdInt() {
    v = _mm512_setzero_si512();
}

SimdInt::SimdInt(int val) {
    v = _mm512_set1_epi32(val);
}

SimdInt SimdInt::load(const int* ptr) {
    SimdInt result;
    result.v = _mm512_loadu_si512(ptr);
    return result;
}

void SimdInt::store(int* ptr) const {
    _mm512_storeu_si512(ptr, v);
}

SimdInt SimdInt::operator&(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm512_and_si512(v, rhs.v);
    return result;
}

SimdInt SimdInt::operator|(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm512_or_si512(v, rhs.v);
    return result;
}

SimdInt SimdInt::operator^(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm512_xor_si512(v, rhs.v);
    return result;
}

#elif defined(USE_AVX) || defined(USE_AVX2)

SimdInt::SimdInt() {
    v = _mm256_setzero_si256();
}

SimdInt::SimdInt(int val) {
    v = _mm256_set1_epi32(val);
}

SimdInt SimdInt::load(const int* ptr) {
    SimdInt result;
    result.v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    return result;
}

void SimdInt::store(int* ptr) const {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v);
}

SimdInt SimdInt::operator&(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm256_and_si256(v, rhs.v);
    return result;
}

SimdInt SimdInt::operator|(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm256_or_si256(v, rhs.v);
    return result;
}

SimdInt SimdInt::operator^(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm256_xor_si256(v, rhs.v);
    return result;
}

#elif defined(USE_SSE)

SimdInt::SimdInt() {
    v = _mm_setzero_si128();
}

SimdInt::SimdInt(int val) {
    v = _mm_set1_epi32(val);
}

SimdInt SimdInt::load(const int* ptr) {
    SimdInt result;
    result.v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    return result;
}

void SimdInt::store(int* ptr) const {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v);
}

SimdInt SimdInt::operator&(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm_and_si128(v, rhs.v);
    return result;
}

SimdInt SimdInt::operator|(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm_or_si128(v, rhs.v);
    return result;
}

SimdInt SimdInt::operator^(const SimdInt& rhs) const {
    SimdInt result;
    result.v = _mm_xor_si128(v, rhs.v);
    return result;
}

#else

SimdInt::SimdInt() {
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        v[i] = 0;
    }
}

SimdInt::SimdInt(int val) {
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        v[i] = val;
    }
}

SimdInt SimdInt::load(const int* ptr) {
    SimdInt result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = ptr[i];
    }
    return result;
}

void SimdInt::store(int* ptr) const {
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        ptr[i] = v[i];
    }
}

SimdInt SimdInt::operator&(const SimdInt& rhs) const {
    SimdInt result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] & rhs.v[i];
    }
    return result;
}

SimdInt SimdInt::operator|(const SimdInt& rhs) const {
    SimdInt result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] | rhs.v[i];
    }
    return result;
}

SimdInt SimdInt::operator^(const SimdInt& rhs) const {
    SimdInt result;
    for (int i = 0; i < SIMD_WIDTH; ++i) {
        result.v[i] = v[i] ^ rhs.v[i];
    }
    return result;
}

#endif

// SimdMask implementation
#if defined(USE_AVX512)

SimdMask::SimdMask() {
    mask = 0;
}

SimdMask::SimdMask(uint32_t bit_mask) {
    mask = bit_mask & 0xFFFF; // 16 lanes for AVX-512
}

bool SimdMask::get(int index) const {
    return (mask & (1 << index)) != 0;
}

void SimdMask::set(int index, bool value) {
    if (value) {
        mask |= (1 << index);
    }
    else {
        mask &= ~(1 << index);
    }
}

SimdMask SimdMask::operator&(const SimdMask& other) const {
    SimdMask result;
    result.mask = mask & other.mask;
    return result;
}

SimdMask SimdMask::operator|(const SimdMask& other) const {
    SimdMask result;
    result.mask = mask | other.mask;
    return result;
}

SimdMask SimdMask::operator^(const SimdMask& other) const {
    SimdMask result;
    result.mask = mask ^ other.mask;
    return result;
}

SimdMask SimdMask::operator~() const {
    SimdMask result;
    result.mask = ~mask & 0xFFFF; // 16 lanes for AVX-512
    return result;
}

bool SimdMask::none() const {
    return mask == 0;
}

bool SimdMask::any() const {
    return mask != 0;
}

bool SimdMask::all() const {
    return mask == 0xFFFF; // 16 lanes for AVX-512
}

#elif defined(USE_AVX2) || defined(USE_AVX)

SimdMask::SimdMask() {
    mask = _mm256_setzero_si256();
}

SimdMask::SimdMask(uint32_t bit_mask) {
    // Convert bit mask to 8 integers (0 or -1) for AVX/AVX2
    int expanded_mask[8];
    for (int i = 0; i < 8; ++i) {
        expanded_mask[i] = (bit_mask & (1 << i)) ? -1 : 0;
    }
    mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(expanded_mask));
}

bool SimdMask::get(int index) const {
    alignas(32) int values[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(values), mask);
    return values[index] != 0;
}

void SimdMask::set(int index, bool value) {
    alignas(32) int values[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(values), mask);
    values[index] = value ? -1 : 0;
    mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(values));
}

SimdMask SimdMask::operator&(const SimdMask& other) const {
    SimdMask result;
    result.mask = _mm256_and_si256(mask, other.mask);
    return result;
}

SimdMask SimdMask::operator|(const SimdMask& other) const {
    SimdMask result;
    result.mask = _mm256_or_si256(mask, other.mask);
    return result;
}

SimdMask SimdMask::operator^(const SimdMask& other) const {
    SimdMask result;
    result.mask = _mm256_xor_si256(mask, other.mask);
    return result;
}

SimdMask SimdMask::operator~() const {
    // Create all ones
    __m256i all_ones = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
    all_ones = _mm256_xor_si256(all_ones, all_ones); // -1 in all lanes

    SimdMask result;
    result.mask = _mm256_xor_si256(mask, all_ones);
    return result;
}

bool SimdMask::none() const {
    return _mm256_testz_si256(mask, mask) != 0;
}

bool SimdMask::any() const {
    return _mm256_testz_si256(mask, mask) == 0;
}

bool SimdMask::all() const {
    __m256i all_ones = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
    all_ones = _mm256_xor_si256(all_ones, all_ones); // -1 in all lanes

    // XOR with all_ones and then check if all zeros
    __m256i temp = _mm256_xor_si256(mask, all_ones);
    return _mm256_testz_si256(temp, temp) != 0;
}

#elif defined(USE_SSE)

SimdMask::SimdMask() {
    mask = _mm_setzero_si128();
}

SimdMask::SimdMask(uint32_t bit_mask) {
    // Convert bit mask to 4 integers (0 or -1) for SSE
    int expanded_mask[4];
    for (int i = 0; i < 4; ++i) {
        expanded_mask[i] = (bit_mask & (1 << i)) ? -1 : 0;
    }
    mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(expanded_mask));
}

bool SimdMask::get(int index) const {
    alignas(16) int values[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(values), mask);
    return values[index] != 0;
}

void SimdMask::set(int index, bool value) {
    alignas(16) int values[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(values), mask);
    values[index] = value ? -1 : 0;
    mask = _mm_load_si128(reinterpret_cast<const __m128i*>(values));
}

SimdMask SimdMask::operator&(const SimdMask& other) const {
    SimdMask result;
    result.mask = _mm_and_si128(mask, other.mask);
    return result;
}

SimdMask SimdMask::operator|(const SimdMask& other) const {
    SimdMask result;
    result.mask = _mm_or_si128(mask, other.mask);
    return result;
}

SimdMask SimdMask::operator^(const SimdMask& other) const {
    SimdMask result;
    result.mask = _mm_xor_si128(mask, other.mask);
    return result;
}

SimdMask SimdMask::operator~() const {
    // Create all ones
    __m128i all_ones = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
    all_ones = _mm_xor_si128(all_ones, all_ones); // -1 in all lanes

    SimdMask result;
    result.mask = _mm_xor_si128(mask, all_ones);
    return result;
}

bool SimdMask::none() const {
#if defined(__SSE4_1__)
    return _mm_testz_si128(mask, mask) != 0;
#else
    // Fallback for SSE2/3
    alignas(16) int values[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(values), mask);
    return values[0] == 0 && values[1] == 0 && values[2] == 0 && values[3] == 0;
#endif
}

bool SimdMask::any() const {
#if defined(__SSE4_1__)
    return _mm_testz_si128(mask, mask) == 0;
#else
    // Fallback for SSE2/3
    alignas(16) int values[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(values), mask);
    return values[0] != 0 || values[1] != 0 || values[2] != 0 || values[3] != 0;
#endif
}

bool SimdMask::all() const {
    __m128i all_ones = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
    all_ones = _mm_xor_si128(all_ones, all_ones); // -1 in all lanes

#if defined(__SSE4_1__)
    // XOR with all_ones and then check if all zeros
    __m128i temp = _mm_xor_si128(mask, all_ones);
    return _mm_testz_si128(temp, temp) != 0;
#else
    // Fallback for SSE2/3
    alignas(16) int values[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(values), mask);
    return values[0] == -1 && values[1] == -1 && values[2] == -1 && values[3] == -1;
#endif
}

#else

SimdMask::SimdMask() {
    mask = 0;
}

SimdMask::SimdMask(uint32_t bit_mask) {
    mask = bit_mask & ((1u << SIMD_WIDTH) - 1);
}

bool SimdMask::get(int index) const {
    return (mask & (1u << index)) != 0;
}

void SimdMask::set(int index, bool value) {
    if (value) {
        mask |= (1u << index);
    }
    else {
        mask &= ~(1u << index);
    }
}

SimdMask SimdMask::operator&(const SimdMask& other) const {
    SimdMask result;
    result.mask = mask & other.mask;
    return result;
}

SimdMask SimdMask::operator|(const SimdMask& other) const { 
    SimdMask result;
    result.mask = mask | other.mask;
    return result;
}

SimdMask SimdMask::operator^(const SimdMask& other) const {
    SimdMask result;
    result.mask = mask ^ other.mask;
    return result;
}

SimdMask SimdMask::operator~() const {
    SimdMask result;
    result.mask = ~mask & ((1u << SIMD_WIDTH) - 1);
    return result;
}

bool SimdMask::none() const {
    return mask == 0;
}

bool SimdMask::any() const {
    return mask != 0;
}

bool SimdMask::all() const {
    return mask == ((1u << SIMD_WIDTH) - 1);
}

#endif

// SimdAABB implementation
#if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_AVX2) || defined(USE_SSE) || !defined(USE_SIMD)

SimdAABB::SimdAABB() {
    min_x = SimdFloat(std::numeric_limits<float>::max());
    min_y = SimdFloat(std::numeric_limits<float>::max());
    min_z = SimdFloat(std::numeric_limits<float>::max());
    max_x = SimdFloat(-std::numeric_limits<float>::max());
    max_y = SimdFloat(-std::numeric_limits<float>::max());
    max_z = SimdFloat(-std::numeric_limits<float>::max());
}

SimdAABB SimdAABB::load(const AABB* boxes) {
    SimdAABB result;

    float min_x_arr[SIMD_WIDTH], min_y_arr[SIMD_WIDTH], min_z_arr[SIMD_WIDTH];
    float max_x_arr[SIMD_WIDTH], max_y_arr[SIMD_WIDTH], max_z_arr[SIMD_WIDTH];

    for (int i = 0; i < SIMD_WIDTH; ++i) {
        min_x_arr[i] = boxes[i].min.x;
        min_y_arr[i] = boxes[i].min.y;
        min_z_arr[i] = boxes[i].min.z;
        max_x_arr[i] = boxes[i].max.x;
        max_y_arr[i] = boxes[i].max.y;
        max_z_arr[i] = boxes[i].max.z;
    }

    result.min_x = SimdFloat::load(min_x_arr);
    result.min_y = SimdFloat::load(min_y_arr);
    result.min_z = SimdFloat::load(min_z_arr);
    result.max_x = SimdFloat::load(max_x_arr);
    result.max_y = SimdFloat::load(max_y_arr);
    result.max_z = SimdFloat::load(max_z_arr);

    return result;
}

void SimdAABB::store(AABB* boxes) const {
    float min_x_arr[SIMD_WIDTH], min_y_arr[SIMD_WIDTH], min_z_arr[SIMD_WIDTH];
    float max_x_arr[SIMD_WIDTH], max_y_arr[SIMD_WIDTH], max_z_arr[SIMD_WIDTH];

    min_x.store(min_x_arr);
    min_y.store(min_y_arr);
    min_z.store(min_z_arr);
    max_x.store(max_x_arr);
    max_y.store(max_y_arr);
    max_z.store(max_z_arr);

    for (int i = 0; i < SIMD_WIDTH; ++i) {
        boxes[i].min.x = min_x_arr[i];
        boxes[i].min.y = min_y_arr[i];
        boxes[i].min.z = min_z_arr[i];
        boxes[i].max.x = max_x_arr[i];
        boxes[i].max.y = max_y_arr[i];
        boxes[i].max.z = max_z_arr[i];
    }
}



SimdMask SimdAABB::overlaps(const SimdAABB& other) const {
#if defined(USE_AVX512)
    __mmask16 overlap_mask = 0;

    // Get SIMD vectors for comparison
    __m512 this_min_x = min_x.v;
    __m512 this_min_y = min_y.v;
    __m512 this_min_z = min_z.v;
    __m512 this_max_x = max_x.v;
    __m512 this_max_y = max_y.v;
    __m512 this_max_z = max_z.v;

    __m512 other_min_x = other.min_x.v;
    __m512 other_min_y = other.min_y.v;
    __m512 other_min_z = other.min_z.v;
    __m512 other_max_x = other.max_x.v;
    __m512 other_max_y = other.max_y.v;
    __m512 other_max_z = other.max_z.v;

    // Check overlap condition for each axis
    __mmask16 x_overlap = _mm512_cmp_ps_mask(this_max_x, other_min_x, _CMP_GE_OQ) &
        _mm512_cmp_ps_mask(this_min_x, other_max_x, _CMP_LE_OQ);

    __mmask16 y_overlap = _mm512_cmp_ps_mask(this_max_y, other_min_y, _CMP_GE_OQ) &
        _mm512_cmp_ps_mask(this_min_y, other_max_y, _CMP_LE_OQ);

    __mmask16 z_overlap = _mm512_cmp_ps_mask(this_max_z, other_min_z, _CMP_GE_OQ) &
        _mm512_cmp_ps_mask(this_min_z, other_max_z, _CMP_LE_OQ);

    // Boxes overlap if they overlap on all axes
    overlap_mask = x_overlap & y_overlap & z_overlap;

    SimdMask result;
    result.mask = overlap_mask;
    return result;

#elif defined(USE_AVX) || defined(USE_AVX2)
    // Using AVX/AVX2 implementation
    // Compare each dimension for overlap
    __m256 x_min_overlap = _mm256_cmp_ps(max_x.v, other.min_x.v, _CMP_GE_OQ);
    __m256 x_max_overlap = _mm256_cmp_ps(min_x.v, other.max_x.v, _CMP_LE_OQ);
    __m256 x_overlaps = _mm256_and_ps(x_min_overlap, x_max_overlap);

    __m256 y_min_overlap = _mm256_cmp_ps(max_y.v, other.min_y.v, _CMP_GE_OQ);
    __m256 y_max_overlap = _mm256_cmp_ps(min_y.v, other.max_y.v, _CMP_LE_OQ);
    __m256 y_overlaps = _mm256_and_ps(y_min_overlap, y_max_overlap);

    __m256 z_min_overlap = _mm256_cmp_ps(max_z.v, other.min_z.v, _CMP_GE_OQ);
    __m256 z_max_overlap = _mm256_cmp_ps(min_z.v, other.max_z.v, _CMP_LE_OQ);
    __m256 z_overlaps = _mm256_and_ps(z_min_overlap, z_max_overlap);

    // All dimensions must overlap
    __m256 all_overlaps = _mm256_and_ps(x_overlaps, _mm256_and_ps(y_overlaps, z_overlaps));

    SimdMask result;
    result.mask = _mm256_castps_si256(all_overlaps);
    return result;

#elif defined(USE_SSE)
    // Using SSE implementation
    // Compare each dimension for overlap
    __m128 x_min_overlap = _mm_cmpge_ps(max_x.v, other.min_x.v);
    __m128 x_max_overlap = _mm_cmple_ps(min_x.v, other.max_x.v);
    __m128 x_overlaps = _mm_and_ps(x_min_overlap, x_max_overlap);

    __m128 y_min_overlap = _mm_cmpge_ps(max_y.v, other.min_y.v);
    __m128 y_max_overlap = _mm_cmple_ps(min_y.v, other.max_y.v);
    __m128 y_overlaps = _mm_and_ps(y_min_overlap, y_max_overlap);

    __m128 z_min_overlap = _mm_cmpge_ps(max_z.v, other.min_z.v);
    __m128 z_max_overlap = _mm_cmple_ps(min_z.v, other.max_z.v);
    __m128 z_overlaps = _mm_and_ps(z_min_overlap, z_max_overlap);

    // All dimensions must overlap
    __m128 all_overlaps = _mm_and_ps(x_overlaps, _mm_and_ps(y_overlaps, z_overlaps));

    SimdMask result;
    result.mask = _mm_castps_si128(all_overlaps);
    return result;

#else
    // Scalar implementation
    SimdMask result;
    result.mask = 0;

    float min_x_a[SIMD_WIDTH], min_y_a[SIMD_WIDTH], min_z_a[SIMD_WIDTH];
    float max_x_a[SIMD_WIDTH], max_y_a[SIMD_WIDTH], max_z_a[SIMD_WIDTH];

    float min_x_b[SIMD_WIDTH], min_y_b[SIMD_WIDTH], min_z_b[SIMD_WIDTH];
    float max_x_b[SIMD_WIDTH], max_y_b[SIMD_WIDTH], max_z_b[SIMD_WIDTH];

    min_x.store(min_x_a); min_y.store(min_y_a); min_z.store(min_z_a);
    max_x.store(max_x_a); max_y.store(max_y_a); max_z.store(max_z_a);

    other.min_x.store(min_x_b); other.min_y.store(min_y_b); other.min_z.store(min_z_b);
    other.max_x.store(max_x_b); other.max_y.store(max_y_b); other.max_z.store(max_z_b);

    for (int i = 0; i < SIMD_WIDTH; ++i) {
        bool x_overlap = max_x_a[i] >= min_x_b[i] && min_x_a[i] <= max_x_b[i];
        bool y_overlap = max_y_a[i] >= min_y_b[i] && min_y_a[i] <= max_y_b[i];
        bool z_overlap = max_z_a[i] >= min_z_b[i] && min_z_a[i] <= max_z_b[i];

        if (x_overlap && y_overlap && z_overlap) {
            result.mask |= (1u << i);
        }
    }

    return result;
#endif
}

SimdMask SimdAABB::contains(const SimdAABB& other) const {
#if defined(USE_AVX512)
    __mmask16 contains_mask = 0;

    // Check containment condition for each axis
    __mmask16 x_contains = _mm512_cmp_ps_mask(min_x.v, other.min_x.v, _CMP_LE_OQ) &
        _mm512_cmp_ps_mask(max_x.v, other.max_x.v, _CMP_GE_OQ);

    __mmask16 y_contains = _mm512_cmp_ps_mask(min_y.v, other.min_y.v, _CMP_LE_OQ) &
        _mm512_cmp_ps_mask(max_y.v, other.max_y.v, _CMP_GE_OQ);

    __mmask16 z_contains = _mm512_cmp_ps_mask(min_z.v, other.min_z.v, _CMP_LE_OQ) &
        _mm512_cmp_ps_mask(max_z.v, other.max_z.v, _CMP_GE_OQ);

    // All axes must be contained
    contains_mask = x_contains & y_contains & z_contains;

    SimdMask result;
    result.mask = contains_mask;
    return result;

#elif defined(USE_AVX) || defined(USE_AVX2)
    // Using AVX/AVX2 implementation
    __m256 x_min_contains = _mm256_cmp_ps(min_x.v, other.min_x.v, _CMP_LE_OQ);
    __m256 x_max_contains = _mm256_cmp_ps(max_x.v, other.max_x.v, _CMP_GE_OQ);
    __m256 x_contains = _mm256_and_ps(x_min_contains, x_max_contains);

    __m256 y_min_contains = _mm256_cmp_ps(min_y.v, other.min_y.v, _CMP_LE_OQ);
    __m256 y_max_contains = _mm256_cmp_ps(max_y.v, other.max_y.v, _CMP_GE_OQ);
    __m256 y_contains = _mm256_and_ps(y_min_contains, y_max_contains);

    __m256 z_min_contains = _mm256_cmp_ps(min_z.v, other.min_z.v, _CMP_LE_OQ);
    __m256 z_max_contains = _mm256_cmp_ps(max_z.v, other.max_z.v, _CMP_GE_OQ);
    __m256 z_contains = _mm256_and_ps(z_min_contains, z_max_contains);

    // All dimensions must contain
    __m256 all_contains = _mm256_and_ps(x_contains, _mm256_and_ps(y_contains, z_contains));

    SimdMask result;
    result.mask = _mm256_castps_si256(all_contains);
    return result;

#elif defined(USE_SSE)
    // Using SSE implementation
    __m128 x_min_contains = _mm_cmple_ps(min_x.v, other.min_x.v);
    __m128 x_max_contains = _mm_cmpge_ps(max_x.v, other.max_x.v);
    __m128 x_contains = _mm_and_ps(x_min_contains, x_max_contains);

    __m128 y_min_contains = _mm_cmple_ps(min_y.v, other.min_y.v);
    __m128 y_max_contains = _mm_cmpge_ps(max_y.v, other.max_y.v);
    __m128 y_contains = _mm_and_ps(y_min_contains, y_max_contains);

    __m128 z_min_contains = _mm_cmple_ps(min_z.v, other.min_z.v);
    __m128 z_max_contains = _mm_cmpge_ps(max_z.v, other.max_z.v);
    __m128 z_contains = _mm_and_ps(z_min_contains, z_max_contains);

    // All dimensions must contain
    __m128 all_contains = _mm_and_ps(x_contains, _mm_and_ps(y_contains, z_contains));

    SimdMask result;
    result.mask = _mm_castps_si128(all_contains);
    return result;

#else
    // Scalar implementation
    SimdMask result;
    result.mask = 0;

    float min_x_a[SIMD_WIDTH], min_y_a[SIMD_WIDTH], min_z_a[SIMD_WIDTH];
    float max_x_a[SIMD_WIDTH], max_y_a[SIMD_WIDTH], max_z_a[SIMD_WIDTH];

    float min_x_b[SIMD_WIDTH], min_y_b[SIMD_WIDTH], min_z_b[SIMD_WIDTH];
    float max_x_b[SIMD_WIDTH], max_y_b[SIMD_WIDTH], max_z_b[SIMD_WIDTH];

    min_x.store(min_x_a); min_y.store(min_y_a); min_z.store(min_z_a);
    max_x.store(max_x_a); max_y.store(max_y_a); max_z.store(max_z_a);

    other.min_x.store(min_x_b); other.min_y.store(min_y_b); other.min_z.store(min_z_b);
    other.max_x.store(max_x_b); other.max_y.store(max_y_b); other.max_z.store(max_z_b);

    for (int i = 0; i < SIMD_WIDTH; ++i) {
        bool x_contains = min_x_a[i] <= min_x_b[i] && max_x_a[i] >= max_x_b[i];
        bool y_contains = min_y_a[i] <= min_y_b[i] && max_y_a[i] >= max_y_b[i];
        bool z_contains = min_z_a[i] <= min_z_b[i] && max_z_a[i] >= max_z_b[i];

        if (x_contains && y_contains && z_contains) {
            result.mask |= (1u << i);
        }
    }

    return result;
#endif
}

#endif

// Mat4SIMD implementation
#if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_AVX2) || defined(USE_SSE) || !defined(USE_SIMD)

Mat4SIMD::Mat4SIMD() {
    for (int i = 0; i < 16; ++i) {
        m[i] = SimdFloat(0.0f);
    }
}

Mat4SIMD::Mat4SIMD(const glm::mat4& mat) {
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float values[SIMD_WIDTH];
            for (int i = 0; i < SIMD_WIDTH; ++i) {
                values[i] = mat[col][row];
            }
            m[col * 4 + row] = SimdFloat::load(values);
        }
    }
}

Mat4SIMD Mat4SIMD::load(const glm::mat4* matrices) {
    Mat4SIMD result;

    // Transpose matrices for SIMD operations
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float values[SIMD_WIDTH];
            for (int i = 0; i < SIMD_WIDTH; ++i) {
                values[i] = matrices[i][col][row];
            }
            result.m[col * 4 + row] = SimdFloat::load(values);
        }
    }

    return result;
}

void Mat4SIMD::store(glm::mat4* matrices) const {
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float values[SIMD_WIDTH];
            m[col * 4 + row].store(values);

            for (int i = 0; i < SIMD_WIDTH; ++i) {
                matrices[i][col][row] = values[i];
            }
        }
    }
}

Mat4SIMD Mat4SIMD::operator*(const Mat4SIMD& rhs) const {
    Mat4SIMD result;

    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            // Compute dot product of row of lhs with column of rhs
            SimdFloat sum = SimdFloat::zero();

            for (int k = 0; k < 4; ++k) {
                SimdFloat a = m[k * 4 + row];
                SimdFloat b = rhs.m[col * 4 + k];
                sum = sum + (a * b);
            }

            result.m[col * 4 + row] = sum;
        }
    }

    return result;
}

Mat4SIMD Mat4SIMD::identity() {
    Mat4SIMD result;

    for (int i = 0; i < 4; ++i) {
        result.m[i * 4 + i] = SimdFloat(1.0f);
    }

    return result;
}

Mat4SIMD Mat4SIMD::translation(const SimdFloat& x, const SimdFloat& y, const SimdFloat& z) {
    Mat4SIMD result = identity();

    result.m[3 * 4 + 0] = x;  // m[12]
    result.m[3 * 4 + 1] = y;  // m[13]
    result.m[3 * 4 + 2] = z;  // m[14]

    return result;
}

Mat4SIMD Mat4SIMD::rotation(const SimdFloat& qx, const SimdFloat& qy, const SimdFloat& qz, const SimdFloat& qw) {
    Mat4SIMD result;

    // Calculate quaternion terms
    SimdFloat qx2 = qx * qx;
    SimdFloat qy2 = qy * qy;
    SimdFloat qz2 = qz * qz;
    SimdFloat qw2 = qw * qw;

    SimdFloat qxqy = qx * qy;
    SimdFloat qxqz = qx * qz;
    SimdFloat qxqw = qx * qw;
    SimdFloat qyqz = qy * qz;
    SimdFloat qyqw = qy * qw;
    SimdFloat qzqw = qz * qw;

    // First row
    result.m[0] = qw2 + qx2 - qy2 - qz2;
    result.m[1] = SimdFloat(2.0f) * (qxqy - qzqw);
    result.m[2] = SimdFloat(2.0f) * (qxqz + qyqw);
    result.m[3] = SimdFloat(0.0f);

    // Second row
    result.m[4] = SimdFloat(2.0f) * (qxqy + qzqw);
    result.m[5] = qw2 - qx2 + qy2 - qz2;
    result.m[6] = SimdFloat(2.0f) * (qyqz - qxqw);
    result.m[7] = SimdFloat(0.0f);

    // Third row
    result.m[8] = SimdFloat(2.0f) * (qxqz - qyqw);
    result.m[9] = SimdFloat(2.0f) * (qyqz + qxqw);
    result.m[10] = qw2 - qx2 - qy2 + qz2;
    result.m[11] = SimdFloat(0.0f);

    // Fourth row
    result.m[12] = SimdFloat(0.0f);
    result.m[13] = SimdFloat(0.0f);
    result.m[14] = SimdFloat(0.0f);
    result.m[15] = SimdFloat(1.0f);

    return result;
}

Mat4SIMD Mat4SIMD::scale(const SimdFloat& x, const SimdFloat& y, const SimdFloat& z) {
    Mat4SIMD result = identity();

    result.m[0] = x;  // m[0]
    result.m[5] = y;  // m[5]
    result.m[10] = z;  // m[10]

    return result;
}

#endif

// SimdMatrixOps namespace implementation
namespace SimdMatrixOps {
    // Matrix multiplication with SIMD
    void multiplyBatch(const glm::mat4* matrices_a, const glm::mat4* matrices_b,
        glm::mat4* results, int count) {

        for (int i = 0; i < count; ++i) {
            results[i] = matrices_a[i] * matrices_b[i];
        }

        // Note: A full SIMD implementation would use the SIMD types
        // defined earlier to process multiple matrices in parallel
    }

    // Transform vectors in batches
    void transformPoints(const glm::mat4* matrices, const glm::vec3* points,
        glm::vec3* results, int count) {

        for (int i = 0; i < count; ++i) {
            glm::vec4 homogeneous(points[i], 1.0f);
            homogeneous = matrices[i] * homogeneous;
            results[i] = glm::vec3(homogeneous) / homogeneous.w;
        }
    }

    // Batch transform updates by hierarchy depth
    void updateTransformsByDepth(const glm::mat4* local, const glm::mat4* parent,
        glm::mat4* world, const int* parent_indices,
        int count, int depth) {

        for (int i = 0; i < count; ++i) {
            int parent_idx = parent_indices[i];
            if (parent_idx >= 0) {
                world[i] = parent[parent_idx] * local[i];
            }
            else {
                world[i] = local[i];
            }
        }
    }
}

// BitMaskOps namespace implementation
namespace BitMaskOps {
    // Find first set bit
    int findFirstSetBit(uint64_t mask) {
        if (mask == 0) return -1;

#if defined(_MSC_VER) && defined(_WIN64)
        unsigned long index;
        _BitScanForward64(&index, mask);
        return static_cast<int>(index);
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_ctzll(mask);
#else
        // Fallback implementation
        for (int i = 0; i < 64; ++i) {
            if (mask & (1ULL << i)) {
                return i;
            }
        }
        return -1;
#endif
    }

    // Count set bits
    int countSetBits(uint64_t mask) {
#if defined(_MSC_VER) && defined(_WIN64)
        return static_cast<int>(__popcnt64(mask));
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_popcountll(mask);
#else
        // Fallback implementation
        int count = 0;
        while (mask) {
            count += mask & 1;
            mask >>= 1;
        }
        return count;
#endif
    }

    // Process entities using bitmasks to avoid branches
    void processEntityBitMasked(void* entity_data, uint64_t mask,
        void (*process_func)(void* data, int entity_idx), int base_idx) {

        while (mask) {
            int bit_idx = findFirstSetBit(mask);
            process_func(entity_data, base_idx + bit_idx);
            mask &= ~(1ULL << bit_idx);
        }
    }
}

// Core Entity System Components
//------------------------------

// AABB implementation
AABB::AABB() : min(0), max(0) {}

AABB::AABB(const glm::vec3& min, const glm::vec3& max) : min(min), max(max) {}

bool AABB::contains(const AABB& other) const {
    return
        min.x <= other.min.x && min.y <= other.min.y && min.z <= other.min.z &&
        max.x >= other.max.x && max.y >= other.max.y && max.z >= other.max.z;
}

bool AABB::overlaps(const AABB& other) const {
    return
        min.x <= other.max.x && max.x >= other.min.x &&
        min.y <= other.max.y && max.y >= other.min.y &&
        min.z <= other.max.z && max.z >= other.min.z;
}

glm::vec3 AABB::center() const {
    return (min + max) * 0.5f;
}

glm::vec3 AABB::extents() const {
    return (max - min) * 0.5f;
}

float AABB::volume() const {
    glm::vec3 diff = max - min;
    return diff.x * diff.y * diff.z;
}
