#include <engine3D/ATMEngine.h>

// SimdVec3 implementation
#if defined(USE_AVX512)
SimdVec3 SimdVec3::load(const float* x_ptr, const float* y_ptr, const float* z_ptr) {
    SimdVec3 result;
    result.x = _mm512_loadu_ps(x_ptr);
    result.y = _mm512_loadu_ps(y_ptr);
    result.z = _mm512_loadu_ps(z_ptr);
    return result;
}

void SimdVec3::store(float* x_ptr, float* y_ptr, float* z_ptr) const {
    _mm512_storeu_ps(x_ptr, x);
    _mm512_storeu_ps(y_ptr, y);
    _mm512_storeu_ps(z_ptr, z);
}

SimdVec3 SimdVec3::add(const SimdVec3& other) const {
    SimdVec3 result;
    result.x = _mm512_add_ps(x, other.x);
    result.y = _mm512_add_ps(y, other.y);
    result.z = _mm512_add_ps(z, other.z);
    return result;
}

SimdVec3 SimdVec3::sub(const SimdVec3& other) const {
    SimdVec3 result;
    result.x = _mm512_sub_ps(x, other.x);
    result.y = _mm512_sub_ps(y, other.y);
    result.z = _mm512_sub_ps(z, other.z);
    return result;
}

SimdVec3 SimdVec3::mul(const SimdVec3& other) const {
    SimdVec3 result;
    result.x = _mm512_mul_ps(x, other.x);
    result.y = _mm512_mul_ps(y, other.y);
    result.z = _mm512_mul_ps(z, other.z);
    return result;
}

SimdVec3 SimdVec3::mul(float scalar) const {
    __m512 scalar_vec = _mm512_set1_ps(scalar);
    SimdVec3 result;
    result.x = _mm512_mul_ps(x, scalar_vec);
    result.y = _mm512_mul_ps(y, scalar_vec);
    result.z = _mm512_mul_ps(z, scalar_vec);
    return result;
}

SimdVec3 SimdVec3::cross(const SimdVec3& other) const {
    SimdVec3 result;

    // Cross product components
    __m512 temp1 = _mm512_mul_ps(y, other.z);
    __m512 temp2 = _mm512_mul_ps(z, other.y);
    result.x = _mm512_sub_ps(temp1, temp2);

    temp1 = _mm512_mul_ps(z, other.x);
    temp2 = _mm512_mul_ps(x, other.z);
    result.y = _mm512_sub_ps(temp1, temp2);

    temp1 = _mm512_mul_ps(x, other.y);
    temp2 = _mm512_mul_ps(y, other.x);
    result.z = _mm512_sub_ps(temp1, temp2);

    return result;
}

#elif defined(USE_AVX) || defined(USE_AVX2)
// AVX implementation
SimdVec3 SimdVec3::load(const float* x_ptr, const float* y_ptr, const float* z_ptr) {
    SimdVec3 result;
    result.x = _mm256_loadu_ps(x_ptr);
    result.y = _mm256_loadu_ps(y_ptr);
    result.z = _mm256_loadu_ps(z_ptr);
    return result;
}

void SimdVec3::store(float* x_ptr, float* y_ptr, float* z_ptr) const {
    _mm256_storeu_ps(x_ptr, x);
    _mm256_storeu_ps(y_ptr, y);
    _mm256_storeu_ps(z_ptr, z);
}

SimdVec3 SimdVec3::add(const SimdVec3& other) const {
    SimdVec3 result;
    result.x = _mm256_add_ps(x, other.x);
    result.y = _mm256_add_ps(y, other.y);
    result.z = _mm256_add_ps(z, other.z);
    return result;
}

SimdVec3 SimdVec3::sub(const SimdVec3& other) const {
    SimdVec3 result;
    result.x = _mm256_sub_ps(x, other.x);
    result.y = _mm256_sub_ps(y, other.y);
    result.z = _mm256_sub_ps(z, other.z);
    return result;
}

SimdVec3 SimdVec3::mul(const SimdVec3& other) const {
    SimdVec3 result;
    result.x = _mm256_mul_ps(x, other.x);
    result.y = _mm256_mul_ps(y, other.y);
    result.z = _mm256_mul_ps(z, other.z);
    return result;
}

SimdVec3 SimdVec3::mul(float scalar) const {
    __m256 scalar_vec = _mm256_set1_ps(scalar);
    SimdVec3 result;
    result.x = _mm256_mul_ps(x, scalar_vec);
    result.y = _mm256_mul_ps(y, scalar_vec);
    result.z = _mm256_mul_ps(z, scalar_vec);
    return result;
}

SimdVec3 SimdVec3::cross(const SimdVec3& other) const {
    SimdVec3 result;

    // Cross product components
    __m256 temp1 = _mm256_mul_ps(y, other.z);
    __m256 temp2 = _mm256_mul_ps(z, other.y);
    result.x = _mm256_sub_ps(temp1, temp2);

    temp1 = _mm256_mul_ps(z, other.x);
    temp2 = _mm256_mul_ps(x, other.z);
    result.y = _mm256_sub_ps(temp1, temp2);

    temp1 = _mm256_mul_ps(x, other.y);
    temp2 = _mm256_mul_ps(y, other.x);
    result.z = _mm256_sub_ps(temp1, temp2);

    return result;
}

#elif defined(USE_SSE)
// SSE implementation
SimdVec3 SimdVec3::load(const float* x_ptr, const float* y_ptr, const float* z_ptr) {
    SimdVec3 result;
    result.x = _mm_loadu_ps(x_ptr);
    result.y = _mm_loadu_ps(y_ptr);
    result.z = _mm_loadu_ps(z_ptr);
    return result;
}

void SimdVec3::store(float* x_ptr, float* y_ptr, float* z_ptr) const {
    _mm_storeu_ps(x_ptr, x);
    _mm_storeu_ps(y_ptr, y);
    _mm_storeu_ps(z_ptr, z);
}

SimdVec3 SimdVec3::add(const SimdVec3& other) const {
    SimdVec3 result;
    result.x = _mm_add_ps(x, other.x);
    result.y = _mm_add_ps(y, other.y);
    result.z = _mm_add_ps(z, other.z);
    return result;
}

SimdVec3 SimdVec3::sub(const SimdVec3& other) const {
    SimdVec3 result;
    result.x = _mm_sub_ps(x, other.x);
    result.y = _mm_sub_ps(y, other.y);
    result.z = _mm_sub_ps(z, other.z);
    return result;
}

SimdVec3 SimdVec3::mul(const SimdVec3& other) const {
    SimdVec3 result;
    result.x = _mm_mul_ps(x, other.x);
    result.y = _mm_mul_ps(y, other.y);
    result.z = _mm_mul_ps(z, other.z);
    return result;
}

SimdVec3 SimdVec3::mul(float scalar) const {
    __m128 scalar_vec = _mm_set1_ps(scalar);
    SimdVec3 result;
    result.x = _mm_mul_ps(x, scalar_vec);
    result.y = _mm_mul_ps(y, scalar_vec);
    result.z = _mm_mul_ps(z, scalar_vec);
    return result;
}

SimdVec3 SimdVec3::cross(const SimdVec3& other) const {
    SimdVec3 result;

    // Cross product components
    __m128 temp1 = _mm_mul_ps(y, other.z);
    __m128 temp2 = _mm_mul_ps(z, other.y);
    result.x = _mm_sub_ps(temp1, temp2);

    temp1 = _mm_mul_ps(z, other.x);
    temp2 = _mm_mul_ps(x, other.z);
    result.y = _mm_sub_ps(temp1, temp2);

    temp1 = _mm_mul_ps(x, other.y);
    temp2 = _mm_mul_ps(y, other.x);
    result.z = _mm_sub_ps(temp1, temp2);

    return result;
}

#else
// Scalar fallback implementation
SimdVec3 SimdVec3::load(const float* x_ptr, const float* y_ptr, const float* z_ptr) {
    SimdVec3 result;
    for (int i = 0; i < SIMD_WIDTH; i++) {
        result.x[i] = x_ptr[i];
        result.y[i] = y_ptr[i];
        result.z[i] = z_ptr[i];
    }
    return result;
}

void SimdVec3::store(float* x_ptr, float* y_ptr, float* z_ptr) const {
    for (int i = 0; i < SIMD_WIDTH; i++) {
        x_ptr[i] = x[i];
        y_ptr[i] = y[i];
        z_ptr[i] = z[i];
    }
}

SimdVec3 SimdVec3::add(const SimdVec3& other) const {
    SimdVec3 result;
    for (int i = 0; i < SIMD_WIDTH; i++) {
        result.x[i] = x[i] + other.x[i];
        result.y[i] = y[i] + other.y[i];
        result.z[i] = z[i] + other.z[i];
    }
    return result;
}

SimdVec3 SimdVec3::sub(const SimdVec3& other) const {
    SimdVec3 result;
    for (int i = 0; i < SIMD_WIDTH; i++) {
        result.x[i] = x[i] - other.x[i];
        result.y[i] = y[i] - other.y[i];
        result.z[i] = z[i] - other.z[i];
    }
    return result;
}

SimdVec3 SimdVec3::mul(const SimdVec3& other) const {
    SimdVec3 result;
    for (int i = 0; i < SIMD_WIDTH; i++) {
        result.x[i] = x[i] * other.x[i];
        result.y[i] = y[i] * other.y[i];
        result.z[i] = z[i] * other.z[i];
    }
    return result;
}

SimdVec3 SimdVec3::mul(float scalar) const {
    SimdVec3 result;
    for (int i = 0; i < SIMD_WIDTH; i++) {
        result.x[i] = x[i] * scalar;
        result.y[i] = y[i] * scalar;
        result.z[i] = z[i] * scalar;
    }
    return result;
}

SimdVec3 SimdVec3::cross(const SimdVec3& other) const {
    SimdVec3 result;
    for (int i = 0; i < SIMD_WIDTH; i++) {
        result.x[i] = y[i] * other.z[i] - z[i] * other.y[i];
        result.y[i] = z[i] * other.x[i] - x[i] * other.z[i];
        result.z[i] = x[i] * other.y[i] - y[i] * other.x[i];
    }
    return result;
}
#endif

// SimdQuat implementation
#if defined(USE_AVX512)
SimdQuat SimdQuat::load(const float* x_ptr, const float* y_ptr, const float* z_ptr, const float* w_ptr) {
    SimdQuat result;
    result.x = _mm512_loadu_ps(x_ptr);
    result.y = _mm512_loadu_ps(y_ptr);
    result.z = _mm512_loadu_ps(z_ptr);
    result.w = _mm512_loadu_ps(w_ptr);
    return result;
}

void SimdQuat::store(float* x_ptr, float* y_ptr, float* z_ptr, float* w_ptr) const {
    _mm512_storeu_ps(x_ptr, x);
    _mm512_storeu_ps(y_ptr, y);
    _mm512_storeu_ps(z_ptr, z);
    _mm512_storeu_ps(w_ptr, w);
}

SimdQuat SimdQuat::mul(const SimdQuat& other) const {
    SimdQuat result;

    // q1 * q2 implementation with SIMD
    __m512 q1x_q2w = _mm512_mul_ps(x, other.w);
    __m512 q1y_q2z = _mm512_mul_ps(y, other.z);
    __m512 q1z_q2y = _mm512_mul_ps(z, other.y);
    __m512 q1w_q2x = _mm512_mul_ps(w, other.x);

    __m512 q1y_q2w = _mm512_mul_ps(y, other.w);
    __m512 q1z_q2x = _mm512_mul_ps(z, other.x);
    __m512 q1w_q2y = _mm512_mul_ps(w, other.y);
    __m512 q1x_q2z = _mm512_mul_ps(x, other.z);

    __m512 q1z_q2w = _mm512_mul_ps(z, other.w);
    __m512 q1w_q2z = _mm512_mul_ps(w, other.z);
    __m512 q1x_q2y = _mm512_mul_ps(x, other.y);
    __m512 q1y_q2x = _mm512_mul_ps(y, other.x);

    __m512 q1w_q2w = _mm512_mul_ps(w, other.w);
    __m512 q1x_q2x = _mm512_mul_ps(x, other.x);
    __m512 q1y_q2y = _mm512_mul_ps(y, other.y);
    __m512 q1z_q2z = _mm512_mul_ps(z, other.z);

    // Calculate components with proper signs
    __m512 temp1, temp2;

    temp1 = _mm512_add_ps(q1x_q2w, q1w_q2x);
    temp2 = _mm512_sub_ps(q1y_q2z, q1z_q2y);
    result.x = _mm512_add_ps(temp1, temp2);

    temp1 = _mm512_add_ps(q1y_q2w, q1w_q2y);
    temp2 = _mm512_sub_ps(q1z_q2x, q1x_q2z);
    result.y = _mm512_add_ps(temp1, temp2);

    temp1 = _mm512_add_ps(q1z_q2w, q1w_q2z);
    temp2 = _mm512_sub_ps(q1x_q2y, q1y_q2x);
    result.z = _mm512_add_ps(temp1, temp2);

    temp1 = _mm512_sub_ps(q1w_q2w, q1x_q2x);
    temp2 = _mm512_sub_ps(temp1, q1y_q2y);
    result.w = _mm512_sub_ps(temp2, q1z_q2z);

    return result;
}

SimdVec3 SimdQuat::rotate(const SimdVec3& v) const {
    // Optimized quaternion-vector rotation
    // v' = q * v * q^-1
    // Optimized formula: v' = v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v)

    // Pure vector part of quaternion
    SimdVec3 q_vec;
    q_vec.x = x;
    q_vec.y = y;
    q_vec.z = z;

    // Cross product of q_vec and v
    SimdVec3 temp = q_vec.cross(v);

    // q.w * v
    SimdVec3 w_mul_v;
    w_mul_v.x = _mm512_mul_ps(w, v.x);
    w_mul_v.y = _mm512_mul_ps(w, v.y);
    w_mul_v.z = _mm512_mul_ps(w, v.z);

    // cross(q.xyz, v) + q.w * v
    SimdVec3 temp2;
    temp2.x = _mm512_add_ps(temp.x, w_mul_v.x);
    temp2.y = _mm512_add_ps(temp.y, w_mul_v.y);
    temp2.z = _mm512_add_ps(temp.z, w_mul_v.z);

    // cross(q.xyz, result of previous)
    SimdVec3 temp3 = q_vec.cross(temp2);

    // Scale by 2
    __m512 two = _mm512_set1_ps(2.0f);
    temp3.x = _mm512_mul_ps(temp3.x, two);
    temp3.y = _mm512_mul_ps(temp3.y, two);
    temp3.z = _mm512_mul_ps(temp3.z, two);

    // Add original vector
    SimdVec3 result;
    result.x = _mm512_add_ps(v.x, temp3.x);
    result.y = _mm512_add_ps(v.y, temp3.y);
    result.z = _mm512_add_ps(v.z, temp3.z);

    return result;
}

#elif defined(USE_AVX) || defined(USE_AVX2)
SimdQuat SimdQuat::load(const float* x_ptr, const float* y_ptr, const float* z_ptr, const float* w_ptr) {
    SimdQuat result;
    result.x = _mm256_loadu_ps(x_ptr);
    result.y = _mm256_loadu_ps(y_ptr);
    result.z = _mm256_loadu_ps(z_ptr);
    result.w = _mm256_loadu_ps(w_ptr);
    return result;
}

void SimdQuat::store(float* x_ptr, float* y_ptr, float* z_ptr, float* w_ptr) const {
    _mm256_storeu_ps(x_ptr, x);
    _mm256_storeu_ps(y_ptr, y);
    _mm256_storeu_ps(z_ptr, z);
    _mm256_storeu_ps(w_ptr, w);
}

SimdQuat SimdQuat::mul(const SimdQuat& other) const {
    SimdQuat result;

    // q1 * q2 implementation with SIMD
    __m256 q1x_q2w = _mm256_mul_ps(x, other.w);
    __m256 q1y_q2z = _mm256_mul_ps(y, other.z);
    __m256 q1z_q2y = _mm256_mul_ps(z, other.y);
    __m256 q1w_q2x = _mm256_mul_ps(w, other.x);

    __m256 q1y_q2w = _mm256_mul_ps(y, other.w);
    __m256 q1z_q2x = _mm256_mul_ps(z, other.x);
    __m256 q1w_q2y = _mm256_mul_ps(w, other.y);
    __m256 q1x_q2z = _mm256_mul_ps(x, other.z);

    __m256 q1z_q2w = _mm256_mul_ps(z, other.w);
    __m256 q1w_q2z = _mm256_mul_ps(w, other.z);
    __m256 q1x_q2y = _mm256_mul_ps(x, other.y);
    __m256 q1y_q2x = _mm256_mul_ps(y, other.x);

    __m256 q1w_q2w = _mm256_mul_ps(w, other.w);
    __m256 q1x_q2x = _mm256_mul_ps(x, other.x);
    __m256 q1y_q2y = _mm256_mul_ps(y, other.y);
    __m256 q1z_q2z = _mm256_mul_ps(z, other.z);

    // Calculate components with proper signs
    __m256 temp1, temp2;

    temp1 = _mm256_add_ps(q1x_q2w, q1w_q2x);
    temp2 = _mm256_sub_ps(q1y_q2z, q1z_q2y);
    result.x = _mm256_add_ps(temp1, temp2);

    temp1 = _mm256_add_ps(q1y_q2w, q1w_q2y);
    temp2 = _mm256_sub_ps(q1z_q2x, q1x_q2z);
    result.y = _mm256_add_ps(temp1, temp2);

    temp1 = _mm256_add_ps(q1z_q2w, q1w_q2z);
    temp2 = _mm256_sub_ps(q1x_q2y, q1y_q2x);
    result.z = _mm256_add_ps(temp1, temp2);

    temp1 = _mm256_sub_ps(q1w_q2w, q1x_q2x);
    temp2 = _mm256_sub_ps(temp1, q1y_q2y);
    result.w = _mm256_sub_ps(temp2, q1z_q2z);

    return result;
}

SimdVec3 SimdQuat::rotate(const SimdVec3& v) const {
    // Optimized quaternion-vector rotation
    // v' = q * v * q^-1
    // Optimized formula: v' = v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v)

    // Pure vector part of quaternion
    SimdVec3 q_vec;
    q_vec.x = x;
    q_vec.y = y;
    q_vec.z = z;

    // Cross product of q_vec and v
    SimdVec3 temp = q_vec.cross(v);

    // q.w * v
    SimdVec3 w_mul_v;
    w_mul_v.x = _mm256_mul_ps(w, v.x);
    w_mul_v.y = _mm256_mul_ps(w, v.y);
    w_mul_v.z = _mm256_mul_ps(w, v.z);

    // cross(q.xyz, v) + q.w * v
    SimdVec3 temp2;
    temp2.x = _mm256_add_ps(temp.x, w_mul_v.x);
    temp2.y = _mm256_add_ps(temp.y, w_mul_v.y);
    temp2.z = _mm256_add_ps(temp.z, w_mul_v.z);

    // cross(q.xyz, result of previous)
    SimdVec3 temp3 = q_vec.cross(temp2);

    // Scale by 2
    __m256 two = _mm256_set1_ps(2.0f);
    temp3.x = _mm256_mul_ps(temp3.x, two);
    temp3.y = _mm256_mul_ps(temp3.y, two);
    temp3.z = _mm256_mul_ps(temp3.z, two);

    // Add original vector
    SimdVec3 result;
    result.x = _mm256_add_ps(v.x, temp3.x);
    result.y = _mm256_add_ps(v.y, temp3.y);
    result.z = _mm256_add_ps(v.z, temp3.z);

    return result;
}

#elif defined(USE_SSE)
SimdQuat SimdQuat::load(const float* x_ptr, const float* y_ptr, const float* z_ptr, const float* w_ptr) {
    SimdQuat result;
    result.x = _mm_loadu_ps(x_ptr);
    result.y = _mm_loadu_ps(y_ptr);
    result.z = _mm_loadu_ps(z_ptr);
    result.w = _mm_loadu_ps(w_ptr);
    return result;
}

void SimdQuat::store(float* x_ptr, float* y_ptr, float* z_ptr, float* w_ptr) const {
    _mm_storeu_ps(x_ptr, x);
    _mm_storeu_ps(y_ptr, y);
    _mm_storeu_ps(z_ptr, z);
    _mm_storeu_ps(w_ptr, w);
}

SimdQuat SimdQuat::mul(const SimdQuat& other) const {
    SimdQuat result;

    // q1 * q2 implementation with SIMD
    __m128 q1x_q2w = _mm_mul_ps(x, other.w);
    __m128 q1y_q2z = _mm_mul_ps(y, other.z);
    __m128 q1z_q2y = _mm_mul_ps(z, other.y);
    __m128 q1w_q2x = _mm_mul_ps(w, other.x);

    __m128 q1y_q2w = _mm_mul_ps(y, other.w);
    __m128 q1z_q2x = _mm_mul_ps(z, other.x);
    __m128 q1w_q2y = _mm_mul_ps(w, other.y);
    __m128 q1x_q2z = _mm_mul_ps(x, other.z);

    __m128 q1z_q2w = _mm_mul_ps(z, other.w);
    __m128 q1w_q2z = _mm_mul_ps(w, other.z);
    __m128 q1x_q2y = _mm_mul_ps(x, other.y);
    __m128 q1y_q2x = _mm_mul_ps(y, other.x);

    __m128 q1w_q2w = _mm_mul_ps(w, other.w);
    __m128 q1x_q2x = _mm_mul_ps(x, other.x);
    __m128 q1y_q2y = _mm_mul_ps(y, other.y);
    __m128 q1z_q2z = _mm_mul_ps(z, other.z);

    // Calculate components with proper signs
    __m128 temp1, temp2;

    temp1 = _mm_add_ps(q1x_q2w, q1w_q2x);
    temp2 = _mm_sub_ps(q1y_q2z, q1z_q2y);
    result.x = _mm_add_ps(temp1, temp2);

    temp1 = _mm_add_ps(q1y_q2w, q1w_q2y);
    temp2 = _mm_sub_ps(q1z_q2x, q1x_q2z);
    result.y = _mm_add_ps(temp1, temp2);

    temp1 = _mm_add_ps(q1z_q2w, q1w_q2z);
    temp2 = _mm_sub_ps(q1x_q2y, q1y_q2x);
    result.z = _mm_add_ps(temp1, temp2);

    temp1 = _mm_sub_ps(q1w_q2w, q1x_q2x);
    temp2 = _mm_sub_ps(temp1, q1y_q2y);
    result.w = _mm_sub_ps(temp2, q1z_q2z);

    return result;
}

SimdVec3 SimdQuat::rotate(const SimdVec3& v) const {
    // Optimized quaternion-vector rotation
    // v' = q * v * q^-1
    // Optimized formula: v' = v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v)

    // Pure vector part of quaternion
    SimdVec3 q_vec;
    q_vec.x = x;
    q_vec.y = y;
    q_vec.z = z;

    // Cross product of q_vec and v
    SimdVec3 temp = q_vec.cross(v);

    // q.w * v
    SimdVec3 w_mul_v;
    w_mul_v.x = _mm_mul_ps(w, v.x);
    w_mul_v.y = _mm_mul_ps(w, v.y);
    w_mul_v.z = _mm_mul_ps(w, v.z);

    // cross(q.xyz, v) + q.w * v
    SimdVec3 temp2;
    temp2.x = _mm_add_ps(temp.x, w_mul_v.x);
    temp2.y = _mm_add_ps(temp.y, w_mul_v.y);
    temp2.z = _mm_add_ps(temp.z, w_mul_v.z);

    // cross(q.xyz, result of previous)
    SimdVec3 temp3 = q_vec.cross(temp2);

    // Scale by 2
    __m128 two = _mm_set1_ps(2.0f);
    temp3.x = _mm_mul_ps(temp3.x, two);
    temp3.y = _mm_mul_ps(temp3.y, two);
    temp3.z = _mm_mul_ps(temp3.z, two);

    // Add original vector
    SimdVec3 result;
    result.x = _mm_add_ps(v.x, temp3.x);
    result.y = _mm_add_ps(v.y, temp3.y);
    result.z = _mm_add_ps(v.z, temp3.z);

    return result;
}

#else
// Scalar fallback implementation for SimdQuat
SimdQuat SimdQuat::load(const float* x_ptr, const float* y_ptr, const float* z_ptr, const float* w_ptr) {
    SimdQuat result;
    for (int i = 0; i < SIMD_WIDTH; i++) {
        result.x[i] = x_ptr[i];
        result.y[i] = y_ptr[i];
        result.z[i] = z_ptr[i];
        result.w[i] = w_ptr[i];
    }
    return result;
}

void SimdQuat::store(float* x_ptr, float* y_ptr, float* z_ptr, float* w_ptr) const {
    for (int i = 0; i < SIMD_WIDTH; i++) {
        x_ptr[i] = x[i];
        y_ptr[i] = y[i];
        z_ptr[i] = z[i];
        w_ptr[i] = w[i];
    }
}

SimdQuat SimdQuat::mul(const SimdQuat& other) const {
    SimdQuat result;
    for (int i = 0; i < SIMD_WIDTH; i++) {
        // Quaternion multiplication: q1 * q2
        result.x[i] = x[i] * other.w[i] + w[i] * other.x[i] + y[i] * other.z[i] - z[i] * other.y[i];
        result.y[i] = y[i] * other.w[i] + w[i] * other.y[i] + z[i] * other.x[i] - x[i] * other.z[i];
        result.z[i] = z[i] * other.w[i] + w[i] * other.z[i] + x[i] * other.y[i] - y[i] * other.x[i];
        result.w[i] = w[i] * other.w[i] - x[i] * other.x[i] - y[i] * other.y[i] - z[i] * other.z[i];
    }
    return result;
}

SimdVec3 SimdQuat::rotate(const SimdVec3& v) const {
    SimdVec3 result;
    for (int i = 0; i < SIMD_WIDTH; i++) {
        // Optimized quaternion-vector rotation: v' = q * v * q^-1
        // Optimized formula: v' = v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v)

        // Create temporary vectors for calculations
        float qvec_x = x[i];
        float qvec_y = y[i];
        float qvec_z = z[i];
        float qvec_w = w[i];

        // q.xyz cross v
        float temp_x = qvec_y * v.z[i] - qvec_z * v.y[i];
        float temp_y = qvec_z * v.x[i] - qvec_x * v.z[i];
        float temp_z = qvec_x * v.y[i] - qvec_y * v.x[i];

        // q.w * v
        float w_mul_v_x = qvec_w * v.x[i];
        float w_mul_v_y = qvec_w * v.y[i];
        float w_mul_v_z = qvec_w * v.z[i];

        // cross(q.xyz, v) + q.w * v
        float temp2_x = temp_x + w_mul_v_x;
        float temp2_y = temp_y + w_mul_v_y;
        float temp2_z = temp_z + w_mul_v_z;

        // cross(q.xyz, result of previous)
        float temp3_x = qvec_y * temp2_z - qvec_z * temp2_y;
        float temp3_y = qvec_z * temp2_x - qvec_x * temp2_z;
        float temp3_z = qvec_x * temp2_y - qvec_y * temp2_x;

        // Scale by 2 and add original vector
        result.x[i] = v.x[i] + 2.0f * temp3_x;
        result.y[i] = v.y[i] + 2.0f * temp3_y;
        result.z[i] = v.z[i] + 2.0f * temp3_z;
    }
    return result;
}
#endif

// SimdMat4 implementation
#if defined(USE_AVX512)
SimdMat4 SimdMat4::load(const glm::mat4* matrices, uint32_t count) {
    SimdMat4 result;

    // Temporary storage for reorganizing data from AoS to SoA
    float row_data[4][16]; // 4 rows, 16 floats per row for AVX512

    // Extract data from glm matrices into row_data
    for (uint32_t i = 0; i < count; i++) {
        const glm::mat4& mat = matrices[i];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                row_data[row][i + col * (SIMD_WIDTH / 4)] = mat[col][row]; // Note: glm is column-major
            }
        }
    }

    // Load data into SIMD registers
    for (int row = 0; row < 4; row++) {
        result.rows[row] = _mm512_loadu_ps(row_data[row]);
    }

    return result;
}

void SimdMat4::store(glm::mat4* matrices, uint32_t count) const {
    // Temporary storage for reorganizing data from SoA to AoS
    float row_data[4][16]; // 4 rows, 16 floats per row for AVX512

    // Store data from SIMD registers into row_data
    for (int row = 0; row < 4; row++) {
        _mm512_storeu_ps(row_data[row], rows[row]);
    }

    // Extract data from row_data into glm matrices
    for (uint32_t i = 0; i < count; i++) {
        glm::mat4& mat = matrices[i];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                mat[col][row] = row_data[row][i + col * (SIMD_WIDTH / 4)]; // Note: glm is column-major
            }
        }
    }
}

SimdMat4 SimdMat4::mul(const SimdMat4& other) const {
    SimdMat4 result;

    for (int i = 0; i < 4; i++) {
        // For each row of the result
        result.rows[i] = _mm512_mul_ps(_mm512_permute_ps(rows[i], 0x00), _mm512_permute_ps(other.rows[0], 0x00));
        result.rows[i] = _mm512_add_ps(result.rows[i],
            _mm512_mul_ps(_mm512_permute_ps(rows[i], 0x55), _mm512_permute_ps(other.rows[1], 0x00)));
        result.rows[i] = _mm512_add_ps(result.rows[i],
            _mm512_mul_ps(_mm512_permute_ps(rows[i], 0xAA), _mm512_permute_ps(other.rows[2], 0x00)));
        result.rows[i] = _mm512_add_ps(result.rows[i],
            _mm512_mul_ps(_mm512_permute_ps(rows[i], 0xFF), _mm512_permute_ps(other.rows[3], 0x00)));
    }

    return result;
}

SimdVec3 SimdMat4::transformPoint(const SimdVec3& point) const {
    SimdVec3 result;

    // For each x component: rows[0].x * point.x + rows[0].y * point.y + rows[0].z * point.z + rows[0].w
    result.x = _mm512_mul_ps(_mm512_permute_ps(rows[0], 0x00), point.x);
    result.x = _mm512_add_ps(result.x, _mm512_mul_ps(_mm512_permute_ps(rows[0], 0x55), point.y));
    result.x = _mm512_add_ps(result.x, _mm512_mul_ps(_mm512_permute_ps(rows[0], 0xAA), point.z));
    result.x = _mm512_add_ps(result.x, _mm512_permute_ps(rows[0], 0xFF));

    // For each y component: rows[1].x * point.x + rows[1].y * point.y + rows[1].z * point.z + rows[1].w
    result.y = _mm512_mul_ps(_mm512_permute_ps(rows[1], 0x00), point.x);
    result.y = _mm512_add_ps(result.y, _mm512_mul_ps(_mm512_permute_ps(rows[1], 0x55), point.y));
    result.y = _mm512_add_ps(result.y, _mm512_mul_ps(_mm512_permute_ps(rows[1], 0xAA), point.z));
    result.y = _mm512_add_ps(result.y, _mm512_permute_ps(rows[1], 0xFF));

    // For each z component: rows[2].x * point.x + rows[2].y * point.y + rows[2].z * point.z + rows[2].w
    result.z = _mm512_mul_ps(_mm512_permute_ps(rows[2], 0x00), point.x);
    result.z = _mm512_add_ps(result.z, _mm512_mul_ps(_mm512_permute_ps(rows[2], 0x55), point.y));
    result.z = _mm512_add_ps(result.z, _mm512_mul_ps(_mm512_permute_ps(rows[2], 0xAA), point.z));
    result.z = _mm512_add_ps(result.z, _mm512_permute_ps(rows[2], 0xFF));

    return result;
}

SimdMat4 SimdMat4::createTransform(const SimdVec3& position, const SimdQuat& rotation, const SimdVec3& scale) {
    SimdMat4 result;

    // Convert quaternion to rotation matrix
    __m512 xx = _mm512_mul_ps(rotation.x, rotation.x);
    __m512 xy = _mm512_mul_ps(rotation.x, rotation.y);
    __m512 xz = _mm512_mul_ps(rotation.x, rotation.z);
    __m512 xw = _mm512_mul_ps(rotation.x, rotation.w);

    __m512 yy = _mm512_mul_ps(rotation.y, rotation.y);
    __m512 yz = _mm512_mul_ps(rotation.y, rotation.z);
    __m512 yw = _mm512_mul_ps(rotation.y, rotation.w);

    __m512 zz = _mm512_mul_ps(rotation.z, rotation.z);
    __m512 zw = _mm512_mul_ps(rotation.z, rotation.w);

    __m512 one = _mm512_set1_ps(1.0f);
    __m512 two = _mm512_set1_ps(2.0f);

    // Calculate rotation matrix elements
    __m512 m00 = _mm512_sub_ps(one, _mm512_mul_ps(two, _mm512_add_ps(yy, zz)));
    __m512 m01 = _mm512_mul_ps(two, _mm512_sub_ps(xy, zw));
    __m512 m02 = _mm512_mul_ps(two, _mm512_add_ps(xz, yw));

    __m512 m10 = _mm512_mul_ps(two, _mm512_add_ps(xy, zw));
    __m512 m11 = _mm512_sub_ps(one, _mm512_mul_ps(two, _mm512_add_ps(xx, zz)));
    __m512 m12 = _mm512_mul_ps(two, _mm512_sub_ps(yz, xw));

    __m512 m20 = _mm512_mul_ps(two, _mm512_sub_ps(xz, yw));
    __m512 m21 = _mm512_mul_ps(two, _mm512_add_ps(yz, xw));
    __m512 m22 = _mm512_sub_ps(one, _mm512_mul_ps(two, _mm512_add_ps(xx, yy)));

    // Apply scaling
    m00 = _mm512_mul_ps(m00, scale.x);
    m01 = _mm512_mul_ps(m01, scale.y);
    m02 = _mm512_mul_ps(m02, scale.z);

    m10 = _mm512_mul_ps(m10, scale.x);
    m11 = _mm512_mul_ps(m11, scale.y);
    m12 = _mm512_mul_ps(m12, scale.z);

    m20 = _mm512_mul_ps(m20, scale.x);
    m21 = _mm512_mul_ps(m21, scale.y);
    m22 = _mm512_mul_ps(m22, scale.z);

    // Create transform matrix
    // First row: (m00, m01, m02, position.x)
    result.rows[0] = _mm512_blend_ps(
        _mm512_blend_ps(
            _mm512_blend_ps(m00, m01, 0xAAAA),
            m02, 0xCCCC),
        position.x, 0xF000);

    // Second row: (m10, m11, m12, position.y)
    result.rows[1] = _mm512_blend_ps(
        _mm512_blend_ps(
            _mm512_blend_ps(m10, m11, 0xAAAA),
            m12, 0xCCCC),
        position.y, 0xF000);

    // Third row: (m20, m21, m22, position.z)
    result.rows[2] = _mm512_blend_ps(
        _mm512_blend_ps(
            _mm512_blend_ps(m20, m21, 0xAAAA),
            m22, 0xCCCC),
        position.z, 0xF000);

    // Fourth row: (0, 0, 0, 1)
    __m512 zero = _mm512_setzero_ps();
    result.rows[3] = _mm512_blend_ps(
        _mm512_blend_ps(
            _mm512_blend_ps(zero, zero, 0xAAAA),
            zero, 0xCCCC),
        one, 0xF000);

    return result;
}

#elif defined(USE_AVX) || defined(USE_AVX2)
SimdMat4 SimdMat4::load(const glm::mat4* matrices, uint32_t count) {
    SimdMat4 result;

    // Temporary storage for reorganizing data from AoS to SoA
    float row_data[4][4][8]; // 4 rows, 4 columns, 8 floats per register for AVX

    // Extract data from glm matrices into row_data
    for (uint32_t i = 0; i < count; i++) {
        const glm::mat4& mat = matrices[i];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                // Distribute values across the registers
                row_data[row][col][i] = mat[col][row]; // Note: glm is column-major
            }
        }
    }

    // Load data into SIMD registers
    for (int row = 0; row < 4; row++) {
        for (int reg = 0; reg < 2; reg++) { // Two registers per row for 8 floats
            result.rows[row][reg] = _mm256_loadu_ps(&row_data[row][reg * 2][0]);
        }
    }

    return result;
}

void SimdMat4::store(glm::mat4* matrices, uint32_t count) const {
    // Temporary storage for reorganizing data from SoA to AoS
    float row_data[4][4][8]; // 4 rows, 4 columns, 8 floats per register for AVX

    // Store data from SIMD registers into row_data
    for (int row = 0; row < 4; row++) {
        for (int reg = 0; reg < 2; reg++) { // Two registers per row for 8 floats
            _mm256_storeu_ps(&row_data[row][reg * 2][0], rows[row][reg]);
        }
    }

    // Extract data from row_data into glm matrices
    for (uint32_t i = 0; i < count; i++) {
        glm::mat4& mat = matrices[i];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                mat[col][row] = row_data[row][col][i]; // Note: glm is column-major
            }
        }
    }
}

SimdMat4 SimdMat4::mul(const SimdMat4& other) const {
    SimdMat4 result;

    for (int i = 0; i < 4; i++) {
        // For each row of the result
        for (int j = 0; j < 2; j++) {
            // Process the two 256-bit chunks
            // Multiply and add the 4 columns of the first matrix with the corresponding row of the second matrix
            __m256 sum = _mm256_mul_ps(_mm256_set1_ps(*((float*)&rows[i][0] + 0)), other.rows[0][j]);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_set1_ps(*((float*)&rows[i][0] + 1)), other.rows[1][j]));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_set1_ps(*((float*)&rows[i][0] + 2)), other.rows[2][j]));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_set1_ps(*((float*)&rows[i][0] + 3)), other.rows[3][j]));
            result.rows[i][j] = sum;
        }
    }

    return result;
}

SimdVec3 SimdMat4::transformPoint(const SimdVec3& point) const {
    SimdVec3 result;

    // For each component (x, y, z), multiply and add
    // Process x component
    result.x = _mm256_mul_ps(_mm256_broadcast_ss((float*)&rows[0][0]), point.x);
    result.x = _mm256_add_ps(result.x, _mm256_mul_ps(_mm256_broadcast_ss((float*)&rows[0][0] + 1), point.y));
    result.x = _mm256_add_ps(result.x, _mm256_mul_ps(_mm256_broadcast_ss((float*)&rows[0][0] + 2), point.z));
    result.x = _mm256_add_ps(result.x, _mm256_broadcast_ss((float*)&rows[0][0] + 3)); // Add translation component

    // Process y component
    result.y = _mm256_mul_ps(_mm256_broadcast_ss((float*)&rows[1][0]), point.x);
    result.y = _mm256_add_ps(result.y, _mm256_mul_ps(_mm256_broadcast_ss((float*)&rows[1][0] + 1), point.y));
    result.y = _mm256_add_ps(result.y, _mm256_mul_ps(_mm256_broadcast_ss((float*)&rows[1][0] + 2), point.z));
    result.y = _mm256_add_ps(result.y, _mm256_broadcast_ss((float*)&rows[1][0] + 3)); // Add translation component

    // Process z component
    result.z = _mm256_mul_ps(_mm256_broadcast_ss((float*)&rows[2][0]), point.x);
    result.z = _mm256_add_ps(result.z, _mm256_mul_ps(_mm256_broadcast_ss((float*)&rows[2][0] + 1), point.y));
    result.z = _mm256_add_ps(result.z, _mm256_mul_ps(_mm256_broadcast_ss((float*)&rows[2][0] + 2), point.z));
    result.z = _mm256_add_ps(result.z, _mm256_broadcast_ss((float*)&rows[2][0] + 3)); // Add translation component

    return result;
}

SimdMat4 SimdMat4::createTransform(const SimdVec3& position, const SimdQuat& rotation, const SimdVec3& scale) {
    SimdMat4 result;

    // Calculate rotation matrix components from quaternion
    __m256 xx = _mm256_mul_ps(rotation.x, rotation.x);
    __m256 xy = _mm256_mul_ps(rotation.x, rotation.y);
    __m256 xz = _mm256_mul_ps(rotation.x, rotation.z);
    __m256 xw = _mm256_mul_ps(rotation.x, rotation.w);

    __m256 yy = _mm256_mul_ps(rotation.y, rotation.y);
    __m256 yz = _mm256_mul_ps(rotation.y, rotation.z);
    __m256 yw = _mm256_mul_ps(rotation.y, rotation.w);

    __m256 zz = _mm256_mul_ps(rotation.z, rotation.z);
    __m256 zw = _mm256_mul_ps(rotation.z, rotation.w);

    __m256 one = _mm256_set1_ps(1.0f);
    __m256 two = _mm256_set1_ps(2.0f);

    // Compute rotation matrix elements
    __m256 m00 = _mm256_sub_ps(one, _mm256_mul_ps(two, _mm256_add_ps(yy, zz)));
    __m256 m01 = _mm256_mul_ps(two, _mm256_sub_ps(xy, zw));
    __m256 m02 = _mm256_mul_ps(two, _mm256_add_ps(xz, yw));

    __m256 m10 = _mm256_mul_ps(two, _mm256_add_ps(xy, zw));
    __m256 m11 = _mm256_sub_ps(one, _mm256_mul_ps(two, _mm256_add_ps(xx, zz)));
    __m256 m12 = _mm256_mul_ps(two, _mm256_sub_ps(yz, xw));

    __m256 m20 = _mm256_mul_ps(two, _mm256_sub_ps(xz, yw));
    __m256 m21 = _mm256_mul_ps(two, _mm256_add_ps(yz, xw));
    __m256 m22 = _mm256_sub_ps(one, _mm256_mul_ps(two, _mm256_add_ps(xx, yy)));

    // Apply scaling
    m00 = _mm256_mul_ps(m00, scale.x);
    m01 = _mm256_mul_ps(m01, scale.y);
    m02 = _mm256_mul_ps(m02, scale.z);

    m10 = _mm256_mul_ps(m10, scale.x);
    m11 = _mm256_mul_ps(m11, scale.y);
    m12 = _mm256_mul_ps(m12, scale.z);

    m20 = _mm256_mul_ps(m20, scale.x);
    m21 = _mm256_mul_ps(m21, scale.y);
    m22 = _mm256_mul_ps(m22, scale.z);

    // Zero vector for identity row
    __m256 zero = _mm256_setzero_ps();

    // Build the first row (m00, m01, m02, position.x)
    result.rows[0][0] = _mm256_set_ps(m01[3], m00[3], m01[2], m00[2], m01[1], m00[1], m01[0], m00[0]);
    result.rows[0][1] = _mm256_set_ps(position.x[3], m02[3], position.x[2], m02[2], position.x[1], m02[1], position.x[0], m02[0]);

    // Build the second row (m10, m11, m12, position.y)
    result.rows[1][0] = _mm256_set_ps(m11[3], m10[3], m11[2], m10[2], m11[1], m10[1], m11[0], m10[0]);
    result.rows[1][1] = _mm256_set_ps(position.y[3], m12[3], position.y[2], m12[2], position.y[1], m12[1], position.y[0], m12[0]);

    // Build the third row (m20, m21, m22, position.z)
    result.rows[2][0] = _mm256_set_ps(m21[3], m20[3], m21[2], m20[2], m21[1], m20[1], m21[0], m20[0]);
    result.rows[2][1] = _mm256_set_ps(position.z[3], m22[3], position.z[2], m22[2], position.z[1], m22[1], position.z[0], m22[0]);

    // Build the fourth row (0, 0, 0, 1)
    result.rows[3][0] = zero;
    result.rows[3][1] = _mm256_set_ps(one[3], zero[3], one[2], zero[2], one[1], zero[1], one[0], zero[0]);

    return result;
}

#elif defined(USE_SSE)
SimdMat4 SimdMat4::load(const glm::mat4* matrices, uint32_t count) {
    SimdMat4 result;

    // Temporary storage for reorganizing data from AoS to SoA
    float row_data[4][4][4]; // 4 rows, 4 columns, 4 floats per register for SSE

    // Extract data from glm matrices into row_data
    for (uint32_t i = 0; i < count; i++) {
        const glm::mat4& mat = matrices[i];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                row_data[row][col][i] = mat[col][row]; // Note: glm is column-major
            }
        }
    }

    // Load data into SIMD registers
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            result.rows[row][col] = _mm_loadu_ps(row_data[row][col]);
        }
    }

    return result;
}

void SimdMat4::store(glm::mat4* matrices, uint32_t count) const {
    // Temporary storage for reorganizing data from SoA to AoS
    float row_data[4][4][4]; // 4 rows, 4 columns, 4 floats per register for SSE

    // Store data from SIMD registers into row_data
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            _mm_storeu_ps(row_data[row][col], rows[row][col]);
        }
    }

    // Extract data from row_data into glm matrices
    for (uint32_t i = 0; i < count; i++) {
        glm::mat4& mat = matrices[i];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                mat[col][row] = row_data[row][col][i]; // Note: glm is column-major
            }
        }
    }
}

SimdMat4 SimdMat4::mul(const SimdMat4& other) const {
    SimdMat4 result;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            // For each element of the result matrix
            // Sum the products of this.rows[i][k] * other.rows[k][j] for all k
            __m128 sum = _mm_mul_ps(_mm_shuffle_ps(rows[i][0], rows[i][0], 0x00), other.rows[0][j]);
            sum = _mm_add_ps(sum, _mm_mul_ps(_mm_shuffle_ps(rows[i][1], rows[i][1], 0x00), other.rows[1][j]));
            sum = _mm_add_ps(sum, _mm_mul_ps(_mm_shuffle_ps(rows[i][2], rows[i][2], 0x00), other.rows[2][j]));
            sum = _mm_add_ps(sum, _mm_mul_ps(_mm_shuffle_ps(rows[i][3], rows[i][3], 0x00), other.rows[3][j]));
            result.rows[i][j] = sum;
        }
    }

    return result;
}

SimdVec3 SimdMat4::transformPoint(const SimdVec3& point) const {
    SimdVec3 result;

    // For x component: rows[0][0] * point.x + rows[0][1] * point.y + rows[0][2] * point.z + rows[0][3]
    result.x = _mm_mul_ps(rows[0][0], point.x);
    result.x = _mm_add_ps(result.x, _mm_mul_ps(rows[0][1], point.y));
    result.x = _mm_add_ps(result.x, _mm_mul_ps(rows[0][2], point.z));
    result.x = _mm_add_ps(result.x, rows[0][3]);

    // For y component: rows[1][0] * point.x + rows[1][1] * point.y + rows[1][2] * point.z + rows[1][3]
    result.y = _mm_mul_ps(rows[1][0], point.x);
    result.y = _mm_add_ps(result.y, _mm_mul_ps(rows[1][1], point.y));
    result.y = _mm_add_ps(result.y, _mm_mul_ps(rows[1][2], point.z));
    result.y = _mm_add_ps(result.y, rows[1][3]);

    // For z component: rows[2][0] * point.x + rows[2][1] * point.y + rows[2][2] * point.z + rows[2][3]
    result.z = _mm_mul_ps(rows[2][0], point.x);
    result.z = _mm_add_ps(result.z, _mm_mul_ps(rows[2][1], point.y));
    result.z = _mm_add_ps(result.z, _mm_mul_ps(rows[2][2], point.z));
    result.z = _mm_add_ps(result.z, rows[2][3]);

    return result;
}

SimdMat4 SimdMat4::createTransform(const SimdVec3& position, const SimdQuat& rotation, const SimdVec3& scale) {
    SimdMat4 result;

    // Calculate rotation matrix components from quaternion
    __m128 xx = _mm_mul_ps(rotation.x, rotation.x);
    __m128 xy = _mm_mul_ps(rotation.x, rotation.y);
    __m128 xz = _mm_mul_ps(rotation.x, rotation.z);
    __m128 xw = _mm_mul_ps(rotation.x, rotation.w);

    __m128 yy = _mm_mul_ps(rotation.y, rotation.y);
    __m128 yz = _mm_mul_ps(rotation.y, rotation.z);
    __m128 yw = _mm_mul_ps(rotation.y, rotation.w);

    __m128 zz = _mm_mul_ps(rotation.z, rotation.z);
    __m128 zw = _mm_mul_ps(rotation.z, rotation.w);

    __m128 one = _mm_set1_ps(1.0f);
    __m128 two = _mm_set1_ps(2.0f);

    // Compute rotation matrix elements
    __m128 m00 = _mm_sub_ps(one, _mm_mul_ps(two, _mm_add_ps(yy, zz)));
    __m128 m01 = _mm_mul_ps(two, _mm_sub_ps(xy, zw));
    __m128 m02 = _mm_mul_ps(two, _mm_add_ps(xz, yw));

    __m128 m10 = _mm_mul_ps(two, _mm_add_ps(xy, zw));
    __m128 m11 = _mm_sub_ps(one, _mm_mul_ps(two, _mm_add_ps(xx, zz)));
    __m128 m12 = _mm_mul_ps(two, _mm_sub_ps(yz, xw));

    __m128 m20 = _mm_mul_ps(two, _mm_sub_ps(xz, yw));
    __m128 m21 = _mm_mul_ps(two, _mm_add_ps(yz, xw));
    __m128 m22 = _mm_sub_ps(one, _mm_mul_ps(two, _mm_add_ps(xx, yy)));

    // Apply scaling
    m00 = _mm_mul_ps(m00, scale.x);
    m01 = _mm_mul_ps(m01, scale.y);
    m02 = _mm_mul_ps(m02, scale.z);

    m10 = _mm_mul_ps(m10, scale.x);
    m11 = _mm_mul_ps(m11, scale.y);
    m12 = _mm_mul_ps(m12, scale.z);

    m20 = _mm_mul_ps(m20, scale.x);
    m21 = _mm_mul_ps(m21, scale.y);
    m22 = _mm_mul_ps(m22, scale.z);

    // Zero vector for identity row
    __m128 zero = _mm_setzero_ps();

    // Build the matrix
    result.rows[0][0] = m00;
    result.rows[0][1] = m01;
    result.rows[0][2] = m02;
    result.rows[0][3] = position.x;

    result.rows[1][0] = m10;
    result.rows[1][1] = m11;
    result.rows[1][2] = m12;
    result.rows[1][3] = position.y;

    result.rows[2][0] = m20;
    result.rows[2][1] = m21;
    result.rows[2][2] = m22;
    result.rows[2][3] = position.z;

    result.rows[3][0] = zero;
    result.rows[3][1] = zero;
    result.rows[3][2] = zero;
    result.rows[3][3] = one;

    return result;
}

#else
// Scalar fallback implementation for SimdMat4
SimdMat4 SimdMat4::load(const glm::mat4* matrices, uint32_t count) {
    SimdMat4 result;

    for (uint32_t i = 0; i < count; i++) {
        const glm::mat4& mat = matrices[i];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                result.rows[row][col][i] = mat[col][row]; // Note: glm is column-major
            }
        }
    }

    return result;
}

void SimdMat4::store(glm::mat4* matrices, uint32_t count) const {
    for (uint32_t i = 0; i < count; i++) {
        glm::mat4& mat = matrices[i];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                mat[col][row] = rows[row][col][i]; // Note: glm is column-major
            }
        }
    }
}

SimdMat4 SimdMat4::mul(const SimdMat4& other) const {
    SimdMat4 result;

    for (int lane = 0; lane < SIMD_WIDTH; lane++) {
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                result.rows[row][col][lane] = 0.0f;
                for (int k = 0; k < 4; k++) {
                    result.rows[row][col][lane] += rows[row][k][lane] * other.rows[k][col][lane];
                }
            }
        }
    }

    return result;
}

SimdVec3 SimdMat4::transformPoint(const SimdVec3& point) const {
    SimdVec3 result;

    for (int i = 0; i < SIMD_WIDTH; i++) {
        result.x[i] = rows[0][0][i] * point.x[i] + rows[0][1][i] * point.y[i] + rows[0][2][i] * point.z[i] + rows[0][3][i];
        result.y[i] = rows[1][0][i] * point.x[i] + rows[1][1][i] * point.y[i] + rows[1][2][i] * point.z[i] + rows[1][3][i];
        result.z[i] = rows[2][0][i] * point.x[i] + rows[2][1][i] * point.y[i] + rows[2][2][i] * point.z[i] + rows[2][3][i];
    }

    return result;
}

SimdMat4 SimdMat4::createTransform(const SimdVec3& position, const SimdQuat& rotation, const SimdVec3& scale) {
    SimdMat4 result;

    for (int i = 0; i < SIMD_WIDTH; i++) {
        // Calculate rotation matrix components from quaternion
        float xx = rotation.x[i] * rotation.x[i];
        float xy = rotation.x[i] * rotation.y[i];
        float xz = rotation.x[i] * rotation.z[i];
        float xw = rotation.x[i] * rotation.w[i];

        float yy = rotation.y[i] * rotation.y[i];
        float yz = rotation.y[i] * rotation.z[i];
        float yw = rotation.y[i] * rotation.w[i];

        float zz = rotation.z[i] * rotation.z[i];
        float zw = rotation.z[i] * rotation.w[i];

        // Calculate rotation matrix elements
        float m00 = (1.0f - 2.0f * (yy + zz)) * scale.x[i];
        float m01 = (2.0f * (xy - zw)) * scale.y[i];
        float m02 = (2.0f * (xz + yw)) * scale.z[i];

        float m10 = (2.0f * (xy + zw)) * scale.x[i];
        float m11 = (1.0f - 2.0f * (xx + zz)) * scale.y[i];
        float m12 = (2.0f * (yz - xw)) * scale.z[i];

        float m20 = (2.0f * (xz - yw)) * scale.x[i];
        float m21 = (2.0f * (yz + xw)) * scale.y[i];
        float m22 = (1.0f - 2.0f * (xx + yy)) * scale.z[i];

        // Create transform matrix
        result.rows[0][0][i] = m00;
        result.rows[0][1][i] = m01;
        result.rows[0][2][i] = m02;
        result.rows[0][3][i] = position.x[i];

        result.rows[1][0][i] = m10;
        result.rows[1][1][i] = m11;
        result.rows[1][2][i] = m12;
        result.rows[1][3][i] = position.y[i];

        result.rows[2][0][i] = m20;
        result.rows[2][1][i] = m21;
        result.rows[2][2][i] = m22;
        result.rows[2][3][i] = position.z[i];

        result.rows[3][0][i] = 0.0f;
        result.rows[3][1][i] = 0.0f;
        result.rows[3][2][i] = 0.0f;
        result.rows[3][3][i] = 1.0f;
    }

    return result;
}
#endif

// SimdAABB implementation
#if defined(USE_AVX512)
SimdAABB SimdAABB::load(const AABB* boxes, uint32_t count) {
    SimdAABB result;

    // Temporary storage for reorganizing data
    float min_x_data[16], min_y_data[16], min_z_data[16];
    float max_x_data[16], max_y_data[16], max_z_data[16];

    // Extract data from AABB boxes
    for (uint32_t i = 0; i < count; i++) {
        const AABB& box = boxes[i];
        min_x_data[i] = box.min.x;
        min_y_data[i] = box.min.y;
        min_z_data[i] = box.min.z;

        max_x_data[i] = box.max.x;
        max_y_data[i] = box.max.y;
        max_z_data[i] = box.max.z;
    }

    // Load data into SIMD registers
    result.min_x = _mm512_loadu_ps(min_x_data);
    result.min_y = _mm512_loadu_ps(min_y_data);
    result.min_z = _mm512_loadu_ps(min_z_data);

    result.max_x = _mm512_loadu_ps(max_x_data);
    result.max_y = _mm512_loadu_ps(max_y_data);
    result.max_z = _mm512_loadu_ps(max_z_data);

    return result;
}

void SimdAABB::store(AABB* boxes, uint32_t count) const {
    // Temporary storage for reorganizing data
    float min_x_data[16], min_y_data[16], min_z_data[16];
    float max_x_data[16], max_y_data[16], max_z_data[16];

    // Store data from SIMD registers
    _mm512_storeu_ps(min_x_data, min_x);
    _mm512_storeu_ps(min_y_data, min_y);
    _mm512_storeu_ps(min_z_data, min_z);

    _mm512_storeu_ps(max_x_data, max_x);
    _mm512_storeu_ps(max_y_data, max_y);
    _mm512_storeu_ps(max_z_data, max_z);

    // Extract data into AABB boxes
    for (uint32_t i = 0; i < count; i++) {
        AABB& box = boxes[i];
        box.min.x = min_x_data[i];
        box.min.y = min_y_data[i];
        box.min.z = min_z_data[i];

        box.max.x = max_x_data[i];
        box.max.y = max_y_data[i];
        box.max.z = max_z_data[i];
    }
}

bool SimdAABB::intersect(const SimdAABB& other) const {
    // Check if this.min <= other.max and this.max >= other.min for all axes
    __mmask16 x_overlap = _mm512_cmp_ps_mask(min_x, other.max_x, _CMP_LE_OS) & _mm512_cmp_ps_mask(max_x, other.min_x, _CMP_GE_OS);
    __mmask16 y_overlap = _mm512_cmp_ps_mask(min_y, other.max_y, _CMP_LE_OS) & _mm512_cmp_ps_mask(max_y, other.min_y, _CMP_GE_OS);
    __mmask16 z_overlap = _mm512_cmp_ps_mask(min_z, other.max_z, _CMP_LE_OS) & _mm512_cmp_ps_mask(max_z, other.min_z, _CMP_GE_OS);

    // Combine results for all axes: boxes intersect if they overlap on all axes
    __mmask16 result = x_overlap & y_overlap & z_overlap;

    // Return true if any of the SIMD lanes has an intersection
    return result != 0;
}

void SimdAABB::transform(const SimdMat4& matrix) {
    // Create vectors for all 8 corners of the AABB
    SimdVec3 corners[8];

    // Original corners
    corners[0] = { min_x, min_y, min_z }; // Min corner
    corners[1] = { max_x, min_y, min_z }; // Max x, min y, min z
    corners[2] = { min_x, max_y, min_z }; // Min x, max y, min z
    corners[3] = { max_x, max_y, min_z }; // Max x, max y, min z
    corners[4] = { min_x, min_y, max_z }; // Min x, min y, max z
    corners[5] = { max_x, min_y, max_z }; // Max x, min y, max z
    corners[6] = { min_x, max_y, max_z }; // Min x, max y, max z
    corners[7] = { max_x, max_y, max_z }; // Max corner

    // Transform all corners
    for (int i = 0; i < 8; i++) {
        corners[i] = matrix.transformPoint(corners[i]);
    }

    // Initialize new bounds with the first corner
    min_x = corners[0].x;
    min_y = corners[0].y;
    min_z = corners[0].z;
    max_x = corners[0].x;
    max_y = corners[0].y;
    max_z = corners[0].z;

    // Expand bounds to include all other corners
    for (int i = 1; i < 8; i++) {
        min_x = _mm512_min_ps(min_x, corners[i].x);
        min_y = _mm512_min_ps(min_y, corners[i].y);
        min_z = _mm512_min_ps(min_z, corners[i].z);

        max_x = _mm512_max_ps(max_x, corners[i].x);
        max_y = _mm512_max_ps(max_y, corners[i].y);
        max_z = _mm512_max_ps(max_z, corners[i].z);
    }
}

#elif defined(USE_AVX) || defined(USE_AVX2)
SimdAABB SimdAABB::load(const AABB* boxes, uint32_t count) {
    SimdAABB result;

    // Temporary storage for reorganizing data
    float min_x_data[8], min_y_data[8], min_z_data[8];
    float max_x_data[8], max_y_data[8], max_z_data[8];

    // Extract data from AABB boxes
    for (uint32_t i = 0; i < count; i++) {
        const AABB& box = boxes[i];
        min_x_data[i] = box.min.x;
        min_y_data[i] = box.min.y;
        min_z_data[i] = box.min.z;

        max_x_data[i] = box.max.x;
        max_y_data[i] = box.max.y;
        max_z_data[i] = box.max.z;
    }

    // Load data into SIMD registers
    result.min_x = _mm256_loadu_ps(min_x_data);
    result.min_y = _mm256_loadu_ps(min_y_data);
    result.min_z = _mm256_loadu_ps(min_z_data);

    result.max_x = _mm256_loadu_ps(max_x_data);
    result.max_y = _mm256_loadu_ps(max_y_data);
    result.max_z = _mm256_loadu_ps(max_z_data);

    return result;
}

void SimdAABB::store(AABB* boxes, uint32_t count) const {
    // Temporary storage for reorganizing data
    float min_x_data[8], min_y_data[8], min_z_data[8];
    float max_x_data[8], max_y_data[8], max_z_data[8];

    // Store data from SIMD registers
    _mm256_storeu_ps(min_x_data, min_x);
    _mm256_storeu_ps(min_y_data, min_y);
    _mm256_storeu_ps(min_z_data, min_z);

    _mm256_storeu_ps(max_x_data, max_x);
    _mm256_storeu_ps(max_y_data, max_y);
    _mm256_storeu_ps(max_z_data, max_z);

    // Extract data into AABB boxes
    for (uint32_t i = 0; i < count; i++) {
        AABB& box = boxes[i];
        box.min.x = min_x_data[i];
        box.min.y = min_y_data[i];
        box.min.z = min_z_data[i];

        box.max.x = max_x_data[i];
        box.max.y = max_y_data[i];
        box.max.z = max_z_data[i];
    }
}

bool SimdAABB::intersect(const SimdAABB& other) const {
    // Check for overlap on each axis
    __m256 x_min_check = _mm256_cmp_ps(min_x, other.max_x, _CMP_LE_OS); // this.min.x <= other.max.x
    __m256 x_max_check = _mm256_cmp_ps(max_x, other.min_x, _CMP_GE_OS); // this.max.x >= other.min.x
    __m256 x_overlap = _mm256_and_ps(x_min_check, x_max_check);

    __m256 y_min_check = _mm256_cmp_ps(min_y, other.max_y, _CMP_LE_OS);
    __m256 y_max_check = _mm256_cmp_ps(max_y, other.min_y, _CMP_GE_OS);
    __m256 y_overlap = _mm256_and_ps(y_min_check, y_max_check);

    __m256 z_min_check = _mm256_cmp_ps(min_z, other.max_z, _CMP_LE_OS);
    __m256 z_max_check = _mm256_cmp_ps(max_z, other.min_z, _CMP_GE_OS);
    __m256 z_overlap = _mm256_and_ps(z_min_check, z_max_check);

    // Combine results: overlap on all axes
    __m256 result = _mm256_and_ps(x_overlap, _mm256_and_ps(y_overlap, z_overlap));

    // Check if any lane has an intersection
    return _mm256_movemask_ps(result) != 0;
}

void SimdAABB::transform(const SimdMat4& matrix) {
    // Create vectors for all 8 corners of the AABB
    SimdVec3 corners[8];

    // Original corners
    corners[0] = { min_x, min_y, min_z }; // Min corner
    corners[1] = { max_x, min_y, min_z }; // Max x, min y, min z
    corners[2] = { min_x, max_y, min_z }; // Min x, max y, min z
    corners[3] = { max_x, max_y, min_z }; // Max x, max y, min z
    corners[4] = { min_x, min_y, max_z }; // Min x, min y, max z
    corners[5] = { max_x, min_y, max_z }; // Max x, min y, max z
    corners[6] = { min_x, max_y, max_z }; // Min x, max y, max z
    corners[7] = { max_x, max_y, max_z }; // Max corner

    // Transform all corners
    for (int i = 0; i < 8; i++) {
        corners[i] = matrix.transformPoint(corners[i]);
    }

    // Initialize new bounds with the first corner
    min_x = corners[0].x;
    min_y = corners[0].y;
    min_z = corners[0].z;
    max_x = corners[0].x;
    max_y = corners[0].y;
    max_z = corners[0].z;

    // Expand bounds to include all other corners
    for (int i = 1; i < 8; i++) {
        min_x = _mm256_min_ps(min_x, corners[i].x);
        min_y = _mm256_min_ps(min_y, corners[i].y);
        min_z = _mm256_min_ps(min_z, corners[i].z);

        max_x = _mm256_max_ps(max_x, corners[i].x);
        max_y = _mm256_max_ps(max_y, corners[i].y);
        max_z = _mm256_max_ps(max_z, corners[i].z);
    }
}

#elif defined(USE_SSE)
SimdAABB SimdAABB::load(const AABB* boxes, uint32_t count) {
    SimdAABB result;

    // Temporary storage for reorganizing data
    float min_x_data[4], min_y_data[4], min_z_data[4];
    float max_x_data[4], max_y_data[4], max_z_data[4];

    // Extract data from AABB boxes
    for (uint32_t i = 0; i < count; i++) {
        const AABB& box = boxes[i];
        min_x_data[i] = box.min.x;
        min_y_data[i] = box.min.y;
        min_z_data[i] = box.min.z;

        max_x_data[i] = box.max.x;
        max_y_data[i] = box.max.y;
        max_z_data[i] = box.max.z;
    }

    // Load data into SIMD registers
    result.min_x = _mm_loadu_ps(min_x_data);
    result.min_y = _mm_loadu_ps(min_y_data);
    result.min_z = _mm_loadu_ps(min_z_data);

    result.max_x = _mm_loadu_ps(max_x_data);
    result.max_y = _mm_loadu_ps(max_y_data);
    result.max_z = _mm_loadu_ps(max_z_data);

    return result;
}

void SimdAABB::store(AABB* boxes, uint32_t count) const {
    // Temporary storage for reorganizing data
    float min_x_data[4], min_y_data[4], min_z_data[4];
    float max_x_data[4], max_y_data[4], max_z_data[4];

    // Store data from SIMD registers
    _mm_storeu_ps(min_x_data, min_x);
    _mm_storeu_ps(min_y_data, min_y);
    _mm_storeu_ps(min_z_data, min_z);

    _mm_storeu_ps(max_x_data, max_x);
    _mm_storeu_ps(max_y_data, max_y);
    _mm_storeu_ps(max_z_data, max_z);

    // Extract data into AABB boxes
    for (uint32_t i = 0; i < count; i++) {
        AABB& box = boxes[i];
        box.min.x = min_x_data[i];
        box.min.y = min_y_data[i];
        box.min.z = min_z_data[i];

        box.max.x = max_x_data[i];
        box.max.y = max_y_data[i];
        box.max.z = max_z_data[i];
    }
}

bool SimdAABB::intersect(const SimdAABB& other) const {
    // Check for overlap on each axis
    __m128 x_min_check = _mm_cmple_ps(min_x, other.max_x); // this.min.x <= other.max.x
    __m128 x_max_check = _mm_cmpge_ps(max_x, other.min_x); // this.max.x >= other.min.x
    __m128 x_overlap = _mm_and_ps(x_min_check, x_max_check);

    __m128 y_min_check = _mm_cmple_ps(min_y, other.max_y);
    __m128 y_max_check = _mm_cmpge_ps(max_y, other.min_y);
    __m128 y_overlap = _mm_and_ps(y_min_check, y_max_check);

    __m128 z_min_check = _mm_cmple_ps(min_z, other.max_z);
    __m128 z_max_check = _mm_cmpge_ps(max_z, other.min_z);
    __m128 z_overlap = _mm_and_ps(z_min_check, z_max_check);

    // Combine results: overlap on all axes
    __m128 result = _mm_and_ps(x_overlap, _mm_and_ps(y_overlap, z_overlap));

    // Check if any lane has an intersection
    return _mm_movemask_ps(result) != 0;
}

void SimdAABB::transform(const SimdMat4& matrix) {
    // Create vectors for all 8 corners of the AABB
    SimdVec3 corners[8];

    // Original corners
    corners[0] = { min_x, min_y, min_z }; // Min corner
    corners[1] = { max_x, min_y, min_z }; // Max x, min y, min z
    corners[2] = { min_x, max_y, min_z }; // Min x, max y, min z
    corners[3] = { max_x, max_y, min_z }; // Max x, max y, min z
    corners[4] = { min_x, min_y, max_z }; // Min x, min y, max z
    corners[5] = { max_x, min_y, max_z }; // Max x, min y, max z
    corners[6] = { min_x, max_y, max_z }; // Min x, max y, max z
    corners[7] = { max_x, max_y, max_z }; // Max corner

    // Transform all corners
    for (int i = 0; i < 8; i++) {
        corners[i] = matrix.transformPoint(corners[i]);
    }

    // Initialize new bounds with the first corner
    min_x = corners[0].x;
    min_y = corners[0].y;
    min_z = corners[0].z;
    max_x = corners[0].x;
    max_y = corners[0].y;
    max_z = corners[0].z;

    // Expand bounds to include all other corners
    for (int i = 1; i < 8; i++) {
        min_x = _mm_min_ps(min_x, corners[i].x);
        min_y = _mm_min_ps(min_y, corners[i].y);
        min_z = _mm_min_ps(min_z, corners[i].z);

        max_x = _mm_max_ps(max_x, corners[i].x);
        max_y = _mm_max_ps(max_y, corners[i].y);
        max_z = _mm_max_ps(max_z, corners[i].z);
    }
}

#else
// Scalar fallback implementation for SimdAABB
SimdAABB SimdAABB::load(const AABB* boxes, uint32_t count) {
    SimdAABB result;

    for (uint32_t i = 0; i < count; i++) {
        const AABB& box = boxes[i];
        result.min_x[i] = box.min.x;
        result.min_y[i] = box.min.y;
        result.min_z[i] = box.min.z;

        result.max_x[i] = box.max.x;
        result.max_y[i] = box.max.y;
        result.max_z[i] = box.max.z;
    }

    return result;
}

void SimdAABB::store(AABB* boxes, uint32_t count) const {
    for (uint32_t i = 0; i < count; i++) {
        AABB& box = boxes[i];
        box.min.x = min_x[i];
        box.min.y = min_y[i];
        box.min.z = min_z[i];

        box.max.x = max_x[i];
        box.max.y = max_y[i];
        box.max.z = max_z[i];
    }
}

bool SimdAABB::intersect(const SimdAABB& other) const {
    bool any_intersection = false;

    for (int i = 0; i < SIMD_WIDTH; i++) {
        // Check if the boxes overlap in all three dimensions
        bool x_overlap = min_x[i] <= other.max_x[i] && max_x[i] >= other.min_x[i];
        bool y_overlap = min_y[i] <= other.max_y[i] && max_y[i] >= other.min_y[i];
        bool z_overlap = min_z[i] <= other.max_z[i] && max_z[i] >= other.min_z[i];

        // Boxes intersect if they overlap in all dimensions
        if (x_overlap && y_overlap && z_overlap) {
            any_intersection = true;
            break;
        }
    }

    return any_intersection;
}

void SimdAABB::transform(const SimdMat4& matrix) {
    for (int lane = 0; lane < SIMD_WIDTH; lane++) {
        // Create temporary arrays for the 8 corners of the AABB
        float corners_x[8], corners_y[8], corners_z[8];

        // Calculate all 8 corners of the original AABB
        corners_x[0] = min_x[lane]; corners_y[0] = min_y[lane]; corners_z[0] = min_z[lane]; // Min corner
        corners_x[1] = max_x[lane]; corners_y[1] = min_y[lane]; corners_z[1] = min_z[lane]; // Max x, min y, min z
        corners_x[2] = min_x[lane]; corners_y[2] = max_y[lane]; corners_z[2] = min_z[lane]; // Min x, max y, min z
        corners_x[3] = max_x[lane]; corners_y[3] = max_y[lane]; corners_z[3] = min_z[lane]; // Max x, max y, min z
        corners_x[4] = min_x[lane]; corners_y[4] = min_y[lane]; corners_z[4] = max_z[lane]; // Min x, min y, max z
        corners_x[5] = max_x[lane]; corners_y[5] = min_y[lane]; corners_z[5] = max_z[lane]; // Max x, min y, max z
        corners_x[6] = min_x[lane]; corners_y[6] = max_y[lane]; corners_z[6] = max_z[lane]; // Min x, max y, max z
        corners_x[7] = max_x[lane]; corners_y[7] = max_y[lane]; corners_z[7] = max_z[lane]; // Max corner

        // Transform each corner
        float transformed_x[8], transformed_y[8], transformed_z[8];

        for (int corner = 0; corner < 8; corner++) {
            // Apply matrix transformation to the corner
            transformed_x[corner] = matrix.rows[0][0][lane] * corners_x[corner] +
                matrix.rows[0][1][lane] * corners_y[corner] +
                matrix.rows[0][2][lane] * corners_z[corner] +
                matrix.rows[0][3][lane];

            transformed_y[corner] = matrix.rows[1][0][lane] * corners_x[corner] +
                matrix.rows[1][1][lane] * corners_y[corner] +
                matrix.rows[1][2][lane] * corners_z[corner] +
                matrix.rows[1][3][lane];

            transformed_z[corner] = matrix.rows[2][0][lane] * corners_x[corner] +
                matrix.rows[2][1][lane] * corners_y[corner] +
                matrix.rows[2][2][lane] * corners_z[corner] +
                matrix.rows[2][3][lane];
        }

        // Find the new bounds of the transformed box
        min_x[lane] = transformed_x[0];
        min_y[lane] = transformed_y[0];
        min_z[lane] = transformed_z[0];

        max_x[lane] = transformed_x[0];
        max_y[lane] = transformed_y[0];
        max_z[lane] = transformed_z[0];

        // Find min/max of all corners
        for (int corner = 1; corner < 8; corner++) {
            min_x[lane] = std::min(min_x[lane], transformed_x[corner]);
            min_y[lane] = std::min(min_y[lane], transformed_y[corner]);
            min_z[lane] = std::min(min_z[lane], transformed_z[corner]);

            max_x[lane] = std::max(max_x[lane], transformed_x[corner]);
            max_y[lane] = std::max(max_y[lane], transformed_y[corner]);
            max_z[lane] = std::max(max_z[lane], transformed_z[corner]);
        }
    }
}
#endif