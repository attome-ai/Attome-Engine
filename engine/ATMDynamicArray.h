#ifndef ATM_DYNAMIC_ARRAY_H
#define ATM_DYNAMIC_ARRAY_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

/**
 * DynamicArray<T> - RAII wrapper for dynamic arrays.
 *
 * Provides automatic memory management (no manual delete[] needed).
 * The EntityContainer already tracks `count` - this wrapper only manages the
 * raw storage.
 *
 * Usage:
 *   DynamicArray<float> velocities(capacity);
 *   velocities[0] = 1.0f;
 *   velocities.resize(newCapacity, count);  // Preserves `count` elements
 *   velocities.resize(newCapacity);         // Preserves min(old, new) elements
 *
 * Indexing is bounds-checked with assert() in debug builds only; release
 * builds compile to a plain pointer access.
 */
template <typename T> class DynamicArray {
private:
  T *data_;
  int capacity_;

public:
  DynamicArray() noexcept : data_(nullptr), capacity_(0) {}

  // Construct with initial capacity
  explicit DynamicArray(int capacity)
      : data_(capacity > 0 ? new T[capacity] : nullptr),
        capacity_(capacity > 0 ? capacity : 0) {
    if (data_) {
      std::fill(data_, data_ + capacity_, T{});
    }
  }

  // Construct with initial capacity and default value
  DynamicArray(int capacity, const T &defaultValue)
      : data_(capacity > 0 ? new T[capacity] : nullptr),
        capacity_(capacity > 0 ? capacity : 0) {
    if (data_) {
      std::fill(data_, data_ + capacity_, defaultValue);
    }
  }

  // Destructor - RAII cleanup
  ~DynamicArray() { delete[] data_; }

  // Move constructor
  DynamicArray(DynamicArray &&other) noexcept
      : data_(other.data_), capacity_(other.capacity_) {
    other.data_ = nullptr;
    other.capacity_ = 0;
  }

  // Move assignment
  DynamicArray &operator=(DynamicArray &&other) noexcept {
    if (this != &other) {
      delete[] data_;
      data_ = other.data_;
      capacity_ = other.capacity_;
      other.data_ = nullptr;
      other.capacity_ = 0;
    }
    return *this;
  }

  // Disable copy (prevent double-free)
  DynamicArray(const DynamicArray &) = delete;
  DynamicArray &operator=(const DynamicArray &) = delete;

  // Array access
  T &operator[](int index) {
    assert(index >= 0 && index < capacity_);
    return data_[index];
  }
  const T &operator[](int index) const {
    assert(index >= 0 && index < capacity_);
    return data_[index];
  }

  // Get raw pointer (for compatibility with existing code)
  T *data() noexcept { return data_; }
  const T *data() const noexcept { return data_; }

  // Implicit conversion to T* for seamless integration
  operator T *() noexcept { return data_; }
  operator const T *() const noexcept { return data_; }

  int capacity() const noexcept { return capacity_; }

  // Resize array, preserving `count` elements and filling new elements with
  // default
  void resize(int newCapacity, int count) { resize(newCapacity, count, T{}); }

  // Resize with a specific default value for new elements
  void resize(int newCapacity, int count, const T &defaultValue) {
    newCapacity = std::max(newCapacity, 0);
    count = std::clamp(count, 0, std::min(capacity_, newCapacity));

    T *newData = newCapacity > 0 ? new T[newCapacity] : nullptr;

    // Copy existing elements
    if (count > 0 && data_) {
      std::copy(data_, data_ + count, newData);
    }

    // Initialize new elements to default value
    if (newData) {
      std::fill(newData + count, newData + newCapacity, defaultValue);
    }

    delete[] data_;
    data_ = newData;
    capacity_ = newCapacity;
  }

  // Resize keeping as many existing elements as fit.
  void resize(int newCapacity) { resize(newCapacity, capacity_, T{}); }

  // Fill all elements with a value
  void fill(int size, const T &value) {
    if (data_) {
      std::fill(data_, data_ + std::min(size, capacity_), value);
    }
  }

  // Fill a range with a value
  void fillRange(int start, int end, const T &value) {
    if (data_ && start >= 0) {
      std::fill(data_ + start, data_ + std::min(end, capacity_), value);
    }
  }
};

/**
 * AlignedDynamicArray<T> - Cache-aligned RAII wrapper for dynamic arrays.
 *
 * Same as DynamicArray but with cache-line alignment for better performance
 * in hot paths (parallel processing, SIMD, etc.).
 */
template <typename T, size_t Alignment = 64> class AlignedDynamicArray {
private:
  T *data_;
  int capacity_;

  // Aligned allocation helper
  static T *allocateAligned(int count) {
    if (count <= 0)
      return nullptr;
    void *ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(count * sizeof(T), Alignment);
#else
    if (posix_memalign(&ptr, Alignment, count * sizeof(T)) != 0) {
      ptr = nullptr;
    }
#endif
    return static_cast<T *>(ptr);
  }

  // Aligned deallocation helper
  static void deallocateAligned(T *ptr) {
    if (ptr) {
#ifdef _WIN32
      _aligned_free(ptr);
#else
      free(ptr);
#endif
    }
  }

public:
  AlignedDynamicArray() noexcept : data_(nullptr), capacity_(0) {}

  // Construct with initial capacity
  explicit AlignedDynamicArray(int capacity)
      : data_(allocateAligned(capacity)),
        capacity_(capacity > 0 ? capacity : 0) {
    if (data_) {
      std::fill(data_, data_ + capacity_, T{});
    }
  }

  // Construct with initial capacity and default value
  AlignedDynamicArray(int capacity, const T &defaultValue)
      : data_(allocateAligned(capacity)),
        capacity_(capacity > 0 ? capacity : 0) {
    if (data_) {
      std::fill(data_, data_ + capacity_, defaultValue);
    }
  }

  // Destructor - RAII cleanup
  ~AlignedDynamicArray() { deallocateAligned(data_); }

  // Move constructor
  AlignedDynamicArray(AlignedDynamicArray &&other) noexcept
      : data_(other.data_), capacity_(other.capacity_) {
    other.data_ = nullptr;
    other.capacity_ = 0;
  }

  // Move assignment
  AlignedDynamicArray &operator=(AlignedDynamicArray &&other) noexcept {
    if (this != &other) {
      deallocateAligned(data_);
      data_ = other.data_;
      capacity_ = other.capacity_;
      other.data_ = nullptr;
      other.capacity_ = 0;
    }
    return *this;
  }

  // Disable copy (prevent double-free)
  AlignedDynamicArray(const AlignedDynamicArray &) = delete;
  AlignedDynamicArray &operator=(const AlignedDynamicArray &) = delete;

  // Array access
  T &operator[](int index) {
    assert(index >= 0 && index < capacity_);
    return data_[index];
  }
  const T &operator[](int index) const {
    assert(index >= 0 && index < capacity_);
    return data_[index];
  }

  // Get raw pointer (for compatibility with existing code)
  T *data() noexcept { return data_; }
  const T *data() const noexcept { return data_; }

  // Implicit conversion to T* for seamless integration
  operator T *() noexcept { return data_; }
  operator const T *() const noexcept { return data_; }

  int capacity() const noexcept { return capacity_; }

  // Resize array, preserving `count` elements
  void resize(int newCapacity, int count) { resize(newCapacity, count, T{}); }

  // Resize with a specific default value for new elements
  void resize(int newCapacity, int count, const T &defaultValue) {
    newCapacity = std::max(newCapacity, 0);
    count = std::clamp(count, 0, std::min(capacity_, newCapacity));

    T *newData = allocateAligned(newCapacity);

    // Copy existing elements
    if (count > 0 && data_) {
      std::copy(data_, data_ + count, newData);
    }

    // Initialize new elements to default value
    if (newData) {
      std::fill(newData + count, newData + newCapacity, defaultValue);
    }

    deallocateAligned(data_);
    data_ = newData;
    capacity_ = newCapacity;
  }

  // Resize keeping as many existing elements as fit.
  void resize(int newCapacity) { resize(newCapacity, capacity_, T{}); }

  // Fill all elements with a value
  void fill(int size, const T &value) {
    if (data_) {
      std::fill(data_, data_ + std::min(size, capacity_), value);
    }
  }

  // Fill a range with a value
  void fillRange(int start, int end, const T &value) {
    if (data_ && start >= 0) {
      std::fill(data_ + start, data_ + std::min(end, capacity_), value);
    }
  }
};

#endif // ATM_DYNAMIC_ARRAY_H
