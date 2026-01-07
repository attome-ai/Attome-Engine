#ifndef ATM_DYNAMIC_ARRAY_H
#define ATM_DYNAMIC_ARRAY_H

#include <algorithm>
#include <cstdint>
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
 */
template <typename T> class DynamicArray {
private:
  T *data_;

public:
  // Construct with initial capacity
  explicit DynamicArray(int capacity)
      : data_(capacity > 0 ? new T[capacity] : nullptr) {
    if (data_) {
      std::fill(data_, data_ + capacity, T{});
    }
  }

  // Construct with initial capacity and default value
  DynamicArray(int capacity, const T &defaultValue)
      : data_(capacity > 0 ? new T[capacity] : nullptr) {
    if (data_) {
      std::fill(data_, data_ + capacity, defaultValue);
    }
  }

  // Destructor - RAII cleanup
  ~DynamicArray() { delete[] data_; }

  // Move constructor
  DynamicArray(DynamicArray &&other) noexcept : data_(other.data_) {
    other.data_ = nullptr;
  }

  // Move assignment
  DynamicArray &operator=(DynamicArray &&other) noexcept {
    if (this != &other) {
      delete[] data_;
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }

  // Disable copy (prevent double-free)
  DynamicArray(const DynamicArray &) = delete;
  DynamicArray &operator=(const DynamicArray &) = delete;

  // Array access
  T &operator[](int index) { return data_[index]; }
  const T &operator[](int index) const { return data_[index]; }

  // Get raw pointer (for compatibility with existing code)
  T *data() noexcept { return data_; }
  const T *data() const noexcept { return data_; }

  // Implicit conversion to T* for seamless integration
  operator T *() noexcept { return data_; }
  operator const T *() const noexcept { return data_; }

  // Get capacity - REMOVED (managed by container)
  // int capacity() const noexcept { return capacity_; }

  // Resize array, preserving `count` elements and filling new elements with
  // default
  void resize(int newCapacity, int count) {
    // Caller checks if newCapacity <= current_capacity

    T *newData = new T[newCapacity];

    // Copy existing elements
    if (count > 0 && data_) {
      std::copy(data_, data_ + count, newData);
    }

    // Initialize new elements to default
    std::fill(newData + count, newData + newCapacity, T{});

    delete[] data_;
    data_ = newData;
  }

  // Resize with a specific default value for new elements
  void resize(int newCapacity, int count, const T &defaultValue) {
    // Caller checks if newCapacity <= current_capacity

    T *newData = new T[newCapacity];

    // Copy existing elements
    if (count > 0 && data_) {
      std::copy(data_, data_ + count, newData);
    }

    // Initialize new elements to default value
    std::fill(newData + count, newData + newCapacity, defaultValue);

    delete[] data_;
    data_ = newData;
  }

  // Fill all elements with a value
  void fill(int size, const T &value) {
    if (data_) {
      std::fill(data_, data_ + size, value);
    }
  }

  // Fill a range with a value
  void fillRange(int start, int end, const T &value) {
    if (data_ && start >= 0) {
      std::fill(data_ + start, data_ + end, value);
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
  // Construct with initial capacity
  explicit AlignedDynamicArray(int capacity)
      : data_(allocateAligned(capacity)) {
    if (data_) {
      std::fill(data_, data_ + capacity, T{});
    }
  }

  // Construct with initial capacity and default value
  AlignedDynamicArray(int capacity, const T &defaultValue)
      : data_(allocateAligned(capacity)) {
    if (data_) {
      std::fill(data_, data_ + capacity, defaultValue);
    }
  }

  // Destructor - RAII cleanup
  ~AlignedDynamicArray() { deallocateAligned(data_); }

  // Move constructor
  AlignedDynamicArray(AlignedDynamicArray &&other) noexcept
      : data_(other.data_) {
    other.data_ = nullptr;
  }

  // Move assignment
  AlignedDynamicArray &operator=(AlignedDynamicArray &&other) noexcept {
    if (this != &other) {
      deallocateAligned(data_);
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }

  // Disable copy (prevent double-free)
  AlignedDynamicArray(const AlignedDynamicArray &) = delete;
  AlignedDynamicArray &operator=(const AlignedDynamicArray &) = delete;

  // Array access
  T &operator[](int index) { return data_[index]; }
  const T &operator[](int index) const { return data_[index]; }

  // Get raw pointer (for compatibility with existing code)
  T *data() noexcept { return data_; }
  const T *data() const noexcept { return data_; }

  // Implicit conversion to T* for seamless integration
  operator T *() noexcept { return data_; }
  operator const T *() const noexcept { return data_; }

  // Get capacity - REMOVED
  // int capacity() const noexcept { return capacity_; }

  // Resize array, preserving `count` elements
  void resize(int newCapacity, int count) {
    // Caller checks capacity

    T *newData = allocateAligned(newCapacity);

    // Copy existing elements
    if (count > 0 && data_) {
      std::copy(data_, data_ + count, newData);
    }

    // Initialize new elements to default
    std::fill(newData + count, newData + newCapacity, T{});

    deallocateAligned(data_);
    data_ = newData;
  }

  // Resize with a specific default value for new elements
  void resize(int newCapacity, int count, const T &defaultValue) {
    // Caller checks capacity

    T *newData = allocateAligned(newCapacity);

    // Copy existing elements
    if (count > 0 && data_) {
      std::copy(data_, data_ + count, newData);
    }

    // Initialize new elements to default value
    std::fill(newData + count, newData + newCapacity, defaultValue);

    deallocateAligned(data_);
    data_ = newData;
  }

  // Fill all elements with a value
  void fill(int size, const T &value) {
    if (data_) {
      std::fill(data_, data_ + size, value);
    }
  }

  // Fill a range with a value
  void fillRange(int start, int end, const T &value) {
    if (data_ && start >= 0) {
      std::fill(data_ + start, data_ + end, value);
    }
  }
};

#endif // ATM_DYNAMIC_ARRAY_H
