#pragma once
#include <string>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <random>
#include <cassert>
#include <algorithm>
#include <vector>
#include <cstdint>      // For uint8_t, etc.
#include <chrono>


class ByteBufferReader {
private:
    const std::vector<uint8_t>& buffer_;
    size_t position_ = 0;
    ByteBufferReader(const ByteBufferReader&) = delete;
    ByteBufferReader& operator=(const ByteBufferReader&) = delete;

    inline void checkRemaining(size_t size) const {
        if (position_ + size > buffer_.size()) {
            throw std::out_of_range("Reading beyond buffer bounds");
        }
    }

public:
    inline explicit ByteBufferReader(const std::vector<uint8_t>& buffer) : buffer_(buffer), position_(0) {}

    inline void reset() { position_ = 0; }

    inline size_t position() const { return position_; }

    inline void position(size_t newPosition) {
        if (newPosition > buffer_.size()) {
            throw std::out_of_range("Position beyond buffer bounds");
        }
        position_ = newPosition;
    }

    inline size_t remaining() const { return buffer_.size() - position_; }

    inline bool hasRemaining() const { return position_ < buffer_.size(); }

    inline uint8_t readByte() {
        checkRemaining(sizeof(uint8_t));
        return buffer_[position_++];
    }

    inline int8_t readInt8() {
        checkRemaining(sizeof(int8_t));
        return static_cast<int8_t>(buffer_[position_++]);
    }

    inline uint16_t readUInt16() {
        checkRemaining(sizeof(uint16_t));
        uint16_t value;
        std::memcpy(&value, &buffer_[position_], sizeof(uint16_t));
        position_ += sizeof(uint16_t);
        return value;
    }

    inline int16_t readInt16() {
        checkRemaining(sizeof(int16_t));
        int16_t value;
        std::memcpy(&value, &buffer_[position_], sizeof(int16_t));
        position_ += sizeof(int16_t);
        return value;
    }

    inline uint32_t readUInt32() {
        checkRemaining(sizeof(uint32_t));
        uint32_t value;
        std::memcpy(&value, &buffer_[position_], sizeof(uint32_t));
        position_ += sizeof(uint32_t);
        return value;
    }

    inline int32_t readInt32() {
        checkRemaining(sizeof(int32_t));
        int32_t value;
        std::memcpy(&value, &buffer_[position_], sizeof(int32_t));
        position_ += sizeof(int32_t);
        return value;
    }

    inline uint64_t readUInt64() {
        checkRemaining(sizeof(uint64_t));
        uint64_t value;
        std::memcpy(&value, &buffer_[position_], sizeof(uint64_t));
        position_ += sizeof(uint64_t);
        return value;
    }

    inline int64_t readInt64() {
        checkRemaining(sizeof(int64_t));
        int64_t value;
        std::memcpy(&value, &buffer_[position_], sizeof(int64_t));
        position_ += sizeof(int64_t);
        return value;
    }

    inline float readFloat() {
        checkRemaining(sizeof(float));
        float value;
        std::memcpy(&value, &buffer_[position_], sizeof(float));
        position_ += sizeof(float);
        return value;
    }

    inline double readDouble() {
        checkRemaining(sizeof(double));
        double value;
        std::memcpy(&value, &buffer_[position_], sizeof(double));
        position_ += sizeof(double);
        return value;
    }

    inline void readBytes(void* dest, size_t length) {
        checkRemaining(length);
        std::memcpy(dest, &buffer_[position_], length);
        position_ += length;
    }

    inline void readBytes(std::vector<uint8_t>& dest, size_t length) {
        checkRemaining(length);
        dest.resize(length);
        if (length > 0) {
            std::memcpy(dest.data(), &buffer_[position_], length);
            position_ += length;
        }
    }

    template<typename T>
    inline void readArray(T* dest, size_t count) {
        const size_t bytesToRead = count * sizeof(T);
        checkRemaining(bytesToRead);
        std::memcpy(dest, &buffer_[position_], bytesToRead);
        position_ += bytesToRead;
    }

    template<typename T>
    inline void readArray(std::vector<T>& dest, size_t count) {
        dest.resize(count);
        if (count > 0) {
            readArray(dest.data(), count);
        }
    }

    inline std::string readString() {
        uint32_t length = readUInt32();
        checkRemaining(length);
        std::string str(reinterpret_cast<const char*>(&buffer_[position_]), length);
        position_ += length;
        return str;
    }

    // Fast functions - no bounds checking

    inline uint8_t readByteFast() {
        return buffer_[position_++];
    }

    inline int8_t readInt8Fast() {
        return static_cast<int8_t>(buffer_[position_++]);
    }

    inline uint16_t readUInt16Fast() {
        uint16_t value;
        std::memcpy(&value, &buffer_[position_], sizeof(uint16_t));
        position_ += sizeof(uint16_t);
        return value;
    }

    inline int16_t readInt16Fast() {
        int16_t value;
        std::memcpy(&value, &buffer_[position_], sizeof(int16_t));
        position_ += sizeof(int16_t);
        return value;
    }

    inline uint32_t readUInt32Fast() {
        uint32_t value;
        std::memcpy(&value, &buffer_[position_], sizeof(uint32_t));
        position_ += sizeof(uint32_t);
        return value;
    }

    inline int32_t readInt32Fast() {
        int32_t value;
        std::memcpy(&value, &buffer_[position_], sizeof(int32_t));
        position_ += sizeof(int32_t);
        return value;
    }

    inline uint64_t readUInt64Fast() {
        uint64_t value;
        std::memcpy(&value, &buffer_[position_], sizeof(uint64_t));
        position_ += sizeof(uint64_t);
        return value;
    }

    inline int64_t readInt64Fast() {
        int64_t value;
        std::memcpy(&value, &buffer_[position_], sizeof(int64_t));
        position_ += sizeof(int64_t);
        return value;
    }

    inline float readFloatFast() {
        float value;
        std::memcpy(&value, &buffer_[position_], sizeof(float));
        position_ += sizeof(float);
        return value;
    }

    inline double readDoubleFast() {
        double value;
        std::memcpy(&value, &buffer_[position_], sizeof(double));
        position_ += sizeof(double);
        return value;
    }

    inline void readBytesFast(void* dest, size_t length) {
        std::memcpy(dest, &buffer_[position_], length);
        position_ += length;
    }

    inline void readBytesFast(std::vector<uint8_t>& dest, size_t length) {
        dest.resize(length);
        if (length > 0) {
            std::memcpy(dest.data(), &buffer_[position_], length);
            position_ += length;
        }
    }

    template<typename T>
    inline void readArrayFast(T* dest, size_t count) {
        const size_t bytesToRead = count * sizeof(T);
        std::memcpy(dest, &buffer_[position_], bytesToRead);
        position_ += bytesToRead;
    }

    template<typename T>
    inline void readArrayFast(std::vector<T>& dest, size_t count) {
        dest.resize(count);
        if (count > 0) {
            readArrayFast(dest.data(), count);
        }
    }

    inline std::string readStringFast() {
        uint32_t length;
        std::memcpy(&length, &buffer_[position_], sizeof(uint32_t));
        position_ += sizeof(uint32_t);

        std::string str(reinterpret_cast<const char*>(&buffer_[position_]), length);
        position_ += length;
        return str;
    }
};

class ByteBufferWriter {
private:
    std::vector<uint8_t>& buffer_;
    size_t position_ = 0;
    ByteBufferWriter(const ByteBufferWriter&) = delete;
    ByteBufferWriter& operator=(const ByteBufferWriter&) = delete;

    inline void ensureCapacity(size_t additionalBytes) {
        const size_t requiredSize = position_ + additionalBytes;
        if (requiredSize > buffer_.size()) {
            buffer_.resize(requiredSize);
        }
    }

public:
    inline explicit ByteBufferWriter(std::vector<uint8_t>& buffer) : buffer_(buffer), position_(0) {}

    inline void reset() { position_ = 0; }

    inline size_t position() const { return position_; }

    inline void position(size_t newPosition) {
        if (newPosition > buffer_.size()) {
            buffer_.resize(newPosition);
        }
        position_ = newPosition;
    }

    inline void writeByte(uint8_t value) {
        ensureCapacity(sizeof(uint8_t));
        buffer_[position_++] = value;
    }

    inline void writeInt8(int8_t value) {
        ensureCapacity(sizeof(int8_t));
        buffer_[position_++] = static_cast<uint8_t>(value);
    }

    inline void writeUInt16(uint16_t value) {
        ensureCapacity(sizeof(uint16_t));
        std::memcpy(&buffer_[position_], &value, sizeof(uint16_t));
        position_ += sizeof(uint16_t);
    }

    inline void writeInt16(int16_t value) {
        ensureCapacity(sizeof(int16_t));
        std::memcpy(&buffer_[position_], &value, sizeof(int16_t));
        position_ += sizeof(int16_t);
    }

    inline void writeUInt32(uint32_t value) {
        ensureCapacity(sizeof(uint32_t));
        std::memcpy(&buffer_[position_], &value, sizeof(uint32_t));
        position_ += sizeof(uint32_t);
    }

    inline void writeInt32(int32_t value) {
        ensureCapacity(sizeof(int32_t));
        std::memcpy(&buffer_[position_], &value, sizeof(int32_t));
        position_ += sizeof(int32_t);
    }

    inline void writeUInt64(uint64_t value) {
        ensureCapacity(sizeof(uint64_t));
        std::memcpy(&buffer_[position_], &value, sizeof(uint64_t));
        position_ += sizeof(uint64_t);
    }

    inline void writeInt64(int64_t value) {
        ensureCapacity(sizeof(int64_t));
        std::memcpy(&buffer_[position_], &value, sizeof(int64_t));
        position_ += sizeof(int64_t);
    }

    inline void writeFloat(float value) {
        ensureCapacity(sizeof(float));
        std::memcpy(&buffer_[position_], &value, sizeof(float));
        position_ += sizeof(float);
    }

    inline void writeDouble(double value) {
        ensureCapacity(sizeof(double));
        std::memcpy(&buffer_[position_], &value, sizeof(double));
        position_ += sizeof(double);
    }

    inline void writeBytes(const void* src, size_t length) {
        ensureCapacity(length);
        std::memcpy(&buffer_[position_], src, length);
        position_ += length;
    }

    inline void writeBytes(const std::vector<uint8_t>& src) {
        const size_t length = src.size();
        ensureCapacity(length);
        if (length > 0) {
            std::memcpy(&buffer_[position_], src.data(), length);
            position_ += length;
        }
    }

    template<typename T>
    inline void writeArray(const T* src, size_t count) {
        const size_t bytesToWrite = count * sizeof(T);
        ensureCapacity(bytesToWrite);
        std::memcpy(&buffer_[position_], src, bytesToWrite);
        position_ += bytesToWrite;
    }

    template<typename T>
    inline void writeArray(const std::vector<T>& src) {
        const size_t count = src.size();
        if (count > 0) {
            writeArray(src.data(), count);
        }
    }

    inline void writeString(const std::string& str) {
        const uint32_t length = static_cast<uint32_t>(str.length());
        writeUInt32(length);
        ensureCapacity(length);
        std::memcpy(&buffer_[position_], str.data(), length);
        position_ += length;
    }

    // Fast functions - no capacity check

    inline void writeByteFast(uint8_t value) {
        buffer_[position_++] = value;
    }

    inline void writeInt8Fast(int8_t value) {
        buffer_[position_++] = static_cast<uint8_t>(value);
    }

    inline void writeUInt16Fast(uint16_t value) {
        std::memcpy(&buffer_[position_], &value, sizeof(uint16_t));
        position_ += sizeof(uint16_t);
    }

    inline void writeInt16Fast(int16_t value) {
        std::memcpy(&buffer_[position_], &value, sizeof(int16_t));
        position_ += sizeof(int16_t);
    }

    inline void writeUInt32Fast(uint32_t value) {
        std::memcpy(&buffer_[position_], &value, sizeof(uint32_t));
        position_ += sizeof(uint32_t);
    }

    inline void writeInt32Fast(int32_t value) {
        std::memcpy(&buffer_[position_], &value, sizeof(int32_t));
        position_ += sizeof(int32_t);
    }

    inline void writeUInt64Fast(uint64_t value) {
        std::memcpy(&buffer_[position_], &value, sizeof(uint64_t));
        position_ += sizeof(uint64_t);
    }

    inline void writeInt64Fast(int64_t value) {
        std::memcpy(&buffer_[position_], &value, sizeof(int64_t));
        position_ += sizeof(int64_t);
    }

    inline void writeFloatFast(float value) {
        std::memcpy(&buffer_[position_], &value, sizeof(float));
        position_ += sizeof(float);
    }

    inline void writeDoubleFast(double value) {
        std::memcpy(&buffer_[position_], &value, sizeof(double));
        position_ += sizeof(double);
    }

    inline void writeBytesFast(const void* src, size_t length) {
        std::memcpy(&buffer_[position_], src, length);
        position_ += length;
    }

    inline void writeBytesFast(const std::vector<uint8_t>& src) {
        const size_t length = src.size();
        if (length > 0) {
            std::memcpy(&buffer_[position_], src.data(), length);
            position_ += length;
        }
    }

    template<typename T>
    inline void writeArrayFast(const T* src, size_t count) {
        const size_t bytesToWrite = count * sizeof(T);
        std::memcpy(&buffer_[position_], src, bytesToWrite);
        position_ += bytesToWrite;
    }

    template<typename T>
    inline void writeArrayFast(const std::vector<T>& src) {
        const size_t count = src.size();
        if (count > 0) {
            writeArrayFast(src.data(), count);
        }
    }

    inline void writeStringFast(const std::string& str) {
        const uint32_t length = static_cast<uint32_t>(str.length());
        std::memcpy(&buffer_[position_], &length, sizeof(uint32_t));
        position_ += sizeof(uint32_t);

        std::memcpy(&buffer_[position_], str.data(), length);
        position_ += length;
    }
};




// --- Forward Declarations ---
class FixedByteBufferReader;
class FixedByteBufferWriter;

// --- FixedByteBufferReader ---
// Reads data from a non-owned, fixed-size buffer. Assumes single-threaded access.
class FixedByteBufferReader {
private:
    const uint8_t* data_ = nullptr; // Pointer to the beginning of the buffer data
    size_t size_ = 0;           // Total size of the buffer
    size_t position_ = 0;       // Current reading position

    // Non-copyable and non-assignable
    FixedByteBufferReader(const FixedByteBufferReader&) = delete;
    FixedByteBufferReader& operator=(const FixedByteBufferReader&) = delete;

    // Checks if 'count' bytes can be read from the current position.
    inline void checkRemaining(size_t count) const {
        if (position_ + count > size_) {
            throw std::out_of_range("Reading beyond fixed buffer bounds");
        }
    }

public:
    // Constructor: Takes a pointer to the data and its total size.
    // The buffer pointed to by 'data' must remain valid for the lifetime of this reader.
    inline explicit FixedByteBufferReader(const uint8_t* data, size_t position, size_t size)
        : data_(data), size_(size), position_(position) {
        if (!data && size > 0) {
            throw std::invalid_argument("Null data pointer with non-zero size");
        }
    }
    inline explicit FixedByteBufferReader(std::string& str) : data_((uint8_t*)str.data()), position_(0), size_(str.size())
    {
    }
    // Resets the reading position to the beginning.
    inline void reset() { position_ = 0; }

    // Gets the current reading position.
    inline size_t position() const { return position_; }

    // Sets the reading position. Throws if the new position is out of bounds.
    inline void position(size_t newPosition) {
        if (newPosition > size_) { // Can position at the end, but not beyond
            throw std::out_of_range("Position set beyond fixed buffer bounds");
        }
        position_ = newPosition;
    }

    // Gets the total size of the underlying buffer.
    inline size_t size() const { return size_; }

    // Gets the number of bytes remaining to be read.
    inline size_t remaining() const { return size_ - position_; }

    // Returns true if there are more bytes to read.
    inline bool hasRemaining() const { return position_ < size_; }

    // Returns a pointer to the current read position in the buffer.
    inline const uint8_t* currentData() const {
        return data_ + position_;
    }

    // --- Read Methods (Bounds Checked) ---

    inline uint8_t readByte() {
        checkRemaining(sizeof(uint8_t));
        return data_[position_++];
    }

    inline int8_t readInt8() {
        checkRemaining(sizeof(int8_t));
        return static_cast<int8_t>(data_[position_++]);
    }

    inline uint16_t readUInt16() {
        checkRemaining(sizeof(uint16_t));
        uint16_t value;
        std::memcpy(&value, data_ + position_, sizeof(uint16_t));
        position_ += sizeof(uint16_t);
        return value;
    }

    inline int16_t readInt16() {
        checkRemaining(sizeof(int16_t));
        int16_t value;
        std::memcpy(&value, data_ + position_, sizeof(int16_t));
        position_ += sizeof(int16_t);
        return value;
    }

    inline uint32_t readUInt32() {
        checkRemaining(sizeof(uint32_t));
        uint32_t value;
        std::memcpy(&value, data_ + position_, sizeof(uint32_t));
        position_ += sizeof(uint32_t);
        return value;
    }

    inline int32_t readInt32() {
        checkRemaining(sizeof(int32_t));
        int32_t value;
        std::memcpy(&value, data_ + position_, sizeof(int32_t));
        position_ += sizeof(int32_t);
        return value;
    }

    inline uint64_t readUInt64() {
        checkRemaining(sizeof(uint64_t));
        uint64_t value;
        std::memcpy(&value, data_ + position_, sizeof(uint64_t));
        position_ += sizeof(uint64_t);
        return value;
    }

    inline int64_t readInt64() {
        checkRemaining(sizeof(int64_t));
        int64_t value;
        std::memcpy(&value, data_ + position_, sizeof(int64_t));
        position_ += sizeof(int64_t);
        return value;
    }

    inline float readFloat() {
        checkRemaining(sizeof(float));
        float value;
        std::memcpy(&value, data_ + position_, sizeof(float));
        position_ += sizeof(float);
        return value;
    }

    inline double readDouble() {
        checkRemaining(sizeof(double));
        double value;
        std::memcpy(&value, data_ + position_, sizeof(double));
        position_ += sizeof(double);
        return value;
    }

    // Reads 'length' bytes into the destination buffer 'dest'.
    inline void readBytes(void* dest, size_t length) {
        checkRemaining(length);
        if (length > 0) {
            std::memcpy(dest, data_ + position_, length);
            position_ += length;
        }
    }

    // Reads 'length' bytes into the destination vector 'dest'. The vector is resized.
    inline void readBytes(std::vector<uint8_t>& dest, size_t length) {
        checkRemaining(length);
        dest.resize(length);
        if (length > 0) {
            std::memcpy(dest.data(), data_ + position_, length);
            position_ += length;
        }
    }

    // Reads 'count' elements of type T into the destination array 'dest'.
    template<typename T>
    inline void readArray(T* dest, size_t count) {
        const size_t bytesToRead = count * sizeof(T);
        checkRemaining(bytesToRead);
        if (bytesToRead > 0) {
            std::memcpy(dest, data_ + position_, bytesToRead);
            position_ += bytesToRead;
        }
    }

    // Reads 'count' elements of type T into the destination vector 'dest'. The vector is resized.
    template<typename T>
    inline void readArray(std::vector<T>& dest, size_t count) {
        dest.resize(count);
        if (count > 0) {
            readArray(dest.data(), count); // Uses the checked pointer version
        }
    }

    // Reads a string prefixed with its uint32 length.
    inline std::string readString() {
        uint32_t length = readUInt32(); // Reads length, advances position, checks bounds
        checkRemaining(length);         // Checks bounds for the string data itself
        std::string str(reinterpret_cast<const char*>(data_ + position_), length);
        position_ += length;
        return str;
    }

    // --- Read Methods (Fast - No Bounds Checking) ---
    // WARNING: Using these methods without ensuring sufficient data remaining
    // can lead to reading out of bounds and undefined behavior.

    inline uint8_t readByteFast() {
        return data_[position_++];
    }

    inline int8_t readInt8Fast() {
        return static_cast<int8_t>(data_[position_++]);
    }

    inline uint16_t readUInt16Fast() {
        uint16_t value;
        std::memcpy(&value, data_ + position_, sizeof(uint16_t));
        position_ += sizeof(uint16_t);
        return value;
    }

    inline int16_t readInt16Fast() {
        int16_t value;
        std::memcpy(&value, data_ + position_, sizeof(int16_t));
        position_ += sizeof(int16_t);
        return value;
    }

    inline uint32_t readUInt32Fast() {
        uint32_t value;
        std::memcpy(&value, data_ + position_, sizeof(uint32_t));
        position_ += sizeof(uint32_t);
        return value;
    }

    inline int32_t readInt32Fast() {
        int32_t value;
        std::memcpy(&value, data_ + position_, sizeof(int32_t));
        position_ += sizeof(int32_t);
        return value;
    }

    inline uint64_t readUInt64Fast() {
        uint64_t value;
        std::memcpy(&value, data_ + position_, sizeof(uint64_t));
        position_ += sizeof(uint64_t);
        return value;
    }

    inline int64_t readInt64Fast() {
        int64_t value;
        std::memcpy(&value, data_ + position_, sizeof(int64_t));
        position_ += sizeof(int64_t);
        return value;
    }

    inline float readFloatFast() {
        float value;
        std::memcpy(&value, data_ + position_, sizeof(float));
        position_ += sizeof(float);
        return value;
    }

    inline double readDoubleFast() {
        double value;
        std::memcpy(&value, data_ + position_, sizeof(double));
        position_ += sizeof(double);
        return value;
    }

    inline void readBytesFast(void* dest, size_t length) {
        if (length > 0) {
            std::memcpy(dest, data_ + position_, length);
            position_ += length;
        }
    }

    inline void readBytesFast(std::vector<uint8_t>& dest, size_t length) {
        dest.resize(length);
        if (length > 0) {
            std::memcpy(dest.data(), data_ + position_, length);
            position_ += length;
        }
    }

    template<typename T>
    inline void readArrayFast(T* dest, size_t count) {
        const size_t bytesToRead = count * sizeof(T);
        if (bytesToRead > 0) {
            std::memcpy(dest, data_ + position_, bytesToRead);
            position_ += bytesToRead;
        }
    }

    template<typename T>
    inline void readArrayFast(std::vector<T>& dest, size_t count) {
        dest.resize(count);
        if (count > 0) {
            readArrayFast(dest.data(), count); // Uses the fast pointer version
        }
    }

    inline std::string readStringFast() {
        // Reads length without check, advances position
        uint32_t length;
        std::memcpy(&length, data_ + position_, sizeof(uint32_t));
        position_ += sizeof(uint32_t);

        // Reads string data without check - relies on caller ensuring length is valid!
        std::string str(reinterpret_cast<const char*>(data_ + position_), length);
        position_ += length;
        return str;
    }
    inline bool canRead(size_t bytes)
    {
        return (position_ + bytes) <= size_;
    }
    // Check if a string (uint32 length + data) can be read
    inline bool canReadString() {
        if (!canRead(sizeof(uint32_t))) {
            return false;
        }

        // Peek at the length without advancing position
        uint32_t length;
        std::memcpy(&length, data_ + position_, sizeof(uint32_t));

        // Check if we have enough bytes for both length and string data
        return canRead(sizeof(uint32_t) + length);
    }

    // Basic numeric type checks
    inline bool canReadByte() { return canRead(sizeof(uint8_t)); }
    inline bool canReadInt8() { return canRead(sizeof(int8_t)); }
    inline bool canReadUInt16() { return canRead(sizeof(uint16_t)); }
    inline bool canReadInt16() { return canRead(sizeof(int16_t)); }
    inline bool canReadUInt32() { return canRead(sizeof(uint32_t)); }
    inline bool canReadInt32() { return canRead(sizeof(int32_t)); }
    inline bool canReadUInt64() { return canRead(sizeof(uint64_t)); }
    inline bool canReadInt64() { return canRead(sizeof(int64_t)); }
    inline bool canReadFloat() { return canRead(sizeof(float)); }
    inline bool canReadDouble() { return canRead(sizeof(double)); }

    // Check if an array of type T with count elements can be read
    template<typename T>
    inline bool canReadArray(size_t count) {
        return canRead(count * sizeof(T));
    }

    // Check if a byte array of specific length can be read
    inline bool canReadBytes(size_t length) {
        return canRead(length);
    }
};


class FixedByteBufferWriter {
private:
    uint8_t* data_ = nullptr;   // Pointer to the beginning of the buffer data
    size_t capacity_ = 0;       // Total capacity of the buffer
    size_t position_ = 0;       // Current writing position

    // Non-copyable and non-assignable
    FixedByteBufferWriter(const FixedByteBufferWriter&) = delete;
    FixedByteBufferWriter& operator=(const FixedByteBufferWriter&) = delete;

    // Checks if 'additionalBytes' can be written without exceeding capacity.
    inline void checkCapacity(size_t additionalBytes) const {
        if (position_ + additionalBytes > capacity_) {
            // Using length_error as it signals exceeding a maximum size constraint
            throw std::length_error("Writing beyond fixed buffer capacity");
        }
    }

public:

    explicit FixedByteBufferWriter(std::string& str) noexcept
        : data_(reinterpret_cast<uint8_t*>(str.data()))
        , capacity_(str.size()), position_(0)
    {
    }

    // Constructor: Takes a pointer to the writable buffer and its total capacity.
    // The buffer pointed to by 'data' must remain valid and writable for the lifetime of this writer.
    inline explicit FixedByteBufferWriter(uint8_t* data, size_t position, size_t capacity)
        : data_(data), capacity_(capacity), position_(position) {
        if (!data && capacity > 0) {
            throw std::invalid_argument("Null data pointer with non-zero capacity");
        }
    }

    // Resets the writing position to the beginning.
    inline void resetPosition() { position_ = 0; }
    inline void reset(uint8_t* data, size_t pos) {
        position_ = pos;
        data_ = data;
    }

    // Gets the current writing position (which is also the number of bytes written so far).
    inline size_t position() const { return position_; }

    // Sets the writing position. Throws if the new position is beyond capacity.
    // Be careful when using this, as it might overwrite existing data or leave gaps.
    inline void position(size_t newPosition) {
        if (newPosition > capacity_) { // Can position at the end, but not beyond
            throw std::out_of_range("Position set beyond fixed buffer capacity");
        }
        position_ = newPosition;
    }

    // Gets the total capacity of the underlying buffer.
    inline size_t capacity() const { return capacity_; }

    // Gets the number of bytes remaining in the buffer's capacity.
    inline size_t remainingCapacity() const { return capacity_ - position_; }

    // Returns true if there is space remaining to write.
    inline bool hasRemainingCapacity() const { return position_ < capacity_; }

    // Returns a pointer to the current write position in the buffer.
    inline uint8_t* currentData() const {
        return data_ + position_;
    }

    // Returns the number of bytes written so far.
    inline size_t writtenSize() const { return position_; }

    // --- Write Methods (Capacity Checked) ---

    inline void writeByte(uint8_t value) {
        checkCapacity(sizeof(uint8_t));
        data_[position_++] = value;
    }

    inline void writeInt8(int8_t value) {
        checkCapacity(sizeof(int8_t));
        data_[position_++] = static_cast<uint8_t>(value);
    }

    inline void writeUInt16(uint16_t value) {
        checkCapacity(sizeof(uint16_t));
        std::memcpy(data_ + position_, &value, sizeof(uint16_t));
        position_ += sizeof(uint16_t);
    }

    inline void writeInt16(int16_t value) {
        checkCapacity(sizeof(int16_t));
        std::memcpy(data_ + position_, &value, sizeof(int16_t));
        position_ += sizeof(int16_t);
    }

    inline void writeUInt32(uint32_t value) {
        checkCapacity(sizeof(uint32_t));
        std::memcpy(data_ + position_, &value, sizeof(uint32_t));
        position_ += sizeof(uint32_t);
    }

    inline void writeInt32(int32_t value) {
        checkCapacity(sizeof(int32_t));
        std::memcpy(data_ + position_, &value, sizeof(int32_t));
        position_ += sizeof(int32_t);
    }

    inline void writeUInt64(uint64_t value) {
        checkCapacity(sizeof(uint64_t));
        std::memcpy(data_ + position_, &value, sizeof(uint64_t));
        position_ += sizeof(uint64_t);
    }

    inline void writeInt64(int64_t value) {
        checkCapacity(sizeof(int64_t));
        std::memcpy(data_ + position_, &value, sizeof(int64_t));
        position_ += sizeof(int64_t);
    }

    inline void writeFloat(float value) {
        checkCapacity(sizeof(float));
        std::memcpy(data_ + position_, &value, sizeof(float));
        position_ += sizeof(float);
    }

    inline void writeDouble(double value) {
        checkCapacity(sizeof(double));
        std::memcpy(data_ + position_, &value, sizeof(double));
        position_ += sizeof(double);
    }

    // Writes 'length' bytes from the source buffer 'src'.
    inline void writeBytes(const void* src, size_t length) {
        checkCapacity(length);
        if (length > 0) {
            std::memcpy(data_ + position_, src, length);
            position_ += length;
        }
    }

    // Writes the contents of the source vector 'src'.
    inline void writeBytes(const std::vector<uint8_t>& src) {
        const size_t length = src.size();
        checkCapacity(length);
        if (length > 0) {
            std::memcpy(data_ + position_, src.data(), length);
            position_ += length;
        }
    }

    // Writes 'count' elements of type T from the source array 'src'.
    template<typename T>
    inline void writeArray(const T* src, size_t count) {
        const size_t bytesToWrite = count * sizeof(T);
        checkCapacity(bytesToWrite);
        if (bytesToWrite > 0) {
            std::memcpy(data_ + position_, src, bytesToWrite);
            position_ += bytesToWrite;
        }
    }

    // Writes the contents of the source vector 'src'.
    template<typename T>
    inline void writeArray(const std::vector<T>& src) {
        const size_t count = src.size();
        if (count > 0) {
            writeArray(src.data(), count); // Uses checked pointer version
        }
    }

    // Writes a string, prefixed with its uint32 length.
    inline void writeString(const std::string& str) {
        const uint32_t length = static_cast<uint32_t>(str.length());
        // Check capacity for both length prefix and string data
        checkCapacity(sizeof(uint32_t) + length);

        // Write length
        std::memcpy(data_ + position_, &length, sizeof(uint32_t));
        position_ += sizeof(uint32_t);

        // Write string data
        if (length > 0) {
            std::memcpy(data_ + position_, str.data(), length);
            position_ += length;
        }
    }

    // --- Write Methods (Fast - No Capacity Checking) ---
    // WARNING: Using these methods without ensuring sufficient capacity
    // can lead to writing out of bounds and undefined behavior.

    inline void writeByteFast(uint8_t value) {
        data_[position_++] = value;
    }

    inline void writeInt8Fast(int8_t value) {
        data_[position_++] = static_cast<uint8_t>(value);
    }

    inline void writeUInt16Fast(uint16_t value) {
        std::memcpy(data_ + position_, &value, sizeof(uint16_t));
        position_ += sizeof(uint16_t);
    }

    inline void writeInt16Fast(int16_t value) {
        std::memcpy(data_ + position_, &value, sizeof(int16_t));
        position_ += sizeof(int16_t);
    }

    inline void writeUInt32Fast(uint32_t value) {
        std::memcpy(data_ + position_, &value, sizeof(uint32_t));
        position_ += sizeof(uint32_t);
    }

    inline void writeInt32Fast(int32_t value) {
        std::memcpy(data_ + position_, &value, sizeof(int32_t));
        position_ += sizeof(int32_t);
    }

    inline void writeUInt64Fast(uint64_t value) {
        std::memcpy(data_ + position_, &value, sizeof(uint64_t));
        position_ += sizeof(uint64_t);
    }

    inline void writeInt64Fast(int64_t value) {
        std::memcpy(data_ + position_, &value, sizeof(int64_t));
        position_ += sizeof(int64_t);
    }

    inline void writeFloatFast(float value) {
        std::memcpy(data_ + position_, &value, sizeof(float));
        position_ += sizeof(float);
    }

    inline void writeDoubleFast(double value) {
        std::memcpy(data_ + position_, &value, sizeof(double));
        position_ += sizeof(double);
    }

    inline void writeBytesFast(const void* src, size_t length) {
        std::memcpy(data_ + position_, src, length);
        position_ += length;

    }

    inline void writeBytesFast(const std::vector<uint8_t>& src) {
        const size_t length = src.size();
        std::memcpy(data_ + position_, src.data(), length);
        position_ += length;

    }

    template<typename T>
    inline void writeArrayFast(const T* src, size_t count) {
        const size_t bytesToWrite = count * sizeof(T);
        std::memcpy(data_ + position_, src, bytesToWrite);
        position_ += bytesToWrite;

    }

    template<typename T>
    inline void writeArrayFast(const std::vector<T>& src) {
        writeArrayFast(src.data(), src.size()); // Uses fast pointer version
    }


    inline void writeStringFast(const std::string& str)
    {
        const uint32_t length = static_cast<uint32_t>(str.length());

        // Write length (no check)
        std::memcpy(data_ + position_, &length, sizeof(uint32_t));
        position_ += sizeof(uint32_t);

        // Write string data (no check)
        std::memcpy(data_ + position_, str.data(), length);
        position_ += length;
    }
};

template<bool IsBigEndian>
class BinaryReader {
public:
    // Constructor taking a string view to read from
    explicit BinaryReader(const std::string_view& data) noexcept
        : data_ptr(reinterpret_cast<const unsigned char*>(data.data()))
        , data_size(data.size()), position(0) {}

    // Constructor taking a const char* and size_t
    explicit BinaryReader(const char* data, size_t size) noexcept
        : data_ptr(reinterpret_cast<const unsigned char*>(data))
        , data_size(size), position(0)
    {}

    // Allow copying - only copies the pointer and position, not the underlying data
    BinaryReader(const BinaryReader&) = default;
    BinaryReader& operator=(const BinaryReader&) = default;

    // Disable moving since we don't own the data
    BinaryReader(BinaryReader&&) = default;
    BinaryReader& operator=(BinaryReader&&) = default;

    template<typename T>
    inline T read() {
        static_assert(std::is_trivially_copyable<T>::value,
            "Only trivially copyable types are supported");
        T value;
        readBytes(reinterpret_cast<unsigned char*>(&value), sizeof(T));
        return value;
    }

    inline std::string readString() {
        const uint32_t length = read<uint32_t>();
        if (position + length > data_size) {
            throw std::out_of_range("Buffer overflow");
        }
        std::string result(reinterpret_cast<const char*>(data_ptr + position), length);
        position += length;
        return result;
    }

    [[nodiscard]] inline size_t size() const noexcept {
        return data_size;
    }

    [[nodiscard]] inline bool canRead(size_t bytes) const noexcept {
        return position + bytes <= data_size;
    }

    // Reset position to beginning
    inline void reset() noexcept {
        position = 0;
    }

    // Get current position
    [[nodiscard]] inline size_t getPosition() const noexcept {
        return position;
    }

    // Set position (with bounds checking)
    inline void setPosition(size_t newPosition) {
        if (newPosition > data_size) {
            throw std::out_of_range("Position out of range");
        }
        position = newPosition;
    }

private:
    inline void readBytes(unsigned char* dest, size_t size) {
        if (position + size > data_size) {
            throw std::out_of_range("Buffer overflow");
        }

        if constexpr (IsBigEndian) {
            std::reverse_copy(data_ptr + position,
                data_ptr + position + size,
                dest);
        }
        else {
            std::memcpy(dest, data_ptr + position, size);
        }
        position += size;
    }

    const unsigned char* data_ptr;
    size_t data_size;
    size_t position;
};


template<bool IsBigEndian>
class BinaryWriter {
public:
    // Construct with reference to external buffer
    explicit BinaryWriter(std::string& buffer)
        : byte_buffer(buffer) {}

    // No need for move operations since we're using a reference
    BinaryWriter(BinaryWriter&&) = delete;
    BinaryWriter& operator=(BinaryWriter&&) = delete;

    // Disable copying
    BinaryWriter(const BinaryWriter&) = delete;
    BinaryWriter& operator=(const BinaryWriter&) = delete;

    inline void reserve(size_t capacity) {
        byte_buffer.reserve(capacity);
    }

    template<typename T>
    inline void write(const T& value) {
        static_assert(std::is_trivially_copyable<T>::value,
            "Only trivially copyable types are supported");
        writeBytes(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    inline void writeString(std::string_view value) {
        const uint32_t length = static_cast<uint32_t>(value.length());
        write(length);
        byte_buffer.append(value);
    }

    [[nodiscard]] inline size_t size() const noexcept {
        return byte_buffer.size();
    }

    inline void clear() noexcept {
        byte_buffer.clear();
    }

    // Get the written data as string_view
    [[nodiscard]] inline std::string_view data() const noexcept {
        return std::string_view(byte_buffer);
    }

private:
    inline void writeBytes(const char* data, size_t size) {
        if constexpr (IsBigEndian) {
            const size_t old_size = byte_buffer.size();
            byte_buffer.resize(old_size + size);
            std::reverse_copy(data, data + size,
                byte_buffer.begin() + old_size);
        }
        else {
            byte_buffer.append(data, size);
        }
    }

    std::string& byte_buffer;
};

// Specialized types
using LittleEndianReader = BinaryReader<false>;
using BigEndianReader = BinaryReader<true>;
using LittleEndianWriter = BinaryWriter<false>;
using BigEndianWriter = BinaryWriter<true>;

// Test and benchmark functions
void ATMByteBufferRunAllTests();

void ATMByteBufferRunAllBenchmarks();