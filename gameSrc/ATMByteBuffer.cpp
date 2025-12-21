#include "ATMByteBuffer.h"
#include <string>
#include <stdexcept>


void ATMByteBufferRunAllTests() {
    std::cout << "Running all ByteBuffer tests...\n";

    // Test basic read/write
    {
        std::vector<uint8_t> buffer;
        ByteBufferWriter writer(buffer);

        // Write various types
        writer.writeByte(42);
        writer.writeInt8(-42);
        writer.writeUInt16(1000);
        writer.writeInt16(-1000);
        writer.writeUInt32(1000000);
        writer.writeInt32(-1000000);
        writer.writeUInt64(1000000000000);
        writer.writeInt64(-1000000000000);
        writer.writeFloat(3.14159f);
        writer.writeDouble(2.718281828);

        // Write arrays
        std::vector<int32_t> intArray = { 1, 2, 3, 4, 5 };
        writer.writeArray(intArray);

        // Write strings
        writer.writeString("Hello, World!");

        // Read everything back
        ByteBufferReader reader(buffer);

        assert(reader.readByte() == 42);
        assert(reader.readInt8() == -42);
        assert(reader.readUInt16() == 1000);
        assert(reader.readInt16() == -1000);
        assert(reader.readUInt32() == 1000000);
        assert(reader.readInt32() == -1000000);
        assert(reader.readUInt64() == 1000000000000);
        assert(reader.readInt64() == -1000000000000);

        float readFloat = reader.readFloat();
        assert(std::abs(readFloat - 3.14159f) < 1e-5f);

        double readDouble = reader.readDouble();
        assert(std::abs(readDouble - 2.718281828) < 1e-10);

        // Read arrays
        std::vector<int32_t> readIntArray;
        reader.readArray(readIntArray, intArray.size());
        assert(readIntArray == intArray);

        // Read strings
        assert(reader.readString() == "Hello, World!");

        std::cout << "Basic read/write test passed!\n";
    }

    // Test Fast methods
    {
        std::vector<uint8_t> buffer(1024); // Pre-allocate buffer
        ByteBufferWriter writer(buffer);

        // Write various types using fast methods
        writer.writeByteFast(42);
        writer.writeInt8Fast(-42);
        writer.writeUInt16Fast(1000);
        writer.writeInt16Fast(-1000);
        writer.writeUInt32Fast(1000000);
        writer.writeInt32Fast(-1000000);
        writer.writeUInt64Fast(1000000000000);
        writer.writeInt64Fast(-1000000000000);
        writer.writeFloatFast(3.14159f);
        writer.writeDoubleFast(2.718281828);

        // Write arrays using fast methods
        std::vector<int32_t> intArray = { 1, 2, 3, 4, 5 };
        writer.writeArrayFast(intArray);

        // Write strings using fast methods
        writer.writeStringFast("Hello, World!");

        // Read everything back using fast methods
        ByteBufferReader reader(buffer);

        assert(reader.readByteFast() == 42);
        assert(reader.readInt8Fast() == -42);
        assert(reader.readUInt16Fast() == 1000);
        assert(reader.readInt16Fast() == -1000);
        assert(reader.readUInt32Fast() == 1000000);
        assert(reader.readInt32Fast() == -1000000);
        assert(reader.readUInt64Fast() == 1000000000000);
        assert(reader.readInt64Fast() == -1000000000000);

        float readFloat = reader.readFloatFast();
        assert(std::abs(readFloat - 3.14159f) < 1e-5f);

        double readDouble = reader.readDoubleFast();
        assert(std::abs(readDouble - 2.718281828) < 1e-10);

        // Read arrays using fast methods
        std::vector<int32_t> readIntArray;
        reader.readArrayFast(readIntArray, intArray.size());
        assert(readIntArray == intArray);

        // Read strings using fast methods
        assert(reader.readStringFast() == "Hello, World!");

        std::cout << "Fast methods test passed!\n";
    }

    // Test edge cases
    {
        // Test empty buffer
        std::vector<uint8_t> emptyBuffer;
        ByteBufferReader reader(emptyBuffer);
        assert(!reader.hasRemaining());
        assert(reader.remaining() == 0);

        // Test position setting
        std::vector<uint8_t> buffer = { 1, 2, 3, 4, 5 };
        ByteBufferReader posReader(buffer);
        posReader.position(3);
        assert(posReader.position() == 3);
        assert(posReader.readByte() == 4);

        // Test reset
        posReader.reset();
        assert(posReader.position() == 0);
        assert(posReader.readByte() == 1);

        // Test  of bounds exception


        std::cout << "Edge cases test passed!\n";
    }

    std::cout << "All tests passed successfully!\n";
}

void ATMByteBufferRunAllBenchmarks() {
    std::cout << "Running all ByteBuffer benchmarks...\n";

    constexpr size_t BENCHMARK_ITERATIONS = 1000000;
    constexpr size_t LARGE_BENCHMARK_ITERATIONS = 100;
    constexpr size_t LARGE_SIZE = 1024 * 1024; // 1MB

    std::random_device rd;
    std::mt19937 gen(rd());

    // Benchmark simple types
    {
        std::vector<uint8_t> buffer(BENCHMARK_ITERATIONS * 8); // Pre-allocate buffer
        ByteBufferWriter writer(buffer);

        // Benchmark writing
        auto startTime = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < BENCHMARK_ITERATIONS; ++i) {
            writer.writeInt32Fast(static_cast<int32_t>(i));
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

        std::cout << "Write int32 (fast) - " << BENCHMARK_ITERATIONS << " iterations: "
            << duration << " microseconds ("
            << static_cast<double>(BENCHMARK_ITERATIONS) / (duration / 1000000.0) << " ops/sec)\n";

        // Benchmark reading
        ByteBufferReader reader(buffer);

        startTime = std::chrono::high_resolution_clock::now();

        int32_t sum = 0; // Prevent optimization
        for (size_t i = 0; i < BENCHMARK_ITERATIONS; ++i) {
            sum += reader.readInt32Fast();
        }

        endTime = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

        std::cout << "Read int32 (fast) - " << BENCHMARK_ITERATIONS << " iterations: "
            << duration << " microseconds ("
            << static_cast<double>(BENCHMARK_ITERATIONS) / (duration / 1000000.0) << " ops/sec)\n";
    }

    // Benchmark blocks
    {
        std::vector<uint8_t> sourceData(LARGE_SIZE);
        std::generate(sourceData.begin(), sourceData.end(), [&gen]() { return static_cast<uint8_t>(gen() % 256); });

        std::vector<uint8_t> buffer(LARGE_SIZE * LARGE_BENCHMARK_ITERATIONS);
        ByteBufferWriter writer(buffer);

        // Benchmark writing blocks
        auto startTime = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < LARGE_BENCHMARK_ITERATIONS; ++i) {
            writer.writeBytesFast(sourceData);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

        std::cout << "Write " << LARGE_SIZE << " bytes (fast) - " << LARGE_BENCHMARK_ITERATIONS << " iterations: "
            << duration << " microseconds ("
            << static_cast<double>(LARGE_BENCHMARK_ITERATIONS) / (duration / 1000000.0) << " ops/sec)\n";

        // Benchmark reading blocks
        ByteBufferReader reader(buffer);

        startTime = std::chrono::high_resolution_clock::now();

        std::vector<uint8_t> readData;
        for (size_t i = 0; i < LARGE_BENCHMARK_ITERATIONS; ++i) {
            reader.readBytesFast(readData, LARGE_SIZE);
        }

        endTime = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

        std::cout << "Read " << LARGE_SIZE << " bytes (fast) - " << LARGE_BENCHMARK_ITERATIONS << " iterations: "
            << duration << " microseconds ("
            << static_cast<double>(LARGE_BENCHMARK_ITERATIONS) / (duration / 1000000.0) << " ops/sec)\n";
    }

    // Benchmark comparison: Standard vs Fast methods
    {
        std::vector<uint8_t> buffer(BENCHMARK_ITERATIONS * 8);
        ByteBufferWriter writer(buffer);

        // Benchmark standard write
        auto startTime = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < BENCHMARK_ITERATIONS; ++i) {
            writer.writeInt32(static_cast<int32_t>(i));
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

        std::cout << "Write int32 (standard) - " << BENCHMARK_ITERATIONS << " iterations: "
            << duration << " microseconds ("
            << static_cast<double>(BENCHMARK_ITERATIONS) / (duration / 1000000.0) << " ops/sec)\n";

        // Reset
        writer.reset();

        // Benchmark fast write
        startTime = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < BENCHMARK_ITERATIONS; ++i) {
            writer.writeInt32Fast(static_cast<int32_t>(i));
        }

        endTime = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

        std::cout << "Write int32 (fast) - " << BENCHMARK_ITERATIONS << " iterations: "
            << duration << " microseconds ("
            << static_cast<double>(BENCHMARK_ITERATIONS) / (duration / 1000000.0) << " ops/sec)\n";
    }

    std::cout << "All benchmarks completed.\n";
}