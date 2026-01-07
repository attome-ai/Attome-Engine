# ATM Engine

A high-performance 2D game engine built with C++20 and SDL3, designed for maximum performance with Data-Oriented Design principles.

## 🚀 Core Technologies

- **Graphics**: SDL3 Software Renderer (batched rendering)
- **Multimedia**: [SDL3](https://www.libsdl.org/) (Windowing, input, audio)
- **UI System**: [ImGui](https://github.com/ocornut/imgui) (SDL3 backend)
- **Math Library**: [GLM](https://github.com/g-truc/glm)
- **Dependency Management**: [vcpkg](https://vcpkg.io/) (Manifest Mode)
- **Build System**: [CMake](https://cmake.org/) (version 3.21+)

## 📁 Project Structure

```
Template3/
├── CMakeLists.txt          # Root build configuration
├── engine/
│   ├── CMakeLists.txt      # Engine static library
│   ├── src/                # Headers + Sources (mixed)
│   │   ├── ATMEngine.h     # Main engine header
│   │   ├── ATMEngine.cpp   # Engine implementation
│   │   └── ...
│   └── vendor/             # Third-party headers (stb, ankerl)
├── games/
│   ├── SnakeGame/          # Snake game example
│   │   └── src/main.cpp
│   └── PlantGame/          # Plant defense game
│       └── src/main.cpp
├── textures/               # Game textures
├── fonts/                  # Font assets
└── resource/               # Additional resources
```

## 🏗️ Building

### Prerequisites
- C++20 compiler (MSVC 2022, Clang 16+, GCC 13+)
- vcpkg installed with `VCPKG_ROOT` set
- CMake 3.21+

### Build Commands
```powershell
# Configure
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

# Build
cmake --build build --config Release

# Run games
./build/games/SnakeGame/Release/SnakeGame.exe
./build/games/PlantGame/Release/PlantGame.exe
```

## ⚡ Performance Features

- **Spatial Partitioning**: O(1) grid-based collision queries
- **Batch Rendering**: Single draw call per Texture Atlas
- **Multithreading**: Parallel entity updates with `std::execution`
- **Structure of Arrays (SOA)**: Cache-friendly memory layout
- **Zero Runtime Allocation**: Memory defined at load time

**Performance**: Handles **1,000,000+ entities** @ ~200 FPS

## 📦 Dependencies (via vcpkg)

- `sdl3`
- `sdl3-image`
- `sdl3-ttf`
- `glm`
- `imgui[sdl3-binding,sdl3-renderer-binding]`

---
*Created with ❤️ by Attome AI*
