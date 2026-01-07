# Attome Engine Template

A high-performance, modern 2D game engine template built with C++20, targeting Windows and HTML5 (via Emscripten). This project is designed for portability and ease of use, utilizing `vcpkg` for dependency management and `CMake` as the build system.

## 🚀 Core Technologies

- **Graphics Backend**: [SDL3 Renderer](https://wiki.libsdl.org/SDL3/SDL_Renderer) (Cross-platform hardware acceleration)
- **Multimedia Layer**: [SDL3](https://www.libsdl.org/) (Windowing, input, audio)
- **UI System**: [ImGui](https://github.com/ocornut/imgui) (Docking branch with SDL3/SDL_Renderer backends)
- **Math Library**: [GLM](https://github.com/g-truc/glm) (OpenGL Mathematics)
- **Data Serialization**: [nlohmann-json](https://github.com/nlohmann/json)
- **Dependency Management**: [vcpkg](https://vcpkg.io/) (Manifest Mode)
- **Build System**: [CMake](https://cmake.org/) (version 3.21+)

## 🛠️ Prerequisites

- **Compiler**: A C++20 compatible compiler (MSVC 2022, Clang 16+, or GCC 13+)
- **vcpkg**: Should be installed and the `VCPKG_ROOT` environment variable set.
- **CMake**: Version 3.21 or higher.

## 🏗️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/attome-ai/Attome-Engine.git
cd Attome-Engine
```

### 2. Build via Visual Studio (Recommended)
Simply open the project folder in Visual Studio 2022. It will automatically detect the `vcpkg.json` manifest and `CMakeSettings.json`, then download and build all dependencies.

### 3. Build via Command Line
```powershell
# Create build directory
mkdir build
cd build

# Configure with vcpkg toolchain
cmake .. -DCMAKE_TOOLCHAIN_FILE=[PATH_TO_VCPKG]/scripts/buildsystems/vcpkg.cmake

# Build the project
cmake --build . --config Release
```

## 📁 Project Structure

- `gameSrc/`: Core engine and game logic implementation (`.cpp` files).
- `game/`: Header files and engine interfaces (`.h` files).
- `windows/`: Windows-specific CMake configuration and entry points.
- `html/`: Emscripten/Web-specific configuration.
- `textures/`: Asset directory for textures and images.
- `vcpkg.json`: Project dependency manifest.

## 🏎️ Dual-Engine Architecture

This repository contains **two complete versions** of the game engine to demonstrate the power of Data-Oriented Design (DOD):

### 1. Basic Engine (Unoptimized)
*Located in `gameBasic/`*
- **Architecture**: Classic OOP with Array of Structures (AOS).
- **Collision**: Naive O(N^2) checks.
- **Rendering**: Individual draw calls per entity.
- **Performance**: Caps at ~10,000 entities @ 15 FPS.
- **Purpose**: Represents "clean" but slow textbook code.

### 2. ATM Engine (Optimized)
*Located in `gameSrc/`*
- **Architecture**: Data-Oriented with Structure of Arrays (SOA).
- **Optimization Features**:
  - **Spatial Partitioning**: O(1) grid-based collision.
  - **Multithreading**: Parallel entity updates using `std::execution`.
  - **Batch Rendering**: 1 draw call per texture/layer via Texture Atlas.
  - **Zero-Allocation**: No dynamic memory during gameplay.
- **Performance**: Handles **1,000,000+ entities** @ 30-60 FPS.

## 📦 Dependency Management

This project uses **vcpkg Manifest Mode**. All libraries listed in `vcpkg.json` are automatically fetched and built during the CMake configuration step. This ensures that every developer uses the exact same versions of the libraries without manual installation.

---
*Created with ❤️ by the Attome AI team.*
