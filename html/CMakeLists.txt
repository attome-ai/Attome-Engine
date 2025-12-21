cmake_minimum_required(VERSION 3.21)

# Set Emscripten-specific options
set(CMAKE_EXECUTABLE_SUFFIX ".html")

# Set optimization flags for Release builds
set(CMAKE_C_FLAGS_RELEASE "-O3 -flto")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto")

# General C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Wno-c++11-narrowing")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Organize source files
#include_directories("${CMAKE_SOURCE_DIR}/SDL/include")
#include_directories("${CMAKE_SOURCE_DIR}/SDL_image/include")
#include_directories("${CMAKE_SOURCE_DIR}/imgui")   
#include_directories("${CMAKE_SOURCE_DIR}/imgui/backends")


#file(GLOB IMGUI_SOURCES ${CMAKE_SOURCE_DIR}/imgui/*.cpp)

set(SOURCE
  ${CMAKE_SOURCE_DIR}/gameSrc/main.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/ATMEngine.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/UI.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/Renderer.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/Networking.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/Entity.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/ATMByteBuffer.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/ATMBufferPool.cpp
)

# Set the correct path to your static SDL library
set(SDL_LIB_DIR "${CMAKE_SOURCE_DIR}/lib")

# Set output directory explicitly
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out/html")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out/html")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out/html")

# Add SDL as a static library
#add_library(SDL3 STATIC IMPORTED)
#set_target_properties(SDL3 PROPERTIES
#    IMPORTED_LOCATION "${SDL_LIB_DIR}/libSDL3.a"
#)

# Add SDL as a static library
#add_library(SDL3_image STATIC IMPORTED)
#set_target_properties(SDL3_image PROPERTIES
#    IMPORTED_LOCATION "${SDL_LIB_DIR}/libSDL3_image.a"
#)
# Ensure the output directory exists
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/out/html")

# Add executable target with source files
add_executable(${PROJECT_NAME} ${SOURCE})

# Link SDL3 library statically
#target_link_libraries(${PROJECT_NAME} PRIVATE SDL3 SDL3_image)

# Set Emscripten linker flags with optimizations
target_link_options(${PROJECT_NAME} PRIVATE
    "-O3"
    "-flto"
    "-sUSE_SDL=3"
    "-sEXPORTED_RUNTIME_METHODS=ccall,cwrap"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sELIMINATE_DUPLICATE_FUNCTIONS=1"
    "--shell-file=${CMAKE_SOURCE_DIR}/shell.html"
)

# Additional target properties to ensure files go to the right place
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out/html"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out/html"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out/html"
)