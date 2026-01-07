cmake_minimum_required(VERSION 3.21)
project(ATMEngineProject)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IFs:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$CONFIG:Debug,RelWithDebInfo:EditAndContinue>,$<$CONFIG:Debug,RelWithDebInfo:ProgramDatabase>>")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies via vcpkg (Global)
find_package(SDL3 CONFIG REQUIRED)
find_package(SDL3_image CONFIG REQUIRED)
find_package(SDL3_ttf CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

# Subdirectories
add_subdirectory(engine)
add_subdirectory(games/SnakeGame)
add_subdirectory(games/PlantGame)