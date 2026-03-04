# =============================================================================
# app/CMakeLists.txt — Android delegate
#
# This file is referenced by app/build.gradle.kts via externalNativeBuild.
# It delegates to the main games/CMakeLists.txt so all game targets are built
# with the same logic, just cross-compiled for Android.
#
# Gradle passes:
#   -DGAME_NAME=<gameName>   (required)
#   -DGAMES_ROOT=<path>      (path to the games/ directory)
# =============================================================================
cmake_minimum_required(VERSION 3.22)
project(AttomeAndroid)

if(NOT DEFINED GAME_NAME)
    message(FATAL_ERROR "GAME_NAME must be set. Pass -DGAME_NAME=<game> to cmake.")
endif()

if(NOT DEFINED GAMES_ROOT)
    set(GAMES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
    message(STATUS "[Android] Inferred GAMES_ROOT: ${GAMES_ROOT}")
endif()

# Delegate to the main games workspace CMakeLists
# This builds AttomeEngine + the selected game as a shared lib (libmain.so)
add_subdirectory("${GAMES_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/games_build")
