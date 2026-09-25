cmake_minimum_required(VERSION 3.21)
project(AttomeGames LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ATTOME_GAMES_ROOT "${CMAKE_CURRENT_LIST_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(AttomeAddGame)

option(ATTOME_BUILD_NET "Build the AttomeNet networking library (needs asio + libsodium)" ON)
option(ATTOME_BUILD_TESTS "Build engine unit tests" ON)
option(ATTOME_BUILD_BENCH "Build engine benchmarks" ON)
# Declared here (not only in engine/CMakeLists.txt) so the attome_online
# check below never depends on the subdirectory having run first.
option(ATTOME_BUILD_VOXEL "Build the voxel engine modules (world, models, net2, Vulkan renderer)" ON)

add_subdirectory(engine)
if(ATTOME_BUILD_NET)
  add_subdirectory(engine/net)
endif()

if(ATTOME_BUILD_TESTS)
  enable_testing()
  add_subdirectory(engine/tests)
endif()

if(ATTOME_BUILD_BENCH)
  add_subdirectory(engine/bench)
endif()

option(ATTOME_BUILD_HELLO_WORLD   "Build Hello World"     ON)
option(ATTOME_BUILD_SNAKE         "Build Snake"           ON)
option(ATTOME_BUILD_TOWER_SWARM   "Build Tower Swarm"     ON)
option(ATTOME_BUILD_PLANET_OPTIMIZED "Build Planet Optimized" ON)
option(ATTOME_BUILD_TEMPLATE      "Build the new-game template" ON)
option(ATTOME_BUILD_ASHLANDS      "Build Ashlands Dominion" ON)

if(ATTOME_BUILD_HELLO_WORLD)
  add_subdirectory(hello-world)
endif()

if(ATTOME_BUILD_SNAKE AND ATTOME_BUILD_NET)  # Snake runs the net self-test
  add_subdirectory(snake)
endif()

if(ATTOME_BUILD_TOWER_SWARM)
  add_subdirectory(tower_swarm)
endif()

if(ATTOME_BUILD_PLANET_OPTIMIZED)
  add_subdirectory(planet_optimized)
endif()

if(ATTOME_BUILD_TEMPLATE)
  add_subdirectory(_template)
endif()

if(ATTOME_BUILD_ASHLANDS)
  add_subdirectory(ashlands-dominion)
endif()

option(ATTOME_BUILD_ONLINE "Build Attome Online (voxel MMO demo: client, server, bots)" ON)
if(ATTOME_BUILD_ONLINE AND ATTOME_BUILD_VOXEL)
  add_subdirectory(attome_online)
endif()
