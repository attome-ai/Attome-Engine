cmake_minimum_required(VERSION 3.21)
project(AttomeGames LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ATTOME_GAMES_ROOT "${CMAKE_CURRENT_LIST_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(AttomeAddGame)

add_subdirectory(engine/net)

option(ATTOME_BUILD_HELLO_WORLD   "Build Hello World"     ON)
option(ATTOME_BUILD_SNAKE         "Build Snake"           ON)
option(ATTOME_BUILD_TOWER_SWARM   "Build Tower Swarm"     ON)

if(ATTOME_BUILD_HELLO_WORLD)
  add_subdirectory(hello-world)
endif()

if(ATTOME_BUILD_SNAKE)
  add_subdirectory(snake)
endif()

if(ATTOME_BUILD_TOWER_SWARM)
  add_subdirectory(tower_swarm)
endif()
