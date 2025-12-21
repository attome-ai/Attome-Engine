cmake_minimum_required(VERSION 3.21)
project(test)
find_package(SDL3 CONFIG REQUIRED)
message(STATUS "SDL3 found: ${SDL3_FOUND}")
