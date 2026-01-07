cmake_minimum_required(VERSION 3.21)
project(Template)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IFs:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$CONFIG:Debug,RelWithDebInfo:EditAndContinue>,$<$CONFIG:Debug,RelWithDebInfo:ProgramDatabase>>")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Project include directories
include_directories("include")
include_directories("vendor")



# Build the appropriate target based on platform
if(EMSCRIPTEN)
  add_subdirectory("html")
else()
  add_subdirectory("windows")
endif()