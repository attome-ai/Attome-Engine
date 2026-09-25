# =============================================================================
# AttomeCore — the 2D engine core, built once and linked by every game
# =============================================================================
add_library(AttomeCore STATIC
  ATMEngine.cpp
  ATMJson.cpp
  ATMConfig.cpp
  ATMInput.cpp
  ATMAudio.cpp
  ATMAssets.cpp
)

target_include_directories(AttomeCore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(AttomeCore PUBLIC cxx_std_20)

find_package(SDL3 CONFIG REQUIRED)
target_link_libraries(AttomeCore PUBLIC SDL3::SDL3)

if(WIN32)
  target_compile_definitions(AttomeCore PUBLIC NOMINMAX)
endif()

# Size the spatial grid from EngineConfig at runtime instead of the
# compile-time constants. Off by default: it costs ~5% on entity moves.
option(ATTOME_RUNTIME_GRID "Size the spatial grid from config at runtime" OFF)
if(ATTOME_RUNTIME_GRID)
  target_compile_definitions(AttomeCore PUBLIC ATM_RUNTIME_GRID=1)
endif()

# Text rendering needs SDL3_ttf. It's optional so a game that doesn't draw
# text (or a platform without the library) doesn't have to pull it in.
option(ATTOME_WITH_TEXT "Build ATMText (SDL3_ttf glyph-cached text)" ON)
if(ATTOME_WITH_TEXT)
  find_package(SDL3_ttf CONFIG QUIET)
  if(SDL3_ttf_FOUND)
    target_sources(AttomeCore PRIVATE ATMText.cpp)
    target_link_libraries(AttomeCore PUBLIC SDL3_ttf::SDL3_ttf)
    target_compile_definitions(AttomeCore PUBLIC ATM_HAS_TEXT=1)
  else()
    message(STATUS "AttomeCore: SDL3_ttf not found, ATMText disabled")
  endif()
endif()

# --- voxel MMO engine modules (docs/DEMO_PLAN.md) ---------------------------
option(ATTOME_BUILD_VOXEL "Build the voxel engine modules (world, models, net2, Vulkan renderer)" ON)
if(ATTOME_BUILD_VOXEL)
  add_subdirectory(voxel)
  add_subdirectory(model)
  add_subdirectory(net2)
  add_subdirectory(render)
endif()
