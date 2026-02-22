cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies via vcpkg
find_package(SDL3 CONFIG REQUIRED)
find_package(SDL3_image CONFIG REQUIRED)
find_package(SDL3_ttf CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

# Common engine sources (Optimized Engine)
set(ENGINE_SOURCES
  "../engine/ATMEngine.cpp"
  "../lib/ATMByteBuffer.cpp"
  "../lib/ATMBufferPool.cpp"
)

# Basic Engine sources (UNOPTIMIZED - for comparison)
set(ENGINE_BASIC_SOURCES
  "../engineBasic/BasicEngine.cpp"
)

# =============================================================================
# Game Executables
# =============================================================================

# 1. Planet Game (Optimized Engine)
add_executable(PlanetGame
  ${CMAKE_SOURCE_DIR}/gameOptimized/planet_game.cpp
  ${ENGINE_SOURCES}
)

# 2. Snake Game (Optimized Engine)
add_executable(SnakeGame
  ${CMAKE_SOURCE_DIR}/gameOptimized/snake_game.cpp
  ${ENGINE_SOURCES}
)

# 3. Planet Game Basic (Unoptimized Engine - for comparison)
add_executable(PlanetGameBasic
  ${CMAKE_SOURCE_DIR}/gameBasic/planet_game_basic.cpp
  ${ENGINE_BASIC_SOURCES}
)

# =============================================================================
# Common Dependencies - Helper Function
# =============================================================================

# Function to link common optimized engine dependencies
function(link_optimized_engine_deps target)
  target_link_libraries(${target} PRIVATE 
    SDL3::SDL3
    SDL3_image::SDL3_image
    SDL3_ttf::SDL3_ttf
    glm::glm
    imgui::imgui
  )
  
  # ImGui backends
  if(TARGET imgui::sdl3)
    target_link_libraries(${target} PRIVATE imgui::sdl3)
  endif()
  if(TARGET imgui::sdlrenderer3)
    target_link_libraries(${target} PRIVATE imgui::sdlrenderer3)
  endif()
  
  # Windows libraries
  target_link_libraries(${target} PRIVATE 
    version.lib
    winmm.lib
    setupapi.lib
    cfgmgr32.lib
    imm32.lib
  )
  
  # Increase stack size for large entity counts
  if(MSVC)
    target_link_options(${target} PRIVATE "/STACK:456777216")
  endif()
  
  # Post-build: Copy assets
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/textures"
      "$<TARGET_FILE_DIR:${target}>/textures"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/fonts"
      "$<TARGET_FILE_DIR:${target}>/fonts"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/resource"
      "$<TARGET_FILE_DIR:${target}>/resource"
  )
endfunction()

# Function to link basic engine dependencies
function(link_basic_engine_deps target)
  target_link_libraries(${target} PRIVATE 
    SDL3::SDL3
    SDL3_image::SDL3_image
    imgui::imgui
  )
  
  # ImGui backends
  if(TARGET imgui::sdl3)
    target_link_libraries(${target} PRIVATE imgui::sdl3)
  endif()
  if(TARGET imgui::sdlrenderer3)
    target_link_libraries(${target} PRIVATE imgui::sdlrenderer3)
  endif()
  
  # Windows libraries
  target_link_libraries(${target} PRIVATE 
    version.lib
    winmm.lib
    setupapi.lib
    cfgmgr32.lib
    imm32.lib
  )
  
  # Stack size for basic engine
  if(MSVC)
    target_link_options(${target} PRIVATE "/STACK:16777216")
  endif()
  
  # Include directories
  target_include_directories(${target} PRIVATE 
    ${CMAKE_SOURCE_DIR}/game
    ${CMAKE_SOURCE_DIR}/include
  )
  
  # Post-build: Copy assets
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/resource"
      "$<TARGET_FILE_DIR:${target}>/resource"
  )
endfunction()

# =============================================================================
# Apply Dependencies to Each Target
# =============================================================================

link_optimized_engine_deps(PlanetGame)
link_optimized_engine_deps(SnakeGame)
link_basic_engine_deps(PlanetGameBasic)

# =============================================================================
# Initial Asset Copy
# =============================================================================

file(COPY "${CMAKE_SOURCE_DIR}/textures" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_SOURCE_DIR}/fonts" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_SOURCE_DIR}/resource" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
