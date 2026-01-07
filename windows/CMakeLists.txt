cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies via vcpkg
find_package(SDL3 CONFIG REQUIRED)
find_package(SDL3_image CONFIG REQUIRED)
find_package(SDL3_ttf CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
# find_package(Vulkan REQUIRED)
find_package(imgui CONFIG REQUIRED)

# The project sources (Optimized Engine)
set(SOURCE
  ${CMAKE_SOURCE_DIR}/gameSrc/zombie_game.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/ATMEngine.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/stb_impl.cpp
  
  ${CMAKE_SOURCE_DIR}/gameSrc/Renderer.cpp
  
  
  ${CMAKE_SOURCE_DIR}/gameSrc/ATMByteBuffer.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/ATMBufferPool.cpp
  
)

# Basic Engine sources (UNOPTIMIZED - for comparison)
set(SOURCE_BASIC
  ${CMAKE_SOURCE_DIR}/gameBasic/zombie_game_basic.cpp
  ${CMAKE_SOURCE_DIR}/gameBasic/BasicEngine.cpp
  ${CMAKE_SOURCE_DIR}/gameBasic/stb_impl.cpp
  
)

# Add executable target (Optimized)
add_executable(Template ${SOURCE})

# Add executable target (Basic/Unoptimized)
add_executable(TemplateBasic ${SOURCE_BASIC})

# Link dependencies
target_link_libraries(Template PRIVATE 
    SDL3::SDL3
    SDL3_image::SDL3_image
    SDL3_ttf::SDL3_ttf
    glm::glm
    imgui::imgui
)

# Link dependencies for Basic Engine (no Vulkan needed)
target_link_libraries(TemplateBasic PRIVATE 
    SDL3::SDL3
    SDL3_image::SDL3_image
    imgui::imgui
)

# Add ImGui backends (if they are not included in the imgui::imgui target provided by vcpkg)
# With vcpkg manifest mode, we can use the targets for backends if we enabled them as features.
if(TARGET imgui::sdl3)
    target_link_libraries(Template PRIVATE imgui::sdl3)
    target_link_libraries(TemplateBasic PRIVATE imgui::sdl3)
endif()
if(TARGET imgui::sdlrenderer3)
    target_link_libraries(Template PRIVATE imgui::sdlrenderer3)
    target_link_libraries(TemplateBasic PRIVATE imgui::sdlrenderer3)
endif()

# Increase stack size (16MB = 16777216 bytes)
if(MSVC)
  target_link_options(Template PRIVATE "/STACK:456777216")
  target_link_options(TemplateBasic PRIVATE "/STACK:16777216")
endif()

# Additional Windows libraries for Template
target_link_libraries(Template PRIVATE 
    version.lib
    winmm.lib
    setupapi.lib
    cfgmgr32.lib
    imm32.lib
)

# Additional Windows libraries for TemplateBasic
target_link_libraries(TemplateBasic PRIVATE 
    version.lib
    winmm.lib
    setupapi.lib
    cfgmgr32.lib
    imm32.lib
)

# Include directories for BasicEngine
target_include_directories(TemplateBasic PRIVATE 
    ${CMAKE_SOURCE_DIR}/game
    ${CMAKE_SOURCE_DIR}/include
)

# Copy assets to build directory
file(COPY "${CMAKE_SOURCE_DIR}/textures" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_SOURCE_DIR}/fonts" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_SOURCE_DIR}/resource" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# Optional: Add post-build command to ensure assets are always up to date
add_custom_command(TARGET Template POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/textures"
        "$<TARGET_FILE_DIR:Template>/textures"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/fonts"
        "$<TARGET_FILE_DIR:Template>/fonts"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/resource"
        "$<TARGET_FILE_DIR:Template>/resource"
)

# Post-build for TemplateBasic
add_custom_command(TARGET TemplateBasic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/resource"
        "$<TARGET_FILE_DIR:TemplateBasic>/resource"
)

