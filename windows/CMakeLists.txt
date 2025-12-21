cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies via vcpkg
find_package(SDL3 CONFIG REQUIRED)
find_package(SDL3_image CONFIG REQUIRED)
find_package(SDL3_ttf CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(imgui CONFIG REQUIRED)

# The project sources
set(SOURCE
  ${CMAKE_SOURCE_DIR}/gameSrc/main.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/ATMEngine.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/UI.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/Renderer.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/Networking.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/Entity.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/ATMByteBuffer.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/ATMBufferPool.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/VulkanRenderer.cpp
  ${CMAKE_SOURCE_DIR}/gameSrc/VulkanLoader.cpp
)

# Add executable target
add_executable(Template ${SOURCE})

# Link dependencies
target_link_libraries(Template PRIVATE 
    SDL3::SDL3
    SDL3_image::SDL3_image
    SDL3_ttf::SDL3_ttf
    glm::glm
    Vulkan::Vulkan
    imgui::imgui
)

# Add ImGui backends (if they are not included in the imgui::imgui target provided by vcpkg)
# With vcpkg manifest mode, we can use the targets for backends if we enabled them as features.
if(TARGET imgui::sdl3)
    target_link_libraries(Template PRIVATE imgui::sdl3)
endif()
if(TARGET imgui::vulkan)
    target_link_libraries(Template PRIVATE imgui::vulkan)
endif()
if(TARGET imgui::sdlrenderer3)
    target_link_libraries(Template PRIVATE imgui::sdlrenderer3)
endif()

# Increase stack size (16MB = 16777216 bytes)
if(MSVC)
  target_link_options(Template PRIVATE "/STACK:456777216")
endif()

# Additional Windows libraries
target_link_libraries(Template PRIVATE 
    version.lib
    winmm.lib
    setupapi.lib
    cfgmgr32.lib
    imm32.lib
)


