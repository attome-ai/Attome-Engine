cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies via vcpkg
find_package(SDL3 CONFIG REQUIRED)
find_package(SDL3_image CONFIG REQUIRED)
find_package(SDL3_ttf CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)


# The project sources (Optimized Engine)
file(GLOB SOURCE "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Add executable target
add_executable(Template ${SOURCE})

# Explicitly add include directories to target
target_include_directories(Template PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/vendor")

# Link dependencies
target_link_libraries(Template PRIVATE 
    SDL3::SDL3
    SDL3_image::SDL3_image
    SDL3_ttf::SDL3_ttf
    glm::glm
    imgui::imgui
)

# Add ImGui backends
if(TARGET imgui::sdl3)
    target_link_libraries(Template PRIVATE imgui::sdl3)
endif()
if(TARGET imgui::sdlrenderer3)
    target_link_libraries(Template PRIVATE imgui::sdlrenderer3)
endif()

# Increase stack size (16MB = 16777216 bytes)
if(MSVC)
  target_link_options(Template PRIVATE "/STACK:456777216")
endif()

# Additional Windows libraries for Template
target_link_libraries(Template PRIVATE 
    version.lib
    winmm.lib
    setupapi.lib
    cfgmgr32.lib
    imm32.lib
)

# Copy assets to build directory
file(COPY "${CMAKE_SOURCE_DIR}/shaders" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_SOURCE_DIR}/textures" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_SOURCE_DIR}/fonts" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_SOURCE_DIR}/resource" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# Optional: Add post-build command to ensure assets are always up to date
add_custom_command(TARGET Template POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/shaders"
        "$<TARGET_FILE_DIR:Template>/shaders"
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

