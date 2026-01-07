# Engine Library
file(GLOB ENGINE_SOURCES "src/*.cpp")

add_library(ATMEngine STATIC ${ENGINE_SOURCES})

# Include Directories (Exported to consumers)
target_include_directories(ATMEngine PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/vendor"
)

# Link Dependencies (Public so games inherit them)
target_link_libraries(ATMEngine PUBLIC
    SDL3::SDL3
    SDL3_image::SDL3_image
    SDL3_ttf::SDL3_ttf
    glm::glm
    imgui::imgui
)

if(WIN32)
    target_link_libraries(ATMEngine PUBLIC 
        version.lib
        winmm.lib
        setupapi.lib
        cfgmgr32.lib
        imm32.lib
    )
endif()

# ImGui bindings
if(TARGET imgui::sdl3)
    target_link_libraries(ATMEngine PUBLIC imgui::sdl3)
endif()
if(TARGET imgui::sdlrenderer3)
    target_link_libraries(ATMEngine PUBLIC imgui::sdlrenderer3)
endif()
