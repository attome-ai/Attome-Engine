# =============================================================================
# planet_optimized — Optimized version of the planet game
# =============================================================================
cmake_minimum_required(VERSION 3.21)
project(PlanetOptimized)

attome_add_game(PlanetOptimized
    ../engine/gameOptimized/planet_game.cpp
)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(PlanetOptimized PRIVATE imgui::imgui)

add_custom_command(TARGET PlanetOptimized POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../engine/resource"
        "$<TARGET_FILE_DIR:PlanetOptimized>/resource"
)
