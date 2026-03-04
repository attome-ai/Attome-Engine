# =============================================================================
# GAMES WORKSPACE — Umbrella CMakeLists
# Add every new game project as its own add_subdirectory() call below.
# The engine library is built once and shared by all games.
# =============================================================================
cmake_minimum_required(VERSION 3.21)
project(AttomeGames)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── vcpkg / toolchain detection ────────────────────────────────────────────────
# When building for web (Emscripten), invoke CMake via:
#   emcmake cmake ..
# When building for desktop, use the vcpkg toolchain file:
#   cmake .. -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
if(EMSCRIPTEN)
    message(STATUS "[AttomeGames] Building for Web (Emscripten / WASM)")
elseif(ANDROID)
    message(STATUS "[AttomeGames] Building for Android (ABI: ${ANDROID_ABI}, API: ${ANDROID_PLATFORM})")
else()
    message(STATUS "[AttomeGames] Building for ${CMAKE_SYSTEM_NAME}")
endif()

# ── find shared packages (used by the engine library) ─────────────────────────
find_package(SDL3       CONFIG REQUIRED)
find_package(SDL3_image CONFIG REQUIRED)
find_package(SDL3_ttf   CONFIG REQUIRED)
find_package(glm        CONFIG REQUIRED)
find_package(imgui      CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# ── engine static library ──────────────────────────────────────────────────────
set(ATM_ENGINE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/engine/engine/ATMEngine.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/engine/lib/ATMByteBuffer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/engine/lib/ATMBufferPool.cpp"
)

add_library(AttomeEngine STATIC ${ATM_ENGINE_SOURCES})

target_include_directories(AttomeEngine PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/engine/engine"
    "${CMAKE_CURRENT_SOURCE_DIR}/engine/game"
)

target_link_libraries(AttomeEngine PUBLIC
    SDL3::SDL3
    SDL3_image::SDL3_image
    SDL3_ttf::SDL3_ttf
    glm::glm
    imgui::imgui
    nlohmann_json::nlohmann_json
)

foreach(_backend IN ITEMS imgui::sdl3 imgui::sdlrenderer3)
    if(TARGET ${_backend})
        target_link_libraries(AttomeEngine PUBLIC ${_backend})
    endif()
endforeach()

if(WIN32 AND NOT EMSCRIPTEN)
    target_link_libraries(AttomeEngine PUBLIC
        version.lib winmm.lib setupapi.lib cfgmgr32.lib imm32.lib
    )
endif()

if(ANDROID)
    # Android system libs required by SDL3
    target_link_libraries(AttomeEngine PUBLIC log android)
endif()

if(MSVC)
    target_link_options(AttomeEngine PUBLIC "/STACK:67108864")
endif()

# ── helper function used by individual game CMakeLists ────────────────────────
# Usage: attome_add_game(MyGame src/main.cpp src/player.cpp)
function(attome_add_game TARGET_NAME)
    set(SOURCES ${ARGN})

    if(ANDROID)
        # Android: build as shared library — SDL3's Java glue loads libmain.so
        add_library(${TARGET_NAME} SHARED ${SOURCES})
        set_target_properties(${TARGET_NAME} PROPERTIES
            OUTPUT_NAME "main"        # produces libmain.so as SDL expects
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/jniLibs/${ANDROID_ABI}"
        )
        # Copy assets to a known path so Gradle can pick them up
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets")
            set(ANDROID_ASSETS_DST "${CMAKE_SOURCE_DIR}/platforms/android/app/src/main/assets")
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${CMAKE_CURRENT_SOURCE_DIR}/assets"
                    "${ANDROID_ASSETS_DST}"
                COMMENT "Copying assets to Android project: ${ANDROID_ASSETS_DST}"
            )
        endif()
    elseif(EMSCRIPTEN)
        add_executable(${TARGET_NAME} ${SOURCES})
        set(WEB_LINK_FLAGS "-s USE_SDL=3 -s ALLOW_MEMORY_GROWTH=1")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/platforms/web/shell.html")
            set(WEB_LINK_FLAGS "${WEB_LINK_FLAGS} --shell-file ${CMAKE_CURRENT_SOURCE_DIR}/platforms/web/shell.html")
        else()
            set(WEB_LINK_FLAGS "${WEB_LINK_FLAGS} --shell-file ${CMAKE_SOURCE_DIR}/_template/platforms/web/shell.html")
        endif()

        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets")
            set(WEB_LINK_FLAGS "${WEB_LINK_FLAGS} --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@/assets")
        endif()

        set_target_properties(${TARGET_NAME} PROPERTIES
            SUFFIX ".html"
            LINK_FLAGS "${WEB_LINK_FLAGS}"
        )
    else()
        add_executable(${TARGET_NAME} ${SOURCES})
        # Copy assets next to the executable after build
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets")
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${CMAKE_CURRENT_SOURCE_DIR}/assets"
                    "$<TARGET_FILE_DIR:${TARGET_NAME}>/assets"
            )
        endif()
    endif()

    target_link_libraries(${TARGET_NAME} PRIVATE AttomeEngine)
    target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
endfunction()

# ── game projects ──────────────────────────────────────────────────────────────
# Each new game gets its own line here. Run Scripts/new-game.ps1 to add one.
add_subdirectory(hello-world)
add_subdirectory(snake)
add_subdirectory(planet_basic)
add_subdirectory(planet_optimized)
# add_subdirectory(my-next-game)   ← new-game.ps1 appends here
