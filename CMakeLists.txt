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
if(ANDROID)
    _add_library(AttomeEngine STATIC
        "${CMAKE_CURRENT_SOURCE_DIR}/gameSrc/ATMEngine.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gameSrc/ATMByteBuffer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gameSrc/ATMBufferPool.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gameSrc/iconv_stub.c"
    )
else()
    _add_library(AttomeEngine STATIC
        "${CMAKE_CURRENT_SOURCE_DIR}/gameSrc/ATMEngine.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gameSrc/ATMByteBuffer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gameSrc/ATMBufferPool.cpp"
    )
endif()

target_include_directories(AttomeEngine PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/game"
    "${CMAKE_CURRENT_SOURCE_DIR}/gameSrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
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
macro(attome_add_game TARGET_NAME)
    set(_atg_SOURCES "")
    foreach(_src IN ITEMS ${ARGN})
        get_filename_component(_abs_src "${_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        list(APPEND _atg_SOURCES "${_abs_src}")
    endforeach()

    if(ANDROID)
        # Android: build as shared library — SDL3's Java glue loads libmain.so
        # Use _add_library to bypass vcpkg's wrapper (which mis-reads ARGC/ARGV)
        _add_library(${TARGET_NAME} SHARED ${_atg_SOURCES})
        set_target_properties(${TARGET_NAME} PROPERTIES
            OUTPUT_NAME "main"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/jniLibs/${TARGET_NAME}/${ANDROID_ABI}"
        )
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets")
            set(_atg_ASSETS_DST "${CMAKE_SOURCE_DIR}/platforms/android/app/src/main/assets")
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${CMAKE_CURRENT_SOURCE_DIR}/assets"
                    "${_atg_ASSETS_DST}"
                COMMENT "Copying assets to Android project"
            )
        endif()
    elseif(EMSCRIPTEN)
        add_executable(${TARGET_NAME} ${_atg_SOURCES})
        set(_atg_WEB_FLAGS "-s USE_SDL=3 -s ALLOW_MEMORY_GROWTH=1")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/platforms/web/shell.html")
            set(_atg_WEB_FLAGS "${_atg_WEB_FLAGS} --shell-file ${CMAKE_CURRENT_SOURCE_DIR}/platforms/web/shell.html")
        else()
            set(_atg_WEB_FLAGS "${_atg_WEB_FLAGS} --shell-file ${CMAKE_SOURCE_DIR}/_template/platforms/web/shell.html")
        endif()
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets")
            set(_atg_WEB_FLAGS "${_atg_WEB_FLAGS} --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@/assets")
        endif()
        set_target_properties(${TARGET_NAME} PROPERTIES
            SUFFIX ".html"
            LINK_FLAGS "${_atg_WEB_FLAGS}"
        )
    else()
        add_executable(${TARGET_NAME} ${_atg_SOURCES})
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
endmacro()

# ── game projects ──────────────────────────────────────────────────────────────
# Each new game gets its own line here. Run Scripts/new-game.ps1 to add one.
add_subdirectory(hello-world)
add_subdirectory(snake)
# add_subdirectory(planet_basic)    # TODO: source files not yet created
# add_subdirectory(planet_optimized) # TODO: source files not yet created
# add_subdirectory(my-next-game)   ← new-game.ps1 appends here
