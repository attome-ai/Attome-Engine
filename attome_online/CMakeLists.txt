# =============================================================================
# Attome Online — voxel MMO demo (docs/DEMO_PLAN.md)
#   ao_shared   game rules shared by client, server and bots
#   ao_serverlib / ao_server   authoritative zone server (library + exe)
#   ao_client   game client (Vulkan); --local runs ao_serverlib in-process
#   ao_bots     headless bot clients for load tests
#   ao_tests    unit tests
# =============================================================================

file(GLOB_RECURSE AO_SHARED_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shared/*.cpp)
add_library(ao_shared STATIC ${AO_SHARED_SRC})
target_include_directories(ao_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ao_shared PUBLIC AttomeVoxel AttomeModel AttomeNet2 AttomeCore)
target_compile_features(ao_shared PUBLIC cxx_std_20)

file(GLOB AO_SERVER_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/server/*.cpp)
list(FILTER AO_SERVER_SRC EXCLUDE REGEX ".*/server_main\\.cpp$")
add_library(ao_serverlib STATIC ${AO_SERVER_SRC})
target_link_libraries(ao_serverlib PUBLIC ao_shared)

add_executable(ao_server ${CMAKE_CURRENT_SOURCE_DIR}/server/server_main.cpp)
target_link_libraries(ao_server PRIVATE ao_serverlib)

file(GLOB_RECURSE AO_CLIENT_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/client/*.cpp)
add_executable(ao_client ${AO_CLIENT_SRC})
# The client uses ImGui directly (HUD, imgui_impl_sdl3); AttomeRender links it
# PRIVATE, which does not propagate include directories.
find_package(imgui CONFIG REQUIRED)
target_include_directories(ao_client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/client)
target_link_libraries(ao_client PRIVATE ao_serverlib AttomeRender imgui::imgui)

file(GLOB AO_BOTS_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bots/*.cpp)
add_executable(ao_bots ${AO_BOTS_SRC})
target_link_libraries(ao_bots PRIVATE ao_shared)

if(ATTOME_BUILD_TESTS)
  file(GLOB AO_TEST_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
  add_executable(ao_tests ${AO_TEST_SRC}
    ${CMAKE_SOURCE_DIR}/engine/tests/test_main.cpp)
  target_include_directories(ao_tests PRIVATE ${CMAKE_SOURCE_DIR}/engine/tests)
  target_link_libraries(ao_tests PRIVATE ao_serverlib)
  add_test(NAME ao_tests COMMAND ao_tests)
endif()

foreach(t ao_shared ao_serverlib ao_server ao_client ao_bots)
  if(MSVC)
    target_compile_options(${t} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${t} PRIVATE -Wall -Wextra)
  endif()
endforeach()

# Runtime data next to the executables.
foreach(t ao_client ao_server ao_bots)
  add_custom_command(TARGET ${t} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_SOURCE_DIR}/config" "$<TARGET_FILE_DIR:${t}>/config")
  add_custom_command(TARGET ${t} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_SOURCE_DIR}/data" "$<TARGET_FILE_DIR:${t}>/data")
endforeach()
add_custom_command(TARGET ao_client POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/assets" "$<TARGET_FILE_DIR:ao_client>/assets")
