function(attome_add_game target_name)
  if(NOT DEFINED ATTOME_GAMES_ROOT)
    message(FATAL_ERROR "ATTOME_GAMES_ROOT is not set. Include from games/CMakeLists.txt.")
  endif()

  add_executable(${target_name}
    ${ARGN}
    "${ATTOME_GAMES_ROOT}/ATMEngine.cpp"
  )

  target_include_directories(${target_name} PRIVATE
    "${ATTOME_GAMES_ROOT}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )

  find_package(SDL3 CONFIG REQUIRED)
  target_link_libraries(${target_name} PRIVATE SDL3::SDL3)
  target_link_libraries(${target_name} PRIVATE AttomeNet)

  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  if(WIN32)
    target_compile_definitions(${target_name} PRIVATE NOMINMAX)
  endif()
endfunction()
