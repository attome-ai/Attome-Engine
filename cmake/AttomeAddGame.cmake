function(attome_add_game target_name)
  add_executable(${target_name} ${ARGN})

  target_include_directories(${target_name} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )

  target_link_libraries(${target_name} PRIVATE AttomeCore)
  if(TARGET AttomeNet)
    target_link_libraries(${target_name} PRIVATE AttomeNet)
  endif()

  # Ship the game's JSON config (editable without rebuilding) and assets next
  # to the executable.
  foreach(data_dir config assets)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${data_dir}")
      add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/${data_dir}"
          "$<TARGET_FILE_DIR:${target_name}>/${data_dir}"
      )
    endif()
  endforeach()

  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()
