add_executable(atm_tests
  test_main.cpp
  test_core.cpp
  test_systems.cpp
)
target_link_libraries(atm_tests PRIVATE AttomeCore)
target_compile_definitions(atm_tests PRIVATE
  ATM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../resource"
  ATM_TEST_FONT="${CMAKE_CURRENT_SOURCE_DIR}/../../snake/assets/fonts/AtomicMd.ttf"
)

if(MSVC)
  target_compile_options(atm_tests PRIVATE /W4)
else()
  target_compile_options(atm_tests PRIVATE -Wall -Wextra)
endif()

# Copy SDL runtime DLLs next to the test binary on Windows so ctest can run it.
if(WIN32)
  add_custom_command(TARGET atm_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_RUNTIME_DLLS:atm_tests> $<TARGET_FILE_DIR:atm_tests>
    COMMAND_EXPAND_LISTS
  )
endif()

add_test(NAME atm_tests COMMAND atm_tests)
