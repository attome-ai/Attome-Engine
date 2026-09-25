add_executable(atm_bench atm_bench.cpp)
target_link_libraries(atm_bench PRIVATE AttomeCore)

if(WIN32)
  add_custom_command(TARGET atm_bench POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_RUNTIME_DLLS:atm_bench> $<TARGET_FILE_DIR:atm_bench>
    COMMAND_EXPAND_LISTS
  )
endif()

# Uses only the original engine API so it can also be built against an older
# engine revision for before/after comparisons.
add_executable(atm_compare compare_legacy.cpp)
target_link_libraries(atm_compare PRIVATE AttomeCore)
