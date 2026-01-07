add_executable(PlantGame src/main.cpp)

target_link_libraries(PlantGame PRIVATE ATMEngine)

if(MSVC)
  target_link_options(PlantGame PRIVATE "/STACK:456777216")
endif()


# Copy assets
file(COPY "${CMAKE_SOURCE_DIR}/textures" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_SOURCE_DIR}/fonts" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_SOURCE_DIR}/resource" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
