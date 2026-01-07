add_executable(SnakeGame src/main.cpp)

target_link_libraries(SnakeGame PRIVATE ATMEngine)

if(MSVC)
  target_link_options(SnakeGame PRIVATE "/STACK:456777216")
endif()


# Copy assets (Using root assets folder or local?)
# For now, copying from root assets to build dir
file(COPY "${CMAKE_SOURCE_DIR}/textures" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_SOURCE_DIR}/fonts" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_SOURCE_DIR}/resource" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
