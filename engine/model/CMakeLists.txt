# AttomeModel — modular voxel characters: rig, animation, parts, demo content
add_library(AttomeModel STATIC
  Rig.cpp
  Animation.cpp
  AnimClips.cpp
  ModelLibrary.cpp
  ModelContent.cpp
  ModelMonsters.cpp
  ModelProps.cpp
  VoxelCodec.cpp
)
target_include_directories(AttomeModel PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_features(AttomeModel PUBLIC cxx_std_20)
find_package(glm CONFIG REQUIRED)
target_link_libraries(AttomeModel PUBLIC AttomeVoxel glm::glm)
if(MSVC)
  target_compile_options(AttomeModel PRIVATE /W4 /permissive-)
else()
  target_compile_options(AttomeModel PRIVATE -Wall -Wextra)
endif()
