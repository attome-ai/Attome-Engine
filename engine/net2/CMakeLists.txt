# =============================================================================
# AttomeNet2 — UDP transport (Host), bit streams and the X-macro schema
# (docs/NETWORK_PLAN.md). Headers are included by relative path
# (../../engine/net2/...) or as <net2/...> via the engine include dir.
# =============================================================================
add_library(AttomeNet2 STATIC
  Net.cpp
  Socket.cpp
)

target_include_directories(AttomeNet2 PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_features(AttomeNet2 PUBLIC cxx_std_20)

find_package(Threads REQUIRED)
find_package(glm CONFIG REQUIRED) # Schema.h Pos/Vel codecs use glm vectors
target_link_libraries(AttomeNet2 PUBLIC Threads::Threads glm::glm)

if(WIN32)
  target_link_libraries(AttomeNet2 PUBLIC ws2_32)
  target_compile_definitions(AttomeNet2 PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

if(MSVC)
  target_compile_options(AttomeNet2 PRIVATE /W4 /permissive-)
else()
  target_compile_options(AttomeNet2 PRIVATE -Wall -Wextra)
endif()
