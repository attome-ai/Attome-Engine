add_library(AttomeNet STATIC
  ActionBuilder.cpp
  ConnectionSlots.cpp
  CryptoLayer.cpp
  MessageBatcher.cpp
  NetSocket.cpp
  NetSelfTest.cpp
  PacketPool.cpp
  ReliabilityLayer.cpp
  Server.cpp
  SimLayer.cpp
  Client.cpp
)

target_compile_features(AttomeNet PUBLIC cxx_std_20)

if(MSVC)
  # /MP is a no-op under Ninja (each TU is already a separate cl.exe); keep it
  # for MSBuild compatibility. /Z7 avoids shared-PDB lock contention in Debug.
  target_compile_options(AttomeNet PRIVATE /MP $<$<CONFIG:Debug>:/Z7>)
endif()

# Precompile the heavy ASIO headers (coroutine + io machinery) once.
# Without this, each TU reparsed ~200k lines of template code; NetSocket.cpp
# alone was taking ~5 min on a Ryzen 9 because of use_awaitable / awaitable.
target_precompile_headers(AttomeNet PRIVATE
  <asio/awaitable.hpp>
  <asio/buffer.hpp>
  <asio/io_context.hpp>
  <asio/ip/udp.hpp>
  <asio/redirect_error.hpp>
  <asio/steady_timer.hpp>
  <asio/use_awaitable.hpp>
)

get_filename_component(ATTOME_GAMES_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
target_include_directories(AttomeNet PUBLIC "${ATTOME_GAMES_ROOT}")

find_package(asio CONFIG REQUIRED)
target_link_libraries(AttomeNet PUBLIC asio::asio)

if(WIN32)
  target_link_libraries(AttomeNet PRIVATE ws2_32 mswsock)
endif()

option(ATTOME_NET_ENCRYPTION "Enable AttomeNet encryption (NET_ENCRYPTION=1)" OFF)
option(ATTOME_NET_SIM "Enable AttomeNet simulation (NET_SIM=1)" OFF)

if(ATTOME_NET_ENCRYPTION)
  target_compile_definitions(AttomeNet PUBLIC NET_ENCRYPTION=1)
  find_package(unofficial-sodium CONFIG REQUIRED)
  target_link_libraries(AttomeNet PUBLIC unofficial-sodium::sodium)
endif()

if(ATTOME_NET_SIM)
  target_compile_definitions(AttomeNet PUBLIC NET_SIM=1)
endif()
