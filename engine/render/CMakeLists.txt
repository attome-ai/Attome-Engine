# =============================================================================
# AttomeRender — Vulkan 1.3 voxel renderer (docs/GPU_3D_PLAN.md §4, §9, §9A)
#   Renderer.h is the only public header; Vulkan stays inside this target.
#   GLSL shaders in shaders/ are compiled to SPIR-V at build time and embedded
#   as uint32_t arrays (no shader files needed at runtime).
# =============================================================================

# Vulkan::Vulkan (loader) + headers. No COMPONENTS: that needs CMake >= 3.24
# (root requires 3.21) and would hard-fail when only vcpkg (no SDK) is used.
find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
if(NOT TARGET SDL3::SDL3)
  find_package(SDL3 CONFIG REQUIRED)
endif()

# --- shader compiler ------------------------------------------------------------
# Vulkan SDK (FindVulkan sets Vulkan_GLSLANG_VALIDATOR_EXECUTABLE) or vcpkg's
# glslang[tools] (installed under <installed>/<triplet>/tools/glslang).
set(ATM_GLSLANG "")
if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
  set(ATM_GLSLANG "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}")
else()
  set(_atm_glslang_hints "")
  if(DEFINED ENV{VULKAN_SDK})
    list(APPEND _atm_glslang_hints "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
  endif()
  if(DEFINED VCPKG_INSTALLED_DIR)
    foreach(_t IN ITEMS "${VCPKG_HOST_TRIPLET}" "${VCPKG_TARGET_TRIPLET}")
      if(_t)
        list(APPEND _atm_glslang_hints "${VCPKG_INSTALLED_DIR}/${_t}/tools/glslang")
      endif()
    endforeach()
  endif()
  find_program(ATM_GLSLANG_VALIDATOR NAMES glslangValidator glslang HINTS ${_atm_glslang_hints})
  if(ATM_GLSLANG_VALIDATOR)
    set(ATM_GLSLANG "${ATM_GLSLANG_VALIDATOR}")
  endif()
endif()
if(NOT ATM_GLSLANG)
  message(FATAL_ERROR "AttomeRender: glslangValidator not found. Install the Vulkan SDK "
                      "(VULKAN_SDK) or add \"glslang\" with feature \"tools\" to vcpkg.json.")
endif()
message(STATUS "AttomeRender: shader compiler ${ATM_GLSLANG}")

# --- shaders -> SPIR-V -> embedded headers -----------------------------------
set(ATM_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(ATM_SHADER_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders_gen")
file(MAKE_DIRECTORY "${ATM_SHADER_GEN_DIR}")

file(GLOB ATM_SHADER_SOURCES CONFIGURE_DEPENDS
  "${ATM_SHADER_DIR}/*.vert" "${ATM_SHADER_DIR}/*.frag" "${ATM_SHADER_DIR}/*.comp")
file(GLOB ATM_SHADER_INCLUDES CONFIGURE_DEPENDS "${ATM_SHADER_DIR}/*.glsl")

set(ATM_SHADER_HEADERS "")
foreach(src IN LISTS ATM_SHADER_SOURCES)
  get_filename_component(name "${src}" NAME_WE)
  get_filename_component(ext "${src}" EXT)
  string(SUBSTRING "${ext}" 1 -1 ext)             # ".vert" -> "vert"
  set(var "kSpv_${name}_${ext}")
  set(spv "${ATM_SHADER_GEN_DIR}/${name}.${ext}.spv")
  set(hdr "${ATM_SHADER_GEN_DIR}/${name}_${ext}.spv.h")
  add_custom_command(
    OUTPUT "${hdr}"
    BYPRODUCTS "${spv}"
    COMMAND "${ATM_GLSLANG}" -V --target-env vulkan1.3
            "-I${ATM_SHADER_DIR}" -o "${spv}" "${src}"
    COMMAND ${CMAKE_COMMAND} -DINPUT=${spv} -DOUTPUT=${hdr} -DVAR=${var}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/EmbedSpirv.cmake"
    DEPENDS "${src}" ${ATM_SHADER_INCLUDES} "${CMAKE_CURRENT_SOURCE_DIR}/EmbedSpirv.cmake"
    COMMENT "GLSL -> SPIR-V: ${name}.${ext}"
    VERBATIM)
  list(APPEND ATM_SHADER_HEADERS "${hdr}")
endforeach()

add_library(AttomeRender STATIC
  Renderer.h
  Renderer.cpp
  RendererImpl.h
  RendererGeometry.cpp
  RendererFrame.cpp
  FaceAllocator.h
  FaceAllocator.cpp
  Culling.h
  Culling.cpp
  GpuLayout.h
  vk/VkCommon.h
  vk/VkContext.h
  vk/VkContext.cpp
  vk/Swapchain.h
  vk/Swapchain.cpp
  vk/Resources.h
  vk/Resources.cpp
  vk/ShaderLibrary.h
  vk/ShaderLibrary.cpp
  vk/ImGuiLayer.h
  vk/ImGuiLayer.cpp
  vk/VmaImpl.cpp
  ${ATM_SHADER_HEADERS}
  ${ATM_SHADER_SOURCES}
  ${ATM_SHADER_INCLUDES})

target_include_directories(AttomeRender
  PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${ATM_SHADER_GEN_DIR}")
target_compile_features(AttomeRender PUBLIC cxx_std_20)

# Vulkan prototypes + the loader for every TU (see vk/VkCommon.h for why volk
# is not used: the prebuilt imgui_impl_vulkan calls prototypes directly).
if(WIN32)
  target_compile_definitions(AttomeRender PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

target_link_libraries(AttomeRender
  PUBLIC  AttomeVoxel glm::glm
  PRIVATE Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator vk-bootstrap::vk-bootstrap
          SDL3::SDL3 imgui::imgui)

if(MSVC)
  target_compile_options(AttomeRender PRIVATE /W4 /permissive- /bigobj)
else()
  target_compile_options(AttomeRender PRIVATE -Wall -Wextra)
endif()
# The third-party implementation TU is noisy; keep warnings for our code only.
if(MSVC)
  set_source_files_properties(vk/VmaImpl.cpp PROPERTIES COMPILE_OPTIONS "/W0")
else()
  set_source_files_properties(vk/VmaImpl.cpp PROPERTIES COMPILE_OPTIONS "-w")
endif()
source_group("shaders" FILES ${ATM_SHADER_SOURCES} ${ATM_SHADER_INCLUDES})
