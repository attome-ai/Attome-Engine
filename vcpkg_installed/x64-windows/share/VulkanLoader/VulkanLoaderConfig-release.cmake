#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Vulkan::Loader" for configuration "Release"
set_property(TARGET Vulkan::Loader APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Vulkan::Loader PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/vulkan-1.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/vulkan-1.dll"
  )

list(APPEND _cmake_import_check_targets Vulkan::Loader )
list(APPEND _cmake_import_check_files_for_Vulkan::Loader "${_IMPORT_PREFIX}/lib/vulkan-1.lib" "${_IMPORT_PREFIX}/bin/vulkan-1.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
