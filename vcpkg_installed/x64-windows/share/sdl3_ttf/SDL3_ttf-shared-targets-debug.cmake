#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SDL3_ttf::SDL3_ttf-shared" for configuration "Debug"
set_property(TARGET SDL3_ttf::SDL3_ttf-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SDL3_ttf::SDL3_ttf-shared PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/SDL3_ttf.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SDL3::SDL3-shared"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/SDL3_ttf.dll"
  )

list(APPEND _cmake_import_check_targets SDL3_ttf::SDL3_ttf-shared )
list(APPEND _cmake_import_check_files_for_SDL3_ttf::SDL3_ttf-shared "${_IMPORT_PREFIX}/debug/lib/SDL3_ttf.lib" "${_IMPORT_PREFIX}/debug/bin/SDL3_ttf.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
