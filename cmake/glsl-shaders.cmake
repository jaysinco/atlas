#
# glslc
#
find_program(GLSLANGVALIDATOR_EXE "glslc")
mark_as_advanced(FORCE GLSLANGVALIDATOR_EXE)
if(GLSLANGVALIDATOR_EXE)
  message(STATUS "Found glslc: ${GLSLANGVALIDATOR_EXE}")
else()
  message(STATUS "Missing glslc!")
endif()

# This function acts much like the 'target_sources' function, as in raw GLSL
# shader files can be passed in and will be compiled using 'glslc',
# provided it is available, where the compiled files will be located where the
# sources files are but with the '.spv' suffix appended.
#
# The first argument is the target that the files are associated with, and will
# be compiled as if it were a source file for it. All provided shaders are also
# only recompiled if the source shader file has been modified since the last
# compilation.
#
# ~~~
# Required:
# TARGET_NAME - Name of the target the shader files are associated with and to be compiled for.
#
# Optional:
# INTERFACE <files> - When the following shader files are added to a target, they are done so as 'INTERFACE' type files
# PUBLIC <files> - When the following shader files are added to a target, they are done so as 'PUBLIC' type files
# PRIVATE <files> - When the following shader files are added to a target, they are done so as 'PRIVATE' type files
# COMPILE_OPTIONS <options> - These are other options passed straight to the 'glslc' call with the source shader file
#
# Example:
# When calling `make vk_lib` the shaders will also be compiled with the library's `.c` files.
#
# add_library(vk_lib lib.c, shader_manager.c)
# target_glsl_shaders(vk_lib
#       PRIVATE test.vert test.frag
#       COMPILE_OPTIONS --target-env vulkan1.1)
# ~~~
function(target_glsl_shaders TARGET_NAME)
  if(NOT GLSLANGVALIDATOR_EXE)
    message(
      FATAL_ERROR "Cannot compile GLSL to SPIR-V is glslc not found!"
    )
  endif()

  set(OPTIONS)
  set(SINGLE_VALUE_KEYWORDS)
  set(MULTI_VALUE_KEYWORDS INTERFACE PUBLIC PRIVATE COMPILE_OPTIONS)
  cmake_parse_arguments(
    target_glsl_shaders "${OPTIONS}" "${SINGLE_VALUE_KEYWORDS}"
    "${MULTI_VALUE_KEYWORDS}" ${ARGN})

  foreach(GLSL_FILE IN LISTS target_glsl_shaders_PRIVATE)
    get_filename_component(GLSL_FILE_ABS "${GLSL_FILE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    set(OUTPUT_SPV "${MY_RUNTIME_DIR}/${GLSL_FILE}.spv")
    add_custom_command(
      OUTPUT ${OUTPUT_SPV}
      COMMAND ${GLSLANGVALIDATOR_EXE} ${target_glsl_shaders_COMPILE_OPTIONS}
              "${GLSL_FILE}" -o "${OUTPUT_SPV}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      MAIN_DEPENDENCY ${GLSL_FILE_ABS}
    )
    target_sources(${TARGET_NAME} PRIVATE ${OUTPUT_SPV})
  endforeach()
endfunction()
