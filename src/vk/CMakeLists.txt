get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

set(MYVK_API_VERSION_MAJOR 1)
set(MYVK_API_VERSION_MINOR 3)

###############################################################################
# library
###############################################################################

set(WAYLAND_PROTO_DIR ${TC_USR_DIR}/share/wayland-protocols)

wayland_gen_protocols(WAYLAND_PROTO_SRC
    ${WAYLAND_PROTO_DIR}/stable/xdg-shell/xdg-shell.xml
)

add_library(${TARGET_NAME}
   ${WAYLAND_PROTO_SRC}
   wayland-helper.cpp
   vulkan-helper.cpp 
)

set_source_files_properties(vulkan-helper.cpp 
    PROPERTIES COMPILE_DEFINITIONS 
    "MYVK_API_VERSION_MAJOR=${MYVK_API_VERSION_MAJOR};MYVK_API_VERSION_MINOR=${MYVK_API_VERSION_MINOR}"
)

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${TARGET_NAME} PUBLIC
    toolkit
    wayland-client
    wayland-cursor
    vulkan
    dl
)

target_compile_options(${TARGET_NAME} PRIVATE -Wno-deprecated)


###############################################################################
# vkwl
###############################################################################

add_executable(vkwl
    keycode.cpp
    scene.cpp
    vkwl.cpp
)

target_glsl_shaders(vkwl
    PRIVATE
        data/vkwl.vert
        data/vkwl.frag
    COMPILE_OPTIONS
        --target-env=vulkan${MYVK_API_VERSION_MAJOR}.${MYVK_API_VERSION_MINOR}
)

target_include_directories(vkwl PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(vkwl
    ${TARGET_NAME} 
    opencv
    assimp
    easy-profiler 
    imgui
)

###############################################################################
# vkcomp
###############################################################################

add_executable(vkcomp
    vkcomp.cpp
)

target_link_libraries(vkcomp
   ${TARGET_NAME} 
)

target_glsl_shaders(vkcomp
    PRIVATE
        data/square.glsl
        data/mul2.glsl
    COMPILE_OPTIONS
        --target-env=vulkan${MYVK_API_VERSION_MAJOR}.${MYVK_API_VERSION_MINOR}
        -fshader-stage=compute
)

file(COPY
    .temp/
    DESTINATION ${MY_RUNTIME_DIR}/data
)
