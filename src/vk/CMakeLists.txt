get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

set(MYVK_API_VERSION_MAJOR 1)
set(MYVK_API_VERSION_MINOR 0)

add_compile_options(-Wno-deprecated)

set_source_files_properties(vulkan-helper.cpp 
    PROPERTIES COMPILE_DEFINITIONS 
    "MYVK_API_VERSION_MAJOR=${MYVK_API_VERSION_MAJOR};MYVK_API_VERSION_MINOR=${MYVK_API_VERSION_MINOR}"
)

###############################################################################
# vkwl
###############################################################################

set(WAYLAND_PROTO_DIR ${TC_USR_DIR}/share/wayland-protocols)
wayland_gen_protocols(WAYLAND_PROTO_SRC
    ${WAYLAND_PROTO_DIR}/stable/xdg-shell/xdg-shell.xml
)

add_executable(vkwl
    ${WAYLAND_PROTO_SRC}
    vulkan-helper.cpp
    scene.cpp
    keycode.cpp
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
    toolkit
    vulkan
    wayland-client
    wayland-cursor
    opencv
    assimp
    easy-profiler 
    imgui
)

###############################################################################
# vkcomp
###############################################################################

add_executable(vkcomp
    vulkan-helper.cpp
    vkcomp.cpp
)

target_link_libraries(vkcomp
    toolkit
    vulkan
)

target_glsl_shaders(vkcomp
    PRIVATE
        data/vkcomp.glsl
    COMPILE_OPTIONS
        --target-env=vulkan${MYVK_API_VERSION_MAJOR}.${MYVK_API_VERSION_MINOR}
        -fshader-stage=compute
)

file(COPY
    .temp/
    DESTINATION ${MY_RUNTIME_DIR}/data
)
