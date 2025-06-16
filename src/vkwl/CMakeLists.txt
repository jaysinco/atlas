get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

set(WAYLAND_PROTO_DIR ${TC_USR_DIR}/share/wayland-protocols)

wayland_gen_protocols(WAYLAND_PROTO_SRC
    ${WAYLAND_PROTO_DIR}/stable/xdg-shell/xdg-shell.xml
)

add_executable(${TARGET_NAME}
    ${WAYLAND_PROTO_SRC}
    app.cpp
    main.cpp
)

target_glsl_shaders(${TARGET_NAME}
    PRIVATE
        data/test.vert
        data/test.frag
    COMPILE_OPTIONS
        --target-env=vulkan1.0
)

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${TARGET_NAME}
    toolkit
    wayland-client
    vulkan
)
