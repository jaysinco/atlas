get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

wayland_gen_protocols(PROTO_FILES
    ${TC_USR_DIR}/share/wayland-protocols/stable/xdg-shell/xdg-shell.xml
)

add_executable(${TARGET_NAME}
    ${PROTO_FILES}
    tinywl.c
)

target_compile_definitions(${TARGET_NAME} PUBLIC WLR_USE_UNSTABLE)

target_include_directories(${TARGET_NAME} PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(${TARGET_NAME} toolkit wlroots)
