get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

wayland_gen_client_protocol(wayland_gen_cpp
    ${TC_INSTALL_DIR}/share/wayland-protocols/stable/xdg-shell/xdg-shell.xml
)

add_executable(${TARGET_NAME}
    ${wayland_gen_cpp}
    keycode-converter.cpp
    wayland-listeners.cpp
    display-context.cpp
    top-window.cpp
    left-panel.cpp
    ime-editor.cpp
    main.cpp
)

target_link_libraries(${TARGET_NAME}
    toolkit
    imgui
    wayland
    librime
    EGL
)

target_include_directories(imgui PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)

file(COPY
    data/FangZhengHeiTi.ttf
    DESTINATION ${MY_RUNTIME_DIR}/data
)
