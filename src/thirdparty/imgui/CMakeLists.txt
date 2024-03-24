add_library(imgui STATIC
    ${TC_INSTALL_DIR}/src/imgui/imgui_demo.cpp
    ${TC_INSTALL_DIR}/src/imgui/imgui_draw.cpp
    ${TC_INSTALL_DIR}/src/imgui/imgui_tables.cpp
    ${TC_INSTALL_DIR}/src/imgui/imgui_widgets.cpp
    ${TC_INSTALL_DIR}/src/imgui/imgui.cpp
    ${TC_INSTALL_DIR}/src/imgui/imgui_demo.cpp
    ${TC_INSTALL_DIR}/src/imgui/backends/imgui_impl_opengl3.cpp
)

target_link_libraries(imgui GLESv2 dl)

target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_ES3)

target_include_directories(imgui PUBLIC
    ${TC_INSTALL_DIR}/src
    ${TC_INSTALL_DIR}/src/imgui
)
