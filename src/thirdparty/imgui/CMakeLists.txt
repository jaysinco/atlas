add_library(imgui STATIC
    ${TC_INSTALL_DIR}/src/imgui/imgui.cpp
    ${TC_INSTALL_DIR}/src/imgui/imgui_demo.cpp
    ${TC_INSTALL_DIR}/src/imgui/imgui_draw.cpp
    ${TC_INSTALL_DIR}/src/imgui/imgui_tables.cpp
    ${TC_INSTALL_DIR}/src/imgui/imgui_widgets.cpp
    ${TC_INSTALL_DIR}/src/imgui/backends/imgui_impl_vulkan.cpp
)

target_compile_definitions(imgui PUBLIC IMGUI_USER_CONFIG="myimconfig.h")

target_link_libraries(imgui vulkan dl)

target_include_directories(imgui PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${TC_INSTALL_DIR}/src
    ${TC_INSTALL_DIR}/src/imgui
)
