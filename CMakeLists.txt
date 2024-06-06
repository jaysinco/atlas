cmake_minimum_required(VERSION 3.16)

project(Atlas C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MY_RUNTIME_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

if(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    add_link_options("-rdynamic" "-Wl,--disable-new-dtags")
elseif(WIN32)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    add_compile_definitions(
        WINVER=0x0601
        _WIN32_WINNT=0x0602
        NOMINMAX
    )
    add_compile_options(-utf-8 -permissive-)
endif()

add_subdirectory(src)
