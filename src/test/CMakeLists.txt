get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MAIN ${CURRENT_DIR_NAME})

add_executable(demo-crash demo-crash.cpp)
target_link_libraries(demo-crash utils)

add_executable(demo-https-server demo-https-server.cpp)
target_link_libraries(demo-https-server utils uWebSockets)

add_executable(demo-https-client demo-https-client.cpp)
target_link_libraries(demo-https-client utils cpr::cpr)

add_executable(demo-opencl demo-opencl.cpp)
target_link_libraries(demo-opencl utils opencl-headers
    $<IF:$<BOOL:${WIN32}>,
        ${CMAKE_CURRENT_SOURCE_DIR}/res/OpenCL.lib,
        /usr/lib/x86_64-linux-gnu/libOpenCL.so.1>
)

if (UNIX)
    add_executable(demo-drm demo-drm.cpp)
    target_include_directories(demo-drm PUBLIC /usr/include/libdrm)
    target_link_libraries(demo-drm utils drm gbm EGL GLESv2)
endif()

add_executable(${MAIN}
    main.cpp
    test-database.cpp
)

target_link_libraries(${MAIN}
    utils
    Catch2::Catch2WithMain
)
