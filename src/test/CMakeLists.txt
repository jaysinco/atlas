get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MAIN ${CURRENT_DIR_NAME})

add_executable(demo-crash demo-crash.cpp)
target_link_libraries(demo-crash utils)

add_executable(demo-https-server demo-https-server.cpp)
target_link_libraries(demo-https-server utils uwebsockets)

add_executable(demo-https-client demo-https-client.cpp)
target_link_libraries(demo-https-client utils libcpr)

add_executable(demo-opencl demo-opencl.cpp)
target_link_libraries(demo-opencl utils opencl-clhpp)

add_executable(${MAIN}
    main.cpp
    test-database.cpp
)

target_link_libraries(${MAIN}
    utils
    catch2
)
