get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MAIN ${CURRENT_DIR_NAME})

add_executable(demo-crash demo-crash.cpp)
target_link_libraries(demo-crash toolkit)

add_executable(demo-https-server demo-https-server.cpp)
target_link_libraries(demo-https-server toolkit uwebsockets zlib openssl libuv)

add_executable(demo-https-client demo-https-client.cpp)
target_link_libraries(demo-https-client toolkit libcpr)

add_executable(demo-opencl demo-opencl.cpp)
target_link_libraries(demo-opencl toolkit opencl-clhpp)

add_executable(${MAIN}
    main.cpp
    test-database.cpp
)

target_link_libraries(${MAIN}
    toolkit
    catch2
)
