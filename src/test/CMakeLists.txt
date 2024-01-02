add_executable(demo-crash demo-crash.cpp)
target_link_libraries(demo-crash toolkit)

add_executable(demo-https-server demo-https-server.cpp)
target_link_libraries(demo-https-server toolkit uwebsockets zlib openssl libuv)

add_executable(demo-https-client demo-https-client.cpp)
target_link_libraries(demo-https-client toolkit libcpr)

xxd_add_resources(demo_opencl_xxd_gen_cpp data/hello.cl)
add_executable(demo-opencl demo-opencl.cpp ${demo_opencl_xxd_gen_cpp})
target_link_libraries(demo-opencl toolkit opencl-clhpp)

add_executable(test
    test-database.cpp
    main.cpp
)

target_link_libraries(test
    toolkit
    catch2
)

file(COPY
    data/cert.pem
    data/key.pem
    DESTINATION ${MY_RUNTIME_DIR}/data
)
