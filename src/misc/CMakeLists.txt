get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

add_executable(${TARGET_NAME}-crash misc-crash.cpp)
target_link_libraries(${TARGET_NAME}-crash toolkit)

add_executable(${TARGET_NAME}-https-server misc-https-server.cpp)
target_link_libraries(${TARGET_NAME}-https-server toolkit uwebsockets zlib openssl libuv)

add_executable(${TARGET_NAME}-https-client misc-https-client.cpp)
target_link_libraries(${TARGET_NAME}-https-client toolkit libcpr)

add_executable(${TARGET_NAME}-gnuplot misc-gnuplot.cpp)
target_link_libraries(${TARGET_NAME}-gnuplot toolkit)

file(COPY
    data/cert.pem
    data/key.pem
    DESTINATION ${MY_RUNTIME_DIR}/data
)