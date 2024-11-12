get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

xxd_add_resources(ocl_device_query_xxd_gen_cpp
    data/hello.cl
)

add_executable(${TARGET_NAME}-device-query
    device-query.cpp
    ${ocl_device_query_xxd_gen_cpp}
)

target_link_libraries(${TARGET_NAME}-device-query
    toolkit
    opencl-clhpp
)
