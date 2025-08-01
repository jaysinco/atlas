get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

xxd_add_resources(ocl_samples_xxd_gen_cpp
    data/txi-guided.cl
    data/spatial-denoise.cl
)

add_executable(${TARGET_NAME}-samples
    common.cpp
    main.cpp
    txi-guided.cpp
    spatial-denoise.cpp
    ${ocl_samples_xxd_gen_cpp}
)

target_link_libraries(${TARGET_NAME}-samples PUBLIC
    toolkit
    opencv
    opencl-clhpp
)

xxd_add_resources(ocl_device_query_xxd_gen_cpp
    data/hello.cl
)

add_executable(${TARGET_NAME}-device-query
    device-query.cpp
    ${ocl_device_query_xxd_gen_cpp}
)

target_link_libraries(${TARGET_NAME}-device-query PUBLIC
    toolkit
    opencl-clhpp
)

file(COPY
    ../cuda/data/hdr.jpg
    DESTINATION ${MY_RUNTIME_DIR}/data
)