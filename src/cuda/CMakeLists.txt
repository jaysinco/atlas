get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION TRUE)

find_package(TensorRT)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

add_executable(${TARGET_NAME}-samples
    common.cu
    main.cpp
    hello-world.cu
    check-device.cu
    sum-matrix.cu
    reduce-integer.cu
    nested-hello-world.cu
    global-variable.cu
    cufft-test.cu
    julia-set.cu
    dot-product.cu
    ray-tracing.cu
    txi-gaussian.cu
    txi-guided.cu
    contrast-lg.cu
    $<IF:$<BOOL:${TensorRT_FOUND}>,trt-mnist.cu,>
)

target_link_libraries(${TARGET_NAME}-samples PUBLIC
    CUDA::cudart_static
    CUDA::cufft
    toolkit
    opencv
    $<IF:$<BOOL:${TensorRT_FOUND}>,trt::nvinfer,>
    $<IF:$<BOOL:${TensorRT_FOUND}>,trt::nvonnxparser,>
)

add_executable(${TARGET_NAME}-device-query
    device-query.cpp
)

target_link_libraries(${TARGET_NAME}-device-query PUBLIC
    CUDA::cudart_static
)

file(COPY
    data/hdr.jpg
    data/mnist.onnx
    data/mnist-input
    DESTINATION ${MY_RUNTIME_DIR}/data
)