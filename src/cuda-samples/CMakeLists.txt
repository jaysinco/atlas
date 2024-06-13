set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION TRUE)

find_package(TensorRT)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

add_executable(cuda-samples
    common.cu
    main.cpp
    hello_world.cu
    check_device.cu
    sum_matrix.cu
    reduce_integer.cu
    nested_hello_world.cu
    global_variable.cu
    test_cufft.cu
    julia_set.cu
    dot_product.cu
    ray_tracing.cu
    txi_gaussian.cu
    txi_guided.cu
    $<IF:$<BOOL:${TensorRT_FOUND}>,test_tensorrt.cu,>
)

target_link_libraries(cuda-samples PUBLIC
    CUDA::cudart_static
    CUDA::cufft
    toolkit
    opencv
    $<IF:$<BOOL:${TensorRT_FOUND}>,trt::nvinfer,>
    $<IF:$<BOOL:${TensorRT_FOUND}>,trt::nvonnxparser,>
)

add_executable(deviceQuery
    deviceQuery.cpp
)

target_link_libraries(deviceQuery PUBLIC
    CUDA::cudart_static
)

file(COPY
    data/hdr.jpg
    data/mnist.onnx
    DESTINATION ${MY_RUNTIME_DIR}/data
)