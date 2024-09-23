list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(xxd-add-resources)
if(UNIX)
    include(wayland-gen-protocols)
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR})

if(TC_CROSS_COMPILE)
    set(TC_USR_DIR ${TC_INSTALL_DIR})
else()
    set(TC_USR_DIR /usr)
endif()

add_subdirectory(thirdparty)
add_subdirectory(toolkit)
add_subdirectory(parser)
add_subdirectory(demo)
add_subdirectory(test)

if(UNIX)
    add_subdirectory(net)
endif()

if(NOT TC_CROSS_COMPILE)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        add_subdirectory(cuda-samples)
    endif()
endif()
