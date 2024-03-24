list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(xxd-add-resources)
if(UNIX)
    include(wayland-gen-client-protocol)
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(thirdparty)
add_subdirectory(toolkit)
add_subdirectory(parser)
add_subdirectory(test)
add_subdirectory(chat-llama)

if(UNIX)
    add_subdirectory(glviewer)
endif()

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    add_subdirectory(cuda-samples)
endif()