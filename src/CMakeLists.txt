list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

include(xxd-add-resources)
include(wayland-gen-client-protocol)

include_directories(${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(thirdparty)
add_subdirectory(toolkit)
add_subdirectory(parser)
add_subdirectory(glview)
add_subdirectory(test)
