get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -no-pie")

add_executable(${TARGET_NAME} main.cpp)
target_link_libraries(${TARGET_NAME} toolkit)

add_executable(${TARGET_NAME}-cpuid cpuid.s)
add_executable(${TARGET_NAME}-printf printf.s)
