get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MAIN ${CURRENT_DIR_NAME})

add_executable(${MAIN} main.cpp)
target_link_libraries(${MAIN} toolkit)

single_file_executable(https-server toolkit uWebSockets)
single_file_executable(concurrent-queue toolkit concurrent-queue)
single_file_executable(thread-pool toolkit threadpool)
