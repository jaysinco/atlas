get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MAIN ${CURRENT_DIR_NAME})

add_executable(${MAIN} main.cpp)
target_link_libraries(${MAIN} utils)

single_file_executable(https-server utils uWebSockets)
single_file_executable(concurrent-queue utils concurrent-queue)
single_file_executable(thread-pool utils threadpool)
