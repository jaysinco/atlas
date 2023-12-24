get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MAIN ${CURRENT_DIR_NAME})

xxd_add_resources(parser_xxd_gen_cpp data/input.txt)
add_executable(${MAIN} main.cpp ${parser_xxd_gen_cpp})
target_link_libraries(${MAIN} toolkit)

add_executable(${MAIN}-autocomplete autocomplete.cpp)
target_link_libraries(${MAIN}-autocomplete toolkit)
