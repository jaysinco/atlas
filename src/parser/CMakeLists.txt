get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

xxd_add_resources(parser_xxd_gen_cpp data/input.txt)
add_executable(${TARGET_NAME} main.cpp ${parser_xxd_gen_cpp})
target_link_libraries(${TARGET_NAME} toolkit)

add_executable(${TARGET_NAME}-autocomplete autocomplete.cpp)
target_link_libraries(${TARGET_NAME}-autocomplete toolkit)
