get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

xxd_add_resources(parser_xxd_gen_cpp data/input.txt)
add_executable(${TARGET_NAME}-spirit spirit.cpp ${parser_xxd_gen_cpp})
target_link_libraries(${TARGET_NAME}-spirit toolkit)

add_executable(${TARGET_NAME}-spirit-autocomplete spirit-autocomplete.cpp)
target_link_libraries(${TARGET_NAME}-spirit-autocomplete toolkit)

add_executable(${TARGET_NAME}-pegtl pegtl.cpp)
target_link_libraries(${TARGET_NAME}-pegtl toolkit)
