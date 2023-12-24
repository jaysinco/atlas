xxd_add_resources(parser_xxd_gen_cpp data/input.txt)
add_executable(parser main.cpp ${parser_xxd_gen_cpp})
target_link_libraries(parser toolkit)

add_executable(parser-autocomplete autocomplete.cpp)
target_link_libraries(parser-autocomplete toolkit)
