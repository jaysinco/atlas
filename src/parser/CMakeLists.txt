get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MAIN ${CURRENT_DIR_NAME})

add_executable(${MAIN} main.cpp)
target_link_libraries(${MAIN} toolkit Boost::headers)

add_executable(${MAIN}-autocomplete autocomplete.cpp)
target_link_libraries(${MAIN}-autocomplete toolkit Boost::headers)
