get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

add_executable(${TARGET_NAME}
    test-database.cpp
    main.cpp
)

target_link_libraries(${TARGET_NAME}
    toolkit
    catch2
)
