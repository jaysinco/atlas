add_executable(test
    test-database.cpp
    main.cpp
)

target_link_libraries(test
    toolkit
    catch2
)
