get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

list(APPEND CMAKE_PREFIX_PATH $ENV{LIBTORCH_HOME}/share/cmake)
find_package(Torch QUIET)
if(NOT Torch_FOUND)
    return()
endif()

add_executable(${TARGET_NAME}
    network.cpp
    train.cpp
    game.cpp
    mcts.cpp
    main.cpp
)

target_link_libraries(${TARGET_NAME}
    ${TORCH_LIBRARIES}
    toolkit
)

if(UNIX)
    set_target_properties(${TARGET_NAME} PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ENV{LIBTORCH_HOME}/lib"
    )
endif()
