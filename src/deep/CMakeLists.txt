get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

list(APPEND CMAKE_PREFIX_PATH $ENV{LIBTORCH_HOME}/share/cmake)
find_package(Torch QUIET)
if(NOT Torch_FOUND)
    return()
endif()

add_executable(${TARGET_NAME}-samples
    main.cpp
    linear-regression.cpp
)

target_link_libraries(${TARGET_NAME}-samples
    ${TORCH_LIBRARIES}
    toolkit
    boost
)

if(UNIX)
    set_target_properties(${TARGET_NAME}-samples PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ENV{LIBTORCH_HOME}/lib"
    )
endif()
