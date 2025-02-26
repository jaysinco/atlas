get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
    return()
endif()

enable_language(ASM)

add_executable(${TARGET_NAME}
    main.cpp
    printf.s
    cpuid.s
)

target_link_libraries(${TARGET_NAME} toolkit)
