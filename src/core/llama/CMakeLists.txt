get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MAIN ${CURRENT_DIR_NAME})

add_executable(${MAIN}
    ggml.c
    ggml-alloc.c
    common.cpp
    console.cpp
    grammar-parser.cpp
    llama.cpp
    main.cpp
)

if (MSVC)
else()
    target_compile_definitions(${MAIN} PUBLIC _GNU_SOURCE)
    target_compile_options(${MAIN} PUBLIC -mavx -mavx2 -mavx512f -mf16c -mfma)
endif()

target_link_libraries(${MAIN} utils)
