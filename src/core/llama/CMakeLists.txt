get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MAIN ${CURRENT_DIR_NAME})

add_executable(${MAIN}
    ggml.c
    ggml-alloc.c
    common.cpp
    k_quants.c
    console.cpp
    grammar-parser.cpp
    llama.cpp
    main.cpp
)

if (WIN32)
    target_compile_options(${MAIN} PUBLIC /arch:AVX /arch:AVX2)
else()
    target_compile_definitions(${MAIN} PUBLIC _GNU_SOURCE)
    target_compile_options(${MAIN} PUBLIC -mavx -mavx2 -mf16c -mfma)
endif()

target_compile_definitions(${MAIN} PUBLIC
    GGML_USE_K_QUANTS
    LOG_DISABLE_LOGS
)

target_link_libraries(${MAIN} utils)

file(COPY res/chat.sh DESTINATION ${MY_RUNTIME_DIR})
