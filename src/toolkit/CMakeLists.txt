get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

add_library(${TARGET_NAME}
    toolkit.cpp
    encoding.cpp
    logging.cpp
    args.cpp
    variant.cpp
    sqlite-helper.cpp
)

execute_process(COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_compile_definitions(${TARGET_NAME} PRIVATE -D_GIT_HASH="${GIT_HASH}")
target_compile_definitions(${TARGET_NAME} PRIVATE -D_GIT_BRANCH="${GIT_BRANCH}")

target_link_libraries(${TARGET_NAME} PUBLIC
    $<IF:$<BOOL:${UNIX}>,pthread,>
    boost
    spdlog
    fmt
    libiconv
    sqlite3
)
