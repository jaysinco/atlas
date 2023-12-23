get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MAIN ${CURRENT_DIR_NAME})

add_library(${MAIN}
    base.cpp
    encoding.cpp
    logging.cpp
    args.cpp
    sqlite-helper.cpp
)

exec_program("git" ${PROJECT_SOURCE_DIR} ARGS "rev-parse HEAD" OUTPUT_VARIABLE GIT_HASH)
target_compile_definitions(${MAIN} PRIVATE -D_GIT_HASH="${GIT_HASH}")
exec_program("git" ${PROJECT_SOURCE_DIR} ARGS "rev-parse --abbrev-ref HEAD" OUTPUT_VARIABLE GIT_BRANCH)
target_compile_definitions(${MAIN} PRIVATE -D_GIT_BRANCH="${GIT_BRANCH}")

target_link_libraries(${MAIN} PUBLIC
    pthread
    boost
    spdlog
    fmt
    libiconv
    sqlite3
)
