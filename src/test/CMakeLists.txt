get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MAIN ${CURRENT_DIR_NAME})

function(single_file_exe_with_prefix)
    set(RAW_EXE_NAME ${ARGV0}-${ARGV1})
    string(REPLACE "/" "-" EXE_NAME ${RAW_EXE_NAME})
    add_executable(${EXE_NAME} ${ARGV1}.cpp)
    math(EXPR LIBS_ARGC "${ARGC}-2")
    list(SUBLIST ARGN 2 ${LIBS_ARGC} DEP_LIBS)
    target_link_libraries(${EXE_NAME} ${DEP_LIBS})
endfunction()

macro(single_file_executable)
    single_file_exe_with_prefix(${MAIN} ${ARGN})
endmacro()

single_file_executable(crash utils)
single_file_executable(https-server utils uWebSockets)
single_file_executable(https-client utils cpr::cpr)
single_file_executable(ocl utils opencl-headers
    $<IF:$<BOOL:${WIN32}>,${CMAKE_CURRENT_SOURCE_DIR}/res/OpenCL.lib,>
)

add_executable(${MAIN}
    main.cpp
    test-database.cpp
)
target_link_libraries(${MAIN}
    utils
    Catch2::Catch2WithMain
)
