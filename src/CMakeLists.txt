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

if (WIN32)
    file(COPY 
        ${qt5_ROOT}/lib/libQt5Widgets.dll
        ${qt5_ROOT}/lib/libQt5Core.dll
        ${qt5_ROOT}/lib/libQt5Gui.dll
        DESTINATION ${ATLAS_RUNTIME_DIR} 
    )
elseif (UNIX)
    file(COPY 
        ${qt5_ROOT}/lib/libQt5Widgets.so.5
        ${qt5_ROOT}/lib/libQt5Core.so.5
        ${qt5_ROOT}/lib/libQt5Gui.so.5
        DESTINATION ${ATLAS_RUNTIME_DIR} 
        FOLLOW_SYMLINK_CHAIN
    )
endif()

add_subdirectory(utils)
add_subdirectory(parser)
add_subdirectory(helloqt)
add_subdirectory(test)
