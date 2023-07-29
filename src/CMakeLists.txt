if (WIN32)
    file(COPY
        ${qt5_ROOT}/bin/Qt5Widgets.dll
        ${qt5_ROOT}/bin/Qt5Core.dll
        ${qt5_ROOT}/bin/Qt5Gui.dll
        ${qt5_ROOT}/plugins/platforms/qwindows.dll
        DESTINATION ${MY_RUNTIME_DIR}
    )
elseif (UNIX)
    file(COPY
        ${qt5_ROOT}/lib/libQt5Widgets.so.5
        ${qt5_ROOT}/lib/libQt5Core.so.5
        ${qt5_ROOT}/lib/libQt5Gui.so.5
        ${qt5_ROOT}/lib/libQt5DBus.so.5
        ${qt5_ROOT}/lib/libQt5EglFSDeviceIntegration.so.5
        ${qt5_ROOT}/plugins/platforms/libqeglfs.so
        # ${qt5_ROOT}/plugins/egldeviceintegrations/libqeglfs-x11-integration.so
        DESTINATION ${MY_RUNTIME_DIR}
        FOLLOW_SYMLINK_CHAIN
    )
endif()

function(xxd_add_resources outfiles)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs OPTIONS)

    cmake_parse_arguments(_RCC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(rcc_files ${_RCC_UNPARSED_ARGUMENTS})

    foreach(it ${rcc_files})
        get_filename_component(outfilename ${it} NAME_WE)
        set(infile ${it})
        set(outfile ${CMAKE_CURRENT_BINARY_DIR}/xxd_${outfilename}.cpp)

        add_custom_command(
            OUTPUT ${outfile}
            COMMAND xxd
            ARGS -i -C ${infile} ${outfile}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            MAIN_DEPENDENCY ${infile}
        )
        list(APPEND ${outfiles} ${outfile})
    endforeach()
    set(${outfiles} ${${outfiles}} PARENT_SCOPE)
endfunction()

add_subdirectory(utils)
add_subdirectory(parser)
add_subdirectory(helloqt)
add_subdirectory(test)
