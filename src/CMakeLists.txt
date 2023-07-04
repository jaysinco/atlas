if (WIN32)
    file(COPY
        ${qt5_ROOT}/bin/Qt5Widgets.dll
        ${qt5_ROOT}/bin/Qt5Core.dll
        ${qt5_ROOT}/bin/Qt5Gui.dll
        ${qt5_ROOT}/plugins/platforms/qwindows.dll
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
