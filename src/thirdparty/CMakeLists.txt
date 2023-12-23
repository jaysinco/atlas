add_subdirectory(${TC_THIRDPARTY}/fmt fmt)
add_subdirectory(${TC_THIRDPARTY}/spdlog spdlog)
add_subdirectory(${TC_THIRDPARTY}/catch2 catch2)
add_subdirectory(${TC_THIRDPARTY}/boost boost)
add_subdirectory(${TC_THIRDPARTY}/libiconv libiconv)
add_subdirectory(${TC_THIRDPARTY}/sqlite3 sqlite3)
add_subdirectory(${TC_THIRDPARTY}/uwebsockets uwebsockets)
add_subdirectory(${TC_THIRDPARTY}/libcpr libcpr)
add_subdirectory(${TC_THIRDPARTY}/opencl-clhpp opencl-clhpp)

file(COPY
    ${TC_INSTALL_DIR}/../lib/libstdc++.so.6
    ${TC_INSTALL_DIR}/lib/libfmt.so
    ${TC_INSTALL_DIR}/lib/libspdlog.so
    ${TC_INSTALL_DIR}/lib/libboost_program_options.so
    ${TC_INSTALL_DIR}/lib/libcharset.so.1
    ${TC_INSTALL_DIR}/lib/libiconv.so.2
    ${TC_INSTALL_DIR}/lib/libsqlite3.so.0

    DESTINATION ${MY_RUNTIME_DIR}
    FOLLOW_SYMLINK_CHAIN
)
