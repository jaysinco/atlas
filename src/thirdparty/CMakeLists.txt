add_subdirectory(imgui)

add_subdirectory(${TC_THIRDPARTY}/zlib zlib)
add_subdirectory(${TC_THIRDPARTY}/openssl openssl)
add_subdirectory(${TC_THIRDPARTY}/libuv libuv)
add_subdirectory(${TC_THIRDPARTY}/fmt fmt)
add_subdirectory(${TC_THIRDPARTY}/spdlog spdlog)
add_subdirectory(${TC_THIRDPARTY}/catch2 catch2)
add_subdirectory(${TC_THIRDPARTY}/boost boost)
add_subdirectory(${TC_THIRDPARTY}/libiconv libiconv)
add_subdirectory(${TC_THIRDPARTY}/sqlite3 sqlite3)
add_subdirectory(${TC_THIRDPARTY}/uwebsockets uwebsockets)
add_subdirectory(${TC_THIRDPARTY}/libcpr libcpr)
add_subdirectory(${TC_THIRDPARTY}/opencl-clhpp opencl-clhpp)
add_subdirectory(${TC_THIRDPARTY}/wayland wayland)
add_subdirectory(${TC_THIRDPARTY}/librime librime)

file(COPY
    ${TC_INSTALL_DIR}/../lib/libstdc++.so.6
    ${TC_INSTALL_DIR}/lib/libz.so
    ${TC_INSTALL_DIR}/lib/libfmt.so
    ${TC_INSTALL_DIR}/lib/libspdlog.so
    ${TC_INSTALL_DIR}/lib/libCatch2.so
    ${TC_INSTALL_DIR}/lib/libboost_program_options.so
    ${TC_INSTALL_DIR}/lib/libcharset.so.1
    ${TC_INSTALL_DIR}/lib/libiconv.so.2
    ${TC_INSTALL_DIR}/lib/libsqlite3.so.0
    ${TC_INSTALL_DIR}/lib/libssl.so
    ${TC_INSTALL_DIR}/lib/libcrypto.so
    ${TC_INSTALL_DIR}/lib/libcpr.so
    ${TC_INSTALL_DIR}/lib/libuv.so
    ${TC_INSTALL_DIR}/lib/libffi.so.8
    ${TC_INSTALL_DIR}/lib/libwayland-client.so
    ${TC_INSTALL_DIR}/lib/libwayland-cursor.so
    ${TC_INSTALL_DIR}/lib/librime.so
    ${TC_INSTALL_DIR}/lib/libboost_system.so
    ${TC_INSTALL_DIR}/lib/libboost_regex.so
    ${TC_INSTALL_DIR}/lib/libboost_filesystem.so
    ${TC_INSTALL_DIR}/lib/libboost_atomic.so
    ${TC_INSTALL_DIR}/lib/libglog.so
    ${TC_INSTALL_DIR}/lib/libyaml-cpp.so
    ${TC_INSTALL_DIR}/lib/libgflags.so
    ${TC_INSTALL_DIR}/lib/libleveldb.so
    ${TC_INSTALL_DIR}/lib/libmarisa.so.0
    ${TC_INSTALL_DIR}/lib/libopencc.so
    ${TC_INSTALL_DIR}/lib/libunwind.so.8
    ${TC_INSTALL_DIR}/lib/liblzma.so
    ${TC_INSTALL_DIR}/bin/llama-cpp
    DESTINATION ${MY_RUNTIME_DIR}
    FOLLOW_SYMLINK_CHAIN
)
