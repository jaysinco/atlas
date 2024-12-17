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
add_subdirectory(${TC_THIRDPARTY}/opencv opencv)

if(UNIX)
    add_subdirectory(imgui)
    add_subdirectory(${TC_THIRDPARTY}/opencl-clhpp opencl-clhpp)
    add_subdirectory(${TC_THIRDPARTY}/ftxui ftxui)

    set(COPY_FILE_LIST
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
        ${TC_INSTALL_DIR}/lib/libunwind.so.8
        ${TC_INSTALL_DIR}/lib/libopencv_core.so
        ${TC_INSTALL_DIR}/lib/libopencv_imgcodecs.so
        ${TC_INSTALL_DIR}/lib/libopencv_imgproc.so
        ${TC_INSTALL_DIR}/lib/libopencv_ximgproc.so
        ${TC_INSTALL_DIR}/lib/libjpeg.so
        ${TC_INSTALL_DIR}/lib/libwebp.so
        ${TC_INSTALL_DIR}/lib/libpng16.so
        ${TC_INSTALL_DIR}/lib/libtiff.so
        ${TC_INSTALL_DIR}/lib/libopenjp2.so
        ${TC_INSTALL_DIR}/lib/libsharpyuv.so
        ${TC_INSTALL_DIR}/lib/liblzma.so
        ${TC_INSTALL_DIR}/lib/libzstd.so
        ${TC_INSTALL_DIR}/lib/libcurl.so
        # ${TC_INSTALL_DIR}/lib/libpcap.so
        ${TC_INSTALL_DIR}/lib/libftxui-screen.so
        ${TC_INSTALL_DIR}/lib/libftxui-dom.so
        ${TC_INSTALL_DIR}/lib/libftxui-component.so
    )

    if(TC_CROSS_COMPILE)
        list(APPEND COPY_FILE_LIST ${TC_INSTALL_DIR}/../lib/libstdc++.so.6)
    endif()

    file(COPY
        ${COPY_FILE_LIST}
        DESTINATION ${MY_RUNTIME_DIR}/lib
        FOLLOW_SYMLINK_CHAIN
    )
elseif(WIN32)
    file(COPY
        ${TC_INSTALL_DIR}/bin/zlib1.dll
        ${TC_INSTALL_DIR}/bin/zstd.dll
        ${TC_INSTALL_DIR}/bin/fmt.dll
        ${TC_INSTALL_DIR}/bin/spdlog.dll
        ${TC_INSTALL_DIR}/bin/Catch2.dll
        ${TC_INSTALL_DIR}/bin/boost_program_options.dll
        ${TC_INSTALL_DIR}/bin/charset-1.dll
        ${TC_INSTALL_DIR}/bin/iconv-2.dll
        ${TC_INSTALL_DIR}/bin/cpr.dll
        ${TC_INSTALL_DIR}/bin/libcrypto-1_1-x64.dll
        ${TC_INSTALL_DIR}/bin/libssl-1_1-x64.dll
        ${TC_INSTALL_DIR}/bin/libcurl.dll
        ${TC_INSTALL_DIR}/bin/uv.dll
        ${TC_INSTALL_DIR}/bin/sqlite3.dll
        ${TC_INSTALL_DIR}/bin/opencv_core481.dll
        ${TC_INSTALL_DIR}/bin/opencv_imgcodecs481.dll
        ${TC_INSTALL_DIR}/bin/opencv_imgproc481.dll
        ${TC_INSTALL_DIR}/bin/jpeg8.dll
        ${TC_INSTALL_DIR}/bin/libwebp.dll
        ${TC_INSTALL_DIR}/bin/tiff.dll
        ${TC_INSTALL_DIR}/bin/libpng16.dll
        ${TC_INSTALL_DIR}/bin/openjp2.dll
        ${TC_INSTALL_DIR}/bin/libsharpyuv.dll
        DESTINATION ${MY_RUNTIME_DIR}
    )
endif()
