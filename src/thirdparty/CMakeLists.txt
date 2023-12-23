add_subdirectory(imgui)
add_subdirectory(Inih)
add_subdirectory(wayland-protocols)

if (WIN32)
    set(WXMSW_INSTALL_DIR ${SV_PREBUILT_INSTALL_DIR}/wxMSW-3.2.2_vc14x_x64_Dev)

    add_subdirectory(${SV_PREBUILT_DIR}/fmt fmt)
    add_subdirectory(${SV_PREBUILT_DIR}/spdlog spdlog)
    add_subdirectory(${SV_PREBUILT_DIR}/catch2 catch2)
    add_subdirectory(${SV_PREBUILT_DIR}/tinyxml2 tinyxml2)
    add_subdirectory(${SV_PREBUILT_DIR}/sqlite3 sqlite3)
    add_subdirectory(${SV_PREBUILT_DIR}/wxwidgets wxwidgets)

    file(COPY
        ${SV_PREBUILT_INSTALL_DIR}/bin/fmt.dll
        ${SV_PREBUILT_INSTALL_DIR}/bin/spdlog.dll
        ${SV_PREBUILT_INSTALL_DIR}/bin/sqlite3.dll
        ${SV_PREBUILT_INSTALL_DIR}/bin/tinyxml2.dll
        ${SV_PREBUILT_INSTALL_DIR}/bin/tinyxml2.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxmsw32u_core_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxbase32u_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxbase32u_net_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxbase32u_xml_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxmsw32u_adv_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxmsw32u_aui_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxmsw32u_gl_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxmsw32u_html_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxmsw32u_media_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxmsw32u_propgrid_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxmsw32u_ribbon_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxmsw32u_richtext_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxmsw32u_stc_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxmsw32u_webview_vc14x_x64.dll
        ${WXMSW_INSTALL_DIR}/lib/vc14x_x64_dll/wxmsw32u_xrc_vc14x_x64.dll
        DESTINATION ${SV_RUNTIME_DIRECTORY}
    )

elseif (UNIX)
    add_subdirectory(${SV_PREBUILT_DIR}/fmt fmt)
    add_subdirectory(${SV_PREBUILT_DIR}/spdlog spdlog)
    add_subdirectory(${SV_PREBUILT_DIR}/catch2 catch2)
    add_subdirectory(${SV_PREBUILT_DIR}/tinyxml2 tinyxml2)
    add_subdirectory(${SV_PREBUILT_DIR}/sqlite3 sqlite3)
    add_subdirectory(${SV_PREBUILT_DIR}/ffmpeg ffmpeg)
    add_subdirectory(${SV_PREBUILT_DIR}/libyuv libyuv)
    add_subdirectory(${SV_PREBUILT_DIR}/sdl sdl)
    add_subdirectory(${SV_PREBUILT_DIR}/opencv opencv)
    add_subdirectory(${SV_PREBUILT_DIR}/librga librga)
    add_subdirectory(${SV_PREBUILT_DIR}/mpp mpp)
    add_subdirectory(${SV_PREBUILT_DIR}/opencl opencl)
    add_subdirectory(${SV_PREBUILT_DIR}/dcmtk dcmtk)
    add_subdirectory(${SV_PREBUILT_DIR}/gtk gtk)
    add_subdirectory(${SV_PREBUILT_DIR}/wxwidgets wxwidgets)
    add_subdirectory(${SV_PREBUILT_DIR}/fftw3 fftw3)
    add_subdirectory(${SV_PREBUILT_DIR}/eigen eigen)
    add_subdirectory(${SV_PREBUILT_DIR}/clfft clfft)
    add_subdirectory(${SV_PREBUILT_DIR}/curl curl)
    add_subdirectory(${SV_PREBUILT_DIR}/librime librime)
    add_subdirectory(${SV_PREBUILT_DIR}/sdbus-cpp sdbus-cpp)
    add_subdirectory(${SV_PREBUILT_DIR}/yaml-cpp yaml-cpp)

    file(COPY
        ${SV_PREBUILT_INSTALL_DIR}/lib/libfmt.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libspdlog.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libsqlite3.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libtinyxml2.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libavformat.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libavdevice.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libswresample.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libavcodec.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libavutil.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libswscale.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libavfilter.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libavformat.so.59
        ${SV_PREBUILT_INSTALL_DIR}/lib/libavdevice.so.59
        ${SV_PREBUILT_INSTALL_DIR}/lib/libswresample.so.4
        ${SV_PREBUILT_INSTALL_DIR}/lib/libavcodec.so.59
        ${SV_PREBUILT_INSTALL_DIR}/lib/libavutil.so.57
        ${SV_PREBUILT_INSTALL_DIR}/lib/libswscale.so.6
        ${SV_PREBUILT_INSTALL_DIR}/lib/libavfilter.so.8
        ${SV_PREBUILT_INSTALL_DIR}/lib/libiconv.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libiconv.so.2
        ${SV_PREBUILT_INSTALL_DIR}/lib/libcharset.so.1
        ${SV_PREBUILT_INSTALL_DIR}/lib/libijg8.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libijg12.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libijg16.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libyuv.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libjpeg.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libpng.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libtiff.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libSDL2.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libSDL2_ttf.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libopencv_imgproc.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libopencv_core.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libopencv_ximgproc.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libopencv_imgcodecs.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libopencv_video.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libopencv_calib3d.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libopencv_features2d.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libopencv_flann.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libx264.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libpostproc.so.56
        ${SV_PREBUILT_INSTALL_DIR}/lib/librga.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/librockchip_mpp.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmdata.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmdsig.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmect.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmfg.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmimage.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmimgle.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmiod.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmjpeg.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmjpls.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmnet.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmpmap.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmpstat.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmqrdb.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmrt.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmseg.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmsr.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmtkcharls.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmtls.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmtract.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libdcmwlm.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libofstd.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/liboflog.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libfftw3f.so.3
        ${SV_PREBUILT_INSTALL_DIR}/lib/libfftw3f_threads.so.3
        ${SV_PREBUILT_INSTALL_DIR}/lib/libwx_gtk3u-3.2.so.0
        ${SV_PREBUILT_INSTALL_DIR}/lib64/libclFFT.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/libcurl.so
        ${SV_PREBUILT_INSTALL_DIR}/lib/librime.so.1
        ${SV_PREBUILT_INSTALL_DIR}/lib/libsdbus-c++.so.1
        # ${SV_PREBUILT_INSTALL_DIR}/lib/libflutter_engine.so
        # ${SV_PREBUILT_INSTALL_DIR}/lib/libglfw.so.3
        ${SV_PREBUILT_INSTALL_DIR}/share/dcmtk/dicom.dic
        DESTINATION ${SV_RUNTIME_DIRECTORY}
        FOLLOW_SYMLINK_CHAIN
    )
endif()