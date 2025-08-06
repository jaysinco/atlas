get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

add_executable(${TARGET_NAME}
    vk-mem-alloc.cpp
    main.cpp
)

target_glsl_shaders(${TARGET_NAME}
    PRIVATE
        data/vkcomp.glsl
    COMPILE_OPTIONS
        --target-env=vulkan1.3
        --target-spv=spv1.3
        -fshader-stage=compute
)

target_link_libraries(${TARGET_NAME}
    toolkit
    vulkan
)

