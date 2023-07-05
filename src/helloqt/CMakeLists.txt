get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MAIN ${CURRENT_DIR_NAME})

add_executable(${MAIN}
    res/app.qrc
    main.cpp
    main-window.cpp
    
)
set_target_properties(${MAIN} PROPERTIES
    AUTOMOC ON AUTORCC ON AUTOUIC ON
)
target_link_libraries(${MAIN} utils Qt5::Widgets)
