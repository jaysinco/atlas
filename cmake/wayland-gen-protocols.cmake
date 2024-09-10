function(wayland_gen_protocols outfiles)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs OPTIONS)

    cmake_parse_arguments(_XML "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(xml_files ${_XML_UNPARSED_ARGUMENTS})

    foreach(it ${xml_files})
        get_filename_component(outfilename ${it} NAME_WE)
        set(infile ${it})
        set(serverheader ${CMAKE_CURRENT_BINARY_DIR}/${outfilename}-protocol.h)
        set(clientheader ${CMAKE_CURRENT_BINARY_DIR}/${outfilename}-client-protocol.h)
        set(privatecode ${CMAKE_CURRENT_BINARY_DIR}/${outfilename}-protocol.c)

        add_custom_command(
            OUTPUT ${serverheader}
            COMMAND wayland-scanner
            ARGS server-header ${infile} ${serverheader}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            MAIN_DEPENDENCY ${infile}
        )

        add_custom_command(
            OUTPUT ${clientheader}
            COMMAND wayland-scanner
            ARGS client-header ${infile} ${clientheader}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            MAIN_DEPENDENCY ${infile}
        )

        add_custom_command(
            OUTPUT ${privatecode}
            COMMAND wayland-scanner
            ARGS private-code ${infile} ${privatecode}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            MAIN_DEPENDENCY ${infile}
        )

        list(APPEND ${outfiles} ${serverheader} ${clientheader} ${privatecode})
    endforeach()
    set(${outfiles} ${${outfiles}} PARENT_SCOPE)
endfunction()
