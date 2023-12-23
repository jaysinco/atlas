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

include_directories(${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(thirdparty)
add_subdirectory(toolkit)
add_subdirectory(parser)
add_subdirectory(test)
