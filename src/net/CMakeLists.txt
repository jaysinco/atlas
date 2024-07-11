get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${CURRENT_DIR_NAME})

add_library(${TARGET_NAME} STATIC
    protocol/addr.cpp
    protocol/protocol.cpp
    protocol/ethernet.cpp
    protocol/arp.cpp
    protocol/ipv4.cpp
    protocol/icmp.cpp
    protocol/udp.cpp
    protocol/tcp.cpp
    protocol/dns.cpp
    protocol/http.cpp
    traffic/adaptor.cpp
    # traffic/port-table.cpp
    # traffic/packet.cpp
    # traffic/transport.cpp
)
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME} toolkit)

# add_executable(lsdev ${CMAKE_SOURCE_DIR}/src/tools/lsdev.cpp)
# target_link_libraries(lsdev ${TARGET_NAME})

# add_executable(arp ${CMAKE_SOURCE_DIR}/src/tools/arp.cpp)
# target_link_libraries(arp ${TARGET_NAME})

# add_executable(ping ${CMAKE_SOURCE_DIR}/src/tools/ping.cpp)
# target_link_libraries(ping ${TARGET_NAME})

# add_executable(mtu ${CMAKE_SOURCE_DIR}/src/tools/mtu.cpp)
# target_link_libraries(mtu ${TARGET_NAME})

# add_executable(traceroute ${CMAKE_SOURCE_DIR}/src/tools/traceroute.cpp)
# target_link_libraries(traceroute ${TARGET_NAME})

# add_executable(dns ${CMAKE_SOURCE_DIR}/src/tools/dns.cpp)
# target_link_libraries(dns ${TARGET_NAME})
