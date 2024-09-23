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
    traffic/packet.cpp
    traffic/transport.cpp
)
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME} toolkit pcap)

add_executable(${TARGET_NAME}-lsdev cli/lsdev.cpp)
target_link_libraries(${TARGET_NAME}-lsdev ${TARGET_NAME})

add_executable(${TARGET_NAME}-sniff
    cli/sniff/main.cpp
)
target_link_libraries(${TARGET_NAME}-sniff ${TARGET_NAME} ftxui)

add_executable(${TARGET_NAME}-arp cli/arp.cpp)
target_link_libraries(${TARGET_NAME}-arp ${TARGET_NAME})

add_executable(${TARGET_NAME}-ping cli/ping.cpp)
target_link_libraries(${TARGET_NAME}-ping ${TARGET_NAME})

add_executable(${TARGET_NAME}-mtu cli/mtu.cpp)
target_link_libraries(${TARGET_NAME}-mtu ${TARGET_NAME})

add_executable(${TARGET_NAME}-traceroute cli/traceroute.cpp)
target_link_libraries(${TARGET_NAME}-traceroute ${TARGET_NAME})

add_executable(${TARGET_NAME}-dns cli/dns.cpp)
target_link_libraries(${TARGET_NAME}-dns ${TARGET_NAME})
