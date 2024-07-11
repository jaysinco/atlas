add_library(common STATIC
  ${CMAKE_SOURCE_DIR}/src/common.cpp
)
target_precompile_headers(common PUBLIC ${CMAKE_SOURCE_DIR}/src/prec.h)
target_link_libraries(common fmt.lib gflags_static.lib glog.lib shlwapi.lib ws2_32.lib)

add_library(protocol STATIC
  ${CMAKE_SOURCE_DIR}/src/protocol/type.cpp
  ${CMAKE_SOURCE_DIR}/src/protocol/protocol.cpp
  ${CMAKE_SOURCE_DIR}/src/protocol/ethernet.cpp
  ${CMAKE_SOURCE_DIR}/src/protocol/arp.cpp
  ${CMAKE_SOURCE_DIR}/src/protocol/ipv4.cpp
  ${CMAKE_SOURCE_DIR}/src/protocol/icmp.cpp
  ${CMAKE_SOURCE_DIR}/src/protocol/udp.cpp
  ${CMAKE_SOURCE_DIR}/src/protocol/tcp.cpp
  ${CMAKE_SOURCE_DIR}/src/protocol/dns.cpp
  ${CMAKE_SOURCE_DIR}/src/protocol/http.cpp
)
target_link_libraries(protocol common)

add_library(transport STATIC
  ${CMAKE_SOURCE_DIR}/src/transport/adaptor.cpp
  ${CMAKE_SOURCE_DIR}/src/transport/port-table.cpp
  ${CMAKE_SOURCE_DIR}/src/transport/packet.cpp
  ${CMAKE_SOURCE_DIR}/src/transport/transport.cpp
)
target_link_libraries(transport protocol wpcap.lib iphlpapi.lib)

add_executable(lsdev ${CMAKE_SOURCE_DIR}/src/tools/lsdev.cpp)
target_link_libraries(lsdev transport)

add_executable(arp ${CMAKE_SOURCE_DIR}/src/tools/arp.cpp)
target_link_libraries(arp transport)

add_executable(ping ${CMAKE_SOURCE_DIR}/src/tools/ping.cpp)
target_link_libraries(ping transport)

add_executable(mtu ${CMAKE_SOURCE_DIR}/src/tools/mtu.cpp)
target_link_libraries(mtu transport)

add_executable(traceroute ${CMAKE_SOURCE_DIR}/src/tools/traceroute.cpp)
target_link_libraries(traceroute transport)

add_executable(dns ${CMAKE_SOURCE_DIR}/src/tools/dns.cpp)
target_link_libraries(dns transport)
