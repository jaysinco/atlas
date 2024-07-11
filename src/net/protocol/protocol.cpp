#include "protocol.h"
#include <random>

namespace net
{

std::map<Protocol::Type, std::map<uint16_t, Protocol::Type>> Protocol::protocol_ports = {
    {Protocol::kUDP,
     {
         {22, Protocol::kSSH},
         {23, Protocol::kTELNET},
         {53, Protocol::kDNS},
         {80, Protocol::kHTTP},
     }},
    {Protocol::kTCP,
     {
         {22, Protocol::kSSH},
         {23, Protocol::kTELNET},
         {53, Protocol::kDNS},
         {80, Protocol::kHTTP},
         {443, Protocol::kHTTPS},
         {3389, Protocol::kRDP},
     }}};

std::string Protocol::typeToStr(Type type)
{
    switch (type) {
        case kEmpty:
            return "empty";
        case kUnknow:
            return "unknow";
        case kEthernet:
            return "ethernet";
        case kIPv4:
            return "ipv4";
        case kIPv6:
            return "ipv6";
        case kARP:
            return "arp";
        case kRARP:
            return "rarp";
        case kICMP:
            return "icmp";
        case kTCP:
            return "tcp";
        case kUDP:
            return "udp";
        case kDNS:
            return "dns";
        case kHTTP:
            return "http";
        case kHTTPS:
            return "https";
        case kSSH:
            return "ssh";
        case kTELNET:
            return "telnet";
        case kRDP:
            return "rdp";
        default:
            return "invalid";
    }
}

uint16_t Protocol::calcChecksum(void const* data, size_t tlen)
{
    uint32_t sum = 0;
    auto buf = static_cast<uint16_t const*>(data);
    while (tlen > 1) {
        sum += *buf++;
        tlen -= 2;
    }
    if (tlen > 0) {
        uint16_t left = 0;
        std::memcpy(&left, buf, 1);
        sum += left;
    }
    while (sum >> 16) {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    return (static_cast<uint16_t>(sum) ^ 0xffff);
}

uint16_t Protocol::randUint16()
{
    static std::random_device rd;
    static std::default_random_engine engine(rd());
    static std::uniform_int_distribution<uint16_t> dist;
    return dist(engine);
}

Protocol::Type Protocol::guessProtocolByPort(uint16_t port, Type type)
{
    if (protocol_ports.count(type) > 0) {
        auto& type_dict = protocol_ports.at(type);
        if (type_dict.count(port) > 0) {
            return type_dict.at(port);
        }
    }
    return kUnknow;
}

}  // namespace net