#include "protocol.h"
#include <random>

namespace net
{

std::map<std::string, std::map<uint16_t, std::string>> protocol::port_dict = {
    {Protocol_Type_UDP,
     {{22, Protocol_Type_SSH},
      {23, Protocol_Type_TELNET},
      {53, Protocol_Type_DNS},
      {80, Protocol_Type_HTTP}}},
    {Protocol_Type_TCP,
     {{22, Protocol_Type_SSH},
      {23, Protocol_Type_TELNET},
      {53, Protocol_Type_DNS},
      {80, Protocol_Type_HTTP},
      {443, Protocol_Type_HTTPS},
      {3389, Protocol_Type_RDP}}}};

bool protocol::is_specific(std::string const& type)
{
    return type != Protocol_Type_Void && type.find("unknow") == std::string::npos;
}

uint16_t protocol::calc_checksum(void const* data, size_t tlen)
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

uint16_t protocol::rand_ushort()
{
    static std::random_device rd;
    static std::default_random_engine engine(rd());
    static std::uniform_int_distribution<uint16_t> dist;
    return dist(engine);
}

std::string protocol::guess_protocol_by_port(uint16_t port, std::string const& type)
{
    if (port_dict.count(type) > 0) {
        auto& type_dict = port_dict.at(type);
        if (type_dict.count(port) > 0) {
            return type_dict.at(port);
        }
    }
    return Protocol_Type_Unknow(-1);
}

}  // namespace net