#pragma once
#include "toolkit/variant.h"

#define Protocol_Type_Void "void"
#define Protocol_Type_Unknow(n) (n < 0 ? "unknow" : "unknow({:#x})"_format(n))
#define Protocol_Type_Ethernet "ethernet"
#define Protocol_Type_IPv4 "ipv4"
#define Protocol_Type_IPv6 "ipv6"
#define Protocol_Type_ARP "arp"
#define Protocol_Type_RARP "rarp"
#define Protocol_Type_ICMP "icmp"
#define Protocol_Type_TCP "tcp"
#define Protocol_Type_UDP "udp"
#define Protocol_Type_DNS "dns"
#define Protocol_Type_HTTP "http"
#define Protocol_Type_HTTPS "https"
#define Protocol_Type_SSH "ssh"
#define Protocol_Type_TELNET "telnet"
#define Protocol_Type_RDP "rdp"

#define ntohx(field, reverse, suffix) field = ((reverse) ? ntoh##suffix : hton##suffix)(field);

namespace net
{

class Protocol
{
public:
    // Destructor should be virtual
    virtual ~Protocol() = default;

    // Serialize current protocol layer and insert in front of `bytes`, which contains
    // raw packet bytes serialized from higher layer.
    virtual void to_bytes(std::vector<uint8_t>& bytes) const = 0;

    // Encode protocol detail as json
    virtual toolkit::Variant to_json() const = 0;

    // Self protocol type
    virtual std::string type() const = 0;

    // Successor protocol type that follows
    virtual std::string succ_type() const = 0;

    // Whether rhs is the response to this
    virtual bool link_to(protocol const& rhs) const = 0;

    // Whether protocol type is specific
    static bool is_specific(std::string const& type);

protected:
    static uint16_t calc_checksum(void const* data, size_t tlen);

    static uint16_t rand_ushort();

    static std::string guess_protocol_by_port(uint16_t port,
                                              std::string const& type = Protocol_Type_TCP);

private:
    static std::map<std::string, std::map<uint16_t, std::string>> port_dict;
};

}  // namespace net