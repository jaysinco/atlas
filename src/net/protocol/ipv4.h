#pragma once
#include "protocol.h"
#include "addr.h"

namespace net
{

class Ipv4: public Protocol
{
public:
    struct Detail
    {
        uint8_t ver_hl;     // Version (4 bits) + Header length (4 bits)
        uint8_t tos;        // Type of service
        uint16_t tlen;      // Total length
        uint16_t id;        // Identification
        uint16_t flags_fo;  // Flags (3 bits) + Fragment offset (13 bits)
        uint8_t ttl;        // Time to live
        uint8_t type;       // IPv4 type
        uint16_t crc;       // Header checksum
        Ip4 sip;            // Source address
        Ip4 dip;            // Destination address
    };

    Ipv4() = default;
    Ipv4(Ip4 const& sip, Ip4 const& dip, uint8_t ttl, Type type, bool forbid_slice);

    ~Ipv4() override = default;
    MyErrCode encode(std::vector<uint8_t>& bytes) const override;
    MyErrCode decode(uint8_t const* const start, uint8_t const*& end,
                     Protocol const* prev) override;
    Variant toVariant() const override;
    Type type() const override;
    Type succType() const override;
    bool linkTo(Protocol const& rhs) const override;

    Detail const& getDetail() const;
    bool operator==(Ipv4 const& rhs) const;
    uint16_t payloadSize() const;

private:
    Detail d_{0};
    static std::map<uint8_t, Type> type_dict;
    static Detail ntoh(Detail const& d, bool reverse = false);
    static Detail hton(Detail const& d);
};

}  // namespace net