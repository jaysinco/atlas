#pragma once
#include "protocol.h"
#include "addr.h"

namespace net
{

class Udp: public Protocol
{
public:
    struct Detail
    {
        uint16_t sport;  // Source port
        uint16_t dport;  // Destination port
        uint16_t len;    // Datagram length, >= 8
        uint16_t crc;    // Checksum
    };

    struct ExtraDetail
    {
        uint16_t crc;  // Computed checksum
    };

    struct PseudoHeader
    {
        Ip4 sip;           // IPv4 Source address
        Ip4 dip;           // IPv4 Destination address
        uint8_t zero_pad;  // Zero
        uint8_t type;      // IPv4 type
        uint16_t len;      // UDP Datagram length
    };

    Udp() = default;

    ~Udp() override = default;
    MyErrCode encode(std::vector<uint8_t>& bytes) const override;
    MyErrCode decode(uint8_t const* const start, uint8_t const*& end,
                     Protocol const* prev) override;
    Variant toVariant() const override;
    Type type() const override;
    Type succType() const override;
    bool linkTo(Protocol const& rhs) const override;

    Detail const& getDetail() const;
    ExtraDetail const& getExtra() const;

private:
    Detail d_{0};
    ExtraDetail extra_;

    static Detail ntoh(Detail const& d, bool reverse = false);
    static Detail hton(Detail const& d);
};

}  // namespace net