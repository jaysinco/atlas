#pragma once
#include "protocol.h"
#include "addr.h"

namespace net
{

class Tcp: public Protocol
{
public:
    struct Detail
    {
        uint16_t sport;     // Source port
        uint16_t dport;     // Destination port
        uint32_t sn;        // Sequence number
        uint32_t an;        // Acknowledgment number
        uint16_t hl_flags;  // Header length (4 bits) + Reserved (3 bits) + Flags (9 bits)
        uint16_t wlen;      // Window Size
        uint16_t crc;       // Checksum
        uint16_t urp;       // Urgent pointer
    };

    struct ExtraDetail
    {
        uint16_t len;  // Datagram length, >= 20
        uint16_t crc;  // Computed checksum
    };

    struct PseudoHeader
    {
        Ip4 sip;           // IPv4 Source address
        Ip4 dip;           // IPv4 Destination address
        uint8_t zero_pad;  // Zero
        uint8_t type;      // IPv4 type
        uint16_t len;      // TCP Datagram length
    };

    Tcp() = default;

    ~Tcp() override = default;
    MyErrCode encode(std::vector<uint8_t>& bytes) const override;
    MyErrCode decode(uint8_t const* const start, uint8_t const*& end,
                     Protocol const* prev) override;
    Variant toVariant() const override;
    Type type() const override;
    Type succType() const override;
    bool linkTo(Protocol const& rhs) const override;

    Detail const& getDetail() const;

private:
    Detail d_{0};
    ExtraDetail extra_;

    static Detail ntoh(Detail const& d, bool reverse = false);
    static Detail hton(Detail const& d);
};

}  // namespace net