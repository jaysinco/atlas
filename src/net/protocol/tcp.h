#pragma once
#include "protocol.h"

class tcp: public protocol
{
public:
    struct detail
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

    struct extra_detail
    {
        uint16_t len;  // Datagram length, >= 20
        uint16_t crc;  // Computed checksum
    };

    struct pseudo_header
    {
        ip4 sip;           // IPv4 Source address
        ip4 dip;           // IPv4 Destination address
        uint8_t zero_pad;  // Zero
        uint8_t type;      // IPv4 type
        uint16_t len;      // TCP Datagram length
    };

    tcp() = default;

    tcp(uint8_t const* const start, uint8_t const*& end, protocol const* prev);

    virtual ~tcp() = default;

    virtual void to_bytes(std::vector<uint8_t>& bytes) const override;

    virtual json to_json() const override;

    virtual std::string type() const override;

    virtual std::string succ_type() const override;

    virtual bool link_to(protocol const& rhs) const override;

    detail const& get_detail() const;

private:
    detail d{0};

    extra_detail extra;

    static detail ntoh(detail const& d, bool reverse = false);

    static detail hton(detail const& d);
};
