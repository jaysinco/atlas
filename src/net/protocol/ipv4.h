#pragma once
#include "protocol.h"

class ipv4: public protocol
{
public:
    struct detail
    {
        uint8_t ver_hl;     // Version (4 bits) + Header length (4 bits)
        uint8_t tos;        // Type of service
        uint16_t tlen;      // Total length
        uint16_t id;        // Identification
        uint16_t flags_fo;  // Flags (3 bits) + Fragment offset (13 bits)
        uint8_t ttl;        // Time to live
        uint8_t type;       // IPv4 type
        uint16_t crc;       // Header checksum
        ip4 sip;            // Source address
        ip4 dip;            // Destination address
    };

    ipv4() = default;

    ipv4(uint8_t const* const start, uint8_t const*& end, protocol const* prev = nullptr);

    ipv4(ip4 const& sip, ip4 const& dip, uint8_t ttl, std::string const& type, bool forbid_slice);

    virtual ~ipv4() = default;

    virtual void to_bytes(std::vector<uint8_t>& bytes) const override;

    virtual json to_json() const override;

    virtual std::string type() const override;

    virtual std::string succ_type() const override;

    virtual bool link_to(protocol const& rhs) const override;

    detail const& get_detail() const;

    bool operator==(ipv4 const& rhs) const;

    uint16_t payload_size() const;

private:
    detail d{0};

    static std::map<uint8_t, std::string> type_dict;

    static detail ntoh(detail const& d, bool reverse = false);

    static detail hton(detail const& d);
};
