#pragma once
#include "protocol.h"

class arp: public protocol
{
public:
    struct detail
    {
        uint16_t hw_type;    // Hardware type
        uint16_t prot_type;  // Protocol type
        uint8_t hw_len;      // Hardware address length
        uint8_t prot_len;    // Protocol address length
        uint16_t op;         // Operation code
        mac smac;            // Source ethernet address
        ip4 sip;             // Source ip address
        mac dmac;            // Destination ethernet address
        ip4 dip;             // Destination ip address
    };

    arp() = default;

    arp(uint8_t const* const start, uint8_t const*& end, protocol const* prev = nullptr);

    arp(mac const& smac, ip4 const& sip, mac const& dmac, ip4 const& dip, bool reply, bool reverse);

    virtual ~arp() = default;

    virtual void to_bytes(std::vector<uint8_t>& bytes) const override;

    virtual json to_json() const override;

    virtual std::string type() const override;

    virtual std::string succ_type() const override;

    virtual bool link_to(protocol const& rhs) const override;

    detail const& get_detail() const;

private:
    detail d{0};

    static detail ntoh(detail const& d, bool reverse = false);

    static detail hton(detail const& d);
};
