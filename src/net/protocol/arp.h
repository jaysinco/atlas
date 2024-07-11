#pragma once
#include "protocol.h"
#include "addr.h"

namespace net
{

class Arp: public Protocol
{
public:
    struct Detail
    {
        uint16_t hw_type;    // Hardware type
        uint16_t prot_type;  // Protocol type
        uint8_t hw_len;      // Hardware address length
        uint8_t prot_len;    // Protocol address length
        uint16_t op;         // Operation code
        Mac smac;            // Source ethernet address
        Ip4 sip;             // Source ip address
        Mac dmac;            // Destination ethernet address
        Ip4 dip;             // Destination ip address
    };

    Arp() = default;
    Arp(Mac const& smac, Ip4 const& sip, Mac const& dmac, Ip4 const& dip, bool reply, bool reverse);

    ~Arp() override = default;
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
    static Detail ntoh(Detail const& d, bool reverse = false);
    static Detail hton(Detail const& d);
};

}  // namespace net