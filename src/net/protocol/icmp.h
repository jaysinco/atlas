#pragma once
#include "protocol.h"
#include "ipv4.h"

namespace net
{

class Icmp: public Protocol
{
public:
    struct Detail
    {
        uint8_t type;  // Type
        uint8_t code;  // Code
        uint16_t crc;  // Checksum as a whole

        union
        {
            struct
            {
                uint16_t id;  // Identification
                uint16_t sn;  // Serial number
            } s;

            uint32_t i;
        } u;
    };

    struct ExtraDetail
    {
        std::string raw;  // Raw data, including ping echo
        Ipv4 eip;         // Error ip header
        uint8_t buf[8];   // At least 8 bytes behind ip header
    };

    Icmp() = default;
    Icmp(std::string const& ping_echo);

    ~Icmp() override = default;
    MyErrCode encode(std::vector<uint8_t>& bytes) const override;
    MyErrCode decode(uint8_t const* const start, uint8_t const*& end,
                     Protocol const* prev) override;
    Variant toVariant() const override;
    Type type() const override;
    Type succType() const override;
    bool linkTo(Protocol const& rhs) const override;

    Detail const& getDetail() const;
    ExtraDetail const& getExtra() const;
    std::string icmpType() const;

private:
    Detail d_{0};
    ExtraDetail extra_;

    static std::map<uint8_t, std::pair<std::string, std::map<uint8_t, std::string>>> type_dict;
    static Detail ntoh(Detail const& d, bool reverse = false);
    static Detail hton(Detail const& d);
};

}  // namespace net