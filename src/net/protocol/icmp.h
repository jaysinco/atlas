#pragma once
#include "protocol.h"
#include "ipv4.h"

class icmp: public protocol
{
public:
    struct detail
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

    struct extra_detail
    {
        std::string raw;  // Raw data, including ping echo
        ipv4 eip;         // Error ip header
        uint8_t buf[8];   // At least 8 bytes behind ip header
    };

    icmp() = default;

    icmp(uint8_t const* const start, uint8_t const*& end, protocol const* prev);

    icmp(std::string const& ping_echo);

    virtual ~icmp() = default;

    virtual void to_bytes(std::vector<uint8_t>& bytes) const override;

    virtual json to_json() const override;

    virtual std::string type() const override;

    virtual std::string succ_type() const override;

    virtual bool link_to(protocol const& rhs) const override;

    detail const& get_detail() const;

    extra_detail const& get_extra() const;

    std::string icmp_type() const;

private:
    detail d{0};

    extra_detail extra;

    static std::map<uint8_t, std::pair<std::string, std::map<uint8_t, std::string>>> type_dict;

    static detail ntoh(detail const& d, bool reverse = false);

    static detail hton(detail const& d);
};
