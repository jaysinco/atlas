#pragma once
#include "protocol.h"
#include <map>

class ethernet: public protocol
{
public:
    struct detail
    {
        mac dmac;       // Destination address
        mac smac;       // Source address
        uint16_t type;  // Ethernet type
    };

    ethernet() = default;

    ethernet(uint8_t const* const start, uint8_t const*& end, protocol const* prev = nullptr);

    ethernet(mac const& smac, mac const& dmac, std::string const& type);

    virtual ~ethernet() = default;

    virtual void to_bytes(std::vector<uint8_t>& bytes) const override;

    virtual json to_json() const override;

    virtual std::string type() const override;

    virtual std::string succ_type() const override;

    virtual bool link_to(protocol const& rhs) const override;

    detail const& get_detail() const;

private:
    detail d{0};

    static std::map<uint16_t, std::string> type_dict;

    static detail ntoh(detail const& d, bool reverse = false);

    static detail hton(detail const& d);
};
