#pragma once
#include "protocol.h"
#include <map>
#include "addr.h"

namespace net
{

class Ethernet: public Protocol
{
public:
    static constexpr int kMinFrameSize = 64;
    static constexpr int kMinFrameSizeNoFCS = kMinFrameSize - 4;  // without frame check sequence

    struct Detail
    {
        Mac dmac;       // Destination address
        Mac smac;       // Source address
        uint16_t type;  // Ethernet type
    };

    Ethernet() = default;
    Ethernet(Mac const& smac, Mac const& dmac, Type type);

    ~Ethernet() override = default;
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
    static std::map<uint16_t, Type> type_dict;
    static Detail ntoh(Detail const& d, bool reverse = false);
    static Detail hton(Detail const& d);
};

}  // namespace net