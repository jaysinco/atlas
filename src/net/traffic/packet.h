#pragma once
#include "protocol/protocol.h"
#include "protocol/addr.h"
#include <optional>
#include <chrono>

namespace net
{

class Packet
{
public:
    using Time = std::chrono::system_clock::time_point;

    struct Detail
    {
        Protocol::Stack layers;  // Protocol layers
        Time time;               // Received time
    };

    Packet();
    MyErrCode encode(std::vector<uint8_t>& bytes) const;
    MyErrCode decode(uint8_t const* const start, uint8_t const* const end,
                     Protocol::Type start_type = Protocol::kEthernet);
    Variant const& toVariant() const;
    bool linkTo(Packet const& rhs) const;
    bool hasType(Protocol::Type type) const;
    bool hasIcmpError() const;
    void setLayers(Protocol::Stack const& st);
    void setLayers(Protocol::Stack&& st);
    void setTime(Time const& tm);
    Detail const& getDetail() const;

public:
    static Packet arp(Mac const& smac, Ip4 const& sip, Mac const& dmac, Ip4 const& dip,
                      bool reply = false, bool reverse = false);

    static Packet ping(Mac const& smac, Ip4 const& sip, Mac const& dmac, Ip4 const& dip,
                       uint8_t ttl = 128, std::string const& echo = "", bool forbid_slice = false);

private:
    Detail d_;
    std::optional<Variant> j_cached_;
    static MyErrCode decodeLayer(Protocol::Type type, uint8_t const* const start,
                                 uint8_t const*& end, Protocol const* prev, Protocol::Ptr& pt);
};

}  // namespace net