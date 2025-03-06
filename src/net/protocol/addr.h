#pragma once
#include <string>
#include <cstdint>
#include "toolkit/error.h"
#include <netinet/in.h>
#include "toolkit/format.h"

namespace net
{

struct Mac
{
    Mac() = default;
    Mac(Mac const&) = default;
    Mac(uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4, uint8_t c5, uint8_t c6);
    explicit Mac(std::string const& s, bool use_colon = false);

    std::string toStr(bool use_colon = false) const;

    bool operator==(Mac const& rhs) const;
    bool operator!=(Mac const& rhs) const;

    static MyErrCode fromDashStr(std::string const& s, Mac* mac = nullptr);
    static MyErrCode fromColonStr(std::string const& s, Mac* mac = nullptr);

    static Mac const kZeros;
    static Mac const kBroadcast;

    uint8_t b1, b2, b3, b4, b5, b6;
};

struct Ip4
{
    Ip4() = default;
    Ip4(Ip4 const&) = default;
    Ip4(uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4);
    explicit Ip4(std::string const& s);
    Ip4(in_addr const& addr);

    operator in_addr() const;
    operator uint32_t() const;
    bool operator==(Ip4 const& rhs) const;
    bool operator!=(Ip4 const& rhs) const;
    uint32_t operator&(Ip4 const& rhs) const;

    bool isLocal(Ip4 const& rhs, Ip4 const& mask) const;
    std::string toStr() const;

    static MyErrCode fromDottedDec(std::string const& s, Ip4* ip = nullptr);
    static MyErrCode fromDomain(std::string const& s, Ip4* ip = nullptr);

    static Ip4 const kZeros;
    static Ip4 const kBroadcast;

    uint8_t b1, b2, b3, b4;
};

}  // namespace net
