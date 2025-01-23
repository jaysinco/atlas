#include "addr.h"
#include <sstream>
#include <arpa/inet.h>
#include <netdb.h>
#include "toolkit/toolkit.h"
#include "toolkit/logging.h"
#include <regex>

namespace net
{

Mac const Mac::kZeros = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
Mac const Mac::kBroadcast = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

Mac::Mac(uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4, uint8_t c5, uint8_t c6)
    : b1(c1), b2(c2), b3(c3), b4(c4), b5(c5), b6(c6)
{
}

Mac::Mac(std::string const& s, bool use_colon)
{
    MyErrCode err = use_colon ? fromColonStr(s, this) : fromDashStr(s, this);
    if (err != MyErrCode::kOk) {
        MY_THROW("invalid mac: {}", s);
    }
}

bool Mac::operator==(Mac const& rhs) const
{
    return b1 == rhs.b1 && b2 == rhs.b2 && b3 == rhs.b3 && b4 == rhs.b4 && b5 == rhs.b5 &&
           b6 == rhs.b6;
}

bool Mac::operator!=(Mac const& rhs) const { return !(*this == rhs); }

std::string Mac::toStr(bool use_colon) const
{
    auto c = reinterpret_cast<uint8_t const*>(this);
    std::ostringstream ss;
    ss << FSTR("{:02x}", c[0]);
    for (int i = 1; i < 6; ++i) {
        ss << FSTR("{}{:02x}", (use_colon ? ":" : "-"), c[i]);
    }
    return ss.str();
}

MyErrCode Mac::fromDashStr(std::string const& s, Mac* mac)
{
    static std::regex pat(R"((\w{2})-(\w{2})-(\w{2})-(\w{2})-(\w{2})-(\w{2}))");
    std::smatch res;
    Mac maddr;
    if (std::regex_match(s, res, pat)) {
        maddr.b1 = std::stoi(res.str(1), nullptr, 16);
        maddr.b2 = std::stoi(res.str(2), nullptr, 16);
        maddr.b3 = std::stoi(res.str(3), nullptr, 16);
        maddr.b4 = std::stoi(res.str(4), nullptr, 16);
        maddr.b5 = std::stoi(res.str(5), nullptr, 16);
        maddr.b6 = std::stoi(res.str(6), nullptr, 16);
    } else {
        DLOG("failed to parse mac: {}", s);
        return MyErrCode::kFailed;
    }
    if (mac) {
        *mac = maddr;
    }
    return MyErrCode::kOk;
}

MyErrCode Mac::fromColonStr(std::string const& s, Mac* mac)
{
    static std::regex pat(R"((\w{2}):(\w{2}):(\w{2}):(\w{2}):(\w{2}):(\w{2}))");
    std::smatch res;
    Mac maddr;
    if (std::regex_match(s, res, pat)) {
        maddr.b1 = std::stoi(res.str(1), nullptr, 16);
        maddr.b2 = std::stoi(res.str(2), nullptr, 16);
        maddr.b3 = std::stoi(res.str(3), nullptr, 16);
        maddr.b4 = std::stoi(res.str(4), nullptr, 16);
        maddr.b5 = std::stoi(res.str(5), nullptr, 16);
        maddr.b6 = std::stoi(res.str(6), nullptr, 16);
    } else {
        DLOG("failed to parse mac: {}", s);
        return MyErrCode::kFailed;
    }
    if (mac) {
        *mac = maddr;
    }
    return MyErrCode::kOk;
}

Ip4 const Ip4::kZeros = {0x0, 0x0, 0x0, 0x0};
Ip4 const Ip4::kBroadcast = {0xff, 0xff, 0xff, 0xff};

Ip4::Ip4(uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4): b1(c1), b2(c2), b3(c3), b4(c4) {}

Ip4::Ip4(std::string const& s)
{
    if (fromDottedDec(s, this) != MyErrCode::kOk) {
        MY_THROW("invalid ip4: {}", s);
    }
}

Ip4::Ip4(in_addr const& addr) { (*reinterpret_cast<uint32_t*>(this)) = addr.s_addr; }

Ip4::operator in_addr() const
{
    in_addr addr;
    addr.s_addr = *reinterpret_cast<uint32_t const*>(this);
    return addr;
}

Ip4::operator uint32_t() const
{
    auto i = reinterpret_cast<uint32_t const*>(this);
    return *i;
}

bool Ip4::operator==(Ip4 const& rhs) const
{
    return b1 == rhs.b1 && b2 == rhs.b2 && b3 == rhs.b3 && b4 == rhs.b4;
}

bool Ip4::operator!=(Ip4 const& rhs) const { return !(*this == rhs); }

uint32_t Ip4::operator&(Ip4 const& rhs) const
{
    auto i = reinterpret_cast<uint32_t const*>(this);
    auto j = reinterpret_cast<uint32_t const*>(&rhs);
    return ntohl(*i) & ntohl(*j);
}

bool Ip4::isLocal(Ip4 const& rhs, Ip4 const& mask) const { return (*this & mask) == (rhs & mask); }

MyErrCode Ip4::fromDottedDec(std::string const& s, Ip4* ip)
{
    in_addr addr;
    if (inet_pton(AF_INET, s.c_str(), &addr) != 1) {
        DLOG("failed to parse ip4: {}", s);
        return MyErrCode::kFailed;
    }
    if (ip) {
        *ip = Ip4(addr);
    }
    return MyErrCode::kOk;
}

MyErrCode Ip4::fromDomain(std::string const& s, Ip4* ip)
{
    addrinfo hints = {0};
    hints.ai_family = AF_INET;
    hints.ai_flags = AI_PASSIVE;
    hints.ai_protocol = 0;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* first_addr;
    auto ret = getaddrinfo(s.c_str(), nullptr, &hints, &first_addr);
    if (ret != 0 || first_addr == nullptr) {
        DLOG("failed to get ip4 from domain: {}", s);
        return MyErrCode::kFailed;
    }
    auto first_addr_guard = toolkit::scopeExit([&] { freeaddrinfo(first_addr); });
    if (ip) {
        *ip = reinterpret_cast<sockaddr_in*>(first_addr->ai_addr)->sin_addr;
    }
    return MyErrCode::kOk;
}

std::string Ip4::toStr() const
{
    auto c = reinterpret_cast<uint8_t const*>(this);
    std::ostringstream ss;
    ss << static_cast<int>(c[0]);
    for (int i = 1; i < 4; ++i) {
        ss << "." << static_cast<int>(c[i]);
    }
    return ss.str();
}

}  // namespace net