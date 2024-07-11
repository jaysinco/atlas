#pragma once
#include "protocol/protocol.h"
#include <memory>
#include <map>
#include <optional>

class packet
{
public:
    struct detail
    {
        timeval time;                                   // Received time
        std::vector<std::shared_ptr<protocol>> layers;  // Protocol layers
        std::string owner;                              // Program that generates this packet
    };

    packet();

    packet(uint8_t const* const start, uint8_t const* const end, timeval const& tv);

    void to_bytes(std::vector<uint8_t>& bytes) const;

    json const& to_json() const;

    bool link_to(packet const& rhs) const;

    detail const& get_detail() const;

    void set_time(timeval const& tv);

    bool is_error() const;

    bool has_type(std::string const& type) const;

    static timeval gettimeofday();

    static packet arp(mac const& smac, ip4 const& sip, mac const& dmac, ip4 const& dip,
                      bool reply = false, bool reverse = false);

    static packet ping(mac const& smac, ip4 const& sip, mac const& dmac, ip4 const& dip,
                       uint8_t ttl = 128, std::string const& echo = "", bool forbid_slice = false);

private:
    detail d;

    std::optional<json> j_cached;

    std::string get_owner() const;

    using decoder = std::shared_ptr<protocol> (*)(uint8_t const* const start, uint8_t const*& end,
                                                  protocol const* prev);

    static std::map<std::string, decoder> decoder_dict;

    template <typename T>
    static std::shared_ptr<protocol> decode(uint8_t const* const start, uint8_t const*& end,
                                            protocol const* prev)
    {
        return std::make_shared<T>(start, end, prev);
    }
};
