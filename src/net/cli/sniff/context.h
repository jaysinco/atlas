#pragma once
#include <mutex>
#include <atomic>
#include <future>
#include "toolkit/logging.h"
#include "traffic/packet.h"

class Context
{
public:
    static Context& instance();

    void setupCapture(std::string const& ipstr, std::string const& filter);
    void startCapture();
    void stopCapture();
    void pushLog(toolkit::LogLevel level, std::string_view mesg);
    bool getLog(int idx, toolkit::LogLevel& level, std::string& mesg);
    int64_t getLogSize();
    bool getPacket(int idx, net::Packet& pac);
    int64_t getPacketSize();

private:
    Context();

    std::string cap_ipstr_;
    std::string cap_filter_;
    std::future<MyErrCode> cap_res_;
    std::atomic<bool> cap_should_stop_;
    std::mutex lck_log_;
    std::vector<std::pair<toolkit::LogLevel, std::string>> log_store_;
    std::mutex lck_packet_;
    std::vector<net::Packet> packet_store_;
};
