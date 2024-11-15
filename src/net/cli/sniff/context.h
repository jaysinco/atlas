#pragma once
#include <mutex>
#include <atomic>
#include "toolkit/logging.h"
#include "traffic/packet.h"

class Context
{
public:
    static Context& instance();

    void startCapture(std::string const& ipstr, std::string const& filter);
    void stopCapture();

    void pushLog(toolkit::LogLevel level, std::string_view mesg);
    void getLog(int idx, toolkit::LogLevel& level, std::string_view& mesg);
    int64_t getLogSize();

private:
    Context();

    std::mutex lck_log_;
    std::vector<std::pair<toolkit::LogLevel, std::string>> log_store_;
    std::mutex lck_packet_;
    std::vector<net::Packet> packet_store_;
    std::atomic<bool> capture_should_stop_;
};
