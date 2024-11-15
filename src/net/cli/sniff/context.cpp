#include "context.h"
#include <thread>
#include "traffic/transport.h"
#include "toolkit/toolkit.h"

Context& Context::instance()
{
    static Context ctx;
    return ctx;
}

Context::Context() = default;

void Context::setupCapture(std::string const& ipstr, std::string const& filter)
{
    cap_ipstr_ = ipstr;
    cap_filter_ = filter;
}

void Context::startCapture()
{
    cap_should_stop_ = false;
    auto task = std::packaged_task<MyErrCode()>([=]() -> MyErrCode {
        auto& apt =
            net::Adaptor::fit(!cap_ipstr_.empty() ? net::Ip4(cap_ipstr_) : net::Ip4::kZeros);
        void* handle;
        CHECK_ERR_RET(net::Transport::open(apt, handle));
        auto handle_guard = toolkit::scopeExit([&] { net::Transport::close(handle); });
        ILOG(apt.toVariant().toJsonStr(3));

        if (!cap_filter_.empty()) {
            ILOG("set filter \"{}\", mask={}", cap_filter_, apt.mask);
            CHECK_ERR_RET(net::Transport::setFilter(handle, cap_filter_, apt.mask));
        }

        ILOG("begin to capture...");
        CHECK_ERR_RET(net::Transport::recv(handle, [&](net::Packet&& p) -> bool {
            std::lock_guard<std::mutex> packet_guard(lck_packet_);
            packet_store_.push_back(std::move(p));
            return cap_should_stop_;
        }));
        ILOG("capture stopped");
        return MyErrCode::kOk;
    });
    cap_res_ = task.get_future();
    std::thread(std::move(task)).detach();
}

void Context::stopCapture()
{
    cap_should_stop_ = true;
    cap_res_.get();
    std::lock_guard<std::mutex> packet_guard(lck_packet_);
    packet_store_.clear();
    packet_store_.shrink_to_fit();
}

void Context::pushLog(toolkit::LogLevel level, std::string_view mesg)
{
    std::lock_guard<std::mutex> log_guard(lck_log_);
    log_store_.emplace_back(level, std::string(mesg));
}

bool Context::getLog(int idx, toolkit::LogLevel& level, std::string& mesg)
{
    std::lock_guard<std::mutex> log_guard(lck_log_);
    if (idx < 0 || idx >= log_store_.size()) {
        return false;
    }
    auto& item = log_store_.at(idx);
    level = item.first;
    mesg = item.second;
    return true;
}

int64_t Context::getLogSize()
{
    std::lock_guard<std::mutex> log_guard(lck_log_);
    return log_store_.size();
}

bool Context::getPacket(int idx, net::Packet& pac)
{
    std::lock_guard<std::mutex> packet_guard(lck_packet_);
    if (idx < 0 || idx >= packet_store_.size()) {
        return false;
    }
    pac = packet_store_.at(idx);
    return true;
}

int64_t Context::getPacketSize()
{
    std::lock_guard<std::mutex> packet_guard(lck_packet_);
    return packet_store_.size();
}