#include "context.h"
#include <thread>
#include "traffic/transport.h"
#include "toolkit/toolkit.h"

Context& Context::instance()
{
    static Context ctx;
    return ctx;
}

Context::Context(): capture_should_stop_(false) {}

void Context::startCapture(std::string const& ipstr, std::string const& filter)
{
    std::thread([=]() -> MyErrCode {
        capture_should_stop_ = false;

        auto& apt = net::Adaptor::fit(!ipstr.empty() ? net::Ip4(ipstr) : net::Ip4::kZeros);
        void* handle;
        CHECK_ERR_RET(net::Transport::open(apt, handle));
        auto handle_guard = toolkit::scopeExit([&] { net::Transport::close(handle); });
        ILOG(apt.toVariant().toJsonStr(3));

        if (!filter.empty()) {
            ILOG("set filter \"{}\", mask={}", filter, apt.mask);
            CHECK_ERR_RET(net::Transport::setFilter(handle, filter, apt.mask));
        }

        ILOG("begin to capture...");
        CHECK_ERR_RET(net::Transport::recv(handle, [&](net::Packet&& p) -> bool {
            std::lock_guard<std::mutex> packet_guard(lck_packet_);
            packet_store_.push_back(std::move(p));
            return capture_should_stop_;
        }));
        ILOG("capture stopped");

        return MyErrCode::kOk;
    }).detach();
}

void Context::stopCapture() { capture_should_stop_ = true; }

void Context::pushLog(toolkit::LogLevel level, std::string_view mesg)
{
    std::lock_guard<std::mutex> log_guard(lck_log_);
    log_store_.emplace_back(level, std::string(mesg));
}

void Context::getLog(int idx, toolkit::LogLevel& level, std::string_view& mesg)
{
    std::lock_guard<std::mutex> log_guard(lck_log_);
    auto& item = log_store_.at(idx);
    level = item.first;
    mesg = item.second;
}

int64_t Context::getLogSize()
{
    std::lock_guard<std::mutex> log_guard(lck_log_);
    return log_store_.size();
}