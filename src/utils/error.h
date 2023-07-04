#pragma once
#include <nonstd/expected.hpp>

#define CHECK_ERR_RET(err) \
    if (auto _err = (err); _err != MyErrCode::kOk) return _err;

enum class MyErrCode
{
    kOk = 0,
    kUnknown,
    kCancelled,
    kInvalidArgument,
    kDeadlineExceeded,
    kNotFound,
    kAlreadyExists,
    kPermissionDenied,
    kUnauthenticated,
    kResourceExhausted,
    kFailedPrecondition,
    kAborted,
    kOutOfRange,
    kUnimplemented,
    kInternal,
    kUnavailable,
    kDataLoss,
};

namespace utils
{

template <typename T>
using Expected = nonstd::expected<T, MyErrCode>;

nonstd::unexpected_type<MyErrCode> unexpected(MyErrCode err);

template <typename T>
struct ScopeExit
{
    explicit ScopeExit(T&& t): t(std::move(t)) {}

    ~ScopeExit() { t(); }

    T t;
};

template <typename T>
ScopeExit<T> scopeExit(T&& t)
{
    return ScopeExit<T>(std::move(t));
}

}  // namespace utils
