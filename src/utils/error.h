#pragma once
#include <nonstd/expected.hpp>

#define CHECK_ERR_RET(err) \
    if (auto _err = (err); _err != MyErrCode::kOk) return _err;

enum class MyErrCode : int
{
    kOk = 0,
    kFailed,
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

}  // namespace utils
