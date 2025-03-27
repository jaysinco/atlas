#pragma once

#define CHECK_ERR_RET(err) \
    if (auto _err = (err); _err != MyErrCode::kOk) return _err;

#define CHECK_ERR_RTI(err) \
    if (auto _err = (err); _err != MyErrCode::kOk) return static_cast<int>(_err);

enum class MyErrCode : int
{
    kOk = 0,
    kFailed,
    kUnknown,
    kException,
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
    kTimeout,
    kUnimplemented,
    kInternal,
    kUnavailable,
    kDataLoss,
};
