#pragma once

#define CHECK_ERR_RET(_err)          \
    do {                             \
        MyErrCode err = (_err);      \
        if (err != MyErrCode::kOk) { \
            return err;              \
        }                            \
    } while (0)

#define MY_MAIN                                                                       \
    static MyErrCode my_main(int argc, char** argv);                                  \
    int main(int argc, char** argv) { return static_cast<int>(my_main(argc, argv)); } \
    static MyErrCode my_main(int argc, char** argv)

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
