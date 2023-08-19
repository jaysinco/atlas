#include "error.h"

namespace utils
{

nonstd::unexpected_type<MyErrCode> unexpected(MyErrCode err)
{
    return nonstd::unexpected_type<MyErrCode>(err);
}

}  // namespace utils
