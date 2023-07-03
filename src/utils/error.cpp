#include "error.h"

namespace utils
{

nonstd::unexpected_type<Error> unexpected(std::string const& s)
{
    return nonstd::unexpected_type<Error>(s);
}

}  // namespace utils
