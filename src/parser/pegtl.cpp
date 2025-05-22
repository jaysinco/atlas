#include "toolkit/args.h"
#include "toolkit/logging.h"
#include <tao/pegtl.hpp>

namespace pegtl = TAO_PEGTL_NAMESPACE;

namespace hello
{

using namespace TAO_PEGTL_NAMESPACE;

struct Prefix: TAO_PEGTL_STRING("hello, ")
{
};

struct Name: plus<alpha>
{
};

struct Grammar: seq<Prefix, Name, one<'!'>, eof>
{
};

template <typename Rule>
struct Action
{
};

template <>
struct Action<Name>
{
    template <typename ActionInput>
    static void apply(ActionInput const& in, std::string& v)
    {
        v = in.string();
        ILOG("name={}", v);
    }
};

}  // namespace hello

int main(int argc, char** argv)
{
    toolkit::Args args(argc, argv);
    args.parse();

    std::string name;
    pegtl::argv_input in(argv, 1);
    if (pegtl::parse<hello::Grammar, hello::Action>(in, name)) {
        ILOG("goodbye, {}!", name);
    } else {
        ELOG("failed to parse");
    }
}