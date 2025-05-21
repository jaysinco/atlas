#include "toolkit/args.h"
#include "toolkit/logging.h"
#include <tao/pegtl.hpp>

namespace pegtl = TAO_PEGTL_NAMESPACE;

namespace hello
{

// clang-format off

   struct Prefix: pegtl::string<'H', 'e', 'l', 'l', 'o', ',', ' '> {};
   struct Name: pegtl::plus<pegtl::alpha> {};
   struct Grammar: pegtl::seq<Prefix, Name, pegtl::one<'!'>, pegtl::eof> {};

// clang-format on

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
        ILOG("Good bye, {}!", name);
    } else {
        ELOG("failed to parse");
    }
}