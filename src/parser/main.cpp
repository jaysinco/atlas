#include "utils/args.h"
#include <boost/phoenix/phoenix.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/include/support_line_pos_iterator.hpp>
#include <fmt/ranges.h>
#include <fmt/ostream.h>

namespace qi = boost::spirit::qi;
namespace enc = boost::spirit::standard_wide;
namespace phx = boost::phoenix;

namespace ast
{
struct Employee
{
    int age;
    std::wstring surname;
    std::wstring forename;
    double salary;
};

}  // namespace ast

BOOST_FUSION_ADAPT_STRUCT(ast::Employee,
                          (int, age)(std::wstring, surname)(std::wstring, forename)(double, salary))

namespace parser
{
template <typename Iterator>
struct ErrorHandler
{
    template <typename, typename, typename, typename>
    struct Result
    {
        typedef void Type;
    };

    explicit ErrorHandler(std::filesystem::path const& source_file): source_file(source_file) {}

    void operator()(Iterator first, Iterator last, Iterator err_pos,
                    boost::spirit::info const& what) const
    {
        Iterator ln_start = boost::spirit::get_line_start(first, err_pos);
        Iterator ln_end = boost::spirit::get_line_end(err_pos, last);
        int ln_pos = std::distance(ln_start, err_pos);
        int line = boost::spirit::get_line(err_pos);
        ELOG("{}({},{}): error: {} expected\n{}\n{}^",
             utils::ws2s(source_file.filename().generic_wstring()), line, ln_pos + 1, what,
             utils::ws2s(std::wstring(ln_start, ln_end)), std::string(ln_pos, ' '));
    }

    std::filesystem::path source_file;
};

template <typename Iterator>
struct Expression: qi::grammar<Iterator, ast::Employee()>
{
    explicit Expression(std::filesystem::path const& source_file)
        : Expression::base_type(start), err_handler(ErrorHandler<Iterator>(source_file))
    {
        quoted = qi::lexeme['"' > *(enc::char_ - '"') > '"'];
        epl = qi::lit(L"employee") > '{' > qi::int_ > ',' > quoted > ',' > quoted > ',' >
              qi::double_ > '}';
        start = qi::skip(enc::space)[qi::eps > epl > qi::eoi];

        qi::on_error<qi::fail>(start, err_handler(qi::_1, qi::_2, qi::_3, qi::_4));
    }

    qi::rule<Iterator, std::wstring()> quoted;
    qi::rule<Iterator, ast::Employee()> epl;
    qi::rule<Iterator, ast::Employee()> start;

    phx::function<ErrorHandler<Iterator>> err_handler;
};

}  // namespace parser

void parsing(std::filesystem::path const& source_file)
{
    using Iterator = boost::spirit::line_pos_iterator<std::wstring::const_iterator>;
    auto raw = utils::readFile(source_file);
    std::wstring input = utils::s2ws(*raw, utils::CodePage::kUTF8);
    Iterator beg(input.begin());
    Iterator end(input.end());
    parser::Expression<Iterator> expr(source_file);
    ast::Employee attr;
    bool ok = qi::parse(beg, end, expr, attr);
    ILOG("{} {}", ok, utils::ws2s(attr.surname));
}

int main(int argc, char** argv)
{
    INIT_LOG(argc, argv);
    parsing(utils::projectRoot() / "src/parser/res/input.txt");
}
