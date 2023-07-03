#include <boost/variant.hpp>
#include <boost/utility/string_view.hpp>
#include <boost/phoenix/phoenix.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/repository/include/qi_distinct.hpp>
#include <map>
#include <vector>
#include <iostream>
#include <iomanip>

using Source = boost::string_view;
using Location = Source::const_iterator;

namespace completion
{
static int fuzzyMatch(Source input, boost::string_view candidate, int rate = 1)
{  // start with first-letter boost
    int score = 0;

    while (!(input.empty() || candidate.empty())) {
        if (input.front() != candidate.front()) {
            return score + std::max(fuzzyMatch(input.substr(1), candidate,
                                               std::max(rate - 2,
                                                        0)),  // penalty for ignoring an input char
                                    fuzzyMatch(input, candidate.substr(1), std::max(rate - 1, 0)));
        }

        input.remove_prefix(1);
        candidate.remove_prefix(1);
        score += ++rate;
    }
    return score;
}

using Candidates = std::vector<std::string>;

class Hints
{
    struct ByLocation
    {
        template <typename T, typename U>
        bool operator()(T const& a, U const& b) const
        {
            return loc(a) < loc(b);
        }

    private:
        static Location loc(Source const& s) { return s.begin(); }

        static Location loc(Location const& l) { return l; }
    };

public:
    std::map<Location, std::string, ByLocation> incomplete;
    std::map<Source, Candidates, ByLocation> suggestions;

    explicit operator bool() const { return incomplete.size() || suggestions.size(); }
};
}  // namespace completion

namespace ast
{
using NumLiteral = double;
using StringLiteral = std::string;  // de-escaped, not source view

struct Identifier: std::string
{
    using std::string::string;
    using std::string::operator=;
};

struct BinaryExpression;
struct CallExpression;

using Expression = boost::variant<NumLiteral, StringLiteral, Identifier,
                                  boost::recursive_wrapper<BinaryExpression>,
                                  boost::recursive_wrapper<CallExpression> >;

struct BinaryExpression
{
    Expression lhs;
    char op;
    Expression rhs;
};

using ArgList = std::vector<Expression>;

struct CallExpression
{
    Identifier function;
    ArgList args;
};
}  // namespace ast

BOOST_FUSION_ADAPT_STRUCT(ast::BinaryExpression, lhs, op, rhs)
BOOST_FUSION_ADAPT_STRUCT(ast::CallExpression, function, args)

// for debug printing:
namespace
{
struct OnceT
{  // an auto-reset flag

    explicit operator bool()
    {
        bool v = flag;
        flag = false;
        return v;
    }

    bool flag = true;
};
}  // namespace

// for debug printing:
namespace ast
{

inline static std::ostream& operator<<(std::ostream& os, std::vector<Expression> const& args)
{
    os << "(";
    OnceT first;
    for (auto& a: args) {
        os << (first ? "" : ", ") << a;
    }
    return os << ")";
}

inline static std::ostream& operator<<(std::ostream& os, BinaryExpression const& e)
{
    return os << boost::fusion::as_vector(e);
}

inline static std::ostream& operator<<(std::ostream& os, CallExpression const& e)
{
    return os << boost::fusion::as_vector(e);
}
}  // namespace ast

namespace parsing
{
namespace qi = boost::spirit::qi;
namespace phx = boost::phoenix;

template <typename It>
struct Expression: qi::grammar<It, ast::Expression()>
{
    explicit Expression(completion::Hints& hints): Expression::base_type(start_), hints_(hints)
    {
        using namespace qi;

        start_ = skip(space)[expression_];

        expression_ =
            term_[_val = _1] >> *(char_("-+") >> expression_)[_val = make_binary_(_val, _1, _2)];
        term_ = factor_[_val = _1] >> *(char_("*/") >> term_)[_val = make_binary_(_val, _1, _2)];
        factor_ = simple_[_val = _1] >> *(char_("^") >> factor_)[_val = make_binary_(_val, _1, _2)];

        simple_ = call_ | variable_ | compound_ | number_ | string_;

        auto implied = [=](char ch) { return copy(omit[lit(ch) | raw[eps][imply_(_1, ch)]]); };

        variable_ = maybe_known_(phx::ref(variables_));

        compound_ %= '(' >> expression_ >> implied(')');

        // The heuristics:
        // - an unknown identifier followed by (
        // - an unclosed argument list implies )
        call_ %= (known_(phx::ref(functions_))  // known -> imply the parens
                  | &(identifier_ >> '(') >> unknown_(phx::ref(functions_))) >>
                 implied('(') >> -(expression_ % (',' | !(')' | eoi) >> implied(','))) >>
                 implied(')');

        // lexemes, primitive rules
        identifier_ = raw[(alpha | '_') >> *(alnum | '_')];

        // imply the closing quotes
        string_ %= '"' >> *('\\' >> char_ | ~char_('"')) >> implied('"');  // TODO more escapes

        number_ = double_;  // TODO integral arguments

        ///////////////////////////////
        // identifier loopkup, suggesting
        {
            maybe_known_ = known_(domain_) | unknown_(domain_);

            // distinct to avoid partially-matching identifiers
            using boost::spirit::repository::qi::distinct;
            auto kw = distinct(copy(alnum | '_'));

            known_ = raw[kw[lazy(domain_)]];
            unknown_ = raw[identifier_[_val = _1]][suggest_for_(_1, domain_)];
        }

        BOOST_SPIRIT_DEBUG_NODES((
            start_)(expression_)(term_)(factor_)(simple_)(compound_)(call_)(variable_)(identifier_)(number_)(string_)
                                 //(maybe_known)(known)(unknown) // qi::symbols<> non-streamable
        )

        variables_ += "foo", "bar", "qux";
        functions_ += "print", "sin", "tan", "sqrt", "frobnicate";
    }

private:
    completion::Hints& hints_;

    using Domain = qi::symbols<char>;
    Domain variables_, functions_;

    qi::rule<It, ast::Expression()> start_;
    qi::rule<It, ast::Expression(), qi::space_type> expression_, term_, factor_, simple_;
    // completables
    qi::rule<It, ast::Expression(), qi::space_type> compound_;
    qi::rule<It, ast::CallExpression(), qi::space_type> call_;

    // implicit lexemes
    qi::rule<It, ast::Identifier()> variable_, identifier_;
    qi::rule<It, ast::NumLiteral()> number_;
    qi::rule<It, ast::StringLiteral()> string_;

    // domain identifier lookups
    qi::_r1_type domain_;
    qi::rule<It, ast::Identifier(Domain const&)> maybe_known_, known_, unknown_;

    ///////////////////////////////
    // binary expression factory
    struct MakeBinaryF
    {
        ast::BinaryExpression operator()(ast::Expression const& lhs, char op,
                                         ast::Expression const& rhs) const
        {
            return {lhs, op, rhs};
        }
    };

    boost::phoenix::function<MakeBinaryF> make_binary_;

    ///////////////////////////////
    // auto-completing incomplete expressions
    struct ImplyF
    {
        completion::Hints& hints;

        void operator()(boost::iterator_range<It> where, char implied_char) const
        {
            auto inserted = hints.incomplete.emplace(&*where.begin(), std::string(1, implied_char));
            // add the implied char to existing completion
            if (!inserted.second) {
                inserted.first->second += implied_char;
            }
        }
    };

    boost::phoenix::function<ImplyF> imply_{ImplyF{hints_}};

    ///////////////////////////////
    // suggest_for
    struct Suggester
    {
        completion::Hints& hints;

        void operator()(boost::iterator_range<It> where, Domain const& symbols) const
        {
            using namespace completion;
            Source id(&*where.begin(), where.size());
            Candidates c;

            symbols.for_each([&](std::string const& k, ...) { c.push_back(k); });

            auto score = [id](Source v) { return fuzzyMatch(id, v); };
            auto byscore = [=](Source a, Source b) { return score(a) > score(b); };

            sort(c.begin(), c.end(), byscore);
            c.erase(remove_if(c.begin(), c.end(), [=](Source s) { return score(s) < 3; }), c.end());

            hints.suggestions.emplace(id, c);
        }
    };

    boost::phoenix::function<Suggester> suggest_for_{Suggester{hints_}};
};
}  // namespace parsing

int main()
{
    for (Source const input: {
             "",                          // invalid
             "(3",                        // incomplete, imply ')'
             "3*(6+sqrt(9))^7 - 1e8",     // completely valid
             "(3*(((6+sqrt(9))^7 - 1e8",  // incomplete, imply ")))"
             "print(\"hello \\\"world!",  // completes the string literal and the parameter list
             "foo",                       // okay, known variable
             "baz",                       // (suggest bar)
             "baz(",                      // incomplete, imply ')', unknown function
             "taz(",                      // incomplete, imply ')', unknown function
             "san(",                      // 2 suggestions (sin/tan)
             "print(1, 2, \"three\", complicated(san(78",
             "(print sqrt sin 9)    -0) \"bye",
         }) {
        std::cout << "-------------- '" << input << "'\n";
        Location f = input.begin(), l = input.end();

        ast::Expression expr;
        completion::Hints hints;
        bool ok = parse(f, l, parsing::Expression<Location>{hints}, expr);

        if (hints) {
            std::cout << "Input: '" << input << "'\n";
        }
        for (auto& c: hints.incomplete) {
            std::cout << "Missing " << std::setw(c.first - input.begin()) << ""
                      << "^ implied: '" << c.second << "'\n";
        }
        for (auto& id: hints.suggestions) {
            std::cout << "Unknown " << std::setw(id.first.begin() - input.begin()) << ""
                      << std::string(id.first.size(), '^');
            if (id.second.empty()) {
                std::cout << " (no suggestions)\n";
            } else {
                std::cout << " (did you mean ";
                OnceT first;
                for (auto& s: id.second) {
                    std::cout << (first ? "" : " or ") << "'" << s << "'";
                }
                std::cout << "?)\n";
            }
        }

        if (ok) {
            std::cout << "AST:    " << expr << "\n";
        } else {
            std::cout << "Parse failed\n";
        }

        if (f != l) {
            std::cout << "Remaining input: '" << std::string(f, l) << "'\n";
        }
    }
}
