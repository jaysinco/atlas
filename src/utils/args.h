#pragma once
#include "logging.h"
#include <boost/program_options.hpp>
#include <iostream>
#define INIT_LOG(argc, argv) (utils::Args(argc, argv).parse())

namespace utils
{

namespace po = boost::program_options;
using po::bool_switch;
using po::value;

class Args
{
public:
    Args(int argc, char** argv);
    void optional(char const* name, po::value_semantic const* s, char const* description);
    void positional(char const* name, po::value_semantic const* s, char const* description,
                    int max_count);
    void parse();
    bool has(std::string const& name) const;
    Args& addSub(std::string const& name, std::string const& desc);
    bool hasSub(std::string const& name) const;

    template <typename T>
    T const& get(std::string const& name) const
    {
        if (!has(name)) {
            MY_THROW("flags not exist: {}", name);
        }
        return vars_[name].as<T>();
    }

private:
    Args();
    bool containOptional() const;
    bool containPositional() const;
    bool containSub() const;
    void addHelpFlags();
    void addLoggingFlags();
    void addSubcommandFlags();
    void printUsage(std::ostream& os = std::cerr);
    void parse(std::vector<std::string> const& args);
    void parse(po::command_line_parser& parser, bool init_logger);

    struct SubCmd
    {
        std::shared_ptr<Args> args;
        std::string desc;
        bool show;
    };

    int argc_;
    char** argv_;
    std::string program_;
    po::options_description opt_args_;
    po::positional_options_description pos_args_;
    po::variables_map vars_;
    std::map<std::string, SubCmd> subs_;
};

}  // namespace utils
