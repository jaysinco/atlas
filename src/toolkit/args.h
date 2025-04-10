#pragma once
#include "error.h"
#include "format.h"
#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

namespace toolkit
{

class Args
{
public:
    Args(int argc, char** argv);
    void optional(char const* name, po::value_semantic const* s, char const* description);
    void positional(char const* name, po::value_semantic const* s, char const* description,
                    int max_count);
    MyErrCode parse(bool init_logger = true);
    bool has(std::string const& name) const;
    Args& addSub(std::string const& name, std::string const& desc);
    bool hasSub(std::string const& name) const;

    template <typename T>
    T const& get(std::string const& name) const
    {
        if (!has(name)) {
            std::cerr << FSTR("Error: argument \"{}\" is not specified", name) << std::endl;
            printUsage();
            std::exit(1);
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
    void printUsage(std::ostream& os = std::cerr) const;
    MyErrCode parse(std::vector<std::string> const& args, bool init_logger);
    MyErrCode parse(po::command_line_parser& parser, bool init_logger);

    struct SubCmd
    {
        std::shared_ptr<Args> args;
        std::string desc;
        bool show;
    };

    int argc_;
    char** argv_;
    std::string program_;
    po::options_description cmd_opt_args_;
    po::options_description log_opt_args_;
    po::options_description pos_opt_args_;
    po::positional_options_description pos_args_;
    po::variables_map vars_;
    std::map<std::string, SubCmd> subs_;
};

}  // namespace toolkit
