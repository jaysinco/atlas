#include "args.h"
#include "logging.h"

namespace toolkit
{

Args::Args()
    : cmd_opt_args_("Optional arguments", 70),
      log_opt_args_("Logging arguments", 70),
      pos_opt_args_("Positional arguments", 70)
{
}

Args::Args(int argc, char** argv): Args()
{
    argc_ = argc;
    argv_ = argv;
    program_ = std::filesystem::path(argv[0]).filename().string();
}

void Args::optional(char const* name, po::value_semantic const* s, char const* description)
{
    cmd_opt_args_.add_options()(name, s, description);
}

void Args::positional(char const* name, po::value_semantic const* s, char const* description,
                      int max_count)
{
    pos_opt_args_.add_options()(name, s, description);
    std::string name_pos = name;
    name_pos = name_pos.substr(0, name_pos.find_first_of(','));
    pos_args_.add(name_pos.c_str(), max_count);
}

Args& Args::addSub(std::string const& name, std::string const& desc)
{
    auto pargs = std::shared_ptr<Args>(new Args);
    pargs->program_ = FSTR("{} {}", program_, name);
    subs_[name] = SubCmd{pargs, desc, false};
    return *pargs;
}

MyErrCode Args::parse(bool init_logger)
{
    po::command_line_parser parser(argc_, argv_);
    CHECK_ERR_RET(parse(parser, init_logger));
    return MyErrCode::kOk;
}

MyErrCode Args::parse(std::vector<std::string> const& args, bool init_logger)
{
    po::command_line_parser parser(args);
    CHECK_ERR_RET(parse(parser, init_logger));
    return MyErrCode::kOk;
}

MyErrCode Args::parse(po::command_line_parser& parser, bool init_logger)
{
    try {
        if (containSub()) {
            parser.allow_unregistered();
            addSubcommandFlags();
            addSub("help", "shows help message and exits");
        } else {
            addHelpFlags();
        }
        if (init_logger) {
            addLoggingFlags();
        }

        po::options_description all_opt_args;
        all_opt_args.add(cmd_opt_args_).add(log_opt_args_).add(pos_opt_args_);
        parser.options(all_opt_args);
        if (containPositional()) {
            parser.positional(pos_args_);
        }

        auto parsed_result =
            parser
                .style(po::command_line_style::unix_style & ~po::command_line_style::allow_guessing)
                .run();
        po::store(parsed_result, vars_);

        if (init_logger) {
            LoggerOption opt;
            opt.program = program_;
            opt.logtostderr = get<bool>("logtostderr");
            opt.logtofile = get<bool>("logtofile");
            opt.loglevel = static_cast<LogLevel>(get<int>("loglevel"));
            opt.logbuflevel = static_cast<LogLevel>(get<int>("logbuflevel"));
            opt.logbufsecs = get<int>("logbufsecs");
            opt.maxlogsize = get<int>("maxlogsize");
            CHECK_ERR_RET(initLogger(std::move(opt)));
        }

        if (containSub()) {
            std::string cmd = get<std::string>("command");
            auto it = subs_.find(cmd);
            if (it == subs_.end()) {
                std::cerr << FSTR("Error: command \"{}\" is not supported", cmd) << std::endl;
                printUsage();
                std::exit(1);
            }
            it->second.show = true;
            if (it->first == "help") {
                printUsage();
                std::exit(0);
            } else {
                auto args = po::collect_unrecognized(parsed_result.options, po::include_positional);
                args.erase(args.begin());
                it->second.args->parse(args, false);
            }
        } else {
            if (get<bool>("help")) {
                printUsage();
                std::exit(0);
            }
        }
        return MyErrCode::kOk;
    } catch (std::exception const& err) {
        std::cerr << FSTR("Error: {}", err.what()) << std::endl;
        printUsage();
        std::exit(1);
    }
}

void Args::printUsage(std::ostream& os) const
{
    os << std::endl;
    os << "Usage: " << program_;
    if (containOptional()) {
        os << " [options]";
    }
    if (containPositional()) {
        std::string last = "";
        for (int i = 0; i < pos_args_.max_total_count(); ++i) {
            auto& name = pos_args_.name_for_position(i);
            if (name == last) {
                os << "...";
                break;
            }
            last = name;
            boost::any dummy;
            auto semantic = pos_opt_args_.find(name, false).semantic();
            bool has_default = semantic->apply_default(dummy);
            os << " " << (has_default ? "[" : "") << name << (has_default ? "]" : "");
        }
    }
    os << std::endl;
    if (containSub()) {
        os << std::endl;
        os << "Subcommands:" << std::endl;
        size_t wmax = 0;
        for (auto& [k, v]: subs_) {
            wmax = std::max(wmax, k.size());
        }
        for (auto& [k, v]: subs_) {
            os << "  " << k << std::string(wmax - k.size() + 4, ' ') << v.desc << std::endl;
        }
    }
    if (!cmd_opt_args_.options().empty()) {
        os << std::endl;
        os << cmd_opt_args_;
    }
    if (!containSub() && containPositional()) {
        os << std::endl;
        os << "Positional arguments:" << std::endl;
        size_t wmax = 0;
        for (auto& v: pos_opt_args_.options()) {
            auto name = FSTR("{}{}", v->long_name(), v->semantic()->name().substr(3));
            wmax = std::max(wmax, name.size());
        }
        for (auto& v: pos_opt_args_.options()) {
            auto name = FSTR("{}{}", v->long_name(), v->semantic()->name().substr(3));
            os << "  " << name << std::string(wmax - name.size() + 4, ' ') << v->description()
               << std::endl;
        }
    }
    if (!log_opt_args_.options().empty()) {
        os << std::endl;
        os << log_opt_args_;
    }
    os << std::endl;
}

bool Args::has(std::string const& name) const { return vars_.count(name) != 0; }

bool Args::hasSub(std::string const& name) const
{
    return subs_.count(name) > 0 && subs_.at(name).show;
}

bool Args::containOptional() const
{
    return !cmd_opt_args_.options().empty() || !log_opt_args_.options().empty() ||
           !pos_opt_args_.options().empty();
}

bool Args::containPositional() const { return pos_args_.max_total_count() > 0; }

bool Args::containSub() const { return !subs_.empty(); }

void Args::addHelpFlags() { optional("help,h", po::bool_switch(), "shows help message and exits"); }

void Args::addLoggingFlags()
{
    auto args_init = log_opt_args_.add_options();
    args_init("loglevel,v", po::value<int>()->default_value(static_cast<int>(LogLevel::kINFO)),
              "log level (0-6)");
    args_init("logbuflevel", po::value<int>()->default_value(static_cast<int>(LogLevel::kERROR)),
              "log buffered level");
    args_init("logbufsecs", po::value<int>()->default_value(30), "max secs logs may be buffered");
    args_init("logtostderr", po::value<bool>()->default_value(true), "log to stderr");
    args_init("logtofile", po::value<bool>()->default_value(false), "log to file");
    args_init("maxlogsize", po::value<int>()->default_value(100), "max log file size (MB)");
}

void Args::addSubcommandFlags()
{
    positional("command", po::value<std::string>()->default_value("help"), "command to execute", 1);
    positional("args", po::value<std::vector<std::string>>(), "arguments for command", -1);
}

}  // namespace toolkit
