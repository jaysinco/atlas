#include "base.h"
#include "encoding.h"
#include "logging.h"
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <limits.h>
#include <libgen.h>
#include <execinfo.h>
#include <fstream>
#include <sstream>

namespace utils
{

std::filesystem::path projectRoot() { return currentExeDir().parent_path().parent_path(); }

std::filesystem::path currentExeDir() { return currentExePath().parent_path(); }

std::filesystem::path currentExePath()
{
    char cep[PATH_MAX] = {0};
    int n = readlink("/proc/self/exe", cep, PATH_MAX);
    return std::string_view(cep, n);
}

std::string currentExeName() { return currentExePath().filename().string(); }

std::filesystem::path defaultLoggingDir() { return currentExeDir() / "logs"; }

MyErrCode readFile(std::filesystem::path const& path, std::string& content)
{
    std::ifstream in_file(path);
    if (!in_file) {
        ELOG("failed to open file: {}", path.string());
        return MyErrCode::kFailed;
    }
    std::stringstream ss;
    ss << in_file.rdbuf();
    content = ss.str();
    return MyErrCode::kOk;
}

MyErrCode setEnv(char const* varname, char const* value)
{
    return ::setenv(varname, value, 1) == 0 ? MyErrCode::kOk : MyErrCode::kFailed;
}

void handleSIGSEGV(int sig)
{
    fprintf(stderr, "signal SIGSEGV(%d) received!\n", sig);
    constexpr int kAddrsLen = 20;
    void* addrs[kAddrsLen] = {nullptr};
    int size = backtrace(addrs, kAddrsLen);
    backtrace_symbols_fd(addrs, size, STDERR_FILENO);
    std::abort();
}

MyErrCode installCrashHook()
{
    signal(SIGSEGV, handleSIGSEGV);
    return MyErrCode::kOk;
}

}  // namespace utils
