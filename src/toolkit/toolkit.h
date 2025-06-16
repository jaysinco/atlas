#pragma once
#include "error.h"
#include <filesystem>
#include <vector>

#define CURR_FILE_DIR() (std::filesystem::path(__FILE__).parent_path())
#define XXD_DECLARE_RES(res)    \
    extern unsigned char res[]; \
    extern unsigned int res##_LEN;
#define XXD_GET_RES(res) (std::string_view(reinterpret_cast<char*>(res), res##_LEN))

namespace toolkit
{

template <typename T>
struct ScopeExit
{
    explicit ScopeExit(T&& t): t(std::move(t)) {}

    ~ScopeExit() { t(); }

    T t;
};

template <typename T>
ScopeExit<T> scopeExit(T&& t)
{
    return ScopeExit<T>(std::move(t));
}

std::string currentExeName();
std::filesystem::path currentExeDir();
std::filesystem::path currentExePath();
std::filesystem::path getLoggingDir();
std::filesystem::path getDataDir();
std::filesystem::path getTempDir();

MyErrCode runAsRoot(int argc, char* argv[]);
MyErrCode installCrashHook();
MyErrCode setEnv(char const* varname, char const* value);
MyErrCode readFile(std::filesystem::path const& path, std::string& content);
MyErrCode readBinaryFile(std::filesystem::path const& path, std::vector<uint8_t>& content);
MyErrCode execsh(std::string const& cmd, std::string& ret);

}  // namespace toolkit
