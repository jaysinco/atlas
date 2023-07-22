#pragma once
#include "error.h"
#include <filesystem>

#define CURR_FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define XXD_DECLARE_RES(res)    \
    extern unsigned char res[]; \
    extern unsigned int res##_LEN;
#define XXD_GET_RES(res) (std::string_view(reinterpret_cast<char*>(res), res##_LEN))

namespace utils
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
std::filesystem::path projectRoot();
std::filesystem::path currentExeDir();
std::filesystem::path currentExePath();
std::filesystem::path defaultLoggingDir();

MyErrCode installCrashHook();
MyErrCode setEnv(char const* varname, char const* value);
Expected<std::string> readFile(std::filesystem::path const& path);

}  // namespace utils
