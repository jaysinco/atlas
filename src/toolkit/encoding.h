#pragma once
#include <string_view>
#include <string>

namespace utils
{

enum class CodePage
{
    kLOCAL,
    kUTF8,
    kGBK,
    kWCHAR,
};

std::string ws2s(std::wstring_view ws, CodePage cp = CodePage::kLOCAL);
std::wstring s2ws(std::string_view s, CodePage cp = CodePage::kLOCAL);

}  // namespace utils
