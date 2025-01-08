#pragma once
#include <filesystem>
#include <fmt/format.h>
#include <fmt/chrono.h>

#define FSTR(...) (toolkit::format(__VA_ARGS__))
#define TOSTR(v) (FSTR("{}", v))
#define MY_VA_ARGS(...) , ##__VA_ARGS__
#define LOG_FSTR(f, ...) (FSTR("[{}:{}] " f, toolkit::filename(__FILE__), __LINE__ MY_VA_ARGS(__VA_ARGS__)))
#define MY_THROW(...) throw std::runtime_error(LOG_FSTR(__VA_ARGS__))

#define DEFINE_FORMATTER(type, x)                                   \
    template <>                                                     \
    struct fmt::formatter<type>                                     \
    {                                                               \
        template <typename Context>                                 \
        constexpr auto parse(Context& ctx)                          \
        {                                                           \
            return ctx.begin();                                     \
        }                                                           \
        template <typename Context>                                 \
        constexpr auto format(type const& item, Context& ctx) const \
        {                                                           \
            return format_to(ctx.out(), "{}", (x));                 \
        }                                                           \
    }

namespace toolkit
{

template <typename... Ts>
std::string format(Ts&&... args)
{
    if constexpr (sizeof...(Ts) == 1) {
        return std::string(std::forward<Ts>(args)...);
    } else {
        return fmt::format(std::forward<Ts>(args)...);
    }
}

inline std::string filename(char const* name)
{
    return std::filesystem::path(name).filename().string();
}

}  // namespace toolkit
