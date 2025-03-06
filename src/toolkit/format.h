#pragma once
#include <filesystem>
#include <fmt/format.h>
#include <fmt/chrono.h>
#include <fmt/ranges.h>
#include <fmt/ostream.h>

#define MY_CONCAT_HELPER(a, b) a##b
#define MY_CONCAT(a, b) MY_CONCAT_HELPER(a, b)
#define FSTR(...) (toolkit::format(__VA_ARGS__))
#define MY_VA_ARGS(...) , ##__VA_ARGS__
#define LOG_FSTR(f, ...) \
    (FSTR("[{}:{}] " f, toolkit::filename(__FILE__), __LINE__ MY_VA_ARGS(__VA_ARGS__)))
#define MY_THROW(...) throw std::runtime_error(LOG_FSTR(__VA_ARGS__))

namespace toolkit
{

template <typename T, typename = void>
struct HasToStr: std::false_type
{
};

template <typename T>
struct HasToStr<T, std::void_t<decltype(std::declval<T>().toStr())>>: std::true_type
{
};

template <typename T>
std::enable_if_t<HasToStr<T>::value, std::string> toString(T&& arg)
{
    return std::forward<T>(arg).toStr();
}

template <typename T, typename = void>
struct CanToString: std::false_type
{
};

template <typename T>
struct CanToString<T, std::void_t<decltype(toString(std::declval<T>()))>>: std::true_type
{
};

template <typename T, typename = void>
struct IsOstreamable: std::false_type
{
};

template <typename T>
struct IsOstreamable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>>
    : std::true_type
{
};

template <typename T>
std::enable_if_t<CanToString<T>::value, std::string> toFormattable(T&& arg)
{
    return toString(std::forward<T>(arg));
}

template <typename T>
std::enable_if_t<!CanToString<T>::value && fmt::is_formattable<T>::value, T&&> toFormattable(
    T&& arg)
{
    return std::forward<T>(arg);
}

template <typename T>
std::enable_if_t<!CanToString<T>::value && !fmt::is_formattable<T>::value &&
                     IsOstreamable<T>::value,
                 decltype(fmt::streamed(std::declval<T>()))>
toFormattable(T&& arg)
{
    return fmt::streamed(std::forward<T>(arg));
}

template <typename T>
std::string format(T&& arg)
{
    return fmt::format("{}", toFormattable(std::forward<T>(arg)));
}

template <typename F, typename... Ts>
std::string format(F&& fmt, Ts&&... args)
{
    return fmt::format(std::forward<F>(fmt), toFormattable(std::forward<Ts>(args))...);
}

inline std::string filename(char const* name)
{
    return std::filesystem::path(name).filename().string();
}

}  // namespace toolkit
