#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <type_traits>
#include <stdexcept>
#include <any>
#include <nlohmann/json_fwd.hpp>

namespace toolkit
{

class Variant
{
public:
    enum Type
    {
        kVoid,
        kBool,    // bool
        kInt,     // int64_t, < 0
        kUint,    // uint64_t, >=0
        kDouble,  // double
        kStr,     // std::string
        kVec,
        kMap,
    };

    using Vec = std::vector<Variant>;
    using Map = std::map<std::string, Variant>;

    // construct
    Variant();
    Variant(bool val);
    Variant(int val);
    Variant(int64_t val);
    Variant(uint32_t val);
    Variant(uint64_t val);
    Variant(double val);
    Variant(char const* val);
    Variant(std::string_view val);
    Variant(std::string const& val);
    Variant(std::string&& val);
    Variant(Vec const& val);
    Variant(Vec&& val);
    Variant(Map const& val);
    Variant(Map&& val);
    Variant(std::initializer_list<Variant> init);

    template <typename T, typename = std::enable_if_t<std::is_enum_v<T>>>
    Variant(T val): Variant(int(val))
    {
    }

    ~Variant();

    // type cast
    bool asBool() const;
    int64_t asInt() const;
    uint64_t asUint() const;
    double asDouble() const;
    std::string& asStr();
    std::string const& asStr() const;
    Vec& asVec();
    Vec const& asVec() const;
    Map& asMap();
    Map const& asMap() const;

    void getTo(bool& val) const;
    void getTo(int& val) const;
    void getTo(int64_t& val) const;
    void getTo(uint64_t& val) const;
    void getTo(float& val) const;
    void getTo(double& val) const;
    void getTo(std::string& val) const;
    void getTo(Vec& val) const;
    void getTo(Map& val) const;

    template <size_t N>
    void getTo(char (&val)[N]) const
    {
        auto& str = asStr();
        if (N < str.size()) {
            throw std::runtime_error("char buffer too small");
        }
        memcpy(val, str.data(), str.size());
    }

    template <typename T, typename = std::enable_if_t<std::is_enum_v<T>>>
    void getTo(T& val) const
    {
        val = static_cast<T>(asInt());
    }

    // getter
    Type getType() const;
    int getSize() const;
    bool isNumber() const;
    bool contains(std::string const& k) const;
    Variant const& operator[](int i) const;
    Variant& operator[](int i);
    Variant const& operator[](std::string const& k) const;
    Variant& operator[](std::string const& k);

    // deleter
    void erase(int i);
    void erase(std::string const& k);

    // export
    nlohmann::json toJson() const;
    std::string toJsonStr(int indent = -1, char indent_char = ' ') const;

    // import
    static Variant fromJson(nlohmann::json const& js);
    static Variant fromJsonStr(std::string_view js);

    // merge & diff
    static void merge(Variant& a, Variant const& b);
    static Variant diff(Variant const& a, Variant const& b);

private:
    static void mergeVec(Vec& a, Vec const& b);
    static void mergeMap(Map& a, Map const& b);
    static Variant diffVec(Vec const& a, Vec const& b);
    static Variant diffMap(Map const& a, Map const& b);

private:
    Type type_;
    std::any val_;
};

bool operator==(Variant const& a, Variant const& b);
bool operator!=(Variant const& a, Variant const& b);

}  // namespace toolkit