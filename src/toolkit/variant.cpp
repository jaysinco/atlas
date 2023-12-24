#include "variant.h"
#include <nlohmann/json.hpp>
#include <fmt/format.h>

namespace toolkit
{

using fmt::enums::format_as;

Variant::Variant(): type_(kVoid) {}

Variant::Variant(bool val): type_(kBool), val_(val) {}

Variant::Variant(int val): Variant(static_cast<int64_t>(val)) {}

Variant::Variant(int64_t val)
{
    if (val >= 0) {
        *this = Variant(static_cast<uint64_t>(val));
    } else {
        type_ = kInt;
        val_ = val;
    }
}

Variant::Variant(uint64_t val): type_(kUint), val_(val) {}

Variant::Variant(double val): type_(kDouble), val_(val) {}

Variant::Variant(char const* val): Variant(std::string(val)) {}

Variant::Variant(std::string_view val): Variant(std::string(val)) {}

Variant::Variant(std::string const& val): type_(kStr), val_(val) {}

Variant::Variant(std::string&& val): type_(kStr), val_(std::move(val)) {}

Variant::Variant(Vec const& val): type_(kVec), val_(val) {}

Variant::Variant(Vec&& val): type_(kVec), val_(std::move(val)) {}

Variant::Variant(Map const& val): type_(kMap), val_(val) {}

Variant::Variant(Map&& val): type_(kMap), val_(std::move(val)) {}

Variant::Variant(std::initializer_list<Variant> init)
{
    bool is_a_map = std::all_of(init.begin(), init.end(), [](Variant const& val) {
        return val.type_ == kVec && val.getSize() == 2 && val[0].type_ == kStr;
    });

    if (is_a_map) {
        type_ = kMap;
        Map map;
        for (auto const& i: init) {
            map.emplace(i[0].asStr(), i[1]);
        }
        val_ = std::move(map);
    } else {
        type_ = kVec;
        Vec vec;
        for (auto const& i: init) {
            vec.push_back(i);
        }
        val_ = std::move(vec);
    }
}

Variant::~Variant() = default;

Variant::Type Variant::getType() const { return type_; }

bool Variant::asBool() const
{
    if (auto* v = std::any_cast<bool>(&val_)) {
        return *v;
    } else if (auto* v = std::any_cast<int64_t>(&val_)) {
        return *v;
    } else if (auto* v = std::any_cast<uint64_t>(&val_)) {
        return *v;
    } else {
        throw std::runtime_error("variant bad cast to bool");
    }
}

int64_t Variant::asInt() const
{
    if (auto* v = std::any_cast<int64_t>(&val_)) {
        return *v;
    } else if (auto* v = std::any_cast<uint64_t>(&val_)) {
        return *v;
    }
    if (auto* v = std::any_cast<double>(&val_)) {
        return *v;
    } else {
        throw std::runtime_error("variant bad cast to int");
    }
}

uint64_t Variant::asUint() const
{
    if (auto* v = std::any_cast<int64_t>(&val_)) {
        return *v;
    } else if (auto* v = std::any_cast<uint64_t>(&val_)) {
        return *v;
    }
    if (auto* v = std::any_cast<double>(&val_)) {
        return *v;
    } else {
        throw std::runtime_error("variant bad cast to uint");
    }
}

double Variant::asDouble() const
{
    if (auto* v = std::any_cast<int64_t>(&val_)) {
        return *v;
    } else if (auto* v = std::any_cast<uint64_t>(&val_)) {
        return *v;
    }
    if (auto* v = std::any_cast<double>(&val_)) {
        return *v;
    } else {
        throw std::runtime_error("variant bad cast to double");
    }
}

std::string& Variant::asStr()
{
    return const_cast<std::string&>((const_cast<Variant const*>(this)->asStr()));
}

std::string const& Variant::asStr() const
{
    if (auto* v = std::any_cast<std::string>(&val_)) {
        return *v;
    } else {
        throw std::runtime_error("variant bad cast to string");
    }
}

Variant::Vec& Variant::asVec()
{
    return const_cast<Variant::Vec&>((const_cast<Variant const*>(this))->asVec());
}

Variant::Vec const& Variant::asVec() const
{
    if (auto* v = std::any_cast<Vec>(&val_)) {
        return *v;
    } else {
        throw std::runtime_error("variant bad cast to vector");
    }
}

Variant::Map& Variant::asMap()
{
    return const_cast<Variant::Map&>((const_cast<Variant const*>(this))->asMap());
}

Variant::Map const& Variant::asMap() const
{
    if (auto* v = std::any_cast<Map>(&val_)) {
        return *v;
    } else {
        throw std::runtime_error("variant bad cast to map");
    }
}

bool Variant::isNumber() const { return type_ == kInt || type_ == kUint || type_ == kDouble; }

void Variant::getTo(bool& val) const { val = asBool(); }

void Variant::getTo(int& val) const { val = asInt(); }

void Variant::getTo(int64_t& val) const { val = asInt(); }

void Variant::getTo(uint64_t& val) const { val = asUint(); }

void Variant::getTo(float& val) const { val = asDouble(); }

void Variant::getTo(double& val) const { val = asDouble(); }

void Variant::getTo(std::string& val) const { val = asStr(); }

void Variant::getTo(Vec& val) const { val = asVec(); }

void Variant::getTo(Map& val) const { val = asMap(); }

int Variant::getSize() const
{
    if (type_ == kVec) {
        return asVec().size();
    } else if (type_ == kMap) {
        return asMap().size();
    } else {
        throw std::runtime_error(fmt::format("variant type don't have size: {}", type_));
    }
}

bool Variant::contains(std::string const& k) const
{
    auto& m = asMap();
    return m.find(k) != m.end();
}

Variant const& Variant::operator[](int i) const { return asVec().at(i); }

Variant& Variant::operator[](int i)
{
    return const_cast<Variant&>((*const_cast<Variant const*>(this))[i]);
}

Variant const& Variant::operator[](std::string const& k) const { return asMap().at(k); }

Variant& Variant::operator[](std::string const& k)
{
    if (type_ == kVoid) {
        *this = Map{};
    }
    auto& m = asMap();
    if (m.find(k) == m.end()) {
        m[k] = Map{};
    }
    return m.at(k);
}

void Variant::erase(int i)
{
    auto& v = asVec();
    if (i < 0 || i >= v.size()) {
        throw std::runtime_error(fmt::format("variant erase wrong index: {}", i));
    }
    v.erase(v.begin() + i);
}

void Variant::erase(std::string const& k)
{
    auto& m = asMap();
    auto it = m.find(k);
    if (it == m.end()) {
        throw std::runtime_error(fmt::format("variant erase wrong key: {}", k));
    }
    m.erase(it);
}

std::string Variant::toJsonStr() const { return toJson().dump(); }

nlohmann::json Variant::toJson() const
{
    switch (type_) {
        case kVoid:
            return nlohmann::json{};
        case kBool:
            return asBool();
        case kInt:
            return asInt();
        case kUint:
            return asUint();
        case kDouble:
            return asDouble();
        case kStr:
            return asStr();
        case kVec: {
            auto js = nlohmann::json::array();
            for (auto& item: asVec()) {
                js.push_back(item.toJson());
            }
            return js;
        }
        case kMap: {
            auto js = nlohmann::json::object();
            for (auto& [name, item]: asMap()) {
                js[name] = item.toJson();
            }
            return js;
        }
        default:
            throw std::runtime_error(fmt::format("variant bad type: {}", type_));
    }
}

Variant Variant::fromJsonStr(std::string_view js) { return fromJson(nlohmann::json::parse(js)); }

Variant Variant::fromJson(nlohmann::json const& js)
{
    auto type = js.type();
    switch (type) {
        case nlohmann::json::value_t::array: {
            Vec v;
            for (auto& i: js) {
                v.push_back(fromJson(i));
            }
            return v;
        }
        case nlohmann::json::value_t::object: {
            Map m;
            for (auto& [k, v]: js.items()) {
                m[k] = fromJson(v);
            }
            return m;
        }
        case nlohmann::json::value_t::string:
            return js.get<std::string>();
        case nlohmann::json::value_t::boolean:
            return js.get<bool>();
        case nlohmann::json::value_t::number_integer:
            return js.get<int64_t>();
        case nlohmann::json::value_t::number_unsigned:
            return js.get<uint64_t>();
        case nlohmann::json::value_t::number_float:
            return js.get<double>();
        case nlohmann::json::value_t::null:
            return {};
        case nlohmann::json::value_t::binary:
        case nlohmann::json::value_t::discarded:
        default:
            throw std::runtime_error(fmt::format("json bad type: {}", fmt::underlying(type)));
    }
}

void Variant::merge(Variant& a, Variant const& b)
{
    Type type_a = a.getType();
    Type type_b = b.getType();
    if (type_a == kVoid) {
        a = b;
        return;
    }
    if (type_b == kVoid) {
        return;
    }
    if (type_a != type_b) {
        if (a.isNumber() && b.isNumber()) {
            a = b;
            return;
        }
        throw std::runtime_error(fmt::format("variant merge bad type: {} <- {}", type_a, type_b));
    }
    switch (type_a) {
        case kVoid:
        case kBool:
        case kInt:
        case kUint:
        case kDouble:
            a = b;
            return;
        case kStr:
            if (a != b) {
                a = b;
            }
            return;
        case kVec: {
            mergeVec(a.asVec(), b.asVec());
            return;
        }
        case kMap: {
            mergeMap(a.asMap(), b.asMap());
            return;
        }
        default:
            throw std::runtime_error(fmt::format("variant bad type: {}", type_a));
    }
}

void Variant::mergeVec(Vec& a, Vec const& b)
{
    if (a.size() != b.size()) {
        throw std::runtime_error(
            fmt::format("variant merge bad size: {} != {}", a.size(), b.size()));
    }
    for (int i = 0; i < a.size(); ++i) {
        merge(a[i], b[i]);
    }
}

void Variant::mergeMap(Map& a, Map const& b)
{
    for (auto& [k, v]: b) {
        if (a.find(k) != a.end()) {
            merge(a.at(k), b.at(k));
        } else {
            a[k] = b.at(k);
        }
    }
}

bool operator!=(Variant const& a, Variant const& b) { return !(a == b); }

bool operator==(Variant const& a, Variant const& b)
{
    Variant::Type type_a = a.getType();
    Variant::Type type_b = b.getType();
    if (type_a != type_b) {
        return false;
    }
    switch (type_a) {
        case Variant::kVoid:
            return true;
        case Variant::kBool:
            return a.asBool() == b.asBool();
        case Variant::kInt:
            return a.asInt() == b.asInt();
        case Variant::kUint:
            return a.asUint() == b.asUint();
        case Variant::kDouble:
            return a.asDouble() == b.asDouble();
        case Variant::kStr:
            return a.asStr() == b.asStr();
        case Variant::kVec: {
            if (a.getSize() != b.getSize()) {
                return false;
            }
            auto& av = a.asVec();
            auto& bv = b.asVec();
            for (int i = 0; i < av.size(); ++i) {
                if (av.at(i) != bv.at(i)) {
                    return false;
                }
            }
            return true;
        }
        case Variant::kMap: {
            if (a.getSize() != b.getSize()) {
                return false;
            }
            auto& am = a.asMap();
            auto& bm = b.asMap();
            for (auto& [k, v]: am) {
                if (bm.find(k) == bm.end() || v != bm.at(k)) {
                    return false;
                }
            }
            return true;
        }
        default:
            throw std::runtime_error(fmt::format("variant bad type: {}", type_a));
    }
}

Variant Variant::diff(Variant const& a, Variant const& b)
{
    Type type_a = a.getType();
    Type type_b = b.getType();
    if (type_a != type_b) {
        if (a.isNumber() && b.isNumber()) {
            return a;
        }
        throw std::runtime_error(fmt::format("variant diff bad type: {} vs {}", type_a, type_b));
    }
    switch (type_a) {
        case kVoid:
        case kBool:
        case kInt:
        case kUint:
        case kDouble:
        case kStr:
            return a == b ? Variant{} : a;
        case kVec:
            return diffVec(a.asVec(), b.asVec());
        case kMap:
            return diffMap(a.asMap(), b.asMap());
        default:
            throw std::runtime_error(fmt::format("variant bad type: {}", type_a));
    }
}

Variant Variant::diffVec(Vec const& a, Vec const& b)
{
    if (a.size() != b.size()) {
        throw std::runtime_error(
            fmt::format("variant diff bad size: {} != {}", a.size(), b.size()));
    }
    if (a == b) {
        return {};
    }
    Vec c;
    for (int i = 0; i < a.size(); ++i) {
        c.push_back(diff(a[i], b[i]));
    }
    return c;
}

Variant Variant::diffMap(Map const& a, Map const& b)
{
    if (a.size() != b.size()) {
        throw std::runtime_error(
            fmt::format("variant diff bad size: {} != {}", a.size(), b.size()));
    }
    if (a == b) {
        return {};
    }
    Map c;
    for (auto& [k, v]: b) {
        if (a.find(k) != a.end()) {
            auto& va = a.at(k);
            if (va != v) {
                c[k] = diff(va, v);
            }
        } else {
            throw std::runtime_error(fmt::format("variant diff bad key: {}", k));
        }
    }
    return c;
}

}  // namespace toolkit