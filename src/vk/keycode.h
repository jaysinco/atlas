#pragma once
#include <cstdint>
#include <map>

class KeyCode
{
public:
    enum Type
    {
        kKeysym,
        kImGui,
        kAscii,
    };

    static int convertTo(Type type, uint32_t key, bool shift);

private:
    struct Data
    {
        int keysym;
        int keysym_shift;
        int imgui;
        int imgui_shift;
        int ascii;
        int ascii_shift;
    };

    static std::map<uint32_t, Data> key_mapping;
};
