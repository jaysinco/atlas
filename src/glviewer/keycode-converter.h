#pragma once
#include <cstdint>
#include <map>

class KeyCodeConverter
{
public:
    enum Type
    {
        KEYSYM,
        IMGUI,
        ASCII,
    };

    static int ConvertTo(Type type, uint32_t key, bool shift);

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

    static std::map<uint32_t, Data> _key_mapping;
};
