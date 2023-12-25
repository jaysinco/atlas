#pragma once
#include "display-context.h"

class ImeEditor
{
public:
    static void Draw();

private:
    static std::string GetComposition(ImeState& state);
    static std::string GetMenu(ImeState& state);
};