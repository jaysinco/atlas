#include "keycode-converter.h"
#include <linux/input.h>
#include <X11/keysym.h>
#include "imgui/imgui.h"
#include "toolkit/logging.h"

using fmt::enums::format_as;

std::map<uint32_t, KeyCodeConverter::Data> KeyCodeConverter::_key_mapping = {
    {KEY_TAB, {XK_Tab, XK_Tab, ImGuiKey_Tab, ImGuiKey_Tab, -1, -1}},
    {KEY_LEFT, {XK_Left, XK_Left, ImGuiKey_LeftArrow, ImGuiKey_LeftArrow, -1, -1}},
    {KEY_RIGHT, {XK_Right, XK_Right, ImGuiKey_RightArrow, ImGuiKey_RightArrow, -1, -1}},
    {KEY_UP, {XK_Up, XK_Up, ImGuiKey_UpArrow, ImGuiKey_UpArrow, -1, -1}},
    {KEY_DOWN, {XK_Down, XK_Down, ImGuiKey_DownArrow, ImGuiKey_DownArrow, -1, -1}},
    {KEY_PAGEUP, {XK_Page_Up, XK_Page_Up, ImGuiKey_PageUp, ImGuiKey_PageUp, -1, -1}},
    {KEY_PAGEDOWN, {XK_Page_Down, XK_Page_Down, ImGuiKey_PageDown, ImGuiKey_PageDown, -1, -1}},

    {KEY_HOME, {XK_Home, XK_Home, ImGuiKey_Home, ImGuiKey_Home, -1, -1}},
    {KEY_END, {XK_End, XK_End, ImGuiKey_End, ImGuiKey_End, -1, -1}},
    {KEY_INSERT, {XK_Insert, XK_Insert, ImGuiKey_Insert, ImGuiKey_Insert, -1, -1}},
    {KEY_DELETE, {XK_Delete, XK_Delete, ImGuiKey_Delete, ImGuiKey_Delete, -1, -1}},
    {KEY_BACKSPACE, {XK_BackSpace, XK_BackSpace, ImGuiKey_Backspace, ImGuiKey_Backspace, -1, -1}},
    {KEY_ENTER, {XK_Return, XK_Return, ImGuiKey_Enter, ImGuiKey_Enter, -1, -1}},
    {KEY_ESC, {XK_Escape, XK_Escape, ImGuiKey_Escape, ImGuiKey_Escape, -1, -1}},
    {KEY_LEFTSHIFT, {XK_Shift_L, XK_Shift_L, ImGuiKey_LeftShift, ImGuiKey_LeftShift, -1, -1}},
    {KEY_LEFTCTRL, {XK_Control_L, XK_Control_L, ImGuiKey_LeftCtrl, ImGuiKey_LeftCtrl, -1, -1}},
    {KEY_LEFTALT, {XK_Alt_L, XK_Alt_L, ImGuiKey_LeftAlt, ImGuiKey_LeftAlt, -1, -1}},
    {KEY_RIGHTSHIFT, {XK_Shift_R, XK_Shift_R, ImGuiKey_RightShift, ImGuiKey_RightShift, -1, -1}},
    {KEY_RIGHTCTRL, {XK_Control_R, XK_Control_R, ImGuiKey_RightCtrl, ImGuiKey_RightCtrl, -1, -1}},
    {KEY_RIGHTALT, {XK_Alt_R, XK_Alt_R, ImGuiKey_RightAlt, ImGuiKey_RightAlt, -1, -1}},
    {KEY_MENU, {XK_Menu, XK_Menu, ImGuiKey_Menu, ImGuiKey_Menu, -1, -1}},

    {KEY_SPACE, {XK_space, XK_space, ImGuiKey_Space, ImGuiKey_Space, ' ', ' '}},
    {KEY_COMMA, {XK_comma, XK_less, ImGuiKey_Comma, ImGuiKey_Comma, ',', '<'}},
    {KEY_DOT, {XK_period, XK_greater, ImGuiKey_Period, ImGuiKey_Period, '.', '>'}},

    {KEY_KP0, {XK_KP_0, XK_KP_0, ImGuiKey_Keypad0, ImGuiKey_Keypad0, '0', '0'}},
    {KEY_KP1, {XK_KP_1, XK_KP_1, ImGuiKey_Keypad1, ImGuiKey_Keypad1, '1', '1'}},
    {KEY_KP2, {XK_KP_2, XK_KP_2, ImGuiKey_Keypad2, ImGuiKey_Keypad2, '2', '2'}},
    {KEY_KP3, {XK_KP_3, XK_KP_3, ImGuiKey_Keypad3, ImGuiKey_Keypad3, '3', '3'}},
    {KEY_KP4, {XK_KP_4, XK_KP_4, ImGuiKey_Keypad4, ImGuiKey_Keypad4, '4', '4'}},
    {KEY_KP5, {XK_KP_5, XK_KP_5, ImGuiKey_Keypad5, ImGuiKey_Keypad5, '5', '5'}},
    {KEY_KP6, {XK_KP_6, XK_KP_6, ImGuiKey_Keypad6, ImGuiKey_Keypad6, '6', '6'}},
    {KEY_KP7, {XK_KP_7, XK_KP_7, ImGuiKey_Keypad7, ImGuiKey_Keypad7, '7', '7'}},
    {KEY_KP8, {XK_KP_8, XK_KP_8, ImGuiKey_Keypad8, ImGuiKey_Keypad8, '8', '8'}},
    {KEY_KP9, {XK_KP_9, XK_KP_9, ImGuiKey_Keypad9, ImGuiKey_Keypad9, '9', '9'}},

    {KEY_0, {XK_0, XK_0, ImGuiKey_0, ImGuiKey_0, '0', ')'}},
    {KEY_1, {XK_1, XK_1, ImGuiKey_1, ImGuiKey_1, '1', '!'}},
    {KEY_2, {XK_2, XK_2, ImGuiKey_2, ImGuiKey_2, '2', '@'}},
    {KEY_3, {XK_3, XK_3, ImGuiKey_3, ImGuiKey_3, '3', '#'}},
    {KEY_4, {XK_4, XK_4, ImGuiKey_4, ImGuiKey_4, '4', '$'}},
    {KEY_5, {XK_5, XK_5, ImGuiKey_5, ImGuiKey_5, '5', '%'}},
    {KEY_6, {XK_6, XK_6, ImGuiKey_6, ImGuiKey_6, '6', '^'}},
    {KEY_7, {XK_7, XK_7, ImGuiKey_7, ImGuiKey_7, '7', '&'}},
    {KEY_8, {XK_8, XK_8, ImGuiKey_8, ImGuiKey_8, '8', '*'}},
    {KEY_9, {XK_9, XK_9, ImGuiKey_9, ImGuiKey_9, '9', '('}},

    {KEY_A, {XK_a, XK_A, ImGuiKey_A, ImGuiKey_A, 'a', 'A'}},
    {KEY_B, {XK_b, XK_B, ImGuiKey_B, ImGuiKey_B, 'b', 'B'}},
    {KEY_C, {XK_c, XK_C, ImGuiKey_C, ImGuiKey_C, 'c', 'C'}},
    {KEY_D, {XK_d, XK_D, ImGuiKey_D, ImGuiKey_D, 'd', 'D'}},
    {KEY_E, {XK_e, XK_E, ImGuiKey_E, ImGuiKey_E, 'e', 'E'}},
    {KEY_F, {XK_f, XK_F, ImGuiKey_F, ImGuiKey_F, 'f', 'F'}},
    {KEY_G, {XK_g, XK_G, ImGuiKey_G, ImGuiKey_G, 'g', 'G'}},
    {KEY_H, {XK_h, XK_H, ImGuiKey_H, ImGuiKey_H, 'h', 'H'}},
    {KEY_I, {XK_i, XK_I, ImGuiKey_I, ImGuiKey_I, 'i', 'I'}},
    {KEY_J, {XK_j, XK_J, ImGuiKey_J, ImGuiKey_J, 'j', 'J'}},
    {KEY_K, {XK_k, XK_K, ImGuiKey_K, ImGuiKey_K, 'k', 'K'}},
    {KEY_L, {XK_l, XK_L, ImGuiKey_L, ImGuiKey_L, 'l', 'L'}},
    {KEY_M, {XK_m, XK_M, ImGuiKey_M, ImGuiKey_M, 'm', 'M'}},
    {KEY_N, {XK_n, XK_N, ImGuiKey_N, ImGuiKey_N, 'n', 'N'}},
    {KEY_O, {XK_o, XK_O, ImGuiKey_O, ImGuiKey_O, 'o', 'O'}},
    {KEY_P, {XK_p, XK_P, ImGuiKey_P, ImGuiKey_P, 'p', 'P'}},
    {KEY_Q, {XK_q, XK_Q, ImGuiKey_Q, ImGuiKey_Q, 'q', 'Q'}},
    {KEY_R, {XK_r, XK_R, ImGuiKey_R, ImGuiKey_R, 'r', 'R'}},
    {KEY_S, {XK_s, XK_S, ImGuiKey_S, ImGuiKey_S, 's', 'S'}},
    {KEY_T, {XK_t, XK_T, ImGuiKey_T, ImGuiKey_T, 't', 'T'}},
    {KEY_U, {XK_u, XK_U, ImGuiKey_U, ImGuiKey_U, 'u', 'U'}},
    {KEY_V, {XK_v, XK_V, ImGuiKey_V, ImGuiKey_V, 'v', 'V'}},
    {KEY_W, {XK_w, XK_W, ImGuiKey_W, ImGuiKey_W, 'w', 'W'}},
    {KEY_X, {XK_x, XK_X, ImGuiKey_X, ImGuiKey_X, 'x', 'X'}},
    {KEY_Y, {XK_y, XK_Y, ImGuiKey_Y, ImGuiKey_Y, 'y', 'Y'}},
    {KEY_Z, {XK_z, XK_Z, ImGuiKey_Z, ImGuiKey_Z, 'z', 'Z'}},
};

int KeyCodeConverter::ConvertTo(Type type, uint32_t key, bool shift)
{
    if (_key_mapping.find(key) == _key_mapping.end()) {
        ELOG("failed to convert event key to {}: {}", type, key);
        switch (type) {
            case KEYSYM:
                return XK_VoidSymbol;
            case IMGUI:
                return ImGuiKey_None;
            case ASCII:
            default:
                return -1;
        }
    }

    Data& data = _key_mapping[key];
    switch (type) {
        case KEYSYM:
            return shift ? data.keysym_shift : data.keysym;
        case IMGUI:
            return shift ? data.imgui_shift : data.imgui;
        case ASCII:
        default:
            return shift ? data.ascii_shift : data.ascii;
    }
}
