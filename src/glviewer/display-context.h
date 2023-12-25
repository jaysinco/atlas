#pragma once
#include <wayland-egl.h>
#include <EGL/egl.h>
#include <wayland-cursor.h>
#include "xdg-shell-client-protocol.h"
#include <mutex>
#include <string>
#include <vector>

struct ImeState
{
    bool isComposing;
    std::string preEdit;
    int cursorPos;
    int selStart;
    int selEnd;
    int pageNo;
    bool isLastPage;
    std::vector<std::string> candidates;
};

class DisplayContext
{
public:
    static DisplayContext& Instance();

public:
    struct
    {
        wl_display* display;
        wl_registry* registry;
        wl_compositor* compositor;
        wl_shell* shell;
        wl_seat* seat;
        wl_pointer* pointer;
        wl_keyboard* keyboard;
        wl_shm* shm;
        wl_cursor_theme* cursor_theme;
        wl_cursor* default_cursor;
        wl_surface* cursor_surface;
        xdg_wm_base* xdg_wm;
        xdg_surface* xdg_surf;
        xdg_toplevel* xdg_top;

        wl_egl_window* native;
        wl_surface* surface;
        // wl_shell_surface* shell_surface;
        EGLSurface egl_surface;
        wl_callback* callback;
        int fullscreen;
        int configured;
        int opaque;
        int running;
    } wl;

    struct
    {
        EGLDisplay dpy;
        EGLContext ctx;
        EGLConfig conf;
    } egl;

    struct
    {
        int width;
        int height;
    } geometry, window_size;

    struct
    {
        float x;
        float y;
    } mouse_pos;

    struct
    {
        unsigned int vbo;
        unsigned int vao;
        unsigned int ebo;
        unsigned int text;
    } gl;

    struct
    {
        struct
        {
            float x;
            float y;
            float lh;
        } input_region;

        uint64_t session;
        bool actived;
        ImeState state;
        bool ascii_mode;
        bool shift_down;
        bool ctrl_down;
    } ime;
};
