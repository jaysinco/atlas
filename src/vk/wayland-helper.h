#pragma once
#include <wayland-client.h>
#include <wayland-cursor.h>
#include <xdg-shell.h>
#include <map>
#include "toolkit/format.h"
#include "toolkit/toolkit.h"

#define CHECK_WL(err) \
    if (!(err)) ELOG("failed to call wl")

#define CHECK_WL_RET(err)              \
    do {                               \
        if (!(err)) {                  \
            ELOG("failed to call wl"); \
            return MyErrCode::kFailed; \
        }                              \
    } while (0)

namespace mywl
{

using Uid = toolkit::Uid;

class Surface
{
public:
    Surface();
    Surface(wl_surface* surface, xdg_surface* shell_surface, xdg_toplevel* toplevel);
    operator wl_surface*() const;
    operator bool() const;

private:
    friend class Context;
    wl_surface* surface_;
    xdg_surface* shell_surface_;
    xdg_toplevel* toplevel_;
    bool need_resize_;
    int width_;
    int height_;
};

enum class EventType
{
    kSurfaceClose,
    kSurfaceResize,
    kPointerMove,
    kPointerPress,
    kPointerScroll,
    kKeyboardPress,
};

struct Event
{
    EventType type;
    int ia;
    int ib;
    double da;
    double db;
};

class Context
{
public:
    MyErrCode createDisplay(char const* name = nullptr);
    MyErrCode createSurface(Uid id, char const* app_id, char const* title);
    wl_display* getDisplay();
    Surface& getSurface(Uid id);
    MyErrCode dispatch();
    MyErrCode destroySurface(Uid id);
    MyErrCode destroy();

protected:
    virtual MyErrCode onEvent(Uid surface_id, Event const& event);

private:
    static void handleRegistry(void* data, wl_registry* registry, uint32_t name,
                               char const* interface, uint32_t version);
    static void handleShellPing(void* data, xdg_wm_base* shell, uint32_t serial);
    static void handleShellSurfaceConfigure(void* data, xdg_surface* shell_surface,
                                            uint32_t serial);
    static void handleToplevelConfigure(void* data, xdg_toplevel* toplevel, int32_t width,
                                        int32_t height, wl_array* states);
    static void handleToplevelClose(void* data, xdg_toplevel* toplevel);
    static void handleSeatCapabilities(void* data, wl_seat* seat, uint32_t caps);
    static void handlePointerEnter(void* data, wl_pointer* pointer, uint32_t serial,
                                   wl_surface* surface, wl_fixed_t sx, wl_fixed_t sy);
    static void handlePointerLeave(void* data, wl_pointer* pointer, uint32_t serial,
                                   wl_surface* surface);
    static void handlePointerMotion(void* data, wl_pointer* pointer, uint32_t time, wl_fixed_t sx,
                                    wl_fixed_t sy);
    static void handlePointerButton(void* data, wl_pointer* pointer, uint32_t serial, uint32_t time,
                                    uint32_t button, uint32_t state);
    static void handlePointerAxis(void* data, wl_pointer* pointer, uint32_t time, uint32_t axis,
                                  wl_fixed_t value);
    static void handleKeyboardKeymap(void* data, wl_keyboard* keyboard, uint32_t format, int fd,
                                     uint32_t size);
    static void handleKeyboardEnter(void* data, wl_keyboard* keyboard, uint32_t serial,
                                    wl_surface* surface, wl_array* keys);
    static void handleKeyboardLeave(void* data, wl_keyboard* keyboard, uint32_t serial,
                                    wl_surface* surface);
    static void handleKeyboardKey(void* data, wl_keyboard* keyboard, uint32_t serial, uint32_t time,
                                  uint32_t key, uint32_t state);
    static void handleKeyboardModifiers(void* data, wl_keyboard* keyboard, uint32_t serial,
                                        uint32_t mods_depressed, uint32_t mods_latched,
                                        uint32_t mods_locked, uint32_t group);

    Uid getSurfaceId(wl_surface* surface);
    Uid getSurfaceId(xdg_surface* shell_surface);
    Uid getSurfaceId(xdg_toplevel* toplevel);

private:
    wl_display* display_;
    wl_registry* registry_;
    wl_compositor* compositor_;
    xdg_wm_base* shell_;
    wl_seat* seat_;
    wl_shm* shm_;
    wl_pointer* pointer_;
    wl_keyboard* keyboard_;
    wl_cursor_theme* cursor_theme_;
    wl_cursor* cursor_;
    wl_surface* cursor_surface_;

    std::map<Uid, Surface> surfaces_;

    Uid pointer_surface_id_ = Uid::kNull;
    Uid keyboard_surface_id_ = Uid::kNull;

    static wl_registry_listener registry_listener;
    static xdg_wm_base_listener shell_listener;
    static xdg_surface_listener shell_surface_listener;
    static xdg_toplevel_listener toplevel_listener;
    static wl_seat_listener seat_listener;
    static wl_pointer_listener pointer_listener;
    static wl_keyboard_listener keyboard_listener;
};

};  // namespace mywl
