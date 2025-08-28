#pragma once
#include <wayland-client.h>
#include <wayland-cursor.h>
#include <xdg-shell.h>
#include <map>
#include "toolkit/format.h"
#include "toolkit/toolkit.h"

namespace mywl
{

using Uid = toolkit::Uid;

class Context
{
public:
    MyErrCode createDisplay(Uid id);

protected:
    virtual void onRegistry(struct wl_registry* registry, uint32_t name, char const* interface,
                            uint32_t version);
    virtual void onShellPing(struct xdg_wm_base* shell, uint32_t serial);
    virtual void onShellSurfaceConfigure(struct xdg_surface* shell_surface, uint32_t serial);
    virtual void onToplevelConfigure(struct xdg_toplevel* toplevel, int32_t width, int32_t height,
                                     struct wl_array* states);
    virtual void onToplevelClose(struct xdg_toplevel* toplevel);
    virtual void onSeatCapabilities(struct wl_seat* seat, uint32_t caps);
    virtual void onPointerEnter(struct wl_pointer* pointer, uint32_t serial,
                                struct wl_surface* surface, wl_fixed_t sx, wl_fixed_t sy);
    virtual void onPointerLeave(struct wl_pointer* pointer, uint32_t serial,
                                struct wl_surface* surface);
    virtual void onPointerMotion(struct wl_pointer* pointer, uint32_t time, wl_fixed_t sx,
                                 wl_fixed_t sy);
    virtual void onPointerButton(struct wl_pointer* wl_pointer, uint32_t serial, uint32_t time,
                                 uint32_t button, uint32_t state);
    virtual void onPointerAxis(struct wl_pointer* wl_pointer, uint32_t time, uint32_t axis,
                               wl_fixed_t value);
    virtual void onKeyboardKeymap(struct wl_keyboard* keyboard, uint32_t format, int fd,
                                  uint32_t size);
    virtual void onKeyboardEnter(struct wl_keyboard* keyboard, uint32_t serial,
                                 struct wl_surface* surface, struct wl_array* keys);
    virtual void onKeyboardLeave(struct wl_keyboard* keyboard, uint32_t serial,
                                 struct wl_surface* surface);
    virtual void onKeyboardKey(struct wl_keyboard* keyboard, uint32_t serial, uint32_t time,
                               uint32_t key, uint32_t state);
    virtual void onKeyboardModifiers(struct wl_keyboard* keyboard, uint32_t serial,
                                     uint32_t mods_depressed, uint32_t mods_latched,
                                     uint32_t mods_locked, uint32_t group);

private:
    static void handleRegistry(void* data, struct wl_registry* registry, uint32_t name,
                               char const* interface, uint32_t version);
    static void handleShellPing(void* data, struct xdg_wm_base* shell, uint32_t serial);
    static void handleShellSurfaceConfigure(void* data, struct xdg_surface* shell_surface,
                                            uint32_t serial);
    static void handleToplevelConfigure(void* data, struct xdg_toplevel* toplevel, int32_t width,
                                        int32_t height, struct wl_array* states);
    static void handleToplevelClose(void* data, struct xdg_toplevel* toplevel);
    static void handleSeatCapabilities(void* data, struct wl_seat* seat, uint32_t caps);
    static void handlePointerEnter(void* data, struct wl_pointer* pointer, uint32_t serial,
                                   struct wl_surface* surface, wl_fixed_t sx, wl_fixed_t sy);
    static void handlePointerLeave(void* data, struct wl_pointer* pointer, uint32_t serial,
                                   struct wl_surface* surface);
    static void handlePointerMotion(void* data, struct wl_pointer* pointer, uint32_t time,
                                    wl_fixed_t sx, wl_fixed_t sy);
    static void handlePointerButton(void* data, struct wl_pointer* wl_pointer, uint32_t serial,
                                    uint32_t time, uint32_t button, uint32_t state);
    static void handlePointerAxis(void* data, struct wl_pointer* wl_pointer, uint32_t time,
                                  uint32_t axis, wl_fixed_t value);
    static void handleKeyboardKeymap(void* data, struct wl_keyboard* keyboard, uint32_t format,
                                     int fd, uint32_t size);
    static void handleKeyboardEnter(void* data, struct wl_keyboard* keyboard, uint32_t serial,
                                    struct wl_surface* surface, struct wl_array* keys);
    static void handleKeyboardLeave(void* data, struct wl_keyboard* keyboard, uint32_t serial,
                                    struct wl_surface* surface);
    static void handleKeyboardKey(void* data, struct wl_keyboard* keyboard, uint32_t serial,
                                  uint32_t time, uint32_t key, uint32_t state);
    static void handleKeyboardModifiers(void* data, struct wl_keyboard* keyboard, uint32_t serial,
                                        uint32_t mods_depressed, uint32_t mods_latched,
                                        uint32_t mods_locked, uint32_t group);

private:
    std::map<Uid, wl_display*> display_;
    std::map<Uid, wl_registry*> registry_;
    std::map<Uid, wl_compositor*> compositor_;
    std::map<Uid, wl_surface*> surface_;
    std::map<Uid, xdg_wm_base*> shell_;
    std::map<Uid, xdg_surface*> shell_surface_;
    std::map<Uid, xdg_toplevel*> toplevel_;
    std::map<Uid, wl_seat*> seat_;
    std::map<Uid, wl_pointer*> pointer_;
    std::map<Uid, wl_keyboard*> keyboard_;
    std::map<Uid, wl_shm*> shm_;
    std::map<Uid, wl_cursor_theme*> cursor_theme_;
    std::map<Uid, wl_cursor*> cursor_;

    static wl_registry_listener registry_listener;
    static xdg_wm_base_listener shell_listener;
    static xdg_surface_listener shell_surface_listener;
    static xdg_toplevel_listener toplevel_listener;
    static wl_seat_listener seat_listener;
    static wl_pointer_listener pointer_listener;
    static wl_keyboard_listener keyboard_listener;
};

};  // namespace mywl
