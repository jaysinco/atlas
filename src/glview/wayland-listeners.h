#pragma once
#include <wayland-client.h>
#include "xdg-shell-client-protocol.h"

class WaylandListeners
{
public:
    static xdg_surface_listener xdg_surface;
    static xdg_toplevel_listener xdg_toplevel;
    static wl_seat_listener seat;
    static wl_registry_listener registry;
    static wl_callback_listener configure_cb;
    static wl_callback_listener frame_cb;
    static wl_pointer_listener pointer;
    static wl_keyboard_listener keyboard;
    static xdg_wm_base_listener xdgwm;

public:
    // frame_cb
    static void redraw(void* data, struct wl_callback* callback, uint32_t time);

    // xdg_surface
    static void handle_xdg_surface_configure(void* data, struct xdg_surface* xdg_surface,
                                             uint32_t serial);

    // xdg_toplevel
    static void handle_xdg_toplevel_configure(void* data, struct xdg_toplevel* xdg_toplevel,
                                              int32_t width, int32_t height,
                                              struct wl_array* states);
    static void handle_xdg_toplevel_close(void* data, struct xdg_toplevel* xdg_toplevel);

    // seat
    static void seat_handle_capabilities(void* data, struct wl_seat* seat, uint32_t caps);

    // registry
    static void registry_handle_global_remove(void* data, struct wl_registry* registry,
                                              uint32_t name);
    static void registry_handle_global(void* data, struct wl_registry* registry, uint32_t name,
                                       char const* interface, uint32_t version);

    // configure_cb
    static void configure_callback(void* data, struct wl_callback* callback, uint32_t time);

    // pointer
    static void pointer_handle_enter(void* data, struct wl_pointer* pointer, uint32_t serial,
                                     struct wl_surface* surface, wl_fixed_t sx, wl_fixed_t sy);
    static void pointer_handle_leave(void* data, struct wl_pointer* pointer, uint32_t serial,
                                     struct wl_surface* surface);
    static void pointer_handle_motion(void* data, struct wl_pointer* pointer, uint32_t time,
                                      wl_fixed_t sx, wl_fixed_t sy);
    static void pointer_handle_button(void* data, struct wl_pointer* wl_pointer, uint32_t serial,
                                      uint32_t time, uint32_t button, uint32_t state);
    static void pointer_handle_axis(void* data, struct wl_pointer* wl_pointer, uint32_t time,
                                    uint32_t axis, wl_fixed_t value);

    // keyboard
    static void keyboard_handle_keymap(void* data, struct wl_keyboard* keyboard, uint32_t format,
                                       int fd, uint32_t size);
    static void keyboard_handle_enter(void* data, struct wl_keyboard* keyboard, uint32_t serial,
                                      struct wl_surface* surface, struct wl_array* keys);
    static void keyboard_handle_leave(void* data, struct wl_keyboard* keyboard, uint32_t serial,
                                      struct wl_surface* surface);
    static void keyboard_handle_key(void* data, struct wl_keyboard* keyboard, uint32_t serial,
                                    uint32_t time, uint32_t key, uint32_t state);
    static void keyboard_handle_modifiers(void* data, struct wl_keyboard* keyboard, uint32_t serial,
                                          uint32_t mods_depressed, uint32_t mods_latched,
                                          uint32_t mods_locked, uint32_t group);
    static void ime_handle_key(uint32_t key, bool down);
    static void toggle_fullscreen(int fullscreen);

    // xdg_wm
    static void handle_xdg_wm_ping(void* data, struct xdg_wm_base* xdg_wm_base, uint32_t serial);
};
