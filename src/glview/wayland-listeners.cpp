#include "wayland-listeners.h"
#include "display-context.h"
#include <cstring>
#include <linux/input.h>
#include <X11/keysym.h>
#include "keycode-converter.h"
#include "top-window.h"
#include "ime-editor.h"
#include "imgui/imgui.h"
#include "toolkit/logging.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include <thread>

xdg_surface_listener WaylandListeners::xdg_surface = {handle_xdg_surface_configure};

xdg_toplevel_listener WaylandListeners::xdg_toplevel = {handle_xdg_toplevel_configure,
                                                        handle_xdg_toplevel_close};

wl_seat_listener WaylandListeners::seat = {
    .capabilities = seat_handle_capabilities,
};

wl_registry_listener WaylandListeners::registry = {registry_handle_global,
                                                   registry_handle_global_remove};

wl_callback_listener WaylandListeners::configure_cb = {configure_callback};

wl_callback_listener WaylandListeners::frame_cb = {redraw};

wl_pointer_listener WaylandListeners::pointer = {pointer_handle_enter, pointer_handle_leave,
                                                 pointer_handle_motion, pointer_handle_button,
                                                 pointer_handle_axis};

wl_keyboard_listener WaylandListeners::keyboard = {keyboard_handle_keymap, keyboard_handle_enter,
                                                   keyboard_handle_leave, keyboard_handle_key,
                                                   keyboard_handle_modifiers};

xdg_wm_base_listener WaylandListeners::xdgwm = {handle_xdg_wm_ping};

void WaylandListeners::handle_xdg_surface_configure(void* data, struct xdg_surface* xdg_surface,
                                                    uint32_t serial)
{
    auto& ctx = DisplayContext::Instance();
    if (!ctx.wl.fullscreen) {
        xdg_surface_set_window_geometry(xdg_surface, 0, 0, ctx.window_size.width,
                                        ctx.window_size.height);
    }
    xdg_surface_ack_configure(xdg_surface, serial);
}

void WaylandListeners::handle_xdg_toplevel_configure(void* data, struct xdg_toplevel* xdg_toplevel,
                                                     int32_t width, int32_t height,
                                                     struct wl_array* states)
{
    auto& ctx = DisplayContext::Instance();

    if (width == 0 || height == 0) {
        width = ctx.window_size.width;
        height = ctx.window_size.height;
    }

    if (ctx.wl.native) {
        wl_egl_window_resize(ctx.wl.native, width, height, 0, 0);
    }

    ctx.geometry.width = width;
    ctx.geometry.height = height;

    if (!ctx.wl.fullscreen) {
        ctx.window_size = ctx.geometry;
    }
}

void WaylandListeners::handle_xdg_toplevel_close(void* data, struct xdg_toplevel* xdg_toplevel) {}

void WaylandListeners::seat_handle_capabilities(void* data, struct wl_seat* seat, uint32_t caps)
{
    auto& ctx = DisplayContext::Instance();

    if ((caps & WL_SEAT_CAPABILITY_POINTER) && !ctx.wl.pointer) {
        ctx.wl.pointer = wl_seat_get_pointer(seat);
        wl_pointer_add_listener(ctx.wl.pointer, &pointer, nullptr);
    } else if (!(caps & WL_SEAT_CAPABILITY_POINTER) && ctx.wl.pointer) {
        wl_pointer_destroy(ctx.wl.pointer);
        ctx.wl.pointer = NULL;
    }

    if ((caps & WL_SEAT_CAPABILITY_KEYBOARD) && !ctx.wl.keyboard) {
        ctx.wl.keyboard = wl_seat_get_keyboard(seat);
        wl_keyboard_add_listener(ctx.wl.keyboard, &keyboard, nullptr);
    } else if (!(caps & WL_SEAT_CAPABILITY_KEYBOARD) && ctx.wl.keyboard) {
        wl_keyboard_destroy(ctx.wl.keyboard);
        ctx.wl.keyboard = NULL;
    }
}

void WaylandListeners::registry_handle_global(void* data, struct wl_registry* registry,
                                              uint32_t name, char const* interface,
                                              uint32_t version)
{
    auto& ctx = DisplayContext::Instance();

    if (strcmp(interface, "wl_compositor") == 0) {
        ctx.wl.compositor =
            (wl_compositor*)wl_registry_bind(registry, name, &wl_compositor_interface, 1);
    } else if (strcmp(interface, "wl_shell") == 0) {
        ctx.wl.shell = (wl_shell*)wl_registry_bind(registry, name, &wl_shell_interface, 1);
    } else if (strcmp(interface, "wl_seat") == 0) {
        ctx.wl.seat = (wl_seat*)wl_registry_bind(registry, name, &wl_seat_interface, 1);
        wl_seat_add_listener(ctx.wl.seat, &seat, nullptr);
    } else if (strcmp(interface, "wl_shm") == 0) {
        ctx.wl.shm = (wl_shm*)wl_registry_bind(registry, name, &wl_shm_interface, 1);
        ctx.wl.cursor_theme = wl_cursor_theme_load(NULL, 32, ctx.wl.shm);
        ctx.wl.default_cursor = wl_cursor_theme_get_cursor(ctx.wl.cursor_theme, "left_ptr");
    } else if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
        ctx.wl.xdg_wm = (xdg_wm_base*)wl_registry_bind(registry, name, &xdg_wm_base_interface,
                                                       std::min(3u, version));
        xdg_wm_base_add_listener(ctx.wl.xdg_wm, &xdgwm, nullptr);
    }
}

void WaylandListeners::registry_handle_global_remove(void* data, struct wl_registry* registry,
                                                     uint32_t name)
{
}

void WaylandListeners::configure_callback(void* data, struct wl_callback* callback, uint32_t time)
{
    auto& ctx = DisplayContext::Instance();

    wl_callback_destroy(callback);

    ctx.wl.configured = 1;

    if (ctx.wl.callback == NULL) {
        redraw(data, NULL, time);
    }
}

void WaylandListeners::redraw(void* data, wl_callback* callback, uint32_t time)
{
    auto& ctx = DisplayContext::Instance();
    wl_region* region;

    assert(ctx.wl.callback == callback);
    ctx.wl.callback = NULL;

    if (callback) wl_callback_destroy(callback);

    if (!ctx.wl.configured) return;

    // draw frame
    FrameWindow::Draw();

    // ==== imgui begin ====
    ImGui_ImplOpenGL3_NewFrame();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)ctx.geometry.width, (float)ctx.geometry.height);
    ImGui::NewFrame();

    ImeEditor::Draw();

    ImGui::Render();
    ImDrawData* raw_imgui_data = ImGui::GetDrawData();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    // ==== imgui end ====

    if (ctx.wl.opaque || ctx.wl.fullscreen) {
        region = wl_compositor_create_region(ctx.wl.compositor);
        wl_region_add(region, 0, 0, ctx.geometry.width, ctx.geometry.height);
        wl_surface_set_opaque_region(ctx.wl.surface, region);
        wl_region_destroy(region);
    } else {
        wl_surface_set_opaque_region(ctx.wl.surface, NULL);
    }

    ctx.wl.callback = wl_surface_frame(ctx.wl.surface);
    wl_callback_add_listener(ctx.wl.callback, &frame_cb, NULL);

    eglSwapBuffers(ctx.egl.dpy, ctx.wl.egl_surface);
}

void WaylandListeners::pointer_handle_enter(void* data, struct wl_pointer* pointer, uint32_t serial,
                                            struct wl_surface* surface, wl_fixed_t sx,
                                            wl_fixed_t sy)
{
    auto& ctx = DisplayContext::Instance();
    wl_buffer* buffer;
    wl_cursor* cursor = ctx.wl.default_cursor;
    wl_cursor_image* image;

    // if (display->window->fullscreen)
    //     wl_pointer_set_cursor(pointer, serial, NULL, 0, 0);

    if (cursor) {
        image = ctx.wl.default_cursor->images[0];
        buffer = wl_cursor_image_get_buffer(image);
        wl_pointer_set_cursor(pointer, serial, ctx.wl.cursor_surface, image->hotspot_x,
                              image->hotspot_y);
        wl_surface_attach(ctx.wl.cursor_surface, buffer, 0, 0);
        wl_surface_damage(ctx.wl.cursor_surface, 0, 0, image->width, image->height);
        wl_surface_commit(ctx.wl.cursor_surface);
    }
}

void WaylandListeners::pointer_handle_leave(void* data, struct wl_pointer* pointer, uint32_t serial,
                                            struct wl_surface* surface)
{
}

void WaylandListeners::pointer_handle_motion(void* data, struct wl_pointer* pointer, uint32_t time,
                                             wl_fixed_t sx, wl_fixed_t sy)
{
    auto& ctx = DisplayContext::Instance();
    ImGuiIO& io = ImGui::GetIO();
    ctx.mouse_pos.x = wl_fixed_to_double(sx);
    ctx.mouse_pos.y = wl_fixed_to_double(sy);
    io.AddMousePosEvent(wl_fixed_to_double(sx), wl_fixed_to_double(sy));
}

void WaylandListeners::pointer_handle_button(void* data, struct wl_pointer* wl_pointer,
                                             uint32_t serial, uint32_t time, uint32_t button,
                                             uint32_t state)
{
    // if (button == BTN_RIGHT && state == WL_POINTER_BUTTON_STATE_PRESSED)
    // {
    //     wl_shell_surface_move(display->window->shell_surface, display->seat, serial);
    // }

    ImGuiIO& io = ImGui::GetIO();
    int imbtn = 0;
    if (button == BTN_LEFT) {
        imbtn = 0;
    } else if (button == BTN_RIGHT) {
        imbtn = 1;
    } else if (button == BTN_MIDDLE) {
        imbtn = 2;
    }
    io.AddMouseButtonEvent(imbtn, state == WL_POINTER_BUTTON_STATE_PRESSED);
}

void WaylandListeners::pointer_handle_axis(void* data, struct wl_pointer* wl_pointer, uint32_t time,
                                           uint32_t axis, wl_fixed_t value)
{
    if (axis == 0) {
        auto& ctx = DisplayContext::Instance();
        ImGuiIO& io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            // float xpos = ctx.mouse_pos.x / ctx.geometry.width;
            // float ypos = ctx.mouse_pos.y / ctx.geometry.height;
            // Zoom(xpos, ypos, wl_fixed_to_double(value) / -10.0);
        } else {
            io.AddMouseWheelEvent(0.0, wl_fixed_to_double(value) / -10.0);
        }
    }
}

void WaylandListeners::keyboard_handle_keymap(void* data, struct wl_keyboard* keyboard,
                                              uint32_t format, int fd, uint32_t size)
{
}

void WaylandListeners::keyboard_handle_enter(void* data, struct wl_keyboard* keyboard,
                                             uint32_t serial, struct wl_surface* surface,
                                             struct wl_array* keys)
{
}

void WaylandListeners::keyboard_handle_leave(void* data, struct wl_keyboard* keyboard,
                                             uint32_t serial, struct wl_surface* surface)
{
}

void WaylandListeners::ime_handle_key(uint32_t key, bool down)
{
    ImGuiIO& io = ImGui::GetIO();
    auto& ctx = DisplayContext::Instance();

    if (key == KEY_LEFTCTRL) {
        ctx.ime.ctrl_down = down;
        return;
    }

    if (key == KEY_LEFTSHIFT) {
        ctx.ime.shift_down = down;
        return;
    }

    // toggle ime mode
    if (ctx.ime.ctrl_down && key == KEY_SPACE) {
        if (down) {
            ctx.ime.ascii_mode = !ctx.ime.ascii_mode;
            ILOG("ascii_mode={}", ctx.ime.ascii_mode);
        }
        return;
    }

    // ascii mode
    if (ctx.ime.ascii_mode) {
        if (key == KEY_BACKSPACE || key == KEY_LEFT || key == KEY_RIGHT) {
            io.AddKeyEvent(
                (ImGuiKey)KeyCodeConverter::ConvertTo(KeyCodeConverter::IMGUI, key, false), down);
            return;
        }

        if (!down) {
            return;
        }

        int c = KeyCodeConverter::ConvertTo(KeyCodeConverter::ASCII, key, ctx.ime.shift_down);
        if (c < 0) {
            return;
        }

        io.AddInputCharacter(c);
    }
    // chinese mode
    else {
        // not composing
        if (!ctx.ime.state.isComposing) {
            if (key == KEY_BACKSPACE || key == KEY_LEFT || key == KEY_RIGHT) {
                io.AddKeyEvent(
                    (ImGuiKey)KeyCodeConverter::ConvertTo(KeyCodeConverter::IMGUI, key, false),
                    down);
                return;
            } else if (key == KEY_SPACE) {
                if (down) {
                    io.AddInputCharacter(' ');
                }
                return;
            } else {
                // pass down to ime engine
            }
        }

        // is composing
        if (!down) {
            return;
        }

        int keysym = KeyCodeConverter::ConvertTo(KeyCodeConverter::KEYSYM, key, false);
        if (keysym == XK_VoidSymbol) {
            return;
        }

        // EsImeProcessKeyX(ctx.ime.session, keysym, 0);

        // EsStringHndl state_json;
        // EsImeGetStateX(ctx.ime.session, &state_json);
        // toolkit::SVConvertFromJsonStr(*(std::string*)state_json, ctx.ime.state);
        // EsStringFree(state_json);

        // if (!ctx.ime.state.isComposing) {
        //     EsStringHndl hs;
        //     EsImeGetCommitX(ctx.ime.session, &hs);
        //     std::string output = *(std::string*)hs;
        //     EsStringFree(hs);

        //     if (output.size() > 0) {
        //         io.AddInputCharactersUTF8(output.c_str());
        //     }
        // }
    }
}

void WaylandListeners::toggle_fullscreen(int fullscreen)
{
    auto& ctx = DisplayContext::Instance();
    struct wl_callback* callback;

    ctx.wl.fullscreen = fullscreen;
    ctx.wl.configured = 0;

    if (fullscreen) {
        xdg_toplevel_set_fullscreen(ctx.wl.xdg_top, NULL);
    } else {
        xdg_toplevel_unset_fullscreen(ctx.wl.xdg_top);
    }

    callback = wl_display_sync(ctx.wl.display);
    wl_callback_add_listener(callback, &configure_cb, nullptr);
}

void WaylandListeners::handle_xdg_wm_ping(void* data, struct xdg_wm_base* xdg_wm_base,
                                          uint32_t serial)
{
    xdg_wm_base_pong(xdg_wm_base, serial);
}

void WaylandListeners::keyboard_handle_key(void* data, struct wl_keyboard* keyboard,
                                           uint32_t serial, uint32_t time, uint32_t key,
                                           uint32_t state)
{
    auto& ctx = DisplayContext::Instance();
    bool down = (state == 1);

    // ime is actived
    if (ctx.ime.actived) {
        ime_handle_key(key, down);
        return;
    }

    // ime not actived
    if (key == KEY_F11 && down) {
        toggle_fullscreen(ctx.wl.fullscreen ^ 1);
    } else if (key == KEY_Q && down) {
        ctx.wl.running = 0;
    }
}

void WaylandListeners::keyboard_handle_modifiers(void* data, struct wl_keyboard* keyboard,
                                                 uint32_t serial, uint32_t mods_depressed,
                                                 uint32_t mods_latched, uint32_t mods_locked,
                                                 uint32_t group)
{
}
