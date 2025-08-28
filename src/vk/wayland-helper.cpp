#include "wayland-helper.h"
#include "toolkit/logging.h"

namespace mywl
{

wl_registry_listener Context::registry_listener = {
    .global = &handleRegistry,
};

xdg_wm_base_listener Context::shell_listener = {
    .ping = handleShellPing,
};

xdg_surface_listener Context::shell_surface_listener = {
    .configure = handleShellSurfaceConfigure,
};

xdg_toplevel_listener Context::toplevel_listener = {
    .configure = handleToplevelConfigure,
    .close = handleToplevelClose,
};

wl_seat_listener Context::seat_listener = {
    .capabilities = handleSeatCapabilities,
};

wl_pointer_listener Context::pointer_listener = {
    .enter = handlePointerEnter,
    .leave = handlePointerLeave,
    .motion = handlePointerMotion,
    .button = handlePointerButton,
    .axis = handlePointerAxis,
};

wl_keyboard_listener Context::keyboard_listener = {
    .keymap = handleKeyboardKeymap,
    .enter = handleKeyboardEnter,
    .leave = handleKeyboardLeave,
    .key = handleKeyboardKey,
    .modifiers = handleKeyboardModifiers,
};

void Context::handleRegistry(void* data, wl_registry* registry, uint32_t name,
                             char const* interface, uint32_t version)
{
    Context* ctx = static_cast<Context*>(data);
    if (strcmp(interface, wl_compositor_interface.name) == 0) {
        CHECK_WL(ctx->compositor_ =
                     (wl_compositor*)wl_registry_bind(registry, name, &wl_compositor_interface, 1));
    } else if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
        CHECK_WL(ctx->shell_ =
                     (xdg_wm_base*)wl_registry_bind(registry, name, &xdg_wm_base_interface, 1));
        xdg_wm_base_add_listener(ctx->shell_, &shell_listener, data);
    } else if (strcmp(interface, wl_seat_interface.name) == 0) {
        CHECK_WL(ctx->seat_ = (wl_seat*)wl_registry_bind(registry, name, &wl_seat_interface, 1));
        wl_seat_add_listener(ctx->seat_, &seat_listener, data);
    } else if (strcmp(interface, wl_shm_interface.name) == 0) {
        CHECK_WL(ctx->shm_ = (wl_shm*)wl_registry_bind(registry, name, &wl_shm_interface, 1));
    }
}

void Context::handleShellPing(void* data, xdg_wm_base* shell, uint32_t serial)
{
    xdg_wm_base_pong(shell, serial);
}

void Context::handleShellSurfaceConfigure(void* data, xdg_surface* shell_surface, uint32_t serial)
{
    Context* ctx = static_cast<Context*>(data);
    Uid surface_id = ctx->getSurfaceId(shell_surface);
    Surface& surface = ctx->getSurface(surface_id);
    xdg_surface_ack_configure(shell_surface, serial);
    if (surface.need_resize_) {
        surface.need_resize_ = false;
        ctx->onSurfaceResize(surface_id, surface.width_, surface.height_);
    }
}

void Context::handleToplevelConfigure(void* data, xdg_toplevel* toplevel, int32_t width,
                                      int32_t height, wl_array* states)
{
    if (width != 0 && height != 0) {
        Context* ctx = static_cast<Context*>(data);
        Uid surface_id = ctx->getSurfaceId(toplevel);
        Surface& surface = ctx->getSurface(surface_id);
        surface.need_resize_ = true;
        surface.width_ = width;
        surface.height_ = height;
    }
}

void Context::handleToplevelClose(void* data, xdg_toplevel* toplevel)
{
    Context* ctx = static_cast<Context*>(data);
    Uid surface_id = ctx->getSurfaceId(toplevel);
    Surface& surface = ctx->getSurface(surface_id);
    surface.need_close_ = true;
}

void Context::handleSeatCapabilities(void* data, wl_seat* seat, uint32_t caps)
{
    Context* ctx = static_cast<Context*>(data);
    if ((caps & WL_SEAT_CAPABILITY_POINTER) && !ctx->pointer_) {
        ctx->pointer_ = wl_seat_get_pointer(seat);
        wl_pointer_add_listener(ctx->pointer_, &pointer_listener, data);
    } else if (!(caps & WL_SEAT_CAPABILITY_POINTER) && ctx->pointer_) {
        wl_pointer_destroy(ctx->pointer_);
        ctx->pointer_ = nullptr;
    }

    if ((caps & WL_SEAT_CAPABILITY_KEYBOARD) && !ctx->keyboard_) {
        ctx->keyboard_ = wl_seat_get_keyboard(seat);
        wl_keyboard_add_listener(ctx->keyboard_, &keyboard_listener, data);
    } else if (!(caps & WL_SEAT_CAPABILITY_KEYBOARD) && ctx->keyboard_) {
        wl_keyboard_destroy(ctx->keyboard_);
        ctx->keyboard_ = nullptr;
    }
}

void Context::handlePointerEnter(void* data, wl_pointer* pointer, uint32_t serial,
                                 wl_surface* surface, wl_fixed_t sx, wl_fixed_t sy)
{
}

void Context::handlePointerLeave(void* data, wl_pointer* pointer, uint32_t serial,
                                 wl_surface* surface)
{
}

void Context::handlePointerMotion(void* data, wl_pointer* pointer, uint32_t time, wl_fixed_t sx,
                                  wl_fixed_t sy)
{
}

void Context::handlePointerButton(void* data, wl_pointer* pointer, uint32_t serial, uint32_t time,
                                  uint32_t button, uint32_t state)
{
}

void Context::handlePointerAxis(void* data, wl_pointer* pointer, uint32_t time, uint32_t axis,
                                wl_fixed_t value)
{
}

void Context::handleKeyboardKeymap(void* data, wl_keyboard* keyboard, uint32_t format, int fd,
                                   uint32_t size)
{
}

void Context::handleKeyboardEnter(void* data, wl_keyboard* keyboard, uint32_t serial,
                                  wl_surface* surface, wl_array* keys)
{
}

void Context::handleKeyboardLeave(void* data, wl_keyboard* keyboard, uint32_t serial,
                                  wl_surface* surface)
{
}

void Context::handleKeyboardKey(void* data, wl_keyboard* keyboard, uint32_t serial, uint32_t time,
                                uint32_t key, uint32_t state)
{
}

void Context::handleKeyboardModifiers(void* data, wl_keyboard* keyboard, uint32_t serial,
                                      uint32_t mods_depressed, uint32_t mods_latched,
                                      uint32_t mods_locked, uint32_t group)
{
}

};  // namespace mywl
