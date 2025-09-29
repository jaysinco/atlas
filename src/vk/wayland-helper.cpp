#include "wayland-helper.h"
#include "toolkit/logging.h"
#include <linux/input.h>

namespace mywl
{

Surface::Surface() = default;

Surface::Surface(wl_surface* surface, xdg_surface* shell_surface, xdg_toplevel* toplevel)
    : surface_(surface), shell_surface_(shell_surface), toplevel_(toplevel)
{
}

Surface::operator wl_surface*() const { return surface_; }

Surface::operator bool() const { return surface_ != nullptr; }

MyErrCode Context::createDisplay(EventHandler* event_handler, char const* name)
{
    CHECK_WL_RET(display_ = wl_display_connect(name));
    CHECK_WL_RET(registry_ = wl_display_get_registry(display_));
    wl_registry_add_listener(registry_, &registry_listener, this);
    wl_display_roundtrip(display_);
    CHECK_WL_RET(cursor_surface_ = wl_compositor_create_surface(compositor_));
    event_handler_ = event_handler;
    return MyErrCode::kOk;
}

wl_display* Context::getDisplay() { return display_; }

MyErrCode Context::destroy()
{
    while (!surfaces_.empty()) {
        CHECK_ERR_RET(destroySurface(surfaces_.begin()->first));
    }
    if (cursor_surface_) {
        wl_surface_destroy(cursor_surface_);
    }
    if (cursor_theme_) {
        wl_cursor_theme_destroy(cursor_theme_);
    }
    if (keyboard_) {
        wl_keyboard_destroy(keyboard_);
    }
    if (pointer_) {
        wl_pointer_destroy(pointer_);
    }
    if (shm_) {
        wl_shm_destroy(shm_);
    }
    if (seat_) {
        wl_seat_destroy(seat_);
    }
    if (shell_) {
        xdg_wm_base_destroy(shell_);
    }
    if (compositor_) {
        wl_compositor_destroy(compositor_);
    }
    if (registry_) {
        wl_registry_destroy(registry_);
    }
    if (display_) {
        wl_display_disconnect(display_);
    }
    return MyErrCode::kOk;
}

MyErrCode Context::createSurface(Uid id, std::string const& app_id, std::string const& title)
{
    if (surfaces_.find(id) != surfaces_.end()) {
        CHECK_ERR_RET(destroySurface(id));
    }

    wl_surface* surface;
    xdg_surface* shell_surface;
    xdg_toplevel* toplevel;

    CHECK_WL_RET(surface = wl_compositor_create_surface(compositor_));
    CHECK_WL_RET(shell_surface = xdg_wm_base_get_xdg_surface(shell_, surface));
    CHECK_WL_RET(toplevel = xdg_surface_get_toplevel(shell_surface));
    surfaces_[id] = {surface, shell_surface, toplevel};

    xdg_surface_add_listener(shell_surface, &shell_surface_listener, this);
    xdg_toplevel_add_listener(toplevel, &toplevel_listener, this);

    xdg_toplevel_set_app_id(toplevel, app_id.c_str());
    xdg_toplevel_set_title(toplevel, title.c_str());

    wl_surface_commit(surface);
    wl_display_roundtrip(display_);
    wl_surface_commit(surface);

    return MyErrCode::kOk;
}

Surface& Context::getSurface(Uid id)
{
    if (surfaces_.find(id) == surfaces_.end()) {
        MY_THROW("surface not exist: {}", id);
    }
    return surfaces_.at(id);
}

wl_surface* Context::getRawSurface(Uid id) { return getSurface(id).surface_; }

MyErrCode Context::destroySurface(Uid id)
{
    if (auto it = surfaces_.find(id); it != surfaces_.end()) {
        xdg_toplevel_destroy(it->second.toplevel_);
        xdg_surface_destroy(it->second.shell_surface_);
        wl_surface_destroy(it->second.surface_);
        surfaces_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("surface not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::dispatch()
{
    wl_display_roundtrip(display_);
    return MyErrCode::kOk;
}

Uid Context::getSurfaceId(wl_surface* surface)
{
    auto it = std::find_if(surfaces_.begin(), surfaces_.end(),
                           [&](auto& s) { return s.second.surface_ == surface; });
    return it == surfaces_.end() ? Uid::kNull : it->first;
}

Uid Context::getSurfaceId(xdg_surface* shell_surface)
{
    auto it = std::find_if(surfaces_.begin(), surfaces_.end(),
                           [&](auto& s) { return s.second.shell_surface_ == shell_surface; });
    return it == surfaces_.end() ? Uid::kNull : it->first;
}

Uid Context::getSurfaceId(xdg_toplevel* toplevel)
{
    auto it = std::find_if(surfaces_.begin(), surfaces_.end(),
                           [&](auto& s) { return s.second.toplevel_ == toplevel; });
    return it == surfaces_.end() ? Uid::kNull : it->first;
}

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
    .wm_capabilities = handleToplevelCapabilities,
};

wl_seat_listener Context::seat_listener = {
    .capabilities = handleSeatCapabilities,
    .name = handleSeatName,
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
    .repeat_info = handleKeyboardRepeat,
};

void Context::handleRegistry(void* data, wl_registry* registry, uint32_t name,
                             char const* interface, uint32_t version)
{
    Context* ctx = static_cast<Context*>(data);
    if (strcmp(interface, wl_compositor_interface.name) == 0) {
        CHECK_WL(ctx->compositor_ = (wl_compositor*)wl_registry_bind(
                     registry, name, &wl_compositor_interface, version));
    } else if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
        CHECK_WL(ctx->shell_ = (xdg_wm_base*)wl_registry_bind(registry, name,
                                                              &xdg_wm_base_interface, version));
        xdg_wm_base_add_listener(ctx->shell_, &shell_listener, data);
    } else if (strcmp(interface, wl_seat_interface.name) == 0) {
        CHECK_WL(ctx->seat_ =
                     (wl_seat*)wl_registry_bind(registry, name, &wl_seat_interface, version));
        wl_seat_add_listener(ctx->seat_, &seat_listener, data);
    } else if (strcmp(interface, wl_shm_interface.name) == 0) {
        CHECK_WL(ctx->shm_ = (wl_shm*)wl_registry_bind(registry, name, &wl_shm_interface, version));
        CHECK_WL(ctx->cursor_theme_ = wl_cursor_theme_load(nullptr, 24, ctx->shm_));
        CHECK_WL(ctx->cursor_ = wl_cursor_theme_get_cursor(ctx->cursor_theme_, "left_ptr"));
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
        ctx->event_handler_->onEvent(
            surface_id,
            {.type = EventType::kSurfaceResize, .ix = surface.width_, .iy = surface.height_});
    }
}

void Context::handleToplevelConfigure(void* data, xdg_toplevel* toplevel, int32_t width,
                                      int32_t height, wl_array* states)
{
    Context* ctx = static_cast<Context*>(data);
    if (width != 0 && height != 0) {
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
    ctx->event_handler_->onEvent(surface_id, {.type = EventType::kSurfaceClose});
}

void Context::handleToplevelCapabilities(void* data, xdg_toplevel* xdg_toplevel,
                                         wl_array* capabilities)
{
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

void Context::handleSeatName(void* data, wl_seat* wl_seat, char const* name) {}

void Context::handlePointerEnter(void* data, wl_pointer* pointer, uint32_t serial,
                                 wl_surface* surface, wl_fixed_t sx, wl_fixed_t sy)
{
    Context* ctx = static_cast<Context*>(data);
    Uid surface_id = ctx->getSurfaceId(surface);
    ctx->pointer_surface_id_ = surface_id;
    if (!ctx->cursor_) {
        return;
    }
    wl_cursor_image* cursor_image = ctx->cursor_->images[0];
    if (!cursor_image) {
        return;
    }
    wl_buffer* cursor_buffer = wl_cursor_image_get_buffer(cursor_image);
    if (!cursor_buffer) {
        return;
    }
    wl_pointer_set_cursor(pointer, serial, ctx->cursor_surface_, cursor_image->hotspot_x,
                          cursor_image->hotspot_y);
    wl_surface_attach(ctx->cursor_surface_, cursor_buffer, 0, 0);
    wl_surface_damage(ctx->cursor_surface_, 0, 0, cursor_image->width, cursor_image->height);
    wl_surface_commit(ctx->cursor_surface_);
}

void Context::handlePointerLeave(void* data, wl_pointer* pointer, uint32_t serial,
                                 wl_surface* surface)
{
}

void Context::handlePointerMotion(void* data, wl_pointer* pointer, uint32_t time, wl_fixed_t sx,
                                  wl_fixed_t sy)
{
    Context* ctx = static_cast<Context*>(data);
    double x = wl_fixed_to_double(sx);
    double y = wl_fixed_to_double(sy);
    ctx->event_handler_->onEvent(ctx->pointer_surface_id_,
                                 {.type = EventType::kPointerMove, .dx = x, .dy = y});
}

void Context::handlePointerButton(void* data, wl_pointer* pointer, uint32_t serial, uint32_t time,
                                  uint32_t button, uint32_t state)
{
    Context* ctx = static_cast<Context*>(data);
    int btn = 0;
    if (button == BTN_LEFT) {
        btn = 0;
    } else if (button == BTN_RIGHT) {
        btn = 1;
    } else if (button == BTN_MIDDLE) {
        btn = 2;
    }
    bool down = state == WL_POINTER_BUTTON_STATE_PRESSED;
    ctx->event_handler_->onEvent(ctx->pointer_surface_id_,
                                 {.type = EventType::kPointerPress, .ix = btn, .iy = down});
}

void Context::handlePointerAxis(void* data, wl_pointer* pointer, uint32_t time, uint32_t axis,
                                wl_fixed_t value)
{
    Context* ctx = static_cast<Context*>(data);
    if (axis == 0) {
        double yoffset = wl_fixed_to_double(value);
        ctx->event_handler_->onEvent(ctx->pointer_surface_id_,
                                     {.type = EventType::kPointerScroll, .dx = 0.0, .dy = yoffset});
    }
}

void Context::handleKeyboardKeymap(void* data, wl_keyboard* keyboard, uint32_t format, int fd,
                                   uint32_t size)
{
}

void Context::handleKeyboardEnter(void* data, wl_keyboard* keyboard, uint32_t serial,
                                  wl_surface* surface, wl_array* keys)
{
    Context* ctx = static_cast<Context*>(data);
    Uid surface_id = ctx->getSurfaceId(surface);
    ctx->keyboard_surface_id_ = surface_id;
}

void Context::handleKeyboardLeave(void* data, wl_keyboard* keyboard, uint32_t serial,
                                  wl_surface* surface)
{
}

void Context::handleKeyboardKey(void* data, wl_keyboard* keyboard, uint32_t serial, uint32_t time,
                                uint32_t key, uint32_t state)
{
    Context* ctx = static_cast<Context*>(data);
    bool down = state == 1;
    ctx->event_handler_->onEvent(ctx->keyboard_surface_id_,
                                 {.type = EventType::kKeyboardPress, .ux = key, .ix = down});
}

void Context::handleKeyboardModifiers(void* data, wl_keyboard* keyboard, uint32_t serial,
                                      uint32_t mods_depressed, uint32_t mods_latched,
                                      uint32_t mods_locked, uint32_t group)
{
}

void Context::handleKeyboardRepeat(void* data, wl_keyboard* wl_keyboard, int32_t rate,
                                   int32_t delay)
{
}
};  // namespace mywl
