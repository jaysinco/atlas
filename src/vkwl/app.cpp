#include "app.h"
#include "toolkit/logging.h"

#define CHECK_VK_ERR_RET(err)                    \
    if (auto _err = (err); _err != VK_SUCCESS) { \
        ELOG("failed to call vulkan: {}", _err); \
        return MyErrCode::kFailed;               \
    }

#define CHECK_WL_ERR(expr)              \
    if (!(expr)) {                      \
        ELOG("failed to call wayland"); \
    }

#define CHECK_WL_ERR_RET(expr)          \
    if (!(expr)) {                      \
        ELOG("failed to call wayland"); \
        return MyErrCode::kFailed;      \
    }

wl_display* Application::display = nullptr;
wl_registry* Application::registry = nullptr;
wl_compositor* Application::compositor = nullptr;
wl_surface* Application::surface = nullptr;
xdg_wm_base* Application::shell = nullptr;
xdg_surface* Application::shell_surface = nullptr;
xdg_toplevel* Application::toplevel = nullptr;

wl_registry_listener Application::registry_listener = {.global = &handleRegistry};
xdg_wm_base_listener Application::shell_listener = {.ping = handleShellPing};
xdg_surface_listener Application::shell_surface_listener = {.configure =
                                                                handleShellSurfaceConfigure};
xdg_toplevel_listener Application::toplevel_listener = {.configure = handleToplevelConfigure,
                                                        .close = handleToplevelClose};

bool Application::quit = false;
bool Application::ready_to_resize = false;
bool Application::resize = false;
int Application::new_width = 0;
int Application::new_height = 0;
uint32_t Application::width = 0;
uint32_t Application::height = 0;

MyErrCode Application::run(char const* win_title, int win_width, int win_height, char const* app_id)
{
    CHECK_ERR_RET(initWayland(win_title, win_width, win_height, app_id));
    CHECK_ERR_RET(initVulkan());
    CHECK_ERR_RET(mainLoop());
    CHECK_ERR_RET(cleanupVulkan());
    CHECK_ERR_RET(cleanupWayland());
    return MyErrCode::kOk;
}

MyErrCode Application::initWayland(char const* win_title, int win_width, int win_height,
                                   char const* app_id)
{
    CHECK_WL_ERR_RET(display = wl_display_connect(nullptr));

    CHECK_WL_ERR_RET(registry = wl_display_get_registry(display));
    wl_registry_add_listener(registry, &registry_listener, nullptr);
    wl_display_roundtrip(display);

    CHECK_WL_ERR_RET(surface = wl_compositor_create_surface(compositor));

    CHECK_WL_ERR_RET(shell_surface = xdg_wm_base_get_xdg_surface(shell, surface));
    xdg_surface_add_listener(shell_surface, &shell_surface_listener, nullptr);

    CHECK_WL_ERR_RET(toplevel = xdg_surface_get_toplevel(shell_surface));
    xdg_toplevel_add_listener(toplevel, &toplevel_listener, nullptr);

    width = win_width;
    height = win_height;
    xdg_toplevel_set_title(toplevel, win_title);
    xdg_toplevel_set_app_id(toplevel, app_id);

    wl_surface_commit(surface);
    wl_display_roundtrip(display);
    wl_surface_commit(surface);

    return MyErrCode::kOk;
}

MyErrCode Application::initVulkan()
{
    uint32_t extension_count = 0;
    CHECK_VK_ERR_RET(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr));
    ILOG("{} extensions supported", extension_count);

    return MyErrCode::kOk;
}

MyErrCode Application::mainLoop() { return MyErrCode::kOk; }

MyErrCode Application::cleanupVulkan() { return MyErrCode::kOk; }

MyErrCode Application::cleanupWayland()
{
    xdg_toplevel_destroy(toplevel);
    xdg_surface_destroy(shell_surface);
    wl_surface_destroy(surface);
    xdg_wm_base_destroy(shell);
    wl_compositor_destroy(compositor);
    wl_registry_destroy(registry);
    wl_display_disconnect(display);
    return MyErrCode::kOk;
}

void Application::handleRegistry(void* data, struct wl_registry* registry, uint32_t name,
                                 char const* interface, uint32_t version)
{
    if (strcmp(interface, wl_compositor_interface.name) == 0) {
        CHECK_WL_ERR(compositor = (wl_compositor*)wl_registry_bind(registry, name,
                                                                   &wl_compositor_interface, 1));
    } else if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
        CHECK_WL_ERR(shell =
                         (xdg_wm_base*)wl_registry_bind(registry, name, &xdg_wm_base_interface, 1));
        xdg_wm_base_add_listener(shell, &shell_listener, nullptr);
    }
}

void Application::handleShellPing(void* data, struct xdg_wm_base* shell, uint32_t serial)
{
    xdg_wm_base_pong(shell, serial);
}

void Application::handleShellSurfaceConfigure(void* data, struct xdg_surface* shell_surface,
                                              uint32_t serial)
{
    xdg_surface_ack_configure(shell_surface, serial);
    if (resize) {
        ready_to_resize = true;
    }
}

void Application::handleToplevelConfigure(void* data, struct xdg_toplevel* toplevel, int32_t width,
                                          int32_t height, struct wl_array* states)
{
    if (width != 0 && height != 0) {
        resize = true;
        new_width = width;
        new_height = height;
    }
}

void Application::handleToplevelClose(void* data, struct xdg_toplevel* toplevel) { quit = true; }
