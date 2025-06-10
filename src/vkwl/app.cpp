#include "app.h"
#include "toolkit/logging.h"

#define GET_VK_EXTENSION_FUNCTION(_id) ((PFN_##_id)(vkGetInstanceProcAddr(instance, #_id)))

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

VkInstance Application::instance = VK_NULL_HANDLE;
VkDebugUtilsMessengerEXT Application::debug_messenger = VK_NULL_HANDLE;

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
    CHECK_ERR_RET(initVulkan(app_id));
    CHECK_ERR_RET(mainLoop());
    CHECK_ERR_RET(cleanupVulkan());
    CHECK_ERR_RET(cleanupWayland());
    return MyErrCode::kOk;
}

MyErrCode Application::mainLoop() { return MyErrCode::kOk; }

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

MyErrCode Application::initVulkan(char const* app_name)
{
    CHECK_ERR_RET(createInstance(app_name));
    CHECK_ERR_RET(setupDebugMessenger());
    return MyErrCode::kOk;
}

MyErrCode Application::cleanupVulkan()
{
    GET_VK_EXTENSION_FUNCTION(vkDestroyDebugUtilsMessengerEXT)(instance, debug_messenger, nullptr);
    vkDestroyInstance(instance, nullptr);
    return MyErrCode::kOk;
}

MyErrCode Application::createInstance(char const* app_name)
{
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = app_name;
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = sizeof(kInstanceExtensionNames) / sizeof(char const*);
    create_info.ppEnabledExtensionNames = kInstanceExtensionNames;

    uint32_t layer_count;
    CHECK_VK_ERR_RET(vkEnumerateInstanceLayerProperties(&layer_count, nullptr));
    std::vector<VkLayerProperties> available_layers(layer_count);
    CHECK_VK_ERR_RET(vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data()));
    size_t found_layers = 0;
    for (uint32_t i = 0; i < layer_count; i++) {
        for (size_t j = 0; j < sizeof(kValidationLayers) / sizeof(char const*); j++) {
            if (strcmp(available_layers[i].layerName, kValidationLayers[j]) == 0) {
                found_layers++;
            }
        }
    }
    if (found_layers >= sizeof(kValidationLayers) / sizeof(char const*)) {
        create_info.enabledLayerCount = sizeof(kValidationLayers) / sizeof(char const*);
        create_info.ppEnabledLayerNames = kValidationLayers;
    }

    CHECK_VK_ERR_RET(vkCreateInstance(&create_info, nullptr, &instance));
    return MyErrCode::kOk;
}

MyErrCode Application::setupDebugMessenger()
{
    VkDebugUtilsMessengerCreateInfoEXT create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = debugCallback;
    create_info.pUserData = nullptr;
    CHECK_VK_ERR_RET(GET_VK_EXTENSION_FUNCTION(vkCreateDebugUtilsMessengerEXT)(
        instance, &create_info, nullptr, &debug_messenger));
    return MyErrCode::kOk;
}

VkBool32 Application::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                    VkDebugUtilsMessageTypeFlagsEXT type,
                                    VkDebugUtilsMessengerCallbackDataEXT const* callback_data,
                                    void* user_data)
{
    switch (severity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            DLOG("{}", callback_data->pMessage);
            break;
        default:
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            ILOG("{}", callback_data->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            WLOG("{}", callback_data->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            ELOG("{}", callback_data->pMessage);
            break;
    }
    return 0;
}