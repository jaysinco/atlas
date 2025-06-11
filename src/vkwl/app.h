#pragma once
#include "toolkit/error.h"
#include <wayland-client.h>
#include "xdg-shell.h"
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_wayland.h>

class Application
{
public:
    static MyErrCode run(char const* win_title, int win_width, int win_height, char const* app_id);

private:
    static MyErrCode mainLoop();

private:
    static MyErrCode initWayland(char const* win_title, int win_width, int win_height,
                                 char const* app_id);
    static MyErrCode cleanupWayland();
    static void handleRegistry(void* data, struct wl_registry* registry, uint32_t name,
                               char const* interface, uint32_t version);
    static void handleShellPing(void* data, struct xdg_wm_base* shell, uint32_t serial);
    static void handleShellSurfaceConfigure(void* data, struct xdg_surface* shell_surface,
                                            uint32_t serial);
    static void handleToplevelConfigure(void* data, struct xdg_toplevel* toplevel, int32_t width,
                                        int32_t height, struct wl_array* states);
    static void handleToplevelClose(void* data, struct xdg_toplevel* toplevel);

private:
    static MyErrCode initVulkan(char const* app_name);
    static MyErrCode cleanupVulkan();
    static MyErrCode createInstance(char const* app_name);
    static MyErrCode setupDebugMessenger();
    static VkBool32 debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                  VkDebugUtilsMessageTypeFlagsEXT type,
                                  VkDebugUtilsMessengerCallbackDataEXT const* callback_data,
                                  void* user_data);
    static MyErrCode createVkSurface();
    static MyErrCode pickPhysicalDevice();
    static MyErrCode rateDeviceSuitability(VkPhysicalDevice device, int& score);
    static MyErrCode createLogicalDevice();

private:
    static wl_display* display;
    static wl_registry* registry;
    static wl_compositor* compositor;
    static wl_surface* surface;
    static xdg_wm_base* shell;
    static xdg_surface* shell_surface;
    static xdg_toplevel* toplevel;

    static wl_registry_listener registry_listener;
    static xdg_wm_base_listener shell_listener;
    static xdg_surface_listener shell_surface_listener;
    static xdg_toplevel_listener toplevel_listener;

    static VkInstance instance;
    constexpr static char const* const kInstanceExtensions[] = {
        "VK_EXT_debug_utils", "VK_KHR_surface", "VK_KHR_wayland_surface"};
    constexpr static char const* const kInstanceLayers[] = {"VK_LAYER_KHRONOS_validation"};
    static VkDebugUtilsMessengerEXT debug_messenger;
    static VkSurfaceKHR vulkan_surface;
    static VkPhysicalDevice physical_device;
    static VkDevice device;
    constexpr static char const* const kDeviceExtensions[] = {"VK_KHR_swapchain"};
    static uint32_t graphics_queue_family_index;
    static VkQueue graphics_queue;

    static bool quit;
    static bool ready_to_resize;
    static bool resize;
    static int new_width;
    static int new_height;
    static uint32_t width;
    static uint32_t height;
};
