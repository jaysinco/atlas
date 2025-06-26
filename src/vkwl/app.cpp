#include "app.h"
#include "toolkit/logging.h"

#define GET_VK_EXTENSION_FUNCTION(_id) ((PFN_##_id)(vkGetInstanceProcAddr(instance, #_id)))

#define CHECK_VK_ERR_RET(err)                        \
    do {                                             \
        if (auto _err = (err); _err != VK_SUCCESS) { \
            ELOG("failed to call vulkan: {}", _err); \
            return MyErrCode::kFailed;               \
        }                                            \
    } while (0)

#define CHECK_WL_ERR(expr) \
    if (!(expr)) ELOG("failed to call wayland")

#define CHECK_WL_ERR_RET(expr)              \
    do {                                    \
        if (!(expr)) {                      \
            ELOG("failed to call wayland"); \
            return MyErrCode::kFailed;      \
        }                                   \
    } while (0)

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
VkSurfaceKHR Application::vulkan_surface = VK_NULL_HANDLE;
VkPhysicalDevice Application::physical_device = VK_NULL_HANDLE;
VkDevice Application::device = VK_NULL_HANDLE;
uint32_t Application::graphics_queue_family_index = 0;
VkQueue Application::graphics_queue = VK_NULL_HANDLE;
VkCommandPool Application::command_pool = VK_NULL_HANDLE;
VkSwapchainKHR Application::swapchain = VK_NULL_HANDLE;
VkRenderPass Application::render_pass = VK_NULL_HANDLE;
VkFormat Application::swapchain_image_format = VK_FORMAT_UNDEFINED;
std::vector<VkImage> Application::swapchain_images;
std::vector<VkImageView> Application::swapchain_image_views;
std::vector<VkFramebuffer> Application::swapchain_frame_buffers;
std::vector<VkCommandBuffer> Application::command_buffers;
std::vector<VkSemaphore> Application::image_available_semaphores;
std::vector<VkSemaphore> Application::render_finished_semaphores;
std::vector<VkFence> Application::in_flight_fences;

bool Application::need_quit = false;
bool Application::ready_to_resize = false;
bool Application::need_resize = false;
int Application::new_width = 0;
int Application::new_height = 0;
uint32_t Application::curr_width = 0;
uint32_t Application::curr_height = 0;
int Application::curr_frame = 0;

MyErrCode Application::run(char const* win_title, int win_width, int win_height, char const* app_id)
{
    CHECK_ERR_RET(initWayland(win_title, win_width, win_height, app_id));
    CHECK_ERR_RET(initVulkan(app_id));
    CHECK_ERR_RET(mainLoop());
    CHECK_ERR_RET(cleanupVulkan());
    CHECK_ERR_RET(cleanupWayland());
    return MyErrCode::kOk;
}

MyErrCode Application::mainLoop()
{
    while (!need_quit) {
        if (need_resize && ready_to_resize) {
            curr_width = new_width;
            curr_height = new_height;
            CHECK_ERR_RET(recreateSwapchain());
            ready_to_resize = false;
            need_resize = false;
            wl_surface_commit(surface);
        }

        CHECK_VK_ERR_RET(
            vkWaitForFences(device, 1, &in_flight_fences[curr_frame], VK_TRUE, UINT64_MAX));
        uint32_t image_index;
        VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                                image_available_semaphores[curr_frame],
                                                VK_NULL_HANDLE, &image_index);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            CHECK_ERR_RET(recreateSwapchain());
            continue;
        } else {
            CHECK_VK_ERR_RET(result);
        }
        CHECK_VK_ERR_RET(vkResetFences(device, 1, &in_flight_fences[curr_frame]));

        CHECK_VK_ERR_RET(vkResetCommandBuffer(command_buffers[curr_frame], 0));
        CHECK_ERR_RET(recordCommandBuffer(command_buffers[curr_frame], image_index));

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &image_available_semaphores[curr_frame];
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        submit_info.pWaitDstStageMask = &wait_stage;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffers[curr_frame];
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &render_finished_semaphores[image_index];
        CHECK_VK_ERR_RET(
            vkQueueSubmit(graphics_queue, 1, &submit_info, in_flight_fences[curr_frame]));

        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &render_finished_semaphores[image_index];
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain;
        present_info.pImageIndices = &image_index;

        result = vkQueuePresentKHR(graphics_queue, &present_info);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            CHECK_ERR_RET(recreateSwapchain());
            continue;
        } else {
            CHECK_VK_ERR_RET(result);
        }

        curr_frame = (curr_frame + 1) % kMaxFramesInFight;
        wl_display_roundtrip(display);
    }

    return MyErrCode::kOk;
}

MyErrCode Application::initWayland(char const* title, int width, int height, char const* app_id)
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

    curr_width = width;
    curr_height = height;
    xdg_toplevel_set_title(toplevel, title);
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
    if (need_resize) {
        ready_to_resize = true;
    }
}

void Application::handleToplevelConfigure(void* data, struct xdg_toplevel* toplevel, int32_t width,
                                          int32_t height, struct wl_array* states)
{
    if (width != 0 && height != 0) {
        need_resize = true;
        new_width = width;
        new_height = height;
    }
}

void Application::handleToplevelClose(void* data, struct xdg_toplevel* toplevel)
{
    need_quit = true;
}

MyErrCode Application::initVulkan(char const* app_name)
{
    CHECK_ERR_RET(createInstance(app_name));
    CHECK_ERR_RET(setupDebugMessenger());
    CHECK_ERR_RET(createVkSurface());
    CHECK_ERR_RET(pickPhysicalDevice());
    CHECK_ERR_RET(createLogicalDevice());
    CHECK_ERR_RET(createSwapchain());
    CHECK_ERR_RET(createImageViews());
    CHECK_ERR_RET(createRenderPass());
    CHECK_ERR_RET(createFramebuffers());
    CHECK_ERR_RET(createCommandPool());
    CHECK_ERR_RET(createCommandBuffers());
    CHECK_ERR_RET(createSyncObjects());
    return MyErrCode::kOk;
}

MyErrCode Application::cleanupVulkan()
{
    CHECK_VK_ERR_RET(vkDeviceWaitIdle(device));
    CHECK_ERR_RET(cleanupSwapchain());
    vkDestroyRenderPass(device, render_pass, nullptr);
    for (int i = 0; i < swapchain_images.size(); ++i) {
        vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
    }
    for (int i = 0; i < kMaxFramesInFight; i++) {
        vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
        vkDestroyFence(device, in_flight_fences[i], nullptr);
    }
    vkFreeCommandBuffers(device, command_pool, command_buffers.size(), command_buffers.data());
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, vulkan_surface, nullptr);
    GET_VK_EXTENSION_FUNCTION(vkDestroyDebugUtilsMessengerEXT)(instance, debug_messenger, nullptr);
    vkDestroyInstance(instance, nullptr);
    return MyErrCode::kOk;
}

MyErrCode Application::recreateSwapchain()
{
    CHECK_VK_ERR_RET(vkDeviceWaitIdle(device));
    CHECK_ERR_RET(cleanupSwapchain());
    CHECK_ERR_RET(createSwapchain());
    CHECK_ERR_RET(createImageViews());
    CHECK_ERR_RET(createFramebuffers());
    return MyErrCode::kOk;
}

MyErrCode Application::cleanupSwapchain()
{
    for (auto frame_buffer: swapchain_frame_buffers) {
        vkDestroyFramebuffer(device, frame_buffer, nullptr);
    }
    for (auto image_view: swapchain_image_views) {
        vkDestroyImageView(device, image_view, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
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
    create_info.enabledExtensionCount = sizeof(kInstanceExtensions) / sizeof(char const*);
    create_info.ppEnabledExtensionNames = kInstanceExtensions;

    uint32_t layer_count;
    CHECK_VK_ERR_RET(vkEnumerateInstanceLayerProperties(&layer_count, nullptr));
    std::vector<VkLayerProperties> available_layers(layer_count);
    CHECK_VK_ERR_RET(vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data()));
    size_t found_layers = 0;
    for (uint32_t i = 0; i < layer_count; i++) {
        for (size_t j = 0; j < sizeof(kInstanceLayers) / sizeof(char const*); j++) {
            if (strcmp(available_layers[i].layerName, kInstanceLayers[j]) == 0) {
                found_layers++;
            }
        }
    }
    if (found_layers >= sizeof(kInstanceLayers) / sizeof(char const*)) {
        create_info.enabledLayerCount = sizeof(kInstanceLayers) / sizeof(char const*);
        create_info.ppEnabledLayerNames = kInstanceLayers;
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
            TLOG("{}", callback_data->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            DLOG("{}", callback_data->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            WLOG("{}", callback_data->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            ELOG("{}", callback_data->pMessage);
            break;
        default:
            break;
    }
    return VK_FALSE;
}

MyErrCode Application::createVkSurface()
{
    VkWaylandSurfaceCreateInfoKHR create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
    create_info.display = display;
    create_info.surface = surface;
    CHECK_VK_ERR_RET(vkCreateWaylandSurfaceKHR(instance, &create_info, nullptr, &vulkan_surface));
    return MyErrCode::kOk;
}

MyErrCode Application::pickPhysicalDevice()
{
    uint32_t device_count = 0;
    CHECK_VK_ERR_RET(vkEnumeratePhysicalDevices(instance, &device_count, nullptr));
    if (device_count == 0) {
        ELOG("failed to find gpu with vulkan support");
        return MyErrCode::kFailed;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    CHECK_VK_ERR_RET(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()));

    int best_score = 0;
    for (auto const& device: devices) {
        int score;
        CHECK_ERR_RET(rateDeviceSuitability(device, score));
        if (score > best_score) {
            physical_device = device;
            best_score = score;
        }
    }
    if (best_score <= 0) {
        ELOG("failed to find a suitable gpu");
        return MyErrCode::kFailed;
    }
    return MyErrCode::kOk;
}

MyErrCode Application::rateDeviceSuitability(VkPhysicalDevice device, int& score)
{
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(device, &properties);
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(device, &features);

    score = 0;
    switch (properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_OTHER:
            score += 1;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            score += 4;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            score += 5;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            score += 3;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            score += 2;
            break;
        default:
            break;
    }

    return MyErrCode::kOk;
}

MyErrCode Application::createLogicalDevice()
{
    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count,
                                             queue_families.data());

    for (uint32_t i = 0; i < queue_family_count; i++) {
        VkBool32 supported = 0;
        CHECK_VK_ERR_RET(
            vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, vulkan_surface, &supported));
        if (supported && (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            graphics_queue_family_index = i;
            break;
        }
    }

    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = graphics_queue_family_index;
    queue_create_info.queueCount = 1;
    float queue_priority = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_create_info;
    VkPhysicalDeviceFeatures device_features;
    vkGetPhysicalDeviceFeatures(physical_device, &device_features);
    create_info.pEnabledFeatures = &device_features;
    create_info.enabledExtensionCount = sizeof(kDeviceExtensions) / sizeof(char const*);
    create_info.ppEnabledExtensionNames = kDeviceExtensions;
    CHECK_VK_ERR_RET(vkCreateDevice(physical_device, &create_info, nullptr, &device));
    vkGetDeviceQueue(device, graphics_queue_family_index, 0, &graphics_queue);
    return MyErrCode::kOk;
}

MyErrCode Application::createCommandPool()
{
    VkCommandPoolCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    create_info.queueFamilyIndex = graphics_queue_family_index;
    create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECK_VK_ERR_RET(vkCreateCommandPool(device, &create_info, nullptr, &command_pool));
    return MyErrCode::kOk;
}

MyErrCode Application::createCommandBuffers()
{
    command_buffers.resize(kMaxFramesInFight);
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool;
    alloc_info.commandBufferCount = command_buffers.size();
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    CHECK_VK_ERR_RET(vkAllocateCommandBuffers(device, &alloc_info, command_buffers.data()));
    return MyErrCode::kOk;
}

MyErrCode Application::createSyncObjects()
{
    int image_cnt = swapchain_images.size();
    image_available_semaphores.resize(kMaxFramesInFight);
    render_finished_semaphores.resize(image_cnt);
    in_flight_fences.resize(kMaxFramesInFight);

    for (int i = 0; i < image_cnt; ++i) {
        {
            VkSemaphoreCreateInfo create_info = {};
            create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            CHECK_VK_ERR_RET(
                vkCreateSemaphore(device, &create_info, nullptr, &render_finished_semaphores[i]));
        }
    }
    for (int i = 0; i < kMaxFramesInFight; i++) {
        {
            VkSemaphoreCreateInfo create_info = {};
            create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            CHECK_VK_ERR_RET(
                vkCreateSemaphore(device, &create_info, nullptr, &image_available_semaphores[i]));
        }

        {
            VkFenceCreateInfo create_info = {};
            create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            CHECK_VK_ERR_RET(vkCreateFence(device, &create_info, nullptr, &in_flight_fences[i]));
        }
    }
    return MyErrCode::kOk;
}

MyErrCode Application::createImageViews()
{
    swapchain_image_views.resize(swapchain_images.size());
    for (size_t i = 0; i < swapchain_images.size(); i++) {
        VkImageViewCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        create_info.image = swapchain_images[i];
        create_info.format = swapchain_image_format;
        create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        create_info.subresourceRange.baseMipLevel = 0;
        create_info.subresourceRange.levelCount = 1;
        create_info.subresourceRange.baseArrayLayer = 0;
        create_info.subresourceRange.layerCount = 1;
        CHECK_VK_ERR_RET(
            vkCreateImageView(device, &create_info, nullptr, &swapchain_image_views[i]));
    }
    return MyErrCode::kOk;
}

MyErrCode Application::createFramebuffers()
{
    swapchain_frame_buffers.resize(swapchain_image_views.size());
    for (size_t i = 0; i < swapchain_image_views.size(); i++) {
        VkFramebufferCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        create_info.renderPass = render_pass;
        create_info.attachmentCount = 1;
        create_info.pAttachments = &swapchain_image_views[i];
        create_info.width = curr_width;
        create_info.height = curr_height;
        create_info.layers = 1;
        CHECK_VK_ERR_RET(
            vkCreateFramebuffer(device, &create_info, nullptr, &swapchain_frame_buffers[i]));
    }
    return MyErrCode::kOk;
}

MyErrCode Application::createSwapchain()
{
    VkSurfaceCapabilitiesKHR capabilities;
    CHECK_VK_ERR_RET(
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, vulkan_surface, &capabilities));

    uint32_t format_count;
    CHECK_VK_ERR_RET(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, vulkan_surface,
                                                          &format_count, nullptr));
    std::vector<VkSurfaceFormatKHR> available_formats(format_count);
    CHECK_VK_ERR_RET(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, vulkan_surface,
                                                          &format_count, available_formats.data()));

    VkSurfaceFormatKHR chosen_format = {};
    for (auto const& available_format: available_formats) {
        if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen_format = available_format;
        }
    }
    if (chosen_format.format == VK_FORMAT_UNDEFINED) {
        ELOG("failed to choose swapchain format");
        return MyErrCode::kFailed;
    }
    swapchain_image_format = chosen_format.format;

    VkSwapchainCreateInfoKHR create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = vulkan_surface;
    create_info.minImageCount =
        std::min(capabilities.minImageCount + 1, capabilities.maxImageCount);
    create_info.imageFormat = chosen_format.format;
    create_info.imageColorSpace = chosen_format.colorSpace;
    create_info.imageExtent.width = curr_width;
    create_info.imageExtent.height = curr_height;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.preTransform = capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    create_info.clipped = VK_TRUE;
    CHECK_VK_ERR_RET(vkCreateSwapchainKHR(device, &create_info, nullptr, &swapchain));

    uint32_t image_cnt;
    CHECK_VK_ERR_RET(vkGetSwapchainImagesKHR(device, swapchain, &image_cnt, nullptr));
    swapchain_images.resize(image_cnt);
    CHECK_VK_ERR_RET(
        vkGetSwapchainImagesKHR(device, swapchain, &image_cnt, swapchain_images.data()));

    DLOG("create {} swapchain elements with size={}x{}, format={}", image_cnt, curr_width,
         curr_height, swapchain_image_format);
    return MyErrCode::kOk;
}

MyErrCode Application::createRenderPass()
{
    VkAttachmentDescription attachment = {};
    attachment.format = swapchain_image_format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref = {};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;

    VkRenderPassCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    create_info.flags = 0;
    create_info.attachmentCount = 1;
    create_info.pAttachments = &attachment;
    create_info.subpassCount = 1;
    create_info.pSubpasses = &subpass;

    CHECK_VK_ERR_RET(vkCreateRenderPass(device, &create_info, nullptr, &render_pass));

    return MyErrCode::kOk;
}

MyErrCode Application::recordCommandBuffer(VkCommandBuffer command_buffer, uint32_t image_index)
{
    VkCommandBufferBeginInfo cmd_begin_info = {};
    cmd_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK_ERR_RET(vkBeginCommandBuffer(command_buffer, &cmd_begin_info));

    VkClearValue clear_value = {{0.5f, 0.5f, 0.5f, 1.0f}};
    VkRenderPassBeginInfo render_begin_info = {};
    render_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_begin_info.renderPass = render_pass;
    render_begin_info.framebuffer = swapchain_frame_buffers[image_index];
    render_begin_info.renderArea.offset.x = 0;
    render_begin_info.renderArea.offset.y = 0;
    render_begin_info.renderArea.extent.width = curr_width;
    render_begin_info.renderArea.extent.height = curr_height;
    render_begin_info.clearValueCount = 1;
    render_begin_info.pClearValues = &clear_value;
    vkCmdBeginRenderPass(command_buffer, &render_begin_info, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdEndRenderPass(command_buffer);

    CHECK_VK_ERR_RET(vkEndCommandBuffer(command_buffer));
    return MyErrCode::kOk;
}