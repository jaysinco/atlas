#include "app.h"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"

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

std::shared_ptr<Scene> Application::scene = nullptr;
VkInstance Application::instance = VK_NULL_HANDLE;
VkDebugUtilsMessengerEXT Application::debug_messenger = VK_NULL_HANDLE;
VkSurfaceKHR Application::vulkan_surface = VK_NULL_HANDLE;
VkPhysicalDevice Application::physical_device = VK_NULL_HANDLE;
VkDevice Application::device = VK_NULL_HANDLE;
VmaAllocator Application::vma_allocator = VK_NULL_HANDLE;
uint32_t Application::graphics_queue_family_index = 0;
VkQueue Application::graphics_queue = VK_NULL_HANDLE;
VkPipeline Application::graphics_pipeline = VK_NULL_HANDLE;
VkPipelineLayout Application::pipeline_layout = VK_NULL_HANDLE;
VkCommandPool Application::command_pool = VK_NULL_HANDLE;
VkBuffer Application::vertex_buffer = VK_NULL_HANDLE;
VmaAllocation Application::vertex_buffer_alloc = VK_NULL_HANDLE;
VkBuffer Application::index_buffer = VK_NULL_HANDLE;
VmaAllocation Application::index_buffer_alloc = VK_NULL_HANDLE;
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
    scene = std::make_shared<Scene>();
    CHECK_ERR_RET(scene->load());
    CHECK_ERR_RET(createInstance(app_name));
    CHECK_ERR_RET(setupDebugMessenger());
    CHECK_ERR_RET(createVkSurface());
    CHECK_ERR_RET(pickPhysicalDevice());
    CHECK_ERR_RET(createLogicalDevice());
    CHECK_ERR_RET(createVkAllocator());
    CHECK_ERR_RET(createSwapchain());
    CHECK_ERR_RET(createImageViews());
    CHECK_ERR_RET(createRenderPass());
    CHECK_ERR_RET(createPipelineLayout());
    CHECK_ERR_RET(createGraphicsPipeline());
    CHECK_ERR_RET(createFramebuffers());
    CHECK_ERR_RET(createCommandPool());
    CHECK_ERR_RET(createVertexBuffer());
    CHECK_ERR_RET(createIndexBuffer());
    CHECK_ERR_RET(createCommandBuffers());
    CHECK_ERR_RET(createSyncObjects());
    CHECK_ERR_RET(scene->unload());
    return MyErrCode::kOk;
}

MyErrCode Application::cleanupVulkan()
{
    CHECK_VK_ERR_RET(vkDeviceWaitIdle(device));
    CHECK_ERR_RET(cleanupSwapchain());
    vkDestroyPipeline(device, graphics_pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyRenderPass(device, render_pass, nullptr);
    for (int i = 0; i < swapchain_images.size(); ++i) {
        vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
    }
    for (int i = 0; i < kMaxFramesInFight; i++) {
        vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
        vkDestroyFence(device, in_flight_fences[i], nullptr);
    }
    vkFreeCommandBuffers(device, command_pool, command_buffers.size(), command_buffers.data());
    vmaDestroyBuffer(vma_allocator, index_buffer, index_buffer_alloc);
    vmaDestroyBuffer(vma_allocator, vertex_buffer, vertex_buffer_alloc);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vmaDestroyAllocator(vma_allocator);
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

    VkDebugUtilsMessengerCreateInfoEXT debug_info{};
    CHECK_ERR_RET(populateDebugMessengerInfo(debug_info));

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.pNext = &debug_info;

    // check extensions
    uint32_t extension_count = 0;
    CHECK_VK_ERR_RET(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr));
    std::vector<VkExtensionProperties> available_extensions(extension_count);
    CHECK_VK_ERR_RET(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count,
                                                            available_extensions.data()));
    for (size_t j = 0; j < sizeof(kInstanceExtensions) / sizeof(char const*); j++) {
        bool found = false;
        for (uint32_t i = 0; i < extension_count; i++) {
            if (strcmp(available_extensions[i].extensionName, kInstanceExtensions[j]) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            ELOG("extension not found: {}", kInstanceExtensions[j]);
        }
    }
    create_info.enabledExtensionCount = sizeof(kInstanceExtensions) / sizeof(char const*);
    create_info.ppEnabledExtensionNames = kInstanceExtensions;

    // check layers
    uint32_t layer_count;
    CHECK_VK_ERR_RET(vkEnumerateInstanceLayerProperties(&layer_count, nullptr));
    std::vector<VkLayerProperties> available_layers(layer_count);
    CHECK_VK_ERR_RET(vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data()));
    for (size_t j = 0; j < sizeof(kInstanceLayers) / sizeof(char const*); j++) {
        bool found = false;
        for (uint32_t i = 0; i < layer_count; i++) {
            if (strcmp(available_layers[i].layerName, kInstanceLayers[j]) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            ELOG("layer not found: {}", kInstanceLayers[j]);
        }
    }
    create_info.enabledLayerCount = sizeof(kInstanceLayers) / sizeof(char const*);
    create_info.ppEnabledLayerNames = kInstanceLayers;

    CHECK_VK_ERR_RET(vkCreateInstance(&create_info, nullptr, &instance));
    return MyErrCode::kOk;
}

MyErrCode Application::populateDebugMessengerInfo(VkDebugUtilsMessengerCreateInfoEXT& create_info)
{
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
    return MyErrCode::kOk;
}

MyErrCode Application::setupDebugMessenger()
{
    VkDebugUtilsMessengerCreateInfoEXT create_info{};
    CHECK_ERR_RET(populateDebugMessengerInfo(create_info));
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

MyErrCode Application::createVkAllocator()
{
    VmaAllocatorCreateInfo create_info = {};
    create_info.vulkanApiVersion = VK_API_VERSION_1_0;
    create_info.physicalDevice = physical_device;
    create_info.device = device;
    create_info.instance = instance;
    CHECK_VK_ERR_RET(vmaCreateAllocator(&create_info, &vma_allocator));
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

MyErrCode Application::createVertexBuffer()
{
    VkDeviceSize buffer_size = scene->getVerticeDataSize();

    VkBuffer staging_buffer;
    VmaAllocation staging_buffer_alloc;
    CHECK_ERR_RET(
        createVkBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       staging_buffer, staging_buffer_alloc));

    void* staging_data;
    CHECK_VK_ERR_RET(vmaMapMemory(vma_allocator, staging_buffer_alloc, &staging_data));
    memcpy(staging_data, scene->getVerticeData(), buffer_size);
    vmaUnmapMemory(vma_allocator, staging_buffer_alloc);

    CHECK_ERR_RET(createVkBuffer(
        buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer, vertex_buffer_alloc));

    CHECK_ERR_RET(copyVkBuffer(staging_buffer, vertex_buffer, buffer_size));
    vmaDestroyBuffer(vma_allocator, staging_buffer, staging_buffer_alloc);
    return MyErrCode::kOk;
}

MyErrCode Application::createIndexBuffer()
{
    VkDeviceSize buffer_size = scene->getIndexDataSize();

    VkBuffer staging_buffer;
    VmaAllocation staging_buffer_alloc;
    CHECK_ERR_RET(
        createVkBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       staging_buffer, staging_buffer_alloc));

    void* staging_data;
    CHECK_VK_ERR_RET(vmaMapMemory(vma_allocator, staging_buffer_alloc, &staging_data));
    memcpy(staging_data, scene->getIndexData(), buffer_size);
    vmaUnmapMemory(vma_allocator, staging_buffer_alloc);

    CHECK_ERR_RET(createVkBuffer(
        buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, index_buffer, index_buffer_alloc));

    CHECK_ERR_RET(copyVkBuffer(staging_buffer, index_buffer, buffer_size));
    vmaDestroyBuffer(vma_allocator, staging_buffer, staging_buffer_alloc);
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

    render_finished_semaphores.resize(image_cnt);
    for (int i = 0; i < image_cnt; ++i) {
        {
            VkSemaphoreCreateInfo create_info = {};
            create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            CHECK_VK_ERR_RET(
                vkCreateSemaphore(device, &create_info, nullptr, &render_finished_semaphores[i]));
        }
    }

    image_available_semaphores.resize(kMaxFramesInFight);
    for (int i = 0; i < kMaxFramesInFight; i++) {
        VkSemaphoreCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        CHECK_VK_ERR_RET(
            vkCreateSemaphore(device, &create_info, nullptr, &image_available_semaphores[i]));
    }

    in_flight_fences.resize(kMaxFramesInFight);
    for (int i = 0; i < kMaxFramesInFight; i++) {
        VkFenceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        CHECK_VK_ERR_RET(vkCreateFence(device, &create_info, nullptr, &in_flight_fences[i]));
    }
    return MyErrCode::kOk;
}

MyErrCode Application::createShaderModule(std::filesystem::path const& fp, VkShaderModule& mod)
{
    std::vector<uint8_t> code;
    CHECK_ERR_RET(toolkit::readBinaryFile(fp, code));
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size();
    create_info.pCode = reinterpret_cast<uint32_t const*>(code.data());
    CHECK_VK_ERR_RET(vkCreateShaderModule(device, &create_info, nullptr, &mod));
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

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    create_info.flags = 0;
    create_info.attachmentCount = 1;
    create_info.pAttachments = &attachment;
    create_info.subpassCount = 1;
    create_info.pSubpasses = &subpass;
    create_info.dependencyCount = 1;
    create_info.pDependencies = &dependency;

    CHECK_VK_ERR_RET(vkCreateRenderPass(device, &create_info, nullptr, &render_pass));

    return MyErrCode::kOk;
}

MyErrCode Application::createPipelineLayout()
{
    VkPipelineLayoutCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    create_info.setLayoutCount = 0;
    create_info.pSetLayouts = nullptr;
    create_info.pushConstantRangeCount = 0;
    create_info.pPushConstantRanges = nullptr;
    CHECK_VK_ERR_RET(vkCreatePipelineLayout(device, &create_info, nullptr, &pipeline_layout));
    return MyErrCode::kOk;
}

MyErrCode Application::createGraphicsPipeline()
{
    // shader modules
    VkShaderModule vert_mod;
    CHECK_ERR_RET(createShaderModule(toolkit::getDataDir() / "test.vert.spv", vert_mod));
    VkShaderModule frag_mod;
    CHECK_ERR_RET(createShaderModule(toolkit::getDataDir() / "test.frag.spv", frag_mod));

    // shader stage
    VkPipelineShaderStageCreateInfo vert_stage_info{};
    vert_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage_info.module = vert_mod;
    vert_stage_info.pName = "main";

    VkPipelineShaderStageCreateInfo frag_stage_info{};
    frag_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage_info.module = frag_mod;
    frag_stage_info.pName = "main";

    VkPipelineShaderStageCreateInfo shader_stages_info[] = {vert_stage_info, frag_stage_info};

    // dynamic state
    std::vector<VkDynamicState> dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,
                                                  VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_state_info{};
    dynamic_state_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state_info.dynamicStateCount = dynamic_states.size();
    dynamic_state_info.pDynamicStates = dynamic_states.data();

    // vertex input
    VkPipelineVertexInputStateCreateInfo vert_input_info{};
    vert_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    auto vert_bind_desc = scene->getVertexBindingDesc();
    vert_input_info.vertexBindingDescriptionCount = 1;
    vert_input_info.pVertexBindingDescriptions = &vert_bind_desc;
    auto vert_attr_descs = scene->getVertexAttrDescs();
    vert_input_info.vertexAttributeDescriptionCount = vert_attr_descs.size();
    vert_input_info.pVertexAttributeDescriptions = vert_attr_descs.data();

    // input assembly
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info{};
    input_assembly_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly_info.primitiveRestartEnable = VK_FALSE;

    // viewports and scissors
    VkPipelineViewportStateCreateInfo viewport_state_info{};
    viewport_state_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state_info.viewportCount = 1;
    viewport_state_info.scissorCount = 1;

    // rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    // multisampling
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    // color blending
    VkPipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.blendEnable = VK_FALSE;
    color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo color_blending{};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;
    color_blending.blendConstants[0] = 0.0f;
    color_blending.blendConstants[1] = 0.0f;
    color_blending.blendConstants[2] = 0.0f;
    color_blending.blendConstants[3] = 0.0f;

    // pipeline
    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages_info;
    pipeline_info.pVertexInputState = &vert_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly_info;
    pipeline_info.pViewportState = &viewport_state_info;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pDepthStencilState = nullptr;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_state_info;
    pipeline_info.layout = pipeline_layout;
    pipeline_info.renderPass = render_pass;
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_info.basePipelineIndex = -1;
    CHECK_VK_ERR_RET(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                               &graphics_pipeline));

    // cleanup
    vkDestroyShaderModule(device, frag_mod, nullptr);
    vkDestroyShaderModule(device, vert_mod, nullptr);

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

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

    VkBuffer vertex_buffers[] = {vertex_buffer};
    VkDeviceSize vertex_buffer_offsets[] = {0};
    vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, vertex_buffer_offsets);
    vkCmdBindIndexBuffer(command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = curr_width;
    viewport.height = curr_height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent.width = curr_width;
    scissor.extent.height = curr_height;
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    vkCmdDrawIndexed(command_buffer, scene->getIndexSize(), 1, 0, 0, 0);

    vkCmdEndRenderPass(command_buffer);
    CHECK_VK_ERR_RET(vkEndCommandBuffer(command_buffer));
    return MyErrCode::kOk;
}

MyErrCode Application::createVkBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                                      VmaAllocation& buffer_alloc)
{
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.requiredFlags = properties;

    CHECK_VK_ERR_RET(
        vmaCreateBuffer(vma_allocator, &buffer_info, &alloc_info, &buffer, &buffer_alloc, nullptr));
    return MyErrCode::kOk;
}

MyErrCode Application::copyVkBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size)
{
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool;
    alloc_info.commandBufferCount = 1;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    VkCommandBuffer copy_command_buffer;
    CHECK_VK_ERR_RET(vkAllocateCommandBuffers(device, &alloc_info, &copy_command_buffer));

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK_ERR_RET(vkBeginCommandBuffer(copy_command_buffer, &begin_info));

    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size;
    vkCmdCopyBuffer(copy_command_buffer, src_buffer, dst_buffer, 1, &copy_region);

    CHECK_VK_ERR_RET(vkEndCommandBuffer(copy_command_buffer));
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &copy_command_buffer;

    CHECK_VK_ERR_RET(vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE));
    CHECK_VK_ERR_RET(vkQueueWaitIdle(graphics_queue));
    vkFreeCommandBuffers(device, command_pool, 1, &copy_command_buffer);

    return MyErrCode::kOk;
}