#include "app.h"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

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
VkDescriptorSetLayout Application::descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool Application::descriptor_pool = VK_NULL_HANDLE;
VkPipeline Application::graphics_pipeline = VK_NULL_HANDLE;
VkPipelineLayout Application::pipeline_layout = VK_NULL_HANDLE;
VkCommandPool Application::command_pool = VK_NULL_HANDLE;
MyVkImage Application::texture_image = {VK_NULL_HANDLE, VK_NULL_HANDLE};
VkImageView Application::texture_image_view = VK_NULL_HANDLE;
VkSampler Application::texture_sampler = VK_NULL_HANDLE;
MyVkBuffer Application::vertex_buffer = {VK_NULL_HANDLE, VK_NULL_HANDLE};
MyVkBuffer Application::index_buffer = {VK_NULL_HANDLE, VK_NULL_HANDLE};
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
std::vector<MyVkBuffer> Application::uniform_buffers;
std::vector<VkDescriptorSet> Application::descriptor_sets;

bool Application::need_quit = false;
bool Application::ready_to_resize = false;
bool Application::need_resize = false;
int Application::new_width = 0;
int Application::new_height = 0;
uint32_t Application::curr_width = 0;
uint32_t Application::curr_height = 0;

MyErrCode Application::run(char const* win_title, char const* app_id)
{
    scene = std::make_shared<Scene>();
    CHECK_ERR_RET(scene->load());
    CHECK_ERR_RET(initWayland(win_title, app_id));
    CHECK_ERR_RET(initVulkan(app_id));
    CHECK_ERR_RET(scene->unload());
    CHECK_ERR_RET(mainLoop());
    CHECK_ERR_RET(cleanupVulkan());
    CHECK_ERR_RET(cleanupWayland());
    return MyErrCode::kOk;
}

MyErrCode Application::mainLoop()
{
    int curr_frame = 0;
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

        memcpy(uniform_buffers[curr_frame].alloc_info.pMappedData, scene->getUniformData(),
               scene->getUniformDataSize());

        CHECK_VK_ERR_RET(vkResetCommandBuffer(command_buffers[curr_frame], 0));
        CHECK_ERR_RET(recordCommandBuffer(curr_frame, image_index));

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

MyErrCode Application::initWayland(char const* win_title, char const* app_id)
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

    auto [width, height] = scene->getInitSize();
    curr_width = width;
    curr_height = height;
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
        scene->onResize(new_width, new_height);
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
    CHECK_ERR_RET(createVkAllocator());
    CHECK_ERR_RET(createSwapchain());
    CHECK_ERR_RET(createSwapchainImageViews());
    CHECK_ERR_RET(createRenderPass());
    CHECK_ERR_RET(createDescriptorSetLayout());
    CHECK_ERR_RET(createPipelineLayout());
    CHECK_ERR_RET(createGraphicsPipeline());
    CHECK_ERR_RET(createFramebuffers());
    CHECK_ERR_RET(createCommandPool());
    CHECK_ERR_RET(createTextureImage());
    CHECK_ERR_RET(createTextureImageView());
    CHECK_ERR_RET(createTextureSampler());
    CHECK_ERR_RET(createVertexBuffer());
    CHECK_ERR_RET(createIndexBuffer());
    CHECK_ERR_RET(createUniformBuffers());
    CHECK_ERR_RET(createDescriptorPool());
    CHECK_ERR_RET(createDescriptorSets());
    CHECK_ERR_RET(createCommandBuffers());
    CHECK_ERR_RET(createSyncObjects());
    return MyErrCode::kOk;
}

MyErrCode Application::cleanupVulkan()
{
    CHECK_VK_ERR_RET(vkDeviceWaitIdle(device));
    CHECK_ERR_RET(cleanupSwapchain());
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    vkDestroyPipeline(device, graphics_pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyRenderPass(device, render_pass, nullptr);
    for (int i = 0; i < swapchain_images.size(); ++i) {
        vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
    }
    for (int i = 0; i < kMaxFramesInFight; i++) {
        vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
        vkDestroyFence(device, in_flight_fences[i], nullptr);
        vmaDestroyBuffer(vma_allocator, uniform_buffers[i].buf, uniform_buffers[i].alloc);
    }
    vmaDestroyBuffer(vma_allocator, index_buffer.buf, index_buffer.alloc);
    vmaDestroyBuffer(vma_allocator, vertex_buffer.buf, vertex_buffer.alloc);
    vkDestroySampler(device, texture_sampler, nullptr);
    vkDestroyImageView(device, texture_image_view, nullptr);
    vmaDestroyImage(vma_allocator, texture_image.img, texture_image.alloc);
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
    CHECK_ERR_RET(createSwapchainImageViews());
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

MyErrCode Application::createVkBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                      VkMemoryPropertyFlags properties,
                                      VmaAllocationCreateFlags flags, MyVkBuffer& buffer)
{
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.flags = flags;
    alloc_info.requiredFlags = properties;

    CHECK_VK_ERR_RET(vmaCreateBuffer(vma_allocator, &buffer_info, &alloc_info, &buffer.buf,
                                     &buffer.alloc, &buffer.alloc_info));
    return MyErrCode::kOk;
}

MyErrCode Application::createVkImage(uint32_t width, uint32_t height, VkFormat format,
                                     VkImageTiling tiling, VkImageUsageFlags usage,
                                     VkMemoryPropertyFlags properties,
                                     VmaAllocationCreateFlags flags, MyVkImage& image)
{
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = width;
    image_info.extent.height = height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = format;
    image_info.tiling = tiling;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = usage;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.flags = flags;
    alloc_info.requiredFlags = properties;

    CHECK_VK_ERR_RET(vmaCreateImage(vma_allocator, &image_info, &alloc_info, &image.img,
                                    &image.alloc, &image.alloc_info));
    return MyErrCode::kOk;
}

MyErrCode Application::beginSingleTimeCommands(VkCommandBuffer& cmd_buf)
{
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool;
    alloc_info.commandBufferCount = 1;
    CHECK_VK_ERR_RET(vkAllocateCommandBuffers(device, &alloc_info, &cmd_buf));

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK_ERR_RET(vkBeginCommandBuffer(cmd_buf, &begin_info));
    return MyErrCode::kOk;
}

MyErrCode Application::endSingleTimeCommands(VkCommandBuffer cmd_buf)
{
    CHECK_VK_ERR_RET(vkEndCommandBuffer(cmd_buf));

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buf;
    CHECK_VK_ERR_RET(vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE));
    CHECK_VK_ERR_RET(vkQueueWaitIdle(graphics_queue));

    vkFreeCommandBuffers(device, command_pool, 1, &cmd_buf);
    return MyErrCode::kOk;
}

MyErrCode Application::copyVkBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size)
{
    VkCommandBuffer cmd_buf;
    CHECK_ERR_RET(beginSingleTimeCommands(cmd_buf));

    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size;
    vkCmdCopyBuffer(cmd_buf, src_buffer, dst_buffer, 1, &copy_region);

    CHECK_ERR_RET(endSingleTimeCommands(cmd_buf));
    return MyErrCode::kOk;
}

MyErrCode Application::copyVkBufferToImage(VkBuffer buffer, VkImage image, uint32_t width,
                                           uint32_t height)
{
    VkCommandBuffer cmd_buf;
    CHECK_ERR_RET(beginSingleTimeCommands(cmd_buf));

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd_buf, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &region);

    CHECK_ERR_RET(endSingleTimeCommands(cmd_buf));
    return MyErrCode::kOk;
}

MyErrCode Application::transitionImageLayout(VkImage image, VkImageLayout old_layout,
                                             VkImageLayout new_layout)
{
    VkCommandBuffer cmd_buf;
    CHECK_ERR_RET(beginSingleTimeCommands(cmd_buf));

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags src_stage;
    VkPipelineStageFlags dest_stage;

    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
        new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dest_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dest_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        ELOG("unsupported layout transition: {} -> {}", old_layout, new_layout);
        return MyErrCode::kFailed;
    }

    vkCmdPipelineBarrier(cmd_buf, src_stage, dest_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    CHECK_ERR_RET(endSingleTimeCommands(cmd_buf));
    return MyErrCode::kOk;
}

MyErrCode Application::createImageView(VkImage image, VkFormat format, VkImageView& image_view)
{
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.format = format;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    CHECK_VK_ERR_RET(vkCreateImageView(device, &view_info, nullptr, &image_view));
    return MyErrCode::kOk;
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

MyErrCode Application::createTextureImage()
{
    auto fpath = scene->getTextureImagePath().string();
    cv::Mat img_file = cv::imread(fpath, cv::IMREAD_COLOR);
    if (img_file.data == nullptr) {
        ELOG("failed to load image file: {}", fpath);
        return MyErrCode::kFailed;
    }
    cv::Mat img;
    cv::cvtColor(img_file, img, cv::COLOR_BGR2BGRA);
    int image_width = img.cols;
    int image_height = img.rows;
    VkDeviceSize image_size = img.total() * img.elemSize();

    MyVkBuffer staging_buffer;
    CHECK_ERR_RET(
        createVkBuffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       VMA_ALLOCATION_CREATE_MAPPED_BIT, staging_buffer));

    memcpy(staging_buffer.alloc_info.pMappedData, img.data, image_size);

    CHECK_ERR_RET(createVkImage(image_width, image_height, VK_FORMAT_B8G8R8A8_SRGB,
                                VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, texture_image));

    CHECK_ERR_RET(transitionImageLayout(texture_image.img, VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));

    CHECK_ERR_RET(
        copyVkBufferToImage(staging_buffer.buf, texture_image.img, image_width, image_height));

    CHECK_ERR_RET(transitionImageLayout(texture_image.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));

    vmaDestroyBuffer(vma_allocator, staging_buffer.buf, staging_buffer.alloc);
    return MyErrCode::kOk;
}

MyErrCode Application::createTextureImageView()
{
    CHECK_ERR_RET(createImageView(texture_image.img, VK_FORMAT_B8G8R8A8_SRGB, texture_image_view));
    return MyErrCode::kOk;
}

MyErrCode Application::createTextureSampler()
{
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_device, &properties);

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.mipLodBias = 0.0f;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = 0.0f;
    CHECK_VK_ERR_RET(vkCreateSampler(device, &sampler_info, nullptr, &texture_sampler));

    return MyErrCode::kOk;
}

MyErrCode Application::createVertexBuffer()
{
    VkDeviceSize buffer_size = scene->getVerticeDataSize();

    MyVkBuffer staging_buffer;
    CHECK_ERR_RET(
        createVkBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       VMA_ALLOCATION_CREATE_MAPPED_BIT, staging_buffer));

    memcpy(staging_buffer.alloc_info.pMappedData, scene->getVerticeData(), buffer_size);

    CHECK_ERR_RET(createVkBuffer(
        buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, vertex_buffer));

    CHECK_ERR_RET(copyVkBuffer(staging_buffer.buf, vertex_buffer.buf, buffer_size));
    vmaDestroyBuffer(vma_allocator, staging_buffer.buf, staging_buffer.alloc);
    return MyErrCode::kOk;
}

MyErrCode Application::createIndexBuffer()
{
    VkDeviceSize buffer_size = scene->getIndexDataSize();

    MyVkBuffer staging_buffer;
    CHECK_ERR_RET(
        createVkBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       VMA_ALLOCATION_CREATE_MAPPED_BIT, staging_buffer));

    memcpy(staging_buffer.alloc_info.pMappedData, scene->getIndexData(), buffer_size);

    CHECK_ERR_RET(createVkBuffer(
        buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, index_buffer));

    CHECK_ERR_RET(copyVkBuffer(staging_buffer.buf, index_buffer.buf, buffer_size));
    vmaDestroyBuffer(vma_allocator, staging_buffer.buf, staging_buffer.alloc);
    return MyErrCode::kOk;
}

MyErrCode Application::createUniformBuffers()
{
    VkDeviceSize buffer_size = scene->getUniformDataSize();
    uniform_buffers.resize(kMaxFramesInFight);
    for (int i = 0; i < kMaxFramesInFight; i++) {
        CHECK_ERR_RET(createVkBuffer(
            buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VMA_ALLOCATION_CREATE_MAPPED_BIT, uniform_buffers[i]));
    }
    return MyErrCode::kOk;
}

MyErrCode Application::createDescriptorPool()
{
    std::array<VkDescriptorPoolSize, 2> pool_sizes{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[0].descriptorCount = kMaxFramesInFight;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[1].descriptorCount = kMaxFramesInFight;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = kMaxFramesInFight;

    CHECK_VK_ERR_RET(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool));
    return MyErrCode::kOk;
}

MyErrCode Application::createDescriptorSets()
{
    descriptor_sets.resize(kMaxFramesInFight);
    std::vector<VkDescriptorSetLayout> layouts(kMaxFramesInFight, descriptor_set_layout);
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool;
    alloc_info.descriptorSetCount = kMaxFramesInFight;
    alloc_info.pSetLayouts = layouts.data();
    CHECK_VK_ERR_RET(vkAllocateDescriptorSets(device, &alloc_info, descriptor_sets.data()));

    for (int i = 0; i < kMaxFramesInFight; i++) {
        VkDescriptorBufferInfo buffer_info{};
        buffer_info.buffer = uniform_buffers[i].buf;
        buffer_info.offset = 0;
        buffer_info.range = scene->getUniformDataSize();

        VkDescriptorImageInfo image_info{};
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_info.imageView = texture_image_view;
        image_info.sampler = texture_sampler;

        std::array<VkWriteDescriptorSet, 2> descriptor_writes{};
        descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[0].dstSet = descriptor_sets[i];
        descriptor_writes[0].dstBinding = 0;
        descriptor_writes[0].dstArrayElement = 0;
        descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptor_writes[0].descriptorCount = 1;
        descriptor_writes[0].pBufferInfo = &buffer_info;

        descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[1].dstSet = descriptor_sets[i];
        descriptor_writes[1].dstBinding = 1;
        descriptor_writes[1].dstArrayElement = 0;
        descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptor_writes[1].descriptorCount = 1;
        descriptor_writes[1].pImageInfo = &image_info;

        vkUpdateDescriptorSets(device, descriptor_writes.size(), descriptor_writes.data(), 0,
                               nullptr);
    }
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

MyErrCode Application::createSwapchainImageViews()
{
    swapchain_image_views.resize(swapchain_images.size());
    for (size_t i = 0; i < swapchain_images.size(); i++) {
        CHECK_ERR_RET(
            createImageView(swapchain_images[i], swapchain_image_format, swapchain_image_views[i]));
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

MyErrCode Application::createDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding ubo_layout_binding{};
    ubo_layout_binding.binding = 0;
    ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ubo_layout_binding.descriptorCount = 1;
    ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding sampler_layout_binding{};
    sampler_layout_binding.binding = 1;
    sampler_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sampler_layout_binding.descriptorCount = 1;
    sampler_layout_binding.pImmutableSamplers = nullptr;
    sampler_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {ubo_layout_binding,
                                                            sampler_layout_binding};
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = bindings.size();
    layout_info.pBindings = bindings.data();
    CHECK_VK_ERR_RET(
        vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout));

    return MyErrCode::kOk;
}

MyErrCode Application::createPipelineLayout()
{
    VkPipelineLayoutCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    create_info.setLayoutCount = 1;
    create_info.pSetLayouts = &descriptor_set_layout;
    create_info.pushConstantRangeCount = 0;
    create_info.pPushConstantRanges = nullptr;
    CHECK_VK_ERR_RET(vkCreatePipelineLayout(device, &create_info, nullptr, &pipeline_layout));
    return MyErrCode::kOk;
}

MyErrCode Application::createGraphicsPipeline()
{
    // shader modules
    VkShaderModule vert_mod;
    CHECK_ERR_RET(createShaderModule(scene->getVertSpvPath(), vert_mod));
    VkShaderModule frag_mod;
    CHECK_ERR_RET(createShaderModule(scene->getFragSpvPath(), frag_mod));

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
    rasterizer.frontFace = scene->getFrontFace();
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

MyErrCode Application::recordCommandBuffer(int curr_frame, int image_index)
{
    VkCommandBuffer command_buffer = command_buffers[curr_frame];

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

    VkBuffer vertex_buffers[] = {vertex_buffer.buf};
    VkDeviceSize vertex_buffer_offsets[] = {0};
    vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, vertex_buffer_offsets);
    vkCmdBindIndexBuffer(command_buffer, index_buffer.buf, 0, VK_INDEX_TYPE_UINT32);

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

    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
                            &descriptor_sets[curr_frame], 0, nullptr);

    vkCmdDrawIndexed(command_buffer, scene->getIndexSize(), 1, 0, 0, 0);

    vkCmdEndRenderPass(command_buffer);
    CHECK_VK_ERR_RET(vkEndCommandBuffer(command_buffer));
    return MyErrCode::kOk;
}
