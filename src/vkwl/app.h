#pragma once
#include "toolkit/error.h"
#include <wayland-client.h>
#include "xdg-shell.h"
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_wayland.h>
#include <vector>
#include <filesystem>
#include <glm/glm.hpp>
#include <vk_mem_alloc.h>

struct VertexData
{
    glm::vec2 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription();
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
};

class Application
{
public:
    static MyErrCode run(char const* win_title, int win_width, int win_height, char const* app_id);

private:
    static MyErrCode mainLoop();

private:
    static MyErrCode initWayland(char const* title, int width, int height, char const* app_id);
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
    static MyErrCode populateDebugMessengerInfo(VkDebugUtilsMessengerCreateInfoEXT& create_info);
    static VkBool32 debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                  VkDebugUtilsMessageTypeFlagsEXT type,
                                  VkDebugUtilsMessengerCallbackDataEXT const* callback_data,
                                  void* user_data);
    static MyErrCode createVkSurface();
    static MyErrCode pickPhysicalDevice();
    static MyErrCode rateDeviceSuitability(VkPhysicalDevice device, int& score);
    static MyErrCode createLogicalDevice();
    static MyErrCode createAllocator();
    static MyErrCode recreateSwapchain();
    static MyErrCode cleanupSwapchain();
    static MyErrCode createSwapchain();
    static MyErrCode createImageViews();
    static MyErrCode createRenderPass();
    static MyErrCode createPipelineLayout();
    static MyErrCode createGraphicsPipeline();
    static MyErrCode createFramebuffers();
    static MyErrCode createCommandPool();
    static MyErrCode createVertexBuffer();
    static MyErrCode createCommandBuffers();
    static MyErrCode createSyncObjects();
    static MyErrCode createShaderModule(std::filesystem::path const& fp, VkShaderModule& mod);
    static MyErrCode recordCommandBuffer(VkCommandBuffer command_buffer, uint32_t image_index);
    static MyErrCode createVkBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                    VkMemoryPropertyFlags properties, VkBuffer& buffer,
                                    VmaAllocation& buffer_alloc);
    static MyErrCode copyVkBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size);

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
    static constexpr char const* const kInstanceExtensions[] = {
        "VK_EXT_debug_utils", "VK_KHR_surface", "VK_KHR_wayland_surface"};
    static constexpr char const* const kInstanceLayers[] = {"VK_LAYER_KHRONOS_validation"};
    static VkDebugUtilsMessengerEXT debug_messenger;
    static VkSurfaceKHR vulkan_surface;
    static VkPhysicalDevice physical_device;
    static VkDevice device;
    static constexpr char const* const kDeviceExtensions[] = {"VK_KHR_swapchain"};
    static VmaAllocator vma_allocator;
    static uint32_t graphics_queue_family_index;
    static VkQueue graphics_queue;
    static VkPipeline graphics_pipeline;
    static VkPipelineLayout pipeline_layout;
    static VkCommandPool command_pool;
    static VkBuffer vertex_buffer;
    static VmaAllocation vertex_buffer_alloc;
    static VkRenderPass render_pass;
    static VkSwapchainKHR swapchain;
    static VkFormat swapchain_image_format;
    static std::vector<VkImage> swapchain_images;
    static std::vector<VkImageView> swapchain_image_views;
    static std::vector<VkFramebuffer> swapchain_frame_buffers;
    static constexpr int kMaxFramesInFight = 2;
    static std::vector<VkCommandBuffer> command_buffers;
    static std::vector<VkSemaphore> image_available_semaphores;
    static std::vector<VkSemaphore> render_finished_semaphores;
    static std::vector<VkFence> in_flight_fences;

    static bool need_quit;
    static bool need_resize;
    static bool ready_to_resize;
    static int new_width;
    static int new_height;
    static uint32_t curr_width;
    static uint32_t curr_height;
    static int curr_frame;
};
