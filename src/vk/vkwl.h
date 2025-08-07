#pragma once
#include "scene.h"
#include <wayland-client.h>
#include <xdg-shell.h>
#include <vulkan/vulkan_wayland.h>
#include <vk_mem_alloc.h>

struct MyVkBuffer
{
    VkBuffer buf;
    VmaAllocation alloc;
    VmaAllocationInfo alloc_info;
};

struct MyVkImage
{
    VkImage img;
    VmaAllocation alloc;
    VmaAllocationInfo alloc_info;
};

class Application
{
public:
    static MyErrCode run(char const* win_title, char const* app_id);

private:
    static MyErrCode initWayland(char const* win_title, char const* app_id);
    static MyErrCode cleanupWayland();
    static void handleRegistry(void* data, struct wl_registry* registry, uint32_t name,
                               char const* interface, uint32_t version);
    static void handleShellPing(void* data, struct xdg_wm_base* shell, uint32_t serial);
    static void handleShellSurfaceConfigure(void* data, struct xdg_surface* shell_surface,
                                            uint32_t serial);
    static void handleToplevelConfigure(void* data, struct xdg_toplevel* toplevel, int32_t width,
                                        int32_t height, struct wl_array* states);
    static void handleToplevelClose(void* data, struct xdg_toplevel* toplevel);
    static void handleSeatCapabilities(void* data, struct wl_seat* seat, uint32_t caps);
    static void handlePointerEnter(void* data, struct wl_pointer* pointer, uint32_t serial,
                                   struct wl_surface* surface, wl_fixed_t sx, wl_fixed_t sy);
    static void handlePointerLeave(void* data, struct wl_pointer* pointer, uint32_t serial,
                                   struct wl_surface* surface);
    static void handlePointerMotion(void* data, struct wl_pointer* pointer, uint32_t time,
                                    wl_fixed_t sx, wl_fixed_t sy);
    static void handlePointerButton(void* data, struct wl_pointer* wl_pointer, uint32_t serial,
                                    uint32_t time, uint32_t button, uint32_t state);
    static void handlePointerAxis(void* data, struct wl_pointer* wl_pointer, uint32_t time,
                                  uint32_t axis, wl_fixed_t value);
    static void handleKeyboardKeymap(void* data, struct wl_keyboard* keyboard, uint32_t format,
                                     int fd, uint32_t size);
    static void handleKeyboardEnter(void* data, struct wl_keyboard* keyboard, uint32_t serial,
                                    struct wl_surface* surface, struct wl_array* keys);
    static void handleKeyboardLeave(void* data, struct wl_keyboard* keyboard, uint32_t serial,
                                    struct wl_surface* surface);
    static void handleKeyboardKey(void* data, struct wl_keyboard* keyboard, uint32_t serial,
                                  uint32_t time, uint32_t key, uint32_t state);
    static void handleKeyboardModifiers(void* data, struct wl_keyboard* keyboard, uint32_t serial,
                                        uint32_t mods_depressed, uint32_t mods_latched,
                                        uint32_t mods_locked, uint32_t group);

private:
    static MyErrCode createShaderModule(std::filesystem::path const& fp, VkShaderModule& mod);
    static MyErrCode createVkBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                    VkMemoryPropertyFlags properties,
                                    VmaAllocationCreateFlags flags, MyVkBuffer& buffer);
    static MyErrCode createVkImage(uint32_t width, uint32_t height, uint32_t mip_levels,
                                   VkSampleCountFlagBits num_samples, VkFormat format,
                                   VkImageTiling tiling, VkImageUsageFlags usage,
                                   VkMemoryPropertyFlags properties, VmaAllocationCreateFlags flags,
                                   MyVkImage& image);
    static MyErrCode createVkImageView(VkImage image, VkFormat format, uint32_t mip_levels,
                                       VkImageAspectFlags aspect_flags, VkImageView& image_view);
    static MyErrCode beginSingleTimeCommands(VkCommandBuffer& cmd_buf);
    static MyErrCode endSingleTimeCommands(VkCommandBuffer cmd_buf);
    static MyErrCode copyVkBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size);
    static MyErrCode copyVkBufferToImage(VkBuffer buffer, VkImage image, uint32_t width,
                                         uint32_t height);
    static MyErrCode transitionImageLayout(VkImage image, VkImageLayout old_layout,
                                           VkImageLayout new_layout, uint32_t mip_levels);
    static MyErrCode generateMipmaps(VkImage image, int32_t width, int32_t height,
                                     uint32_t mip_levels);
    static VkBool32 debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                  VkDebugUtilsMessageTypeFlagsEXT type,
                                  VkDebugUtilsMessengerCallbackDataEXT const* callback_data,
                                  void* user_data);
    static MyErrCode initVulkan(char const* app_name);
    static MyErrCode cleanupVulkan();
    static MyErrCode createInstance(char const* app_name);
    static MyErrCode setupDebugMessenger();
    static MyErrCode populateDebugMessengerInfo(VkDebugUtilsMessengerCreateInfoEXT& create_info);
    static MyErrCode createVkSurface();
    static MyErrCode pickPhysicalDevice();
    static MyErrCode rateDeviceSuitability(VkPhysicalDevice device, int& score);
    static MyErrCode createLogicalDevice();
    static MyErrCode createVkAllocator();
    static MyErrCode cleanupSwapchain();
    static MyErrCode createSwapchain();
    static MyErrCode recreateSwapchain();
    static MyErrCode createSwapchainImageViews();
    static MyErrCode createColorImagesAndViews();
    static MyErrCode createDepthImagesAndViews();
    static MyErrCode createRenderPass();
    static MyErrCode createFramebuffers();
    static MyErrCode createDescriptorSetLayout();
    static MyErrCode createPipelineLayout();
    static MyErrCode createGraphicsPipeline();
    static MyErrCode recreateGraphicsPipeline();
    static MyErrCode createCommandPool();
    static MyErrCode createTextureImage();
    static MyErrCode createTextureImageView();
    static MyErrCode createTextureSampler();
    static MyErrCode createVertexBuffer();
    static MyErrCode createIndexBuffer();
    static MyErrCode createUniformBuffers();
    static MyErrCode createDescriptorPool();
    static MyErrCode createDescriptorSets();
    static MyErrCode createCommandBuffers();
    static MyErrCode createSyncObjects();
    static MyErrCode setupImgui();
    static MyErrCode recordCommandBuffer(int curr_frame, int image_index);
    static MyErrCode mainLoop();

private:
    static wl_display* display;
    static wl_registry* registry;
    static wl_compositor* compositor;
    static wl_surface* surface;
    static xdg_wm_base* shell;
    static xdg_surface* shell_surface;
    static xdg_toplevel* toplevel;
    static wl_seat* seat;
    static wl_pointer* pointer;
    static wl_keyboard* keyboard;
    static wl_registry_listener registry_listener;
    static xdg_wm_base_listener shell_listener;
    static xdg_surface_listener shell_surface_listener;
    static xdg_toplevel_listener toplevel_listener;
    static wl_seat_listener seat_listener;
    static wl_pointer_listener pointer_listener;
    static wl_keyboard_listener keyboard_listener;

    static VkInstance instance;
    static constexpr uint32_t kVkApiVersion = VK_API_VERSION_1_3;
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
    static VkDescriptorPool imgui_descriptor_pool;
    static VkDescriptorSetLayout descriptor_set_layout;
    static VkDescriptorPool descriptor_pool;
    static VkPipeline graphics_pipeline;
    static VkPipelineLayout pipeline_layout;
    static VkCommandPool command_pool;
    static uint32_t texture_mip_levels;
    static MyVkImage texture_image;
    static VkImageView texture_image_view;
    static VkSampler texture_sampler;
    static MyVkBuffer vertex_buffer;
    static MyVkBuffer index_buffer;
    static VkRenderPass render_pass;
    static VkSwapchainKHR swapchain;
    static VkFormat swapchain_image_format;
    static VkFormat depth_image_format;
    static constexpr VkSampleCountFlagBits kMsaaSamples = VK_SAMPLE_COUNT_8_BIT;
    static std::vector<VkImage> swapchain_images;
    static std::vector<VkImageView> swapchain_image_views;
    static std::vector<MyVkImage> color_images;
    static std::vector<VkImageView> color_image_views;
    static std::vector<MyVkImage> depth_images;
    static std::vector<VkImageView> depth_image_views;
    static std::vector<VkFramebuffer> swapchain_frame_buffers;
    static std::vector<VkSemaphore> render_finished_semaphores;
    static constexpr int kMaxFramesInFight = 2;
    static std::vector<VkCommandBuffer> command_buffers;
    static std::vector<VkSemaphore> image_available_semaphores;
    static std::vector<VkFence> in_flight_fences;
    static std::vector<MyVkBuffer> uniform_buffers;
    static std::vector<VkDescriptorSet> descriptor_sets;

    static std::shared_ptr<Scene> scene;
    static bool need_quit;
    static bool need_resize;
    static bool ready_to_resize;
    static int new_width;
    static int new_height;
    static uint32_t curr_width;
    static uint32_t curr_height;
};
