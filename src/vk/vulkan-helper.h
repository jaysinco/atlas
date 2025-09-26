#pragma once
#define VULKAN_HPP_NO_NODISCARD_WARNINGS
#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include <parallel_hashmap/phmap.h>
#include <functional>
#include <set>
#include <map>

namespace toolkit
{

template <typename T, typename = void>
struct CanVkToString: std::false_type
{
};

template <typename T>
struct CanVkToString<T, std::void_t<decltype(vk::to_string(std::declval<T>()))>>
    : std::is_same<decltype(vk::to_string(std::declval<T>())), std::string>
{
};

template <typename T>
std::enable_if_t<CanVkToString<T>::value, std::string> toFormattable(T&& arg)
{
    return vk::to_string(std::forward<T>(arg));
}

}  // namespace toolkit

#include "toolkit/format.h"
#include "toolkit/toolkit.h"

#define CHECK_VK_RET(err)                            \
    do {                                             \
        if (auto _err = (err); _err != VK_SUCCESS) { \
            ELOG("failed to call vk: {}", _err);     \
            return MyErrCode::kFailed;               \
        }                                            \
    } while (0)

#define CHECK_VKHPP_RET(err)                                   \
    do {                                                       \
        if (auto _err = (err); _err != vk::Result::eSuccess) { \
            ELOG("failed to call vk: {}", _err);               \
            return MyErrCode::kFailed;                         \
        }                                                      \
    } while (0)

#define CHECK_VKHPP_VAL(err)                                  \
    ({                                                        \
        auto MY_CONCAT(res_vk_, __LINE__) = (err);            \
        CHECK_VKHPP_RET(MY_CONCAT(res_vk_, __LINE__).result); \
        MY_CONCAT(res_vk_, __LINE__).value;                   \
    })

namespace myvk
{

class Context;

class CommandBuffer;

using Uid = toolkit::Uid;

template <typename T>
using UidMap = phmap::parallel_node_hash_map_m<Uid, T>;

using DeviceRater = std::function<int(vk::PhysicalDevice const&)>;

using QueuePicker = std::function<bool(uint32_t family_index, vk::QueueFamilyProperties const&)>;

using CmdSubmitter = std::function<MyErrCode(CommandBuffer&)>;

constexpr uint32_t kRemainingMipLevels = ~0U;
constexpr uint32_t kRemainingArrayLayers = ~0U;
constexpr vk::DeviceSize kRemainingSize = ~0ULL;
constexpr vk::Extent3D kRemainingExtend = {~0U, ~0U, ~0U};
constexpr vk::ImageAspectFlags kAllImageAspects = vk::ImageAspectFlagBits::eNone;

struct BufferMeta
{
    uint64_t size = 0;
    vk::BufferUsageFlags usages = vk::BufferUsageFlagBits::eStorageBuffer;
};

struct ImageMeta
{
    vk::ImageType type = vk::ImageType::e2D;
    vk::Format format = vk::Format::eB8G8R8A8Srgb;
    vk::ImageAspectFlags aspects = vk::ImageAspectFlagBits::eColor;
    vk::Extent3D extent = {0, 0, 0};
    uint32_t layers = 1;
    uint32_t mip_levels = 1;
    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
    vk::ImageTiling tiling = vk::ImageTiling::eOptimal;
    vk::ImageLayout init_layout = vk::ImageLayout::eUndefined;
    vk::ImageUsageFlags usages = vk::ImageUsageFlagBits::eSampled;
};

struct ImageSubLayers
{
    ImageSubLayers(uint32_t base_level = 0, uint32_t base_layer = 0,
                   uint32_t num_layers = kRemainingArrayLayers,
                   vk::ImageAspectFlags aspects = kAllImageAspects);

    uint32_t base_level;
    uint32_t base_layer;
    uint32_t num_layers;
    vk::ImageAspectFlags aspects;
};

struct ImageSubRange: ImageSubLayers
{
    ImageSubRange(uint32_t base_level = 0, uint32_t num_levels = kRemainingMipLevels,
                  uint32_t base_layer = 0, uint32_t num_layers = kRemainingArrayLayers,
                  vk::ImageAspectFlags aspects = kAllImageAspects);

    uint32_t num_levels;
};

struct ImageSubArea: ImageSubLayers
{
    ImageSubArea(uint32_t base_level = 0, vk::Offset3D const& offset = {0, 0, 0},
                 vk::Extent3D const& extent = kRemainingExtend, uint32_t base_layer = 0,
                 uint32_t num_layers = kRemainingArrayLayers,
                 vk::ImageAspectFlags aspects = kAllImageAspects);

    vk::Offset3D offset;
    vk::Extent3D extent;
};

struct BufferImageArea
{
    vk::DeviceSize offset = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct ImageViewMeta: ImageSubRange
{
    ImageViewMeta(vk::ImageViewType type = vk::ImageViewType::e2D,
                  vk::ComponentMapping swizzle = {}, uint32_t base_level = 0,
                  uint32_t num_levels = kRemainingMipLevels, uint32_t base_layer = 0,
                  uint32_t num_layers = kRemainingArrayLayers,
                  vk::ImageAspectFlags aspects = kAllImageAspects);

    vk::ImageViewType type;
    vk::ComponentMapping swizzle;
};

struct SwapchainMeta
{
    uint32_t image_count = 1;
    vk::SurfaceFormatKHR surface_format = {vk::Format::eB8G8R8A8Srgb,
                                           vk::ColorSpaceKHR::eSrgbNonlinear};
    vk::Extent2D extent = {0, 0};
    vk::PresentModeKHR mode = vk::PresentModeKHR::eMailbox;
    vk::ImageUsageFlags usages = vk::ImageUsageFlagBits::eColorAttachment;
};

struct RenderPassMeta
{
    std::vector<vk::AttachmentDescription> attachments;
    std::vector<vk::AttachmentReference> attachment_refs;
    std::vector<vk::SubpassDescription> subpasses;
    std::vector<vk::SubpassDependency> dependencies;
};

struct SamplerMeta
{
    vk::Filter filter = vk::Filter::eLinear;
    vk::SamplerAddressMode address_mode = vk::SamplerAddressMode::eRepeat;
    vk::BorderColor border_color = vk::BorderColor::eIntOpaqueBlack;
    float min_lod = 0;
    float max_lod = 1;
};

struct MemoryBarrierMeta
{
    vk::PipelineStageFlags2 src_stage;
    vk::AccessFlags2 src_access;
    vk::PipelineStageFlags2 dst_stage;
    vk::AccessFlags2 dst_access;
};

struct ImageBarrierMeta: MemoryBarrierMeta
{
    vk::ImageLayout old_layout;
    vk::ImageLayout new_layout;
};

struct GraphicPipelineMeta
{
    Uid pipeline_layout_id;
    Uid render_pass_id;
    Uid vert_shader_id = Uid::kNull;
    Uid frag_shader_id = Uid::kNull;
    uint32_t subpass = 0;
    uint32_t viewport_count = 1;
    uint32_t scissor_count = 1;
    bool enable_primitive_restart = false;
    bool enable_depth_clamp = false;
    bool enable_raster_discard = false;
    bool enable_sample_shading = true;
    bool enable_blend = false;
    bool enable_depth_test = true;
    bool enable_depth_write = true;
    bool enable_depth_bounds_test = false;
    float min_depth_bounds = 0.0f;
    float max_depth_bounds = 1.0f;
    float raster_line_width = 1.0f;
    vk::PrimitiveTopology prmitive_topology = vk::PrimitiveTopology::eTriangleList;
    vk::PolygonMode polygon_mode = vk::PolygonMode::eFill;
    vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eBack;
    vk::FrontFace front_face = vk::FrontFace::eClockwise;
    vk::SampleCountFlagBits raster_samples = vk::SampleCountFlagBits::e1;
    vk::CompareOp depth_compare_op = vk::CompareOp::eLess;
    std::vector<vk::VertexInputBindingDescription> vert_input_binds;
    std::vector<vk::VertexInputAttributeDescription> vert_input_attrs;
    std::vector<vk::DynamicState> dynamic_states = {vk::DynamicState::eViewport,
                                                    vk::DynamicState::eScissor};
};

class Allocation
{
public:
    Allocation(VmaAllocation alloc, VmaAllocator allocator);
    vk::MemoryPropertyFlags getMemProp() const;
    VmaAllocationInfo getAllocInfo() const;
    bool canBeMapped() const;
    bool isMapped() const;
    void* map();
    void unmap();
    void* getMappedData() const;
    void invalid(vk::DeviceSize offset = 0, vk::DeviceSize size = vk::WholeSize);
    void flush(vk::DeviceSize offset = 0, vk::DeviceSize size = vk::WholeSize);

private:
    friend class Context;
    VmaAllocation alloc_;
    VmaAllocator allocator_;
};

class Buffer: public Allocation
{
public:
    Buffer(BufferMeta const& meta, vk::Buffer buf, VmaAllocation alloc, VmaAllocator allocator);
    BufferMeta const& getMeta() const;
    operator vk::Buffer() const;
    operator bool() const;

private:
    friend class Context;
    BufferMeta meta_;
    vk::Buffer buf_;
};

class Image: public Allocation
{
public:
    Image(ImageMeta const& meta, vk::Image img, VmaAllocation alloc, VmaAllocator allocator);
    ImageMeta const& getMeta() const;
    vk::Extent3D getMipExtent(uint32_t mip_level) const;
    operator vk::Image() const;
    operator bool() const;
    static uint32_t getMaxMipLevels(vk::Extent3D const& base);

private:
    friend class Context;
    ImageMeta meta_;
    vk::Image img_;
};

class ImageView
{
public:
    ImageView(ImageViewMeta const& meta, ImageMeta const& img_meta, vk::ImageView view);
    ImageViewMeta const& getMeta() const;
    operator vk::ImageView() const;
    operator bool() const;

private:
    friend class Context;
    ImageViewMeta meta_;
    ImageMeta const* img_meta_;
    vk::ImageView view_;
};

class Semaphore
{
public:
    Semaphore(vk::SemaphoreType type, vk::Semaphore sem);
    bool isTimeline() const;
    operator vk::Semaphore() const;
    operator bool() const;

private:
    friend class Context;
    vk::SemaphoreType type_;
    vk::Semaphore sem_;
};

class SemaphoreSubmitInfo
{
public:
    SemaphoreSubmitInfo(Uid id, vk::PipelineStageFlags2 stages);
    SemaphoreSubmitInfo(Uid id, vk::PipelineStageFlags2 stages, uint64_t val);
    SemaphoreSubmitInfo(Uid id, uint64_t val);

private:
    friend class Context;
    Uid id_;
    vk::PipelineStageFlags2 stages_;
    uint64_t val_;
};

class Swapchain
{
public:
    Swapchain(SwapchainMeta const& meta, vk::SwapchainKHR swapchain,
              std::vector<Uid> const& image_ids, std::vector<Uid> const& image_view_ids);
    SwapchainMeta const& getMeta() const;
    operator vk::SwapchainKHR() const;
    operator bool() const;

private:
    friend class Context;
    SwapchainMeta meta_;
    vk::SwapchainKHR swapchain_;
    std::vector<Uid> image_ids_;
    std::vector<Uid> image_view_ids_;
};

class Queue
{
public:
    Queue(vk::Queue queue, uint32_t family_index);
    uint32_t getFamilyIndex() const;
    operator vk::Queue() const;
    operator bool() const;

private:
    friend class Context;
    vk::Queue queue_;
    uint32_t family_index_;
};

class DescriptorSetLayoutBinding
{
public:
    DescriptorSetLayoutBinding(vk::DescriptorType type, vk::ShaderStageFlags stages,
                               uint32_t count = 1);
    operator vk::DescriptorSetLayoutBinding() const;

private:
    friend class Context;
    vk::DescriptorSetLayoutBinding layout_;
};

class DescriptorSetLayout
{
public:
    DescriptorSetLayout(vk::DescriptorSetLayout layout,
                        std::vector<vk::DescriptorSetLayoutBinding> const& bindings);
    operator vk::DescriptorSetLayout() const;
    operator bool() const;

private:
    friend class Context;
    vk::DescriptorSetLayout layout_;
    std::vector<vk::DescriptorSetLayoutBinding> bindings_;
};

class DescriptorSet
{
public:
    DescriptorSet(vk::DescriptorSet set, vk::DescriptorPool pool,
                  std::vector<vk::DescriptorSetLayoutBinding> const& layout_bindings);
    operator vk::DescriptorSet() const;
    operator bool() const;

private:
    friend class Context;
    vk::DescriptorSet set_;
    vk::DescriptorPool pool_;
    std::vector<vk::DescriptorSetLayoutBinding> const* layout_bindings_;
};

class WriteDescriptorSetBinding
{
public:
    WriteDescriptorSetBinding(int id);
    WriteDescriptorSetBinding(int id_0, int id_1, vk::ImageLayout layout);

private:
    friend class Context;
    Uid id_0_;
    Uid id_1_;
    vk::ImageLayout layout_;
};

class CommandBuffer: public vk::CommandBuffer
{
public:
    CommandBuffer(vk::CommandBuffer buf, vk::CommandPool pool, Context& ctx);

    MyErrCode copyBufferToBuffer(Uid src_buf_id, Uid dst_buf_id, vk::DeviceSize src_offset = 0,
                                 vk::DeviceSize dst_offset = 0,
                                 vk::DeviceSize size = kRemainingSize);
    MyErrCode copyBufferToImage(Uid src_buf_id, Uid dst_img_id, vk::ImageLayout dst_img_layout,
                                BufferImageArea src_buf_area = {}, ImageSubArea dst_img_area = {});
    MyErrCode blitImage(Uid src_img_id, vk::ImageLayout src_img_layout, Uid dst_img_id,
                        vk::ImageLayout dst_img_layout, ImageSubArea src_img_area = {},
                        ImageSubArea dst_img_area = {});
    MyErrCode pipelineMemoryBarrier(MemoryBarrierMeta const& meta);
    MyErrCode pipelineImageBarrier(Uid image_id, ImageBarrierMeta const& meta,
                                   ImageSubRange range = {});
    MyErrCode pipelineImageBarrier(Uid image_id, vk::ImageLayout old_layout,
                                   vk::ImageLayout new_layout, ImageSubRange range = {});
    MyErrCode pushConstants(Uid pipeline_layout_id, vk::ShaderStageFlags stages, uint32_t offset,
                            uint32_t size, void const* data);
    MyErrCode bindComputePipeline(Uid pipeline_id);
    MyErrCode bindDescriptorSets(vk::PipelineBindPoint bind_point, Uid pipeline_layout_id,
                                 std::vector<Uid> const& set_ids);

    using vk::CommandBuffer::bindDescriptorSets;
    using vk::CommandBuffer::blitImage;
    using vk::CommandBuffer::copyBufferToImage;
    using vk::CommandBuffer::pushConstants;

private:
    static void completeImageSubLayers(Image const& image, ImageSubLayers& layers);
    static void completeImageSubRange(Image const& image, ImageSubRange& range);
    static void completeImageSubArea(Image const& image, ImageSubArea& area);
    static ImageBarrierMeta deduceImageBarrier(vk::ImageLayout old_layout,
                                               vk::ImageLayout new_layout);

private:
    friend class Context;
    vk::CommandPool pool_;
    Context* ctx_;
};

class Context
{
public:
    MyErrCode createInstance(char const* app_name, std::vector<char const*> const& extensions);
    MyErrCode createPhysicalDevice(DeviceRater const& device_rater = defaultDeviceRater);
    MyErrCode createDeviceAndQueues(std::vector<char const*> const& extensions,
                                    std::map<uint32_t, std::set<Uid>> const& queue_ids);
    MyErrCode createAllocator();
    MyErrCode createSurface(Uid id, vk::SurfaceKHR surface);
    MyErrCode createCommandPool(Uid id, Uid queue_id, vk::CommandPoolCreateFlags flags = {});
    MyErrCode createDescriptorPool(Uid id, uint32_t max_sets,
                                   std::map<vk::DescriptorType, uint32_t> const& sizes,
                                   vk::DescriptorPoolCreateFlags flags = {});
    MyErrCode createBuffer(Uid id, BufferMeta const& meta, vk::MemoryPropertyFlags properties,
                           VmaAllocationCreateFlags flags = 0);
    MyErrCode createImage(Uid id, ImageMeta const& meta, vk::MemoryPropertyFlags properties,
                          VmaAllocationCreateFlags flags = 0);
    MyErrCode createImageView(Uid id, Uid image_id, ImageViewMeta meta);
    MyErrCode createSampler(Uid id, SamplerMeta const& meta);
    MyErrCode createCommandBuffer(Uid id, Uid command_pool_id);
    MyErrCode createBinarySemaphore(Uid id);
    MyErrCode createTimelineSemaphore(Uid id, uint64_t init_val);
    MyErrCode createFence(Uid id, bool init_signaled = false);
    MyErrCode createShaderModule(Uid id, std::filesystem::path const& file_path);
    MyErrCode createDescriptorSetLayout(Uid id,
                                        std::vector<DescriptorSetLayoutBinding> const& bindings);
    MyErrCode createDescriptorSet(Uid id, Uid set_layout_id, Uid descriptor_pool_id);
    MyErrCode createPipelineLayout(Uid id, std::vector<Uid> const& set_layout_ids,
                                   std::vector<vk::PushConstantRange> const& push_ranges = {});
    MyErrCode createComputePipeline(Uid id, Uid pipeline_layout_id, Uid shader_id);
    MyErrCode createGraphicPipeline(Uid id, GraphicPipelineMeta const& meta);
    MyErrCode createSwapchain(Uid id, Uid surface_id, SwapchainMeta const& meta);
    MyErrCode createRenderPass(Uid id, RenderPassMeta const& meta);
    MyErrCode createFramebuffer(Uid id, Uid render_pass_id, std::vector<Uid> const& image_view_ids);

    vk::Instance& getInstance();
    vk::SurfaceKHR& getSurface(Uid id);
    vk::PhysicalDevice& getPhysicalDevice();
    vk::Device& getDevice();
    Queue& getQueue(Uid id);
    vk::CommandPool& getCommandPool(Uid id);
    vk::DescriptorPool& getDescriptorPool(Uid id);
    Buffer& getBuffer(Uid id);
    Image& getImage(Uid id);
    ImageView& getImageView(Uid id);
    vk::Sampler& getSampler(Uid id);
    CommandBuffer& getCommandBuffer(Uid id);
    Semaphore& getSemaphore(Uid id);
    vk::Fence& getFence(Uid id);
    vk::ShaderModule& getShaderModule(Uid id);
    DescriptorSetLayout& getDescriptorSetLayout(Uid id);
    vk::PipelineLayout& getPipelineLayout(Uid id);
    vk::Pipeline& getPipeline(Uid id);
    DescriptorSet& getDescriptorSet(Uid id);
    Swapchain& getSwapchain(Uid id);
    vk::RenderPass& getRenderPass(Uid id);
    vk::Framebuffer& getFramebuffer(Uid id);

    MyErrCode pickQueueFamily(QueuePicker const& queue_picker, uint32_t& family_index);
    MyErrCode copyBufferToBuffer(Uid queue_id, Uid command_pool_id, Uid src_buf_id, Uid dst_buf_id,
                                 vk::DeviceSize src_offset = 0, vk::DeviceSize dst_offset = 0,
                                 vk::DeviceSize size = kRemainingSize);
    MyErrCode copyBufferToImage(Uid queue_id, Uid command_pool_id, Uid src_buf_id, Uid dst_img_id,
                                vk::ImageLayout dst_img_layout, BufferImageArea src_buf_area = {},
                                ImageSubArea dst_img_area = {});
    MyErrCode copyHostToBuffer(Uid queue_id, Uid command_pool_id, void const* src_host,
                               Uid dst_buf_id);
    MyErrCode copyHostToImage(Uid queue_id, Uid command_pool_id, void const* src_host,
                              Uid dst_img_id, vk::ImageLayout dst_img_layout,
                              ImageSubArea dst_img_area = {});
    MyErrCode generateMipmaps(Uid queue_id, Uid command_pool_id, Uid image_id,
                              vk::ImageLayout image_layout);
    MyErrCode updateDescriptorSet(
        Uid set_id, std::map<uint32_t, WriteDescriptorSetBinding> const& write_bindings);
    MyErrCode waitDeviceIdle();
    MyErrCode waitQueueIdle(Uid queue_id);
    MyErrCode waitFences(std::vector<Uid> const& fence_ids, bool wait_all = true,
                         uint64_t timeout = UINT64_MAX);
    MyErrCode resetFences(std::vector<Uid> const& fence_ids);
    MyErrCode waitSemaphores(std::vector<SemaphoreSubmitInfo> const& wait_semaphores,
                             uint64_t timeout = UINT64_MAX);
    MyErrCode signalSemaphore(SemaphoreSubmitInfo const& signal_semaphore);
    MyErrCode acquireNextImage(Uid swapchain_id, uint32_t& image_index, bool& recreate_swapchain,
                               Uid signal_semaphore_id = Uid::kNull,
                               Uid signal_fence_id = Uid::kNull, uint64_t timeout = UINT64_MAX);
    MyErrCode recordCommand(Uid command_buffer_id, vk::CommandBufferUsageFlags usage,
                            CmdSubmitter const& submitter, bool reset_command_buffer = false);
    MyErrCode oneTimeSubmit(Uid queue_id, Uid command_pool_id, CmdSubmitter const& submitter);
    MyErrCode submit(Uid queue_id, Uid command_buffer_id,
                     std::vector<SemaphoreSubmitInfo> const& wait_semaphores = {},
                     std::vector<SemaphoreSubmitInfo> const& signal_semaphores = {},
                     Uid fence_id = Uid::kNull);
    MyErrCode present(Uid queue_id, Uid swapchain_id, uint32_t image_index,
                      bool& recreate_swapchain, std::vector<Uid> const& wait_semaphores = {});

    MyErrCode destroySurface(Uid id);
    MyErrCode destroyCommandPool(Uid id);
    MyErrCode destroyDescriptorPool(Uid id);
    MyErrCode destroyBuffer(Uid id);
    MyErrCode destroyImage(Uid id);
    MyErrCode destroyImageView(Uid id);
    MyErrCode destroySampler(Uid id);
    MyErrCode destroyCommandBuffer(Uid id);
    MyErrCode destroySemaphore(Uid id);
    MyErrCode destroyFence(Uid id);
    MyErrCode destroyShaderModule(Uid id);
    MyErrCode destroyDescriptorSetLayout(Uid id);
    MyErrCode destroyPipelineLayout(Uid id);
    MyErrCode destroyPipeline(Uid id);
    MyErrCode destroyDescriptorSet(Uid id);
    MyErrCode destroySwapchain(Uid id);
    MyErrCode destroyRenderPass(Uid id);
    MyErrCode destroyFramebuffer(Uid id);
    MyErrCode destroy();

    static int defaultDeviceRater(vk::PhysicalDevice const& dev);

private:
    static vk::DebugUtilsMessengerCreateInfoEXT getDebugMessengerInfo();
    static MyErrCode logDeviceInfo(vk::PhysicalDevice const& physical_device);

    template <typename T, typename = std::enable_if<vk::isVulkanHandleType<T>::value>>
    MyErrCode setDebugObjectId(T const& obj, Uid id);
    MyErrCode setDebugObjectId(Queue const& queue, Uid id);
    MyErrCode setDebugObjectId(Buffer const& buffer, Uid id);
    MyErrCode setDebugObjectId(Image const& image, Uid id);
    MyErrCode setDebugObjectId(ImageView const& image_view, Uid id);
    MyErrCode setDebugObjectId(Semaphore const& semaphore, Uid id);
    MyErrCode setDebugObjectId(DescriptorSetLayout const& descriptor_set_layout, Uid id);
    MyErrCode setDebugObjectId(DescriptorSet const& descriptor_set, Uid id);
    MyErrCode setDebugObjectId(Swapchain const& swapchain, Uid id);

    template <typename T>
    MyErrCode create(UidMap<std::remove_reference_t<T>>& map, Uid id, T&& val);
    template <typename T>
    T& get(UidMap<T>& map, Uid id);

    template <typename T>
    MyErrCode destroy(UidMap<T>& map);
    template <typename T>
    MyErrCode destroy(UidMap<T>& map, Uid id);
    MyErrCode destroy(Queue const& queue);
    MyErrCode destroy(vk::SurfaceKHR const& surface);
    MyErrCode destroy(vk::CommandPool const& command_pool);
    MyErrCode destroy(vk::DescriptorPool const& descriptor_pool);
    MyErrCode destroy(Buffer const& buffer);
    MyErrCode destroy(Image const& image);
    MyErrCode destroy(ImageView const& image_view);
    MyErrCode destroy(vk::Sampler const& sampler);
    MyErrCode destroy(CommandBuffer const& command_buffer);
    MyErrCode destroy(Semaphore const& semaphore);
    MyErrCode destroy(vk::Fence const& fence);
    MyErrCode destroy(vk::ShaderModule const& shader_module);
    MyErrCode destroy(vk::PipelineLayout const& pipeline_layout);
    MyErrCode destroy(vk::Pipeline const& pipeline);
    MyErrCode destroy(DescriptorSetLayout const& descriptor_set_layout);
    MyErrCode destroy(DescriptorSet const& descriptor_set);
    MyErrCode destroy(Swapchain const& swapchain);
    MyErrCode destroy(vk::RenderPass const& render_pass);
    MyErrCode destroy(vk::Framebuffer const& framebuffer);

private:
    vk::Instance instance_;
    vk::DebugUtilsMessengerEXT debug_messenger_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    VmaAllocator allocator_;

    UidMap<Queue> queues_;
    UidMap<vk::SurfaceKHR> surfaces_;
    UidMap<vk::CommandPool> command_pools_;
    UidMap<vk::DescriptorPool> descriptor_pools_;
    UidMap<Buffer> buffers_;
    UidMap<Image> images_;
    UidMap<ImageView> image_views_;
    UidMap<vk::Sampler> samplers_;
    UidMap<CommandBuffer> command_buffers_;
    UidMap<Semaphore> semaphores_;
    UidMap<vk::Fence> fences_;
    UidMap<vk::ShaderModule> shader_modules_;
    UidMap<vk::PipelineLayout> pipeline_layouts_;
    UidMap<vk::Pipeline> pipelines_;
    UidMap<DescriptorSetLayout> descriptor_set_layouts_;
    UidMap<DescriptorSet> descriptor_sets_;
    UidMap<Swapchain> swapchains_;
    UidMap<vk::RenderPass> render_passes_;
    UidMap<vk::Framebuffer> framebuffers_;
};

}  // namespace myvk
