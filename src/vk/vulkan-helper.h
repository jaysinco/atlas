#pragma once
#define VULKAN_HPP_NO_NODISCARD_WARNINGS
#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include <map>
#include <functional>

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

using DeviceRater =
    std::function<int(vk::PhysicalDeviceProperties const&, vk::PhysicalDeviceFeatures const&)>;

using QueuePicker = std::function<bool(uint32_t family_index, vk::QueueFamilyProperties const&)>;

using CmdSubmitter = std::function<MyErrCode(vk::CommandBuffer&)>;

class Uid
{
public:
    Uid(int id);
    bool operator<(Uid rhs) const;
    bool operator==(Uid rhs) const;
    bool operator!=(Uid rhs) const;
    std::string toStr() const;
    static Uid const kNull;
    static Uid temp();

private:
    int id_;
    static int temp_id;
};

class Allocation
{
public:
    Allocation();
    Allocation(VmaAllocation alloc, VmaAllocator allocator);
    vk::MemoryPropertyFlags getMemProp() const;
    VmaAllocationInfo getAllocInfo() const;
    bool canBeMapped() const;
    bool isMapped() const;
    void* map();
    void unmap();
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
    Buffer();
    Buffer(uint64_t size, vk::Buffer buf, VmaAllocation alloc, VmaAllocator allocator);
    uint64_t getSize() const;
    operator vk::Buffer() const;
    operator bool() const;

private:
    friend class Context;
    uint64_t size_;
    vk::Buffer buf_;
};

class Image: public Allocation

{
public:
    Image();
    Image(vk::Format format, vk::Extent2D extent, vk::ImageLayout layout, uint32_t mip_levels,
          vk::Image img, vk::ImageView img_view, VmaAllocation alloc, VmaAllocator allocator);
    uint64_t getSize() const;
    vk::ImageLayout getLayout() const;
    void setLayout(vk::ImageLayout layout);
    operator vk::Image() const;
    operator vk::ImageView() const;
    operator bool() const;

private:
    friend class Context;
    vk::Format format_;
    vk::Extent2D extent_;
    vk::ImageLayout layout_;
    uint32_t mip_levels_;
    vk::Image img_;
    vk::ImageView img_view_;
};

class CommandBuffer
{
public:
    CommandBuffer();
    CommandBuffer(vk::CommandBuffer buf, vk::CommandPool pool);
    operator vk::CommandBuffer() const;
    operator bool() const;

private:
    friend class Context;
    vk::CommandBuffer buf_;
    vk::CommandPool pool_;
};

class Semaphore
{
public:
    Semaphore();
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
    Swapchain();
    Swapchain(vk::Format format, vk::Extent2D extent, vk::SwapchainKHR swapchain,
              std::vector<vk::Image> const& images, std::vector<vk::ImageView> const& image_views);
    operator vk::SwapchainKHR() const;
    operator bool() const;

private:
    friend class Context;
    vk::Format format_;
    vk::Extent2D extent_;
    vk::SwapchainKHR swapchain_;
    std::vector<vk::Image> images_;
    std::vector<vk::ImageView> image_views_;
};

class Queue
{
public:
    Queue();
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

class DescriptorSet
{
public:
    DescriptorSet();
    DescriptorSet(vk::DescriptorSet set, vk::DescriptorPool pool);
    operator vk::DescriptorSet() const;
    operator bool() const;

private:
    friend class Context;
    vk::DescriptorSet set_;
    vk::DescriptorPool pool_;
};

class WriteDescriptorSet
{
public:
    WriteDescriptorSet(uint32_t binding, vk::DescriptorType type, Buffer& buffer);
    operator vk::WriteDescriptorSet() const;

private:
    friend class Context;
    vk::WriteDescriptorSet write_;
    std::vector<vk::DescriptorImageInfo> images_;
    std::vector<vk::DescriptorBufferInfo> buffers_;
};

class Context

{
public:
    MyErrCode createInstance(char const* app_name, std::vector<char const*> const& extensions);
    MyErrCode createPhysicalDevice(DeviceRater const& device_rater = defaultDeviceRater);
    MyErrCode createDeviceAndQueues(std::vector<char const*> const& extensions,
                                    std::map<Uid, QueuePicker> const& queue_pickers);
    MyErrCode createAllocator();
    MyErrCode createSurface(Uid id, vk::SurfaceKHR surface);
    MyErrCode createCommandPool(Uid id, Uid queue_id, vk::CommandPoolCreateFlags flags = {});
    MyErrCode createDescriptorPool(Uid id, uint32_t max_sets,
                                   std::map<vk::DescriptorType, uint32_t> const& size);
    MyErrCode createBuffer(Uid id, uint64_t size, vk::BufferUsageFlags usage,
                           vk::MemoryPropertyFlags properties, VmaAllocationCreateFlags flags = 0);
    MyErrCode createImage(Uid id, vk::Format format, vk::Extent2D extent, vk::ImageTiling tiling,
                          vk::ImageLayout initial_layout, uint32_t mip_levels,
                          vk::SampleCountFlagBits num_samples, vk::ImageAspectFlags aspect_mask,
                          vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties,
                          VmaAllocationCreateFlags flags = 0);
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
    MyErrCode createSwapchain(Uid id, Uid surface_id, vk::SurfaceFormatKHR surface_format,
                              vk::Extent2D extent, vk::PresentModeKHR mode,
                              vk::ImageUsageFlags usage);

    vk::Instance& getInstance();
    vk::SurfaceKHR& getSurface(Uid id);
    vk::PhysicalDevice& getPhysicalDevice();
    vk::Device& getDevice();
    Queue& getQueue(Uid id);
    vk::CommandPool& getCommandPool(Uid id);
    vk::DescriptorPool& getDescriptorPool(Uid id);
    Buffer& getBuffer(Uid id);
    Image& getImage(Uid id);
    CommandBuffer& getCommandBuffer(Uid id);
    Semaphore& getSemaphore(Uid id);
    vk::Fence& getFence(Uid id);
    vk::ShaderModule& getShaderModule(Uid id);
    vk::DescriptorSetLayout& getDescriptorSetLayout(Uid id);
    vk::PipelineLayout& getPipelineLayout(Uid id);
    vk::Pipeline& getPipeline(Uid id);
    DescriptorSet& getDescriptorSet(Uid id);
    Swapchain& getSwapchain(Uid id);

    MyErrCode copyBufferToBuffer(Uid queue_id, Uid command_pool_id, Uid dst_buf_id, Uid src_buf_id);
    MyErrCode copyBufferToImage(Uid queue_id, Uid command_pool_id, Uid dst_img_id, Uid src_buf_id);
    MyErrCode copyHostToBuffer(Uid queue_id, Uid command_pool_id, Uid dst_buf_id,
                               void const* src_host);
    MyErrCode copyHostToImage(Uid queue_id, Uid command_pool_id, Uid dst_img_id,
                              void const* src_host);
    MyErrCode transitionImageLayout(Uid queue_id, Uid command_pool_id, Uid image_id,
                                    vk::ImageLayout new_layout);
    MyErrCode updateDescriptorSet(Uid set_id, std::vector<WriteDescriptorSet> const& writes);
    MyErrCode waitQueueIdle(Uid queue_id);
    MyErrCode waitFences(std::vector<Uid> const& fence_ids, bool wait_all = true,
                         uint64_t timeout = UINT64_MAX);
    MyErrCode waitSemaphores(std::vector<SemaphoreSubmitInfo> const& wait_semaphores,
                             uint64_t timeout = UINT64_MAX);
    MyErrCode signalSemaphore(SemaphoreSubmitInfo const& signal_semaphore);
    MyErrCode recordCommand(Uid command_buffer_id, vk::CommandBufferUsageFlags usage,
                            CmdSubmitter const& submitter);
    MyErrCode oneTimeSubmit(Uid queue_id, Uid command_pool_id, CmdSubmitter const& submitter);
    MyErrCode submit(Uid queue_id, Uid command_buffer_id,
                     std::vector<SemaphoreSubmitInfo> const& wait_semaphores = {},
                     std::vector<SemaphoreSubmitInfo> const& signal_semaphores = {},
                     Uid fence_id = Uid::kNull);

    MyErrCode destroySurface(Uid id);
    MyErrCode destroyCommandPool(Uid id);
    MyErrCode destroyDescriptorPool(Uid id);
    MyErrCode destroyBuffer(Uid id);
    MyErrCode destroyImage(Uid id);
    MyErrCode destroyCommandBuffer(Uid id);
    MyErrCode destroySemaphore(Uid id);
    MyErrCode destroyFence(Uid id);
    MyErrCode destroyShaderModule(Uid id);
    MyErrCode destroyDescriptorSetLayout(Uid id);
    MyErrCode destroyPipelineLayout(Uid id);
    MyErrCode destroyPipeline(Uid id);
    MyErrCode destroyDescriptorSet(Uid id);
    MyErrCode destroySwapchain(Uid id);
    MyErrCode destroy();

    static int defaultDeviceRater(vk::PhysicalDeviceProperties const& prop,
                                  vk::PhysicalDeviceFeatures const& feat);

protected:
    template <typename T, typename = std::enable_if<vk::isVulkanHandleType<T>::value>>
    MyErrCode setDebugObjectId(T obj, Uid id);
    static vk::DebugUtilsMessengerCreateInfoEXT getDebugMessengerInfo();
    static MyErrCode logDeviceInfo(vk::PhysicalDevice const& physical_device);

private:
    vk::Instance instance_;
    vk::DebugUtilsMessengerEXT debug_messenger_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    VmaAllocator allocator_;

    std::map<Uid, Queue> queues_;
    std::map<Uid, vk::SurfaceKHR> surfaces_;
    std::map<Uid, vk::CommandPool> command_pools_;
    std::map<Uid, vk::DescriptorPool> descriptor_pools_;
    std::map<Uid, Buffer> buffers_;
    std::map<Uid, Image> images_;
    std::map<Uid, CommandBuffer> command_buffers_;
    std::map<Uid, Semaphore> semaphores_;
    std::map<Uid, vk::Fence> fences_;
    std::map<Uid, vk::ShaderModule> shader_modules_;
    std::map<Uid, vk::PipelineLayout> pipeline_layouts_;
    std::map<Uid, vk::Pipeline> pipelines_;
    std::map<Uid, vk::DescriptorSetLayout> descriptor_set_layouts_;
    std::map<Uid, DescriptorSet> descriptor_sets_;
    std::map<Uid, Swapchain> swapchains_;
};

}  // namespace myvk
