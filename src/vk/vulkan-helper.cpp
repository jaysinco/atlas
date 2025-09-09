#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#include "vulkan-helper.h"
#include <vulkan/vulkan_format_traits.hpp>
#include "toolkit/toolkit.h"
#include "toolkit/logging.h"
#include <set>

#define MYVK_API_VERSION VK_MAKE_API_VERSION(0, MYVK_API_VERSION_MAJOR, MYVK_API_VERSION_MINOR, 0)

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace myvk
{

Allocation::Allocation(VmaAllocation alloc, VmaAllocator allocator)
    : alloc_(alloc), allocator_(allocator)
{
}

VmaAllocationInfo Allocation::getAllocInfo() const
{
    VmaAllocationInfo alloc_info = {};
    vmaGetAllocationInfo(allocator_, alloc_, &alloc_info);
    return alloc_info;
}

vk::MemoryPropertyFlags Allocation::getMemProp() const
{
    VkMemoryPropertyFlags flags;
    vmaGetAllocationMemoryProperties(allocator_, alloc_, &flags);
    return static_cast<vk::MemoryPropertyFlags>(flags);
}

bool Allocation::canBeMapped() const
{
    return static_cast<bool>(getMemProp() & vk::MemoryPropertyFlagBits::eHostVisible);
}

bool Allocation::isMapped() const { return getMappedData() != nullptr; }

void* Allocation::map()
{
    void* data = nullptr;
    if (auto err = vmaMapMemory(allocator_, alloc_, &data); err != VK_SUCCESS) {
        MY_THROW("failed to map memory: {}", err);
    }
    return data;
}

void Allocation::unmap() { vmaUnmapMemory(allocator_, alloc_); }

void* Allocation::getMappedData() const { return getAllocInfo().pMappedData; }

void Allocation::invalid(vk::DeviceSize offset, vk::DeviceSize size)
{
    if (auto err = vmaInvalidateAllocation(allocator_, alloc_, offset, size); err != VK_SUCCESS) {
        MY_THROW("failed to invalidate memory: {}", err);
    }
}

void Allocation::flush(vk::DeviceSize offset, vk::DeviceSize size)
{
    if (auto err = vmaFlushAllocation(allocator_, alloc_, offset, size); err != VK_SUCCESS) {
        MY_THROW("failed to invalidate memory: {}", err);
    }
}

Buffer::Buffer(BufferMeta const& meta, vk::Buffer buf, VmaAllocation alloc, VmaAllocator allocator)
    : meta_(meta), buf_(buf), Allocation(alloc, allocator)
{
}

BufferMeta const& Buffer::getMeta() const { return meta_; }

Buffer::operator vk::Buffer() const { return buf_; }

Buffer::operator bool() const { return buf_; }

Image::Image(ImageMeta const& meta, vk::Image img, VmaAllocation alloc, VmaAllocator allocator)
    : meta_(meta), img_(img), Allocation(alloc, allocator)
{
}

ImageMeta const& Image::getMeta() const { return meta_; }

Image::operator vk::Image() const { return img_; }

Image::operator bool() const { return img_; }

ImageView::ImageView(ImageViewMeta const& meta, ImageMeta const& img_meta, vk::ImageView view)
    : meta_(meta), img_meta_(&img_meta), view_(view)
{
}

ImageViewMeta const& ImageView::getMeta() const { return meta_; }

ImageView::operator vk::ImageView() const { return view_; }

ImageView::operator bool() const { return view_; }

Semaphore::Semaphore(vk::SemaphoreType type, vk::Semaphore sem): type_(type), sem_(sem) {}

bool Semaphore::isTimeline() const { return type_ == vk::SemaphoreType::eTimeline; }

Semaphore::operator vk::Semaphore() const { return sem_; }

Semaphore::operator bool() const { return sem_; }

SemaphoreSubmitInfo::SemaphoreSubmitInfo(Uid id, vk::PipelineStageFlags2 stages)
    : id_(id), stages_(stages), val_(0)
{
}

SemaphoreSubmitInfo::SemaphoreSubmitInfo(Uid id, vk::PipelineStageFlags2 stages, uint64_t val)
    : id_(id), stages_(stages), val_(val)
{
}

SemaphoreSubmitInfo::SemaphoreSubmitInfo(Uid id, uint64_t val)
    : id_(id), stages_(vk::PipelineStageFlagBits2::eNone), val_(val)
{
}

Swapchain::Swapchain(SwapchainMeta const& meta, vk::SwapchainKHR swapchain,
                     std::vector<Uid> const& image_ids, std::vector<Uid> const& image_view_ids)
    : meta_(meta), swapchain_(swapchain), image_ids_(image_ids), image_view_ids_(image_view_ids)
{
    meta_.image_count = image_ids.size();
}

SwapchainMeta const& Swapchain::getMeta() const { return meta_; }

Swapchain::operator vk::SwapchainKHR() const { return swapchain_; }

Swapchain::operator bool() const { return swapchain_; }

Queue::Queue(vk::Queue queue, uint32_t family_index): queue_(queue), family_index_(family_index) {}

uint32_t Queue::getFamilyIndex() const { return family_index_; }

Queue::operator vk::Queue() const { return queue_; }

Queue::operator bool() const { return queue_; }

DescriptorSetLayout::DescriptorSetLayout(
    vk::DescriptorSetLayout layout, std::vector<vk::DescriptorSetLayoutBinding> const& bindings)
    : layout_(layout), bindings_(bindings)
{
}

DescriptorSetLayout::operator vk::DescriptorSetLayout() const { return layout_; }

DescriptorSetLayout::operator bool() const { return layout_; }

DescriptorSet::DescriptorSet(vk::DescriptorSet set, vk::DescriptorPool pool,
                             std::vector<vk::DescriptorSetLayoutBinding> const& layout_bindings)
    : set_(set), pool_(pool), layout_bindings_(&layout_bindings)
{
}

DescriptorSet::operator vk::DescriptorSet() const { return set_; }

DescriptorSet::operator bool() const { return set_; }

DescriptorSetLayoutBinding::DescriptorSetLayoutBinding(vk::DescriptorType type,
                                                       vk::ShaderStageFlags stages, uint32_t count)
    : layout_(0, type, count, stages)
{
}

DescriptorSetLayoutBinding::operator vk::DescriptorSetLayoutBinding() const { return layout_; }

WriteDescriptorSetBinding::WriteDescriptorSetBinding(int id): id_0_(id) {}

WriteDescriptorSetBinding::WriteDescriptorSetBinding(int id_0, int id_1, vk::ImageLayout layout)
    : id_0_(id_0), id_1_(id_1), layout_(layout)
{
}

static VkBool32 debugMessengerUserCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type,
    vk::DebugUtilsMessengerCallbackDataEXT const* callback_data, void* user_data)
{
    switch (severity) {
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
            TLOG("[vkdbg] {}", callback_data->pMessage);
            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
            TLOG("[vkdbg] {}", callback_data->pMessage);
            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
            ILOG("[vkdbg] {}", callback_data->pMessage);
            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
            ELOG("[vkdbg] {}", callback_data->pMessage);
            break;
        default:
            break;
    }
    return VK_FALSE;
}

vk::DebugUtilsMessengerCreateInfoEXT Context::getDebugMessengerInfo()
{
    vk::DebugUtilsMessengerCreateInfoEXT create_info;
    create_info.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                                   vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                                   vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                   vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
    create_info.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                               vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                               vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);
    create_info.pfnUserCallback = &debugMessengerUserCallback;
    return create_info;
}

MyErrCode Context::createInstance(char const* app_name, std::vector<char const*> const& extensions)
{
    VULKAN_HPP_DEFAULT_DISPATCHER.init();
    vk::ApplicationInfo app_info{app_name, VK_MAKE_VERSION(0, 1, 0), "No Engine",
                                 VK_MAKE_VERSION(0, 1, 0), MYVK_API_VERSION};
    auto debug_info = getDebugMessengerInfo();
    instance_ = CHECK_VKHPP_VAL(
        vk::createInstance({vk::InstanceCreateFlags(), &app_info, {}, extensions, &debug_info}));

    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);

    debug_messenger_ = CHECK_VKHPP_VAL(instance_.createDebugUtilsMessengerEXT(debug_info));

    return MyErrCode::kOk;
}

vk::Instance& Context::getInstance() { return instance_; }

template <typename T, typename>
MyErrCode Context::setDebugObjectId(T const& obj, Uid id)
{
    std::string name = FSTR("{}", id);
    CHECK_VKHPP_RET(device_.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT(
        T::objectType, reinterpret_cast<uint64_t>(static_cast<typename T::CType>(obj)),
        name.c_str())));
    return MyErrCode::kOk;
}

MyErrCode Context::setDebugObjectId(Queue const& queue, Uid id)
{
    return setDebugObjectId(queue.queue_, id);
}

MyErrCode Context::setDebugObjectId(Buffer const& buffer, Uid id)
{
    return setDebugObjectId(buffer.buf_, id);
}

MyErrCode Context::setDebugObjectId(Image const& image, Uid id)
{
    return setDebugObjectId(image.img_, id);
}

MyErrCode Context::setDebugObjectId(ImageView const& image_view, Uid id)
{
    return setDebugObjectId(image_view.view_, id);
}

MyErrCode Context::setDebugObjectId(Semaphore const& semaphore, Uid id)
{
    return setDebugObjectId(semaphore.sem_, id);
}

MyErrCode Context::setDebugObjectId(DescriptorSetLayout const& descriptor_set_layout, Uid id)
{
    return setDebugObjectId(descriptor_set_layout.layout_, id);
}

MyErrCode Context::setDebugObjectId(DescriptorSet const& descriptor_set, Uid id)
{
    return setDebugObjectId(descriptor_set.set_, id);
}

MyErrCode Context::setDebugObjectId(Swapchain const& swapchain, Uid id)
{
    return setDebugObjectId(swapchain.swapchain_, id);
}

template <typename T>
MyErrCode Context::create(UidMap<std::remove_reference_t<T>>& map, Uid id, T&& val)
{
    CHECK_ERR_RET(setDebugObjectId(val, id));
    map.try_emplace_l(
        id,
        [&](auto& old) {
            destroy(old.second);
            old.second = std::move(val);
        },
        std::move(val));
    return MyErrCode::kOk;
}

template <typename T>
T& Context::get(UidMap<T>& map, Uid id)
{
    if (auto it = map.find(id); it == map.end()) {
        MY_THROW("id not exist: {}", id);
    } else {
        return it->second;
    }
}

template <typename T>
MyErrCode Context::destroy(UidMap<T>& map)
{
    map.for_each([&](auto const& v) { destroy(v.second); });
    map.clear();
    return MyErrCode::kOk;
}

template <typename T>
MyErrCode Context::destroy(UidMap<T>& map, Uid id)
{
    bool erased = surfaces_.erase_if(id, [&](auto& old) {
        destroy(old.second);
        return true;
    });
    if (!erased) {
        ELOG("id not exist: {}", id);
        return MyErrCode::kFailed;
    }
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(Queue const& queue) { return MyErrCode::kOk; }

MyErrCode Context::destroy(vk::SurfaceKHR const& surface)
{
    instance_.destroy(surface);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(vk::CommandPool const& command_pool)
{
    device_.destroy(command_pool);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(vk::DescriptorPool const& descriptor_pool)
{
    device_.destroy(descriptor_pool);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(Buffer const& buffer)
{
    vmaDestroyBuffer(buffer.allocator_, buffer.buf_, buffer.alloc_);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(Image const& image)
{
    if (image.alloc_) {
        vmaDestroyImage(image.allocator_, image.img_, image.alloc_);
    }
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(ImageView const& image_view)
{
    device_.destroy(image_view);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(vk::Sampler const& sampler)
{
    device_.destroy(sampler);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(CommandBuffer const& command_buffer)
{
    device_.freeCommandBuffers(command_buffer.pool_, command_buffer);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(Semaphore const& semaphore)
{
    device_.destroy(semaphore);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(vk::Fence const& fence)
{
    device_.destroy(fence);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(vk::ShaderModule const& shader_module)
{
    device_.destroy(shader_module);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(vk::PipelineLayout const& pipeline_layout)
{
    device_.destroy(pipeline_layout);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(vk::Pipeline const& pipeline)
{
    device_.destroy(pipeline);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(DescriptorSetLayout const& descriptor_set_layout)
{
    device_.destroy(descriptor_set_layout);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(DescriptorSet const& descriptor_set)
{
    device_.freeDescriptorSets(descriptor_set.pool_, {descriptor_set.set_});
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(Swapchain const& swapchain)
{
    device_.destroy(swapchain);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(vk::RenderPass const& render_pass)
{
    device_.destroy(render_pass);
    return MyErrCode::kOk;
}

MyErrCode Context::destroy(vk::Framebuffer const& framebuffer)
{
    device_.destroy(framebuffer);
    return MyErrCode::kOk;
}

MyErrCode Context::createSurface(Uid id, vk::SurfaceKHR surface)
{
    return create(surfaces_, id, surface);
}

vk::SurfaceKHR& Context::getSurface(Uid id) { return get(surfaces_, id); }

MyErrCode Context::destroySurface(Uid id) { return destroy(surfaces_, id); }

MyErrCode Context::logDeviceInfo(vk::PhysicalDevice const& physical_device)
{
    auto device_props = physical_device.getProperties();
    auto device_limits = device_props.limits;
    auto memory_props = physical_device.getMemoryProperties();
    auto queue_families = physical_device.getQueueFamilyProperties();
    auto device_features = physical_device.getFeatures();
    auto extensions = CHECK_VKHPP_VAL(physical_device.enumerateDeviceExtensionProperties());

    // basic device
    DLOG("====== Vulkan Device Information ======");
    DLOG("Device Name: {}", device_props.deviceName);
    DLOG("Device ID: 0x{:x}", device_props.deviceID);
    DLOG("Vendor ID: 0x{:x}", device_props.vendorID);
    DLOG("Device Type: {}", device_props.deviceType);
    DLOG("Driver Version: {}.{}.{}", VK_VERSION_MAJOR(device_props.driverVersion),
         VK_VERSION_MINOR(device_props.driverVersion),
         VK_VERSION_PATCH(device_props.driverVersion));
    DLOG("Vulkan API Device Version: {}.{}.{}", VK_VERSION_MAJOR(device_props.apiVersion),
         VK_VERSION_MINOR(device_props.apiVersion), VK_VERSION_PATCH(device_props.apiVersion));
    DLOG("Vulkan API App Version: {}.{}.{}", VK_VERSION_MAJOR(MYVK_API_VERSION),
         VK_VERSION_MINOR(MYVK_API_VERSION), VK_VERSION_PATCH(MYVK_API_VERSION));

    // pipeline
    DLOG("====== Pipeline Limits ======");
    DLOG("Max Compute Shared Memory Size: {} KB", device_limits.maxComputeSharedMemorySize / 1024);
    DLOG("Max Bound Descriptor Sets: {}", device_limits.maxBoundDescriptorSets);
    DLOG("Max Push Constants Size: {} bytes", device_limits.maxPushConstantsSize);
    DLOG("Max Uniform Buffer Range: {} bytes", device_limits.maxUniformBufferRange);
    DLOG("Max Storage Buffer Range: {} bytes", device_limits.maxStorageBufferRange);
    DLOG("Max Per Stage Descriptor Samplers: {}", device_limits.maxPerStageDescriptorSamplers);
    DLOG("Max Per Stage Descriptor Uniform Buffers: {}",
         device_limits.maxPerStageDescriptorUniformBuffers);
    DLOG("Max Per Stage Descriptor Storage Buffers: {}",
         device_limits.maxPerStageDescriptorStorageBuffers);

    // memory
    DLOG("====== Memory Properties ======");
    DLOG("Memory Heaps: {}", memory_props.memoryHeapCount);
    for (uint32_t i = 0; i < memory_props.memoryHeapCount; ++i) {
        auto const& heap = memory_props.memoryHeaps[i];
        DLOG("  Heap {}: Size = {} MB, Flags = {}", i, heap.size / (1024 * 1024), heap.flags);
    }

    DLOG("Memory Types:");
    for (uint32_t i = 0; i < memory_props.memoryTypeCount; ++i) {
        auto const& type = memory_props.memoryTypes[i];
        DLOG("  Type {}: Heap Index = {}, Flags = {}", i, type.heapIndex, type.propertyFlags);
    }

    // queue
    DLOG("====== Queue Families ======");
    for (uint32_t i = 0; i < queue_families.size(); ++i) {
        auto const& queue = queue_families[i];
        DLOG("Queue Family {}:", i);
        DLOG("  Queue Count: {}", queue.queueCount);
        DLOG("  Queue Flags: {}", queue.queueFlags);
        DLOG("  Timestamp Valid Bits: {}", queue.timestampValidBits);
        DLOG("  Min Image Transfer Granularity: {}x{}x{}", queue.minImageTransferGranularity.width,
             queue.minImageTransferGranularity.height, queue.minImageTransferGranularity.depth);
    }

    DLOG("============================");
    return MyErrCode::kOk;
}

int Context::defaultDeviceRater(vk::PhysicalDeviceProperties const& prop,
                                vk::PhysicalDeviceFeatures const& feat)
{
    int score = 0;
    switch (prop.deviceType) {
        case vk::PhysicalDeviceType::eOther:
            score += 1;
            break;
        case vk::PhysicalDeviceType::eIntegratedGpu:
            score += 4;
            break;
        case vk::PhysicalDeviceType::eDiscreteGpu:
            score += 5;
            break;
        case vk::PhysicalDeviceType::eVirtualGpu:
            score += 3;
            break;
        case vk::PhysicalDeviceType::eCpu:
            score += 2;
            break;
        default:
            break;
    }
    return score;
}

MyErrCode Context::createPhysicalDevice(DeviceRater const& device_rater)
{
    auto physical_devices = CHECK_VKHPP_VAL(instance_.enumeratePhysicalDevices());
    int best_score = 0;
    for (auto& d: physical_devices) {
        int score = device_rater(d.getProperties(), d.getFeatures());
        if (score > best_score) {
            physical_device_ = d;
            best_score = score;
        }
    }
    if (best_score <= 0) {
        ELOG("no proper device found");
        return MyErrCode::kFailed;
    }
    CHECK_ERR_RET(logDeviceInfo(physical_device_));
    return MyErrCode::kOk;
}

vk::PhysicalDevice& Context::getPhysicalDevice() { return physical_device_; }

MyErrCode Context::pickQueueFamily(QueuePicker const& queue_picker, uint32_t& family_index)
{
    auto family_props = physical_device_.getQueueFamilyProperties();
    for (uint32_t i = 0; i < family_props.size(); ++i) {
        if (queue_picker(i, family_props[i])) {
            family_index = i;
            return MyErrCode::kOk;
        }
    }
    ELOG("no proper queue family found");
    return MyErrCode::kFailed;
}

MyErrCode Context::createDeviceAndQueues(std::vector<char const*> const& extensions,
                                         std::map<uint32_t, std::set<Uid>> const& queue_ids)
{
    std::vector<vk::DeviceQueueCreateInfo> queue_infos;
    float const queue_priority = 1.0f;
    for (auto& [family, ids]: queue_ids) {
        queue_infos.emplace_back(vk::DeviceQueueCreateFlags(), family, ids.size(), &queue_priority);
    }

    vk::PhysicalDeviceVulkan11Features feat_11;
    vk::PhysicalDeviceVulkan12Features feat_12;
    vk::PhysicalDeviceVulkan13Features feat_13;

    feat_12.timelineSemaphore = true;
    feat_13.synchronization2 = true;

    vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceVulkan11Features,
                       vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features>
        c = {vk::DeviceCreateInfo{vk::DeviceCreateFlags(), queue_infos, {}, extensions}, feat_11,
             feat_12, feat_13};

    device_ = CHECK_VKHPP_VAL(physical_device_.createDevice(c.get<vk::DeviceCreateInfo>()));

    VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);

    auto family_props = physical_device_.getQueueFamilyProperties();
    for (auto& [family, ids]: queue_ids) {
        int i = 0;
        for (auto id: ids) {
            auto queue = device_.getQueue(family, i++);
            CHECK_ERR_RET(create(queues_, id, Queue{queue, family}));
            DLOG("create queue {}: family {}, {}", id, family, family_props[family].queueFlags);
        }
    }

    return MyErrCode::kOk;
}

vk::Device& Context::getDevice() { return device_; }

Queue& Context::getQueue(Uid id) { return get(queues_, id); }

MyErrCode Context::createCommandPool(Uid id, Uid queue_id, vk::CommandPoolCreateFlags flags)
{
    auto pool =
        CHECK_VKHPP_VAL(device_.createCommandPool({flags, getQueue(queue_id).getFamilyIndex()}));
    return create(command_pools_, id, pool);
}

vk::CommandPool& Context::getCommandPool(Uid id) { return get(command_pools_, id); }

MyErrCode Context::destroyCommandPool(Uid id) { return destroy(command_pools_, id); }

MyErrCode Context::createDescriptorPool(Uid id, uint32_t max_sets,
                                        std::map<vk::DescriptorType, uint32_t> const& sizes,
                                        vk::DescriptorPoolCreateFlags flags)
{
    std::vector<vk::DescriptorPoolSize> pool_size;
    for (auto [type, count]: sizes) {
        pool_size.emplace_back(type, count);
    }
    auto pool = CHECK_VKHPP_VAL(device_.createDescriptorPool({flags, max_sets, pool_size}));
    return create(descriptor_pools_, id, pool);
}

vk::DescriptorPool& Context::getDescriptorPool(Uid id) { return get(descriptor_pools_, id); }

MyErrCode Context::destroyDescriptorPool(Uid id) { return destroy(descriptor_pools_, id); }

MyErrCode Context::createAllocator()
{
    VmaAllocatorCreateInfo create_info = {};
    create_info.vulkanApiVersion = MYVK_API_VERSION;
    create_info.physicalDevice = physical_device_;
    create_info.device = device_;
    create_info.instance = instance_;
    CHECK_VK_RET(vmaCreateAllocator(&create_info, &allocator_));
    return MyErrCode::kOk;
}

MyErrCode Context::createShaderModule(Uid id, std::filesystem::path const& file_path)
{
    std::vector<uint8_t> code;
    CHECK_ERR_RET(toolkit::readBinaryFile(file_path, code));
    vk::ShaderModuleCreateInfo create_info(vk::ShaderModuleCreateFlags(), code.size(),
                                           reinterpret_cast<uint32_t const*>(code.data()));
    auto shader = CHECK_VKHPP_VAL(device_.createShaderModule(create_info));
    return create(shader_modules_, id, shader);
}

vk::ShaderModule& Context::getShaderModule(Uid id) { return get(shader_modules_, id); }

MyErrCode Context::destroyShaderModule(Uid id) { return destroy(shader_modules_, id); }

MyErrCode Context::createBuffer(Uid id, BufferMeta const& meta, vk::MemoryPropertyFlags properties,
                                VmaAllocationCreateFlags flags)
{
    vk::BufferCreateInfo buffer_info(vk::BufferCreateFlags{}, meta.size, meta.usages,
                                     vk::SharingMode::eExclusive);

    VmaAllocationCreateInfo creation_info = {};
    creation_info.flags = flags;
    creation_info.requiredFlags = static_cast<VkMemoryPropertyFlags>(properties);

    VkBuffer buf;
    VmaAllocation alloc;
    CHECK_VK_RET(vmaCreateBuffer(allocator_, buffer_info, &creation_info, &buf, &alloc, nullptr));

    CHECK_ERR_RET(create(buffers_, id, Buffer{meta, buf, alloc, allocator_}));
    DLOG("create buffer {}: {} bytes, {}", id, meta.size, buffers_.at(id).getMemProp());
    return MyErrCode::kOk;
}

Buffer& Context::getBuffer(Uid id) { return get(buffers_, id); }

MyErrCode Context::destroyBuffer(Uid id) { return destroy(buffers_, id); }

MyErrCode Context::createImage(Uid id, ImageMeta const& meta, vk::MemoryPropertyFlags properties,
                               VmaAllocationCreateFlags flags)
{
    vk::ImageCreateInfo image_info(vk::ImageCreateFlags(), meta.type, meta.format, meta.extent,
                                   meta.mip_levels, meta.layers, meta.samples, meta.tiling,
                                   meta.usages, vk::SharingMode::eExclusive, {}, meta.init_layout);

    VmaAllocationCreateInfo vma_info = {};
    vma_info.flags = flags;
    vma_info.requiredFlags = static_cast<VkMemoryPropertyFlags>(properties);

    VkImage img;
    VmaAllocation alloc;
    CHECK_VK_RET(vmaCreateImage(allocator_, image_info, &vma_info, &img, &alloc, nullptr));

    return create(images_, id, Image{meta, img, alloc, allocator_});
}

Image& Context::getImage(Uid id) { return get(images_, id); }

MyErrCode Context::destroyImage(Uid id) { return destroy(images_, id); }

MyErrCode Context::createImageView(Uid id, Uid image_id, ImageViewMeta const& meta)
{
    auto& image = getImage(image_id);
    vk::ImageViewCreateInfo view_info({}, image.img_, meta.type, image.getMeta().format,
                                      meta.swizzle,
                                      {image.getMeta().aspects, meta.base_level, meta.num_levels,
                                       meta.base_layer, meta.num_layers});
    vk::ImageView img_view = CHECK_VKHPP_VAL(device_.createImageView(view_info));
    return create(image_views_, id, ImageView{meta, image.getMeta(), img_view});
}

ImageView& Context::getImageView(Uid id) { return get(image_views_, id); }

MyErrCode Context::destroyImageView(Uid id) { return destroy(image_views_, id); }

MyErrCode Context::createSampler(Uid id, SamplerMeta const& meta)
{
    auto device_props = physical_device_.getProperties();
    auto sampler = CHECK_VKHPP_VAL(device_.createSampler({{},
                                                          meta.filter,
                                                          meta.filter,
                                                          vk::SamplerMipmapMode::eLinear,
                                                          meta.address_mode,
                                                          meta.address_mode,
                                                          meta.address_mode,
                                                          0.0f,
                                                          true,
                                                          device_props.limits.maxSamplerAnisotropy,
                                                          false,
                                                          vk::CompareOp::eNever,
                                                          meta.min_lod,
                                                          meta.max_lod,
                                                          meta.border_color,
                                                          false}));
    return create(samplers_, id, sampler);
}

vk::Sampler& Context::getSampler(Uid id) { return get(samplers_, id); }

MyErrCode Context::destroySampler(Uid id) { return destroy(samplers_, id); }

MyErrCode Context::createCommandBuffer(Uid id, Uid command_pool_id)
{
    auto& pool = getCommandPool(command_pool_id);
    auto bufs = CHECK_VKHPP_VAL(
        device_.allocateCommandBuffers({pool, vk::CommandBufferLevel::ePrimary, 1}));
    return create(command_buffers_, id, CommandBuffer{bufs.front(), pool, *this});
}

CommandBuffer& Context::getCommandBuffer(Uid id) { return get(command_buffers_, id); }

MyErrCode Context::destroyCommandBuffer(Uid id) { return destroy(command_buffers_, id); }

MyErrCode Context::createBinarySemaphore(Uid id)
{
    auto sem = CHECK_VKHPP_VAL(device_.createSemaphore({}));
    return create(semaphores_, id, Semaphore{vk::SemaphoreType::eBinary, sem});
}

MyErrCode Context::createTimelineSemaphore(Uid id, uint64_t init_val)
{
    vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo> c{
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{vk::SemaphoreType::eTimeline, init_val},
    };
    auto sem = CHECK_VKHPP_VAL(device_.createSemaphore(c.get<vk::SemaphoreCreateInfo>()));
    return create(semaphores_, id, Semaphore{vk::SemaphoreType::eTimeline, sem});
}

Semaphore& Context::getSemaphore(Uid id) { return get(semaphores_, id); }

MyErrCode Context::destroySemaphore(Uid id) { return destroy(semaphores_, id); }

MyErrCode Context::createFence(Uid id, bool init_signaled)
{
    auto fence = CHECK_VKHPP_VAL(device_.createFence(
        {init_signaled ? vk::FenceCreateFlagBits::eSignaled : vk::FenceCreateFlags{}}));
    return create(fences_, id, fence);
}

vk::Fence& Context::getFence(Uid id) { return get(fences_, id); }

MyErrCode Context::destroyFence(Uid id) { return destroy(fences_, id); }

MyErrCode Context::createPipelineLayout(Uid id, std::vector<Uid> const& set_layout_ids,
                                        std::vector<vk::PushConstantRange> const& push_ranges)
{
    std::vector<vk::DescriptorSetLayout> set_layouts;
    for (auto id: set_layout_ids) {
        set_layouts.push_back(getDescriptorSetLayout(id));
    }
    auto layout = CHECK_VKHPP_VAL(device_.createPipelineLayout({{}, set_layouts, push_ranges}));
    return create(pipeline_layouts_, id, layout);
}

vk::PipelineLayout& Context::getPipelineLayout(Uid id) { return get(pipeline_layouts_, id); }

MyErrCode Context::destroyPipelineLayout(Uid id) { return destroy(pipeline_layouts_, id); }

MyErrCode Context::createComputePipeline(Uid id, Uid pipeline_layout_id, Uid shader_id)
{
    vk::PipelineShaderStageCreateInfo shader_info(vk::PipelineShaderStageCreateFlags(),
                                                  vk::ShaderStageFlagBits::eCompute,
                                                  getShaderModule(shader_id), "main");
    vk::ComputePipelineCreateInfo create_info(vk::PipelineCreateFlags(), shader_info,
                                              getPipelineLayout(pipeline_layout_id));
    auto pipeline = CHECK_VKHPP_VAL(device_.createComputePipeline({}, create_info));
    return create(pipelines_, id, pipeline);
}

vk::Pipeline& Context::getPipeline(Uid id) { return get(pipelines_, id); }

MyErrCode Context::destroyPipeline(Uid id) { return destroy(pipelines_, id); }

MyErrCode Context::createDescriptorSetLayout(
    Uid id, std::vector<DescriptorSetLayoutBinding> const& bindings)
{
    std::vector<vk::DescriptorSetLayoutBinding> layout_bindings;
    for (int i = 0; i < bindings.size(); ++i) {
        vk::DescriptorSetLayoutBinding b = bindings[i];
        b.setBinding(i);
        layout_bindings.push_back(b);
    }
    auto layout = CHECK_VKHPP_VAL(
        device_.createDescriptorSetLayout({vk::DescriptorSetLayoutCreateFlags(), layout_bindings}));
    return create(descriptor_set_layouts_, id, DescriptorSetLayout{layout, layout_bindings});
}

DescriptorSetLayout& Context::getDescriptorSetLayout(Uid id)
{
    return get(descriptor_set_layouts_, id);
}

MyErrCode Context::destroyDescriptorSetLayout(Uid id)
{
    return destroy(descriptor_set_layouts_, id);
}

MyErrCode Context::createDescriptorSet(Uid id, Uid set_layout_id, Uid descriptor_pool_id)
{
    auto& pool = getDescriptorPool(descriptor_pool_id);
    DescriptorSetLayout& layout = getDescriptorSetLayout(set_layout_id);
    auto sets = CHECK_VKHPP_VAL(device_.allocateDescriptorSets({pool, layout.layout_}));
    return create(descriptor_sets_, id, DescriptorSet{sets.front(), pool, layout.bindings_});
}

DescriptorSet& Context::getDescriptorSet(Uid id) { return get(descriptor_sets_, id); }

MyErrCode Context::destroyDescriptorSet(Uid id) { return destroy(descriptor_sets_, id); }

CommandBuffer::CommandBuffer(vk::CommandBuffer buf, vk::CommandPool pool, Context& ctx)
    : vk::CommandBuffer(buf), pool_(pool), ctx_(&ctx)
{
}

MyErrCode CommandBuffer::copyBufferToBuffer(Uid src_buf_id, Uid dst_buf_id,
                                            vk::DeviceSize src_offset, vk::DeviceSize dst_offset,
                                            vk::DeviceSize size)
{
    Buffer& dst_buf = ctx_->getBuffer(dst_buf_id);
    Buffer& src_buf = ctx_->getBuffer(src_buf_id);
    if (src_offset + size > src_buf.getMeta().size || dst_offset + size > dst_buf.getMeta().size) {
        ELOG("invalid buffer size for copy");
        return MyErrCode::kFailed;
    }
    vk::BufferCopy region(src_offset, dst_offset,
                          size == vk::WholeSize ? dst_buf.getMeta().size : size);
    copyBuffer(src_buf, dst_buf, region);
    return MyErrCode::kOk;
}

MyErrCode CommandBuffer::copyBufferToImage(Uid src_buf_id, Uid dst_img_id,
                                           vk::ImageLayout dst_img_layout, uint32_t dst_img_layer)
{
    Image& dst_img = ctx_->getImage(dst_img_id);
    Buffer& src_buf = ctx_->getBuffer(src_buf_id);
    vk::BufferImageCopy region(
        0, 0, 0, vk::ImageSubresourceLayers{dst_img.getMeta().aspects, 0, dst_img_layer, 1},
        {0, 0, 0}, dst_img.getMeta().extent);
    copyBufferToImage(src_buf, dst_img, dst_img_layout, region);
    return MyErrCode::kOk;
}

MyErrCode CommandBuffer::pipelineMemoryBarrier(vk::PipelineStageFlags2 src_stage,
                                               vk::AccessFlags2 src_access,
                                               vk::PipelineStageFlags2 dst_stage,
                                               vk::AccessFlags2 dst_access)
{
    vk::MemoryBarrier2 barrier(src_stage, src_access, dst_stage, dst_access);
    pipelineBarrier2(vk::DependencyInfo{{}, barrier});
    return MyErrCode::kOk;
}

MyErrCode CommandBuffer::pipelineImageBarrier(Uid image_id, vk::ImageLayout old_layout,
                                              vk::ImageLayout new_layout,
                                              vk::PipelineStageFlags2 src_stage,
                                              vk::AccessFlags2 src_access,
                                              vk::PipelineStageFlags2 dst_stage,
                                              vk::AccessFlags2 dst_access)
{
    Image image = ctx_->getImage(image_id);
    vk::ImageMemoryBarrier2 barrier(
        src_stage, src_access, dst_stage, dst_access, old_layout, new_layout,
        vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, image,
        {image.getMeta().aspects, 0, image.getMeta().mip_levels, 0, image.getMeta().layers});
    pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, barrier});
    return MyErrCode::kOk;
}

MyErrCode CommandBuffer::pushConstants(Uid pipeline_layout_id, vk::ShaderStageFlags stages,
                                       uint32_t offset, uint32_t size, void const* data)
{
    pushConstants(ctx_->getPipelineLayout(pipeline_layout_id), stages, offset, size, data);
    return MyErrCode::kOk;
}

MyErrCode CommandBuffer::bindComputePipeline(Uid pipeline_id)
{
    bindPipeline(vk::PipelineBindPoint::eCompute, ctx_->getPipeline(pipeline_id));
    return MyErrCode::kOk;
}

MyErrCode CommandBuffer::bindDescriptorSets(vk::PipelineBindPoint bind_point,
                                            Uid pipeline_layout_id, std::vector<Uid> const& set_ids)
{
    std::vector<vk::DescriptorSet> sets;
    for (auto id: set_ids) {
        sets.push_back(ctx_->getDescriptorSet(id));
    }
    bindDescriptorSets(bind_point, ctx_->getPipelineLayout(pipeline_layout_id), 0, sets, {});
    return MyErrCode::kOk;
}

MyErrCode Context::copyBufferToBuffer(Uid queue_id, Uid command_pool_id, Uid src_buf_id,
                                      Uid dst_buf_id)
{
    CHECK_ERR_RET(oneTimeSubmit(queue_id, command_pool_id, [&](CommandBuffer& cmd) -> MyErrCode {
        CHECK_ERR_RET(cmd.copyBufferToBuffer(src_buf_id, dst_buf_id));
        return MyErrCode::kOk;
    }));
    return MyErrCode::kOk;
}

MyErrCode Context::copyBufferToImage(Uid queue_id, Uid command_pool_id, Uid src_buf_id,
                                     Uid dst_img_id, vk::ImageLayout dst_img_layout,
                                     uint32_t dst_img_layer)
{
    CHECK_ERR_RET(oneTimeSubmit(queue_id, command_pool_id, [&](CommandBuffer& cmd) -> MyErrCode {
        CHECK_ERR_RET(cmd.copyBufferToImage(src_buf_id, dst_img_id, dst_img_layout, dst_img_layer));
        return MyErrCode::kOk;
    }));
    return MyErrCode::kOk;
}

MyErrCode Context::copyHostToBuffer(Uid queue_id, Uid command_pool_id, void const* src_host,
                                    Uid dst_buf_id)
{
    Buffer& dst_buf = getBuffer(dst_buf_id);
    if (dst_buf.canBeMapped()) {
        ELOG("no need to call this function for buffer {} that can be mapped", dst_buf_id);
        return MyErrCode::kFailed;
    }

    Uid staging_buf_id = Uid::temp();
    uint64_t buffer_size = dst_buf.getMeta().size;

    CHECK_ERR_RET(createBuffer(
        staging_buf_id, {buffer_size, vk::BufferUsageFlagBits::eTransferSrc},
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_ALLOCATION_CREATE_MAPPED_BIT));

    memcpy(getBuffer(staging_buf_id).getMappedData(), src_host, buffer_size);
    CHECK_ERR_RET(copyBufferToBuffer(queue_id, command_pool_id, staging_buf_id, dst_buf_id));
    CHECK_ERR_RET(destroyBuffer(staging_buf_id));
    return MyErrCode::kOk;
}

MyErrCode Context::transitionImageLayout(Uid queue_id, Uid command_pool_id, Uid image_id,
                                         vk::ImageLayout old_layout, vk::ImageLayout new_layout)
{
    if (old_layout == new_layout) {
        return MyErrCode::kOk;
    }

    vk::PipelineStageFlags2 src_stage;
    vk::AccessFlags2 src_access;
    vk::PipelineStageFlags2 dst_stage;
    vk::AccessFlags2 dst_access;

    if (old_layout == vk::ImageLayout::eUndefined &&
        new_layout == vk::ImageLayout::eTransferDstOptimal) {
        src_stage = vk::PipelineStageFlagBits2::eNone;
        src_access = vk::AccessFlagBits2::eNone;
        dst_stage = vk::PipelineStageFlagBits2::eTransfer;
        dst_access = vk::AccessFlagBits2::eTransferWrite;
    } else if (old_layout == vk::ImageLayout::eTransferDstOptimal &&
               new_layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        src_stage = vk::PipelineStageFlagBits2::eTransfer;
        src_access = vk::AccessFlagBits2::eTransferWrite;
        dst_stage = vk::PipelineStageFlagBits2::eFragmentShader;
        dst_access = vk::AccessFlagBits2::eShaderRead;
    } else {
        ELOG("unsupported layout transition: {} -> {}", old_layout, new_layout);
        return MyErrCode::kFailed;
    }

    CHECK_ERR_RET(oneTimeSubmit(queue_id, command_pool_id, [&](CommandBuffer& cmd) -> MyErrCode {
        CHECK_ERR_RET(cmd.pipelineImageBarrier(image_id, old_layout, new_layout, src_stage,
                                               src_access, dst_stage, dst_access));
        return MyErrCode::kOk;
    }));
    return MyErrCode::kOk;
}

MyErrCode Context::copyHostToImage(Uid queue_id, Uid command_pool_id, void const* src_host,
                                   Uid dst_img_id, vk::ImageLayout dst_img_layout,
                                   uint32_t dst_img_layer)
{
    Image& dst_img = getImage(dst_img_id);
    ImageMeta const& image_meta = dst_img.getMeta();
    if (dst_img.canBeMapped()) {
        ELOG("no need to call this function for image {} that can be mapped", dst_img_id);
        return MyErrCode::kFailed;
    }

    Uid staging_buffer_id = Uid::temp();
    uint64_t buffer_size = vk::blockSize(image_meta.format) * image_meta.extent.width *
                           image_meta.extent.height * image_meta.extent.depth;

    CHECK_ERR_RET(createBuffer(
        staging_buffer_id, {buffer_size, vk::BufferUsageFlagBits::eTransferSrc},
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_ALLOCATION_CREATE_MAPPED_BIT));

    memcpy(getBuffer(staging_buffer_id).getMappedData(), src_host, buffer_size);
    CHECK_ERR_RET(transitionImageLayout(queue_id, command_pool_id, dst_img_id, dst_img_layout,
                                        vk::ImageLayout::eTransferDstOptimal));
    CHECK_ERR_RET(copyBufferToImage(queue_id, command_pool_id, staging_buffer_id, dst_img_id,
                                    vk::ImageLayout::eTransferDstOptimal, dst_img_layer));
    CHECK_ERR_RET(destroyBuffer(staging_buffer_id));
    return MyErrCode::kOk;
}

MyErrCode Context::updateDescriptorSet(
    Uid set_id, std::map<uint32_t, WriteDescriptorSetBinding> const& write_bindings)
{
    auto& set = getDescriptorSet(set_id);

    std::vector<vk::DescriptorImageInfo> image_infos;
    std::vector<vk::DescriptorBufferInfo> buffer_infos;
    image_infos.reserve(write_bindings.size());
    buffer_infos.reserve(write_bindings.size());

    std::vector<vk::WriteDescriptorSet> set_writes;
    for (auto& [binding, write]: write_bindings) {
        vk::DescriptorType type = set.layout_bindings_->at(binding).descriptorType;
        switch (type) {
            case vk::DescriptorType::eStorageBuffer: {
                auto& buffer = getBuffer(write.id_0_);
                buffer_infos.emplace_back(buffer, 0, buffer.getMeta().size);
                set_writes.emplace_back(set, binding, 0, 1, type, nullptr, &buffer_infos.back());
                break;
            }
            case vk::DescriptorType::eCombinedImageSampler: {
                auto& view = getImageView(write.id_0_);
                auto& sampler = getSampler(write.id_1_);
                image_infos.emplace_back(sampler, view, write.layout_);
                set_writes.emplace_back(set, binding, 0, 1, type, &image_infos.back());
                break;
            }
            default:
                ELOG("unsupported write descriptor type: {}", type);
                return MyErrCode::kFailed;
        }
    }

    device_.updateDescriptorSets(set_writes, {});
    return MyErrCode::kOk;
}

MyErrCode Context::createSwapchain(Uid id, Uid surface_id, SwapchainMeta const& meta)
{
    auto& surface = getSurface(surface_id);

    bool format_found = false;
    for (auto format: CHECK_VKHPP_VAL(physical_device_.getSurfaceFormatsKHR(surface))) {
        if (format == meta.surface_format) {
            format_found = true;
        }
    }
    if (!format_found) {
        ELOG("surface format not found: {} {}", meta.surface_format.format,
             meta.surface_format.colorSpace);
        return MyErrCode::kFailed;
    }

    vk::SurfaceCapabilitiesKHR surface_caps =
        CHECK_VKHPP_VAL(physical_device_.getSurfaceCapabilitiesKHR(surface));

    uint32_t image_count = std::max(surface_caps.minImageCount + 1, meta.image_count);
    if (surface_caps.maxImageCount > 0) {
        image_count = std::min(image_count, surface_caps.maxImageCount);
    }

    vk::SwapchainCreateInfoKHR swapchain_info(
        {}, surface, image_count, meta.surface_format.format, meta.surface_format.colorSpace,
        meta.extent, 1, meta.usages, vk::SharingMode::eExclusive, {}, surface_caps.currentTransform,
        vk::CompositeAlphaFlagBitsKHR::eOpaque, meta.mode, true);

    auto swap = CHECK_VKHPP_VAL(device_.createSwapchainKHR(swapchain_info));
    auto images = CHECK_VKHPP_VAL(device_.getSwapchainImagesKHR(swap));
    ILOG("create {} swapchain images with size={}x{}, format={}", images.size(), meta.extent.width,
         meta.extent.height, meta.surface_format.format);

    ImageMeta image_meta = {.type = vk::ImageType::e2D,
                            .format = meta.surface_format.format,
                            .aspects = vk::ImageAspectFlagBits::eColor,
                            .extent = vk::Extent3D(meta.extent, 1),
                            .layers = 1,
                            .mip_levels = 1,
                            .samples = vk::SampleCountFlagBits::e1,
                            .tiling = vk::ImageTiling::eOptimal,
                            .init_layout = vk::ImageLayout::eUndefined,
                            .usages = meta.usages};

    ImageViewMeta view_meta = {.type = vk::ImageViewType::e2D,
                               .base_level = 0,
                               .num_levels = 1,
                               .base_layer = 0,
                               .num_layers = 1,
                               .swizzle = {}};

    std::vector<Uid> image_ids;
    std::vector<Uid> view_ids;
    for (auto& image: images) {
        Uid image_id = Uid::temp();
        CHECK_ERR_RET(
            create(images_, image_id, Image{image_meta, image, VK_NULL_HANDLE, VK_NULL_HANDLE}));
        image_ids.push_back(image_id);

        Uid view_id = Uid::temp();
        CHECK_ERR_RET(createImageView(view_id, image_id, view_meta));
        view_ids.push_back(view_id);
    }

    return create(swapchains_, id, Swapchain{meta, swap, image_ids, view_ids});
}

Swapchain& Context::getSwapchain(Uid id) { return get(swapchains_, id); }

MyErrCode Context::destroySwapchain(Uid id) { return destroy(swapchains_, id); }

MyErrCode Context::createRenderPass(Uid id, RenderPassMeta const& meta)
{
    auto render_pass = CHECK_VKHPP_VAL(
        device_.createRenderPass({{}, meta.attachments, meta.subpasses, meta.dependencies}));
    return create(render_passes_, id, render_pass);
}

vk::RenderPass& Context::getRenderPass(Uid id) { return get(render_passes_, id); }

MyErrCode Context::destroyRenderPass(Uid id) { return destroy(render_passes_, id); }

MyErrCode Context::createFramebuffer(Uid id, Uid render_pass_id,
                                     std::vector<Uid> const& image_view_ids)
{
    std::vector<vk::ImageView> attachments;
    vk::Extent3D extent = {0, 0, 0};
    uint32_t layers = 0;
    for (auto view_id: image_view_ids) {
        ImageView& view = getImageView(view_id);
        if (extent.width != 0 && extent != view.img_meta_->extent) {
            ELOG("inconsistent image extent: {}", view_id);
            return MyErrCode::kFailed;
        }
        extent = view.img_meta_->extent;
        if (layers != 0 && layers != view.meta_.num_layers) {
            ELOG("inconsistent image layers: {}", view_id);
            return MyErrCode::kFailed;
        }
        layers = view.meta_.num_layers;
        attachments.push_back(view.view_);
    }
    auto framebuffer = CHECK_VKHPP_VAL(device_.createFramebuffer(
        {{}, getRenderPass(render_pass_id), attachments, extent.width, extent.height, layers}));
    return create(framebuffers_, id, framebuffer);
}

vk::Framebuffer& Context::getFramebuffer(Uid id) { return get(framebuffers_, id); }

MyErrCode Context::destroyFramebuffer(Uid id) { return destroy(framebuffers_, id); }

MyErrCode Context::waitQueueIdle(Uid queue_id)
{
    vk::Queue queue = getQueue(queue_id);
    CHECK_VKHPP_RET(queue.waitIdle());
    return MyErrCode::kOk;
}

MyErrCode Context::waitFences(std::vector<Uid> const& fence_ids, bool wait_all, uint64_t timeout)
{
    std::vector<vk::Fence> fences;
    for (auto id: fence_ids) {
        fences.push_back(getFence(id));
    }
    CHECK_VKHPP_RET(device_.waitForFences(fences, wait_all, timeout));
    return MyErrCode::kOk;
}

MyErrCode Context::waitSemaphores(std::vector<SemaphoreSubmitInfo> const& wait_semaphores,
                                  uint64_t timeout)
{
    std::vector<vk::Semaphore> sems;
    std::vector<uint64_t> vals;
    for (auto& s: wait_semaphores) {
        auto& sem = getSemaphore(s.id_);
        if (!sem.isTimeline()) {
            ELOG("semaphore not timeline: {}", s.id_);
            return MyErrCode::kFailed;
        }
        sems.push_back(sem.sem_);
        vals.push_back(s.val_);
    }
    CHECK_VKHPP_RET(device_.waitSemaphores({{}, sems, vals}, timeout));
    return MyErrCode::kOk;
}

MyErrCode Context::signalSemaphore(SemaphoreSubmitInfo const& signal_semaphore)
{
    auto& sem = getSemaphore(signal_semaphore.id_);
    if (!sem.isTimeline()) {
        ELOG("semaphore not timeline: {}", signal_semaphore.id_);
        return MyErrCode::kFailed;
    }
    CHECK_VKHPP_RET(device_.signalSemaphore(vk::SemaphoreSignalInfo{sem, signal_semaphore.val_}));
    return MyErrCode::kOk;
}

MyErrCode Context::recordCommand(Uid command_buffer_id, vk::CommandBufferUsageFlags usage,
                                 CmdSubmitter const& submitter)
{
    CommandBuffer& command_buffer = getCommandBuffer(command_buffer_id);
    CHECK_VKHPP_RET(command_buffer.begin({usage}));
    CHECK_ERR_RET(submitter(command_buffer));
    CHECK_VKHPP_RET(command_buffer.end());
    return MyErrCode::kOk;
}

MyErrCode Context::oneTimeSubmit(Uid queue_id, Uid command_pool_id, CmdSubmitter const& submitter)
{
    Uid command_buffer_id = Uid::temp();
    CHECK_ERR_RET(createCommandBuffer(command_buffer_id, command_pool_id));
    CHECK_ERR_RET(recordCommand(command_buffer_id, vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
                                submitter));
    CHECK_ERR_RET(submit(queue_id, command_buffer_id));
    CHECK_ERR_RET(waitQueueIdle(queue_id));
    CHECK_ERR_RET(destroyCommandBuffer(command_buffer_id));
    return MyErrCode::kOk;
}

MyErrCode Context::submit(Uid queue_id, Uid command_buffer_id,
                          std::vector<SemaphoreSubmitInfo> const& wait_semaphores,
                          std::vector<SemaphoreSubmitInfo> const& signal_semaphores, Uid fence_id)
{
    std::vector<vk::SemaphoreSubmitInfo> wait_sems;
    std::vector<vk::SemaphoreSubmitInfo> signal_sems;
    for (auto s: wait_semaphores) {
        wait_sems.emplace_back(getSemaphore(s.id_), s.val_, s.stages_);
    }
    for (auto s: signal_semaphores) {
        signal_sems.emplace_back(getSemaphore(s.id_), s.val_, s.stages_);
    }
    vk::Queue queue = getQueue(queue_id);
    CommandBuffer& command_buffer = getCommandBuffer(command_buffer_id);
    vk::CommandBufferSubmitInfo cmdbuf_info = {command_buffer};
    vk::SubmitInfo2 submit_info = {vk::SubmitFlags{}, wait_sems, cmdbuf_info, signal_sems};
    CHECK_VKHPP_RET(
        queue.submit2(submit_info, fence_id != Uid::kNull ? getFence(fence_id) : vk::Fence{}));
    return MyErrCode::kOk;
}

MyErrCode Context::destroy()
{
    CHECK_VKHPP_RET(device_.waitIdle());

    CHECK_ERR_RET(destroy(framebuffers_));
    CHECK_ERR_RET(destroy(render_passes_));
    CHECK_ERR_RET(destroy(pipelines_));
    CHECK_ERR_RET(destroy(pipeline_layouts_));
    CHECK_ERR_RET(destroy(descriptor_sets_));
    CHECK_ERR_RET(destroy(descriptor_set_layouts_));
    CHECK_ERR_RET(destroy(fences_));
    CHECK_ERR_RET(destroy(semaphores_));
    CHECK_ERR_RET(destroy(command_buffers_));
    CHECK_ERR_RET(destroy(image_views_));
    CHECK_ERR_RET(destroy(images_));
    CHECK_ERR_RET(destroy(swapchains_));
    CHECK_ERR_RET(destroy(buffers_));
    CHECK_ERR_RET(destroy(samplers_));
    CHECK_ERR_RET(destroy(shader_modules_));
    CHECK_ERR_RET(destroy(descriptor_pools_));
    CHECK_ERR_RET(destroy(command_pools_));
    CHECK_ERR_RET(destroy(surfaces_));
    CHECK_ERR_RET(destroy(queues_));

    if (allocator_) {
        vmaDestroyAllocator(allocator_);
    }
    if (device_) {
        device_.destroy();
    }
    if (debug_messenger_) {
        instance_.destroy(debug_messenger_);
    }
    if (instance_) {
        instance_.destroy();
    }
    return MyErrCode::kOk;
}

}  // namespace myvk
