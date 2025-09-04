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

bool Allocation::isMapped() const { return getAllocInfo().pMappedData != nullptr; }

void* Allocation::map()
{
    void* data = nullptr;
    if (auto err = vmaMapMemory(allocator_, alloc_, &data); err != VK_SUCCESS) {
        MY_THROW("failed to map memory: {}", err);
    }
    return data;
}

void Allocation::unmap() { vmaUnmapMemory(allocator_, alloc_); }

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

Image::Image(ImageMeta const& meta, vk::Image img, vk::ImageView img_view, VmaAllocation alloc,
             VmaAllocator allocator)
    : meta_(meta), img_(img), img_view_(img_view), Allocation(alloc, allocator)
{
    meta_.size = vk::blockSize(meta_.format) * meta_.extent.width * meta_.extent.height;
}

ImageMeta const& Image::getMeta() const { return meta_; }

Image::operator vk::Image() const { return img_; }

Image::operator vk::ImageView() const { return img_view_; }

Image::operator bool() const { return img_; }

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
                     std::vector<Uid> const& image_ids)
    : meta_(meta), swapchain_(swapchain), image_ids_(image_ids)
{
    meta_.image_count = image_ids.size();
}

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
    : set_(set), pool_(pool), layout_bindings_(layout_bindings)
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
MyErrCode Context::setDebugObjectId(T obj, Uid id)
{
    std::string name = FSTR("{}", id);
    CHECK_VKHPP_RET(device_.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT(
        T::objectType, reinterpret_cast<uint64_t>(static_cast<typename T::CType>(obj)),
        name.c_str())));
    return MyErrCode::kOk;
}

MyErrCode Context::createSurface(Uid id, vk::SurfaceKHR surface)
{
    if (surfaces_.find(id) != surfaces_.end()) {
        CHECK_ERR_RET(destroySurface(id));
    }
    surfaces_.emplace(id, surface);
    CHECK_ERR_RET(setDebugObjectId(surface, id));
    return MyErrCode::kOk;
}

vk::SurfaceKHR& Context::getSurface(Uid id)
{
    if (surfaces_.find(id) == surfaces_.end()) {
        MY_THROW("surface not exist: {}", id);
    }
    return surfaces_.at(id);
}

MyErrCode Context::destroySurface(Uid id)
{
    if (auto it = surfaces_.find(id); it != surfaces_.end()) {
        instance_.destroy(it->second);
        surfaces_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("surface not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

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
            queues_.emplace(id, Queue{queue, family});
            CHECK_ERR_RET(setDebugObjectId(queue, id));
            DLOG("create queue {}: family {}, {}", id, family, family_props[family].queueFlags);
        }
    }

    return MyErrCode::kOk;
}

vk::Device& Context::getDevice() { return device_; }

Queue& Context::getQueue(Uid id)
{
    if (queues_.find(id) == queues_.end()) {
        MY_THROW("queue not exist: {}", id);
    }
    return queues_.at(id);
}

MyErrCode Context::createCommandPool(Uid id, Uid queue_id, vk::CommandPoolCreateFlags flags)
{
    if (command_pools_.find(id) != command_pools_.end()) {
        CHECK_ERR_RET(destroyCommandPool(id));
    }
    auto pool =
        CHECK_VKHPP_VAL(device_.createCommandPool({flags, getQueue(queue_id).getFamilyIndex()}));
    command_pools_.emplace(id, pool);
    CHECK_ERR_RET(setDebugObjectId(pool, id));
    return MyErrCode::kOk;
}

vk::CommandPool& Context::getCommandPool(Uid id)
{
    if (command_pools_.find(id) == command_pools_.end()) {
        MY_THROW("command pool not exist: {}", id);
    }
    return command_pools_.at(id);
}

MyErrCode Context::destroyCommandPool(Uid id)
{
    if (auto it = command_pools_.find(id); it != command_pools_.end()) {
        device_.destroy(it->second);
        command_pools_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("command pool not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createDescriptorPool(Uid id, uint32_t max_sets,
                                        std::map<vk::DescriptorType, uint32_t> const& sizes,
                                        vk::DescriptorPoolCreateFlags flags)
{
    if (descriptor_pools_.find(id) != descriptor_pools_.end()) {
        CHECK_ERR_RET(destroyDescriptorPool(id));
    }
    std::vector<vk::DescriptorPoolSize> pool_size;
    for (auto [type, count]: sizes) {
        pool_size.emplace_back(type, count);
    }
    auto pool = CHECK_VKHPP_VAL(device_.createDescriptorPool({flags, max_sets, pool_size}));
    descriptor_pools_.emplace(id, pool);
    CHECK_ERR_RET(setDebugObjectId(pool, id));
    return MyErrCode::kOk;
}

vk::DescriptorPool& Context::getDescriptorPool(Uid id)
{
    if (descriptor_pools_.find(id) == descriptor_pools_.end()) {
        MY_THROW("descriptor pool not exist: {}", id);
    }
    return descriptor_pools_.at(id);
}

MyErrCode Context::destroyDescriptorPool(Uid id)
{
    if (auto it = descriptor_pools_.find(id); it != descriptor_pools_.end()) {
        device_.destroy(it->second);
        descriptor_pools_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("descriptor pool not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

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
    if (shader_modules_.find(id) != shader_modules_.end()) {
        CHECK_ERR_RET(destroyShaderModule(id));
    }
    std::vector<uint8_t> code;
    CHECK_ERR_RET(toolkit::readBinaryFile(file_path, code));
    vk::ShaderModuleCreateInfo create_info(vk::ShaderModuleCreateFlags(), code.size(),
                                           reinterpret_cast<uint32_t const*>(code.data()));
    auto shader = CHECK_VKHPP_VAL(device_.createShaderModule(create_info));
    shader_modules_.emplace(id, shader);
    CHECK_ERR_RET(setDebugObjectId(shader, id));
    return MyErrCode::kOk;
}

vk::ShaderModule& Context::getShaderModule(Uid id)
{
    if (shader_modules_.find(id) == shader_modules_.end()) {
        MY_THROW("shader module not exist: {}", id);
    }
    return shader_modules_.at(id);
}

MyErrCode Context::destroyShaderModule(Uid id)
{
    if (auto it = shader_modules_.find(id); it != shader_modules_.end()) {
        device_.destroy(it->second);
        shader_modules_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("shader module not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createBuffer(Uid id, BufferMeta const& meta, vk::MemoryPropertyFlags properties,
                                VmaAllocationCreateFlags flags)
{
    if (buffers_.find(id) != buffers_.end()) {
        CHECK_ERR_RET(destroyBuffer(id));
    }

    vk::BufferCreateInfo buffer_info(vk::BufferCreateFlags{}, meta.size, meta.usages,
                                     vk::SharingMode::eExclusive);

    VmaAllocationCreateInfo creation_info = {};
    creation_info.flags = flags;
    creation_info.requiredFlags = static_cast<VkMemoryPropertyFlags>(properties);

    VkBuffer buf;
    VmaAllocation alloc;
    CHECK_VK_RET(vmaCreateBuffer(allocator_, buffer_info, &creation_info, &buf, &alloc, nullptr));

    buffers_.emplace(id, Buffer{meta, buf, alloc, allocator_});
    CHECK_ERR_RET(setDebugObjectId(buffers_.at(id).buf_, id));
    DLOG("create buffer {}: {} bytes, {}", id, meta.size, buffers_.at(id).getMemProp());
    return MyErrCode::kOk;
}

Buffer& Context::getBuffer(Uid id)
{
    if (buffers_.find(id) == buffers_.end()) {
        MY_THROW("buffer not exist: {}", id);
    }
    return buffers_.at(id);
}

MyErrCode Context::destroyBuffer(Uid id)
{
    if (auto it = buffers_.find(id); it != buffers_.end()) {
        vmaDestroyBuffer(it->second.allocator_, it->second.buf_, it->second.alloc_);
        buffers_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("buffer not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createImage(Uid id, ImageMeta const& meta, vk::MemoryPropertyFlags properties,
                               VmaAllocationCreateFlags flags)
{
    if (images_.find(id) != images_.end()) {
        CHECK_ERR_RET(destroyImage(id));
    }

    vk::ImageCreateInfo image_info(vk::ImageCreateFlags(), vk::ImageType::e2D, meta.format,
                                   vk::Extent3D(meta.extent, 1), meta.mip_levels, 1, meta.samples,
                                   meta.tiling, meta.usages, vk::SharingMode::eExclusive, {},
                                   meta.init_layout);

    VmaAllocationCreateInfo vma_info = {};
    vma_info.flags = flags;
    vma_info.requiredFlags = static_cast<VkMemoryPropertyFlags>(properties);

    VkImage img;
    VmaAllocation alloc;
    VmaAllocationInfo alloc_info;
    CHECK_VK_RET(vmaCreateImage(allocator_, image_info, &vma_info, &img, &alloc, &alloc_info));

    vk::ImageViewCreateInfo view_info({}, img, vk::ImageViewType::e2D, meta.format, {},
                                      {meta.aspects, 0, meta.mip_levels, 0, 1});
    vk::ImageView img_view = CHECK_VKHPP_VAL(device_.createImageView(view_info));

    images_.emplace(id, Image{meta, img, img_view, alloc, allocator_});
    CHECK_ERR_RET(setDebugObjectId(images_.at(id).img_, id));
    CHECK_ERR_RET(setDebugObjectId(img_view, id));
    return MyErrCode::kOk;
}

Image& Context::getImage(Uid id)
{
    if (images_.find(id) == images_.end()) {
        MY_THROW("image not exist: {}", id);
    }
    return images_.at(id);
}

MyErrCode Context::destroyImage(Uid id)
{
    if (auto it = images_.find(id); it != images_.end()) {
        device_.destroy(it->second.img_view_);
        if (it->second.alloc_) {
            vmaDestroyImage(it->second.allocator_, it->second.img_, it->second.alloc_);
        }
        images_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("image not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createSampler() { return MyErrCode::kOk; }

vk::Sampler& Context::getSampler(Uid id)
{
    if (samplers_.find(id) == samplers_.end()) {
        MY_THROW("sampler not exist: {}", id);
    }
    return samplers_.at(id);
}

MyErrCode Context::destroySampler(Uid id)
{
    if (auto it = samplers_.find(id); it != samplers_.end()) {
        device_.destroy(it->second);
        samplers_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("sampler not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createCommandBuffer(Uid id, Uid command_pool_id)
{
    if (command_buffers_.find(id) != command_buffers_.end()) {
        CHECK_ERR_RET(destroyDescriptorSet(id));
    }
    auto& pool = getCommandPool(command_pool_id);
    auto bufs = CHECK_VKHPP_VAL(
        device_.allocateCommandBuffers({pool, vk::CommandBufferLevel::ePrimary, 1}));
    command_buffers_.emplace(id, CommandBuffer{bufs.front(), pool, *this});
    CHECK_ERR_RET(setDebugObjectId(bufs.front(), id));
    return MyErrCode::kOk;
}

CommandBuffer& Context::getCommandBuffer(Uid id)
{
    if (command_buffers_.find(id) == command_buffers_.end()) {
        MY_THROW("command buffer not exist: {}", id);
    }
    return command_buffers_.at(id);
}

MyErrCode Context::destroyCommandBuffer(Uid id)
{
    if (auto it = command_buffers_.find(id); it != command_buffers_.end()) {
        device_.freeCommandBuffers(it->second.pool_, it->second);
        command_buffers_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("command buffer not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createBinarySemaphore(Uid id)
{
    if (semaphores_.find(id) != semaphores_.end()) {
        CHECK_ERR_RET(destroySemaphore(id));
    }
    auto sem = CHECK_VKHPP_VAL(device_.createSemaphore({}));
    semaphores_.emplace(id, Semaphore{vk::SemaphoreType::eBinary, sem});
    CHECK_ERR_RET(setDebugObjectId(sem, id));
    return MyErrCode::kOk;
}

MyErrCode Context::createTimelineSemaphore(Uid id, uint64_t init_val)
{
    if (semaphores_.find(id) != semaphores_.end()) {
        CHECK_ERR_RET(destroySemaphore(id));
    }
    vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo> c{
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{vk::SemaphoreType::eTimeline, init_val},
    };
    auto sem = CHECK_VKHPP_VAL(device_.createSemaphore(c.get<vk::SemaphoreCreateInfo>()));
    semaphores_.emplace(id, Semaphore{vk::SemaphoreType::eTimeline, sem});
    CHECK_ERR_RET(setDebugObjectId(sem, id));
    return MyErrCode::kOk;
}

Semaphore& Context::getSemaphore(Uid id)
{
    if (semaphores_.find(id) == semaphores_.end()) {
        MY_THROW("semaphore not exist: {}", id);
    }
    return semaphores_.at(id);
}

MyErrCode Context::destroySemaphore(Uid id)
{
    if (auto it = semaphores_.find(id); it != semaphores_.end()) {
        device_.destroy(it->second.sem_);
        semaphores_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("semaphore not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createFence(Uid id, bool init_signaled)
{
    if (fences_.find(id) != fences_.end()) {
        CHECK_ERR_RET(destroyFence(id));
    }
    auto fence = CHECK_VKHPP_VAL(device_.createFence(
        {init_signaled ? vk::FenceCreateFlagBits::eSignaled : vk::FenceCreateFlags{}}));
    fences_.emplace(id, fence);
    CHECK_ERR_RET(setDebugObjectId(fence, id));
    return MyErrCode::kOk;
}

vk::Fence& Context::getFence(Uid id)
{
    if (fences_.find(id) == fences_.end()) {
        MY_THROW("fence not exist: {}", id);
    }
    return fences_.at(id);
}

MyErrCode Context::destroyFence(Uid id)
{
    if (auto it = fences_.find(id); it != fences_.end()) {
        device_.destroy(it->second);
        fences_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("fence not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createPipelineLayout(Uid id, std::vector<Uid> const& set_layout_ids,
                                        std::vector<vk::PushConstantRange> const& push_ranges)
{
    if (pipeline_layouts_.find(id) != pipeline_layouts_.end()) {
        CHECK_ERR_RET(destroyPipelineLayout(id));
    }
    std::vector<vk::DescriptorSetLayout> set_layouts;
    for (auto id: set_layout_ids) {
        set_layouts.push_back(getDescriptorSetLayout(id));
    }
    auto layout = CHECK_VKHPP_VAL(device_.createPipelineLayout({{}, set_layouts, push_ranges}));
    pipeline_layouts_.emplace(id, layout);
    CHECK_ERR_RET(setDebugObjectId(layout, id));
    return MyErrCode::kOk;
}

vk::PipelineLayout& Context::getPipelineLayout(Uid id)
{
    if (pipeline_layouts_.find(id) == pipeline_layouts_.end()) {
        MY_THROW("pipeline layout not exist: {}", id);
    }
    return pipeline_layouts_.at(id);
}

MyErrCode Context::destroyPipelineLayout(Uid id)
{
    if (auto it = pipeline_layouts_.find(id); it != pipeline_layouts_.end()) {
        device_.destroy(it->second);
        pipeline_layouts_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("pipeline layout not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createComputePipeline(Uid id, Uid pipeline_layout_id, Uid shader_id)
{
    if (pipelines_.find(id) != pipelines_.end()) {
        CHECK_ERR_RET(destroyPipeline(id));
    }
    vk::PipelineShaderStageCreateInfo shader_info(vk::PipelineShaderStageCreateFlags(),
                                                  vk::ShaderStageFlagBits::eCompute,
                                                  getShaderModule(shader_id), "main");
    vk::ComputePipelineCreateInfo create_info(vk::PipelineCreateFlags(), shader_info,
                                              getPipelineLayout(pipeline_layout_id));
    auto pipeline = CHECK_VKHPP_VAL(device_.createComputePipeline({}, create_info));
    pipelines_.emplace(id, pipeline);
    CHECK_ERR_RET(setDebugObjectId(pipeline, id));
    return MyErrCode::kOk;
}

vk::Pipeline& Context::getPipeline(Uid id)
{
    if (pipelines_.find(id) == pipelines_.end()) {
        MY_THROW("pipeline not exist: {}", id);
    }
    return pipelines_.at(id);
}

MyErrCode Context::destroyPipeline(Uid id)
{
    if (auto it = pipelines_.find(id); it != pipelines_.end()) {
        device_.destroy(it->second);
        pipelines_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("pipeline not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createDescriptorSetLayout(
    Uid id, std::vector<DescriptorSetLayoutBinding> const& bindings)
{
    if (descriptor_set_layouts_.find(id) != descriptor_set_layouts_.end()) {
        CHECK_ERR_RET(destroyDescriptorSetLayout(id));
    }
    std::vector<vk::DescriptorSetLayoutBinding> layout_bindings;
    for (int i = 0; i < bindings.size(); ++i) {
        vk::DescriptorSetLayoutBinding b = bindings[i];
        b.setBinding(i);
        layout_bindings.push_back(b);
    }
    auto layout = CHECK_VKHPP_VAL(
        device_.createDescriptorSetLayout({vk::DescriptorSetLayoutCreateFlags(), layout_bindings}));
    descriptor_set_layouts_.emplace(id, DescriptorSetLayout{layout, layout_bindings});
    CHECK_ERR_RET(setDebugObjectId(layout, id));
    return MyErrCode::kOk;
}

DescriptorSetLayout& Context::getDescriptorSetLayout(Uid id)
{
    if (descriptor_set_layouts_.find(id) == descriptor_set_layouts_.end()) {
        MY_THROW("descriptor set layout not exist: {}", id);
    }
    return descriptor_set_layouts_.at(id);
}

MyErrCode Context::destroyDescriptorSetLayout(Uid id)
{
    if (auto it = descriptor_set_layouts_.find(id); it != descriptor_set_layouts_.end()) {
        device_.destroy(it->second);
        descriptor_set_layouts_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("descriptor set layout not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createDescriptorSet(Uid id, Uid set_layout_id, Uid descriptor_pool_id)
{
    if (descriptor_sets_.find(id) != descriptor_sets_.end()) {
        CHECK_ERR_RET(destroyDescriptorSet(id));
    }
    auto& pool = getDescriptorPool(descriptor_pool_id);
    DescriptorSetLayout& layout = getDescriptorSetLayout(set_layout_id);
    auto sets = CHECK_VKHPP_VAL(device_.allocateDescriptorSets({pool, layout.layout_}));
    descriptor_sets_.emplace(id, DescriptorSet{sets.front(), pool, layout.bindings_});
    CHECK_ERR_RET(setDebugObjectId(sets.front(), id));
    return MyErrCode::kOk;
}

DescriptorSet& Context::getDescriptorSet(Uid id)
{
    if (descriptor_sets_.find(id) == descriptor_sets_.end()) {
        MY_THROW("descriptor set not exist: {}", id);
    }
    return descriptor_sets_.at(id);
}

MyErrCode Context::destroyDescriptorSet(Uid id)
{
    if (auto it = descriptor_sets_.find(id); it != descriptor_sets_.end()) {
        device_.freeDescriptorSets(it->second.pool_, {it->second.set_});
        descriptor_sets_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("descriptor set not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

CommandBuffer::CommandBuffer(vk::CommandBuffer buf, vk::CommandPool pool, Context& ctx)
    : vk::CommandBuffer(buf), pool_(pool), ctx_(ctx)
{
}

MyErrCode CommandBuffer::copyBufferToBuffer(Uid src_buf_id, Uid dst_buf_id,
                                            vk::DeviceSize src_offset, vk::DeviceSize dst_offset,
                                            vk::DeviceSize size)
{
    Buffer& dst_buf = ctx_.getBuffer(dst_buf_id);
    Buffer& src_buf = ctx_.getBuffer(src_buf_id);
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
                                           vk::ImageLayout dst_img_layout)
{
    Image& dst_img = ctx_.getImage(dst_img_id);
    Buffer& src_buf = ctx_.getBuffer(src_buf_id);
    if (src_buf.getMeta().size != dst_img.getMeta().size) {
        ELOG("invalid buffer size for copy: {} != {}", src_buf.getMeta().size,
             dst_img.getMeta().size);
        return MyErrCode::kFailed;
    }
    vk::BufferImageCopy region(0, 0, 0,
                               vk::ImageSubresourceLayers{dst_img.getMeta().aspects, 0, 0, 1},
                               {0, 0, 0}, vk::Extent3D(dst_img.getMeta().extent, 1));
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
    Image image = ctx_.getImage(image_id);
    vk::ImageMemoryBarrier2 barrier(src_stage, src_access, dst_stage, dst_access, old_layout,
                                    new_layout, vk::QueueFamilyIgnored, vk::QueueFamilyIgnored,
                                    image,
                                    {image.getMeta().aspects, 0, image.getMeta().mip_levels, 0, 1});
    pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, barrier});
    return MyErrCode::kOk;
}

MyErrCode CommandBuffer::pushConstants(Uid pipeline_layout_id, vk::ShaderStageFlags stages,
                                       uint32_t offset, uint32_t size, void const* data)
{
    pushConstants(ctx_.getPipelineLayout(pipeline_layout_id), stages, offset, size, data);
    return MyErrCode::kOk;
}

MyErrCode CommandBuffer::bindComputePipeline(Uid pipeline_id)
{
    bindPipeline(vk::PipelineBindPoint::eCompute, ctx_.getPipeline(pipeline_id));
    return MyErrCode::kOk;
}

MyErrCode CommandBuffer::bindDescriptorSets(vk::PipelineBindPoint bind_point,
                                            Uid pipeline_layout_id, std::vector<Uid> const& set_ids)
{
    std::vector<vk::DescriptorSet> sets;
    for (auto id: set_ids) {
        sets.push_back(ctx_.getDescriptorSet(id));
    }
    bindDescriptorSets(bind_point, ctx_.getPipelineLayout(pipeline_layout_id), 0, sets, {});
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
                                     Uid dst_img_id, vk::ImageLayout dst_img_layout)
{
    CHECK_ERR_RET(oneTimeSubmit(queue_id, command_pool_id, [&](CommandBuffer& cmd) -> MyErrCode {
        CHECK_ERR_RET(cmd.copyBufferToImage(src_buf_id, dst_img_id, dst_img_layout));
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

    memcpy(getBuffer(staging_buf_id).getAllocInfo().pMappedData, src_host, buffer_size);
    CHECK_ERR_RET(copyBufferToBuffer(queue_id, command_pool_id, dst_buf_id, staging_buf_id));
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
                                   Uid dst_img_id, vk::ImageLayout dst_img_layout)
{
    Image& dst_img = getImage(dst_img_id);
    if (dst_img.canBeMapped()) {
        ELOG("no need to call this function for image {} that can be mapped", dst_img_id);
        return MyErrCode::kFailed;
    }

    Uid staging_buffer_id = Uid::temp();
    uint64_t buffer_size = dst_img.getMeta().size;

    CHECK_ERR_RET(createBuffer(
        staging_buffer_id, {buffer_size, vk::BufferUsageFlagBits::eTransferSrc},
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_ALLOCATION_CREATE_MAPPED_BIT));

    memcpy(getBuffer(staging_buffer_id).getAllocInfo().pMappedData, src_host, buffer_size);
    CHECK_ERR_RET(transitionImageLayout(queue_id, command_pool_id, dst_img_id, dst_img_layout,
                                        vk::ImageLayout::eTransferDstOptimal));
    CHECK_ERR_RET(copyBufferToImage(queue_id, command_pool_id, dst_img_id, staging_buffer_id,
                                    vk::ImageLayout::eTransferDstOptimal));
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
        vk::DescriptorType type = set.layout_bindings_.at(binding).descriptorType;
        switch (type) {
            case vk::DescriptorType::eStorageBuffer: {
                auto& buffer = getBuffer(write.id_0_);
                buffer_infos.emplace_back(buffer, 0, buffer.getMeta().size);
                set_writes.emplace_back(set, binding, 0, 1, type, nullptr, &buffer_infos.back());
                break;
            }
            case vk::DescriptorType::eCombinedImageSampler: {
                auto& image = getImage(write.id_0_);
                auto& sampler = getSampler(write.id_1_);
                image_infos.emplace_back(sampler, image.img_view_, write.layout_);
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
    if (swapchains_.find(id) != swapchains_.end()) {
        CHECK_ERR_RET(destroySwapchain(id));
    }

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

    ImageMeta image_meta = {.format = meta.surface_format.format,
                            .extent = meta.extent,
                            .tiling = vk::ImageTiling::eOptimal,
                            .init_layout = vk::ImageLayout::eUndefined,
                            .mip_levels = 1,
                            .samples = vk::SampleCountFlagBits::e1,
                            .aspects = vk::ImageAspectFlagBits::eColor,
                            .usages = meta.usages};

    std::vector<Uid> image_ids;
    for (auto& image: images) {
        vk::ImageViewCreateInfo view_info({}, image, vk::ImageViewType::e2D,
                                          meta.surface_format.format, {},
                                          {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        vk::ImageView image_view = CHECK_VKHPP_VAL(device_.createImageView(view_info));
        Uid image_id = Uid::temp();
        images_.emplace(image_id,
                        Image{image_meta, image, image_view, VK_NULL_HANDLE, VK_NULL_HANDLE});
        CHECK_ERR_RET(setDebugObjectId(image, image_id));
        CHECK_ERR_RET(setDebugObjectId(image_view, image_id));
        image_ids.push_back(image_id);
    }

    Swapchain swapchain{meta, swap, image_ids};
    swapchains_.emplace(id, std::move(swapchain));
    CHECK_ERR_RET(setDebugObjectId(swap, id));
    return MyErrCode::kOk;
}

Swapchain& Context::getSwapchain(Uid id)
{
    if (swapchains_.find(id) == swapchains_.end()) {
        MY_THROW("swapchain not exist: {}", id);
    }
    return swapchains_.at(id);
}

MyErrCode Context::destroySwapchain(Uid id)
{
    if (auto it = swapchains_.find(id); it != swapchains_.end()) {
        device_.destroy(it->second.swapchain_);
        swapchains_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("swapchain not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createRenderPass(Uid id, RenderPassMeta const& meta)
{
    if (render_passes_.find(id) != render_passes_.end()) {
        CHECK_ERR_RET(destroyRenderPass(id));
    }
    auto render_pass = CHECK_VKHPP_VAL(
        device_.createRenderPass({{}, meta.attachments, meta.subpasses, meta.dependencies}));
    render_passes_.emplace(id, render_pass);
    CHECK_ERR_RET(setDebugObjectId(render_pass, id));
    return MyErrCode::kOk;
}

vk::RenderPass& Context::getRenderPass(Uid id)
{
    if (render_passes_.find(id) == render_passes_.end()) {
        MY_THROW("render pass not exist: {}", id);
    }
    return render_passes_.at(id);
}

MyErrCode Context::destroyRenderPass(Uid id)
{
    if (auto it = render_passes_.find(id); it != render_passes_.end()) {
        device_.destroy(it->second);
        render_passes_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("render pass not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createFramebuffer(Uid id, Uid render_pass_id, std::vector<Uid> const& image_ids)
{
    if (framebuffers_.find(id) != framebuffers_.end()) {
        CHECK_ERR_RET(destroyFramebuffer(id));
    }
    std::vector<vk::ImageView> attachments;
    vk::Extent2D extent = {0, 0};
    for (auto id: image_ids) {
        Image& image = getImage(id);
        if (extent.width != 0 && extent != image.meta_.extent) {
            ELOG("inconsistent image extent: {}", id);
            return MyErrCode::kFailed;
        }
        extent = image.meta_.extent;
        attachments.push_back(image.img_view_);
    }
    auto framebuffer = CHECK_VKHPP_VAL(device_.createFramebuffer(
        {{}, getRenderPass(render_pass_id), attachments, extent.width, extent.height, 1}));
    framebuffers_.emplace(id, framebuffer);
    CHECK_ERR_RET(setDebugObjectId(framebuffer, id));
    return MyErrCode::kOk;
}

vk::Framebuffer& Context::getFramebuffer(Uid id)
{
    if (framebuffers_.find(id) == framebuffers_.end()) {
        MY_THROW("framebuffer not exist: {}", id);
    }
    return framebuffers_.at(id);
}

MyErrCode Context::destroyFramebuffer(Uid id)
{
    if (auto it = framebuffers_.find(id); it != framebuffers_.end()) {
        device_.destroy(it->second);
        framebuffers_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("framebuffer not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

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
    while (!render_passes_.empty()) {
        CHECK_ERR_RET(destroyRenderPass(render_passes_.begin()->first));
    }
    while (!pipelines_.empty()) {
        CHECK_ERR_RET(destroyPipeline(pipelines_.begin()->first));
    }
    while (!pipeline_layouts_.empty()) {
        CHECK_ERR_RET(destroyPipelineLayout(pipeline_layouts_.begin()->first));
    }
    while (!descriptor_sets_.empty()) {
        CHECK_ERR_RET(destroyDescriptorSet(descriptor_sets_.begin()->first));
    }
    while (!descriptor_set_layouts_.empty()) {
        CHECK_ERR_RET(destroyDescriptorSetLayout(descriptor_set_layouts_.begin()->first));
    }
    while (!fences_.empty()) {
        CHECK_ERR_RET(destroyFence(fences_.begin()->first));
    }
    while (!semaphores_.empty()) {
        CHECK_ERR_RET(destroySemaphore(semaphores_.begin()->first));
    }
    while (!command_buffers_.empty()) {
        CHECK_ERR_RET(destroyCommandBuffer(command_buffers_.begin()->first));
    }
    while (!images_.empty()) {
        CHECK_ERR_RET(destroyImage(images_.begin()->first));
    }
    while (!swapchains_.empty()) {
        CHECK_ERR_RET(destroySwapchain(swapchains_.begin()->first));
    }
    while (!buffers_.empty()) {
        CHECK_ERR_RET(destroyBuffer(buffers_.begin()->first));
    }
    while (!samplers_.empty()) {
        CHECK_ERR_RET(destroySampler(samplers_.begin()->first));
    }
    while (!shader_modules_.empty()) {
        CHECK_ERR_RET(destroyShaderModule(shader_modules_.begin()->first));
    }
    while (!descriptor_pools_.empty()) {
        CHECK_ERR_RET(destroyDescriptorPool(descriptor_pools_.begin()->first));
    }
    while (!command_pools_.empty()) {
        CHECK_ERR_RET(destroyCommandPool(command_pools_.begin()->first));
    }
    while (!surfaces_.empty()) {
        CHECK_ERR_RET(destroySurface(surfaces_.begin()->first));
    }
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
