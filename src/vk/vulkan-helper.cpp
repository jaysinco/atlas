#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#include "vulkan-helper.h"
#include "toolkit/toolkit.h"
#include "toolkit/logging.h"
#include <set>

#define MYVK_API_VERSION VK_MAKE_API_VERSION(0, MYVK_API_VERSION_MAJOR, MYVK_API_VERSION_MINOR, 0)

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace myvk
{

Buffer::Buffer() = default;

Buffer::Buffer(uint64_t size, vk::Buffer buf, VmaAllocation alloc, VmaAllocator allocator)
    : size_(size), buf_(buf), alloc_(alloc), allocator_(allocator)
{
}

VmaAllocationInfo Buffer::getAllocInfo() const
{
    VmaAllocationInfo alloc_info = {};
    vmaGetAllocationInfo(allocator_, alloc_, &alloc_info);
    return alloc_info;
}

uint64_t Buffer::getSize() const { return size_; }

Buffer::operator vk::Buffer() const { return buf_; }

Buffer::operator bool() const { return buf_; }

Image::Image() = default;

Image::Image(vk::Format format, vk::Extent2D extent, vk::Image img, vk::ImageView img_view,
             VmaAllocation alloc, VmaAllocator allocator)
    : format_(format),
      extent_(extent),
      img_(img),
      img_view_(img_view),
      alloc_(alloc),
      allocator_(allocator)
{
}

Image::operator vk::Image() const { return img_; }

Image::operator vk::ImageView() const { return img_view_; }

Image::operator bool() const { return img_; }

CommandBuffer::CommandBuffer() = default;

CommandBuffer::CommandBuffer(vk::CommandBuffer buf, vk::CommandPool pool): buf_(buf), pool_(pool) {}

CommandBuffer::operator vk::CommandBuffer() const { return buf_; }

CommandBuffer::operator bool() const { return buf_; }

Semaphore::Semaphore() = default;

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

Swapchain::Swapchain() = default;

Swapchain::Swapchain(vk::Format format, vk::Extent2D extent, vk::SwapchainKHR swapchain,
                     std::vector<vk::Image> const& images,
                     std::vector<vk::ImageView> const& image_views)
    : format_(format),
      extent_(extent),
      swapchain_(swapchain),
      images_(images),
      image_views_(image_views)
{
}

Swapchain::operator vk::SwapchainKHR() const { return swapchain_; }

Swapchain::operator bool() const { return swapchain_; }

Queue::Queue() = default;

Queue::Queue(vk::Queue queue, uint32_t family_index): queue_(queue), family_index_(family_index) {}

uint32_t Queue::getFamilyIndex() const { return family_index_; }

Queue::operator vk::Queue() const { return queue_; }

Queue::operator bool() const { return queue_; }

DescriptorSet::DescriptorSet() = default;

DescriptorSet::DescriptorSet(vk::DescriptorSet set, vk::DescriptorPool pool): set_(set), pool_(pool)
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

WriteDescriptorSet::WriteDescriptorSet(uint32_t binding, vk::DescriptorType type, Buffer& buffer)
{
    vk::DescriptorBufferInfo buf_info(buffer, 0, buffer.getSize());
    buffers_.emplace_back(buf_info);
    write_ = vk::WriteDescriptorSet({}, binding, 0, 1, type, nullptr, &buffers_.back());
}

WriteDescriptorSet::operator vk::WriteDescriptorSet() const { return write_; }

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
    std::string name = FSTR("UID #{}", id);
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
    surfaces_[id] = surface;
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
    ILOG("Device Name: {}", device_props.deviceName);
    DLOG("Device ID: 0x{:x}", device_props.deviceID);
    DLOG("Vendor ID: 0x{:x}", device_props.vendorID);
    ILOG("Device Type: {}", vk::to_string(device_props.deviceType));
    DLOG("Driver Version: {}.{}.{}", VK_VERSION_MAJOR(device_props.driverVersion),
         VK_VERSION_MINOR(device_props.driverVersion),
         VK_VERSION_PATCH(device_props.driverVersion));
    ILOG("Vulkan API Device Version: {}.{}.{}", VK_VERSION_MAJOR(device_props.apiVersion),
         VK_VERSION_MINOR(device_props.apiVersion), VK_VERSION_PATCH(device_props.apiVersion));
    ILOG("Vulkan API App Version: {}.{}.{}", VK_VERSION_MAJOR(MYVK_API_VERSION),
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
        DLOG("  Heap {}: Size = {} MB, Flags = {}", i, heap.size / (1024 * 1024),
             vk::to_string(heap.flags));
    }

    DLOG("Memory Types:");
    for (uint32_t i = 0; i < memory_props.memoryTypeCount; ++i) {
        auto const& type = memory_props.memoryTypes[i];
        DLOG("  Type {}: Heap Index = {}, Flags = {}", i, type.heapIndex,
             vk::to_string(type.propertyFlags));
    }

    // queue
    DLOG("====== Queue Families ======");
    for (uint32_t i = 0; i < queue_families.size(); ++i) {
        auto const& queue = queue_families[i];
        DLOG("Queue Family {}:", i);
        DLOG("  Queue Count: {}", queue.queueCount);
        DLOG("  Queue Flags: {}", vk::to_string(queue.queueFlags));
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

MyErrCode Context::createDeviceAndQueues(std::vector<char const*> const& extensions,
                                         std::map<Uid, QueuePicker> const& queue_pickers)
{
    std::vector<vk::DeviceQueueCreateInfo> queue_infos;
    std::map<Uid, uint32_t> family_indices;
    std::set<uint32_t> family_indices_dup;
    auto family_props = physical_device_.getQueueFamilyProperties();
    float const queue_priority = 1.0f;

    for (auto& [id, picker]: queue_pickers) {
        bool found = false;
        for (uint32_t i = 0; i < family_props.size(); ++i) {
            if (picker(i, family_props[i])) {
                if (family_indices_dup.find(i) != family_indices_dup.end()) {
                    ELOG("duplicated queue family index: {}", i);
                    return MyErrCode::kFailed;
                }
                family_indices_dup.insert(i);
                queue_infos.emplace_back(vk::DeviceQueueCreateFlags(), i, 1, &queue_priority);
                family_indices[id] = i;
                found = true;
                break;
            }
        }
        if (!found) {
            ELOG("no proper queue found");
            return MyErrCode::kFailed;
        }
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

    for (auto& [id, family_index]: family_indices) {
        ILOG("Queue[{}] Flags: {}", id, vk::to_string(family_props[family_index].queueFlags));
        auto queue = device_.getQueue(family_index, 0);
        queues_[id] = {queue, family_index};
        CHECK_ERR_RET(setDebugObjectId(queue, id));
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
    command_pools_[id] =
        CHECK_VKHPP_VAL(device_.createCommandPool({flags, getQueue(queue_id).getFamilyIndex()}));
    CHECK_ERR_RET(setDebugObjectId(command_pools_[id], id));
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
                                        std::map<vk::DescriptorType, uint32_t> const& size)
{
    if (descriptor_pools_.find(id) != descriptor_pools_.end()) {
        CHECK_ERR_RET(destroyDescriptorPool(id));
    }
    std::vector<vk::DescriptorPoolSize> pool_size;
    for (auto [type, count]: size) {
        pool_size.emplace_back(type, count);
    }
    descriptor_pools_[id] = CHECK_VKHPP_VAL(
        device_.createDescriptorPool({vk::DescriptorPoolCreateFlags(), max_sets, pool_size}));
    CHECK_ERR_RET(setDebugObjectId(descriptor_pools_[id], id));
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
    shader_modules_[id] = CHECK_VKHPP_VAL(device_.createShaderModule(create_info));
    CHECK_ERR_RET(setDebugObjectId(shader_modules_[id], id));
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

MyErrCode Context::createBuffer(Uid id, uint64_t size, vk::BufferUsageFlags usage,
                                vk::MemoryPropertyFlags properties, VmaAllocationCreateFlags flags)
{
    if (buffers_.find(id) != buffers_.end()) {
        CHECK_ERR_RET(destroyBuffer(id));
    }

    vk::BufferCreateInfo buffer_info(vk::BufferCreateFlags{}, size, usage,
                                     vk::SharingMode::eExclusive);

    VmaAllocationCreateInfo creation_info = {};
    creation_info.flags = flags;
    creation_info.requiredFlags = static_cast<VkMemoryPropertyFlags>(properties);

    VkBuffer buf;
    VmaAllocation alloc;
    CHECK_VK_RET(vmaCreateBuffer(allocator_, buffer_info, &creation_info, &buf, &alloc, nullptr));

    buffers_[id] = {size, buf, alloc, allocator_};
    CHECK_ERR_RET(setDebugObjectId(buffers_[id].buf_, id));
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

MyErrCode Context::createImage(Uid id, vk::Format format, vk::Extent2D extent,
                               vk::ImageTiling tiling, vk::ImageLayout initial_layout,
                               uint32_t mip_levels, vk::SampleCountFlagBits num_samples,
                               vk::ImageAspectFlags aspect_mask, vk::ImageUsageFlags usage,
                               vk::MemoryPropertyFlags properties, VmaAllocationCreateFlags flags)
{
    if (images_.find(id) != images_.end()) {
        CHECK_ERR_RET(destroyImage(id));
    }

    vk::ImageCreateInfo image_info(vk::ImageCreateFlags(), vk::ImageType::e2D, format,
                                   vk::Extent3D(extent, 1), mip_levels, 1, num_samples, tiling,
                                   usage, vk::SharingMode::eExclusive, {}, initial_layout);

    VmaAllocationCreateInfo vma_info = {};
    vma_info.flags = flags;
    vma_info.requiredFlags = static_cast<VkMemoryPropertyFlags>(properties);

    VkImage img;
    VmaAllocation alloc;
    VmaAllocationInfo alloc_info;
    CHECK_VK_RET(vmaCreateImage(allocator_, image_info, &vma_info, &img, &alloc, &alloc_info));

    vk::ImageViewCreateInfo view_info({}, img, vk::ImageViewType::e2D, format, {},
                                      {aspect_mask, 0, mip_levels, 0, 1});
    vk::ImageView img_view = CHECK_VKHPP_VAL(device_.createImageView(view_info));

    images_[id] = Image{format, extent, img, img_view, alloc, allocator_};
    CHECK_ERR_RET(setDebugObjectId(images_[id].img_, id));
    CHECK_ERR_RET(setDebugObjectId(images_[id].img_view_, id));
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
        vmaDestroyImage(it->second.allocator_, it->second.img_, it->second.alloc_);
        images_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("image not exist: {}", id);
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
    command_buffers_[id] = {bufs.front(), pool};
    CHECK_ERR_RET(setDebugObjectId(command_buffers_[id].buf_, id));
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
        device_.freeCommandBuffers(it->second.pool_, it->second.buf_);
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
    semaphores_[id] = {vk::SemaphoreType::eBinary, sem};
    CHECK_ERR_RET(setDebugObjectId(semaphores_[id].sem_, id));
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
    semaphores_[id] = {vk::SemaphoreType::eTimeline, sem};
    CHECK_ERR_RET(setDebugObjectId(semaphores_[id].sem_, id));
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
    fences_[id] = fence;
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

MyErrCode Context::createPipelineLayout(Uid id, std::vector<Uid> const& set_layout_ids)
{
    if (pipeline_layouts_.find(id) != pipeline_layouts_.end()) {
        CHECK_ERR_RET(destroyPipelineLayout(id));
    }
    std::vector<vk::DescriptorSetLayout> set_layouts;
    for (auto id: set_layout_ids) {
        set_layouts.push_back(getDescriptorSetLayout(id));
    }
    pipeline_layouts_[id] = CHECK_VKHPP_VAL(device_.createPipelineLayout({{}, set_layouts}));
    CHECK_ERR_RET(setDebugObjectId(pipeline_layouts_[id], id));
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
    pipelines_[id] = CHECK_VKHPP_VAL(device_.createComputePipeline({}, create_info));
    CHECK_ERR_RET(setDebugObjectId(pipelines_[id], id));
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
    descriptor_set_layouts_[id] = CHECK_VKHPP_VAL(
        device_.createDescriptorSetLayout({vk::DescriptorSetLayoutCreateFlags(), layout_bindings}));
    CHECK_ERR_RET(setDebugObjectId(descriptor_set_layouts_[id], id));
    return MyErrCode::kOk;
}

vk::DescriptorSetLayout& Context::getDescriptorSetLayout(Uid id)
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
    auto sets = CHECK_VKHPP_VAL(
        device_.allocateDescriptorSets({pool, 1, &getDescriptorSetLayout(set_layout_id)}));
    descriptor_sets_[id] = {sets.front(), pool};
    CHECK_ERR_RET(setDebugObjectId(descriptor_sets_[id].set_, id));
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

MyErrCode Context::updateDescriptorSet(Uid set_id, std::vector<WriteDescriptorSet> const& writes)
{
    auto& set = getDescriptorSet(set_id);
    std::vector<vk::WriteDescriptorSet> write_set;
    for (vk::WriteDescriptorSet w: writes) {
        w.setDstSet(set);
        write_set.push_back(w);
    }
    device_.updateDescriptorSets(write_set, {});
    return MyErrCode::kOk;
}

MyErrCode Context::createSwapchain(Uid id, Uid surface_id, vk::SurfaceFormatKHR surface_format,
                                   vk::Extent2D extent, vk::PresentModeKHR mode,
                                   vk::ImageUsageFlags usage)
{
    if (swapchains_.find(id) != swapchains_.end()) {
        CHECK_ERR_RET(destroySwapchain(id));
    }

    auto& surface = getSurface(surface_id);

    bool format_found = false;
    for (auto format: CHECK_VKHPP_VAL(physical_device_.getSurfaceFormatsKHR(surface))) {
        if (format == surface_format) {
            format_found = true;
        }
    }
    if (!format_found) {
        ELOG("surface format not found: {} {}", vk::to_string(surface_format.format),
             vk::to_string(surface_format.colorSpace));
        return MyErrCode::kFailed;
    }

    vk::SurfaceCapabilitiesKHR surface_caps =
        CHECK_VKHPP_VAL(physical_device_.getSurfaceCapabilitiesKHR(surface));

    if (surface_caps.currentExtent.width == std::numeric_limits<uint32_t>::max()) {
        extent.width = std::clamp(extent.width, surface_caps.minImageExtent.width,
                                  surface_caps.maxImageExtent.width);
        extent.height = std::clamp(extent.height, surface_caps.minImageExtent.height,
                                   surface_caps.maxImageExtent.height);
    } else {
        extent = surface_caps.currentExtent;
    }

    uint32_t image_count = surface_caps.minImageCount + 1;
    if (surface_caps.maxImageCount > 0) {
        image_count = std::min(image_count, surface_caps.maxImageCount);
    }

    vk::SwapchainCreateInfoKHR swapchain_info(
        {}, surface, image_count, surface_format.format, surface_format.colorSpace, extent, 1,
        usage, vk::SharingMode::eExclusive, {}, surface_caps.currentTransform,
        vk::CompositeAlphaFlagBitsKHR::eOpaque, mode, true);

    ILOG("create {} swapchain images with size={}x{}, format={}", image_count, extent.width,
         extent.height, vk::to_string(surface_format.format));

    Swapchain swapchain;
    swapchain.format_ = surface_format.format;
    swapchain.extent_ = extent;
    swapchain.swapchain_ = CHECK_VKHPP_VAL(device_.createSwapchainKHR(swapchain_info));

    for (auto& image: CHECK_VKHPP_VAL(device_.getSwapchainImagesKHR(swapchain))) {
        swapchain.images_.push_back(image);
    }

    vk::ImageViewCreateInfo view_info({}, {}, vk::ImageViewType::e2D, surface_format.format, {},
                                      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    for (auto image: swapchain.images_) {
        view_info.image = image;
        swapchain.image_views_.push_back(CHECK_VKHPP_VAL(device_.createImageView(view_info)));
    }

    swapchains_[id] = std::move(swapchain);
    CHECK_ERR_RET(setDebugObjectId(swapchains_[id].swapchain_, id));
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
        for (auto& view: it->second.image_views_) {
            device_.destroy(view);
        }
        device_.destroy(it->second.swapchain_);
        swapchains_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("swapchain not exist: {}", id);
        return MyErrCode::kFailed;
    }
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
    vk::CommandBuffer command_buffer = getCommandBuffer(command_buffer_id);
    CHECK_VKHPP_RET(command_buffer.begin({usage}));
    CHECK_ERR_RET(submitter(command_buffer));
    CHECK_VKHPP_RET(command_buffer.end());
    return MyErrCode::kOk;
}

MyErrCode Context::oneTimeSubmit(Uid queue_id, Uid command_pool_id, CmdSubmitter const& submitter)
{
    auto& command_pool = getCommandPool(command_pool_id);
    auto command_buffers = CHECK_VKHPP_VAL(
        device_.allocateCommandBuffers({command_pool, vk::CommandBufferLevel::ePrimary, 1}));
    auto command_buffers_guard =
        toolkit::scopeExit([&] { device_.freeCommandBuffers(command_pool, command_buffers); });

    auto& command_buffer = command_buffers.front();
    CHECK_VKHPP_RET(command_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
    CHECK_ERR_RET(submitter(command_buffer));
    CHECK_VKHPP_RET(command_buffer.end());

    vk::Queue queue = getQueue(queue_id);
    CHECK_VKHPP_RET(queue.submit(vk::SubmitInfo{{}, {}, command_buffer}, nullptr));
    CHECK_VKHPP_RET(queue.waitIdle());
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
    vk::CommandBuffer command_buffer = getCommandBuffer(command_buffer_id);
    vk::CommandBufferSubmitInfo cmdbuf_info = {command_buffer};
    vk::SubmitInfo2 submit_info = {vk::SubmitFlags{}, wait_sems, cmdbuf_info, signal_sems};
    CHECK_VKHPP_RET(
        queue.submit2(submit_info, fence_id != kUidNull ? getFence(fence_id) : vk::Fence{}));
    return MyErrCode::kOk;
}

MyErrCode Context::destroy()
{
    CHECK_VKHPP_RET(device_.waitIdle());
    while (!swapchains_.empty()) {
        CHECK_ERR_RET(destroySwapchain(swapchains_.begin()->first));
    }
    while (!pipelines_.empty()) {
        CHECK_ERR_RET(destroyPipeline(pipelines_.begin()->first));
    }
    while (!pipeline_layouts_.empty()) {
        CHECK_ERR_RET(destroyPipelineLayout(pipeline_layouts_.begin()->first));
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
    while (!images_.empty()) {
        CHECK_ERR_RET(destroyImage(images_.begin()->first));
    }
    while (!buffers_.empty()) {
        CHECK_ERR_RET(destroyBuffer(buffers_.begin()->first));
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
