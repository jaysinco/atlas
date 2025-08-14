#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#include "vulkan-helper.h"
#include "toolkit/toolkit.h"
#include "toolkit/logging.h"

#define MYVK_API_VERSION VK_MAKE_API_VERSION(0, MYVK_API_VERSION_MAJOR, MYVK_API_VERSION_MINOR, 0)

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace myvk
{

Buffer::Buffer(): buf_(VK_NULL_HANDLE), alloc_(VK_NULL_HANDLE) {}

Buffer::Buffer(VkBuffer buf, VmaAllocation alloc, VmaAllocationInfo const& alloc_info)
    : buf_(buf), alloc_(alloc), alloc_info_(alloc_info)
{
}

void* Buffer::getMappedData() const { return alloc_info_.pMappedData; }

Buffer::operator VkBuffer() const { return buf_; }

Buffer::operator vk::Buffer() const { return buf_; }

Buffer::operator bool() const { return buf_ != VK_NULL_HANDLE; }

Queue::Queue(): queue_(VK_NULL_HANDLE), family_index_(-1) {}

Queue::Queue(VkQueue queue, uint32_t family_index): queue_(queue), family_index_(family_index) {}

uint32_t Queue::getFamilyIndex() const { return family_index_; }

Queue::operator VkQueue() const { return queue_; }

Queue::operator vk::Queue() const { return queue_; }

Queue::operator bool() const { return queue_ != VK_NULL_HANDLE; }

static VkBool32 debugMessengerUserCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type,
    vk::DebugUtilsMessengerCallbackDataEXT const* callback_data, void* user_data)
{
    switch (severity) {
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
            TLOG("{}", callback_data->pMessage);
            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
            DLOG("{}", callback_data->pMessage);
            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
            WLOG("{}", callback_data->pMessage);
            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
            ELOG("{}", callback_data->pMessage);
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

MyErrCode Context::createInstance(char const* name, std::vector<char const*> const& extensions)
{
    VULKAN_HPP_DEFAULT_DISPATCHER.init();
    vk::ApplicationInfo app_info{name, VK_MAKE_VERSION(0, 1, 0), "No Engine",
                                 VK_MAKE_VERSION(0, 1, 0), MYVK_API_VERSION};
    auto debug_info = getDebugMessengerInfo();
    instance_ = CHECK_VKHPP_RET(
        vk::createInstance({vk::InstanceCreateFlags(), &app_info, {}, extensions, &debug_info}));

    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);

    debug_messenger_ = CHECK_VKHPP_RET(instance_.createDebugUtilsMessengerEXT(debug_info));

    return MyErrCode::kOk;
}

MyErrCode Context::createPhysicalDevice(DevicePicker const& device_picker)
{
    auto physical_devices = CHECK_VKHPP_RET(instance_.enumeratePhysicalDevices());
    bool found = false;
    for (auto& d: physical_devices) {
        if (device_picker(d.getProperties(), d.getFeatures())) {
            physical_device_ = d;
            found = true;
            break;
        }
    }
    if (!found) {
        ELOG("no proper device found");
        return MyErrCode::kFailed;
    }

    auto device_props = physical_device_.getProperties();
    auto device_limits = device_props.limits;
    ILOG("Device Name: {}", device_props.deviceName);
    ILOG("Vulkan Version: {}.{}.{}", VK_VERSION_MAJOR(device_props.apiVersion),
         VK_VERSION_MINOR(device_props.apiVersion), VK_VERSION_PATCH(device_props.apiVersion));
    ILOG("Max Compute Shared Memory Size: {} KB", device_limits.maxComputeSharedMemorySize / 1024);

    return MyErrCode::kOk;
}

MyErrCode Context::createDevice(std::vector<QueuePicker> const& queue_pickers,
                                std::vector<char const*> const& extensions)
{
    auto queue_props = physical_device_.getQueueFamilyProperties();
    float const queue_priority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queue_infos;
    for (auto& picker: queue_pickers) {
        bool found = false;
        for (uint32_t i = 0; i < queue_props.size(); ++i) {
            if (picker(queue_props[i])) {
                queue_infos.emplace_back(vk::DeviceQueueCreateFlags(), i, 1, &queue_priority);
                found = true;
                break;
            }
        }
        if (!found) {
            ELOG("no proper queue found");
            return MyErrCode::kFailed;
        }
    }

    device_ = CHECK_VKHPP_RET(
        physical_device_.createDevice({vk::DeviceCreateFlags(), queue_infos, {}, extensions}));

    VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);

    for (auto& info: queue_infos) {
        auto family_index = info.queueFamilyIndex;
        auto queue = device_.getQueue(family_index, 0);
        queues_.emplace_back(queue, family_index);
    }

    return MyErrCode::kOk;
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

MyErrCode Context::createCommandPools(std::vector<vk::CommandPoolCreateFlags> const& flags)
{
    for (int i = 0; i < flags.size(); ++i) {
        command_pools_.push_back(
            CHECK_VKHPP_RET(device_.createCommandPool({flags[i], getQueue(i).getFamilyIndex()})));
    }
    return MyErrCode::kOk;
}

MyErrCode Context::createDescriptorPool(uint32_t max_sets,
                                        std::vector<vk::DescriptorPoolSize> const& pool_sizes)
{
    descriptor_pool_ = CHECK_VKHPP_RET(
        device_.createDescriptorPool({vk::DescriptorPoolCreateFlags(), max_sets, pool_sizes}));
    return MyErrCode::kOk;
}

MyErrCode Context::destroy()
{
    for (auto& [_, i]: buffers_) {
        vmaDestroyBuffer(allocator_, i.buf_, i.alloc_);
    }
    for (auto& [_, i]: shader_modules_) {
        device_.destroyShaderModule(i);
    }
    if (descriptor_pool_) {
        device_.destroyDescriptorPool(descriptor_pool_);
    }
    for (auto& i: command_pools_) {
        device_.destroyCommandPool(i);
    }
    if (allocator_) {
        vmaDestroyAllocator(allocator_);
    }
    if (device_) {
        device_.destroy();
    }
    if (debug_messenger_) {
        instance_.destroyDebugUtilsMessengerEXT(debug_messenger_);
    }
    if (instance_) {
        instance_.destroy();
    }
    return MyErrCode::kOk;
}

Queue& Context::getQueue(int i) { return queues_.at(i); }

vk::CommandPool& Context::getCommandPool(int i) { return command_pools_.at(i); };

MyErrCode Context::createShaderModule(char const* name, std::filesystem::path const& spv_path)
{
    if (shader_modules_.find(name) != shader_modules_.end()) {
        CHECK_ERR_RET(destroyShaderModule(name));
    }
    std::vector<uint8_t> code;
    CHECK_ERR_RET(toolkit::readBinaryFile(spv_path, code));
    vk::ShaderModuleCreateInfo create_info(vk::ShaderModuleCreateFlags(), code.size(),
                                           reinterpret_cast<uint32_t const*>(code.data()));
    shader_modules_[name] = CHECK_VKHPP_RET(device_.createShaderModule(create_info));
    return MyErrCode::kOk;
}

vk::ShaderModule& Context::getShaderModule(char const* name) { return shader_modules_.at(name); }

MyErrCode Context::destroyShaderModule(char const* name)
{
    if (auto it = shader_modules_.find(name); it != shader_modules_.end()) {
        device_.destroyShaderModule(it->second);
        shader_modules_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("name not exist: {}", name);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createBuffer(char const* name, uint64_t size, vk::BufferUsageFlags usage,
                                vk::MemoryPropertyFlags properties, VmaAllocationCreateFlags flags)
{
    if (buffers_.find(name) != buffers_.end()) {
        CHECK_ERR_RET(destroyBuffer(name));
    }

    vk::BufferCreateInfo buffer_info(vk::BufferCreateFlags{}, size, usage,
                                     vk::SharingMode::eExclusive);

    VmaAllocationCreateInfo creation_info = {};
    creation_info.flags = flags;
    creation_info.requiredFlags = static_cast<VkMemoryPropertyFlags>(properties);

    VkBuffer buf;
    VmaAllocation alloc;
    VmaAllocationInfo alloc_info;
    CHECK_VK_RET(
        vmaCreateBuffer(allocator_, buffer_info, &creation_info, &buf, &alloc, &alloc_info));

    buffers_[name] = {buf, alloc, alloc_info};
    return MyErrCode::kOk;
}

Buffer& Context::getBuffer(char const* name) { return buffers_.at(name); }

MyErrCode Context::destroyBuffer(char const* name)
{
    if (auto it = buffers_.find(name); it != buffers_.end()) {
        vmaDestroyBuffer(allocator_, it->second.buf_, it->second.alloc_);
        buffers_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("name not exist: {}", name);
        return MyErrCode::kFailed;
    }
}

}  // namespace myvk
