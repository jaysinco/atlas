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
    std::vector<char const*> const layers = {};
    instance_ = CHECK_VKHPP_RET(
        vk::createInstance({vk::InstanceCreateFlags(), &app_info, layers, extensions}));

    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);

    debug_messenger_ =
        CHECK_VKHPP_RET(instance_.createDebugUtilsMessengerEXT(getDebugMessengerInfo()));

    return kOk;
}

MyErrCode Context ::createPhysicalDevice(
    std::function<bool(vk::PhysicalDeviceProperties const&)> const& picker)
{
    auto physical_devices = CHECK_VKHPP_RET(instance_.enumeratePhysicalDevices());
    for (auto& d: physical_devices) {
        if (picker(d.getProperties())) {
            physical_device_ = d;
            break;
        }
    }

    auto device_props = physical_device_.getProperties();
    auto device_limits = device_props.limits;
    ILOG("Device Name: {}", device_props.deviceName);
    ILOG("Vulkan Version: {}.{}.{}", VK_VERSION_MAJOR(device_props.apiVersion),
         VK_VERSION_MINOR(device_props.apiVersion), VK_VERSION_PATCH(device_props.apiVersion));
    ILOG("Max Compute Shared Memory Size: {} KB", device_limits.maxComputeSharedMemorySize / 1024);

    return kOk;
}

MyErrCode Context::createDevice(
    std::function<bool(vk::QueueFamilyProperties const&)> const& queue_picker,
    std::vector<char const*> const& extensions)
{
    auto queue_props = physical_device_.getQueueFamilyProperties();
    auto queue_picked = std::find_if(queue_props.begin(), queue_props.end(), queue_picker);
    queue_family_index_ = std::distance(queue_props.begin(), queue_picked);
    ILOG("Compute Queue Family Index: {}", queue_family_index_);

    float const queue_priority = 1.0f;
    vk::DeviceQueueCreateInfo queue_create_info(vk::DeviceQueueCreateFlags(), queue_family_index_,
                                                1, &queue_priority);
    device_ = CHECK_VKHPP_RET(physical_device_.createDevice(
        {vk::DeviceCreateFlags(), queue_create_info, {}, extensions}));

    VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);
    queue_ = device_.getQueue(queue_family_index_, 0);

    return kOk;
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

MyErrCode Context::destroy()
{
    for (auto [_, buf]: buffers_) {
        vmaDestroyBuffer(allocator_, buf.buf_, buf.alloc_);
    }
    for (auto [_, shm]: shader_modules_) {
        device_.destroyShaderModule(shm);
    }
    vmaDestroyAllocator(allocator_);
    device_.destroy();
    instance_.destroyDebugUtilsMessengerEXT(debug_messenger_);
    instance_.destroy();
    return kOk;
}

MyErrCode Context::createShaderModule(char const* name, std::filesystem::path const& spv_path)
{
    if (shader_modules_.find(name) != shader_modules_.end()) {
        ELOG("duplicated name: {}", name);
        return MyErrCode::kFailed;
    }
    std::vector<uint8_t> code;
    CHECK_ERR_RET(toolkit::readBinaryFile(spv_path, code));
    vk::ShaderModuleCreateInfo create_info(vk::ShaderModuleCreateFlags(), code.size(),
                                           reinterpret_cast<uint32_t const*>(code.data()));
    shader_modules_[name] = CHECK_VKHPP_RET(device.createShaderModule(create_info));
    return MyErrCode::kOk;
}

vk::ShaderModule& Context::getShaderModule(char const* name) {}

MyErrCode Context::destroyShaderModule(char const* name) {}

MyErrCode Context::destroyShaderModule(vk::ShaderModule& shader_module) {}

MyErrCode Context::createBuffer(char const* name, uint64_t size, vk::BufferUsageFlags usage,
                                vk::MemoryPropertyFlags properties, VmaAllocationCreateFlags flags)
{
}

Buffer& Context::getBuffer(char const* name) {}

MyErrCode Context::destroyBuffer(char const* name) {}

MyErrCode Context::destroyBuffer(Buffer& buffer) {}

// void Allocator::destroy() {  }
//
//
//
// MyErrCode Allocator::createBuffer(uint64_t size, vk::BufferUsageFlags usage,
//                                   vk::MemoryPropertyFlags properties,
//                                   VmaAllocationCreateFlags flags, Buffer& buffer)
// {
//     vk::BufferCreateInfo buffer_info(vk::BufferCreateFlags{}, size, usage,
//                                      vk::SharingMode::eExclusive);
//
//     VmaAllocationCreateInfo creation_info = {};
//     creation_info.flags = flags;
//     creation_info.requiredFlags = static_cast<VkMemoryPropertyFlags>(properties);
//
//     VkBuffer buf;
//     VmaAllocation alloc;
//     VmaAllocationInfo alloc_info;
//     CHECK_VK_RET(vmaCreateBuffer(*this, buffer_info, &creation_info, &buf, &alloc, &alloc_info));
//
//     buffer = {buf, alloc, alloc_info};
//     return MyErrCode::kOk;
// }
//
// void Allocator::destroyBuffer(Buffer& buffer)
// {
//     vmaDestroyBuffer(*this, buffer.buf_, buffer.alloc_);
// }

// MyErrCode createAllocator(vk::PhysicalDevice physical_device, vk::Device device,
//                           vk::Instance instance, Allocator& allocator)
// {
// }
//
// MyErrCode createShaderModule(vk::Device device, std::filesystem::path const& spv_path,
//                              vk::ShaderModule& shader_module)
// {
// }

}  // namespace myvk
