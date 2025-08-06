#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#include "vulkan-helper.h"
#include "toolkit.h"

namespace toolkit::myvk
{

Allocator Allocator::default_allocator;

Allocator::Allocator(): allocator_(VK_NULL_HANDLE) {}

Allocator::Allocator(VmaAllocator allocator): allocator_(allocator) {}

void Allocator::destory() { vmaDestroyAllocator(allocator_); }

Allocator::operator VmaAllocator() const { return allocator_; }

Allocator::operator bool() const { return allocator_ != VK_NULL_HANDLE; }

Allocator Allocator::getDefault() { return default_allocator; }

void Allocator::setDefault(Allocator allocator)
{
    if (default_allocator) {
        default_allocator.destory();
    }
    default_allocator = allocator;
}

Buffer::Buffer(): buf_(VK_NULL_HANDLE), alloc_(VK_NULL_HANDLE), allocator_(VK_NULL_HANDLE) {}

Buffer::Buffer(VkBuffer buf, VmaAllocation alloc, VmaAllocationInfo const& alloc_info,
               VmaAllocator allocator)
    : buf_(buf), alloc_(alloc), alloc_info_(alloc_info), allocator_(allocator)
{
}

void Buffer::destory() { vmaDestroyBuffer(allocator_, buf_, alloc_); }

void* Buffer::getMappedData() const { return alloc_info_.pMappedData; }

Buffer::operator VkBuffer() const { return buf_; }

Buffer::operator vk::Buffer() const { return buf_; }

Buffer::operator bool() const { return buf_ != VK_NULL_HANDLE; }

Allocator createAllocator(uint32_t vk_api_version, vk::PhysicalDevice physical_device,
                          vk::Device device, vk::Instance instance, bool set_default)
{
    VmaAllocatorCreateInfo create_info = {};
    create_info.vulkanApiVersion = vk_api_version;
    create_info.physicalDevice = physical_device;
    create_info.device = device;
    create_info.instance = instance;

    VmaAllocator allocator;
    CHECK_VK_ERR_THROW(vmaCreateAllocator(&create_info, &allocator));
    if (set_default) {
        Allocator::setDefault(allocator);
    }

    return allocator;
}

Buffer createBuffer(uint64_t size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                    VmaAllocationCreateFlags flags, Allocator allocator)
{
    vk::BufferCreateInfo buffer_info(vk::BufferCreateFlags{}, size, usage,
                                     vk::SharingMode::eExclusive);

    VmaAllocationCreateInfo creation_info = {};
    creation_info.flags = flags;
    creation_info.requiredFlags = static_cast<VkMemoryPropertyFlags>(properties);

    VkBuffer buf;
    VmaAllocation alloc;
    VmaAllocationInfo alloc_info;
    CHECK_VK_ERR_THROW(
        vmaCreateBuffer(allocator, buffer_info, &creation_info, &buf, &alloc, &alloc_info));

    return {buf, alloc, alloc_info, allocator};
}

vk::ShaderModule createShaderModule(vk::Device device, std::filesystem::path const& spv)
{
    std::vector<uint8_t> code;
    CHECK_ERR_THROW(toolkit::readBinaryFile(spv, code));
    vk::ShaderModuleCreateInfo create_info(vk::ShaderModuleCreateFlags(), code.size(),
                                           reinterpret_cast<uint32_t const*>(code.data()));
    return device.createShaderModule(create_info);
}

// NOLINTBEGIN

#define FOR_EACH_INSTANCE_PROC        \
    X(vkCreateDebugUtilsMessengerEXT) \
    X(vkDestroyDebugUtilsMessengerEXT)

#define X(_id) static PFN_##_id _id = nullptr;
FOR_EACH_INSTANCE_PROC
#undef X

void loadInstanceProcAddr(vk::Instance instance)
{
#define X(_id) _id = ((PFN_##_id)(vkGetInstanceProcAddr(instance, #_id)));
    FOR_EACH_INSTANCE_PROC
#undef X
}

void loadDeviceProcAddr(vk::Device device) {}

}  // namespace toolkit::myvk

VkResult vkCreateDebugUtilsMessengerEXT(VkInstance instance,
                                        VkDebugUtilsMessengerCreateInfoEXT const* pCreateInfo,
                                        VkAllocationCallbacks const* pAllocator,
                                        VkDebugUtilsMessengerEXT* pMessenger)
{
    return myvk::vkCreateDebugUtilsMessengerEXT(instance, pCreateInfo, pAllocator, pMessenger);
}

void vkDestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT messenger,
                                     VkAllocationCallbacks const* pAllocator)
{
    return myvk::vkDestroyDebugUtilsMessengerEXT(instance, messenger, pAllocator);
}

// NOLINTEND
