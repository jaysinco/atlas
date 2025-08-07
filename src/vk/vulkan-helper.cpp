#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#include "vulkan-helper.h"
#include "toolkit/toolkit.h"

namespace toolkit::myvk
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

Allocator::Allocator(): allocator_(VK_NULL_HANDLE) {}

Allocator::Allocator(VmaAllocator allocator): allocator_(allocator) {}

void Allocator::destroy() { vmaDestroyAllocator(allocator_); }

Allocator::operator VmaAllocator() const { return allocator_; }

Allocator::operator bool() const { return allocator_ != VK_NULL_HANDLE; }

Buffer Allocator::createBuffer(uint64_t size, vk::BufferUsageFlags usage,
                               vk::MemoryPropertyFlags properties, VmaAllocationCreateFlags flags)
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
        vmaCreateBuffer(*this, buffer_info, &creation_info, &buf, &alloc, &alloc_info));

    return {buf, alloc, alloc_info};
}

void Allocator::destroyBuffer(Buffer& buffer)
{
    vmaDestroyBuffer(*this, buffer.buf_, buffer.alloc_);
}

Allocator createAllocator(vk::PhysicalDevice physical_device, vk::Device device,
                          vk::Instance instance)
{
    VmaAllocatorCreateInfo create_info = {};
    create_info.vulkanApiVersion = MYVK_API_VERSION;
    create_info.physicalDevice = physical_device;
    create_info.device = device;
    create_info.instance = instance;

    VmaAllocator allocator;
    CHECK_VK_ERR_THROW(vmaCreateAllocator(&create_info, &allocator));
    return allocator;
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
