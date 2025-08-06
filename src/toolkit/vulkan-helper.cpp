#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#include "vulkan-helper.h"

namespace toolkit::myvk
{

Allocator Allocator::default_a;

Allocator::Allocator(): allocator_(VK_NULL_HANDLE) {}

Allocator::Allocator(VmaAllocator allocator): allocator_(allocator) {}

void Allocator::destory() { vmaDestroyAllocator(allocator_); }

Allocator::operator VmaAllocator() const { return allocator_; }

Allocator::operator bool() const { return allocator_ != VK_NULL_HANDLE; }

Allocator Allocator::getDefault() { return default_a; }

void Allocator::setDefault(Allocator allocator)
{
    if (default_a) {
        default_a.destory();
    }
    default_a = allocator;
}

Buffer::Buffer(): buf_(VK_NULL_HANDLE), alloc_(VK_NULL_HANDLE), allocator_(VK_NULL_HANDLE) {}

Buffer::Buffer(VkBuffer buf, VmaAllocation alloc, VmaAllocationInfo const& alloc_info,
               VmaAllocator allocator)
    : buf_(buf), alloc_(alloc), alloc_info_(alloc_info), allocator_(allocator)
{
}

void Buffer::destory() { vmaDestroyBuffer(allocator_, buf_, alloc_); }

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

}  // namespace toolkit::myvk
