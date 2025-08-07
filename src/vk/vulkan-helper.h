#pragma once
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include "toolkit/format.h"

#define MYVK_API_VERSION VK_MAKE_API_VERSION(0, MYVK_API_VERSION_MAJOR, MYVK_API_VERSION_MINOR, 0)
#define CHECK_VK_ERR_THROW(err)                          \
    do {                                                 \
        if (auto _err = (err); _err != VK_SUCCESS) {     \
            MY_THROW("failed to call vulkan: {}", _err); \
        }                                                \
    } while (0)

namespace toolkit::myvk
{

class Buffer
{
public:
    Buffer();
    Buffer(VkBuffer buf, VmaAllocation alloc, VmaAllocationInfo const& alloc_info);
    void* getMappedData() const;
    operator VkBuffer() const;
    operator vk::Buffer() const;
    operator bool() const;

private:
    friend class Allocator;
    VkBuffer buf_;
    VmaAllocation alloc_;
    VmaAllocationInfo alloc_info_;
};

class Allocator
{
public:
    Allocator();
    Allocator(VmaAllocator allocator);
    void destroy();
    operator VmaAllocator() const;
    operator bool() const;

    Buffer createBuffer(uint64_t size, vk::BufferUsageFlags usage,
                        vk::MemoryPropertyFlags properties, VmaAllocationCreateFlags flags);
    void destroyBuffer(Buffer& buffer);

private:
    VmaAllocator allocator_;
};

void loadInstanceProcAddr(vk::Instance instance);
void loadDeviceProcAddr(vk::Device device);
Allocator createAllocator(vk::PhysicalDevice physical_device, vk::Device device,
                          vk::Instance instance);
vk::ShaderModule createShaderModule(vk::Device device, std::filesystem::path const& spv);

}  // namespace toolkit::myvk

namespace myvk = toolkit::myvk;
