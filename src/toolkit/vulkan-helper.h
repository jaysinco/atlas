#pragma once
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include "format.h"

#define CHECK_VK_ERR_THROW(err)                          \
    do {                                                 \
        if (auto _err = (err); _err != VK_SUCCESS) {     \
            MY_THROW("failed to call vulkan: {}", _err); \
        }                                                \
    } while (0)

namespace toolkit::myvk
{

class Allocator
{
public:
    Allocator();
    Allocator(VmaAllocator allocator);
    void destory();
    operator VmaAllocator() const;
    operator bool() const;

    static Allocator getDefault();
    static void setDefault(Allocator allocator);

private:
    VmaAllocator allocator_;
    static Allocator default_allocator;
};

class Buffer
{
public:
    Buffer();
    Buffer(VkBuffer buf, VmaAllocation alloc, VmaAllocationInfo const& alloc_info,
           VmaAllocator allocator);
    void destory();
    void* getMappedData() const;
    operator VkBuffer() const;
    operator vk::Buffer() const;
    operator bool() const;

private:
    VkBuffer buf_;
    VmaAllocation alloc_;
    VmaAllocationInfo alloc_info_;
    VmaAllocator allocator_;
};

void loadInstanceProcAddr(vk::Instance instance);
void loadDeviceProcAddr(vk::Device device);
Allocator createAllocator(uint32_t vk_api_version, vk::PhysicalDevice physical_device,
                          vk::Device device, vk::Instance instance, bool set_default = true);
Buffer createBuffer(uint64_t size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                    VmaAllocationCreateFlags flags, Allocator allocator = Allocator::getDefault());
vk::ShaderModule createShaderModule(vk::Device device, std::filesystem::path const& spv);

}  // namespace toolkit::myvk

namespace myvk = toolkit::myvk;
