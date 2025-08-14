#pragma once
#define VULKAN_HPP_NO_NODISCARD_WARNINGS
#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include "toolkit/format.h"
#include <map>
#include <functional>

#define CHECK_VK_RET(err)                            \
    do {                                             \
        if (auto _err = (err); _err != VK_SUCCESS) { \
            ELOG("failed to call vk: {}", _err);     \
            return MyErrCode::kFailed;               \
        }                                            \
    } while (0)

#define CHECK_VKHPP_RET(err)                                                                   \
    ({                                                                                         \
        auto MY_CONCAT(res_vk_, __LINE__) = (err);                                             \
        if (MY_CONCAT(res_vk_, __LINE__).result != vk::Result::eSuccess) {                     \
            ELOG("failed to call vk: {}", vk::to_string(MY_CONCAT(res_vk_, __LINE__).result)); \
            return MyErrCode::kFailed;                                                         \
        }                                                                                      \
        MY_CONCAT(res_vk_, __LINE__).value;                                                    \
    })

namespace myvk
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
    friend class Context;
    VkBuffer buf_;
    VmaAllocation alloc_;
    VmaAllocationInfo alloc_info_;
};

class Context
{
public:
    MyErrCode createInstance(char const* name, std::vector<char const*> const& extensions);
    MyErrCode createPhysicalDevice(
        std::function<bool(vk::PhysicalDeviceProperties const&)> const& picker);
    MyErrCode createDevice(
        std::function<bool(vk::QueueFamilyProperties const&)> const& queue_picker,
        std::vector<char const*> const& extensions);
    MyErrCode createAllocator();
    MyErrCode destroy();

    MyErrCode createShaderModule(char const* name, std::filesystem::path const& spv_path);
    vk::ShaderModule& getShaderModule(char const* name);
    MyErrCode destroyShaderModule(char const* name);

    MyErrCode createBuffer(char const* name, uint64_t size, vk::BufferUsageFlags usage,
                           vk::MemoryPropertyFlags properties, VmaAllocationCreateFlags flags);
    Buffer& getBuffer(char const* name);
    MyErrCode destroyBuffer(char const* name);

protected:
    vk::DebugUtilsMessengerCreateInfoEXT getDebugMessengerInfo();
    MyErrCode destroyShaderModule(vk::ShaderModule& shader_module);
    MyErrCode destroyBuffer(Buffer& buffer);

private:
    vk::Instance instance_;
    vk::DebugUtilsMessengerEXT debug_messenger_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    uint32_t queue_family_index_;
    vk::Queue queue_;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    std::map<std::string, vk::ShaderModule> shader_modules_;
    std::map<std::string, Buffer> buffers_;
};

}  // namespace myvk
