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

class Queue
{
public:
    Queue();
    Queue(VkQueue queue, uint32_t family_index);
    uint32_t getFamilyIndex() const;
    operator VkQueue() const;
    operator vk::Queue() const;
    operator bool() const;

private:
    friend class Context;
    VkQueue queue_;
    uint32_t family_index_;
};

class Context
{
public:
    using DevicePicker =
        std::function<bool(vk::PhysicalDeviceProperties const&, vk::PhysicalDeviceFeatures const&)>;
    using QueuePicker = std::function<bool(vk::QueueFamilyProperties const&)>;

    MyErrCode createInstance(char const* name, std::vector<char const*> const& extensions);
    MyErrCode createPhysicalDevice(DevicePicker const& device_picker);
    MyErrCode createDevice(std::vector<QueuePicker> const& queue_pickers,
                           std::vector<char const*> const& extensions);
    MyErrCode createAllocator();
    MyErrCode createCommandPools(std::vector<vk::CommandPoolCreateFlags> const& flags);
    MyErrCode createDescriptorPool(uint32_t max_sets,
                                   std::vector<vk::DescriptorPoolSize> const& pool_sizes);

    Queue& getQueue(int i = 0);
    vk::CommandPool& getCommandPool(int i = 0);
    MyErrCode destroy();

    MyErrCode createBuffer(char const* name, uint64_t size, vk::BufferUsageFlags usage,
                           vk::MemoryPropertyFlags properties, VmaAllocationCreateFlags flags);
    Buffer& getBuffer(char const* name);
    MyErrCode destroyBuffer(char const* name);

    MyErrCode createShaderModule(char const* name, std::filesystem::path const& spv_path);
    vk::ShaderModule& getShaderModule(char const* name);
    MyErrCode destroyShaderModule(char const* name);

    MyErrCode createDescriptorSetLayout(
        char const* name, std::vector<vk::DescriptorSetLayoutBinding> const& bindings);
    vk::DescriptorSetLayout& getDescriptorSetLayout(char const* name);
    MyErrCode destroyDescriptorSetLayout(char const* name);

    MyErrCode createPipelineLayout(char const* name, std::vector<char const*> const& set_layouts);
    vk::PipelineLayout& getPipelineLayout(char const* name);
    MyErrCode destroyPipelineLayout(char const* name);

    MyErrCode createComputePipeline(char const* name);
    vk::PipelineLayout& getPipeline(char const* name);
    MyErrCode destroyPipeline(char const* name);

protected:
    vk::DebugUtilsMessengerCreateInfoEXT getDebugMessengerInfo();

private:
    vk::Instance instance_;
    vk::DebugUtilsMessengerEXT debug_messenger_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    std::vector<Queue> queues_;
    std::vector<vk::CommandPool> command_pools_;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    vk::DescriptorPool descriptor_pool_;

    std::map<std::string, Buffer> buffers_;
    std::map<std::string, vk::ShaderModule> shader_modules_;
    std::map<std::string, vk::DescriptorSetLayout> descriptor_set_layouts_;
    std::map<std::string, vk::PipelineLayout> pipeline_layouts_;
    std::map<std::string, vk::Pipeline> pipelines_;
};

}  // namespace myvk
