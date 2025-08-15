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

#define CHECK_VKHPP_RET(err)                                    \
    do {                                                        \
        if (auto _err = (err); _err != vk::Result::eSuccess) {  \
            ELOG("failed to call vk: {}", vk::to_string(_err)); \
            return MyErrCode::kFailed;                          \
        }                                                       \
    } while (0)

#define CHECK_VKHPP_VAL(err)                                  \
    ({                                                        \
        auto MY_CONCAT(res_vk_, __LINE__) = (err);            \
        CHECK_VKHPP_RET(MY_CONCAT(res_vk_, __LINE__).result); \
        MY_CONCAT(res_vk_, __LINE__).value;                   \
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
    using TaskSubmitter = std::function<MyErrCode(vk::CommandBuffer&)>;

    MyErrCode createInstance(char const* name, std::vector<char const*> const& extensions);
    MyErrCode createPhysicalDevice(DevicePicker const& device_picker);
    MyErrCode createDevice(std::vector<char const*> const& extensions,
                           std::map<std::string, QueuePicker> const& queue_pickers);
    MyErrCode createCommandPools(std::map<std::string, vk::CommandPoolCreateFlags> const& flags);
    MyErrCode createDescriptorPool(vk::DescriptorPoolCreateInfo const& info);
    MyErrCode createAllocator();
    MyErrCode createBuffer(char const* name, uint64_t size, vk::BufferUsageFlags usage,
                           vk::MemoryPropertyFlags properties, VmaAllocationCreateFlags flags);
    MyErrCode createShaderModule(char const* name, std::filesystem::path const& file_path);
    MyErrCode createDescriptorSetLayout(char const* name,
                                        vk::DescriptorSetLayoutCreateInfo const& info);
    MyErrCode createPipelineLayout(char const* name, vk::PipelineLayoutCreateInfo const& info);
    MyErrCode createComputePipeline(char const* name, vk::ComputePipelineCreateInfo const& info);
    MyErrCode createDescriptorSet(char const* name, char const* set_layout_name);

    Queue& getQueue(char const* name);
    vk::CommandPool& getCommandPool(char const* name);
    Buffer& getBuffer(char const* name);
    vk::ShaderModule& getShaderModule(char const* name);
    vk::DescriptorSetLayout& getDescriptorSetLayout(char const* name);
    vk::PipelineLayout& getPipelineLayout(char const* name);
    vk::Pipeline& getPipeline(char const* name);
    vk::DescriptorSet& getDescriptorSet(char const* name);

    MyErrCode updateDescriptorSets(std::vector<vk::WriteDescriptorSet> const& writes);
    MyErrCode oneTimeSubmit(char const* queue_name, TaskSubmitter const& submitter);

    MyErrCode destroyBuffer(char const* name);
    MyErrCode destroyShaderModule(char const* name);
    MyErrCode destroyDescriptorSetLayout(char const* name);
    MyErrCode destroyPipelineLayout(char const* name);
    MyErrCode destroyPipeline(char const* name);
    MyErrCode destroyDescriptorSet(char const* name);
    MyErrCode destroy();

protected:
    static vk::DebugUtilsMessengerCreateInfoEXT getDebugMessengerInfo();
    static MyErrCode logDeviceInfo(vk::PhysicalDevice const& physical_device);

private:
    vk::Instance instance_;
    vk::DebugUtilsMessengerEXT debug_messenger_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    vk::DescriptorPool descriptor_pool_;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    std::map<std::string, Queue> queues_;
    std::map<std::string, vk::CommandPool> command_pools_;
    std::map<std::string, Buffer> buffers_;
    std::map<std::string, vk::ShaderModule> shader_modules_;
    std::map<std::string, vk::DescriptorSetLayout> descriptor_set_layouts_;
    std::map<std::string, vk::PipelineLayout> pipeline_layouts_;
    std::map<std::string, vk::Pipeline> pipelines_;
    std::map<std::string, vk::DescriptorSet> descriptor_sets_;
};

}  // namespace myvk
