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
    instance_ = CHECK_VKHPP_VAL(
        vk::createInstance({vk::InstanceCreateFlags(), &app_info, {}, extensions, &debug_info}));

    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);

    debug_messenger_ = CHECK_VKHPP_VAL(instance_.createDebugUtilsMessengerEXT(debug_info));

    return MyErrCode::kOk;
}

MyErrCode Context::createPhysicalDevice(DevicePicker const& device_picker)
{
    auto physical_devices = CHECK_VKHPP_VAL(instance_.enumeratePhysicalDevices());
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

    device_ = CHECK_VKHPP_VAL(
        physical_device_.createDevice({vk::DeviceCreateFlags(), queue_infos, {}, extensions}));

    VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);

    for (auto& info: queue_infos) {
        auto family_index = info.queueFamilyIndex;
        auto queue = device_.getQueue(family_index, 0);
        queues_.emplace_back(queue, family_index);
    }

    return MyErrCode::kOk;
}

MyErrCode Context::createCommandPools(std::vector<vk::CommandPoolCreateFlags> const& flags)
{
    if (flags.size() != queues_.size()) {
        ELOG("flags size must be {}", queues_.size());
        return MyErrCode::kFailed;
    }
    for (int i = 0; i < flags.size(); ++i) {
        command_pools_.push_back(
            CHECK_VKHPP_VAL(device_.createCommandPool({flags[i], getQueue(i).getFamilyIndex()})));
    }
    return MyErrCode::kOk;
}

MyErrCode Context::createDescriptorPool(vk::DescriptorPoolCreateInfo const& info)
{
    descriptor_pool_ = CHECK_VKHPP_VAL(device_.createDescriptorPool(info));
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

Queue& Context::getQueue(int i) { return queues_.at(i); }

vk::CommandPool& Context::getCommandPool(int i) { return command_pools_.at(i); };

MyErrCode Context::createShaderModule(char const* name, std::filesystem::path const& file_path)
{
    if (shader_modules_.find(name) != shader_modules_.end()) {
        CHECK_ERR_RET(destroyShaderModule(name));
    }
    std::vector<uint8_t> code;
    CHECK_ERR_RET(toolkit::readBinaryFile(file_path, code));
    vk::ShaderModuleCreateInfo create_info(vk::ShaderModuleCreateFlags(), code.size(),
                                           reinterpret_cast<uint32_t const*>(code.data()));
    shader_modules_[name] = CHECK_VKHPP_VAL(device_.createShaderModule(create_info));
    return MyErrCode::kOk;
}

vk::ShaderModule& Context::getShaderModule(char const* name) { return shader_modules_.at(name); }

MyErrCode Context::destroyShaderModule(char const* name)
{
    if (auto it = shader_modules_.find(name); it != shader_modules_.end()) {
        device_.destroy(it->second);
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

MyErrCode Context::createDescriptorSetLayout(char const* name,
                                             vk::DescriptorSetLayoutCreateInfo const& info)
{
    if (descriptor_set_layouts_.find(name) != descriptor_set_layouts_.end()) {
        CHECK_ERR_RET(destroyDescriptorSetLayout(name));
    }
    descriptor_set_layouts_[name] = CHECK_VKHPP_VAL(device_.createDescriptorSetLayout(info));
    return MyErrCode::kOk;
}

vk::DescriptorSetLayout& Context::getDescriptorSetLayout(char const* name)
{
    return descriptor_set_layouts_.at(name);
}

MyErrCode Context::destroyDescriptorSetLayout(char const* name)
{
    if (auto it = descriptor_set_layouts_.find(name); it != descriptor_set_layouts_.end()) {
        device_.destroy(it->second);
        descriptor_set_layouts_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("name not exist: {}", name);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createPipelineLayout(char const* name, vk::PipelineLayoutCreateInfo const& info)
{
    if (pipeline_layouts_.find(name) != pipeline_layouts_.end()) {
        CHECK_ERR_RET(destroyPipelineLayout(name));
    }
    pipeline_layouts_[name] = CHECK_VKHPP_VAL(device_.createPipelineLayout(info));
    return MyErrCode::kOk;
}

vk::PipelineLayout& Context::getPipelineLayout(char const* name)
{
    return pipeline_layouts_.at(name);
}

MyErrCode Context::destroyPipelineLayout(char const* name)
{
    if (auto it = pipeline_layouts_.find(name); it != pipeline_layouts_.end()) {
        device_.destroy(it->second);
        pipeline_layouts_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("name not exist: {}", name);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createComputePipeline(char const* name,
                                         vk::ComputePipelineCreateInfo const& info)
{
    if (pipelines_.find(name) != pipelines_.end()) {
        CHECK_ERR_RET(destroyPipeline(name));
    }
    pipelines_[name] = CHECK_VKHPP_VAL(device_.createComputePipeline({}, info));
    return MyErrCode::kOk;
}

vk::Pipeline& Context::getPipeline(char const* name) { return pipelines_.at(name); }

MyErrCode Context::destroyPipeline(char const* name)
{
    if (auto it = pipelines_.find(name); it != pipelines_.end()) {
        device_.destroy(it->second);
        pipelines_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("name not exist: {}", name);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::createDescriptorSet(char const* name, char const* set_layout_name)
{
    if (descriptor_sets_.find(name) != descriptor_sets_.end()) {
        CHECK_ERR_RET(destroyDescriptorSet(name));
    }
    auto descriptor_sets = CHECK_VKHPP_VAL(device_.allocateDescriptorSets(
        {descriptor_pool_, 1, &getDescriptorSetLayout(set_layout_name)}));
    descriptor_sets_[name] = descriptor_sets.front();
    return MyErrCode::kOk;
}

vk::DescriptorSet& Context::getDescriptorSet(char const* name) { return descriptor_sets_.at(name); }

MyErrCode Context::updateDescriptorSets(std::vector<vk::WriteDescriptorSet> const& writes)
{
    device_.updateDescriptorSets(writes, {});
    return MyErrCode::kOk;
}

MyErrCode Context::destroyDescriptorSet(char const* name)
{
    if (auto it = descriptor_sets_.find(name); it != descriptor_sets_.end()) {
        device_.freeDescriptorSets(descriptor_pool_, it->second);
        descriptor_sets_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("name not exist: {}", name);
        return MyErrCode::kFailed;
    }
}

MyErrCode Context::oneTimeSubmit(int queue_index, TaskSubmitter const& submitter)
{
    auto command_buffers = CHECK_VKHPP_VAL(device_.allocateCommandBuffers(
        {getCommandPool(queue_index), vk::CommandBufferLevel::ePrimary, 1}));
    auto& command_buffer = command_buffers.front();

    CHECK_VKHPP_RET(command_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
    CHECK_ERR_RET(submitter(command_buffer));
    CHECK_VKHPP_RET(command_buffer.end());

    vk::Queue queue = getQueue(queue_index);
    CHECK_VKHPP_RET(queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &command_buffer}, nullptr));
    CHECK_VKHPP_RET(queue.waitIdle());

    return MyErrCode::kOk;
}

MyErrCode Context::destroy()
{
    for (auto& [_, i]: pipelines_) {
        device_.destroy(i);
    }
    for (auto& [_, i]: pipeline_layouts_) {
        device_.destroy(i);
    }
    for (auto& [_, i]: descriptor_set_layouts_) {
        device_.destroy(i);
    }
    for (auto& [_, i]: buffers_) {
        vmaDestroyBuffer(allocator_, i.buf_, i.alloc_);
    }
    for (auto& [_, i]: shader_modules_) {
        device_.destroy(i);
    }
    if (descriptor_pool_) {
        device_.destroy(descriptor_pool_);
    }
    for (auto& i: command_pools_) {
        device_.destroy(i);
    }
    if (allocator_) {
        vmaDestroyAllocator(allocator_);
    }
    if (device_) {
        device_.destroy();
    }
    if (debug_messenger_) {
        instance_.destroy(debug_messenger_);
    }
    if (instance_) {
        instance_.destroy();
    }
    return MyErrCode::kOk;
}

}  // namespace myvk
