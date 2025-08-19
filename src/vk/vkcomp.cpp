#include "vulkan-helper.h"
#include "toolkit/toolkit.h"
#include "toolkit/args.h"
#include "toolkit/logging.h"

enum AppUid : int
{
    kQueueCompute,
    kBufferIn,
    kBufferOut,
    kShaderCompute,
    kDescriptorPoolMain,
    kDescriptorLayoutCompute,
    kDescriptorSetCompute,
    kPipelineLayoutCompute,
    kPipelineCompute,
};

MY_MAIN
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.parse();

    myvk::Context ctx;
    std::vector<char const*> instance_extensions = {"VK_EXT_debug_utils"};
    CHECK_ERR_RET(ctx.createInstance("vkcomp", instance_extensions));
    auto device_picker = [](vk::PhysicalDeviceProperties const& prop,
                            vk::PhysicalDeviceFeatures const& feat) -> bool {
        return prop.deviceType == vk::PhysicalDeviceType::eDiscreteGpu ||
               prop.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
    };
    CHECK_ERR_RET(ctx.createPhysicalDevice(device_picker));
    auto queue_picker = [](uint32_t family_index, vk::QueueFamilyProperties const& prop) -> bool {
        return static_cast<bool>(prop.queueFlags & vk::QueueFlagBits::eCompute);
    };
    CHECK_ERR_RET(ctx.createDeviceAndQueues({}, {{kQueueCompute, queue_picker}}));
    CHECK_ERR_RET(ctx.createCommandPool(kQueueCompute, vk::CommandPoolCreateFlags()));

    vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, 2);
    CHECK_ERR_RET(ctx.createDescriptorPool(
        kDescriptorPoolMain, {vk::DescriptorPoolCreateFlags(), 1, descriptor_pool_size}));
    CHECK_ERR_RET(ctx.createAllocator());

    uint32_t const num_elements = 10;
    uint32_t const buffer_size = num_elements * sizeof(int32_t);

    CHECK_ERR_RET(ctx.createBuffer(
        kBufferIn, buffer_size, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_ALLOCATION_CREATE_MAPPED_BIT));

    CHECK_ERR_RET(ctx.createBuffer(
        kBufferOut, buffer_size, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_ALLOCATION_CREATE_MAPPED_BIT));

    int32_t* in_buffer_data =
        reinterpret_cast<int32_t*>(ctx.getBuffer(kBufferIn).getAllocInfo().pMappedData);
    for (int32_t i = 0; i < num_elements; ++i) {
        in_buffer_data[i] = i;
    }

    CHECK_ERR_RET(
        ctx.createShaderModule(kShaderCompute, toolkit::getDataDir() / "vkcomp.glsl.spv"));

    auto descriptor_set_layout_bindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eCompute}};
    CHECK_ERR_RET(ctx.createDescriptorSetLayout(
        kDescriptorLayoutCompute,
        {vk::DescriptorSetLayoutCreateFlags(), descriptor_set_layout_bindings}));

    CHECK_ERR_RET(ctx.createPipelineLayout(
        kPipelineLayoutCompute,
        {vk::PipelineLayoutCreateFlags(), ctx.getDescriptorSetLayout(kDescriptorLayoutCompute)}));

    vk::PipelineShaderStageCreateInfo pipeline_shader_info(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute,
        ctx.getShaderModule(kShaderCompute), "main");
    vk::ComputePipelineCreateInfo pipeline_create_info(
        vk::PipelineCreateFlags(), pipeline_shader_info,
        ctx.getPipelineLayout(kPipelineLayoutCompute));
    CHECK_ERR_RET(ctx.createComputePipeline(kPipelineCompute, pipeline_create_info));

    CHECK_ERR_RET(ctx.createDescriptorSet(kDescriptorSetCompute, kDescriptorLayoutCompute,
                                          kDescriptorPoolMain));

    vk::DescriptorBufferInfo in_buffer_info(ctx.getBuffer(kBufferIn), 0, buffer_size);
    vk::DescriptorBufferInfo out_buffer_info(ctx.getBuffer(kBufferOut), 0, buffer_size);
    std::vector<vk::WriteDescriptorSet> const write_descriptor_sets = {
        vk::WriteDescriptorSet{ctx.getDescriptorSet(kDescriptorSetCompute), 0, 0, 1,
                               vk::DescriptorType::eStorageBuffer, nullptr, &in_buffer_info},
        vk::WriteDescriptorSet{ctx.getDescriptorSet(kDescriptorSetCompute), 1, 0, 1,
                               vk::DescriptorType::eStorageBuffer, nullptr, &out_buffer_info},
    };
    CHECK_ERR_RET(ctx.updateDescriptorSets(write_descriptor_sets));

    auto compute_submitter = [&](vk::CommandBuffer& command_buffer) -> MyErrCode {
        command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                    ctx.getPipeline(kPipelineCompute));
        command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                          ctx.getPipelineLayout(kPipelineLayoutCompute), 0,
                                          {ctx.getDescriptorSet(kDescriptorSetCompute)}, {});
        command_buffer.dispatch(num_elements, 1, 1);
        return MyErrCode::kOk;
    };
    CHECK_ERR_RET(ctx.oneTimeSubmit(kQueueCompute, compute_submitter));

    for (uint32_t i = 0; i < num_elements; ++i) {
        std::cout << in_buffer_data[i] << " ";
    }
    std::cout << std::endl;

    int32_t* out_buffer_data =
        reinterpret_cast<int32_t*>(ctx.getBuffer(kBufferOut).getAllocInfo().pMappedData);
    for (uint32_t i = 0; i < num_elements; ++i) {
        std::cout << out_buffer_data[i] << " ";
    }
    std::cout << std::endl;

    CHECK_ERR_RET(ctx.destroy());

    return MyErrCode::kOk;
    MY_CATCH_RET
}
