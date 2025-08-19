#include "vulkan-helper.h"
#include "toolkit/toolkit.h"
#include "toolkit/args.h"
#include "toolkit/logging.h"

// NOLINTBEGIN
enum AppUid : int
{
    UID_vkQueue_compute,
    UID_vkCommandPool_compute = UID_vkQueue_compute,
    UID_vkBuffer_in,
    UID_vkBuffer_out,
    UID_vkShader_compute,
    UID_vkDescriptorPool_main,
    UID_vkDescriptorSet_compute,
    UID_vkDescriptorSetLayout_compute = UID_vkDescriptorSet_compute,
    UID_vkPipeline_compute,
    UID_vkPipelineLayout_compute,
};

// NOLINTEND

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
    CHECK_ERR_RET(ctx.createDeviceAndQueues({}, {{UID_vkQueue_compute, queue_picker}}));
    CHECK_ERR_RET(ctx.createCommandPool(UID_vkQueue_compute, vk::CommandPoolCreateFlags()));
    CHECK_ERR_RET(ctx.createDescriptorPool(UID_vkDescriptorPool_main, 1,
                                           {{vk::DescriptorType::eStorageBuffer, 2}}));
    CHECK_ERR_RET(ctx.createAllocator());

    uint32_t const num_elements = 10;
    uint32_t const buffer_size = num_elements * sizeof(int32_t);

    CHECK_ERR_RET(ctx.createBuffer(
        UID_vkBuffer_in, buffer_size, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_ALLOCATION_CREATE_MAPPED_BIT));

    CHECK_ERR_RET(ctx.createBuffer(
        UID_vkBuffer_out, buffer_size, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_ALLOCATION_CREATE_MAPPED_BIT));

    int32_t* in_buffer_data =
        reinterpret_cast<int32_t*>(ctx.getBuffer(UID_vkBuffer_in).getAllocInfo().pMappedData);
    for (int32_t i = 0; i < num_elements; ++i) {
        in_buffer_data[i] = i;
    }

    CHECK_ERR_RET(ctx.createDescriptorSetAndLayout(
        UID_vkDescriptorSet_compute, UID_vkDescriptorPool_main,
        {
            ctx.bindBufferDescriptor(vk::DescriptorType::eStorageBuffer,
                                     vk::ShaderStageFlagBits::eCompute, UID_vkBuffer_in),
            ctx.bindBufferDescriptor(vk::DescriptorType::eStorageBuffer,
                                     vk::ShaderStageFlagBits::eCompute, UID_vkBuffer_out),
        }));

    CHECK_ERR_RET(
        ctx.createShaderModule(UID_vkShader_compute, toolkit::getDataDir() / "vkcomp.glsl.spv"));

    CHECK_ERR_RET(ctx.createPipelineLayout(UID_vkPipelineLayout_compute,
                                           {UID_vkDescriptorSetLayout_compute}));

    vk::PipelineShaderStageCreateInfo pipeline_shader_info(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute,
        ctx.getShaderModule(UID_vkShader_compute), "main");

    vk::ComputePipelineCreateInfo pipeline_create_info(
        vk::PipelineCreateFlags(), pipeline_shader_info,
        ctx.getPipelineLayout(UID_vkPipelineLayout_compute));

    CHECK_ERR_RET(ctx.createComputePipeline(UID_vkPipeline_compute, pipeline_create_info));

    auto compute_submitter = [&](vk::CommandBuffer& command_buffer) -> MyErrCode {
        command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                    ctx.getPipeline(UID_vkPipeline_compute));
        command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                          ctx.getPipelineLayout(UID_vkPipelineLayout_compute), 0,
                                          {ctx.getDescriptorSet(UID_vkDescriptorSet_compute)}, {});
        command_buffer.dispatch(num_elements, 1, 1);
        return MyErrCode::kOk;
    };
    CHECK_ERR_RET(ctx.oneTimeSubmit(UID_vkQueue_compute, compute_submitter));

    for (uint32_t i = 0; i < num_elements; ++i) {
        std::cout << in_buffer_data[i] << " ";
    }
    std::cout << std::endl;

    int32_t* out_buffer_data =
        reinterpret_cast<int32_t*>(ctx.getBuffer(UID_vkBuffer_out).getAllocInfo().pMappedData);
    for (uint32_t i = 0; i < num_elements; ++i) {
        std::cout << out_buffer_data[i] << " ";
    }
    std::cout << std::endl;

    CHECK_ERR_RET(ctx.destroy());

    return MyErrCode::kOk;
    MY_CATCH_RET
}
