#include "vulkan-helper.h"
#include "toolkit/toolkit.h"
#include "toolkit/args.h"
#include "toolkit/logging.h"

// NOLINTBEGIN
enum AppUid : int
{
    UID_vkQueue_compute,
    UID_vkCommandPool_compute,
    UID_vkBuffer_a,
    UID_vkBuffer_b,
    UID_vkShader_square,
    UID_vkShader_mul2,
    UID_vkDescriptorPool_main,
    UID_vkDescriptorSetLayout_compute,
    UID_vkDescriptorSet_square,
    UID_vkDescriptorSet_mul2,
    UID_vkPipelineLayout_compute,
    UID_vkPipeline_square,
    UID_vkPipeline_mul2,
};

// NOLINTEND

MY_MAIN
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.parse();

    myvk::Context ctx;
    CHECK_ERR_RET(ctx.createInstance("vkcomp", {"VK_EXT_debug_utils"}));
    CHECK_ERR_RET(ctx.createPhysicalDevice());
    auto queue_picker = [](uint32_t family_index, vk::QueueFamilyProperties const& prop) -> bool {
        return static_cast<bool>(prop.queueFlags & vk::QueueFlagBits::eCompute);
    };
    CHECK_ERR_RET(ctx.createDeviceAndQueues({}, {{UID_vkQueue_compute, queue_picker}}));
    CHECK_ERR_RET(ctx.createCommandPool(UID_vkCommandPool_compute, UID_vkQueue_compute,
                                        vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
                                            vk::CommandPoolCreateFlagBits::eTransient));
    CHECK_ERR_RET(ctx.createDescriptorPool(UID_vkDescriptorPool_main, 2,
                                           {{vk::DescriptorType::eStorageBuffer, 4}}));
    CHECK_ERR_RET(ctx.createAllocator());

    uint32_t const num_elements = 10;
    uint32_t const buffer_size = num_elements * sizeof(int32_t);

    CHECK_ERR_RET(ctx.createBuffer(
        UID_vkBuffer_a, buffer_size, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_ALLOCATION_CREATE_MAPPED_BIT));

    CHECK_ERR_RET(ctx.createBuffer(
        UID_vkBuffer_b, buffer_size, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_ALLOCATION_CREATE_MAPPED_BIT));

    int32_t* a_buffer_data =
        reinterpret_cast<int32_t*>(ctx.getBuffer(UID_vkBuffer_a).getAllocInfo().pMappedData);

    int32_t* b_buffer_data =
        reinterpret_cast<int32_t*>(ctx.getBuffer(UID_vkBuffer_b).getAllocInfo().pMappedData);

    for (int32_t i = 0; i < num_elements; ++i) {
        a_buffer_data[i] = i;
    }

    CHECK_ERR_RET(ctx.createDescriptorSetLayout(
        UID_vkDescriptorSetLayout_compute,
        {
            {vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute},
            {vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute},
        }));

    CHECK_ERR_RET(ctx.createDescriptorSet(
        UID_vkDescriptorSet_square, UID_vkDescriptorSetLayout_compute, UID_vkDescriptorPool_main));

    CHECK_ERR_RET(ctx.createDescriptorSet(
        UID_vkDescriptorSet_mul2, UID_vkDescriptorSetLayout_compute, UID_vkDescriptorPool_main));

    CHECK_ERR_RET(ctx.updateDescriptorSet(
        UID_vkDescriptorSet_square,
        {
            {0, vk::DescriptorType::eStorageBuffer, ctx.getBuffer(UID_vkBuffer_a)},
            {1, vk::DescriptorType::eStorageBuffer, ctx.getBuffer(UID_vkBuffer_b)},
        }));

    CHECK_ERR_RET(ctx.updateDescriptorSet(
        UID_vkDescriptorSet_mul2,
        {
            {0, vk::DescriptorType::eStorageBuffer, ctx.getBuffer(UID_vkBuffer_b)},
            {1, vk::DescriptorType::eStorageBuffer, ctx.getBuffer(UID_vkBuffer_a)},
        }));

    CHECK_ERR_RET(
        ctx.createShaderModule(UID_vkShader_square, toolkit::getDataDir() / "square.glsl.spv"));
    CHECK_ERR_RET(
        ctx.createShaderModule(UID_vkShader_mul2, toolkit::getDataDir() / "mul2.glsl.spv"));

    CHECK_ERR_RET(ctx.createPipelineLayout(UID_vkPipelineLayout_compute,
                                           {UID_vkDescriptorSetLayout_compute}));

    CHECK_ERR_RET(ctx.createComputePipeline(UID_vkPipeline_square, UID_vkPipelineLayout_compute,
                                            UID_vkShader_square));
    CHECK_ERR_RET(ctx.createComputePipeline(UID_vkPipeline_mul2, UID_vkPipelineLayout_compute,
                                            UID_vkShader_mul2));

    auto submit_compute = [&ctx](vk::CommandBuffer& cmd) -> MyErrCode {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, ctx.getPipeline(UID_vkPipeline_square));
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                               ctx.getPipelineLayout(UID_vkPipelineLayout_compute), 0,
                               {ctx.getDescriptorSet(UID_vkDescriptorSet_square)}, {});
        cmd.dispatch(num_elements, 1, 1);

        vk::MemoryBarrier barrier(vk::AccessFlagBits::eMemoryWrite,
                                  vk::AccessFlagBits::eMemoryRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags{},
                            barrier, {}, {});

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, ctx.getPipeline(UID_vkPipeline_mul2));
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                               ctx.getPipelineLayout(UID_vkPipelineLayout_compute), 0,
                               {ctx.getDescriptorSet(UID_vkDescriptorSet_mul2)}, {});
        cmd.dispatch(num_elements, 1, 1);
        return MyErrCode::kOk;
    };
    CHECK_ERR_RET(
        ctx.oneTimeSubmit(UID_vkQueue_compute, UID_vkCommandPool_compute, submit_compute));

    ILOG("a = [{}]", fmt::join(a_buffer_data, a_buffer_data + num_elements, ", "));
    ILOG("b = [{}]", fmt::join(b_buffer_data, b_buffer_data + num_elements, ", "));

    CHECK_ERR_RET(ctx.destroy());
    return MyErrCode::kOk;
    MY_CATCH_RET
}
