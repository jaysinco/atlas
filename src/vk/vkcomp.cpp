#include "vulkan-helper.h"
#include "toolkit/toolkit.h"
#include "toolkit/args.h"
#include "toolkit/logging.h"
#include <thread>
#include <glm/glm.hpp>

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
    UID_vkCommandBuffer_0,
    UID_vkSemaphore_0,
};

// NOLINTEND

struct PushConstants
{
    glm::ivec4 data;
};

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
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));

    CHECK_ERR_RET(ctx.createBuffer(
        UID_vkBuffer_b, buffer_size, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));

    int32_t* a_buffer_data = reinterpret_cast<int32_t*>(ctx.getBuffer(UID_vkBuffer_a).map());

    int32_t* b_buffer_data = reinterpret_cast<int32_t*>(ctx.getBuffer(UID_vkBuffer_b).map());

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

    CHECK_ERR_RET(ctx.createPipelineLayout(
        UID_vkPipelineLayout_compute, {UID_vkDescriptorSetLayout_compute},
        {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants)}}));

    CHECK_ERR_RET(ctx.createComputePipeline(UID_vkPipeline_square, UID_vkPipelineLayout_compute,
                                            UID_vkShader_square));

    CHECK_ERR_RET(ctx.createComputePipeline(UID_vkPipeline_mul2, UID_vkPipelineLayout_compute,
                                            UID_vkShader_mul2));

    CHECK_ERR_RET(ctx.createTimelineSemaphore(UID_vkSemaphore_0, 0));

    CHECK_ERR_RET(ctx.createCommandBuffer(UID_vkCommandBuffer_0, UID_vkCommandPool_compute));

    PushConstants contants;
    contants.data.x = 3;
    contants.data.y = 1;

    auto record_compute = [&](vk::CommandBuffer& cmd) -> MyErrCode {
        cmd.pushConstants(ctx.getPipelineLayout(UID_vkPipelineLayout_compute),
                          vk::ShaderStageFlagBits::eCompute, 0, sizeof(contants), &contants);

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, ctx.getPipeline(UID_vkPipeline_square));
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                               ctx.getPipelineLayout(UID_vkPipelineLayout_compute), 0,
                               {ctx.getDescriptorSet(UID_vkDescriptorSet_square)}, {});
        cmd.dispatch(num_elements, 1, 1);

        vk::MemoryBarrier2 barrier(
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
        cmd.pipelineBarrier2(vk::DependencyInfo{{}, barrier});

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, ctx.getPipeline(UID_vkPipeline_mul2));
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                               ctx.getPipelineLayout(UID_vkPipelineLayout_compute), 0,
                               {ctx.getDescriptorSet(UID_vkDescriptorSet_mul2)}, {});
        cmd.dispatch(num_elements, 1, 1);
        return MyErrCode::kOk;
    };

    CHECK_ERR_RET(ctx.recordCommand(
        UID_vkCommandBuffer_0, vk::CommandBufferUsageFlagBits::eOneTimeSubmit, record_compute));

    CHECK_ERR_RET(ctx.submit(UID_vkQueue_compute, UID_vkCommandBuffer_0,
                             {{UID_vkSemaphore_0, vk::PipelineStageFlagBits2::eComputeShader, 1}},
                             {{UID_vkSemaphore_0, 2}}));

    std::this_thread::sleep_for(std::chrono::seconds(1));
    for (int32_t i = 0; i < num_elements; ++i) {
        a_buffer_data[i] = i;
    }

    CHECK_ERR_RET(ctx.signalSemaphore({UID_vkSemaphore_0, 1}));
    CHECK_ERR_RET(ctx.waitSemaphores({{UID_vkSemaphore_0, 2}}));

    ILOG("a = [{}]", fmt::join(a_buffer_data, a_buffer_data + num_elements, ", "));
    ILOG("b = [{}]", fmt::join(b_buffer_data, b_buffer_data + num_elements, ", "));

    CHECK_ERR_RET(ctx.destroy());
    return MyErrCode::kOk;
    MY_CATCH_RET
}
