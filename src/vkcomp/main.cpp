#include "toolkit/vulkan-helper.h"
#include "toolkit/toolkit.h"
#include "toolkit/args.h"
#include "toolkit/logging.h"

int main(int argc, char** argv)
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.parse();

    vk::ApplicationInfo app_info{"vkcomp", VK_MAKE_VERSION(0, 1, 0), "No Engine",
                                 VK_MAKE_VERSION(0, 1, 0), VK_API_VERSION_1_3};
    std::vector<char const*> const layers = {"VK_LAYER_KHRONOS_validation"};
    auto instance = vk::createInstance({vk::InstanceCreateFlags(), &app_info, layers});

    auto physical_device = instance.enumeratePhysicalDevices().front();
    auto device_props = physical_device.getProperties();
    ILOG("Device Name: {}", device_props.deviceName);
    ILOG("Vulkan Version: {}.{}.{}", VK_VERSION_MAJOR(device_props.apiVersion),
         VK_VERSION_MINOR(device_props.apiVersion), VK_VERSION_PATCH(device_props.apiVersion));
    auto device_limits = device_props.limits;
    ILOG("Max Compute Shared Memory Size: {} KB", device_limits.maxComputeSharedMemorySize / 1024);

    auto queue_family_props = physical_device.getQueueFamilyProperties();
    auto prop_it = std::find_if(queue_family_props.begin(), queue_family_props.end(),
                                [](vk::QueueFamilyProperties const& prop) {
                                    return prop.queueFlags & vk::QueueFlagBits::eCompute;
                                });
    uint32_t const compute_queue_family_index = std::distance(queue_family_props.begin(), prop_it);
    ILOG("Compute Queue Family Index: {}", compute_queue_family_index);

    float const queue_priority = 1.0f;
    vk::DeviceQueueCreateInfo queue_create_info(vk::DeviceQueueCreateFlags(),
                                                compute_queue_family_index, 1, &queue_priority);
    auto device = physical_device.createDevice({vk::DeviceCreateFlags(), queue_create_info});
    auto allocator = myvk::createAllocator(VK_API_VERSION_1_3, physical_device, device, instance);

    uint32_t const num_elements = 10;
    uint32_t const buffer_size = num_elements * sizeof(int32_t);

    auto in_buffer = myvk::createBuffer(
        buffer_size, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_ALLOCATION_CREATE_MAPPED_BIT);

    auto out_buffer = myvk::createBuffer(
        buffer_size, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_ALLOCATION_CREATE_MAPPED_BIT);

    int32_t* in_buffer_data = reinterpret_cast<int32_t*>(in_buffer.getMappedData());
    for (int32_t i = 0; i < num_elements; ++i) {
        in_buffer_data[i] = i;
    }

    auto shader_module =
        myvk::createShaderModule(device, toolkit::getDataDir() / "vkcomp.glsl.spv");

    std::vector<vk::DescriptorSetLayoutBinding> const descriptor_set_layout_bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}};
    vk::DescriptorSetLayout descriptor_set_layout = device.createDescriptorSetLayout(
        {vk::DescriptorSetLayoutCreateFlags(), descriptor_set_layout_bindings});

    vk::PipelineLayout pipeline_layout =
        device.createPipelineLayout({vk::PipelineLayoutCreateFlags(), descriptor_set_layout});

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, shader_module,
        "main");
    vk::ComputePipelineCreateInfo pipeline_create_info(
        vk::PipelineCreateFlags(), pipeline_shader_create_info, pipeline_layout);
    vk::Pipeline pipeline = device.createComputePipeline({}, pipeline_create_info).value;

    vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, 2);
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info(vk::DescriptorPoolCreateFlags(), 1,
                                                             descriptor_pool_size);
    vk::DescriptorPool descriptor_pool = device.createDescriptorPool(descriptor_pool_create_info);

    vk::DescriptorSetAllocateInfo DescriptorSetAllocInfo(descriptor_pool, 1,
                                                         &descriptor_set_layout);
    std::vector<vk::DescriptorSet> const DescriptorSets =
        device.allocateDescriptorSets(DescriptorSetAllocInfo);
    vk::DescriptorSet DescriptorSet = DescriptorSets.front();
    vk::DescriptorBufferInfo InBufferInfo(in_buffer, 0, num_elements * sizeof(int32_t));
    vk::DescriptorBufferInfo OutBufferInfo(out_buffer, 0, num_elements * sizeof(int32_t));

    std::vector<vk::WriteDescriptorSet> const WriteDescriptorSets = {
        {DescriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &InBufferInfo},
        {DescriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &OutBufferInfo},
    };
    device.updateDescriptorSets(WriteDescriptorSets, {});

    vk::CommandPoolCreateInfo CommandPoolCreateInfo(vk::CommandPoolCreateFlags(),
                                                    compute_queue_family_index);
    vk::CommandPool CommandPool = device.createCommandPool(CommandPoolCreateInfo);

    vk::CommandBufferAllocateInfo CommandBufferAllocInfo(CommandPool,  // Command Pool
                                                         vk::CommandBufferLevel::ePrimary,  // Level
                                                         1);  // Num Command Buffers
    std::vector<vk::CommandBuffer> const CmdBuffers =
        device.allocateCommandBuffers(CommandBufferAllocInfo);
    vk::CommandBuffer CmdBuffer = CmdBuffers.front();

    vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    CmdBuffer.begin(CmdBufferBeginInfo);
    CmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    CmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,  // Bind point
                                 pipeline_layout,                  // Pipeline Layout
                                 0,                                // First descriptor set
                                 {DescriptorSet},                  // List of descriptor sets
                                 {});                              // Dynamic offsets
    CmdBuffer.dispatch(num_elements, 1, 1);
    CmdBuffer.end();

    vk::Queue Queue = device.getQueue(compute_queue_family_index, 0);
    vk::Fence Fence = device.createFence(vk::FenceCreateInfo());

    vk::SubmitInfo SubmitInfo(0,            // Num Wait Semaphores
                              nullptr,      // Wait Semaphores
                              nullptr,      // Pipeline Stage Flags
                              1,            // Num Command Buffers
                              &CmdBuffer);  // List of command buffers
    Queue.submit({SubmitInfo}, Fence);
    auto result = device.waitForFences({Fence},        // List of fences
                                       true,           // Wait All
                                       uint64_t(-1));  // Timeout

    for (uint32_t i = 0; i < num_elements; ++i) {
        std::cout << in_buffer_data[i] << " ";
    }
    std::cout << std::endl;

    int32_t* out_buffer_data = reinterpret_cast<int32_t*>(out_buffer.getMappedData());
    for (uint32_t i = 0; i < num_elements; ++i) {
        std::cout << out_buffer_data[i] << " ";
    }
    std::cout << std::endl;

    in_buffer.destory();
    out_buffer.destory();
    allocator.destory();
    device.destroyFence(Fence);
    device.destroyDescriptorSetLayout(descriptor_set_layout);
    device.destroyPipelineLayout(pipeline_layout);
    device.destroyShaderModule(shader_module);
    device.destroyPipeline(pipeline);
    device.destroyDescriptorPool(descriptor_pool);
    device.destroyCommandPool(CommandPool);
    device.destroy();
    instance.destroy();

    return 0;
    MY_CATCH_RTI
}
