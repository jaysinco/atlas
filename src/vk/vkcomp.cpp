#include "vulkan-helper.h"
#include "toolkit/toolkit.h"
#include "toolkit/args.h"
#include "toolkit/logging.h"

int main(int argc, char** argv)
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.parse();

    // instance
    vk::ApplicationInfo app_info{"vkcomp", VK_MAKE_VERSION(0, 1, 0), "No Engine",
                                 VK_MAKE_VERSION(0, 1, 0), MYVK_API_VERSION};
    std::vector<char const*> const instance_layers = {
#ifdef MYVK_ENABLE_VALIDATION_LAYER
        "VK_LAYER_KHRONOS_validation"
#endif
    };
    std::vector<char const*> instance_extensions = {"VK_EXT_debug_utils"};
    auto instance = vk::createInstance(
        {vk::InstanceCreateFlags(), &app_info, instance_layers, instance_extensions});

    myvk::loadInstanceProcAddr(instance);
    auto debug_messenger = instance.createDebugUtilsMessengerEXT(myvk::getDebugMessengerInfo());

    // physical device
    auto physical_device = instance.enumeratePhysicalDevices().front();
    auto device_props = physical_device.getProperties();
    auto device_limits = device_props.limits;

    ILOG("Device Name: {}", device_props.deviceName);
    ILOG("Vulkan Version: {}.{}.{}", VK_VERSION_MAJOR(device_props.apiVersion),
         VK_VERSION_MINOR(device_props.apiVersion), VK_VERSION_PATCH(device_props.apiVersion));
    ILOG("Max Compute Shared Memory Size: {} KB", device_limits.maxComputeSharedMemorySize / 1024);

    // device & queue
    auto queue_family_props = physical_device.getQueueFamilyProperties();
    auto prop_chosen = std::find_if(queue_family_props.begin(), queue_family_props.end(),
                                    [](vk::QueueFamilyProperties const& prop) {
                                        return prop.queueFlags & vk::QueueFlagBits::eCompute;
                                    });
    uint32_t const queue_family_index = std::distance(queue_family_props.begin(), prop_chosen);
    ILOG("Compute Queue Family Index: {}", queue_family_index);

    float const queue_priority = 1.0f;
    vk::DeviceQueueCreateInfo queue_create_info(vk::DeviceQueueCreateFlags(), queue_family_index, 1,
                                                &queue_priority);

    auto device = physical_device.createDevice({vk::DeviceCreateFlags(), queue_create_info});
    vk::Queue queue = device.getQueue(queue_family_index, 0);

    // allocator & buffer
    auto allocator = myvk::createAllocator(physical_device, device, instance);

    uint32_t const num_elements = 10;
    uint32_t const buffer_size = num_elements * sizeof(int32_t);

    auto in_buffer = allocator.createBuffer(
        buffer_size, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_ALLOCATION_CREATE_MAPPED_BIT);

    auto out_buffer = allocator.createBuffer(
        buffer_size, vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        VMA_ALLOCATION_CREATE_MAPPED_BIT);

    int32_t* in_buffer_data = reinterpret_cast<int32_t*>(in_buffer.getMappedData());
    for (int32_t i = 0; i < num_elements; ++i) {
        in_buffer_data[i] = i;
    }

    // pipeline
    auto shader_module =
        myvk::createShaderModule(device, toolkit::getDataDir() / "vkcomp.glsl.spv");

    auto descriptor_set_layout_bindings = {
        vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eCompute}};

    auto descriptor_set_layout = device.createDescriptorSetLayout(
        {vk::DescriptorSetLayoutCreateFlags(), descriptor_set_layout_bindings});

    auto pipeline_layout =
        device.createPipelineLayout({vk::PipelineLayoutCreateFlags(), descriptor_set_layout});

    vk::PipelineShaderStageCreateInfo pipeline_shader(vk::PipelineShaderStageCreateFlags(),
                                                      vk::ShaderStageFlagBits::eCompute,
                                                      shader_module, "main");

    vk::ComputePipelineCreateInfo pipeline_create_info(vk::PipelineCreateFlags(), pipeline_shader,
                                                       pipeline_layout);

    auto pipeline = device.createComputePipeline({}, pipeline_create_info).value;

    // descriptor
    vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, 2);
    auto descriptor_pool =
        device.createDescriptorPool({vk::DescriptorPoolCreateFlags(), 1, descriptor_pool_size});

    auto descriptor_sets =
        device.allocateDescriptorSets({descriptor_pool, 1, &descriptor_set_layout});
    auto descriptor_set = descriptor_sets.front();

    vk::DescriptorBufferInfo in_buffer_info(in_buffer, 0, buffer_size);
    vk::DescriptorBufferInfo out_buffer_info(out_buffer, 0, buffer_size);

    std::vector<vk::WriteDescriptorSet> const write_descriptor_sets = {
        vk::WriteDescriptorSet{descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
                               &in_buffer_info},
        vk::WriteDescriptorSet{descriptor_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
                               &out_buffer_info},
    };

    device.updateDescriptorSets(write_descriptor_sets, {});

    // command buffer
    auto command_pool =
        device.createCommandPool({vk::CommandPoolCreateFlags(), queue_family_index});
    auto command_buffers =
        device.allocateCommandBuffers({command_pool, vk::CommandBufferLevel::ePrimary, 1});
    auto command_buffer = command_buffers.front();

    // dispatch
    command_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0,
                                      descriptor_set, {});
    command_buffer.dispatch(num_elements, 1, 1);
    command_buffer.end();

    auto fence = device.createFence({});
    vk::SubmitInfo submit_info(0, nullptr, nullptr, 1, &command_buffer);
    queue.submit({submit_info}, fence);
    auto result = device.waitForFences(fence, true, -1);

    // print output
    for (uint32_t i = 0; i < num_elements; ++i) {
        std::cout << in_buffer_data[i] << " ";
    }
    std::cout << std::endl;

    int32_t* out_buffer_data = reinterpret_cast<int32_t*>(out_buffer.getMappedData());
    for (uint32_t i = 0; i < num_elements; ++i) {
        std::cout << out_buffer_data[i] << " ";
    }
    std::cout << std::endl;

    // cleanup
    allocator.destroyBuffer(in_buffer);
    allocator.destroyBuffer(out_buffer);
    allocator.destroy();

    device.destroyFence(fence);
    device.destroyDescriptorSetLayout(descriptor_set_layout);
    device.destroyPipelineLayout(pipeline_layout);
    device.destroyShaderModule(shader_module);
    device.destroyPipeline(pipeline);
    device.destroyDescriptorPool(descriptor_pool);
    device.destroyCommandPool(command_pool);
    device.destroy();

    instance.destroyDebugUtilsMessengerEXT(debug_messenger);
    instance.destroy();

    return 0;
    MY_CATCH_RTI
}
