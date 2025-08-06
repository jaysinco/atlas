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
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags(), &app_info, layers);
    auto instance = vk::createInstance(instance_create_info);

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
    vk::DeviceQueueCreateInfo device_queue_create_info(
        vk::DeviceQueueCreateFlags(), compute_queue_family_index, 1, &queue_priority);
    vk::DeviceCreateInfo device_create_info(vk::DeviceCreateFlags(), device_queue_create_info);
    auto device = physical_device.createDevice(device_create_info);

    uint32_t const num_elements = 10;
    uint32_t const buffer_size = num_elements * sizeof(int32_t);

    vk::BufferCreateInfo buffer_create_info{vk::BufferCreateFlags(), buffer_size,
                                            vk::BufferUsageFlagBits::eStorageBuffer,
                                            vk::SharingMode::eExclusive};
    auto vkBufferCreateInfo = static_cast<VkBufferCreateInfo>(buffer_create_info);

    VmaAllocatorCreateInfo AllocatorInfo = {};
    AllocatorInfo.vulkanApiVersion = device_props.apiVersion;
    AllocatorInfo.physicalDevice = physical_device;
    AllocatorInfo.device = device;
    AllocatorInfo.instance = instance;

    VmaAllocator Allocator;
    vmaCreateAllocator(&AllocatorInfo, &Allocator);

    VkBuffer InBufferRaw;
    VkBuffer OutBufferRaw;

    VmaAllocationCreateInfo AllocationInfo = {};
    AllocationInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    VmaAllocation InBufferAllocation;
    vmaCreateBuffer(Allocator, &vkBufferCreateInfo, &AllocationInfo, &InBufferRaw,
                    &InBufferAllocation, nullptr);

    AllocationInfo.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
    VmaAllocation OutBufferAllocation;
    vmaCreateBuffer(Allocator, &vkBufferCreateInfo, &AllocationInfo, &OutBufferRaw,
                    &OutBufferAllocation, nullptr);

    vk::Buffer InBuffer = InBufferRaw;
    vk::Buffer OutBuffer = OutBufferRaw;

    int32_t* InBufferPtr = nullptr;
    vmaMapMemory(Allocator, InBufferAllocation, reinterpret_cast<void**>(&InBufferPtr));
    for (int32_t I = 0; I < num_elements; ++I) {
        InBufferPtr[I] = I;
    }
    vmaUnmapMemory(Allocator, InBufferAllocation);

    std::vector<char> ShaderContents;
    if (std::ifstream ShaderFile{toolkit::getDataDir() / "vkcomp.glsl.spv",
                                 std::ios::binary | std::ios::ate}) {
        size_t const FileSize = ShaderFile.tellg();
        ShaderFile.seekg(0);
        ShaderContents.resize(FileSize, '\0');
        ShaderFile.read(ShaderContents.data(), FileSize);
    }

    vk::ShaderModuleCreateInfo ShaderModuleCreateInfo(
        vk::ShaderModuleCreateFlags(),                              // Flags
        ShaderContents.size(),                                      // Code size
        reinterpret_cast<uint32_t const*>(ShaderContents.data()));  // Code
    vk::ShaderModule ShaderModule = device.createShaderModule(ShaderModuleCreateInfo);

    std::vector<vk::DescriptorSetLayoutBinding> const DescriptorSetLayoutBinding = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}};
    vk::DescriptorSetLayoutCreateInfo DescriptorSetLayoutCreateInfo(
        vk::DescriptorSetLayoutCreateFlags(), DescriptorSetLayoutBinding);
    vk::DescriptorSetLayout DescriptorSetLayout =
        device.createDescriptorSetLayout(DescriptorSetLayoutCreateInfo);

    vk::PipelineLayoutCreateInfo PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(),
                                                          DescriptorSetLayout);
    vk::PipelineLayout PipelineLayout = device.createPipelineLayout(PipelineLayoutCreateInfo);
    vk::PipelineCache PipelineCache = device.createPipelineCache(vk::PipelineCacheCreateInfo());

    vk::PipelineShaderStageCreateInfo PipelineShaderCreateInfo(
        vk::PipelineShaderStageCreateFlags(),  // Flags
        vk::ShaderStageFlagBits::eCompute,     // Stage
        ShaderModule,                          // Shader Module
        "main");                               // Shader Entry Point
    vk::ComputePipelineCreateInfo ComputePipelineCreateInfo(
        vk::PipelineCreateFlags(),  // Flags
        PipelineShaderCreateInfo,   // Shader Create Info struct
        PipelineLayout);            // Pipeline Layout
    vk::Pipeline ComputePipeline =
        device.createComputePipeline(PipelineCache, ComputePipelineCreateInfo).value;

    vk::DescriptorPoolSize DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 2);
    vk::DescriptorPoolCreateInfo DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1,
                                                          DescriptorPoolSize);
    vk::DescriptorPool DescriptorPool = device.createDescriptorPool(DescriptorPoolCreateInfo);

    vk::DescriptorSetAllocateInfo DescriptorSetAllocInfo(DescriptorPool, 1, &DescriptorSetLayout);
    std::vector<vk::DescriptorSet> const DescriptorSets =
        device.allocateDescriptorSets(DescriptorSetAllocInfo);
    vk::DescriptorSet DescriptorSet = DescriptorSets.front();
    vk::DescriptorBufferInfo InBufferInfo(InBuffer, 0, num_elements * sizeof(int32_t));
    vk::DescriptorBufferInfo OutBufferInfo(OutBuffer, 0, num_elements * sizeof(int32_t));

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
    CmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, ComputePipeline);
    CmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,  // Bind point
                                 PipelineLayout,                   // Pipeline Layout
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

    vmaMapMemory(Allocator, InBufferAllocation, reinterpret_cast<void**>(&InBufferPtr));
    for (uint32_t I = 0; I < num_elements; ++I) {
        std::cout << InBufferPtr[I] << " ";
    }
    std::cout << std::endl;
    vmaUnmapMemory(Allocator, InBufferAllocation);

    int32_t* OutBufferPtr = nullptr;
    vmaMapMemory(Allocator, OutBufferAllocation, reinterpret_cast<void**>(&OutBufferPtr));
    for (uint32_t I = 0; I < num_elements; ++I) {
        std::cout << OutBufferPtr[I] << " ";
    }
    std::cout << std::endl;
    vmaUnmapMemory(Allocator, OutBufferAllocation);

    struct BufferInfo
    {
        VkBuffer Buffer;
        VmaAllocation Allocation;
    };

    // Lets allocate a couple of buffers to see how they are layed out in memory
    auto AllocateBuffer = [Allocator, compute_queue_family_index](size_t SizeInBytes,
                                                                  VmaMemoryUsage Usage) {
        vk::BufferCreateInfo BufferCreateInfo{
            vk::BufferCreateFlags(),                  // Flags
            SizeInBytes,                              // Size
            vk::BufferUsageFlagBits::eStorageBuffer,  // Usage
            vk::SharingMode::eExclusive,              // Sharing mode
            1,                                        // Number of queue family indices
            &compute_queue_family_index               // List of queue family indices
        };

        auto vkBufferCreateInfo = static_cast<VkBufferCreateInfo>(BufferCreateInfo);

        VmaAllocationCreateInfo AllocationInfo = {};
        AllocationInfo.usage = Usage;

        BufferInfo Info;
        vmaCreateBuffer(Allocator, &vkBufferCreateInfo, &AllocationInfo, &Info.Buffer,
                        &Info.Allocation, nullptr);

        return Info;
    };

    auto DestroyBuffer = [Allocator](BufferInfo Info) {
        vmaDestroyBuffer(Allocator, Info.Buffer, Info.Allocation);
    };

    constexpr size_t MB = 1024 * 1024;
    BufferInfo B1 = AllocateBuffer(4 * MB, VMA_MEMORY_USAGE_CPU_TO_GPU);
    BufferInfo B2 = AllocateBuffer(10 * MB, VMA_MEMORY_USAGE_GPU_TO_CPU);
    BufferInfo B3 = AllocateBuffer(20 * MB, VMA_MEMORY_USAGE_GPU_ONLY);
    BufferInfo B4 = AllocateBuffer(100 * MB, VMA_MEMORY_USAGE_CPU_ONLY);

    DestroyBuffer(B1);
    DestroyBuffer(B2);
    DestroyBuffer(B3);
    DestroyBuffer(B4);

    vmaDestroyBuffer(Allocator, InBuffer, InBufferAllocation);
    vmaDestroyBuffer(Allocator, OutBuffer, OutBufferAllocation);
    vmaDestroyAllocator(Allocator);

    device.resetCommandPool(CommandPool, vk::CommandPoolResetFlags());
    device.destroyFence(Fence);
    device.destroyDescriptorSetLayout(DescriptorSetLayout);
    device.destroyPipelineLayout(PipelineLayout);
    device.destroyPipelineCache(PipelineCache);
    device.destroyShaderModule(ShaderModule);
    device.destroyPipeline(ComputePipeline);
    device.destroyDescriptorPool(DescriptorPool);
    device.destroyCommandPool(CommandPool);
    device.destroy();
    instance.destroy();

    return 0;
    MY_CATCH_RTI
}
