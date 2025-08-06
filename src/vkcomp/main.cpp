#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include "toolkit/args.h"
#include "toolkit/toolkit.h"
#include "toolkit/logging.h"

int main(int argc, char** argv)
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.parse();
    std::cout << "Hello Vulkan Compute" << std::endl;
    vk::ApplicationInfo AppInfo{
        "VulkanCompute",    // Application Name
        1,                  // Application Version
        nullptr,            // Engine Name or nullptr
        0,                  // Engine Version
        VK_API_VERSION_1_1  // Vulkan API version
    };

    std::vector<char const*> const Layers = {"VK_LAYER_KHRONOS_validation"};
    vk::InstanceCreateInfo InstanceCreateInfo(vk::InstanceCreateFlags(),  // Flags
                                              &AppInfo,                   // Application Info
                                              Layers.size(),              // Layers count
                                              Layers.data());             // Layers
    vk::Instance Instance = vk::createInstance(InstanceCreateInfo);

    vk::PhysicalDevice PhysicalDevice = Instance.enumeratePhysicalDevices().front();
    vk::PhysicalDeviceProperties DeviceProps = PhysicalDevice.getProperties();
    std::cout << "Device Name    : " << DeviceProps.deviceName << std::endl;
    uint32_t const ApiVersion = DeviceProps.apiVersion;
    std::cout << "Vulkan Version : " << VK_VERSION_MAJOR(ApiVersion) << "."
              << VK_VERSION_MINOR(ApiVersion) << "." << VK_VERSION_PATCH(ApiVersion) << std::endl;
    vk::PhysicalDeviceLimits DeviceLimits = DeviceProps.limits;
    std::cout << "Max Compute Shared Memory Size: "
              << DeviceLimits.maxComputeSharedMemorySize / 1024 << " KB" << std::endl;

    std::vector<vk::QueueFamilyProperties> QueueFamilyProps =
        PhysicalDevice.getQueueFamilyProperties();
    auto PropIt = std::find_if(QueueFamilyProps.begin(), QueueFamilyProps.end(),
                               [](vk::QueueFamilyProperties const& Prop) {
                                   return Prop.queueFlags & vk::QueueFlagBits::eCompute;
                               });
    uint32_t const ComputeQueueFamilyIndex = std::distance(QueueFamilyProps.begin(), PropIt);
    std::cout << "Compute Queue Family Index: " << ComputeQueueFamilyIndex << std::endl;

    // Just to avoid a warning from the Vulkan Validation Layer
    float const QueuePriority = 1.0f;
    vk::DeviceQueueCreateInfo DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(),  // Flags
                                                    ComputeQueueFamilyIndex,  // Queue Family Index
                                                    1,                        // Number of Queues
                                                    &QueuePriority);
    vk::DeviceCreateInfo DeviceCreateInfo(
        vk::DeviceCreateFlags(),  // Flags
        DeviceQueueCreateInfo);   // Device Queue Create Info struct
    vk::Device Device = PhysicalDevice.createDevice(DeviceCreateInfo);

    uint32_t const NumElements = 10;
    uint32_t const BufferSize = NumElements * sizeof(int32_t);

    vk::BufferCreateInfo BufferCreateInfo{
        vk::BufferCreateFlags(),                  // Flags
        BufferSize,                               // Size
        vk::BufferUsageFlagBits::eStorageBuffer,  // Usage
        vk::SharingMode::eExclusive,              // Sharing mode
        1,                                        // Number of queue family indices
        &ComputeQueueFamilyIndex                  // List of queue family indices
    };
    auto vkBufferCreateInfo = static_cast<VkBufferCreateInfo>(BufferCreateInfo);

    VmaAllocatorCreateInfo AllocatorInfo = {};
    AllocatorInfo.vulkanApiVersion = DeviceProps.apiVersion;
    AllocatorInfo.physicalDevice = PhysicalDevice;
    AllocatorInfo.device = Device;
    AllocatorInfo.instance = Instance;

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
    for (int32_t I = 0; I < NumElements; ++I) {
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
    vk::ShaderModule ShaderModule = Device.createShaderModule(ShaderModuleCreateInfo);

    std::vector<vk::DescriptorSetLayoutBinding> const DescriptorSetLayoutBinding = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}};
    vk::DescriptorSetLayoutCreateInfo DescriptorSetLayoutCreateInfo(
        vk::DescriptorSetLayoutCreateFlags(), DescriptorSetLayoutBinding);
    vk::DescriptorSetLayout DescriptorSetLayout =
        Device.createDescriptorSetLayout(DescriptorSetLayoutCreateInfo);

    vk::PipelineLayoutCreateInfo PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(),
                                                          DescriptorSetLayout);
    vk::PipelineLayout PipelineLayout = Device.createPipelineLayout(PipelineLayoutCreateInfo);
    vk::PipelineCache PipelineCache = Device.createPipelineCache(vk::PipelineCacheCreateInfo());

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
        Device.createComputePipeline(PipelineCache, ComputePipelineCreateInfo).value;

    vk::DescriptorPoolSize DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 2);
    vk::DescriptorPoolCreateInfo DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1,
                                                          DescriptorPoolSize);
    vk::DescriptorPool DescriptorPool = Device.createDescriptorPool(DescriptorPoolCreateInfo);

    vk::DescriptorSetAllocateInfo DescriptorSetAllocInfo(DescriptorPool, 1, &DescriptorSetLayout);
    std::vector<vk::DescriptorSet> const DescriptorSets =
        Device.allocateDescriptorSets(DescriptorSetAllocInfo);
    vk::DescriptorSet DescriptorSet = DescriptorSets.front();
    vk::DescriptorBufferInfo InBufferInfo(InBuffer, 0, NumElements * sizeof(int32_t));
    vk::DescriptorBufferInfo OutBufferInfo(OutBuffer, 0, NumElements * sizeof(int32_t));

    std::vector<vk::WriteDescriptorSet> const WriteDescriptorSets = {
        {DescriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &InBufferInfo},
        {DescriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &OutBufferInfo},
    };
    Device.updateDescriptorSets(WriteDescriptorSets, {});

    vk::CommandPoolCreateInfo CommandPoolCreateInfo(vk::CommandPoolCreateFlags(),
                                                    ComputeQueueFamilyIndex);
    vk::CommandPool CommandPool = Device.createCommandPool(CommandPoolCreateInfo);

    vk::CommandBufferAllocateInfo CommandBufferAllocInfo(CommandPool,  // Command Pool
                                                         vk::CommandBufferLevel::ePrimary,  // Level
                                                         1);  // Num Command Buffers
    std::vector<vk::CommandBuffer> const CmdBuffers =
        Device.allocateCommandBuffers(CommandBufferAllocInfo);
    vk::CommandBuffer CmdBuffer = CmdBuffers.front();

    vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    CmdBuffer.begin(CmdBufferBeginInfo);
    CmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, ComputePipeline);
    CmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,  // Bind point
                                 PipelineLayout,                   // Pipeline Layout
                                 0,                                // First descriptor set
                                 {DescriptorSet},                  // List of descriptor sets
                                 {});                              // Dynamic offsets
    CmdBuffer.dispatch(NumElements, 1, 1);
    CmdBuffer.end();

    vk::Queue Queue = Device.getQueue(ComputeQueueFamilyIndex, 0);
    vk::Fence Fence = Device.createFence(vk::FenceCreateInfo());

    vk::SubmitInfo SubmitInfo(0,            // Num Wait Semaphores
                              nullptr,      // Wait Semaphores
                              nullptr,      // Pipeline Stage Flags
                              1,            // Num Command Buffers
                              &CmdBuffer);  // List of command buffers
    Queue.submit({SubmitInfo}, Fence);
    auto result = Device.waitForFences({Fence},        // List of fences
                                       true,           // Wait All
                                       uint64_t(-1));  // Timeout

    vmaMapMemory(Allocator, InBufferAllocation, reinterpret_cast<void**>(&InBufferPtr));
    for (uint32_t I = 0; I < NumElements; ++I) {
        std::cout << InBufferPtr[I] << " ";
    }
    std::cout << std::endl;
    vmaUnmapMemory(Allocator, InBufferAllocation);

    int32_t* OutBufferPtr = nullptr;
    vmaMapMemory(Allocator, OutBufferAllocation, reinterpret_cast<void**>(&OutBufferPtr));
    for (uint32_t I = 0; I < NumElements; ++I) {
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
    auto AllocateBuffer = [Allocator, ComputeQueueFamilyIndex](size_t SizeInBytes,
                                                               VmaMemoryUsage Usage) {
        vk::BufferCreateInfo BufferCreateInfo{
            vk::BufferCreateFlags(),                  // Flags
            SizeInBytes,                              // Size
            vk::BufferUsageFlagBits::eStorageBuffer,  // Usage
            vk::SharingMode::eExclusive,              // Sharing mode
            1,                                        // Number of queue family indices
            &ComputeQueueFamilyIndex                  // List of queue family indices
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

    Device.resetCommandPool(CommandPool, vk::CommandPoolResetFlags());
    Device.destroyFence(Fence);
    Device.destroyDescriptorSetLayout(DescriptorSetLayout);
    Device.destroyPipelineLayout(PipelineLayout);
    Device.destroyPipelineCache(PipelineCache);
    Device.destroyShaderModule(ShaderModule);
    Device.destroyPipeline(ComputePipeline);
    Device.destroyDescriptorPool(DescriptorPool);
    Device.destroyCommandPool(CommandPool);
    Device.destroy();
    Instance.destroy();

    return 0;
    MY_CATCH_RTI
}
