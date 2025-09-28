#include "vulkan-helper.h"
#include "wayland-helper.h"
#include "scene.h"
#include "toolkit/args.h"
#include "toolkit/logging.h"
#include <thread>
#include <moodycamel/concurrentqueue.h>
#define BUILD_WITH_EASY_PROFILER
#define USING_EASY_PROFILER
#include <easy/profiler.h>
#include <easy/arbitrary_value.h>

static constexpr int kMaxAppInstance = 2;
static constexpr int kMaxSwapchainImage = 10;
static constexpr int kMaxFrameInfight = 2;

// NOLINTBEGIN
enum AppUid : int
{

    UID_vkQueue_0,
    UID_vkQueue_end = UID_vkQueue_0 + kMaxAppInstance,
    UID_vkCommandPool_0,
    UID_vkCommandPool_end = UID_vkCommandPool_0 + kMaxAppInstance,
    UID_vkDescriptorPool_0,
    UID_vkDescriptorPool_end = UID_vkDescriptorPool_0 + kMaxAppInstance,
    UID_vkDescriptorSetLayout_0,
    UID_vkDescriptorSetLayout_end = UID_vkDescriptorSetLayout_0 + kMaxAppInstance,
    UID_vkPipelineLayout_0,
    UID_vkPipelineLayout_end = UID_vkPipelineLayout_0 + kMaxAppInstance,
    UID_vkPipeline_0,
    UID_vkPipeline_end = UID_vkPipeline_0 + kMaxAppInstance,
    UID_vkSurface_0,
    UID_vkSurface_end = UID_vkSurface_0 + kMaxAppInstance,

    UID_vkImage_color_0,
    UID_vkImage_color_0_end = UID_vkImage_color_0 + kMaxAppInstance * kMaxSwapchainImage,
    UID_vkImageView_color_0,
    UID_vkImageView_color_end = UID_vkImageView_color_0 + kMaxAppInstance * kMaxSwapchainImage,
    UID_vkImage_depth_0,
    UID_vkImage_depth_0_end = UID_vkImage_depth_0 + kMaxAppInstance * kMaxSwapchainImage,
    UID_vkImageView_depth_0,
    UID_vkImageView_depth_end = UID_vkImageView_depth_0 + kMaxAppInstance * kMaxSwapchainImage,
    UID_vkFrameBuffer_0,
    UID_vkFrameBuffer_end = UID_vkFrameBuffer_0 + kMaxAppInstance * kMaxSwapchainImage,
    UID_vkSemaphore_render_finished_0,
    UID_vkSemaphore_render_finished_end =
        UID_vkSemaphore_render_finished_0 + kMaxAppInstance * kMaxSwapchainImage,

    UID_vkCommandBuffer_0,
    UID_vkCommandBuffer_end = UID_vkCommandBuffer_0 + kMaxAppInstance * kMaxFrameInfight,
    UID_vkSemaphore_image_available_0,
    UID_vkSemaphore_image_available_end =
        UID_vkSemaphore_image_available_0 + kMaxAppInstance * kMaxFrameInfight,
    UID_vkFence_inflight_0,
    UID_vkFence_inflight_end = UID_vkFence_inflight_0 + kMaxAppInstance * kMaxFrameInfight,
    UID_vkBuffer_uniform_0,
    UID_vkBuffer_uniform_end = UID_vkBuffer_uniform_0 + kMaxAppInstance * kMaxFrameInfight,
    UID_vkDescriptorSet_0,
    UID_vkDescriptorSet_end = UID_vkDescriptorSet_0 + kMaxAppInstance * kMaxFrameInfight,

    UID_vkShader_vert,
    UID_vkShader_frag,

    UID_wlSurface_0,
    UID_wlSurface_end = UID_wlSurface_0 + kMaxAppInstance,
};

// NOLINTEND

using Uid = toolkit::Uid;
using Event = mywl::EventData;

class AppInstance
{
public:
    explicit AppInstance(int id): id_(id) {}

    MyErrCode init(wl_surface* wls, myvk::Context* vk, std::filesystem::path const& model_path,
                   std::filesystem::path const& texture_path)
    {
        wls_ = wls;
        vk_ = vk;

        CHECK_ERR_RET(vk_->createCommandPool(UID_vkCommandPool_0 + id_, UID_vkQueue_0 + id_,
                                             vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
                                                 vk::CommandPoolCreateFlagBits::eTransient));
        CHECK_ERR_RET(vk_->createDescriptorPool(
            UID_vkDescriptorPool_0 + id_, 2, {{vk::DescriptorType::eStorageBuffer, 4}},
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet));

        return MyErrCode::kOk;
    }

    MyErrCode recreateSwapchain() { return MyErrCode::kOk; }

    MyErrCode drawLoop()
    {
        while (true) {
            Event event;
            while (evq_.try_dequeue(event)) {
            }
        }
        return MyErrCode::kOk;
    }

    MyErrCode postEvent(Event const& event)
    {
        evq_.enqueue(event);
        return MyErrCode::kOk;
    }

private:
    int id_;
    wl_surface* wls_;
    myvk::Context* vk_;
    moodycamel::ConcurrentQueue<Event> evq_;
    scene::Scene sc_;
};

class AppEventHandler: public mywl::EventHandler
{
public:
    MyErrCode onEvent(Uid surface_id, mywl::EventData const& event) override
    {
        CHECK_ERR_RET(app_surfaces_.at(surface_id)->postEvent(event));
        return MyErrCode::kOk;
    }

    MyErrCode registerAppSurface(Uid surface_id, AppInstance* app)
    {
        app_surfaces_[surface_id] = app;
        return MyErrCode::kOk;
    }

private:
    std::map<Uid, AppInstance*> app_surfaces_;
};

MyErrCode run()

{
    mywl::Context wl;
    myvk::Context vk;

    std::vector<AppInstance> apps;
    for (int i = 0; i < kMaxAppInstance; ++i) {
        apps.emplace_back(i);
    }

    AppEventHandler app_event;
    CHECK_ERR_RET(wl.createDisplay(&app_event));
    for (int i = 0; i < kMaxAppInstance; ++i) {
        CHECK_ERR_RET(app_event.registerAppSurface(UID_wlSurface_0 + i, &apps[i]));
        CHECK_ERR_RET(wl.createSurface(UID_wlSurface_0 + i, FSTR("vkwl{}", i), FSTR("vkwl{}", i)));
    }

    CHECK_ERR_RET(vk.createInstance("vkwl", {"VK_EXT_debug_utils"}));
    CHECK_ERR_RET(vk.createPhysicalDevice());
    uint32_t queue_family;
    auto queue_picker = [](uint32_t family_index, vk::QueueFamilyProperties const& prop) -> bool {
        return static_cast<bool>(prop.queueFlags & vk::QueueFlagBits::eGraphics);
    };
    CHECK_ERR_RET(vk.pickQueueFamily(queue_picker, queue_family));
    std::set<Uid> queue_ids;
    for (int i = 0; i < kMaxAppInstance; ++i) {
        queue_ids.insert(UID_vkQueue_0 + i);
    }
    CHECK_ERR_RET(vk.createDeviceAndQueues({}, {{queue_family, queue_ids}}));
    CHECK_ERR_RET(vk.createAllocator());

    std::vector<std::thread> app_threads;
    for (int i = 0; i < kMaxAppInstance; ++i) {
        wl_surface* surface = wl.getRawSurface(UID_wlSurface_0 + i);
        CHECK_ERR_RET(apps[i].init(surface, &vk, "model_path", "texture_path"));
        app_threads.emplace_back([&apps, i] { apps[i].drawLoop(); });
    }

    while (true) {
        CHECK_ERR_RET(wl.dispatch());
    }

    for (int i = 0; i < kMaxAppInstance; ++i) {
        app_threads[i].join();
    }

    CHECK_ERR_RET(vk.destroy());
    CHECK_ERR_RET(wl.destroy());
    return MyErrCode::kOk;
}

MY_MAIN
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.parse();
    EASY_MAIN_THREAD;
    profiler::startListen(28077);
    CHECK_ERR_RET(run());
    ILOG("end!");
    return MyErrCode::kOk;
    MY_CATCH_RET
}
