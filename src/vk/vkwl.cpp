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

using Uid = toolkit::Uid;
using Event = mywl::EventData;
using EventType = mywl::EventType;

static std::atomic<int> g_app_still_run = kMaxAppInstance;

// NOLINTBEGIN
enum AppUid : int
{
    UID_vkShader_vert,
    UID_vkShader_frag,

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
    UID_vkImage_texture_0,
    UID_vkImage_texture_end = UID_vkImage_texture_0 + kMaxAppInstance,
    UID_vkImageView_texture_0,
    UID_vkImageView_texture_end = UID_vkImageView_texture_0 + kMaxAppInstance,
    UID_vkSampler_texture_0,
    UID_vkSampler_texture_end = UID_vkSampler_texture_0 + kMaxAppInstance,
    UID_vkBuffer_vertex_0,
    UID_vkBuffer_vertex_end = UID_vkBuffer_vertex_0 + kMaxAppInstance,
    UID_vkBuffer_index_0,
    UID_vkBuffer_index_end = UID_vkBuffer_index_0 + kMaxAppInstance,
    UID_vkRenderPass_0,
    UID_vkRenderPass_end = UID_vkRenderPass_0 + kMaxAppInstance,
    UID_vkSwapchain_0,
    UID_vkSwapchain_end = UID_vkSwapchain_0 + kMaxAppInstance,

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

    UID_wlSurface_0,
    UID_wlSurface_end = UID_wlSurface_0 + kMaxAppInstance,
};

// NOLINTEND

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
        CHECK_ERR_RET(
            vk_->createDescriptorPool(UID_vkDescriptorPool_0 + id_, 1000,
                                      {
                                          {vk::DescriptorType::eSampler, 1000},
                                          {vk::DescriptorType::eCombinedImageSampler, 1000},
                                          {vk::DescriptorType::eSampledImage, 1000},
                                          {vk::DescriptorType::eStorageImage, 1000},
                                          {vk::DescriptorType::eUniformTexelBuffer, 1000},
                                          {vk::DescriptorType::eStorageTexelBuffer, 1000},
                                          {vk::DescriptorType::eUniformBuffer, 1000},
                                          {vk::DescriptorType::eStorageBuffer, 1000},
                                          {vk::DescriptorType::eUniformBufferDynamic, 1000},
                                          {vk::DescriptorType::eStorageBufferDynamic, 1000},
                                          {vk::DescriptorType::eInputAttachment, 1000},
                                      },
                                      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet));

        return MyErrCode::kOk;
    }

    MyErrCode destroy()
    {
        CHECK_ERR_RET(sc_.destroy());
        return MyErrCode::kOk;
    }

    MyErrCode recreateSwapchain() { return MyErrCode::kOk; }

    MyErrCode drawLoop()
    {
        auto exit_guard = toolkit::scopeExit([] { g_app_still_run -= 1; });
        quit_ = false;
        while (!quit_) {
            // vk render...
            // ...
            // ...
            CHECK_ERR_RET(handleEvent());
        }

        return MyErrCode::kOk;
    }

    MyErrCode handleEvent()
    {
        Event ev;
        while (evq_.try_dequeue(ev)) {
            ILOG("{} receive event {}", id_, static_cast<int>(ev.type));
            switch (ev.type) {
                case EventType::kSurfaceClose: {
                    quit_ = true;
                    break;
                }
                case EventType::kSurfaceResize: {
                    CHECK_ERR_RET(sc_.onSurfaceResize(ev.ix, ev.iy));
                    break;
                }
                case EventType::kPointerMove: {
                    CHECK_ERR_RET(sc_.onPointerMove(ev.dx, ev.dy));
                    break;
                }
                case EventType::kPointerPress: {
                    CHECK_ERR_RET(sc_.onPointerPress(ev.ix, ev.iy));
                    break;
                }
                case EventType::kPointerScroll: {
                    CHECK_ERR_RET(sc_.onPointerScroll(ev.dx, ev.dy));
                    break;
                }
                case EventType::kKeyboardPress: {
                    CHECK_ERR_RET(sc_.onKeyboardPress(ev.ux, ev.ix, quit_));
                    break;
                }
                default:
                    break;
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
    bool quit_;
};

class AppEventHandler: public mywl::EventHandler
{
public:
    MyErrCode onEvent(Uid surface_id, mywl::EventData const& event) override
    {
        ILOG("on event {} {}", surface_id, static_cast<int>(event.type));
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

    CHECK_ERR_RET(vk.createInstance(
        "vkwl", {"VK_EXT_debug_utils", "VK_KHR_surface", "VK_KHR_wayland_surface"}));
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
    CHECK_ERR_RET(vk.createDeviceAndQueues({"VK_KHR_swapchain"}, {{queue_family, queue_ids}}));
    CHECK_ERR_RET(vk.createAllocator());
    CHECK_ERR_RET(
        vk.createShaderModule(UID_vkShader_vert, toolkit::getDataDir() / "vkwl.vert.spv"));
    CHECK_ERR_RET(
        vk.createShaderModule(UID_vkShader_frag, toolkit::getDataDir() / "vkwl.frag.spv"));

    std::vector<std::thread> app_threads;
    for (int i = 0; i < kMaxAppInstance; ++i) {
        wl_surface* wl_surface = wl.getSurface(UID_wlSurface_0 + i);
        VkSurfaceKHR vk_surface;
        VkWaylandSurfaceCreateInfoKHR surface_ci = {};
        surface_ci.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
        surface_ci.display = wl.getDisplay();
        surface_ci.surface = wl_surface;
        CHECK_VK_RET(
            vkCreateWaylandSurfaceKHR(vk.getInstance(), &surface_ci, nullptr, &vk_surface));
        CHECK_ERR_RET(vk.createSurface(UID_vkSurface_0 + i, vk_surface));
        CHECK_ERR_RET(apps[i].init(wl_surface, &vk, toolkit::getDataDir() / "lyran.obj",
                                   toolkit::getDataDir() / "lyran-diffuse.jpg"));
        app_threads.emplace_back([&apps, i] { apps[i].drawLoop(); });
    }

    while (g_app_still_run > 0) {
        CHECK_ERR_RET(wl.dispatch());
    }

    for (int i = 0; i < kMaxAppInstance; ++i) {
        app_threads[i].join();
        CHECK_ERR_RET(apps[i].destroy());
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
