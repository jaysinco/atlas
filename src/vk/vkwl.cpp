#include "vulkan-helper.h"
#include "wayland-helper.h"
#include "scene.h"
#include "toolkit/args.h"
#include "toolkit/logging.h"
#include <thread>
#define BUILD_WITH_EASY_PROFILER
#define USING_EASY_PROFILER
#include <easy/profiler.h>
#include <easy/arbitrary_value.h>

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

MY_MAIN
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.parse();
    EASY_MAIN_THREAD;
    profiler::startListen(28077);
    // Application::run("vkwl", "vkwl");
    ILOG("end!");
    return MyErrCode::kOk;
    MY_CATCH_RET
}
