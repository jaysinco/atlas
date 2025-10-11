#include "vulkan-helper.h"
#include "wayland-helper.h"
#include "scene.h"
#include "toolkit/args.h"
#include "toolkit/logging.h"
#include <thread>
#include <moodycamel/concurrentqueue.h>
#include <imgui/backends/imgui_impl_vulkan.h>
#define BUILD_WITH_EASY_PROFILER
#define USING_EASY_PROFILER
#include <easy/profiler.h>
#include <easy/arbitrary_value.h>

static constexpr int kMaxAppInstance = 1;
static constexpr int kMaxSwapchainImage = 10;
static constexpr int kMaxFrameInfight = 2;

using Uid = toolkit::Uid;
using Event = mywl::EventData;
using EventType = mywl::EventType;

thread_local ImGuiContext* MyImGuiTLS;
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

    UID_scModel_0,
    UID_scModel_end = UID_scModel_0 + kMaxAppInstance,
    UID_scTexture_0,
    UID_scTexture_end = UID_scTexture_0 + kMaxAppInstance,
};

// NOLINTEND

class AppInstance
{
public:
    explicit AppInstance(int id): id_(id) {}

    MyErrCode init(myvk::Context* vk, int init_width, int init_height,
                   std::filesystem::path const& model_path,
                   std::filesystem::path const& texture_path)
    {
        vk_ = vk;
        curr_width_ = init_width;
        curr_height_ = init_height;
        swapchain_image_format_ = vk::Format::eB8G8R8A8Srgb;
        depth_image_format_ = vk::Format::eD32SfloatS8Uint;
        msaa_sample_count_ = vk::SampleCountFlagBits::e4;

        CHECK_ERR_RET(createRenderPass());
        CHECK_ERR_RET(createSwapImages(false));

        CHECK_ERR_RET(vk_->createDescriptorSetLayout(
            UID_vkDescriptorSetLayout_0 + id_,
            {
                {vk::DescriptorType::eUniformBuffer,
                 vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment},
                {vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment},
            }));

        CHECK_ERR_RET(vk_->createPipelineLayout(UID_vkPipelineLayout_0 + id_,
                                                {UID_vkDescriptorSetLayout_0 + id_}));
        CHECK_ERR_RET(createPipeline(false));

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

        // upload texture
        CHECK_ERR_RET(sc_.createTexture(UID_scTexture_0 + id_, texture_path));
        auto& texture = sc_.getTexture(UID_scTexture_0 + id_);
        auto texture_size = texture.getSize();
        auto texture_mip_levels = texture.getMaxMipLevels();

        CHECK_ERR_RET(vk_->createImage(
            UID_vkImage_texture_0 + id_,
            {
                .format = vk::Format::eB8G8R8A8Srgb,
                .aspects = vk::ImageAspectFlagBits::eColor,
                .extent = {texture_size.first, texture_size.second, 1},
                .mip_levels = texture_mip_levels,
                .samples = vk::SampleCountFlagBits::e1,
                .usages = vk::ImageUsageFlagBits::eTransferSrc |
                          vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            },
            vk::MemoryPropertyFlagBits::eDeviceLocal));

        CHECK_ERR_RET(vk_->copyHostToImage(UID_vkQueue_0 + id_, UID_vkCommandPool_0 + id_,
                                           texture.getData().data(), UID_vkImage_texture_0 + id_,
                                           vk::ImageLayout::eUndefined));

        texture.unload();

        CHECK_ERR_RET(vk_->generateMipmaps(
            UID_vkQueue_0 + id_, UID_vkCommandPool_0 + id_, UID_vkImage_texture_0 + id_,
            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal));

        CHECK_ERR_RET(
            vk_->createImageView(UID_vkImageView_texture_0 + id_, UID_vkImage_texture_0 + id_, {}));

        CHECK_ERR_RET(vk_->createSampler(
            UID_vkSampler_texture_0 + id_,
            {.min_lod = 0.0f, .max_lod = static_cast<float>(texture_mip_levels)}));

        // upload model
        CHECK_ERR_RET(sc_.createModel(UID_scModel_0 + id_, model_path));
        auto& model = sc_.getModel(UID_scModel_0 + id_);
        uint64_t model_vertex_size = model.getVertices().size() * sizeof(scene::Vertex);
        uint64_t model_index_size = model.getIndices().size() * sizeof(uint32_t);

        CHECK_ERR_RET(vk_->createBuffer(
            UID_vkBuffer_vertex_0 + id_,
            {model_vertex_size,
             vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer},
            vk::MemoryPropertyFlagBits::eDeviceLocal));

        CHECK_ERR_RET(vk_->copyHostToBuffer(UID_vkQueue_0 + id_, UID_vkCommandPool_0 + id_,
                                            model.getVertices().data(),
                                            UID_vkBuffer_vertex_0 + id_));

        CHECK_ERR_RET(vk_->createBuffer(
            UID_vkBuffer_index_0 + id_,
            {model_index_size,
             vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer},
            vk::MemoryPropertyFlagBits::eDeviceLocal));

        CHECK_ERR_RET(vk_->copyHostToBuffer(UID_vkQueue_0 + id_, UID_vkCommandPool_0 + id_,
                                            model.getIndices().data(), UID_vkBuffer_index_0 + id_));

        model.unload();

        // group
        for (int i = 0; i < swapchain_image_count_; ++i) {
            int id_offset = kMaxSwapchainImage * id_ + i;
            CHECK_ERR_RET(
                vk_->createBinarySemaphore(UID_vkSemaphore_render_finished_0 + id_offset));
        }

        for (int i = 0; i < kMaxFrameInfight; ++i) {
            int id_offset = kMaxFrameInfight * id_ + i;
            CHECK_ERR_RET(
                vk_->createBinarySemaphore(UID_vkSemaphore_image_available_0 + id_offset));
            CHECK_ERR_RET(vk_->createFence(UID_vkFence_inflight_0 + id_offset, true));
            CHECK_ERR_RET(vk_->createCommandBuffer(UID_vkCommandBuffer_0 + id_offset,
                                                   UID_vkCommandPool_0 + id_));

            CHECK_ERR_RET(vk_->createBuffer(
                UID_vkBuffer_uniform_0 + id_offset,
                {sizeof(scene::UniformBuffer), vk::BufferUsageFlagBits::eUniformBuffer},
                vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent,
                VMA_ALLOCATION_CREATE_MAPPED_BIT));

            CHECK_ERR_RET(vk_->createDescriptorSet(UID_vkDescriptorSet_0 + id_offset,
                                                   UID_vkDescriptorSetLayout_0 + id_,
                                                   UID_vkDescriptorPool_0 + id_));

            CHECK_ERR_RET(vk_->updateDescriptorSet(
                UID_vkDescriptorSet_0 + id_offset,
                {
                    {0, UID_vkBuffer_uniform_0 + id_offset},
                    {1,
                     {UID_vkImageView_texture_0 + id_, UID_vkSampler_texture_0 + id_,
                      vk::ImageLayout::eShaderReadOnlyOptimal}},
                }));
        }

        CHECK_ERR_RET(setupImgui());

        return MyErrCode::kOk;
    }

    MyErrCode destroy()
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui::DestroyContext();
        CHECK_ERR_RET(sc_.destroy());
        return MyErrCode::kOk;
    }

    MyErrCode setupImgui()
    {
        ImGui::SetCurrentContext(ImGui::CreateContext());

        ImGuiIO& io = ImGui::GetIO();
        io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;

        ImGuiStyle& style = ImGui::GetStyle();
        style.Alpha = 1.0;
        style.ButtonTextAlign = ImVec2(0.0, 0.0);

        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = vk_->getInstance();
        init_info.PhysicalDevice = vk_->getPhysicalDevice();
        init_info.Device = vk_->getDevice();
        init_info.Queue = vk_->getQueue(UID_vkQueue_0 + id_);
        init_info.DescriptorPool = vk_->getDescriptorPool(UID_vkDescriptorPool_0 + id_);
        init_info.MinImageCount = swapchain_image_count_;
        init_info.ImageCount = swapchain_image_count_;
        init_info.MSAASamples = static_cast<VkSampleCountFlagBits>(msaa_sample_count_);
        ImGui_ImplVulkan_Init(&init_info, vk_->getRenderPass(UID_vkRenderPass_0 + id_));

        CHECK_ERR_RET(vk_->oneTimeSubmit(UID_vkQueue_0 + id_, UID_vkCommandPool_0 + id_,
                                         [&](myvk::CommandBuffer& cmd) -> MyErrCode {
                                             ImGui_ImplVulkan_CreateFontsTexture(cmd);
                                             return MyErrCode::kOk;
                                         }));

        ImGui_ImplVulkan_DestroyFontUploadObjects();

        return MyErrCode::kOk;
    }

    MyErrCode createRenderPass()
    {
        vk::AttachmentDescription color_attach{{},
                                               swapchain_image_format_,
                                               msaa_sample_count_,
                                               vk::AttachmentLoadOp::eClear,
                                               vk::AttachmentStoreOp::eStore,
                                               vk::AttachmentLoadOp::eDontCare,
                                               vk::AttachmentStoreOp::eDontCare,
                                               vk::ImageLayout::eUndefined,
                                               vk::ImageLayout::eColorAttachmentOptimal};

        vk::AttachmentReference color_attach_ref{0, vk::ImageLayout::eColorAttachmentOptimal};

        vk::AttachmentDescription depth_attach{{},
                                               depth_image_format_,
                                               msaa_sample_count_,
                                               vk::AttachmentLoadOp::eClear,
                                               vk::AttachmentStoreOp::eDontCare,
                                               vk::AttachmentLoadOp::eDontCare,
                                               vk::AttachmentStoreOp::eDontCare,
                                               vk::ImageLayout::eUndefined,
                                               vk::ImageLayout::eDepthStencilAttachmentOptimal};

        vk::AttachmentReference depth_attach_ref{1,
                                                 vk::ImageLayout::eDepthStencilAttachmentOptimal};

        vk::AttachmentDescription resolve_attach{{},
                                                 swapchain_image_format_,
                                                 vk::SampleCountFlagBits::e1,
                                                 vk::AttachmentLoadOp::eDontCare,
                                                 vk::AttachmentStoreOp::eStore,
                                                 vk::AttachmentLoadOp::eDontCare,
                                                 vk::AttachmentStoreOp::eDontCare,
                                                 vk::ImageLayout::eUndefined,
                                                 vk::ImageLayout::ePresentSrcKHR};

        vk::AttachmentReference resolve_attach_ref{2, vk::ImageLayout::eColorAttachmentOptimal};

        myvk::RenderPassMeta meta;
        meta.attachments = {color_attach, depth_attach, resolve_attach};
        meta.attachment_refs = {color_attach_ref, depth_attach_ref, resolve_attach_ref};

        vk::SubpassDescription subpass{{},
                                       vk::PipelineBindPoint::eGraphics,
                                       0,
                                       nullptr,
                                       1,
                                       &meta.attachment_refs[0],
                                       &meta.attachment_refs[2],
                                       &meta.attachment_refs[1]};

        vk::SubpassDependency dependency{vk::SubpassExternal,
                                         0,
                                         vk::PipelineStageFlagBits::eColorAttachmentOutput |
                                             vk::PipelineStageFlagBits::eEarlyFragmentTests,
                                         vk::PipelineStageFlagBits::eColorAttachmentOutput |
                                             vk::PipelineStageFlagBits::eEarlyFragmentTests,
                                         vk::AccessFlagBits::eNone,
                                         vk::AccessFlagBits::eColorAttachmentWrite |
                                             vk::AccessFlagBits::eDepthStencilAttachmentWrite};

        meta.subpasses = {subpass};
        meta.dependencies = {dependency};

        CHECK_ERR_RET(vk_->createRenderPass(UID_vkRenderPass_0 + id_, meta));
        return MyErrCode::kOk;
    }

    MyErrCode createSwapImages(bool recreate)
    {
        if (recreate) {
            CHECK_ERR_RET(destroySwapImages());
        }

        CHECK_ERR_RET(vk_->createSwapchain(
            UID_vkSwapchain_0 + id_, UID_vkSurface_0 + id_,
            {.surface_format = {swapchain_image_format_, vk::ColorSpaceKHR::eSrgbNonlinear},
             .extent = {curr_width_, curr_height_}}));

        auto& swapchain = vk_->getSwapchain(UID_vkSwapchain_0 + id_);
        swapchain_image_count_ = swapchain.getImageCount();

        for (int i = 0; i < swapchain_image_count_; ++i) {
            int id_offset = kMaxSwapchainImage * id_ + i;
            CHECK_ERR_RET(
                vk_->createImage(UID_vkImage_color_0 + id_offset,
                                 {
                                     .format = swapchain_image_format_,
                                     .aspects = vk::ImageAspectFlagBits::eColor,
                                     .extent = {curr_width_, curr_height_, 1},
                                     .mip_levels = 1,
                                     .samples = msaa_sample_count_,
                                     .usages = vk::ImageUsageFlagBits::eTransientAttachment |
                                               vk::ImageUsageFlagBits::eColorAttachment,
                                 },
                                 vk::MemoryPropertyFlagBits::eDeviceLocal));

            CHECK_ERR_RET(vk_->createImageView(UID_vkImageView_color_0 + id_offset,
                                               UID_vkImage_color_0 + id_offset, {}));

            CHECK_ERR_RET(
                vk_->createImage(UID_vkImage_depth_0 + id_offset,
                                 {
                                     .format = depth_image_format_,
                                     .aspects = vk::ImageAspectFlagBits::eDepth,
                                     .extent = {curr_width_, curr_height_, 1},
                                     .mip_levels = 1,
                                     .samples = msaa_sample_count_,
                                     .usages = vk::ImageUsageFlagBits::eDepthStencilAttachment,
                                 },
                                 vk::MemoryPropertyFlagBits::eDeviceLocal));

            CHECK_ERR_RET(vk_->createImageView(UID_vkImageView_depth_0 + id_offset,
                                               UID_vkImage_depth_0 + id_offset, {}));

            CHECK_ERR_RET(vk_->createFramebuffer(
                UID_vkFrameBuffer_0 + id_offset, UID_vkRenderPass_0 + id_,
                {UID_vkImageView_color_0 + id_offset, UID_vkImageView_depth_0 + id_offset,
                 swapchain.getImageViewId(i)}));
        }
        return MyErrCode::kOk;
    }

    MyErrCode destroySwapImages()
    {
        CHECK_ERR_RET(vk_->waitQueueIdle(UID_vkQueue_0 + id_));
        for (int i = 0; i < swapchain_image_count_; ++i) {
            int id_offset = kMaxSwapchainImage * id_ + i;
            CHECK_ERR_RET(vk_->destroyFramebuffer(UID_vkFrameBuffer_0 + id_offset));
            CHECK_ERR_RET(vk_->destroyImageView(UID_vkImageView_depth_0 + id_offset));
            CHECK_ERR_RET(vk_->destroyImage(UID_vkImage_depth_0 + id_offset));
            CHECK_ERR_RET(vk_->destroyImageView(UID_vkImageView_color_0 + id_offset));
            CHECK_ERR_RET(vk_->destroyImage(UID_vkImage_color_0 + id_offset));
        }
        CHECK_ERR_RET(vk_->destroySwapchain(UID_vkSwapchain_0 + id_));
        return MyErrCode::kOk;
    }

    MyErrCode createPipeline(bool recreate)
    {
        if (recreate) {
            CHECK_ERR_RET(vk_->waitQueueIdle(UID_vkQueue_0 + id_));
            CHECK_ERR_RET(vk_->destroyPipeline(UID_vkPipeline_0 + id_));
        }

        vk::VertexInputBindingDescription input_bind{0, sizeof(scene::Vertex),
                                                     vk::VertexInputRate::eVertex};

        std::vector<vk::VertexInputAttributeDescription> input_attrs(3);
        input_attrs[0] = {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(scene::Vertex, pos)};
        input_attrs[1] = {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(scene::Vertex, normal)};
        input_attrs[2] = {2, 0, vk::Format::eR32G32Sfloat, offsetof(scene::Vertex, tex_coord)};

        auto& gui_state = sc_.getGuiState();
        CHECK_ERR_RET(vk_->createGraphicPipeline(
            UID_vkPipeline_0 + id_,
            {
                .pipeline_layout_id = UID_vkPipelineLayout_0 + id_,
                .render_pass_id = UID_vkRenderPass_0 + id_,
                .vert_shader_id = UID_vkShader_vert,
                .frag_shader_id = UID_vkShader_frag,
                .polygon_mode =
                    gui_state.wire_frame ? vk::PolygonMode::eLine : vk::PolygonMode::eFill,
                .front_face = gui_state.face_clockwise ? vk::FrontFace::eClockwise
                                                       : vk::FrontFace::eCounterClockwise,
                .raster_samples = msaa_sample_count_,
                .vert_input_binds = {input_bind},
                .vert_input_attrs = input_attrs,
            }));
        return MyErrCode::kOk;
    }

    MyErrCode recordFrame(myvk::CommandBuffer& cmd, int curr_frame, int image_index)
    {
        std::vector<vk::ClearValue> clear_values(2);
        clear_values[0] = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f};
        clear_values[1] = vk::ClearDepthStencilValue{1.0f, 0};

        cmd.beginRenderPass(
            UID_vkRenderPass_0 + id_, UID_vkFrameBuffer_0 + kMaxSwapchainImage * id_ + image_index,
            {{0, 0}, {curr_width_, curr_height_}}, clear_values, vk::SubpassContents::eInline);

        vk::Viewport viewport{
            0.0f, 0.0f, static_cast<float>(curr_width_), static_cast<float>(curr_height_),
            0.0f, 1.0f};
        cmd.setViewport(0, viewport);

        vk::Rect2D scissor{{0, 0}, {curr_width_, curr_height_}};
        cmd.setScissor(0, scissor);

        cmd.bindPipeline(UID_vkPipeline_0 + id_, vk::PipelineBindPoint::eGraphics);
        cmd.bindVertexBuffer(UID_vkBuffer_vertex_0 + id_);
        cmd.bindIndexBuffer(UID_vkBuffer_index_0 + id_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, UID_vkPipelineLayout_0 + id_,
                               {UID_vkDescriptorSet_0 + kMaxFrameInfight * id_ + curr_frame});

        cmd.drawIndexed(sc_.getModel(UID_scModel_0 + id_).getIndicesCount(), 1, 0, 0, 0);

        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

        cmd.endRenderPass();
        return MyErrCode::kOk;
    }

    MyErrCode drawLoop()
    {
        quit_ = false;
        int curr_frame = 0;
        auto last_time = std::chrono::high_resolution_clock::now();

        while (!quit_) {
            int inflight_frame_id_offset = kMaxFrameInfight * id_ + curr_frame;

            CHECK_ERR_RET(vk_->waitFences({UID_vkFence_inflight_0 + inflight_frame_id_offset}));

            uint32_t image_index;
            bool recreate_swapchain;
            CHECK_ERR_RET(vk_->acquireNextImage(
                UID_vkSwapchain_0 + id_, image_index, recreate_swapchain,
                UID_vkSemaphore_image_available_0 + inflight_frame_id_offset));
            if (recreate_swapchain) {
                ILOG("acquire next image need recreate swapchain");
                CHECK_ERR_RET(createSwapImages(true));
                continue;
            }

            int swap_image_id_offset = kMaxSwapchainImage * id_ + image_index;

            CHECK_ERR_RET(vk_->resetFences({UID_vkFence_inflight_0 + inflight_frame_id_offset}));

            auto curr_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> delta_duration = curr_time - last_time;
            float delta_time = delta_duration.count();
            last_time = curr_time;

            ImGuiIO& io = ImGui::GetIO();
            io.DisplaySize = ImVec2(curr_width_, curr_height_);
            io.DeltaTime = delta_time;
            ImGui_ImplVulkan_NewFrame();
            ImGui::NewFrame();
            bool recreate_pipeline = false;
            CHECK_ERR_RET(sc_.onFrameDraw(recreate_pipeline));
            if (recreate_pipeline) {
                CHECK_ERR_RET(createPipeline(true));
            }
            ImGui::Render();

            void* uniform_mapped_data =
                vk_->getBuffer(UID_vkBuffer_uniform_0 + inflight_frame_id_offset).getMappedData();
            scene::UniformBuffer uniform_data = sc_.getUniformBuffer(UID_scModel_0 + id_);
            memcpy(uniform_mapped_data, &uniform_data, sizeof(scene::UniformBuffer));

            CHECK_ERR_RET(vk_->recordCommand(
                UID_vkCommandBuffer_0 + inflight_frame_id_offset,
                vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
                [&](myvk::CommandBuffer& cmd) { return recordFrame(cmd, curr_frame, image_index); },
                true));

            CHECK_ERR_RET(
                vk_->submit(UID_vkQueue_0 + id_, UID_vkCommandBuffer_0 + inflight_frame_id_offset,
                            {{UID_vkSemaphore_image_available_0 + inflight_frame_id_offset,
                              vk::PipelineStageFlagBits2::eColorAttachmentOutput}},
                            {{UID_vkSemaphore_render_finished_0 + swap_image_id_offset}},
                            UID_vkFence_inflight_0 + inflight_frame_id_offset));

            CHECK_ERR_RET(vk_->present(UID_vkQueue_0 + id_, UID_vkSwapchain_0 + id_, image_index,
                                       recreate_swapchain,
                                       {UID_vkSemaphore_render_finished_0 + swap_image_id_offset}));
            if (recreate_swapchain) {
                ILOG("present need recreate swapchain");
                CHECK_ERR_RET(createSwapImages(true));
            }

            CHECK_ERR_RET(handleEvent());
            curr_frame = (curr_frame + 1) % kMaxFrameInfight;
        }

        CHECK_ERR_RET(destroySwapImages());
        ILOG("quit app {}", id_);
        return MyErrCode::kOk;
    }

    MyErrCode handleEvent()
    {
        Event ev;
        while (evq_.try_dequeue(ev)) {
            switch (ev.type) {
                case EventType::kSurfaceClose: {
                    quit_ = true;
                    break;
                }
                case EventType::kSurfaceResize: {
                    if (ev.ix != curr_width_ || ev.iy != curr_height_) {
                        CHECK_ERR_RET(sc_.onSurfaceResize(ev.ix, ev.iy));
                        curr_width_ = ev.ix;
                        curr_height_ = ev.iy;
                        CHECK_ERR_RET(createSwapImages(true));
                    }
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
    myvk::Context* vk_;
    moodycamel::ConcurrentQueue<Event> evq_;
    scene::Scene sc_;

    bool quit_;
    uint32_t curr_width_;
    uint32_t curr_height_;
    uint32_t swapchain_image_count_;
    vk::Format swapchain_image_format_;
    vk::Format depth_image_format_;
    uint32_t texture_mip_levels_;
    vk::SampleCountFlagBits msaa_sample_count_;
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
        CHECK_ERR_RET(wl.createSurface(UID_wlSurface_0 + i, "", FSTR("vkwl-{}", i)));
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

    std::vector<std::pair<std::string, std::string>> app_files = {
        {"lyran.obj", "lyran-diffuse.jpg"},
        {"lloyd.obj", "lloyd-diffuse.jpg"},
    };

    std::vector<std::thread> app_threads;
    for (int i = 0; i < kMaxAppInstance; ++i) {
        VkSurfaceKHR vk_surface;
        VkWaylandSurfaceCreateInfoKHR surface_ci = {};
        surface_ci.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
        surface_ci.display = wl.getDisplay();
        surface_ci.surface = wl.getSurface(UID_wlSurface_0 + i);
        CHECK_VK_RET(
            vkCreateWaylandSurfaceKHR(vk.getInstance(), &surface_ci, nullptr, &vk_surface));
        CHECK_ERR_RET(vk.createSurface(UID_vkSurface_0 + i, vk_surface));
        app_threads.emplace_back([&, i] {
            apps[i].init(&vk, 350, 300, toolkit::getDataDir() / app_files.at(i).first,
                         toolkit::getDataDir() / app_files.at(i).second);
            apps[i].drawLoop();
            apps[i].destroy();
            wl.destroySurface(UID_wlSurface_0 + i);
            g_app_still_run -= 1;
        });
    }

    while (g_app_still_run > 0) {
        CHECK_ERR_RET(wl.dispatch(false));
    }
    CHECK_ERR_RET(wl.dispatch(true));

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
    CHECK_ERR_RET(toolkit::installCrashHook());
    EASY_MAIN_THREAD;
    profiler::startListen(28077);
    CHECK_ERR_RET(run());
    ILOG("end!");
    return MyErrCode::kOk;
    MY_CATCH_RET
}
