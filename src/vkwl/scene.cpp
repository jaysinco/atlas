#include "scene.h"
#include "toolkit/toolkit.h"
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>

Scene::Scene() = default;

Scene::~Scene() = default;

MyErrCode Scene::load()
{
    vertices_ = {
        {{-1.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
        {{1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
        {{1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
        {{-1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},
    };

    indices_ = {0, 1, 2, 2, 3, 0};
    index_size_ = indices_.size();

    ubo_.model = glm::mat4(1.0f);
    ubo_.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                            glm::vec3(0.0f, 1.0f, 0.0f));
    auto [width, height] = getInitSize();
    CHECK_ERR_RET(onResize(width, height));

    return MyErrCode::kOk;
}

MyErrCode Scene::unload()
{
    vertices_.clear();
    vertices_.shrink_to_fit();
    indices_.clear();
    indices_.shrink_to_fit();
    return MyErrCode::kOk;
}

std::filesystem::path Scene::getVertSpvPath() const
{
    return toolkit::getDataDir() / "test.vert.spv";
}

std::filesystem::path Scene::getFragSpvPath() const
{
    return toolkit::getDataDir() / "test.frag.spv";
}

std::filesystem::path Scene::getTextureImagePath() const
{
    return toolkit::getDataDir() / "statue-1275469.jpg";
}

VkDeviceSize Scene::getVerticeDataSize() const { return sizeof(vertices_[0]) * vertices_.size(); }

void const* Scene::getVerticeData() const { return vertices_.data(); }

uint32_t Scene::getIndexSize() const { return index_size_; }

VkDeviceSize Scene::getIndexDataSize() const { return sizeof(indices_[0]) * indices_.size(); }

void const* Scene::getIndexData() const { return indices_.data(); }

VkVertexInputBindingDescription Scene::getVertexBindingDesc() const
{
    VkVertexInputBindingDescription desc{};
    desc.binding = 0;
    desc.stride = sizeof(Vertex);
    desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return desc;
}

VkFrontFace Scene::getFrontFace() const { return VK_FRONT_FACE_COUNTER_CLOCKWISE; }

std::vector<VkVertexInputAttributeDescription> Scene::getVertexAttrDescs() const
{
    std::vector<VkVertexInputAttributeDescription> descs{3};
    descs[0].binding = 0;
    descs[0].location = 0;
    descs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    descs[0].offset = offsetof(Vertex, pos);

    descs[1].binding = 0;
    descs[1].location = 1;
    descs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    descs[1].offset = offsetof(Vertex, color);

    descs[2].binding = 0;
    descs[2].location = 2;
    descs[2].format = VK_FORMAT_R32G32_SFLOAT;
    descs[2].offset = offsetof(Vertex, tex_coord);
    return descs;
}

VkDeviceSize Scene::getUniformDataSize() const { return sizeof(ubo_); }

void const* Scene::getUniformData() const { return &ubo_; }

std::pair<int, int> Scene::getInitSize() const { return {300, 200}; }

MyErrCode Scene::onResize(int width, int height)
{
    ubo_.proj =
        glm::perspective(glm::radians(45.0f), width / static_cast<float>(height), 0.1f, 10.0f);
    ubo_.proj[1][1] *= -1;
    return MyErrCode::kOk;
}
