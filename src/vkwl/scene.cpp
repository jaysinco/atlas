#include "scene.h"
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>

Scene::Scene() = default;

Scene::~Scene() = default;

MyErrCode Scene::load()
{
    vertices_ = {
        {{-1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}},
        {{1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
        {{1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
        {{-1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
    };

    indices_ = {0, 1, 2, 2, 3, 0};
    index_size_ = indices_.size();

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

VkDeviceSize Scene::getVerticeDataSize() const { return sizeof(vertices_[0]) * vertices_.size(); }

void const* Scene::getVerticeData() const { return vertices_.data(); }

uint32_t Scene::getIndexSize() const { return index_size_; }

VkDeviceSize Scene::getIndexDataSize() const { return sizeof(indices_[0]) * indices_.size(); }

void const* Scene::getIndexData() const { return indices_.data(); }

VkVertexInputBindingDescription Scene::getVertexBindingDesc()
{
    VkVertexInputBindingDescription desc{};
    desc.binding = 0;
    desc.stride = sizeof(Vertex);
    desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return desc;
}

std::vector<VkVertexInputAttributeDescription> Scene::getVertexAttrDescs()
{
    std::vector<VkVertexInputAttributeDescription> descs{2};
    descs[0].binding = 0;
    descs[0].location = 0;
    descs[0].format = VK_FORMAT_R32G32_SFLOAT;
    descs[0].offset = offsetof(Vertex, pos);

    descs[1].binding = 0;
    descs[1].location = 1;
    descs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    descs[1].offset = offsetof(Vertex, color);
    return descs;
}

VkDeviceSize Scene::getUniformDataSize() const { return sizeof(ubo_); }

void const* Scene::getUniformData() const { return &ubo_; }

MyErrCode Scene::updateUniformData(int win_width, int win_height)
{
    static auto start_time = std::chrono::high_resolution_clock::now();
    auto curr_time = std::chrono::high_resolution_clock::now();
    float time =
        std::chrono::duration<float, std::chrono::seconds::period>(curr_time - start_time).count();
    ubo_.model =
        glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo_.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f));
    ubo_.proj = glm::perspective(glm::radians(45.0f), win_width / static_cast<float>(win_height),
                                 0.1f, 10.0f);
    ubo_.proj[1][1] *= -1;

    return MyErrCode::kOk;
}
