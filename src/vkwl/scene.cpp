#include "scene.h"

MockScene::MockScene()
{
    vertices_ = {
        {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
        {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
    };

    indices_ = {0, 1, 2, 2, 3, 0};
}

MockScene::~MockScene() {}

MyErrCode MockScene::load() { return MyErrCode::kOk; }

MyErrCode MockScene::unload() { return MyErrCode::kOk; }

VkDeviceSize MockScene::getVerticeDataSize() const
{
    return sizeof(vertices_[0]) * vertices_.size();
}

void const* MockScene::getVerticeData() const { return vertices_.data(); }

uint32_t MockScene::getIndexSize() const { return indices_.size(); }

VkDeviceSize MockScene::getIndexDataSize() const { return sizeof(indices_[0]) * indices_.size(); }

void const* MockScene::getIndexData() const { return indices_.data(); }

VkVertexInputBindingDescription MockScene::getVertexBindingDesc() const
{
    VkVertexInputBindingDescription desc{};
    desc.binding = 0;
    desc.stride = sizeof(Vertex);
    desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return desc;
}

std::vector<VkVertexInputAttributeDescription> MockScene::getVertexAttrDescs() const
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
