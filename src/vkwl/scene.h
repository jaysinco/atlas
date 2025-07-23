#pragma once
#include "toolkit/error.h"
#include <vulkan/vulkan.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <vector>
#include <filesystem>

class Scene
{
public:
    Scene();
    ~Scene();

    MyErrCode load();
    MyErrCode unload();

    std::filesystem::path getVertSpvPath() const;
    std::filesystem::path getFragSpvPath() const;

    VkDeviceSize getVerticeDataSize() const;
    void const* getVerticeData() const;

    VkVertexInputBindingDescription getVertexBindingDesc();
    std::vector<VkVertexInputAttributeDescription> getVertexAttrDescs();

    uint32_t getIndexSize() const;
    VkDeviceSize getIndexDataSize() const;
    void const* getIndexData() const;

    VkDeviceSize getUniformDataSize() const;
    void const* getUniformData() const;
    MyErrCode updateUniformData(int win_width, int win_height);

private:
    struct Vertex
    {
        glm::vec2 pos;
        glm::vec3 color;
    };

    struct UniformBufferObject
    {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
    };

    std::vector<Vertex> vertices_;
    uint32_t index_size_;
    std::vector<uint32_t> indices_;
    UniformBufferObject ubo_;
};
