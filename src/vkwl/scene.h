#pragma once
#include "toolkit/error.h"
#include <vulkan/vulkan.h>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <vector>

class Scene
{
public:
    Scene();
    ~Scene();

    MyErrCode load();
    MyErrCode unload();

    VkDeviceSize getVerticeDataSize() const;
    void const* getVerticeData() const;

    uint32_t getIndexSize() const;
    VkDeviceSize getIndexDataSize() const;
    void const* getIndexData() const;

    static VkVertexInputBindingDescription getVertexBindingDesc();
    static std::vector<VkVertexInputAttributeDescription> getVertexAttrDescs();

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
