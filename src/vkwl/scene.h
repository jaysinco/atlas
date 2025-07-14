#pragma once
#include "toolkit/error.h"
#include <vulkan/vulkan.h>
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

private:
    struct Vertex
    {
        glm::vec2 pos;
        glm::vec3 color;
    };

    std::vector<Vertex> vertices_;
    std::vector<uint32_t> indices_;
    uint32_t index_size_;
};
