#pragma once
#include "toolkit/error.h"
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>

class Scene
{
public:
    virtual ~Scene() = default;
    virtual MyErrCode load() = 0;
    virtual MyErrCode unload() = 0;
    virtual VkDeviceSize getVerticeDataSize() const = 0;
    virtual void const* getVerticeData() const = 0;
    virtual uint32_t getIndexSize() const = 0;
    virtual VkDeviceSize getIndexDataSize() const = 0;
    virtual void const* getIndexData() const = 0;
    virtual VkVertexInputBindingDescription getVertexBindingDesc() const = 0;
    virtual std::vector<VkVertexInputAttributeDescription> getVertexAttrDescs() const = 0;
};

class MockScene: public Scene
{
public:
    MockScene();
    ~MockScene() override;
    MyErrCode load() override;
    MyErrCode unload() override;
    VkDeviceSize getVerticeDataSize() const override;
    void const* getVerticeData() const override;
    uint32_t getIndexSize() const override;
    VkDeviceSize getIndexDataSize() const override;
    void const* getIndexData() const override;
    VkVertexInputBindingDescription getVertexBindingDesc() const override;
    std::vector<VkVertexInputAttributeDescription> getVertexAttrDescs() const override;

private:
    struct Vertex
    {
        glm::vec2 pos;
        glm::vec3 color;
    };

    std::vector<Vertex> vertices_;
    std::vector<uint32_t> indices_;
};
