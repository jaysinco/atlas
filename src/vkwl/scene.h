#pragma once
#include "toolkit/error.h"
#include <vulkan/vulkan.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <vector>
#include <filesystem>

class aiScene;
class aiNode;
class aiMesh;

class Scene
{
public:
    Scene();
    ~Scene();

    MyErrCode load();
    MyErrCode unload();

    std::filesystem::path getVertSpvPath() const;
    std::filesystem::path getFragSpvPath() const;
    std::filesystem::path getTextureImagePath() const;

    VkDeviceSize getVerticeDataSize() const;
    void const* getVerticeData() const;

    uint32_t getIndexSize() const;
    VkDeviceSize getIndexDataSize() const;
    void const* getIndexData() const;

    VkDeviceSize getUniformDataSize() const;
    void const* getUniformData() const;

    VkFrontFace getFrontFace() const;
    VkVertexInputBindingDescription getVertexBindingDesc() const;
    std::vector<VkVertexInputAttributeDescription> getVertexAttrDescs() const;

    std::pair<int, int> getInitSize() const;
    MyErrCode onResize(int width, int height);

private:
    MyErrCode visitNode(aiScene const* scene, aiNode const* node);
    MyErrCode visitMesh(aiMesh const* mesh);
    MyErrCode loadModel();
    MyErrCode unloadModel();

private:
    struct Vertex
    {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 tex_coord;
    };

    struct UniformBufferObject
    {
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
    };

    struct BoundingBox
    {
        glm::vec3 lower;
        glm::vec3 high;
    };

    std::vector<Vertex> vertices_;
    std::vector<uint32_t> indices_;
    uint32_t index_size_;
    BoundingBox bbox_;
    UniformBufferObject ubo_;
};
