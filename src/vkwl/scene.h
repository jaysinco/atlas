#pragma once
#include "toolkit/error.h"
#include <vulkan/vulkan.h>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <vector>
#include <filesystem>

class aiScene;
class aiNode;
class aiMesh;

namespace glm
{
std::string toString(vec3 const& v);
}  // namespace glm

struct Axis
{
    glm::vec3 front;
    glm::vec3 right;
    glm::vec3 up;
};

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 tex_coord;
};

struct BoundingBox
{
    glm::vec3 lower;
    glm::vec3 high;
};

struct Trackball
{
    double last_mouse_x;
    double last_mouse_y;
    bool left_mouse_pressed;
    bool right_mouse_pressed;
    bool middle_mouse_pressed;
};

struct UniformBufferObject
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

class Camera
{
public:
    enum Face
    {
        kForward,
        kBackward,
        kLeft,
        kRight,
        kUp,
        kDown,
    };

    Camera(float aspect = 1.0f, glm::vec3 init_pos = glm::vec3(3.0f, 0.0f, 0.0f), float near = 0.1f,
           float far = 10.0f, float fov = 45.0f);
    void reset();
    void onScreenResize(int width, int height);
    void move(Face face, float distance);
    void move(float dx, float dy, float dz);
    void shake(float ddegree);
    void nod(float ddegree);
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    Axis getAxis() const;

private:
    glm::vec3 const init_pos_;
    float const near_, far_, fov_;
    float aspect_;
    glm::vec3 pos_;
    float yaw_, pitch_;
};

class Model
{
public:
    MyErrCode load(std::string const& model_path);
    MyErrCode unload();
    void reset();
    void move(float dx, float dy, float dz);
    void spin(float ddegree, float axis_x, float axis_y, float axis_z);
    void zoom(float dx, float dy, float dz);
    glm::mat4 getModelMatrix() const;

private:
    MyErrCode visitNode(aiScene const* scene, aiNode const* node);
    MyErrCode visitMesh(aiMesh const* mesh);

    friend class Scene;
    uint32_t index_size_;
    std::vector<Vertex> vertices_;
    std::vector<uint32_t> indices_;
    BoundingBox bbox_;
    glm::mat4 translate_, rotate_, scale_, init_;
};

class Scene
{
public:
    std::pair<int, int> getScreenInitSize() const;
    std::filesystem::path getModelPath() const;
    std::filesystem::path getTextureImagePath() const;
    std::filesystem::path getVertSpvPath() const;
    std::filesystem::path getFragSpvPath() const;

    MyErrCode load();
    MyErrCode unload();

    VkDeviceSize getUniformDataSize() const;
    void const* getUniformData() const;

    VkDeviceSize getVerticeDataSize() const;
    void const* getVerticeData() const;
    uint32_t getIndexSize() const;

    VkDeviceSize getIndexDataSize() const;
    void const* getIndexData() const;

    VkFrontFace getFrontFace() const;
    VkVertexInputBindingDescription getVertexBindingDesc() const;
    std::vector<VkVertexInputAttributeDescription> getVertexAttrDescs() const;

    MyErrCode onFrameDraw();
    MyErrCode onScreenResize(int width, int height);
    MyErrCode onMouseMove(double xpos, double ypos);
    MyErrCode onMousePress(int button, bool down);
    MyErrCode onMouseScroll(double xoffset, double yoffset);
    MyErrCode onKeyboardPress(int key, bool down);

private:
    Camera camera_;
    Model model_;
    Trackball trackball_;
    UniformBufferObject ubo_;
};
