#pragma once
#include "toolkit/error.h"
#include "toolkit/toolkit.h"
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <vector>
#include <map>
#include <filesystem>

class aiScene;
class aiNode;
class aiMesh;

namespace glm
{
std::string toString(vec3 const& v);
}  // namespace glm

namespace scene
{

using Uid = toolkit::Uid;

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

    struct Axis
    {
        glm::vec3 front;
        glm::vec3 right;
        glm::vec3 up;
    };

    // z is up, right hand
    Camera(float aspect = 1.0f, glm::vec3 init_pos = glm::vec3(0.6f, 3.0f, 0.0f),
           glm::vec3 init_center = glm::vec3(0.6f, 0.0f, 0.0f), float near = 0.1f,
           float far = 100.0f, float fov = 45.0f);
    void reset();
    void onSurfaceResize(int width, int height);
    void move(Face face, float distance);
    void move(float dx, float dy, float dz);
    void shake(float ddegree);
    void nod(float ddegree);
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    Axis getAxis() const;

private:
    glm::vec3 const init_pos_;
    glm::vec3 const init_center_;
    float const near_, far_, fov_;
    float aspect_;
    glm::vec3 pos_;
    float yaw_, pitch_;
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

class Model
{
public:
    MyErrCode load(std::filesystem::path const& file_path);
    void move(float dx, float dy, float dz);
    void spin(float ddegree, float axis_x, float axis_y, float axis_z);
    void zoom(float dx, float dy, float dz);
    void reset();
    void unload();

    glm::mat4 getModelMatrix() const;
    std::vector<Vertex> const& getVertices() const;
    std::vector<uint32_t> const& getIndices() const;
    uint32_t getIndexSize() const;

private:
    MyErrCode visitNode(aiScene const* scene, aiNode const* node);
    MyErrCode visitMesh(aiMesh const* mesh);

private:
    friend class Scene;
    uint32_t index_size_;
    std::vector<Vertex> vertices_;
    std::vector<uint32_t> indices_;
    BoundingBox bbox_;
    glm::mat4 translate_, rotate_, scale_, init_;
};

class Texture
{
public:
    MyErrCode load(std::filesystem::path const& file_path);
    std::vector<uint8_t> const& getData() const;
    std::pair<int, int> getSize() const;
    void unload();

private:
    int width_;
    int height_;
    std::vector<uint8_t> data_;
};

struct UniformBuffer
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec3 light_pos;
    alignas(16) glm::vec3 light_color;
};

struct GuiState
{
    double last_mouse_x;
    double last_mouse_y;
    bool left_mouse_pressed;
    bool right_mouse_pressed;
    bool middle_mouse_pressed;
    bool shift_down;
    bool ctrl_down;
    bool show_demo;
    bool wire_frame;
    bool face_clockwise;
};

class Scene
{
public:
    MyErrCode createModel(Uid id, std::filesystem::path const& file_path);
    MyErrCode createTexture(Uid id, std::filesystem::path const& file_path);

    GuiState const& getGuiState() const;
    Model& getModel(Uid id);
    Texture& getTexture(Uid id);
    UniformBuffer getUniformBuffer(Uid model_id);

    MyErrCode onFrameDraw(bool& recreate_pipeline);
    MyErrCode onSurfaceResize(int width, int height);
    MyErrCode onPointerMove(double xpos, double ypos);
    MyErrCode onPointerPress(int button, bool down);
    MyErrCode onPointerScroll(double xoffset, double yoffset);
    MyErrCode onKeyboardPress(int key, bool down, bool& need_quit);

    MyErrCode destroyModel(Uid id);
    MyErrCode destroyTexture(Uid id);
    MyErrCode destroy();

private:
    MyErrCode drawGui(bool& recreate_pipeline);

private:
    Camera camera_;
    GuiState gs_;
    std::map<Uid, Model> models_;
    std::map<Uid, Texture> textures_;
};

}  // namespace scene
