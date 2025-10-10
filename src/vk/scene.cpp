#include "scene.h"
#include "toolkit/toolkit.h"
#include "toolkit/logging.h"
#include "keycode.h"
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <linux/input.h>
#include <imgui/imgui.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace glm
{
std::string toString(glm::vec3 const& v) { return FSTR("({}, {}, {})", v.x, v.y, v.z); }
}  // namespace glm

namespace scene
{

Camera::Camera(float aspect, glm::vec3 init_pos, glm::vec3 init_center, float near, float far,
               float fov)
    : aspect_(aspect),
      init_pos_(init_pos),
      init_center_(init_center),
      near_(near),
      far_(far),
      fov_(fov)
{
    reset();
}

void Camera::reset()
{
    pos_ = init_pos_;
    glm::vec3 lookat = glm::normalize(init_center_ - init_pos_);
    pitch_ = glm::degrees(asin(lookat.z));
    yaw_ = glm::degrees(atan2(lookat.y, lookat.x));
}

void Camera::onSurfaceResize(int width, int height)
{
    aspect_ = static_cast<float>(width) / height;
}

void Camera::move(Face face, float distance)
{
    Axis axis = getAxis();
    switch (face) {
        case kForward:
            pos_ += axis.front * distance;
            break;
        case kBackward:
            pos_ -= axis.front * distance;
            break;
        case kLeft:
            pos_ -= axis.right * distance;
            break;
        case kRight:
            pos_ += axis.right * distance;
            break;
        case kUp:
            pos_ += axis.up * distance;
            break;
        case kDown:
            pos_ -= axis.up * distance;
            break;
    }
}

void Camera::move(float dx, float dy, float dz) { pos_ += glm::vec3(dx, dy, dz); }

void Camera::shake(float ddegree) { yaw_ += ddegree; }

void Camera::nod(float ddegree) { pitch_ = std::max(std::min(pitch_ + ddegree, 89.0f), -89.0f); }

Camera::Axis Camera::getAxis() const
{
    Axis axis;
    float x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    float y = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    float z = sin(glm::radians(pitch_));
    axis.front = glm::normalize(glm::vec3(x, y, z));
    axis.right = glm::normalize(glm::cross(axis.front, glm::vec3(0.0f, 0.0f, 1.0f)));
    axis.up = glm::normalize(glm::cross(axis.right, axis.front));
    return axis;
}

glm::mat4 Camera::getViewMatrix() const
{
    Axis axis = getAxis();
    auto center = pos_ + axis.front;
    return glm::lookAt(pos_, center, axis.up);
}

glm::mat4 Camera::getProjectionMatrix() const
{
    auto proj = glm::perspective(glm::radians(fov_), aspect_, near_, far_);
    proj[1][1] *= -1;
    return proj;
}

MyErrCode Model::visitMesh(aiMesh const* mesh)
{
    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
        Vertex v = {};
        v.pos = glm::vec3(mesh->mVertices[i].x, -mesh->mVertices[i].z, mesh->mVertices[i].y);
        if (mesh->HasNormals()) {
            v.normal = glm::vec3(mesh->mNormals[i].x, -mesh->mNormals[i].z, mesh->mNormals[i].y);
        }
        if (mesh->mTextureCoords[0]) {
            v.tex_coord = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
        }
        vertices_.push_back(v);
    }
    for (unsigned i = 0; i < mesh->mNumFaces; ++i) {
        aiFace const& face = mesh->mFaces[i];
        for (unsigned j = 0; j < face.mNumIndices; ++j) {
            indices_.push_back(face.mIndices[j]);
            bbox_.lower.x = std::min(bbox_.lower.x, vertices_[face.mIndices[j]].pos.x);
            bbox_.lower.y = std::min(bbox_.lower.y, vertices_[face.mIndices[j]].pos.y);
            bbox_.lower.z = std::min(bbox_.lower.z, vertices_[face.mIndices[j]].pos.z);
            bbox_.high.x = std::max(bbox_.high.x, vertices_[face.mIndices[j]].pos.x);
            bbox_.high.y = std::max(bbox_.high.y, vertices_[face.mIndices[j]].pos.y);
            bbox_.high.z = std::max(bbox_.high.z, vertices_[face.mIndices[j]].pos.z);
        }
    }
    ILOG("-- {}, faces={}, box=[{}, {}]", mesh->mName.C_Str(), mesh->mNumFaces, bbox_.lower,
         bbox_.high);
    return MyErrCode::kOk;
}

MyErrCode Model::visitNode(aiScene const* scene, aiNode const* node)
{
    for (unsigned i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        CHECK_ERR_RET(visitMesh(mesh));
    }
    for (unsigned i = 0; i < node->mNumChildren; i++) {
        CHECK_ERR_RET(visitNode(scene, node->mChildren[i]));
    }
    return MyErrCode::kOk;
}

MyErrCode Model::load(std::filesystem::path const& file_path)
{
    Assimp::Importer importer;
    aiScene const* scene = importer.ReadFile(
        file_path.string(), aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        ELOG("failed to import model {}: {}", file_path, importer.GetErrorString());
        return MyErrCode::kFailed;
    }

    ILOG("loading {}", file_path);
    bbox_.lower = glm::vec3(std::numeric_limits<float>::max());
    bbox_.high = glm::vec3(std::numeric_limits<float>::min());
    CHECK_ERR_RET(visitNode(scene, scene->mRootNode));
    index_size_ = indices_.size();

    glm::vec3 center = (bbox_.high + bbox_.lower) / 2.0f;
    glm::vec3 size = bbox_.high - bbox_.lower;
    float factor = 2.0f / std::max({size.x, size.y, size.z});
    init_ =
        glm::scale(glm::mat4(1.0f), glm::vec3(factor)) * glm::translate(glm::mat4(1.0f), -center);
    reset();

    return MyErrCode::kOk;
}

void Model::unload()
{
    vertices_.clear();
    vertices_.shrink_to_fit();
    indices_.clear();
    indices_.shrink_to_fit();
}

void Model::reset()
{
    translate_ = glm::mat4(1.0f);
    rotate_ = glm::mat4(1.0f);
    scale_ = glm::mat4(1.0f);
}

void Model::move(float dx, float dy, float dz)
{
    translate_ = glm::translate(glm::mat4(1.0f), glm::vec3(dx, dy, dz)) * translate_;
}

void Model::spin(float ddegree, float axis_x, float axis_y, float axis_z)
{
    rotate_ =
        glm::rotate(glm::mat4(1.0f), glm::radians(ddegree), glm::vec3(axis_x, axis_y, axis_z)) *
        rotate_;
}

void Model::zoom(float dx, float dy, float dz)
{
    scale_ = glm::scale(glm::mat4(1.0f), glm::vec3(dx, dy, dz)) * scale_;
}

glm::mat4 Model::getModelMatrix() const { return translate_ * rotate_ * scale_ * init_; }

std::vector<Vertex> const& Model::getVertices() const { return vertices_; }

std::vector<uint32_t> const& Model::getIndices() const { return indices_; }

uint32_t Model::getIndexSize() const { return index_size_; }

MyErrCode Texture::load(std::filesystem::path const& file_path)
{
    cv::Mat src = cv::imread(file_path, cv::IMREAD_COLOR);
    if (src.data == nullptr) {
        ELOG("failed to load image file: {}", file_path);
        return MyErrCode::kFailed;
    }

    width_ = src.cols;
    height_ = src.rows;
    data_.resize(height_ * width_ * 4);

    cv::Mat dst(height_, width_, CV_8UC4, data_.data());
    cv::cvtColor(src, dst, cv::COLOR_BGR2BGRA);
    return MyErrCode::kOk;
}

std::vector<uint8_t> const& Texture::getData() const { return data_; }

std::pair<uint32_t, uint32_t> Texture::getSize() const { return {width_, height_}; }

uint32_t Texture::getMaxMipLevels() const
{
    return std::floor(std::log2(std::max(width_, height_))) + 1;
}

void Texture::unload()
{
    data_.clear();
    data_.shrink_to_fit();
}

MyErrCode Scene::createModel(Uid id, std::filesystem::path const& file_path)
{
    if (models_.find(id) != models_.end()) {
        CHECK_ERR_RET(destroyModel(id));
    }
    models_[id] = {};
    CHECK_ERR_RET(models_[id].load(file_path));
    return MyErrCode::kOk;
}

Model& Scene::getModel(Uid id)
{
    if (models_.find(id) == models_.end()) {
        MY_THROW("model not exist: {}", id);
    }
    return models_.at(id);
}

MyErrCode Scene::destroyModel(Uid id)
{
    if (auto it = models_.find(id); it != models_.end()) {
        models_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("model not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

MyErrCode Scene::createTexture(Uid id, std::filesystem::path const& file_path)
{
    if (textures_.find(id) != textures_.end()) {
        CHECK_ERR_RET(destroyTexture(id));
    }
    textures_[id] = {};
    CHECK_ERR_RET(textures_[id].load(file_path));
    return MyErrCode::kOk;
}

Texture& Scene::getTexture(Uid id)
{
    if (textures_.find(id) == textures_.end()) {
        MY_THROW("texture not exist: {}", id);
    }
    return textures_.at(id);
}

MyErrCode Scene::destroyTexture(Uid id)
{
    if (auto it = textures_.find(id); it != textures_.end()) {
        textures_.erase(it);
        return MyErrCode::kOk;
    } else {
        ELOG("texture not exist: {}", id);
        return MyErrCode::kFailed;
    }
}

UniformBuffer Scene::getUniformBuffer(Uid model_id)
{
    Model& model = getModel(model_id);
    UniformBuffer ubo;
    ubo.model = model.getModelMatrix();
    ubo.view = camera_.getViewMatrix();
    ubo.proj = camera_.getProjectionMatrix();
    ubo.light_pos = ubo.view * ubo.model * glm::vec4(3.0f, 3.0f, 5.0f, 1.0f);
    ubo.light_color = glm::vec3(1.0f, 1.0f, 1.0f);
    return ubo;
}

GuiState const& Scene::getGuiState() const { return gs_; }

MyErrCode Scene::destroy()
{
    while (!models_.empty()) {
        CHECK_ERR_RET(destroyModel(models_.begin()->first));
    }
    while (!textures_.empty()) {
        CHECK_ERR_RET(destroyTexture(textures_.begin()->first));
    }
    return MyErrCode::kOk;
}

MyErrCode Scene::onFrameDraw(bool& recreate_pipeline)
{
    CHECK_ERR_RET(drawGui(recreate_pipeline));
    return MyErrCode::kOk;
}

MyErrCode Scene::onSurfaceResize(int width, int height)
{
    camera_.onSurfaceResize(width, height);
    return MyErrCode::kOk;
}

MyErrCode Scene::onPointerMove(double xpos, double ypos)
{
    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(xpos, ypos);
    if (io.WantCaptureMouse) {
        return MyErrCode::kOk;
    }

    float dx = xpos - gs_.last_mouse_x;
    float dy = ypos - gs_.last_mouse_y;
    if (gs_.middle_mouse_pressed) {
        camera_.shake(-0.15 * dx);
        camera_.nod(-0.15 * dy);
    } else if (gs_.left_mouse_pressed) {
        camera_.move(Camera::kRight, -0.01 * dx);
        camera_.move(Camera::kUp, 0.01 * dy);
    } else if (gs_.right_mouse_pressed) {
        glm::vec3 spin_dir = camera_.getAxis().up;
        models_.begin()->second.spin(dx, spin_dir.x, spin_dir.y, spin_dir.z);
    }
    gs_.last_mouse_x = xpos;
    gs_.last_mouse_y = ypos;
    return MyErrCode::kOk;
}

MyErrCode Scene::onPointerPress(int button, bool down)
{
    ImGuiIO& io = ImGui::GetIO();
    io.AddMouseButtonEvent(button, down);
    if (io.WantCaptureMouse) {
        return MyErrCode::kOk;
    }

    if (button == 0) {
        gs_.left_mouse_pressed = down;
    } else if (button == 1) {
        gs_.right_mouse_pressed = down;
    } else if (button == 2) {
        gs_.middle_mouse_pressed = down;
    }
    return MyErrCode::kOk;
}

MyErrCode Scene::onPointerScroll(double xoffset, double yoffset)
{
    ImGuiIO& io = ImGui::GetIO();
    io.AddMouseWheelEvent(0.0, yoffset / -10.0);
    if (io.WantCaptureMouse) {
        return MyErrCode::kOk;
    }

    double sensitivity = -0.03;
    camera_.move(Camera::kForward, yoffset * sensitivity);
    return MyErrCode::kOk;
}

MyErrCode Scene::onKeyboardPress(int key, bool down, bool& need_quit)
{
    if (key == KEY_LEFTSHIFT) {
        gs_.shift_down = down;
    } else if (key == KEY_LEFTCTRL) {
        gs_.ctrl_down = down;
    }

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard) {
        if (key == KEY_BACKSPACE || key == KEY_LEFT || key == KEY_RIGHT) {
            io.AddKeyEvent(static_cast<ImGuiKey>(KeyCode::convertTo(KeyCode::kImGui, key, false)),
                           down);
        } else if (down) {
            if (int c = KeyCode::convertTo(KeyCode::kAscii, key, gs_.shift_down); c >= 0) {
                io.AddInputCharacter(c);
            }
        }
        return MyErrCode::kOk;
    }

    if (!down) {
        return MyErrCode::kOk;
    }
    if (key == KEY_Q) {
        need_quit = true;
    } else if (key == KEY_R) {
        camera_.reset();
        models_.begin()->second.reset();
    }
    return MyErrCode::kOk;
}

MyErrCode Scene::drawGui(bool& recreate_pipeline)
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
    ImGui::Begin("vkwl");
    ImGui::Checkbox("show demo", &gs_.show_demo);
    if (ImGui::Checkbox("wire frame", &gs_.wire_frame)) {
        recreate_pipeline = true;
    }
    if (ImGui::Checkbox("face clockwise", &gs_.face_clockwise)) {
        recreate_pipeline = true;
    }
    ImGui::End();

    if (gs_.show_demo) {
        ImGui::ShowDemoWindow();
    }

    return MyErrCode::kOk;
}

}  // namespace scene
