#include "scene.h"
#include "toolkit/toolkit.h"
#include "toolkit/logging.h"
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace glm
{
std::string toString(glm::vec3 const& v) { return FSTR("({}, {}, {})", v.x, v.y, v.z); }
}  // namespace glm

Camera::Camera(float aspect, glm::vec3 init_pos, float near, float far, float fov)
    : aspect_(aspect), init_pos_(init_pos), near_(near), far_(far), fov_(fov)
{
    this->reset();
}

void Camera::reset()
{
    this->pos_ = this->init_pos_;
    this->yaw_ = 90.0f;
    this->pitch_ = 0.0f;
}

void Camera::onScreenResize(int width, int height)
{
    this->aspect_ = static_cast<float>(width) / height;
}

void Camera::move(Face face, float distance)
{
    Axis axis = this->getAxis();
    switch (face) {
        case kForward:
            this->pos_ += axis.front * distance;
            break;
        case kBackward:
            this->pos_ -= axis.front * distance;
            break;
        case kLeft:
            this->pos_ -= axis.right * distance;
            break;
        case kRight:
            this->pos_ += axis.right * distance;
            break;
        case kUp:
            this->pos_ += axis.up * distance;
            break;
        case kDown:
            this->pos_ -= axis.up * distance;
            break;
    }
}

void Camera::move(float dx, float dy, float dz) { this->pos_ += glm::vec3(dx, dy, dz); }

void Camera::shake(float ddegree) { this->yaw_ += ddegree; }

void Camera::nod(float ddegree)
{
    this->pitch_ = std::max(std::min(this->pitch_ + ddegree, 89.0f), -89.0f);
}

Camera::Axis Camera::getAxis() const
{
    Axis axis;
    float x = cos(glm::radians(this->yaw_)) * cos(glm::radians(this->pitch_));
    float y = sin(glm::radians(this->yaw_)) * cos(glm::radians(this->pitch_));
    float z = sin(glm::radians(this->pitch_));
    axis.front = glm::normalize(glm::vec3(x, y, z));
    axis.right = glm::normalize(glm::cross(axis.front, glm::vec3(0.0f, 0.0f, 1.0f)));
    axis.up = glm::normalize(glm::cross(axis.right, axis.front));
    return axis;
}

glm::mat4 Camera::getViewMatrix() const
{
    Axis axis = this->getAxis();
    auto center = this->pos_ + axis.front;
    return glm::lookAt(this->pos_, center, axis.up);
}

glm::mat4 Camera::getProjectionMatrix() const
{
    auto proj = glm::perspective(glm::radians(this->fov_), this->aspect_, this->near_, this->far_);
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

MyErrCode Model::load(std::string const& model_path)
{
    Assimp::Importer importer;
    aiScene const* scene = importer.ReadFile(
        model_path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        ELOG("failed to import model {}: {}", model_path, importer.GetErrorString());
        return MyErrCode::kFailed;
    }

    ILOG("loading {}", model_path);
    bbox_.lower = glm::vec3(std::numeric_limits<float>::max());
    bbox_.high = glm::vec3(std::numeric_limits<float>::min());
    CHECK_ERR_RET(visitNode(scene, scene->mRootNode));
    index_size_ = indices_.size();

    glm::vec3 center = (bbox_.high + bbox_.lower) / 2.0f;
    glm::vec3 size = bbox_.high - bbox_.lower;
    float factor = 2.0f / std::max({size.x, size.y, size.z});
    this->init_ =
        glm::scale(glm::mat4(1.0f), glm::vec3(factor)) * glm::translate(glm::mat4(1.0f), -center);
    this->reset();

    return MyErrCode::kOk;
}

MyErrCode Model::unload()
{
    vertices_.clear();
    vertices_.shrink_to_fit();
    indices_.clear();
    indices_.shrink_to_fit();
    return MyErrCode::kOk;
}

void Model::reset()
{
    this->translate_ = glm::mat4(1.0f);
    this->rotate_ = glm::mat4(1.0f);
    this->scale_ = glm::mat4(1.0f);
}

void Model::move(float dx, float dy, float dz)
{
    this->translate_ = glm::translate(glm::mat4(1.0f), glm::vec3(dx, dy, dz)) * this->translate_;
}

void Model::spin(float ddegree, float axis_x, float axis_y, float axis_z)
{
    this->rotate_ =
        glm::rotate(glm::mat4(1.0f), glm::radians(ddegree), glm::vec3(axis_x, axis_y, axis_z)) *
        this->rotate_;
}

void Model::zoom(float dx, float dy, float dz)
{
    this->scale_ = glm::scale(glm::mat4(1.0f), glm::vec3(dx, dy, dz)) * this->scale_;
}

glm::mat4 Model::getModelMatrix() const
{
    return this->translate_ * this->rotate_ * this->scale_ * this->init_;
}

std::pair<int, int> Scene::getScreenInitSize() const { return {300, 200}; }

std::filesystem::path Scene::getModelPath() const
{
    return toolkit::getDataDir() / "watermill.obj";
}

std::filesystem::path Scene::getTextureImagePath() const
{
    return toolkit::getDataDir() / "watermill-diffuse.png";
}

std::filesystem::path Scene::getVertSpvPath() const
{
    return toolkit::getDataDir() / "test.vert.spv";
}

std::filesystem::path Scene::getFragSpvPath() const
{
    return toolkit::getDataDir() / "test.frag.spv";
}

MyErrCode Scene::load()
{
    CHECK_ERR_RET(model_.load(getModelPath().string()));
    auto [width, height] = getScreenInitSize();
    CHECK_ERR_RET(onScreenResize(width, height));
    return MyErrCode::kOk;
}

MyErrCode Scene::unload()
{
    CHECK_ERR_RET(model_.unload());
    return MyErrCode::kOk;
}

VkDeviceSize Scene::getVerticeDataSize() const
{
    return sizeof(model_.vertices_[0]) * model_.vertices_.size();
}

void const* Scene::getVerticeData() const { return model_.vertices_.data(); }

uint32_t Scene::getIndexSize() const { return model_.index_size_; }

VkDeviceSize Scene::getIndexDataSize() const
{
    return sizeof(model_.indices_[0]) * model_.indices_.size();
}

void const* Scene::getIndexData() const { return model_.indices_.data(); }

VkVertexInputBindingDescription Scene::getVertexBindingDesc() const
{
    VkVertexInputBindingDescription desc{};
    desc.binding = 0;
    desc.stride = sizeof(Vertex);
    desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return desc;
}

VkFrontFace Scene::getFrontFace() const { return VK_FRONT_FACE_COUNTER_CLOCKWISE; }

std::vector<VkVertexInputAttributeDescription> Scene::getVertexAttrDescs() const
{
    std::vector<VkVertexInputAttributeDescription> descs{3};
    descs[0].binding = 0;
    descs[0].location = 0;
    descs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    descs[0].offset = offsetof(Vertex, pos);

    descs[1].binding = 0;
    descs[1].location = 1;
    descs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    descs[1].offset = offsetof(Vertex, normal);

    descs[2].binding = 0;
    descs[2].location = 2;
    descs[2].format = VK_FORMAT_R32G32_SFLOAT;
    descs[2].offset = offsetof(Vertex, tex_coord);
    return descs;
}

VkDeviceSize Scene::getUniformDataSize() const { return sizeof(ubo_); }

void const* Scene::getUniformData() const { return &ubo_; }

MyErrCode Scene::onFrameDraw()
{
    ubo_.model = model_.getModelMatrix();
    ubo_.view = camera_.getViewMatrix();
    ubo_.proj = camera_.getProjectionMatrix();
    return MyErrCode::kOk;
}

MyErrCode Scene::onScreenResize(int width, int height)
{
    camera_.onScreenResize(width, height);
    return MyErrCode::kOk;
}
