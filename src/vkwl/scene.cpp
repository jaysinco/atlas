#include "scene.h"
#include "toolkit/toolkit.h"
#include "toolkit/logging.h"
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

Scene::Scene() = default;

Scene::~Scene() = default;

MyErrCode Scene::load()
{
    CHECK_ERR_RET(loadModel());
    ubo_.model = glm::mat4(1.0f);
    ubo_.view = glm::lookAt(glm::vec3(-3.0f, -3.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f));
    auto [width, height] = getInitSize();
    CHECK_ERR_RET(onResize(width, height));
    return MyErrCode::kOk;
}

MyErrCode Scene::unload()
{
    CHECK_ERR_RET(unloadModel());
    return MyErrCode::kOk;
}

std::filesystem::path Scene::getVertSpvPath() const
{
    return toolkit::getDataDir() / "test.vert.spv";
}

std::filesystem::path Scene::getFragSpvPath() const
{
    return toolkit::getDataDir() / "test.frag.spv";
}

std::filesystem::path Scene::getTextureImagePath() const
{
    return toolkit::getDataDir() / "statue-1275469.jpg";
}

VkDeviceSize Scene::getVerticeDataSize() const { return sizeof(vertices_[0]) * vertices_.size(); }

void const* Scene::getVerticeData() const { return vertices_.data(); }

uint32_t Scene::getIndexSize() const { return index_size_; }

VkDeviceSize Scene::getIndexDataSize() const { return sizeof(indices_[0]) * indices_.size(); }

void const* Scene::getIndexData() const { return indices_.data(); }

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

std::pair<int, int> Scene::getInitSize() const { return {300, 200}; }

MyErrCode Scene::onResize(int width, int height)
{
    ubo_.proj =
        glm::perspective(glm::radians(45.0f), width / static_cast<float>(height), 0.1f, 10.0f);
    ubo_.proj[1][1] *= -1;
    return MyErrCode::kOk;
}

MyErrCode Scene::visitMesh(aiMesh const* mesh)
{
    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
        Vertex v = {};
        v.pos = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
        if (mesh->HasNormals()) {
            v.normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
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
            bbox_.lower.x = std::min(bbox_.lower.x, mesh->mVertices[face.mIndices[j]].x);
            bbox_.lower.y = std::min(bbox_.lower.y, mesh->mVertices[face.mIndices[j]].y);
            bbox_.lower.z = std::min(bbox_.lower.z, mesh->mVertices[face.mIndices[j]].z);
            bbox_.high.x = std::max(bbox_.high.x, mesh->mVertices[face.mIndices[j]].x);
            bbox_.high.y = std::max(bbox_.high.y, mesh->mVertices[face.mIndices[j]].y);
            bbox_.high.z = std::max(bbox_.high.z, mesh->mVertices[face.mIndices[j]].z);
        }
    }
    ILOG("-- {}, faces={}", mesh->mName.C_Str(), mesh->mNumFaces);
    return MyErrCode::kOk;
}

MyErrCode Scene::visitNode(aiScene const* scene, aiNode const* node)
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

MyErrCode Scene::loadModel()
{
    Assimp::Importer importer;
    auto mode_path = toolkit::getDataDir() / "test.obj";
    aiScene const* scene = importer.ReadFile(
        mode_path.string(), aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        ELOG("failed to import model {}: {}", mode_path, importer.GetErrorString());
        return MyErrCode::kFailed;
    }
    ILOG("loading {}", mode_path);
    bbox_.lower = glm::vec3(std::numeric_limits<float>::max());
    bbox_.high = glm::vec3(std::numeric_limits<float>::min());
    CHECK_ERR_RET(visitNode(scene, scene->mRootNode));
    index_size_ = indices_.size();
    return MyErrCode::kOk;
}

MyErrCode Scene::unloadModel()
{
    vertices_.clear();
    vertices_.shrink_to_fit();
    indices_.clear();
    indices_.shrink_to_fit();
    return MyErrCode::kOk;
}
