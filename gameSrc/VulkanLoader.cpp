#include "VulkanLoader.h"
#include <iostream>
#include <filesystem>

// For GLB parsing
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

// SDL for textures
#include <SDL3/SDL.h>
#include <SDL3_image/SDL_image.h>

// Additional GLM includes
#include <glm/gtc/type_ptr.hpp>


VulkanMeshLoader::VulkanMeshLoader() {
}

VulkanMeshLoader::~VulkanMeshLoader() {
}

bool VulkanMeshLoader::LoadFromGLB(const std::string& filename, std::vector<VulkanStaticMeshInfo>& outMeshes) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);

    if (!warn.empty()) {
        std::cout << "GLTF Warning: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cout << "GLTF Error: " << err << std::endl;
    }

    if (!ret) {
        std::cout << "Failed to load GLB file: " << filename << std::endl;
        return false;
    }

    // Process all scenes and nodes
    const tinygltf::Scene& scene = model.scenes[model.defaultScene];
    for (int nodeIndex : scene.nodes) {
        ProcessNode(model, nodeIndex, outMeshes);
    }

    // Apply vertex merging to all loaded meshes
    for (auto& mesh : outMeshes) {
        MergeVerticesByDistance(mesh);
    }

    return true;
}

bool VulkanMeshLoader::ProcessNode(const tinygltf::Model& model, int nodeIndex,
    std::vector<VulkanStaticMeshInfo>& outMeshes,
    const glm::mat4& parentTransform) {
    const tinygltf::Node& node = model.nodes[nodeIndex];

    // Calculate node's transform
    glm::mat4 localTransform = GetNodeTransform(node);
    glm::mat4 globalTransform = parentTransform * localTransform;

    // Process mesh if this node has one
    if (node.mesh >= 0) {
        VulkanStaticMeshInfo meshInfo;
        if (ProcessMesh(model, node.mesh, meshInfo)) {
            meshInfo.transform = globalTransform;
            meshInfo.name = node.name;
            outMeshes.push_back(meshInfo);
        }
    }

    // Process child nodes
    for (int childIndex : node.children) {
        ProcessNode(model, childIndex, outMeshes, globalTransform);
    }

    return true;
}

bool VulkanMeshLoader::ProcessMesh(const tinygltf::Model& model, int meshIndex,
    VulkanStaticMeshInfo& outMesh) {
    const tinygltf::Mesh& mesh = model.meshes[meshIndex];

    // Process each primitive in the mesh
    for (const tinygltf::Primitive& primitive : mesh.primitives) {
        // Get vertex positions
        if (primitive.attributes.find("POSITION") != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("POSITION")];
            std::vector<uint8_t> buffer;
            ProcessAccessor(model, accessor, buffer);
            outMesh.vertices.resize(accessor.count);
            memcpy(outMesh.vertices.data(), buffer.data(), buffer.size());
        }

        // Get vertex normals
        if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("NORMAL")];
            std::vector<uint8_t> buffer;
            ProcessAccessor(model, accessor, buffer);
            outMesh.normals.resize(accessor.count);
            memcpy(outMesh.normals.data(), buffer.data(), buffer.size());
        }

        // Get texture coordinates
        if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
            std::vector<uint8_t> buffer;
            ProcessAccessor(model, accessor, buffer);
            outMesh.texCoords.resize(accessor.count);
            memcpy(outMesh.texCoords.data(), buffer.data(), buffer.size());
        }

        // Get tangents
        if (primitive.attributes.find("TANGENT") != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("TANGENT")];
            std::vector<uint8_t> buffer;
            ProcessAccessor(model, accessor, buffer);
            outMesh.tangents.resize(accessor.count);
            memcpy(outMesh.tangents.data(), buffer.data(), buffer.size());
        }

        // Get indices with proper type conversion
        if (primitive.indices >= 0) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
            std::vector<uint8_t> buffer;
            ProcessAccessor(model, accessor, buffer);
            outMesh.indices.resize(accessor.count);

            // Convert indices based on component type
            switch (accessor.componentType) {
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                const uint8_t* src = reinterpret_cast<const uint8_t*>(buffer.data());
                for (size_t i = 0; i < accessor.count; ++i) {
                    outMesh.indices[i] = static_cast<uint32_t>(src[i]);
                }
                break;
            }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                const uint16_t* src = reinterpret_cast<const uint16_t*>(buffer.data());
                for (size_t i = 0; i < accessor.count; ++i) {
                    outMesh.indices[i] = static_cast<uint32_t>(src[i]);
                }
                break;
            }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                const uint32_t* src = reinterpret_cast<const uint32_t*>(buffer.data());
                for (size_t i = 0; i < accessor.count; ++i) {
                    outMesh.indices[i] = src[i];
                }
                break;
            }
            default:
                std::cout << "Warning: Unsupported index component type: " << accessor.componentType << std::endl;
                break;
            }
        }

        // Process material
        if (primitive.material >= 0) {
            const tinygltf::Material& material = model.materials[primitive.material];
            LoadMaterialTextures(model, material, outMesh.material);
        }

        // Process skinning data if present
        if (primitive.attributes.find("JOINTS_0") != primitive.attributes.end() &&
            primitive.attributes.find("WEIGHTS_0") != primitive.attributes.end()) {

            outMesh.hasAnimation = true;

            // Load joint indices
            const tinygltf::Accessor& jointsAccessor = model.accessors[primitive.attributes.at("JOINTS_0")];
            std::vector<uint8_t> jointsBuffer;
            ProcessAccessor(model, jointsAccessor, jointsBuffer);
            outMesh.joints.resize(jointsAccessor.count);
            memcpy(outMesh.joints.data(), jointsBuffer.data(), jointsBuffer.size());

            // Load weights
            const tinygltf::Accessor& weightsAccessor = model.accessors[primitive.attributes.at("WEIGHTS_0")];
            std::vector<uint8_t> weightsBuffer;
            ProcessAccessor(model, weightsAccessor, weightsBuffer);
            outMesh.weights.resize(weightsAccessor.count);
            memcpy(outMesh.weights.data(), weightsBuffer.data(), weightsBuffer.size());
        }
    }

    return true;
}

bool VulkanMeshLoader::LoadMaterialTextures(const tinygltf::Model& model,
    const tinygltf::Material& material,
    VulkanStaticMeshInfo::Material& outMaterial) {
    // Base color
    if (material.pbrMetallicRoughness.baseColorTexture.index >= 0) {
        outMaterial.baseColorTextureIndex = material.pbrMetallicRoughness.baseColorTexture.index;
        SDL_Texture* texture = LoadTextureFromGLTF(model, material.pbrMetallicRoughness.baseColorTexture.index);
        if (texture) {
            outMaterial.baseColorTextureIndex = outMaterial.textures.size();
            outMaterial.textures.push_back(texture);
        }
    }

    // Normal map
    if (material.normalTexture.index >= 0) {
        SDL_Texture* texture = LoadTextureFromGLTF(model, material.normalTexture.index);
        if (texture) {
            outMaterial.normalTextureIndex = outMaterial.textures.size();
            outMaterial.textures.push_back(texture);
        }
    }

    // Metallic-roughness
    if (material.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0) {
        SDL_Texture* texture = LoadTextureFromGLTF(model, material.pbrMetallicRoughness.metallicRoughnessTexture.index);
        if (texture) {
            outMaterial.metallicRoughnessTextureIndex = outMaterial.textures.size();
            outMaterial.textures.push_back(texture);
        }
    }

    // Material factors
    outMaterial.metallicFactor = material.pbrMetallicRoughness.metallicFactor;
    outMaterial.roughnessFactor = material.pbrMetallicRoughness.roughnessFactor;

    if (material.pbrMetallicRoughness.baseColorFactor.size() == 4) {
        outMaterial.baseColorFactor = glm::vec4(
            material.pbrMetallicRoughness.baseColorFactor[0],
            material.pbrMetallicRoughness.baseColorFactor[1],
            material.pbrMetallicRoughness.baseColorFactor[2],
            material.pbrMetallicRoughness.baseColorFactor[3]
        );
    }

    outMaterial.doubleSided = material.doubleSided;
    outMaterial.alphaCutoff = material.alphaCutoff;

    return true;
}

bool VulkanMeshLoader::ProcessAccessor(const tinygltf::Model& model,
    const tinygltf::Accessor& accessor,
    std::vector<uint8_t>& outputBuffer) {
    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

    size_t stride = accessor.ByteStride(bufferView);
    size_t offset = accessor.byteOffset + bufferView.byteOffset;
    size_t elementSize = tinygltf::GetComponentSizeInBytes(accessor.componentType) *
        tinygltf::GetNumComponentsInType(accessor.type);

    outputBuffer.resize(accessor.count * elementSize);

    const uint8_t* dataPtr = buffer.data.data() + offset;
    if (stride == elementSize) {
        memcpy(outputBuffer.data(), dataPtr, accessor.count * elementSize);
    }
    else {
        for (size_t i = 0; i < accessor.count; ++i) {
            memcpy(&outputBuffer[i * elementSize], dataPtr + (i * stride), elementSize);
        }
    }

    return true;
}

SDL_Texture* VulkanMeshLoader::LoadTextureFromGLTF(const tinygltf::Model& model, int textureIndex) {
    if (textureIndex < 0 || textureIndex >= model.textures.size()) {
        return nullptr;
    }

    const tinygltf::Texture& texture = model.textures[textureIndex];
    const tinygltf::Image& image = model.images[texture.source];

    // Implementation depends on your specific SDL setup
    // This is a placeholder - you'll need to implement actual texture loading
    return nullptr;
}

glm::mat4 VulkanMeshLoader::GetNodeTransform(const tinygltf::Node& node) {
    glm::mat4 transform(1.0f);

    // If matrix is provided, use it directly
    if (node.matrix.size() == 16) {
        transform = glm::mat4(
            node.matrix[0], node.matrix[1], node.matrix[2], node.matrix[3],
            node.matrix[4], node.matrix[5], node.matrix[6], node.matrix[7],
            node.matrix[8], node.matrix[9], node.matrix[10], node.matrix[11],
            node.matrix[12], node.matrix[13], node.matrix[14], node.matrix[15]
        );
    }
    else {
        // Otherwise compose transform from TRS components
        glm::vec3 translation(0.0f);
        glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 scale(1.0f);

        if (node.translation.size() == 3) {
            translation = glm::vec3(
                node.translation[0],
                node.translation[1],
                node.translation[2]
            );
        }

        if (node.rotation.size() == 4) {
            rotation = glm::quat(
                node.rotation[3],  // w
                node.rotation[0],  // x
                node.rotation[1],  // y
                node.rotation[2]   // z
            );
        }

        if (node.scale.size() == 3) {
            scale = glm::vec3(
                node.scale[0],
                node.scale[1],
                node.scale[2]
            );
        }

        transform = glm::translate(glm::mat4(1.0f), translation) *
            glm::mat4_cast(rotation) *
            glm::scale(glm::mat4(1.0f), scale);
    }

    return transform;
}

bool VulkanMeshLoader::ProcessAnimations(const tinygltf::Model& model, VulkanStaticMeshInfo& outMesh) {
    if (model.animations.empty()) {
        return false;
    }

    outMesh.hasAnimation = true;

    // Process each animation
    for (const auto& animation : model.animations) {
        // Create a new vector for this animation's keyframes
        std::vector<std::pair<float, glm::mat4>> currentAnimationKeyframes;

        for (const auto& channel : animation.channels) {
            const auto& sampler = animation.samplers[channel.sampler];
            const auto& targetNode = model.nodes[channel.target_node];

            // Get input times
            const auto& timeAccessor = model.accessors[sampler.input];
            std::vector<uint8_t> timeData;
            ProcessAccessor(model, timeAccessor, timeData);
            const float* times = reinterpret_cast<const float*>(timeData.data());

            // Get output values
            const auto& valueAccessor = model.accessors[sampler.output];
            std::vector<uint8_t> valueData;
            ProcessAccessor(model, valueAccessor, valueData);

            // Process keyframes based on the animation path
            if (channel.target_path == "translation" ||
                channel.target_path == "rotation" ||
                channel.target_path == "scale") {

                // Create transformation matrices for each keyframe
                for (size_t i = 0; i < timeAccessor.count; ++i) {
                    glm::mat4 transform(1.0f);
                    // Implementation depends on the path type
                    // Store the keyframe
                    currentAnimationKeyframes.push_back(
                        std::make_pair(times[i], transform)
                    );
                }
            }
        }

        // Add this animation's keyframes to the mesh's animations
        outMesh.animationKeyframes.push_back(currentAnimationKeyframes);
    }

    return true;
}


void VulkanMeshLoader::MergeVerticesByDistance(VulkanStaticMeshInfo& mesh, float threshold ) {
    if (mesh.vertices.empty() || mesh.indices.empty()) return;

    std::vector<glm::vec3> uniqueVertices;
    std::vector<uint32_t> newIndices;
    std::vector<glm::vec3> newNormals;
    std::vector<glm::vec2> newTexCoords;
    std::vector<glm::vec4> newTangents;
    std::vector<glm::vec3> newBitangents;
    std::vector<glm::uvec4> newJoints;
    std::vector<glm::vec4> newWeights;

    // Map to store unique vertex data
    struct VertexData {
        uint32_t index;
        glm::vec3 normal;
        glm::vec2 texCoord;
        glm::vec4 tangent;
        glm::vec3 bitangent;
        glm::uvec4 joint;
        glm::vec4 weight;
    };

    std::unordered_map<uint32_t, std::vector<VertexData>> gridCells;
    const float gridSize = threshold * 2.0f;

    // Helper function to generate grid cell key
    auto getGridKey = [gridSize](const glm::vec3& pos) -> uint32_t {
        int x = static_cast<int>(pos.x / gridSize);
        int y = static_cast<int>(pos.y / gridSize);
        int z = static_cast<int>(pos.z / gridSize);
        return static_cast<uint32_t>((x * 73856093) ^ (y * 19349663) ^ (z * 83492791));
        };

    // First pass: Build spatial hash grid
    for (size_t i = 0; i < mesh.vertices.size(); i++) {
        uint32_t key = getGridKey(mesh.vertices[i]);
        VertexData vd;
        vd.index = i;
        if (!mesh.normals.empty()) vd.normal = mesh.normals[i];
        if (!mesh.texCoords.empty()) vd.texCoord = mesh.texCoords[i];
        if (!mesh.tangents.empty()) vd.tangent = mesh.tangents[i];
        if (!mesh.bitangents.empty()) vd.bitangent = mesh.bitangents[i];
        if (!mesh.joints.empty()) vd.joint = mesh.joints[i];
        if (!mesh.weights.empty()) vd.weight = mesh.weights[i];
        gridCells[key].push_back(vd);
    }

    std::vector<uint32_t> oldToNewIndex(mesh.vertices.size(), UINT32_MAX);

    // Second pass: Merge vertices within threshold
    for (size_t i = 0; i < mesh.vertices.size(); i++) {
        if (oldToNewIndex[i] != UINT32_MAX) continue;

        uint32_t key = getGridKey(mesh.vertices[i]);
        const auto& cell = gridCells[key];

        uint32_t newIndex = static_cast<uint32_t>(uniqueVertices.size());
        uniqueVertices.push_back(mesh.vertices[i]);

        if (!mesh.normals.empty()) newNormals.push_back(mesh.normals[i]);
        if (!mesh.texCoords.empty()) newTexCoords.push_back(mesh.texCoords[i]);
        if (!mesh.tangents.empty()) newTangents.push_back(mesh.tangents[i]);
        if (!mesh.bitangents.empty()) newBitangents.push_back(mesh.bitangents[i]);
        if (!mesh.joints.empty()) newJoints.push_back(mesh.joints[i]);
        if (!mesh.weights.empty()) newWeights.push_back(mesh.weights[i]);

        oldToNewIndex[i] = newIndex;

        // Check nearby vertices
        for (const auto& vd : cell) {
            if (oldToNewIndex[vd.index] != UINT32_MAX) continue;

            float dist = glm::length(mesh.vertices[i] - mesh.vertices[vd.index]);
            if (dist <= threshold) {
                oldToNewIndex[vd.index] = newIndex;
            }
        }
    }

    // Update indices
    newIndices.reserve(mesh.indices.size());
    for (uint32_t idx : mesh.indices) {
        newIndices.push_back(oldToNewIndex[idx]);
    }

    // Update mesh data
    mesh.vertices = std::move(uniqueVertices);
    mesh.indices = std::move(newIndices);
    if (!mesh.normals.empty()) mesh.normals = std::move(newNormals);
    if (!mesh.texCoords.empty()) mesh.texCoords = std::move(newTexCoords);
    if (!mesh.tangents.empty()) mesh.tangents = std::move(newTangents);
    if (!mesh.bitangents.empty()) mesh.bitangents = std::move(newBitangents);
    if (!mesh.joints.empty()) mesh.joints = std::move(newJoints);
    if (!mesh.weights.empty()) mesh.weights = std::move(newWeights);
}