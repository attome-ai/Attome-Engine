#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

// Forward declaration for SDL structures
struct SDL_Texture;

// Forward declaration for TinyGLTF
namespace tinygltf {
    class Model;
    struct Accessor;
    struct Material;
    class Node;
}

// GLM for math operations
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>



// Complete VulkanStaticMeshInfo structure with advanced fields
struct VulkanStaticMeshInfo
{
    std::vector<glm::vec3> vertices;      // 3D positions
    std::vector<uint32_t> indices;
    std::vector<uint32_t> color;
    std::vector<glm::vec3> normals;       // 3D normal vectors
    std::vector<glm::vec2> texCoords;     // 2D UV coordinates
    std::vector<SDL_Texture*> textures;

    // Additional fields for advanced cases
    std::vector<glm::vec4> tangents;      // Tangent vectors for normal mapping (vec4 for handedness)
    std::vector<glm::vec3> bitangents;    // Bitangent vectors for normal mapping
    std::vector<glm::uvec4> joints;       // Joint indices for skinning (4 influences per vertex)
    std::vector<glm::vec4> weights;       // Weights for skinning (4 weights per vertex)
    std::string name;                     // Mesh name
    std::vector<VulkanStaticMeshInfo> submeshes; // For hierarchical models
    glm::mat4 transform;                  // Local transform matrix

    // Material properties
    struct Material {
        glm::vec4 baseColorFactor;        // Base color multiplier
        glm::vec3 emissiveFactor;         // Emissive color
        float metallicFactor;             // Metallic value
        float roughnessFactor;            // Roughness value
        float alphaCutoff;                // Alpha cutoff value
        bool doubleSided;                 // Whether the material is double-sided
        std::vector< SDL_Texture*> textures; // Textures for the material
        // PBR texture indices
        int baseColorTextureIndex;        // Diffuse/albedo texture
        int normalTextureIndex;           // Normal map texture
        int metallicRoughnessTextureIndex; // Metallic-roughness texture
        int emissiveTextureIndex;         // Emissive texture
        int occlusionTextureIndex;        // Ambient occlusion texture

        Material() :
            baseColorFactor(1.0f),
            emissiveFactor(0.0f),
            metallicFactor(1.0f),
            roughnessFactor(1.0f),
            alphaCutoff(0.5f),
            doubleSided(false),
            baseColorTextureIndex(-1),
            normalTextureIndex(-1),
            metallicRoughnessTextureIndex(-1),
            emissiveTextureIndex(-1),
            occlusionTextureIndex(-1) {}
    } material;

    // Animation data
    bool hasAnimation;
    std::vector<std::string> boneNames;
    std::vector<glm::mat4> inverseBindMatrices;
    std::vector<std::vector<std::pair<float, glm::mat4>>> animationKeyframes;

    // Constructor with default values
    VulkanStaticMeshInfo() : hasAnimation(false), transform(1.0f) {}

    // Clean up textures when the mesh is destroyed
    ~VulkanStaticMeshInfo() {};
};

class VulkanMeshLoader {
public:
    VulkanMeshLoader();
    ~VulkanMeshLoader();

    // Main loading function
    bool LoadFromGLB(const std::string& filename, std::vector<VulkanStaticMeshInfo>& outMeshes);

private:
    // Helper functions for processing GLTF data
    bool ProcessNode(const tinygltf::Model& model, int nodeIndex,
        std::vector<VulkanStaticMeshInfo>& outMeshes,
        const glm::mat4& parentTransform = glm::mat4(1.0f));

    bool ProcessMesh(const tinygltf::Model& model, int meshIndex,
        VulkanStaticMeshInfo& outMesh);

    bool LoadMaterialTextures(const tinygltf::Model& model,
        const tinygltf::Material& material,
        VulkanStaticMeshInfo::Material& outMaterial);

    bool ProcessAccessor(const tinygltf::Model& model,
        const tinygltf::Accessor& accessor,
        std::vector<uint8_t>& outputBuffer);

    SDL_Texture* LoadTextureFromGLTF(const tinygltf::Model& model, int textureIndex);

    // Helper for matrix transformations
    glm::mat4 GetNodeTransform(const tinygltf::Node& node);
    // Animation processing
    bool ProcessAnimations(const tinygltf::Model& model, VulkanStaticMeshInfo& outMesh);

   
    void MergeVerticesByDistance(VulkanStaticMeshInfo& mesh, float threshold = 0.00001);
};