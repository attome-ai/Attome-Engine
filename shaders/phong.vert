#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;
layout(location = 0) out vec3 outPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outTexCoord;
layout(set = 0, binding = 0) uniform Globals {
    mat4 viewMatrix;
    mat4 projectionMatrix;
    mat4 modelMatrix;
    vec4 lightPos;
};
void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * viewMatrix * worldPos;
    outPos = worldPos.xyz;
    outNormal = mat3(modelMatrix) * normal;
    outTexCoord = texCoord;
}