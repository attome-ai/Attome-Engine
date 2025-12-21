#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;
layout(location = 0) out vec2 outTexCoord;
layout(set = 0, binding = 0) uniform Globals {
    mat4 viewMatrix;
    mat4 projectionMatrix;
};
void main() {
    gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
    outTexCoord = texCoord;
}