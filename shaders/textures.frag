#version 450
layout(location = 0) in vec2 outTexCoord;
layout(location = 0) out vec4 fragColor;
layout(binding = 0) uniform sampler2D texSampler;
void main() {
    fragColor = texture(texSampler, outTexCoord);
}