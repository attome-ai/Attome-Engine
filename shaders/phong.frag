#version 450
layout(location = 0) in vec3 outPos;
layout(location = 1) in vec3 outNormal;
layout(location = 2) in vec2 outTexCoord;
layout(location = 0) out vec4 fragColor;
layout(set = 0, binding = 0) uniform Globals {
    vec4 lightPos;
    vec4 viewPos;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float shininess;
};
void main() {
    vec3 norm = normalize(outNormal);
    vec3 lightDir = normalize(lightPos.xyz - outPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 viewDir = normalize(viewPos.xyz - outPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 result = ambient.rgb + diff * diffuse.rgb + spec * specular.rgb;
    fragColor = vec4(result, 1.0);
}