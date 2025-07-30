#version 450
#include "common.glsl"

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
layout(binding = 1) uniform sampler2D texSampler;

void phongLighting() {
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(ubo.lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec4 texColor = texture(texSampler, fragTexCoord);
    vec3 viewPos = vec3(0.0f, 0.0f, 0.0f);
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);

    vec3 ambient = 0.05f * ubo.lightColor;
    vec3 diffuse = diff * ubo.lightColor;
    vec3 specular = 0.5f * spec * ubo.lightColor;

    outColor = vec4(ambient + diffuse + specular, 1.0) * texColor;
}

void main() {
    phongLighting();
}
