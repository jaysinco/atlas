#version 450
#include "vkwl.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPos, 1.0);
    fragPos = vec3(ubo.view * ubo.model * vec4(inPos, 1.0));
    fragNormal = mat3(transpose(inverse(ubo.view * ubo.model))) * inNormal;
    fragTexCoord = inTexCoord;
}
