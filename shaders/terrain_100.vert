#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 proj;
    mat4 view;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 view;
} pcs;

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.proj * pcs.view * ubo.model * vec4(inPosition);
    // gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition);
    fragColor = vec3(inColor);
}
