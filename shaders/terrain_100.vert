#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 proj;
    mat4 view;
} ubo;
// 


layout(push_constant) uniform PushConstants {
    mat4 view;
} pcs;

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inColor; // ? What to do with this?  Need normals.
// The whole uniform object needs to be expanded as well.

layout(location = 0) out vec3 fragColor;

void main() {
    // gl_Position = ubo.proj * pcs.view * ubo.model * vec4(inPosition);
    // vec4 pos = vec4(inPosition);
    vec4 pos4_0 = vec4(inPosition);

    // pos4_0.x = pos4_0.x + (ubo.proj[0][0] * 100.0);
    pos4_0.x = pos4_0.x + 0.5;
    // pos4_0.y = pos4_0.y + 0.50;
    pos4_0.y = pos4_0.y + ubo.proj[2][2];
    // vec4 pos4_1 = vec4(pos4_0[0] + ubo.proj[2], pos4_0[1], pos4_0[2], pos4_0[3]);
    gl_Position = pos4_0;
    // gl_Position = ubo.proj * pos;
    // gl_Position = vec4(inPosition);
    // gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition);
    // vec4 pos = vec4(inPosition);
    // gl_Position = ubo.model * pos;
    fragColor = vec3(inColor);
}
