#version 450
// Built-in clears (vk_exec.c): one triangle over the whole target at the clear depth.
layout(push_constant) uniform P { vec4 color; float depth; } pc;
void main() {
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(p * 2.0 - 1.0, pc.depth, 1.0);
}
