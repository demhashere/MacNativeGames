#version 450
// FSR's RCAS pass (vk_fsr.c): one triangle over the whole target; the viewport and scissor take the picture's rectangle.
void main() {
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
