#version 450
// Built-in masked colour clear: the colour of the push constants into attachment LOC (TYPE: vec4 / uvec4 / ivec4).
layout(push_constant) uniform P { vec4 color; float depth; } pc;
layout(location = LOC) out TYPE o;
void main() { o = TYPE(pc.color); }
