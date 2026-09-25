#version 460
// Fullscreen triangle (no vertex buffer). z = 0: the far plane in reverse-Z,
// so the sky pass (depth test EQUAL) only shades pixels no geometry covered.
layout(location = 0) out vec2 vUv;

void main() {
  vec2 uv = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
  vUv = uv;
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
