

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 normal;


layout(location = 0) out vec4 posit;
layout(location = 1) out vec4 norm;

void main() {

    posit = vec4(position.xyzw);
    norm = vec4(normal.xyzw);
  
}