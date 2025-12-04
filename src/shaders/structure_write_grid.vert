

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shader_storage_buffer_object : enable
#extension GL_EXT_debug_printf : enable

layout(location = 0) in float height;
layout(location = 1) in vec3 xyzPos;
layout(location = 2) in float raster;

layout(location = 0) out float hei;
layout(location = 1) out vec4 pos;
layout(location = 2) out float rasterr;



void main() {
    
    hei = height;
    rasterr = raster;
    pos = vec4(xyzPos.xyz,1.0);  

}
