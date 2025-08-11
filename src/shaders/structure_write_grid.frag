

#version 450
#extension GL_ARB_separate_shader_objects : enable

// #extension GL_EXT_debug_printf : enable


layout(location = 0) in vec4 fragColor;
layout(location = 1) flat in int frag_id;

layout(location = 0) out vec4 outColor;



void main() {
     

        outColor = fragColor;
   

}
    
    
