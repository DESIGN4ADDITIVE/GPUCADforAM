

#version 450
#extension GL_ARB_separate_shader_objects : enable

// #extension GL_EXT_debug_printf : enable


layout(location = 0) in vec4 fragColor;
layout(location = 1) flat in int frag_id;
layout(location = 2) flat in float r_val;

layout(location = 0) out vec4 outColor;



void main() {
     
        if(fragColor.w >= 0.7)
        {
                if(fragColor.w == 0.7)
                {
                        outColor = vec4(0.0,0.8,1.0,1.0);
                }
                
                else if(fragColor.w == 0.8)
                {
                        outColor = vec4(0.5,1.0,0.5,1.0);
                }


                else if(fragColor.w == 0.9)
                {
                        outColor = vec4(1.0,0.0,0.5,1.0);
                }


                else if(fragColor.w == 1.0)
                {
                        outColor = vec4(1.0,0.0,1.0,1.0);
                }
        }
        else
        {
                discard;
        }
     
}
    
    
