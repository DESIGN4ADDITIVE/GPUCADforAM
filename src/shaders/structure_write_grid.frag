

#version 450
#extension GL_ARB_separate_shader_objects : enable

// #extension GL_EXT_debug_printf : enable


layout(location = 0) in vec4 fragColor;
layout(location = 1) flat in int frag_id;
layout(location = 2) flat in float r_val;


layout( push_constant, std430) uniform push_constants
{
	vec4 eyes;
        float p_size_1;
        float p_size_2;
        float p_size_3;
        float p_size_4;
      	float mouse_x;
        float mouse_y;
        int mouse_click;
        float pix_delta;
        int support;
        float point_size;
        int boundary;
        float alpha_val;

} ;

layout(location = 0) out vec4 outColor;



void main() {
     

        
        if(boundary > 0)
        {
                if(fragColor.w >= 0.6)
                {
                        outColor = vec4(0.0,0.0,1.0,1.0);
                }
                else 
                {       
                        discard;
                }
        }
        else
        {
                outColor = vec4(0.0,0.0,1.0,alpha_val);
        }
        
        
   
     
}
    
    
