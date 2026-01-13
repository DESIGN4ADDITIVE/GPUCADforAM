

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shader_storage_buffer_object : enable
#extension GL_EXT_debug_printf : enable


layout(location = 0) in vec4 fragColorout;
layout(location = 1) flat in int frag_id ;
layout(location = 2) flat in float r_val;




layout (input_attachment_index = 0, binding = 2) uniform subpassInput inputColor;
layout (input_attachment_index = 1, binding = 3) uniform subpassInput inputDepth;

layout (binding = 1) buffer storageBbuffer
{
  float val[]; 
} ; 


layout(location = 0) out vec4 outColor;


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
        int make_region;

} ;

float a = mouse_x - pix_delta;
float b = mouse_x + pix_delta;
float c = mouse_y - pix_delta;
float d = mouse_y + pix_delta;

void main() {
     
        
        

        
        vec4 colorr = subpassLoad(inputColor).rgba;
        float ch_depth = subpassLoad(inputDepth).r;


        if(boundary > 0)
        {
                if((gl_FragCoord.z == ch_depth) && (fragColorout.w >= 0.6))
                {
                        
                        if(fragColorout.w == 0.7)
                        {
                                outColor = vec4(0.0,0.8,1.0,alpha_val);
                        }
                        
                        else if(fragColorout.w == 0.8)
                        {
                                outColor = vec4(0.5,1.0,0.5,alpha_val);
                        }


                        else if(fragColorout.w == 0.9)
                        {
                                outColor = vec4(1.0,0.0,0.5,alpha_val);
                        }


                        else if(fragColorout.w == 0.925)
                        {
                                outColor = vec4(fragColorout.xyz,alpha_val);
                        }


                        else if(fragColorout.w == 1.0)
                        {
                                outColor = vec4(1.0,0.0,1.0,alpha_val);
                        }



                        else if(fragColorout.w == 0.625)
                        {
                                outColor = vec4(0.85,1.0,0.5,alpha_val);
                        }

                        else
                        {
                                outColor = vec4(fragColorout.xyz,alpha_val);
                        }
                        
                
                        if ((mouse_click == 2) && (support == -1) && (a < gl_FragCoord.x) && (gl_FragCoord.x < b) && (c < gl_FragCoord.y) && (gl_FragCoord.y < d) )
                        {
                
                                val[frag_id] = -1.0;
                        
                        }

                        else if ((mouse_click == 1) && (support == 1) && (a < gl_FragCoord.x) && (gl_FragCoord.x < b) && (c < gl_FragCoord.y) && (gl_FragCoord.y < d))
                        {
                                val[frag_id] = 1.0;
                        }

                        
                        else if ((mouse_click == -1) && (support == -1) && (a < gl_FragCoord.x) && (gl_FragCoord.x < b) && (c < gl_FragCoord.y) && (gl_FragCoord.y < d))
                        {
                                val[frag_id] = 0.0;
                        }
                      
                        
                }

                else
                {
                        
                        discard;
                }
        }

        else
        {       

                

                if((gl_FragCoord.z == ch_depth))
                {
                 
                        
                        outColor = vec4(fragColorout.xyz,alpha_val);
                     
                  
                        
                        if ((mouse_click == 2) && (support == -1) && (a < gl_FragCoord.x) && (gl_FragCoord.x < b) && (c < gl_FragCoord.y) && (gl_FragCoord.y < d) )
                        {
                
                                val[frag_id] = -1.0;
                        
                        }

                        else if ((mouse_click == 1) && (support == 1) && (a < gl_FragCoord.x) && (gl_FragCoord.x < b) && (c < gl_FragCoord.y) && (gl_FragCoord.y < d))
                        {
                                val[frag_id] = 1.0;
                        }

                        
                        else if ((mouse_click == -1) && (support == -1) && (a < gl_FragCoord.x) && (gl_FragCoord.x < b) && (c < gl_FragCoord.y) && (gl_FragCoord.y < d))
                        {
                                val[frag_id] = 0.0;
                        }
                

        

                
                        
                        
                        
                }

                else
                {
                        
                        discard;
                }

        }
   
}
    
    
