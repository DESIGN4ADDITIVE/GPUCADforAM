

#version 450
#extension GL_ARB_separate_shader_objects : enable


layout (input_attachment_index = 0, binding = 2) uniform subpassInput inputColor;
layout (input_attachment_index = 1, binding = 3) uniform subpassInput inputDepth;

layout (binding = 1) buffer storageBbuffer
{
  float val[]; 
} ; 

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



layout(location = 0) in  vec4 fragColor;
layout(location = 1) flat in int prim_id ;


layout(location = 0)  out vec4 outColor;


float a = mouse_x - pix_delta;
float b = mouse_x + pix_delta;
float c = mouse_y - pix_delta;
float d = mouse_y + pix_delta;

void main() 
{
        vec4 colorr = subpassLoad(inputColor).rgba;
        float ch_depth = subpassLoad(inputDepth).r;


        if(make_region > 0)
        {
                outColor = fragColor;

                if ((mouse_click == 2) && (support == -1) && (a < gl_FragCoord.x) && (gl_FragCoord.x < b) && (c < gl_FragCoord.y) && (gl_FragCoord.y < d) )
                {

                        val[prim_id] = -1.0;

                }

                else if ((mouse_click == 1) && (support == 1) && (a < gl_FragCoord.x) && (gl_FragCoord.x < b) && (c < gl_FragCoord.y) && (gl_FragCoord.y < d))
                {
                        val[prim_id] = 1.0;
                }


                else if ((mouse_click == -1) && (support == -1) && (a < gl_FragCoord.x) && (gl_FragCoord.x < b) && (c < gl_FragCoord.y) && (gl_FragCoord.y < d))
                {
                        val[prim_id] = 0.0;
                }

                
        }

        else
        {
                if(gl_FragCoord.z == ch_depth)
                {

                        outColor = fragColor;

                        if ((mouse_click == 2) && (support == -1) && (a < gl_FragCoord.x) && (gl_FragCoord.x < b) && (c < gl_FragCoord.y) && (gl_FragCoord.y < d) )
                        {
                
                                val[prim_id] = -1.0;
                        
                        }

                        else if ((mouse_click == 1) && (support == 1) && (a < gl_FragCoord.x) && (gl_FragCoord.x < b) && (c < gl_FragCoord.y) && (gl_FragCoord.y < d))
                        {
                                val[prim_id] = 1.0;
                        }

                        
                        else if ((mouse_click == -1) && (support == -1) && (a < gl_FragCoord.x) && (gl_FragCoord.x < b) && (c < gl_FragCoord.y) && (gl_FragCoord.y < d))
                        {
                                val[prim_id] = 0.0;
                        }
                      
                }
                else
                {
                        discard;
                }
        }

}