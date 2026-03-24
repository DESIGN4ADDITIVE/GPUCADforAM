

#version 450

#extension GL_ARB_viewport_array : enable

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;//triangle_strip

const int num =  1;
layout(invocations = num) in;

layout(binding = 0) uniform UniformBufferObject {
	mat4 modelViewProj[1];
} ubo;

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
        int show_region;
        int show_domain;
        int analysis;

} ;

layout(location = 0) in vec4 posit[];
layout(location = 1) in vec4 norm[];
layout(location = 2) in float field[];



layout (location = 0) out vec4 fragColor;
layout (location = 1) out int  prim_id;


float saturate (float x)
{
    return min(1.0, max(0.0,x));
}
vec3 saturate (vec3 x)
{
    return min(vec3(1.,1.,1.), max(vec3(0.,0.,0.),x));
}

vec3 spectral_jet(float x)
{
 
	vec3 c;

	if (x < 0.25)
		c = vec3(0.0, 4.0 * x, 1.0);
	else if (x < 0.5)
		c = vec3(0.0, 1.0, 1.0 + 4.0 * (0.25 - x));
	else if (x < 0.75)
		c = vec3(4.0 * (x - 0.5), 1.0, 0.0);
	else
		c = vec3(1.0, 1.0 + 4.0 * (0.75 - x), 0.0);

	// Clamp colour components in [0,1]
	return saturate(c);
}


void main(void)
{	
    prim_id = gl_PrimitiveIDIn;
    
	for(int i = 0; i < 3 ; i++)
	{

	
        vec3 n_normal = normalize(norm[i].xyz*1.0);
	
		vec3 lightvector = normalize(eyes.xyz - posit[i].xyz);
		
		vec3 lightcolor1 = vec3(1.0,1.0,1.0);
	
        vec3 lightcolor2 = vec3(0.0,1.0,1.0);

        vec3 lightcolor3 = vec3(1.0,1.0,0.0);

        vec3 lightcolor4 = vec3(0.6,1.0,0.3);

		float amg = abs(dot(lightvector.xyz,n_normal.xyz));

        float coll = norm[i].w;
		
		if (gl_InvocationID == 0)
			{
                gl_Position =ubo.modelViewProj[0]*posit[i];
                if(analysis == 0)
                {
                    if((make_region > 0) || (show_domain > 0) )
                    {
                        if(norm[i].w == 0.25)
                        {
                            
                            fragColor = vec4(lightcolor4*amg,norm[i].w);
                        }
                        else if (norm[i].w == 0.5)
                        {
                    
                            fragColor = vec4(lightcolor3*amg,norm[i].w);
                        }
                        else
                        {
                            fragColor = vec4(lightcolor2*amg,norm[i].w);
                        }
                    }
                    else if(show_region > 0)
                    {
                        fragColor = vec4(lightcolor4*amg,norm[i].w);
                    }
               
    
                    else
                    {
                        fragColor = vec4(lightcolor1*amg,norm[i].w);
                    }
                }

                if(analysis > 0)
                {
                    fragColor = vec4(spectral_jet(coll).xyz*amg, 1.0);
                }

                if(val[prim_id] == 1)
                {
                    fragColor = vec4(1.0,0.0,0.0,1.0);
                }

                if(val[prim_id] == -1)
                {
                    fragColor = vec4(1.0,1.0,0.0,1.0);
                }

                gl_PointSize =float(2);

			}

		gl_ViewportIndex =gl_InvocationID;

		
        
		EmitVertex();
	}

	EndPrimitive();

}