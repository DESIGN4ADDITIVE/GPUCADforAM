


#version 450 
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shader_storage_buffer_object : enable
// #extension GL_EXT_debug_printf : enable
#extension GL_ARB_viewport_array : enable

layout (points) in;
layout (points, max_vertices = 1) out;//triangle_strip

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

} ;


layout(location = 0) in float hei[];
layout(location = 1) in vec4 pos[];
layout(location = 2) in float raster[];

layout (location = 0) out vec4 fragColor;
layout (location = 1) out int  frag_id;
layout (location = 2) out float r_val;




float saturate (float x)
{
    return min(1.0, max(0.0,x));
}

vec3 saturate (vec3 x)
{
    return min(vec3(1.,1.,1.), max(vec3(0.,0.,0.),x));
}

vec3 spectral_jet(float w)
{

	float x = saturate((w - 0.0)/ 1.0);
	vec3 c;

	if (x < 0.25)
		c = vec3(0.0, 4.0 * x, 1.0);
	else if (x < 0.5)
		c = vec3(0.0, 1.0, 1.0 + 4.0 * (0.25 - x));
	else if (x < 0.75)
		c = vec3(4.0 * (x - 0.5), 1.0, 0.0);
	else
		c = vec3(1.0, 1.0 + 4.0 * (0.75 - x), 0.0);

	return saturate(c);
}


void main(void)
{	

	for(int i = 0; i < 1 ; i++)//
	{

	
		gl_Position = ubo.modelViewProj[0]*pos[i];
		
		if (gl_InvocationID == 0)
		{
			frag_id = gl_PrimitiveIDIn;
			
			if(boundary > 0)
			{
				fragColor = vec4(0.0,0.0,1.0,0.5);
			}
			else
			{
				fragColor = vec4(0.0,0.0,1.0,0.0);
			}

			
			fragColor.w = raster[i];
			
			if(hei[i] > 0.5)
			{
				gl_PointSize = p_size_1;
			}
			else if(hei[i] > 0.25)
			{
				gl_PointSize = p_size_2;
				
			}
			else if(hei[i] > 0.0)
			{
				gl_PointSize = p_size_3;
			}
			else
			{
				gl_PointSize = p_size_4;
			}

			if(val[frag_id] == -1.0)
			{
				gl_PointSize = point_size;
				fragColor = vec4(1.0,0.0,0.0,1.0);
				fragColor.w = 1.0;
				
			}
			if(val[frag_id] == 1.0)
			{
				gl_PointSize = point_size;
				fragColor = vec4(1.0,1.0,0.0,1.0);
				fragColor.w = 1.0;

				
				
			}
			
			
		}

		
		gl_ViewportIndex = gl_InvocationID;

		gl_PrimitiveID = gl_PrimitiveIDIn;

		EmitVertex();
	}
	
	EndPrimitive();
}

