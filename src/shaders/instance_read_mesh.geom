#version 450

#extension GL_ARB_viewport_array : enable
// #extension GL_EXT_debug_printf : enable

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


layout( push_constant, std430) uniform Inst_push_constants
{
	
    vec4 eyes;
	vec4 upaxis;
    vec4 force_dir;
	vec4 Scale_load;
	vec4 Scale_support;
	ivec2 mouse_delta;
} ;

layout (location = 0) in vec3 Pos[];
layout (location = 1) in vec3 Normal[];

layout (location = 2) in vec3 instPos[];
layout (location = 3) in vec3 instNormal[];
layout (location = 4) in vec3 instforce_dir[];
layout (location = 5) in float instVal[];
layout (location = 6) in int inst_loadindex[];

layout (location = 0) out vec4 fragColor;


void main(void)
{	
	

	for(int i = 0; i < 3 ; i++)
	{

       	vec3 n_normal = normalize(Normal[i].xyz*1.0);
	
		vec3 norm = instNormal[i].xyz;

		vec3 up_axis =  normalize(vec3(0,1,0));

		vec3 dir_force ;

		vec3 Pos_scale;

		vec3 lightcolor ;

		if(instVal[i] == 1.0)
		{
			lightcolor = vec3(1.0,0.550,0.0);
			
			dir_force = normalize(instforce_dir[i].xyz);

			Pos_scale = Pos[i].xyz * Scale_load.xyz;
		}
		if(instVal[i] == -1.0)
		{
			lightcolor = vec3(1.0,0.0,1.0);

			dir_force = norm;

			Pos_scale = Pos[i].xyz * Scale_support.xyz;
		}
		
		vec3 cross_vec = normalize(cross(up_axis,dir_force.xyz));

		vec3 locPos;
		vec3 new_norm;

		float angle = -1 * (acos((dir_force.x * up_axis.x )+ (dir_force.y * up_axis.y) + (dir_force.z * up_axis.z)));

		if(length(cross_vec) > 0)
		{

			locPos = Pos_scale.xyz * cos(angle) + (cross_vec * dot(cross_vec , Pos_scale.xyz) * (1 - cos(angle))) + (cross(Pos_scale.xyz, cross_vec ) * sin(angle));
			new_norm = n_normal.xyz * cos(angle) + (cross_vec * dot(cross_vec , n_normal.xyz) * (1 - cos(angle))) + (cross(n_normal.xyz, cross_vec ) * sin(angle));
			
		}
		else
		{
			
			locPos = Pos_scale.xyz;
			new_norm = n_normal;

			if(cos(angle) == -1)
			{
				locPos.y *= -1;
				new_norm.y *= -1;
			}
			
		}
		
		vec4 pos = vec4((locPos.xyz) + instPos[i].xyz, 1.0);

		vec3 lightvector = normalize(eyes.xyz);
	
		float amg = abs(dot(lightvector.xyz,new_norm.xyz));
		
		if (gl_InvocationID == 0)
			{
                
				vec4 proj_pos = ubo.modelViewProj[0]*pos;
				proj_pos.xy += mouse_delta * 0.002; 
				gl_Position = proj_pos ;
			

                fragColor = vec4(lightcolor*amg,1.0);

                gl_PointSize =float(2);

			}


        
		EmitVertex();
	}
	
	EndPrimitive();

}