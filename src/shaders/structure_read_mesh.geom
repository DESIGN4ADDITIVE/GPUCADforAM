

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

} ;

layout(location = 0) in vec4 posit[];
layout(location = 1) in vec4 norm[];

layout (location = 1) out vec3 fragColor;


/////////////////////////////////////////////////////////////////////////////
float colormap_red(float x) {
    if (x < 100.0) {
        return (-9.55123422981038E-02 * x + 5.86981763554179E+00) * x - 3.13964093701986E+00;
    } else {
        return 5.25591836734694E+00 * x - 8.32322857142857E+02;
    }
}

float colormap_green(float x) {
    if (x < 150.0) {
        return 5.24448979591837E+00 * x - 3.20842448979592E+02;
    } else {
        return -5.25673469387755E+00 * x + 1.34195877551020E+03;
    }
}

float colormap_blue(float x) {
    if (x < 80.0) {
        return 4.59774436090226E+00 * x - 2.26315789473684E+00;
    } else {
        return -5.25112244897959E+00 * x + 8.30385102040816E+02;
    }
}

vec4 colormap(float x) {
    float t = x * 255.0;
    float r = clamp(colormap_red(t) / 255.0, 0.0, 1.0);
    float g = clamp(colormap_green(t) / 255.0, 0.0, 1.0);
    float b = clamp(colormap_blue(t) / 255.0, 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}



void main(void)
{	

	for(int i = 0; i < 3 ; i++)
	{

	
        vec3 n_normal = normalize(norm[i].xyz*1.0);
	
		vec3 lightvector = normalize(eyes.xyz - posit[i].xyz);
		
		vec3 lightcolor = vec3(1.0,1.0,1.0);
	
		float amg = abs(dot(lightvector.xyz,n_normal.xyz));
		
		if (gl_InvocationID == 0)
			{
                gl_Position =ubo.modelViewProj[0]*posit[i];
                
                fragColor = lightcolor*amg;
                gl_PointSize =float(2);

			}

		gl_ViewportIndex =gl_InvocationID;

		gl_PrimitiveID = gl_PrimitiveIDIn;
        
		EmitVertex();
	}
	
	EndPrimitive();

}