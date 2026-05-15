#version 450



layout (location = 0) in vec4 fragColor;

layout (input_attachment_index = 0, binding = 2) uniform subpassInput inputColor;
layout (input_attachment_index = 1, binding = 3) uniform subpassInput inputDepth;

layout( push_constant, std430) uniform Inst_push_constants
{
	
    vec4 eyes;
	vec4 upaxis;
    vec4 force_dir;
	vec4 Scale_load;
	vec4 Scale_support;
} ;

layout (location = 0) out vec4 outFragColor;

void main() 
{

    float ch_depth = subpassLoad(inputDepth).r;
    
    if((gl_FragCoord.z == ch_depth))
    {
        outFragColor = fragColor;
    }
    else
    {
        discard;
    }
}