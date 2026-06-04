#version 450


// #extension GL_EXT_debug_printf : enable

// Vertex attributes
layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;

// Instanced attributes
layout (location = 2) in vec3 instancePos;
layout (location = 3) in vec3 instanceRot;
layout (location = 4) in float instanceval;


layout (location = 0) out vec3 Pos;
layout (location = 1) out vec3 Normal;

layout (location = 2) out vec3 instPos;
layout (location = 3) out vec3 instNormal;
layout (location = 4) out float instVal;

void main() 
{

    Pos = inPos;
    Normal = inNormal;

    instPos = instancePos;
    instNormal = instanceRot;
    instVal = instanceval;

}
