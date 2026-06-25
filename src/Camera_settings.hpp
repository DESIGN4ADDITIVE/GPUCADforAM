#pragma once 

#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "linmath.h"
#include "helper_math.h"


class Camera_settings
{
    public:
    
    static inline void set_ZYX_pos(mat4x4 viewMatrix, float3 position, float3 rotation)
    {

        const float c3 = cos(rotation.x);
        const float s3 = sin(rotation.x);
        const float c2 = cos(rotation.y);
        const float s2 = sin(rotation.y);
        const float c1 = cos(rotation.z);
        const float s1 = sin(rotation.z);


        const float3 u{(c1 * c2),(c1 * s2 * s3) - (c3 * s1),(s1 * s3) + (c1 * c3 * s2)};
        const float3 v{(c2 * s1),(c1 * c3) + (s1 * s2 *s3),(c3 * s1 * s2) - (c1 * s3)};
        const float3 w{(-s2),(c2 * s3),(c2 * c3)};

        mat4x4_identity(viewMatrix);
        viewMatrix[0][0] = u.x;
        viewMatrix[1][0] = u.y;
        viewMatrix[2][0] = u.z;
        viewMatrix[0][1] = v.x;
        viewMatrix[1][1] = v.y;
        viewMatrix[2][1] = v.z;
        viewMatrix[0][2] = w.x;
        viewMatrix[1][2] = w.y;
        viewMatrix[2][2] = w.z;
        viewMatrix[3][0] = -1.0f * dot(u, position);
        viewMatrix[3][1] = -1.0f * dot(v, position);
        viewMatrix[3][2] = -1.0f * dot(w, position);

    }


    static inline void rotate_plane(float angle,float view_type,float3 *camera_rot)
    {

        if((view_type == 1) || (view_type == 2) )
        {
            camera_rot->x = 0.0f;
            camera_rot->y = angle;
            camera_rot->z = 0.0f;

        }
        
        else if((view_type == 3) || (view_type == 4))
        {
            
            camera_rot->x = angle;
            camera_rot->y = 0.0f;
            camera_rot->z = 0.0f;
        }

        else if((view_type == 5) || (view_type == 6))
        {
            camera_rot->x = 0.0f;
            camera_rot->y = 0.0f;
            camera_rot->z = angle;
        }
        
    }

    static inline void camera_view(float3 *camera_rot, uint view_type)
    {
        switch (view_type)
        {
        case 1:
            *camera_rot = {0.0f,0.0f,0.0f};
            break;
        case 2:
            *camera_rot = {0.0f,3.14f,0.0f};
            break;

        case 3:
            *camera_rot = {1.57f,0.0f,0.0f};
            break;
        case 4:
            *camera_rot = {4.712f,0.0f,0.0f};
            break;

        case 5:
            *camera_rot = {0.0f,4.712f,0.0f};
            break;
        case 6:
            *camera_rot = {0.0f,1.57f,0.0f};
            break;
        
        }
    }

};

#endif