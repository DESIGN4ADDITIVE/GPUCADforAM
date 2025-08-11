#pragma once
#ifndef __MODELLING_H__
#define __MODELLING_H__

#include <cuda_runtime_api.h>




class Modelling
{
    int gridsize;

    public :

    static float a;

    static float b;
 
    Modelling(int Nx, int Ny, int Nz);

    ~Modelling();
    
    void concentric_cylinder(float* data_1,float3 center,float radius_1,float thickness_radial, float thickness_axial, int Nx,int Ny, int Nz);

    void distance_from_line(float* data_1,float3 center,float3 axis,float radius_1,float thickness_radial,float thickness_axial, int Nx,int Ny, int Nz, bool onetime);

    void sphere_with_center(float* data_1,float3 center,float radius_1,float thickness_wall,int Nx,int Ny, int Nz,bool onetime);

    void cuboid(float* data_1,float3 center,float3 angles,float x_width, float y_width, float z_width,int Nx,int Ny, int Nz);

    void cuboid_shell(float* data_1,float3 center,float3 angles, float x_width,float y_width,float z_width, float thickness, int Nx,int Ny, int Nz);

    void torus_with_center(float* data_1,float3 center,float3 angles,float torus_radius,float torus_circle_radius,int Nx,int Ny, int Nz);

    void cone_with_base_radius_height(float* data_1,float3 center,float3 angles,float base_radius,float cone_height,int Nx,int Ny, int Nz);

    void retain_boundary(float* data_1, float* data_2, float* data_3,int Nx,int Ny, int Nz);

    void GPU_buffer_normalise_four(float *d_vec1,float *d_vec2,float *d_vec3, size_t size, int Nx, int Ny, int Nz, float isoval_1);

    void GPU_buffer_normalise_dual(float *d_vec1,float *d_vec2,float *d_vec3, size_t size, int Nx, int Ny, int Nz, float isovalue);

    void init_final_boundary(float* data_1,int Nx,int Ny, int Nz);

};


#endif