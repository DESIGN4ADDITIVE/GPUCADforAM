#pragma once

#ifndef __INTERPOLATIONS_H_
#define __INTERPOLATIONS_H_

#include "helper_cuda.h"

class Interpolations
{    
    protected:

    cudaTextureObject_t     texObj;

    cudaTextureObject_t     texObj3d;
    
    public:
    
    void setupTexture(int dx, int dy, int dz);

    void setup_3DTexture(int dx, int dy, int dz);

    void copytotexture(float *d_phi,cudaPitchedPtr data_ptr, int NX, int NY, int NZ);

    void copytotexture_results(float3 *d_displacement,cudaPitchedPtr data_ptr,int NX,int NY,int NZ, bool x_result, bool y_result, bool z_result);

    void copytotexture_3d_results(float3 *d_displacement,cudaPitchedPtr data_ptr,int NX,int NY,int NZ);

    void updateTexture(cudaPitchedPtr data_ptr);

    void update_3dTexture(cudaPitchedPtr data_ptr);

    void deleteTexture();

    void delete_3dTexture();

    void refine_3d(float4 *disp,int NX2, int NY2, int NZ2,float dx, float dy, float dz,
    float *d_result, bool x_val, bool y_val, bool z_val);

    void grid_displacement(float3 *d_grid_pos, float3 *grid_disp, int Nx, int Ny, int Nz, bool disp_grid);

    void limit_displacement(float3 *d_xyz, float3 *disp, int Nx, int Ny, int Nz, bool disp_grid, uint magnify);
};


#endif