
#pragma once

#ifndef _ISOSURFACE_H_
#define _ISOSURFACE_H_

#include "MarchingCubes_kernel.h"
#include <helper_cuda.h>

class Isosurface : public MarchingCubeCuda
{

    public:

    Isosurface();
    ~Isosurface();

    void computeIsosurface(float* vol, float4* pos , float4* norm, float isoValue,
        uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
        uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
        uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, float* vol_one,float* vol_two, 
        float isovalue1,float iso1, bool retain);

    void computeIsosurface_2(float4* pos , float4* norm, float isoValue,
        uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
        uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
        uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, float* vol_one, float isovalue1);


    void computeIsosurface_lattice(float* vol, float4* pos , float4* norm, float &isoValue,
        uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
        uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
        uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, float* vol_one,float* vol_two, 
        float isovalue1,float isovalue2,float iso1, float iso2);

    void computeIsosurface_latticeone(float* vol, float4* pos , float4* norm, float &isoValue,
        uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
        uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
        uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, float* vol_one,
        float isovalue1,float isovalue2);

    void patch_grid(float *d_vec1 , int Nx, int Ny, int Nz ,float isoval);



};


#endif //_ISOSURFACE_H_



