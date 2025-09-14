
#pragma once
#ifndef _MARCHING_CUBES_KERNEL_H_
#define _MARCHING_CUBES_KERNEL_H_

#include <stdint.h>
#include <cuda_runtime_api.h>

#define NTHREADS 32



struct grid_points
{
    int val = 0;
    float t_x = 0.0;
    float t_y = 0.0;
    float t_z = 0.0;
};

class MarchingCubeCuda
{
    
    public:

        void classify_copy_Voxel_lattice(dim3 grid, dim3 threads, uint *voxel_verts, float *volume_two, grid_points *vol_one,
                     uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
                     float3 voxelSize, float isoValue);

        void classifyVoxel_lattice(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied,float *volume_two,
                     uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
                     float3 voxelSize, float isoValue, grid_points *volume_one);

        void classifyVoxel_lattice_2(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, float *volume_one,
                     uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
                     float3 voxelSize, float isoValue);

        void classifyVoxel_lattice_3(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, float *volume_one,
                     uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
                     float3 voxelSize, float isoValue);

        void classifyVoxel_lattice_new(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, float *volume,
                     uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
                     float3 voxelSize, float isoValue);

        void compactVoxels_lattice(dim3 grid, dim3 threads, uint *compactedVoxelArray, 
                            uint *voxelOccupied,uint *voxelOccupiedScan, uint numVoxels);

        void generateTriangles_lattice(dim3 grid, dim3 threads,float4 *pos, float4 *norm, uint *compactVoxelArray,
                    uint *numVertsScanned,uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                    float3 voxelSize,float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts, uint totalverts_1,
                    grid_points *volume_one,float *volume_two,float isovalue1,float iso1, bool retain, uint *voxel_verts);

        
        void generateTriangles_lattice_2(dim3 grid, dim3 threads,float4 *pos, float4 *norm, 
                            uint *compactedVoxelArray, uint *numVertsScanned,
                            uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                            float3 voxelSize,float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts, uint totalverts,
                            float *volume_one,float isovalue1);

        
        void generateTriangles_lattice_3(dim3 grid, dim3 threads,float4 *pos, float4 *norm, 
                            uint *compactedVoxelArray, uint *numVertsScanned,
                            uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                            float3 voxelSize,float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts, uint totalverts,
                            float *volume_one,float isovalue1);

        void generateTriangles_lattice_new(dim3 grid, dim3 threads,
                          float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, float *volume,
                          uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                          float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts, uint totalverts,float *volume_one,float *volume_two,float isovalue1,float isovalue2, float iso1, float iso2);

        void generateTriangles_lattice_newone(dim3 grid, dim3 threads,
                          float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, float *volume,
                          uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                          float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts, uint totalverts,float *volume_one,float isovalue1,float isovalue2);


        void allocateTextures_s(uint **d_triTable,  uint **d_numVertsTable);

        void destroyAllTextureObjects();
        
        void ThrustScanWrapper_lattice(unsigned int *output, unsigned int *input, unsigned int numElements);
};
#endif // _MARCHING_CUBES_KERNEL_H_