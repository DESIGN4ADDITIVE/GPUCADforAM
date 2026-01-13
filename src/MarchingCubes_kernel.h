
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

        void classify_copy_Voxel_lattice(dim3 grid, dim3 threads, uint3 raster_grid, uint *voxel_verts,grid_points *vol_one, float *volume_two,float *vol_lattice,bool fixed, bool dynamic,float iso1, float iso2, 
                     uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
                     float3 voxelSize, float isoValue, bool obj_union, bool obj_diff, bool obj_intersect);

        void classifyVoxel_lattice(dim3 grid, dim3 threads,float *vol,uint3 raster_grid, uint *voxelVerts, uint *voxelOccupied,grid_points  *primitive_fixed,float *primitive_dynamic, float *topo_field,float *lattice_field, 
                     uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,float iso1, float iso2,
                     float3 voxelSize, float isoValue, bool obj_union, bool obj_diff, bool obj_intersect, bool primitive, bool topo, bool compute_lattice, bool fixed, bool dynamic, bool make_region);

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
                    grid_points  *primitive_fixed,float *primitive_dynamic, float *topo_field,float *lattice_field, float isovalue1,float iso1, uint *voxel_verts, bool obj_union, bool obj_diff, bool obj_intersect,
                     bool primitive, bool topo, bool compute_lattice, bool fixed, bool dynamic, bool make_region);

        
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