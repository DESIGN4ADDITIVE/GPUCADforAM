
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


    void copy_parameter( uint *voxel_verts, float isoValue,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize,uint numVoxels,
    grid_points *vol_one,float* vol_two,float *vol_lattice,bool fixed, bool dynamic,float iso1, float iso2, bool obj_union, bool obj_diff, bool obj_intersect);

    void copy_regions( uint *voxel_verts, float isoValue,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize,uint numVoxels,grid_points *vol_topo,
    grid_points *vol_one,float* vol_two,float *vol_lattice,bool fixed, bool dynamic,float iso1, float iso2, bool obj_union, bool obj_diff, bool obj_intersect );

    
    void computeIsosurface(float *vol,uint3 raster_grid, float4* pos , float4* norm, float isoValue,
    uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
    uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, grid_points  *primitive_fixed,float *primitive_dynamic, float *topo_field,float *lattice_field, 
    float iso1,float iso2, bool obj_union, bool obj_diff, bool obj_intersect , bool primitive, bool topo, bool compute_lattice, bool fixed, bool dynamic,
    bool make_region, size_t *nfacets);

    void compute_solid_voxels(float isoValue,uint numVoxels, uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, grid_points  *primitive_fixed,float *d_solid_field);


    void computeIsosurface_region( float4* pos , float4* norm, float isoValue,
    uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
    uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts,grid_points *vol_topo, grid_points  *primitive_fixed,float *primitive_dynamic, float *topo_field,float *lattice_field, 
    float iso1,float iso2, bool obj_union, bool obj_diff, bool obj_intersect , bool primitive, bool topo, bool compute_lattice, bool fixed, bool dynamic,
    bool make_region , bool show_region, bool show_domain, triangle_metadata *triangle_data);


    void computeIsosurface_2(float4* pos , float4* norm, float isoValue,
        uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
        uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
        uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, grid_points *vol_topo, 
        grid_points *vol_one,float* vol_two, float *d_solid, float isovalue1, float *d_result, triangle_metadata *triangle_data);

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

    void patch_topo_field(float *d_vec1 , int Nx, int Ny, int Nz, float isoval ,float *solid_field,float minDens, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask);


};


#endif //_ISOSURFACE_H_



