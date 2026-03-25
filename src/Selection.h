#pragma once
#ifndef __SELECTION_H__
#define __SELECTION_H__

#include "../src/general/DataTypes.h"
#include "Isosurface.h"

class Selection
{
    public:

    void vertex_selection(float* d_storagebuffer, float* d_volume, int Nx, int Ny, int Nz);
    
    void vertex_selection_two(float* d_storagebuffer, float* d_volume, int Nx, int Ny, int Nz, bool load_selection, 
    bool boundary_selection , bool delete_selection);

    void facet_selection(float* d_storagebuffer_1, float* d_storagebuffer_2, int nfacets, bool load_selection,
    bool boundary_selection, bool delete_selection);

    void update_load_condition(REAL3* d_us,float* d_cudastoragebuffer,int Nx, int Ny, int Nz, bool x_axis, bool y_axis, bool z_axis );

    void raster_update(float isoval_fixed, float iso_dynamic, float iso1, float iso2, float *raster, float *d_solid, grid_points *vol_one, float *boundary, float *lattice_field, float *fix_lat_field,  bool fixed, bool dynamic,int Nx,int Ny, int Nz ,
    bool obj_union, bool obj_difference, bool obj_intersect);

    void raster_make_region(float isoval_fixed, float iso_dynamic, float iso1, float iso2, float *raster, float *d_solid, grid_points *vol_one, float *boundary, float *lattice_field, float *fix_lat_field,  bool fixed, bool dynamic,int Nx,int Ny, int Nz);

    void constrained_vol(REAL *raster, grid_points *vol_topo, uint *solid_voxels, float volfrac, int Nx, int Ny, int Nz);

    void fixed_free(int *fixed_free, REAL *raster, int Nx, int Ny, int Nz);

    void facet_to_points(float *storagebuffer, triangle_metadata *triangle_data, uint active_facets, float3 *d_u, bool update_load, bool update_support, bool clear_load, bool clear_support,
    uint3 gridSizeShift, uint3 gridSizeMask, uint Nx, uint Ny, grid_points *d_vol_one, float *d_selection, float isoval);

    void apply_to_lower(REAL *d_selection, REAL *d_selection2, grid_points *d_vol_one, int Nx, int Ny, int Nz, uint3 gridSizeMask, uint3 gridSizeShift, float isoval);

    void raster_region_update(float isoval_fixed_region,  float *raster, grid_points *vol_topo, grid_points *vol_one, int Nx,int Ny, int Nz, bool show_domain);

};


#endif