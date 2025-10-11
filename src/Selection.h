#pragma once
#ifndef __SELECTION_H__
#define __SELECTION_H__

#include "../src/general/DataTypes.h"

class Selection
{
    public:

    void vertex_selection(float* d_storagebuffer, float* d_volume, int Nx, int Ny, int Nz);
    
    void vertex_selection_two(float* d_storagebuffer, float* d_volume, int Nx, int Ny, int Nz, bool load_selection, 
    bool boundary_selection , bool delete_selection);

    void update_load_condition(REAL3* d_us,float* d_cudastoragebuffer,int Nx, int Ny, int Nz, bool x_axis, bool y_axis, bool z_axis );

};


#endif