#pragma once

#ifndef __OPTIMISATION_KERNELS_H__
#define __OPTIMISATION_KERNELS_H__

#include "../general/DataTypes.h"

#include <helper_cuda.h>

class Optimisation_kernels {

public :

int NX;
int NY;
int NZ;

void init_d_den(REAL *d_u, REAL volfrac, int size);

void GPUMeshFilter(REAL3 *d_u,REAL *d_den, REAL rmin, REAL *d_grad,const int pitchX, int NX,int NY,int NZ);

void Update_den_GPU(REAL3 *d_u,REAL *d_den, REAL VolFrac, REAL *d_grad, float *d_volume,int NX, int NY, int NZ,const REAL lmid,const REAL move,const REAL MinDens,REAL *d_new_den, REAL *d_new_den_result,int block_num);

void copy_den(REAL *d_den, REAL *d_new_den, int NX, int NY, int NZ);

void Update_s_one(REAL3 *d_u,REAL *d_den, REAL VolFrac,REAL MinDens, REAL *d_grad, float *d_volume,int pitchX, int NX, int NY, int NZ);


};

#endif