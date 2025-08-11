
#pragma once

#ifndef __THERMALSIM_H_
#define __THERMALSIM_H_

#include "../general/DataTypes.h"
#include "../gpu_kernels/general_kernels.h"

class Thermalsim
{

    bool GPUInitialized = false;
    REAL *d_ResReduction_t = NULL;

    public:

    int NX;
    int NY;
    int NZ;

    
    void InitGPU(REAL EleStiffone[8][8]);

    void GPUMatVec(REAL3 *d_u, REAL *d_den, REAL *d_selection, REAL3 *d_res, size_t pitch_bytes, REAL pexp);

    void GPURes(REAL3 *d_u, REAL *d_den, REAL *d_selection, REAL3 *d_res, size_t pitch_bytes, REAL pexp);

    void GPUCG(REAL3 *d_u, REAL *d_den, REAL *d_selection,const int iter, const int OptIter, const REAL EndRes, int &FinalIter, REAL &FinalRes, REAL pexp);

    void GPUCompGrad(REAL3 *d_u, REAL *d_den, REAL *d_grad, REAL &Obj, REAL &Vol, const size_t u_pitch_bytes, size_t &grad_pitch_bytes, REAL pexp);

    void GPUCleanUp();

};


#endif