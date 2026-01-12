
#pragma once


#ifndef __STRUCTURALSIM_H_
#define __STRUCTURALSIM_H_

#include "../general/DataTypes.h"
#include "../gpu_kernels/general_kernels.h"


class  Structuralsim 
{
    ///////////////////////////////////////////////////////////////////////////

    bool GPUInitialized = false;
    dim3 MatVecGrid;
    dim3 MatVecBlock;
    dim3 Scalar4Grid;
    dim3 Scalar4Block;

    REAL *d_ResReduction = NULL;

    public:
    
    int NX;
    int NY;
    int NZ;
    

    void InitGPU (REAL EleStiff [24][24]);

    void GPUMatVec (REAL3 *d_u, REAL *d_den, REAL *d_selection, REAL3 *d_res, size_t pitch_bytes, const REAL pexp);

    void GPURes (REAL3 *d_u,REAL *d_den, REAL *d_selection, REAL3 *d_res, size_t pitch_bytes, const REAL pexp);

    void GPUCG (REAL3 *d_u,REAL *d_den, REAL *d_selection, const int iter, const int OptIter, const REAL EndRes, int &FinalIter, REAL &FinalRes, const REAL pexp);

    void GPUCompGrad (REAL3 *d_u,REAL *d_den, REAL *d_grad, REAL &Obj, REAL &Vol, const size_t u_pitch_bytes, size_t grad_pitch_bytes, const REAL pexp);


    void GPUCleanUp ();

};


#endif