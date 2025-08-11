#pragma once

#ifndef __GENERAL_KERNELS_H__
#define __GENERAL_KERNELS_H__


#include "../general/DataTypes.h"


#define INDEX(i,j,j_off)  (i +__mul24(j,j_off))

#define INDEX3D(i,j,k, j_off, i_off) (i + __mul24((j), (j_off)) + __mul24((k), (__mul24((i_off), (j_off)))))

#define IOFF 1

#define p2 512


class  General_Kernels
{
	public :
	
	int NX;
    int NY;
    int NZ;

	__device__ static inline int GPUGetLocalID(const int i, const int j, const int k)
	{
		return i+2*j+4*k;
	}

	static void GPUScalar(REAL* d_result, REAL3 *d_vec1, REAL3 *d_vec2, const size_t pitch_bytes, int NX, int NY, int NZ);

    static void GPUSum(REAL *d_result, REAL *d_vec, const size_t pitch_bytes, int NX, int NY, int NZ);

	static void GPUVolume(REAL *d_result, REAL *d_den, const size_t pitch_bytes, int NX, int NY, int NZ);

	static void VecSMultAdd(REAL3 *d_v, REAL a1, REAL3 *d_w, const REAL a2, const size_t pitch_bytes, const int NX, const int NY, const int NZ);

	static void GPUScalarone(REAL* d_result, REAL3 *d_vec1, REAL3 *d_vec2, const size_t pitch_bytes, int NX, int NY, int NZ);

};


#endif