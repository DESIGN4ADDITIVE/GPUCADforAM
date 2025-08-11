
#include "../general/DataTypes.h"
#include "general_kernels.h"



__global__ void Sum1Kernel(REAL *d_result, REAL *a, const int pitchX, const int NX, const int NY, const int NZ)
{
///////////////////////////////////////////////////////////////////////////////////////
	__shared__ REAL Accum[1024];
	REAL sum = 0.0;
	int i = threadIdx.x;
	int j = threadIdx.y;
	const int ind = INDEX(i,j,blockDim.x);
	
	i = INDEX(i, blockIdx.x, blockDim.x);
	j = INDEX(j, blockIdx.y, blockDim.y);
	const int active = (i >= 0 && i<NX-1 && j >=0 && j<NY-1);
	int indg = INDEX(i,j,pitchX);
	for(int k=0;k<NZ-1;k++)
	{
		REAL Result;
		
		if(active)
		{
			Result = a[indg];
		}
		else
		{
			Result = 0.0;
		}
	
	
		Accum[ind] = Result;
		__syncthreads();

		for(int stride = 1024/2; stride>0;stride/=2)
		{
			if(ind < stride)
			{
				
				Result = Accum[ind];
				Result += Accum[ind+stride];
				Accum[ind] = Result;
			}
			__syncthreads();
		}
		sum += Accum[0];
		
		indg = INDEX(indg, NY, pitchX);
	}
	
	if(ind == 0)
	{
	
		d_result[blockIdx.x+blockIdx.y*gridDim.x] = sum;
		
	}

}



__global__ void SumVolumeKernel(REAL *d_result, REAL *d_den, const int pitchX, const int NX, const int NY, const int NZ)
{
	__shared__ REAL Accum[1024];
	REAL sum = 0.0;
	int i = threadIdx.x;
	int j = threadIdx.y;
	const int ind = INDEX(i,j,blockDim.x);
	
	i = INDEX(i, blockIdx.x, blockDim.x);
	j = INDEX(j, blockIdx.y, blockDim.y);
	const int active = (i >= 0 && i<NX-1 && j >=0 && j<NY-1);
	int indg = INDEX(i,j,pitchX);
	
	//loop through z direction
    for(int k=0;k<NZ-1;k++)
    {
		
		
		REAL MyA;
		if(active)
        {
			MyA = d_den[indg];
		}
        else
        {
			MyA = 0.0;
		}
		Accum[ind] = MyA;
		__syncthreads();
     
		for(int stride = 1024/2; stride>0; stride/=2)
        {
			if(ind < stride)
            {
				REAL Result = Accum[ind];
                Result += Accum[ind+stride];
                Accum[ind] = Result;
			}
            __syncthreads();
		}
		sum += Accum[0];
		indg = INDEX(indg, NY, pitchX);
	}
	if(ind == 0)
    {
		d_result[blockIdx.x+blockIdx.y*gridDim.x] = sum;
	}
}



__global__ void MyReduction(REAL *d_DataIn, REAL *d_DataOut)
{
	__shared__ REAL sdata[p2];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = d_DataIn[i];
	__syncthreads();

	for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
	{
		if (tid < s) 
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) d_DataOut[blockIdx.x] = sdata[0];
}




__global__ void MyVecSMultAddKernel(REAL3 *v, const REAL a1, REAL3 *w, const REAL a2, const int pitchX, const int NX, const int NY, const int NZ)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	i = INDEX(i, blockIdx.x, blockDim.x);
	j = INDEX(j, blockIdx.y, blockDim.y);
	const int active = (i >= 0 && i<NX && j >=0 && j<NY);
	int indg = INDEX(i,j,pitchX);
	if(active)
	{
		for(int k=0;k<NZ;k++)
		{
			REAL3 MyV = v[indg];
			REAL3 MyW = w[indg];
			REAL3 result;

			result.x = a1*MyV.x + a2*MyW.x;
			result.y = a1*MyV.y + a2*MyW.y;
			result.z = a1*MyV.z + a2*MyW.z;

			v[indg] = result;
			indg = INDEX(indg, NY, pitchX);
		}
	}
}






__global__ void ScalarProd4Kernel(REAL *d_result, REAL3 *a, REAL3 *b, const int pitchX, const int NX, const int NY, const int NZ)
{
	__shared__ REAL3 Accum[1024];
	REAL sum = 0.0;
	int i = threadIdx.x;
	int j = threadIdx.y;
	const int ind = INDEX(i,j,blockDim.x);
	
	i = INDEX(i, blockIdx.x, blockDim.x);
	j = INDEX(j, blockIdx.y, blockDim.y);
	const int active = (i >= 0 && i<NX && j >=0 && j<NY);
	int indg = INDEX(i,j,pitchX);
	for(int k=0;k<NZ;k++)
	{
		REAL3 MyA;
		REAL3 MyB;
		if(active)
		{
			MyA = a[indg];
			MyB = b[indg];
		}
		else
		{
			MyA.x = 0.0;
			MyA.y = 0.0;
			MyA.z = 0.0;
			MyB.x = 0.0;
			MyB.y = 0.0;
			MyB.z = 0.0;
		}
		REAL3 Result;
		Result.x = MyA.x*MyB.x;
		Result.y = MyA.y*MyB.y;
		Result.z = MyA.z*MyB.z;
		Accum[ind] = Result;
		__syncthreads();

		for(int stride = 1024/2; stride>0;stride/=2)
		{
			if(ind < stride)
			{
				
				Result = Accum[ind];
				const REAL3 Result2 = Accum[ind+stride];
				Result.x += Result2.x;
				Result.y += Result2.y;
				Result.z += Result2.z;
				Accum[ind] = Result;
			}
			__syncthreads();
		}
		sum += Accum[0].x + Accum[0].y + Accum[0].z;
		
		indg = INDEX(indg, NY, pitchX);
	}
	
	if(ind == 0)
	{
		d_result[blockIdx.x+blockIdx.y*gridDim.x] = sum;
		
	}
}


void General_Kernels::GPUScalar(REAL* d_result, REAL3 *d_vec1, REAL3 *d_vec2, const size_t pitch_bytes, int NX, int NY, int NZ)
{

	const int pitch = pitch_bytes/sizeof(REAL3);

	dim3 threads(32,32,1);
        
    dim3 grids(ceil((NX)/ float(threads.x)), ceil((NY)/ float(threads.y)), 1);
	ScalarProd4Kernel<<<grids, threads>>>(d_result, d_vec1, d_vec2, pitch,  NX,  NY,  NZ);
	cudaDeviceSynchronize();
	MyReduction<<<1, p2>>>(d_result, d_result);
	cudaDeviceSynchronize();

}





void General_Kernels::GPUSum(REAL *d_result, REAL *d_vec, const size_t pitch_bytes, int NX, int NY, int NZ)
{


	dim3 threads(32,32,1);
    dim3 grids(ceil((NX)/ float(threads.x)), ceil((NY)/ float(threads.y)), 1);


	const int pitch = pitch_bytes/sizeof(REAL);
	cudaMemset(d_result, (REAL)0.0, p2*sizeof(REAL));
	cudaDeviceSynchronize();
	Sum1Kernel<<<grids, threads>>>(d_result, d_vec, pitch,  NX,  NY,  NZ);
	cudaDeviceSynchronize();
	MyReduction<<<1, p2>>>(d_result, d_result);
	cudaDeviceSynchronize();
}


void General_Kernels::GPUVolume(REAL *d_result, REAL *d_den, const size_t pitch_bytes, int NX, int NY, int NZ)
{
	dim3 threads(32,32,1);
    dim3 grids(ceil((NX)/ float(threads.x)), ceil((NY)/ float(threads.y)), 1);

	const int pitch = pitch_bytes/sizeof(REAL3);
	cudaMemset(d_result, (REAL)0.0, p2*sizeof(REAL));
	cudaDeviceSynchronize();
	SumVolumeKernel<<<grids, threads>>>(d_result, d_den, pitch,  NX,  NY,  NZ);
	cudaDeviceSynchronize();
	MyReduction<<<1, p2>>>(d_result, d_result);
	cudaDeviceSynchronize();
}

void General_Kernels::VecSMultAdd(REAL3 *d_v, REAL a1, REAL3 *d_w, const REAL a2, const size_t pitch_bytes, const int NX, const int NY, const int NZ)
{
	dim3 threads(32,32,1);        
    dim3 grids(ceil((NX)/ float(threads.x)), ceil((NY)/ float(threads.y)), 1);
	const int pitch = pitch_bytes/sizeof(REAL3);
	MyVecSMultAddKernel<<<grids,threads>>>(d_v, a1, d_w, a2, pitch,  NX,  NY,  NZ);
	cudaDeviceSynchronize();

}


__global__ void ScalarProd4Kernelone_t(REAL *d_result, REAL3 *a, REAL3 *b, const int pitchX, const int NX, const int NY, const int NZ)
{
	__shared__ REAL3 Accum[1024];
	REAL sum = 0.0;
	int i = threadIdx.x;
	int j = threadIdx.y;
	const int ind = INDEX(i,j,32);
	
	i = INDEX(i, blockIdx.x, 32);
	j = INDEX(j, blockIdx.y, 32);
	const int active = (i >= 0 && i<NX && j >=0 && j<NY);
	int indg = INDEX(i,j,pitchX);
	for(int k=0;k<NZ;k++)
	{
		REAL3 MyA;
		REAL3 MyB;
		if(active)
		{
			MyA = a[indg];
			MyB = b[indg];
		}
		else
		{
			MyA.x = 0.0;
			MyA.y = 0.0;
			MyA.z = 0.0;
			MyB.x = 0.0;
			MyB.y = 0.0;
			MyB.z = 0.0;
		}
		REAL3 Result;
		Result.x = MyA.x*MyB.x;
	
	
		Accum[ind] = Result;
		__syncthreads();
		
		for(int stride = (32*32)/2; stride>0;stride/=2)
		{
			if(ind < stride)
			{
				
				Result = Accum[ind];
				const REAL3 Result2 = Accum[ind+stride];
				Result.x += Result2.x;
			
				Accum[ind] = Result;
			}
			__syncthreads();
		}
		sum += Accum[0].x ;
		indg = INDEX(indg, NY, pitchX);
	}
	
	if(ind == 0)
	{
		
		d_result[blockIdx.x+blockIdx.y*gridDim.x] = sum;
		
	}
}


void General_Kernels::GPUScalarone(REAL* d_result, REAL3 *d_vec1, REAL3 *d_vec2, const size_t pitch_bytes, int NX, int NY, int NZ)
{
	dim3 threads(32,32,1);
        
    dim3 grids(ceil((NX)/ float(threads.x)), ceil((NY)/ float(threads.y)), 1);
	const int pitch = pitch_bytes/sizeof(REAL3);
	ScalarProd4Kernelone_t<<<grids, threads>>>(d_result, d_vec1, d_vec2, pitch, NX, NY, NZ);
	cudaDeviceSynchronize();
	MyReduction<<<1, p2>>>(d_result, d_result);
	cudaDeviceSynchronize();

}