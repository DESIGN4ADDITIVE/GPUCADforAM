
#include "Selection.h"

#include <iostream>




__global__ void vertex_selection_kernel(float2* d_result, float* d_storagebuffer, int Nx, int Ny, int Nz)
{
    int tx = threadIdx.x;
	int ind = blockIdx.x*blockDim.x+tx;
    int n = Nx*Ny*Nz;
	__shared__ float cc[1024];
	__shared__ float dd[1024];
	
	float ch ;
	float id ;
	cc[tx] = 1000;

	if (ind < n)
	{
		cc[tx] = d_storagebuffer[ind];
		dd[tx] = ind * 1.0;
	}

	__syncthreads();

	for(int stride = blockDim.x/2; stride>0; stride/=2)
	{
		
		if(tx < stride)
		{
			
			ch = min(cc[tx],cc[tx+stride]);
			if((ch == cc[tx]))
			{
				id = dd[tx] ;

			}
			else
			{
			 	id = dd[tx + stride];
			}
		

			cc[tx] = ch;

			dd[tx] = id;	

		}
		__syncthreads();
	}
	
	if (tx == 0)
	{
		d_result[blockIdx.x].x = cc[tx];
		d_result[blockIdx.x].y = dd[tx];
		 
	}

	d_storagebuffer[ind] = 1000.0;

	__syncthreads();

	
}


__global__ void Selection_Min_reduction(float2 *d_DataIn,float* d_volume , int block_num)
{
	__shared__ float2 sdata_min[1024];

	unsigned int tid = threadIdx.x;
	sdata_min[tid].x = 1000;
	sdata_min[tid].y = 1000;
	int index;
	int e;
	e = (block_num/1024) + (!(block_num%1024)?0:1);
	
	float c ;
	float d ;
	float ff;


	int f_ind = 0;

	for (int k = 0; k< e;k++)
	{
		index = tid + k*1024;
		if(index < block_num)
		{
			c = sdata_min[tid].x;
			d = sdata_min[tid].y;		

			sdata_min[tid] = d_DataIn[index];

			
			ff = min(c,sdata_min[tid].x);
			if(c == ff)
			{
				sdata_min[tid].x = c;
				sdata_min[tid].y = d;
				
			}
			
		}


	
	}

	
	__syncthreads();

	

	for(unsigned int s=blockDim.x/2; s>0;s/=2) 
	{
		
		
		if (tid < s) 
		{
			c = sdata_min[tid].x;
			d = sdata_min[tid].y;
			sdata_min[tid].x = min(sdata_min[tid].x,sdata_min[tid + s].x);
			if(c == sdata_min[tid].x)
			{
				sdata_min[tid].y = d;
			}
			else
			{
				sdata_min[tid].y = sdata_min[tid + s].y;
			}
			
		}
		__syncthreads();
	}

	
	if (tid == 0) 
	{
		d_DataIn[0] = sdata_min[0];

		f_ind = (int)sdata_min[0].y;
		
		if (!(f_ind == 1000))
		{
			d_volume[f_ind] = 25.0;
		}
		
	}
	
}



void Selection::vertex_selection(float* d_storagebuffer, float* d_volume, int Nx, int Ny, int Nz)
{
    dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	float2 gb;
    int block_num = grids.x;

	float2 *d_Reduction_min;

	cudaMalloc((void **)&d_Reduction_min, sizeof(float2)* (block_num));

	cudaMemset(d_Reduction_min, 0.0, sizeof(float2)* (block_num));

    vertex_selection_kernel<<<grids,tids>>>(d_Reduction_min,d_storagebuffer,Nx,Ny,Nz);

	unsigned int  x_grid = 1;
	unsigned int  x_thread = 1024;

	Selection_Min_reduction<<<x_grid, x_thread>>>(d_Reduction_min,d_volume,block_num);

	cudaDeviceSynchronize();

	cudaMemcpy(&gb, d_Reduction_min, sizeof(float2), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cudaFree(d_Reduction_min);
}


__global__ void vertex_selection_two_kernel(float* d_storagebuffer_1, float* d_storagebuffer_2, int Nx, int Ny, int Nz)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int size = Nx*Ny*Nz;
	float val1 = 0;
	float val2 = 0;
	if(idx < size)
	{
		val1 = d_storagebuffer_1[idx];
		val2 = d_storagebuffer_2[idx];
		
		
		if ((val1 == -1.0) && (val2 != -1.0))
		{
		
			d_storagebuffer_2[idx] = -1.0;
		}

		else if ((val1 == 1.0) && (val2 != 1.0))
		{
		
			d_storagebuffer_2[idx] = 1.0;
		}
	
	}
}

void Selection::vertex_selection_two(float* d_storagebuffer_1, float* d_storagebuffer_2, int Nx, int Ny, int Nz)
{
    dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	
	vertex_selection_two_kernel<<<grids,tids>>>(d_storagebuffer_1,d_storagebuffer_2,Nx,Ny,Nz);

	cudaDeviceSynchronize();
}

__global__ void update_load_kernel(REAL3* d_us,float* d_cudastoragebuffer,int Nx, int Ny, int Nz, bool x_axis,bool y_axis,bool z_axis)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int size = Nx*Ny*Nz;
	REAL3 a;
	float b;
	if(idx < size)
	{
		b = d_cudastoragebuffer[idx];

		if(b == 1.0)
		{
			a = d_us[idx];
			if(x_axis)
			{
				a.x = -1.0;
			}
			if(y_axis)
			{
				a.y = -1.0;
			}
			if(z_axis)
			{
				a.z = -1.0;
			}
			d_us[idx] = a;
		
		}
	}
}

void Selection::update_load_condition(REAL3* d_us,float* d_cudastoragebuffer,int Nx, int Ny, int Nz, bool x_axis, bool y_axis, bool z_axis )
{
	dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	update_load_kernel<<<grids,tids>>>(d_us, d_cudastoragebuffer, Nx, Ny, Nz, x_axis, y_axis, z_axis);
	cudaDeviceSynchronize();
}