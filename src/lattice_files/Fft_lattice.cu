
#include "Fft_lattice.h"

#include <cufft.h>
#include <helper_cuda.h>
#include <helper_math.h>

extern cufftHandle planr2c;
extern cufftHandle planc2r;
extern cufftHandle planc2c;

__global__ void create_lattice_kernel(float *d_latticevol,uint NX, uint NY, uint NZ, uint size, uint lattice_index_type)
{


	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	uint index = x + (y * NX) + (z * (NX * NY));
	float aa;



	if((x < NX) && (y < NY ) && ( z < NZ))
	{
		
		float xx = (((x * 1.0)/(NX - 1)) - 0.5) / 0.5;
		float yy = (((y * 1.0)/(NY - 1)) - 0.5) / 0.5;
		float zz = (((z * 1.0)/(NZ - 1)) - 0.5) / 0.5;

		if(lattice_index_type == 0)
		{
			aa = cosf(3.28 * xx)*sinf(3.28 * yy) + cosf(3.28 * yy) * sinf(3.28 * zz) + cosf(3.28 * zz) * sinf(3.28 * xx);

		}
		else if(lattice_index_type == 1)
		{
			aa = cosf(3.28  * xx) + cosf(3.28  * yy) + cosf(3.28  * zz) ;
		}
		else if (lattice_index_type == 2)
		{
			aa = 4 * (cosf(xx) * cosf(yy) * cosf(zz)) - (cosf(2*xx)*cosf(2*yy) + cosf(2*yy)*cosf(2*zz) + cosf(2*zz)*cosf(2*xx));
		}
		else if (lattice_index_type == 3)
		{
			aa = 2 * (cosf(3.14 * xx)*cosf(3.14 * yy) + cosf(3.14 * yy)*cosf(3.14 * zz) + cosf(3.14 * zz)*cosf(3.14 * xx)) - (cosf( 2 * 3.14 * xx) + cosf(2 * 3.14 * yy) + cosf(2 * 3.14 * zz));
		}
		else if (lattice_index_type == 4)
		{
			aa = min(min((powf(xx,2) + pow(yy,2)), (pow(yy,2) + pow(zz,2))),(pow(zz,2) + pow(xx,2)));
		}

		else if (lattice_index_type == 5)
		{
			aa = cos(3.14 * xx)*cosf(3.14 * yy) *cosf(3.14 * zz) -  sinf(3.14 *xx) * sinf(3.14 * yy) * sinf(3.14 * zz);
			
		}

		d_latticevol[index] = aa;
		
		__syncthreads();

	}

};

void Fft_lattice::create_lattice(float *d_latticevol, uint NX, uint NY, uint NZ, uint size, uint lattice_index_type)
{
	dim3 grids(ceil((NX)/float(16)),ceil((NY)/float(8)),ceil((NZ)/float(8)));
	dim3 tids(16,8,8);
	create_lattice_kernel<<<grids,tids>>>(d_latticevol,NX,NY,NZ,size,lattice_index_type);
	cudaDeviceSynchronize();
}


void normalise_bufferr(float *dataone, float *datatwo, size_t size)
{
	float *h_B;
	h_B = (float *)malloc((size) * sizeof(*dataone));
	cudaMemcpy(h_B, dataone, (size) * sizeof(*dataone), cudaMemcpyHostToHost);
	float a,b;
	for (int i=0;i<size;i++)
	{
		if(i==0)
		{
			a = h_B[i];
			b = h_B[i];
		}
		
		a=min(a,h_B[i]);
		b =max(b,h_B[i]);
	}

	for(int i=0;i<size;i++)
	{
		h_B[i] = (h_B[i] - a)/(b-a);
	}

	cudaMemcpy(datatwo, h_B, (size) * sizeof(*h_B), cudaMemcpyHostToHost);
	free(h_B);
}




void Fft_lattice::fft_func(float2 *fft_data)
{

		checkCudaErrors(cufftExecR2C(planr2c, (cufftReal *)fft_data, (cufftComplex *)fft_data));

}

void Fft_lattice::ifft_func(float2 *fft_data)
{

	checkCudaErrors(cufftExecC2R(planc2r, (cufftComplex *)fft_data, (cufftReal *)fft_data));

}

void Fft_lattice::ifft_func_complex(float2 *fft_data)
{

	checkCudaErrors(cufftExecC2C(planc2c, (cufftComplex *)fft_data, (cufftComplex *)fft_data,CUFFT_INVERSE));

}


__global__ void fft_scalar_kernel(float2 *fft_data_compute,float scalar_val,int size)
{


	int tx = blockIdx.x * blockDim.x + threadIdx.x;

	float2 a ;

	if(tx < size)
	{
		a = fft_data_compute[tx];

        a.x /= scalar_val;
        a.y /= scalar_val;
	
		fft_data_compute[tx] = a;

		__syncthreads();

	}

};


void Fft_lattice::fft_scalar(float2 *fft_data_compute,float scalar_val,int size)
{
    dim3 grids(ceil((size)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	fft_scalar_kernel<<<grids,tids>>>(fft_data_compute,scalar_val,size);
	cudaDeviceSynchronize();
}



__global__ void fft_fill_kernel(float2 *fft_compute, float2 *fft_compute_fill,int Nx, int Ny , int Nz ,size_t size, uint Nx2, uint mid_index)
{

		int tx = blockIdx.x * blockDim.x + threadIdx.x;
		int mid_x = floorf(Nx/2);
		int mid_y = floorf(Ny/2);
		int mid_z = floorf(Nz/2);
		if(tx < Nx2*Ny*Nz)
		{
	

			int z = tx/(Nx2*Ny);
			int y = (tx%(Nx2*Ny))/Nx2;
			int x = (tx%(Nx2*Ny))%Nx2;
	
			int e,f,g;

			float2 s1;
			s1.x = 0.0f;
			s1.y = 0.0f;

			if((x == 0) && (y == 0) && (z == 0))
			{
				
				fft_compute_fill[0] = fft_compute[0];
				
			}
			else
			{
			
				if(x == 0)
				{
					e = 0;
				}

				else if(x > 0)
				{
					e = Nx - x;	
				}

				if( y == 0)
				{
					f = 0;
				}
				else if(y > 0)
				{
					f = Nx - y;
				}

				if(z == 0)
				{
					g = 0;
				}
				else if(z > 0)
				{
					g = Nx - z;
				}

				int indd =  x + y *Nx2 + z * (Nx2 *Ny);
				int indd1 = x + y *Nx + z * (Nx*Ny);
				int indd2 = e + f *Nx + g * (Nx*Ny);

	
			 	s1.x = fft_compute[indd].x;
				s1.y = fft_compute[indd].y;
			
				fft_compute_fill[indd1].x = s1.x;
				fft_compute_fill[indd1].y = s1.y;

				fft_compute_fill[indd2].x = s1.x;
				fft_compute_fill[indd2].y = -1.0f * s1.y;
			
			}
        }
		
    	__syncthreads();
}

void Fft_lattice::fft_fill(float2 *fft_compute, float2 *fft_compute_fill,int Nx, int Ny , int Nz, uint mid_index)
{
	
	uint Nx2 = floor(Nx/2.0) +1;
	size_t size = Nx2*Ny*Nz;
	dim3 grids(ceil((size)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	fft_fill_kernel<<<grids,tids>>>(fft_compute,fft_compute_fill,Nx,Ny,Nz,size,Nx2,mid_index);
	cudaDeviceSynchronize();
}




__global__ void sum_individual_gratings_kernel(float2 *sum_gratings,float* fft_function,int Nx,int Ny, int Nz)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;

	if(tx < Nx*Ny*Nz)
	{
		float2 a = sum_gratings[tx];
		fft_function[tx] += a.x;
		
	}
}


void Fft_lattice::indiviual_grating_sum(float2 *sum_gratings,float* fft_function,int Nx,int Ny, int Nz)
{
	dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	sum_individual_gratings_kernel<<<grids,tids>>>(sum_gratings,fft_function,Nx,Ny,Nz);
	cudaDeviceSynchronize();
}


__global__ void add_fft_constant_kernel(float2* individual_grating, float2 fft_constant,int index)
{
	float2 a = fft_constant;
	
	individual_grating[index] = a;

}

void Fft_lattice::add_fft_constants(float2* indvidual_grating, float2 fft_constant, int index)
{
	
	add_fft_constant_kernel<<<1,1>>>(indvidual_grating,fft_constant,index);
	cudaDeviceSynchronize();
}