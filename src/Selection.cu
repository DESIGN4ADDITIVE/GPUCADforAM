

#include "Selection.h"
#include <helper_math.h>
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


__global__ void vertex_selection_two_kernel(float* d_storagebuffer_1, float* d_storagebuffer_2, int Nx, int Ny, int Nz, bool load_selection,
bool boundary_selection, bool delete_selection)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int size = Nx*Ny*Nz;
	float val1 = 0;
	float val2 = 0;
	if(idx < size)
	{
		val1 = d_storagebuffer_1[idx];
		val2 = d_storagebuffer_2[idx];
		
		if(boundary_selection )
		{
			if ((val1 == -1.0) && (val2 != -1.0))
			{
			
				d_storagebuffer_2[idx] = -1.0;
			}
		}
		else if (load_selection)
		{
			if ((val1 == 1.0) && (val2 != 1.0))
			{
			
				d_storagebuffer_2[idx] = 1.0;
			}
		}
		
		else if(delete_selection)
		{
			if (((val2 != 0.0) && (val1 == 0.0)))
			{
				d_storagebuffer_2[idx] = 0.0;
			}
		}
	
	}
}

void Selection::vertex_selection_two(float* d_storagebuffer_1, float* d_storagebuffer_2, int Nx, int Ny, int Nz, bool load_selection,
bool boundary_selection, bool delete_selection)
{
    dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	
	vertex_selection_two_kernel<<<grids,tids>>>(d_storagebuffer_1,d_storagebuffer_2,Nx,Ny,Nz, load_selection,
	boundary_selection, delete_selection);

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



__device__
uint3 calcGridPos_sel(uint i, uint3 gridSizeShift)
{
    uint3 gridPos;
    
    uint z_quo = i / gridSizeShift.z;
    uint z_rem = i % gridSizeShift.z;
    uint y_quo = (z_rem)/gridSizeShift.y;
    uint x_rem = (z_rem) % gridSizeShift.y;

    gridPos.x = x_rem;
    gridPos.y = y_quo;
    gridPos.z = z_quo; 

    return gridPos;
}


__device__
float sampleVolume_hgrid(float *data, uint3 p, uint3 gridSize)
{

    uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
    return (float) data[i];
}

__device__
float sampleVolume_hgrid_val(grid_points *data, uint3 p, uint3 gridSize)
{

    uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
    return (float) data[i].val;
}


__device__
int sampleVolume_hgrid2(grid_points *data, uint3 p, uint3 gridSize, int j)
{
    p.x = min(p.x, gridSize.x);
    p.y = min(p.y, gridSize.y);
    p.z = min(p.z, gridSize.z);
   
    uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;

    return (float) data[i].val;
}


__global__ void update_raster_kernel(float isoval_fixed, float iso_dynamic, float iso1, float iso2,float *raster,float *d_solid, grid_points *vol_one, float *boundary, float *lattice_field,float *fix_lat_field, bool fixed, bool dynamic,int Nx,int Ny, int Nz)
{
	int ind =  (blockDim.x * blockIdx.x) + threadIdx.x;
	uint3 gridPos = calcGridPos_sel(ind,make_uint3(1,Nx,Ny*Nx));
	uint3 gridSize = make_uint3(Nx * 2,  Ny * 2,Nz * 2);

	if((gridPos.x < Nx) && (gridPos.y < Ny) && (gridPos.z < Nz))
	{

		uint3 gridPosone = gridPos * 2;

		int r_val = 0.0;

		float a,b,c,d,e,f,g;
		float p,q,s,u;

		a = b = c = d = e = f = g = 0.0;

		p = q = s = u = 0.0;

		if(!fixed)
		{
			
			float f_fixed[7];
			f_fixed[0] = sampleVolume_hgrid_val(vol_one, gridPosone, gridSize);
			f_fixed[1] = (int(gridPos.x - 1) >= 0 ) ? sampleVolume_hgrid_val(vol_one, gridPosone - make_uint3(1,0,0), gridSize) : nanf("") ;
			f_fixed[2] = ((gridPos.x + 1) < Nx) ? sampleVolume_hgrid_val(vol_one, gridPosone + make_uint3(1,0,0), gridSize) : nanf("") ;
			f_fixed[3] = (int(gridPos.y - 1) >= 0 ) ? sampleVolume_hgrid_val(vol_one, gridPosone - make_uint3(0,1,0), gridSize) : nanf("") ;
			f_fixed[4] = ((gridPos.y + 1) < Ny) ? sampleVolume_hgrid_val(vol_one, gridPosone + make_uint3(0,1,0), gridSize) : nanf("") ;
			f_fixed[5] = (int(gridPos.z - 1) >= 0)  ? sampleVolume_hgrid_val(vol_one, gridPosone - make_uint3(0,0,1), gridSize) : nanf("") ;
			f_fixed[6] = ((gridPos.z + 1) < Nz) ? sampleVolume_hgrid_val(vol_one, gridPosone + make_uint3(0,0,1), gridSize) : nanf("") ;
				
			
		
				
		
				if(f_fixed[1] != nanf("") )
				{
					r_val = ((f_fixed[1] < isoval_fixed) && (f_fixed[0] >= isoval_fixed)) ? 1 : ((f_fixed[1] >= isoval_fixed) && (f_fixed[0] < isoval_fixed)) ? 2 : 0;

					if(r_val == 1)
					{
						a = 1.0;
						
					}
					else if (r_val == 2)
					{
						b = 1.0;
						q = 1.0;
					}

				}

				if((f_fixed[2] != nanf("")))
				{
					r_val = ((f_fixed[2] < isoval_fixed) && (f_fixed[0] >= isoval_fixed)) ? 1 : ((f_fixed[2] >= isoval_fixed) && (f_fixed[0] < isoval_fixed)) ? 3 : 0 ;

					if(r_val == 1)
					{
						a = 1.0;
						p = 1.0;
					}
					else if (r_val == 3)
					{
						
						c = 1.0;
					}
				
				
				}

				if((f_fixed[3] != nanf("")))
				{
					r_val = ((f_fixed[3] < isoval_fixed) && (f_fixed[0] >= isoval_fixed)) ? 1 : ((f_fixed[3] >= isoval_fixed) && (f_fixed[0] < isoval_fixed)) ? 4 : 0 ;
				
					if(r_val == 1)
					{
						a = 1.0;
						
					}
					else if (r_val == 4)
					{
						d = 1.0;
						s = 1.0;
					}
				
				}

				if((f_fixed[4] != nanf("")))
				{
					r_val = ((f_fixed[4] < isoval_fixed) && (f_fixed[0] >= isoval_fixed)) ? 1 : ((f_fixed[4] >= isoval_fixed) && (f_fixed[0] < isoval_fixed)) ? 5 : 0 ;
				
				
					if(r_val == 1)
					{
						a = 1.0;
						p = 1.0;
					}
					else if (r_val == 5)
					{
						e = 1.0;
					}
				
				
				}



				if((f_fixed[5] != nanf("")))
				{
					r_val = ((f_fixed[5] < isoval_fixed) && (f_fixed[0] >= isoval_fixed)) ? 1.0 : ((f_fixed[5] >= isoval_fixed) && (f_fixed[0] < isoval_fixed)) ? 6 : 0 ;

				
					if(r_val == 1)
					{
						a = 1.0;
						
					}
					else if (r_val == 6)
					{
						f = 1.0;
						u = 1.0;
					}
				
				
				
				}

				

				if((f_fixed[6] != nanf("")))
				{
					r_val = ((f_fixed[6] < isoval_fixed) && (f_fixed[0] >= isoval_fixed)) ? 1 : ((f_fixed[6] >= isoval_fixed) && (f_fixed[0] < isoval_fixed)) ? 7 : 0 ;


					if(r_val == 1)
					{
						a = 1.0;
						p = 1.0;
					}
					else if (r_val == 7)
					{
						g = 1.0;
					}
				
				
				}


			
		
		}

		__syncthreads();
		
	
		/* To do for lattice structure */

		if((!dynamic) && (r_val == 0))
		{
			float field[7];
			
			field[0] = sampleVolume_hgrid(boundary, gridPosone, gridSize);
			field[1] = (int(gridPos.x - 1) >= 0 ) ? sampleVolume_hgrid(boundary, gridPosone - make_uint3(1,0,0), gridSize) : nanf("") ;
			field[2] = ((gridPos.x + 1) < Nx) ? sampleVolume_hgrid(boundary, gridPosone + make_uint3(1,0,0), gridSize) : nanf("") ;
			field[3] = (int(gridPos.y - 1) >= 0 ) ? sampleVolume_hgrid(boundary, gridPosone - make_uint3(0,1,0), gridSize) : nanf("") ;
			field[4] = ((gridPos.y + 1) < Ny) ? sampleVolume_hgrid(boundary, gridPosone + make_uint3(0,1,0), gridSize) : nanf("") ;
			field[5] = (int(gridPos.z - 1) >= 0)  ? sampleVolume_hgrid(boundary, gridPosone - make_uint3(0,0,1), gridSize) : nanf("") ;
			field[6] = ((gridPos.z + 1) < Nz) ? sampleVolume_hgrid(boundary, gridPosone + make_uint3(0,0,1), gridSize) : nanf("") ;
			
			
			if(r_val == 0)
			{
				if(field[1] != nanf("") )
				{
					r_val = ((field[1] < iso_dynamic) && (field[0] >= iso_dynamic)) ? 1 : ((field[1] >= iso_dynamic) && (field[0] < iso_dynamic)) ? 2 : 0 ;

					if(r_val == 1)
					{
						a = 1.0;
						
						q = 1.0;
						
						
					}
					else if (r_val == 2)
					{
						
						b = 1.0;

						q = 1.0;
						
					}

				}

				

				if((field[2] != nanf("")))
				{
					r_val = ((field[2] < iso_dynamic) && (field[0] >= iso_dynamic)) ? 1 : ((field[2] >= iso_dynamic) && (field[0] < iso_dynamic)) ? 3 : 0 ;

					if(r_val == 1)
					{
						a = 1.0;
						p = 1.0;
					}
					else if (r_val == 3)
					{
						c = 1.0;
						p = 1.0;
					}
				
				
				}

				

				if((field[3] != nanf("")))
				{
					r_val = ((field[3] < iso_dynamic) && (field[0] >= iso_dynamic)) ? 1 : ((field[3] >= iso_dynamic) && (field[0] < iso_dynamic)) ? 4 : 0 ;
				
					if(r_val == 1)
					{
						a = 1.0;
						s = 1.0;
						
					}
					else if (r_val == 4)
					{
						d = 1.0;
						s = 1.0;
					}
				
				}

				

				if((field[4] != nanf("")))
				{
					r_val = ((field[4] < iso_dynamic) && (field[0] >= iso_dynamic)) ? 1 : ((field[4] >= iso_dynamic) && (field[0] < iso_dynamic)) ? 5 : 0 ;
				
				
					if(r_val == 1)
					{
						a = 1.0;
						p = 1.0;
					}
					else if (r_val == 5)
					{
						e = 1.0;
						p = 1.0;
					}
				
				
				}

				


				if((field[5] != nanf("")))
				{
					r_val = ((field[5] < iso_dynamic) && (field[0] >= iso_dynamic)) ? 1 : ((field[5] >= iso_dynamic) && (field[0] < iso_dynamic)) ? 6 : 0 ;

				
					if(r_val == 1)
					{
						a = 1.0;
						u = 1.0;
						
					}
					else if (r_val == 6)
					{
						f = 1.0;
						u = 1.0;
					}
				
				
				
				}

				

				if((field[6] != nanf("")))
				{
					r_val = ((field[6] < iso_dynamic) && (field[0] >= iso_dynamic)) ? 1 : ((field[6] >= iso_dynamic) && (field[0] < iso_dynamic)) ? 7 : 0 ;


					if(r_val == 1)
					{
						a = 1.0;
						p = 1;
					}
					else if (r_val == 7)
					{
						g = 1.0;
						p = 1.0;
					}
				
				
				}


			
			}

		
		}

		__syncthreads();	

		if(r_val == 0)
		{
			
			float f_fixed = sampleVolume_hgrid_val(vol_one, gridPosone, gridSize);

			float f_dynamic = sampleVolume_hgrid(boundary, gridPosone, gridSize);

			float latt_check = sampleVolume_hgrid(fix_lat_field, gridPosone, gridSize) ;

			if((f_fixed < isoval_fixed ) || (f_dynamic < iso_dynamic ) || ((latt_check > iso1) && latt_check <= iso2))
			{
				a = 0.625;

				p = 1;
			}
			
		}

		__syncthreads();

		if(a > 0)
		{
			raster[ind] = a;
		}

		__syncthreads();

		if((b > 0) && (gridPos.x > 0) )
		{
			raster[(gridPos.x - 1) +(gridPos.y * Nx) + (gridPos.z * Nx * Ny)] = b;	
		}

		__syncthreads();
		
		if((c > 0) && (gridPos.x < (Nx - 1)))
		{
			raster[(gridPos.x + 1) +(gridPos.y * Nx) + (gridPos.z * Nx * Ny)] = c;
		}

		__syncthreads();

		if((d > 0) && (gridPos.y > 0))
		{
			raster[(gridPos.x) +((gridPos.y - 1) * Nx) + (gridPos.z * Nx * Ny)] = d;
		}

		__syncthreads();

		if((e > 0) && (gridPos.y < (Ny - 1)))
		{
			raster[(gridPos.x) +((gridPos.y + 1) * Nx) + (gridPos.z * Nx * Ny)] = e;
		}

		__syncthreads();

		if((f > 0) && (gridPos.z > 0))
		{
			raster[(gridPos.x) +((gridPos.y) * Nx) + ((gridPos.z - 1) * Nx * Ny)] = f;
		}

		__syncthreads();

		if((g > 0) && (gridPos.z < (Nz - 1)))
		{
			raster[(gridPos.x) +((gridPos.y) * Nx) + ((gridPos.z + 1) * Nx * Ny)] = g;
		}

		__syncthreads();



		if(p > 0)
		{
			d_solid[ind] = p;
		}

		__syncthreads();
	
		if((q > 0) && (gridPos.x > 0))
		{
			d_solid[(gridPos.x - 1) +(gridPos.y * Nx) + (gridPos.z * Nx * Ny)] = q;
		}

		__syncthreads();


		if((s > 0) && (gridPos.y > 0))
		{
			d_solid[(gridPos.x) +((gridPos.y - 1) * Nx) + (gridPos.z * Nx * Ny)] = s;
		}

		__syncthreads();


		if((u > 0) && (gridPos.z > 0))
		{
			d_solid[(gridPos.x) +(gridPos.y * Nx) + ((gridPos.z - 1 ) * Nx * Ny)] = u;
		}

		__syncthreads();
		
	
	}

	__syncthreads();


}


__global__ void Reduction_sel(uint *d_DataIn, uint *d_DataOut, int block_num)
{
	__shared__ uint sdata[1024];

	for (int j=threadIdx.x; j<1024; j+= 32*blockDim.x)  
	
	sdata[j]=0;

	unsigned int tid = threadIdx.x;

	int index;
	int e;
	e = (block_num/1024) + (!(block_num%1024)?0:1);
	
	uint c = 0;

	for (int k = 0; k< e;k++)
	{
		index = tid + k*1024;
		if(index < block_num)
		{
			sdata[tid] = d_DataIn[index];
		
			c += sdata[tid];			
		}
		__syncthreads();
	
	}

	sdata[tid] = c;
	__syncthreads();

	

	for(unsigned int s=blockDim.x/2; s>0;s/=2) 
	{
		
		
		if (tid < s) 
		{
			
			sdata[tid] += sdata[tid + s];
			
		}
		__syncthreads();
	}

	
	if (tid == 0) 
	{
		d_DataOut[0] = sdata[0];

		
		
	}
	
}

void Selection::raster_update(float isoval_fixed, float iso_dynamic, float iso1, float iso2, float *raster, float *d_solid, grid_points *vol_one, float *boundary, float *lattice_field, float *fix_lat_field,  bool fixed, bool dynamic,int Nx,int Ny, int Nz)
{
	dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);
	dim3 tids(1024,1,1);

	update_raster_kernel<<<grids,tids>>>(isoval_fixed, iso_dynamic, iso1, iso2,raster,d_solid, vol_one, boundary, lattice_field,fix_lat_field, fixed, dynamic, Nx, Ny, Nz);
	cudaDeviceSynchronize();

	getLastCudaError("Raster copy failed");
}

__global__ void update_make_region_kernel(float isoval_fixed, float iso_dynamic, float iso1, float iso2,float *raster,float *d_solid, grid_points *vol_one, float *boundary, float *lattice_field,float *fix_lat_field, bool fixed, bool dynamic,int Nx,int Ny, int Nz)
{
	int ind =  (blockDim.x * blockIdx.x) + threadIdx.x;
	uint3 gridPos = calcGridPos_sel(ind,make_uint3(1,Nx,Ny*Nx));
	uint3 gridSize = make_uint3(Nx * 2,  Ny * 2,Nz * 2);

	if((gridPos.x < Nx) && (gridPos.y < Ny) && (gridPos.z < Nz))
	{
		

		uint3 gridPosone = gridPos * 2;

		int r_val = 0.0;

	
		float a,b,c,d,e,f,g;


		a = b = c = d = e = f = g = 0.0;


		float field[7];
			

		field[0] = sampleVolume_hgrid(boundary, gridPosone, gridSize);
		field[1] = (int(gridPos.x - 1) >= 0 ) ? sampleVolume_hgrid(boundary, gridPosone - make_uint3(1,0,0), gridSize) : nanf("") ;
		field[2] = ((gridPos.x + 1) < Nx) ? sampleVolume_hgrid(boundary, gridPosone + make_uint3(1,0,0), gridSize) : nanf("") ;
		field[3] = (int(gridPos.y - 1) >= 0 ) ? sampleVolume_hgrid(boundary, gridPosone - make_uint3(0,1,0), gridSize) : nanf("") ;
		field[4] = ((gridPos.y + 1) < Ny) ? sampleVolume_hgrid(boundary, gridPosone + make_uint3(0,1,0), gridSize) : nanf("") ;
		field[5] = (int(gridPos.z - 1) >= 0)  ? sampleVolume_hgrid(boundary, gridPosone - make_uint3(0,0,1), gridSize) : nanf("") ;
		field[6] = ((gridPos.z + 1) < Nz) ? sampleVolume_hgrid(boundary, gridPosone + make_uint3(0,0,1), gridSize) : nanf("") ;
		
		
		if(r_val == 0)
		{
			if(field[1] != nanf("") )
			{
				r_val = ((field[1] < iso_dynamic) && (field[0] >= iso_dynamic)) ? 1 : ((field[1] >= iso_dynamic) && (field[0] < iso_dynamic)) ? 2 : 0 ;

				if(r_val == 1)
				{
					a = 1.0;
					
		
					
					
				}
				else if (r_val == 2)
				{
					
					b = 1.0;

					
					
				}

			}

			

			if((field[2] != nanf("")))
			{
				r_val = ((field[2] < iso_dynamic) && (field[0] >= iso_dynamic)) ? 1 : ((field[2] >= iso_dynamic) && (field[0] < iso_dynamic)) ? 3 : 0 ;

				if(r_val == 1)
				{
					a = 1.0;
				
				}
				else if (r_val == 3)
				{
					c = 1.0;
					
				}
			
			
			}

			

			if((field[3] != nanf("")))
			{
				r_val = ((field[3] < iso_dynamic) && (field[0] >= iso_dynamic)) ? 1 : ((field[3] >= iso_dynamic) && (field[0] < iso_dynamic)) ? 4 : 0 ;
			
				if(r_val == 1)
				{
					a = 1.0;
					
					
				}
				else if (r_val == 4)
				{
					d = 1.0;
					
				}
			
			}

			

			if((field[4] != nanf("")))
			{
				r_val = ((field[4] < iso_dynamic) && (field[0] >= iso_dynamic)) ? 1 : ((field[4] >= iso_dynamic) && (field[0] < iso_dynamic)) ? 5 : 0 ;
			
			
				if(r_val == 1)
				{
					a = 1.0;
				}
				else if (r_val == 5)
				{
					e = 1.0;
					
				}
			
			
			}

			


			if((field[5] != nanf("")))
			{
				r_val = ((field[5] < iso_dynamic) && (field[0] >= iso_dynamic)) ? 1 : ((field[5] >= iso_dynamic) && (field[0] < iso_dynamic)) ? 6 : 0 ;

			
				if(r_val == 1)
				{
					a = 1.0;
					
					
				}
				else if (r_val == 6)
				{
					f = 1.0;
					
				}
			
			
			
			}

			

			if((field[6] != nanf("")))
			{
				r_val = ((field[6] < iso_dynamic) && (field[0] >= iso_dynamic)) ? 1 : ((field[6] >= iso_dynamic) && (field[0] < iso_dynamic)) ? 7 : 0 ;


				if(r_val == 1)
				{
					a = 1.0;
					
				}
				else if (r_val == 7)
				{
					g = 1.0;
				
				}
			
			
			}


		
		}



		__syncthreads();	

		if(r_val == 0)
		{
			
			float f_fixed = sampleVolume_hgrid_val(vol_one, gridPosone, gridSize);

			float f_dynamic = sampleVolume_hgrid(boundary, gridPosone, gridSize);

			float latt_check = sampleVolume_hgrid(fix_lat_field, gridPosone, gridSize) ;

			if(((f_fixed < isoval_fixed ) && (f_dynamic < iso_dynamic )) || ((f_dynamic < iso_dynamic ) && ((latt_check > iso1) && latt_check <= iso2)))
			{
				a = 0.643;
				

			}
			
		}

		if(a > 0)
		{
			raster[ind] = a;
		}

		__syncthreads();

		if((b > 0) && (gridPos.x > 0) )
		{
			raster[(gridPos.x - 1) +(gridPos.y * Nx) + (gridPos.z * Nx * Ny)] = b;	
		}

		__syncthreads();
		
		if((c > 0) && (gridPos.x < (Nx - 1)))
		{
			raster[(gridPos.x + 1) +(gridPos.y * Nx) + (gridPos.z * Nx * Ny)] = c;
		}

		__syncthreads();

		if((d > 0) && (gridPos.y > 0))
		{
			raster[(gridPos.x) +((gridPos.y - 1) * Nx) + (gridPos.z * Nx * Ny)] = d;
		}

		__syncthreads();

		if((e > 0) && (gridPos.y < (Ny - 1)))
		{
			raster[(gridPos.x) +((gridPos.y + 1) * Nx) + (gridPos.z * Nx * Ny)] = e;
		}

		__syncthreads();

		if((f > 0) && (gridPos.z > 0))
		{
			raster[(gridPos.x) +((gridPos.y) * Nx) + ((gridPos.z - 1) * Nx * Ny)] = f;
		}

		__syncthreads();

		if((g > 0) && (gridPos.z < (Nz - 1)))
		{
			raster[(gridPos.x) +((gridPos.y) * Nx) + ((gridPos.z + 1) * Nx * Ny)] = g;
		}

		__syncthreads();



		

	}

}


void Selection::raster_make_region(float isoval_fixed, float iso_dynamic, float iso1, float iso2, float *raster, float *d_solid, grid_points *vol_one, float *boundary, float *lattice_field, float *fix_lat_field,  bool fixed, bool dynamic,int Nx,int Ny, int Nz)
{
	dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);
	dim3 tids(1024,1,1);


	update_make_region_kernel<<<grids,tids>>>(isoval_fixed, iso_dynamic, iso1, iso2,raster,d_solid, vol_one, boundary, lattice_field,fix_lat_field, fixed, dynamic, Nx, Ny, Nz);
	cudaDeviceSynchronize();


	getLastCudaError("Raster make region copy failed");
}




__global__ void constrained_vol_kernel(REAL *solid, int *fixed_free, uint *d_sum_solid, float volfrac, int Nx , int Ny, int Nz)
{
	int index = threadIdx.x + (blockDim.x*blockIdx.x);
	int tx  = threadIdx.x;

	__shared__ float cc[1024];

  	uint z = index / ((Nx) * (Ny));
    uint z_rem = index % ((Nx) * (Ny));
    uint y = (z_rem)/ ((Nx));
    uint x = (z_rem) % ((Nx));
	float a = 0;
	int b = 0;
	if((x < (Nx - 1)) && (y < (Ny - 1)) && (z < (Nz -1 )))
	{
		a = solid[index];
		b = fixed_free[index];
		if(a == 1)
		{
			if(b == -1)
			{
				cc[tx] = 0.0;
			}
			else
			{
				cc[tx] = volfrac;
			}
		}
		else
		{
			cc[tx] = 0.0;
		}
	}
	else
	{
		cc[tx] = 0.0;
	}

	__syncthreads();

	for(int stride = blockDim.x/2; stride>0; stride/=2)
	{
		if(tx < stride)
		{
			float Result = cc[tx];
			Result += cc[tx+stride];
			cc[tx] = Result;

		}
		__syncthreads();
	}
	

	if (tx == 0)
	{
		d_sum_solid[blockIdx.x] = cc[tx];
		
	}

	__syncthreads();
}


void Selection::constrained_vol(REAL *solid, int *fixed_free, uint *solid_voxels, float volfrac, int Nx, int Ny, int Nz)
{
	
	dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	
	uint *d_sum_solid;
	checkCudaErrors(cudaMalloc((void **)&d_sum_solid, sizeof(uint)*(grids.x)));
  	cudaMemset(d_sum_solid, 0.0, sizeof(uint)*grids.x);

	constrained_vol_kernel<<<grids,tids>>>(solid,fixed_free, d_sum_solid,volfrac, Nx , Ny, Nz);

	unsigned int  x_grid = 1;
	unsigned int  x_thread = 1024;
	Reduction_sel<<<x_grid, x_thread>>>(d_sum_solid,d_sum_solid,grids.x);
	cudaDeviceSynchronize();
	
    cudaMemcpy(solid_voxels, d_sum_solid, sizeof(uint), cudaMemcpyDeviceToHost);
	cudaFree(d_sum_solid);
}



__global__ void Fixed_free_kernel(int *fixed_free, REAL *raster, int Nx, int Ny, int Nz)
{
	int index = threadIdx.x + (blockDim.x*blockIdx.x);


  	uint z = index / ((Nx) * (Ny));
    uint z_rem = index % ((Nx) * (Ny));
    uint y = (z_rem)/ ((Nx));
    uint x = (z_rem) % ((Nx));

	float a = raster[index];

	if((x < (Nx - 1)) && (y < (Ny - 1)) && (z < (Nz -1 )))
	{
		if(a == 0.643f)
		{
			fixed_free[index] = -1;
			
		}
	
	}
}


void Selection::fixed_free(int *fixed_free, REAL *raster, int Nx, int Ny, int Nz)
{
	
	dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	

	Fixed_free_kernel<<<grids,tids>>>(fixed_free,raster, Nx , Ny, Nz);

	cudaDeviceSynchronize();

}