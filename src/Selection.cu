

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

__global__ void facet_selection_kernel(float* d_storagebuffer_1, float* d_storagebuffer_2, int nfacets, bool load_selection,
bool boundary_selection, bool delete_selection)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int size = nfacets;
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

void Selection::facet_selection(float* d_storagebuffer_1, float* d_storagebuffer_2, int nfacets, bool load_selection,
bool boundary_selection, bool delete_selection)
{
    dim3 grids(ceil((nfacets)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	
	facet_selection_kernel<<<grids,tids>>>(d_storagebuffer_1,d_storagebuffer_2,nfacets, load_selection,
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

__global__ void update_raster_kernel(float isoval_fixed, float iso_dynamic, float iso1, float iso2,float *raster,float *d_solid, grid_points *vol_one, float *boundary, float *lattice_field,float *fix_lat_field, bool fixed, bool dynamic,int Nx,int Ny, int Nz,
bool obj_union, bool obj_difference, bool obj_intersect)
{
	int ind =  (blockDim.x * blockIdx.x) + threadIdx.x;
	uint3 gridPos = calcGridPos_sel(ind,make_uint3(1,Nx,Ny*Nx));
	uint3 gridSize = make_uint3(Nx * 2,  Ny * 2,Nz * 2);

	if((gridPos.x < (Nx)) && (gridPos.y < (Ny)) && (gridPos.z < (Nz)))
	{

		uint3 gridPosone = gridPos * 2;

		int r_val = 0.0;

		float a,b,c,d,e,f,g;

		float p,q,r,s,t,u,v;
	
		a = b = c = d = e = f = g = 0.0;

		p = q = r = s = t = u = v = 0.0;


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
					p = 0.5;
			
				}
				else if (r_val == 2)
				{
					b = 1.0;
					q = 0.5;
				}

			}

			if((f_fixed[2] != nanf("")))
			{
				r_val = ((f_fixed[2] < isoval_fixed) && (f_fixed[0] >= isoval_fixed)) ? 1 : ((f_fixed[2] >= isoval_fixed) && (f_fixed[0] < isoval_fixed)) ? 3 : 0 ;

				if(r_val == 1)
				{
					a = 1.0;
					p = 0.5;
				}
				else if (r_val == 3)
				{
					
					c = 1.0;
					r = 0.5;
				}
			
			
			}

			if((f_fixed[3] != nanf("")))
			{
				r_val = ((f_fixed[3] < isoval_fixed) && (f_fixed[0] >= isoval_fixed)) ? 1 : ((f_fixed[3] >= isoval_fixed) && (f_fixed[0] < isoval_fixed)) ? 4 : 0 ;
			
				if(r_val == 1)
				{
					a = 1.0;
					p = 0.5;
					
				}
				else if (r_val == 4)
				{
					d = 1.0;
					s = 0.5;
				}
			
			}

			if((f_fixed[4] != nanf("")))
			{
				r_val = ((f_fixed[4] < isoval_fixed) && (f_fixed[0] >= isoval_fixed)) ? 1 : ((f_fixed[4] >= isoval_fixed) && (f_fixed[0] < isoval_fixed)) ? 5 : 0 ;
			
			
				if(r_val == 1)
				{
					a = 1.0;
					p = 0.5;
				}
				else if (r_val == 5)
				{
					e = 1.0;
					t = 0.5;
				}
			
			
			}



			if((f_fixed[5] != nanf("")))
			{
				r_val = ((f_fixed[5] < isoval_fixed) && (f_fixed[0] >= isoval_fixed)) ? 1.0 : ((f_fixed[5] >= isoval_fixed) && (f_fixed[0] < isoval_fixed)) ? 6 : 0 ;

			
				if(r_val == 1)
				{
					a = 1.0;
					p = 0.5;
					
				}
				else if (r_val == 6)
				{
					f = 1.0;
					u = 0.5;
				}
			
			
			
			}

			

			if((f_fixed[6] != nanf("")))
			{
				r_val = ((f_fixed[6] < isoval_fixed) && (f_fixed[0] >= isoval_fixed)) ? 1 : ((f_fixed[6] >= isoval_fixed) && (f_fixed[0] < isoval_fixed)) ? 7 : 0 ;


				if(r_val == 1)
				{
					a = 1.0;
					p = 0.5;
				}
				else if (r_val == 7)
				{
					g = 1.0;
					v = 0.5;
				}
			
			
			}


			
		
		}

		__syncthreads();
		
	
		// /* To do f	or lattice structure */
		// /* To do f	or dynamic region  */
		

		if(r_val == 0)
		{
			
			float f_fixed = sampleVolume_hgrid_val(vol_one, gridPosone, gridSize);

			if((f_fixed < isoval_fixed ))
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


		if(gridPos.x < (Nx - 1)  && (gridPos.y < (Ny - 1) && (gridPos.z < (Nz - 1))))
		{

			if((p > 0) || (r > 0) || (t > 0) || (v > 0))
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
			
	}

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

void Selection::raster_update(float isoval_fixed, float iso_dynamic, float iso1, float iso2, float *raster, float *d_solid, grid_points *vol_one, float *boundary, float *lattice_field, float *fix_lat_field,  bool fixed, bool dynamic,int Nx,int Ny, int Nz, bool obj_union,
bool obj_difference, bool obj_intersect)
{
	dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);
	dim3 tids(1024,1,1);

	update_raster_kernel<<<grids,tids>>>(isoval_fixed, iso_dynamic, iso1, iso2,raster,d_solid,vol_one, boundary, lattice_field,fix_lat_field, fixed, dynamic, Nx, Ny, Nz, obj_union, obj_difference, obj_intersect);
	cudaDeviceSynchronize();

	getLastCudaError("Raster copy failed");
}


__global__ void update_raster_fixed_region_kernel(float isoval_fixed_region,float *raster, grid_points *vol_topo, grid_points *vol_one, int Nx,int Ny, int Nz, bool show_domain)
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

		float f_fixed[7];
		f_fixed[0] = sampleVolume_hgrid_val(vol_topo, gridPosone, gridSize);
		f_fixed[1] = (int(gridPos.x - 1) >= 0 ) ? sampleVolume_hgrid_val(vol_topo, gridPosone - make_uint3(1,0,0), gridSize) : nanf("") ;
		f_fixed[2] = ((gridPos.x + 1) < Nx) ? sampleVolume_hgrid_val(vol_topo, gridPosone + make_uint3(1,0,0), gridSize) : nanf("") ;
		f_fixed[3] = (int(gridPos.y - 1) >= 0 ) ? sampleVolume_hgrid_val(vol_topo, gridPosone - make_uint3(0,1,0), gridSize) : nanf("") ;
		f_fixed[4] = ((gridPos.y + 1) < Ny) ? sampleVolume_hgrid_val(vol_topo, gridPosone + make_uint3(0,1,0), gridSize) : nanf("") ;
		f_fixed[5] = (int(gridPos.z - 1) >= 0)  ? sampleVolume_hgrid_val(vol_topo, gridPosone - make_uint3(0,0,1), gridSize) : nanf("") ;
		f_fixed[6] = ((gridPos.z + 1) < Nz) ? sampleVolume_hgrid_val(vol_topo, gridPosone + make_uint3(0,0,1), gridSize) : nanf("") ;
			
		
		
		float f_free[7];
		f_free[0] = sampleVolume_hgrid_val(vol_one, gridPosone, gridSize);
		f_free[1] = (int(gridPos.x - 1) >= 0 ) ? sampleVolume_hgrid_val(vol_one, gridPosone - make_uint3(1,0,0), gridSize) : nanf("") ;
		f_free[2] = ((gridPos.x + 1) < Nx) ? sampleVolume_hgrid_val(vol_one, gridPosone + make_uint3(1,0,0), gridSize) : nanf("") ;
		f_free[3] = (int(gridPos.y - 1) >= 0 ) ? sampleVolume_hgrid_val(vol_one, gridPosone - make_uint3(0,1,0), gridSize) : nanf("") ;
		f_free[4] = ((gridPos.y + 1) < Ny) ? sampleVolume_hgrid_val(vol_one, gridPosone + make_uint3(0,1,0), gridSize) : nanf("") ;
		f_free[5] = (int(gridPos.z - 1) >= 0)  ? sampleVolume_hgrid_val(vol_one, gridPosone - make_uint3(0,0,1), gridSize) : nanf("") ;
		f_free[6] = ((gridPos.z + 1) < Nz) ? sampleVolume_hgrid_val(vol_one, gridPosone + make_uint3(0,0,1), gridSize) : nanf("") ;
		

		if(f_fixed[1] != nanf("") )
		{
			r_val = ((f_fixed[1] < isoval_fixed_region) && (f_fixed[0] >= isoval_fixed_region)) ? 1 : ((f_fixed[1] >= isoval_fixed_region) && (f_fixed[0] < isoval_fixed_region)) ? 2 : 0;

			if(r_val == 1)
			{
				a = 1.0;
				
			}
			else if (r_val == 2)
			{
				b = 1.0;
				
			}

			if((r_val == 0) && show_domain)
			{
				if(f_free[1] != nanf("") )
				{
					r_val = ((f_free[1] < isoval_fixed_region) && (f_free[0] >= isoval_fixed_region)) ? 1 : ((f_free[1] >= isoval_fixed_region) && (f_free[0] < isoval_fixed_region)) ? 2 : 0;
				
					if(r_val == 1)
					{
						a = 1.0;

					}
					else if (r_val == 2)
					{
						b = 1.0;
						
					}
				
				}
		
			}

		}

		if((f_fixed[2] != nanf("")))
		{
			r_val = ((f_fixed[2] < isoval_fixed_region) && (f_fixed[0] >= isoval_fixed_region)) ? 1 : ((f_fixed[2] >= isoval_fixed_region) && (f_fixed[0] < isoval_fixed_region)) ? 3 : 0 ;

			if(r_val == 1)
			{
				a = 1.0;
			
			}
			else if (r_val == 3)
			{
				
				c = 1.0;
			}

			if((r_val == 0) && show_domain)
			{
				if((f_free[2] != nanf("")))
				{
					r_val = ((f_free[2] < isoval_fixed_region) && (f_free[0] >= isoval_fixed_region)) ? 1 : ((f_free[2] >= isoval_fixed_region) && (f_free[0] < isoval_fixed_region)) ? 3 : 0 ;

					if(r_val == 1)
					{
						a = 1.0;
						
					}
					else if (r_val == 3)
					{
						
						c = 1.0;
					}
				}
			}
		
		
		}

		if((f_fixed[3] != nanf("")))
		{
			r_val = ((f_fixed[3] < isoval_fixed_region) && (f_fixed[0] >= isoval_fixed_region)) ? 1 : ((f_fixed[3] >= isoval_fixed_region) && (f_fixed[0] < isoval_fixed_region)) ? 4 : 0 ;
		
			if(r_val == 1)
			{
				a = 1.0;
				
			}
			else if (r_val == 4)
			{
				d = 1.0;
			
			}

			if((r_val == 0) && show_domain)
			{
				if((f_free[3] != nanf("")))
				{
					r_val = ((f_free[3] < isoval_fixed_region) && (f_free[0] >= isoval_fixed_region)) ? 1 : ((f_free[3] >= isoval_fixed_region) && (f_free[0] < isoval_fixed_region)) ? 4 : 0 ;
			
					if(r_val == 1)
					{
						a = 1.0;
						
					}
					else if (r_val == 4)
					{
						d = 1.0;
						
					}

				}
			}
		
		}

		if((f_fixed[4] != nanf("")))
		{
			r_val = ((f_fixed[4] < isoval_fixed_region) && (f_fixed[0] >= isoval_fixed_region)) ? 1 : ((f_fixed[4] >= isoval_fixed_region) && (f_fixed[0] < isoval_fixed_region)) ? 5 : 0 ;
		
		
			if(r_val == 1)
			{
				a = 1.0;
				
			}
			else if (r_val == 5)
			{
				e = 1.0;
			}

			if((r_val == 0) && show_domain)
			{
				if((f_free[4] != nanf("")))
				{
					r_val = ((f_free[4] < isoval_fixed_region) && (f_free[0] >= isoval_fixed_region)) ? 1 : ((f_free[4] >= isoval_fixed_region) && (f_free[0] < isoval_fixed_region)) ? 5 : 0 ;


					if(r_val == 1)
					{
						a = 1.0;
						
					}
					else if (r_val == 5)
					{
						e = 1.0;
					}
				}

			}	
		
		
		}



		if((f_fixed[5] != nanf("")))
		{
			r_val = ((f_fixed[5] < isoval_fixed_region) && (f_fixed[0] >= isoval_fixed_region)) ? 1.0 : ((f_fixed[5] >= isoval_fixed_region) && (f_fixed[0] < isoval_fixed_region)) ? 6 : 0 ;

		
			if(r_val == 1)
			{
				a = 1.0;
				
			}
			else if (r_val == 6)
			{
				f = 1.0;
			
			}
			if((r_val == 0) && show_domain)
			{
				if((f_free[5] != nanf("")))
				{
					r_val = ((f_free[5] < isoval_fixed_region) && (f_free[0] >= isoval_fixed_region)) ? 1.0 : ((f_free[5] >= isoval_fixed_region) && (f_free[0] < isoval_fixed_region)) ? 6 : 0 ;

				
					if(r_val == 1)
					{
						a = 1.0;
						
					}
					else if (r_val == 6)
					{
						f = 1.0;
						
					}

				}
			}
		
		}

		

		if((f_fixed[6] != nanf("")))
		{
			r_val = ((f_fixed[6] < isoval_fixed_region) && (f_fixed[0] >= isoval_fixed_region)) ? 1 : ((f_fixed[6] >= isoval_fixed_region) && (f_fixed[0] < isoval_fixed_region)) ? 7 : 0 ;


			if(r_val == 1)
			{
				a = 1.0;
				
			}
			else if (r_val == 7)
			{
				g = 1.0;
			}


			if((r_val == 0) && show_domain)
			{

				if((f_free[6] != nanf("")))
				{
					r_val = ((f_free[6] < isoval_fixed_region) && (f_free[0] >= isoval_fixed_region)) ? 1 : ((f_free[6] >= isoval_fixed_region) && (f_free[0] < isoval_fixed_region)) ? 7 : 0 ;


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
		
		
		}

	

		__syncthreads();	

		if(r_val == 0)
		{
			
			float f_fixed = sampleVolume_hgrid_val(vol_topo, gridPosone, gridSize);
			float f_free = sampleVolume_hgrid_val(vol_one, gridPosone, gridSize);
			if((f_fixed < isoval_fixed_region ) )
			{
				a = 0.625;

				
			}
			if(show_domain && (f_free < isoval_fixed_region ))
			{
				a = 0.625;

				
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
		}

		__syncthreads();


}


void Selection::raster_region_update(float isoval_fixed_region,  float *raster, grid_points *vol_topo,grid_points *vol_one, int Nx,int Ny, int Nz, bool show_domain)
{
	dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);
	dim3 tids(1024,1,1);

	update_raster_fixed_region_kernel<<<grids,tids>>>(isoval_fixed_region,raster, vol_topo, vol_one, Nx, Ny, Nz,show_domain);
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




__global__ void constrained_vol_kernel(REAL *solid, grid_points *vol_topo, uint *d_sum_solid, float volfrac, int Nx , int Ny, int Nz)
{
	int index = threadIdx.x + (blockDim.x*blockIdx.x);
	int tx  = threadIdx.x;

	__shared__ float cc[1024];

  	uint z = index / ((Nx) * (Ny));
    uint z_rem = index % ((Nx) * (Ny));
    uint y = (z_rem)/ ((Nx));
    uint x = (z_rem) % ((Nx));
	uint index2 = (2*x) + ((2*y) * (2*Nx)) + ((2*z)*((2*Nx)*(2*Ny)));
	float a = 0;
	int b = 0;
	if((x < (Nx - 1)) && (y < (Ny - 1)) && (z < (Nz -1 )))
	{
		a = solid[index];
		b = int(vol_topo[index2].val);
		if((a == 1) || (a == 0.5))
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


void Selection::constrained_vol(REAL *solid, grid_points *vol_topo, uint *solid_voxels, float volfrac, int Nx, int Ny, int Nz)
{
	
	dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	
	uint *d_sum_solid;
	checkCudaErrors(cudaMalloc((void **)&d_sum_solid, sizeof(uint)*(grids.x)));
  	cudaMemset(d_sum_solid, 0.0, sizeof(uint)*grids.x);

	constrained_vol_kernel<<<grids,tids>>>(solid,vol_topo, d_sum_solid,volfrac, Nx , Ny, Nz);

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

__device__
uint3 calcGridPos_selection(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
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

__device__ uint2 index_cal(uint edge, uint3 gridpos,uint3 gridSizeMask)
{
	uint2 id = {0,0};

	
	switch (edge)
	{
		case 0:
			id.x = (gridpos.x) + (gridpos.y * (gridSizeMask.x + 1) )+ (gridpos.z *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			id.y = (gridpos.x + 1) + (gridpos.y * (gridSizeMask.x + 1)) + (gridpos.z *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			return id;
			break;
		case 1:
			id.x = (gridpos.x + 1) + (gridpos.y * (gridSizeMask.x + 1) )+ (gridpos.z *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			id.y = (gridpos.x + 1) + ((gridpos.y + 1) * (gridSizeMask.x + 1)) + (gridpos.z *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			return id;
			break;


		case 2:
			id.x = (gridpos.x) + ((gridpos.y + 1) * (gridSizeMask.x + 1) )+ (gridpos.z *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			id.y = (gridpos.x + 1) + ((gridpos.y + 1) * (gridSizeMask.x + 1)) + (gridpos.z *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			return id;
			break;

		case 3:
			id.x = (gridpos.x) + ((gridpos.y) * (gridSizeMask.x + 1) )+ (gridpos.z *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			id.y = (gridpos.x) + ((gridpos.y + 1) * (gridSizeMask.x + 1)) + (gridpos.z *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			return id;
			break;

		case 4:
			id.x = (gridpos.x) + (gridpos.y * (gridSizeMask.x + 1) )+ ((gridpos.z + 1) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			id.y = (gridpos.x + 1) + (gridpos.y * (gridSizeMask.x + 1)) + ((gridpos.z + 1) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			return id;
			break;
		case 5:
			id.x = (gridpos.x + 1) + (gridpos.y * (gridSizeMask.x + 1) )+ ((gridpos.z + 1) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			id.y = (gridpos.x + 1) + ((gridpos.y + 1) * (gridSizeMask.x + 1)) + ((gridpos.z + 1) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			return id;
			break;

		case 6:
			id.x = (gridpos.x) + ((gridpos.y + 1) * (gridSizeMask.x + 1) )+ ((gridpos.z + 1) * ((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			id.y = (gridpos.x + 1) + ((gridpos.y + 1) * (gridSizeMask.x + 1)) + ((gridpos.z + 1) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			return id;
			break;

		case 7:
			id.x = (gridpos.x) + ((gridpos.y) * (gridSizeMask.x + 1) )+ ((gridpos.z + 1) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			id.y = (gridpos.x) + ((gridpos.y + 1) * (gridSizeMask.x + 1)) + ((gridpos.z + 1) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			return id;
			break;

		case 8:
			id.x = (gridpos.x) + (gridpos.y * (gridSizeMask.x + 1) )+ ((gridpos.z) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			id.y = (gridpos.x) + (gridpos.y * (gridSizeMask.x + 1)) + ((gridpos.z + 1) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			return id;
			break;
		case 9:
			id.x = (gridpos.x + 1) + (gridpos.y * (gridSizeMask.x + 1) )+ ((gridpos.z) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			id.y = (gridpos.x + 1) + ((gridpos.y) * (gridSizeMask.x + 1)) + ((gridpos.z + 1) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			return id;
			break;

		case 10:
			id.x = (gridpos.x + 1) + ((gridpos.y + 1) * (gridSizeMask.x + 1) )+ ((gridpos.z ) * ((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			id.y = (gridpos.x + 1) + ((gridpos.y + 1) * (gridSizeMask.x + 1)) + ((gridpos.z + 1) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			return id;
			break;

		case 11:
			id.x = (gridpos.x) + ((gridpos.y + 1) * (gridSizeMask.x + 1) )+ ((gridpos.z ) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			id.y = (gridpos.x) + ((gridpos.y + 1) * (gridSizeMask.x + 1)) + ((gridpos.z + 1) *((gridSizeMask.x + 1) * (gridSizeMask.y + 1)));
			return id;
			break;
		
	}
	return id;
}


__device__ uint final_idx(uint2 indxx , float isoval, float *d_vol)
{
	

	
	float a = d_vol[indxx.x];
	float b = d_vol[indxx.y];
	if((a < isoval) && (b >= isoval))
	{
		return indxx.y;
	}
	else 
	{
		return indxx.x;
	}


}


__device__ int final_idx_val(uint2 indxx , grid_points *vol_one, float isoval, uint voxel , uint edge)
{
	

	
	int a = vol_one[indxx.x].val ;
	int b = vol_one[indxx.y].val ;


	if((a < isoval) && (b >= isoval))
	{
		return indxx.y;
	}
	else if((a >= isoval) && (b < isoval))
	{
		return indxx.x;
	}
	else
	{
		return -1;
	}


}


__device__ void face_to_grid(float a, uint2 indx1, uint2 indx2, uint2 indx3,grid_points *d_vol_one,uint voxel,uint edge_1,uint edge_2, uint edge_3, REAL *d_selection, float isoval)
{
	
	
	int3 v_index = {0,0,0};
	
	v_index.x = final_idx_val(indx1,d_vol_one,isoval, voxel, edge_1);

	v_index.y = final_idx_val(indx2,d_vol_one,isoval, voxel, edge_2);

	v_index.z = final_idx_val(indx3,d_vol_one,isoval, voxel, edge_3);


	if( a == -1.0)
	{
		if(v_index.x > -1)
		{
			d_selection[v_index.x] = -1.0;
		}
		if(v_index.y > -1)
		{
			d_selection[v_index.y] = -1.0;
		}
		if(v_index.z > -1)
		{
			d_selection[v_index.z] = -1.0;
		}
		
	}
	else if( a == 1.0)
	{
		if(v_index.x > -1)
		{
			d_selection[v_index.x] = 1.0;
		}
		if(v_index.y > -1)
		{
			d_selection[v_index.y] = 1.0;
		}
		if(v_index.z > -1)
		{
			d_selection[v_index.z] = 1.0;
		}

	}	
}



__global__ void facet_to_point_kernel(float *storagebuffer, triangle_metadata *triangle_data, uint active_facets, float3 *d_u, bool update_load,bool update_support, bool clear_load, bool clear_support,
uint3 gridSizeShift, uint3 gridSizeMask, uint Nx, uint Ny, grid_points *d_vol_one, float *d_selection, float isoval)
{

	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	
	

	if(tx < active_facets)
	{

		float a = storagebuffer[tx];

		uint voxel  = triangle_data[tx].voxel;
		uint edge_1 = triangle_data[tx].edge_1;
		uint edge_2 = triangle_data[tx].edge_2;
		uint edge_3 = triangle_data[tx].edge_3;
		uint l_index = triangle_data[tx].l_index;

		uint3 gridPos = calcGridPos_selection(voxel, gridSizeShift, gridSizeMask);

		uint2 indx1 = index_cal(edge_1,gridPos,gridSizeMask);
		uint2 indx2 = index_cal(edge_2,gridPos,gridSizeMask);
		uint2 indx3 = index_cal(edge_3,gridPos,gridSizeMask);


		if(l_index == 0)
		{
			
			face_to_grid(a,indx1,indx2,indx3,d_vol_one,voxel, edge_1, edge_2, edge_3, d_selection, isoval);
			
		}
		__syncthreads();

		if(l_index == 1)
		{
			
			face_to_grid(a,indx1,indx2,indx3,d_vol_one,voxel, edge_1, edge_2, edge_3, d_selection, isoval);
			
		}
		__syncthreads();

		if(l_index == 2)
		{
			
			face_to_grid(a,indx1,indx2,indx3,d_vol_one,voxel, edge_1, edge_2, edge_3, d_selection, isoval);
			
		}
		__syncthreads();

		if(l_index == 3)
		{
			
			face_to_grid(a,indx1,indx2,indx3,d_vol_one,voxel, edge_1, edge_2, edge_3, d_selection, isoval);
			
		}
		__syncthreads();

		if(l_index == 4)
		{
			
			face_to_grid(a,indx1,indx2,indx3,d_vol_one,voxel, edge_1, edge_2, edge_3, d_selection, isoval);
			
		}
		__syncthreads();


	}

}

void Selection::facet_to_points(float *storagebuffer, triangle_metadata *triangle_data, uint active_facets, float3 *d_u, bool update_load, bool update_support, bool clear_load, bool clear_support,
uint3 gridSizeShift, uint3 gridSizeMask, uint Nx, uint Ny, grid_points *d_vol_one, float *d_selection, float isoval)
{
	dim3 grids(ceil( (active_facets)/ 1024.0),1,1);
    dim3 tids(1024,1,1);
	facet_to_point_kernel<<<grids,tids>>>(storagebuffer, triangle_data,active_facets, d_u,update_load, update_support, clear_load, clear_support, gridSizeShift, gridSizeMask, Nx, Ny, d_vol_one, d_selection, isoval);
	cudaDeviceSynchronize();
}


__global__ void apply_to_lower_kernel(REAL *d_selection, REAL *d_selection2, grid_points *d_vol_one, int Nx, int Ny, int Nz, uint3 gridSizeMask, uint3 gridSizeShift, float isoval)
{
	int tx = blockIdx.x*blockDim.x + threadIdx.x;

	uint zz = tx / (Nx * Ny);
    uint z_rem = tx % (Nx * Ny);
    uint yy = (z_rem)/(Nx);
    uint xx = (z_rem) % (Nx);


	uint3 shift = make_uint3(1,Nx,Nx*Ny);
	uint3 mask = make_uint3(Nx,Ny,Nz);

	if(tx < (Nx * Ny * Nz) )
	{
		uint3 gridPos = calcGridPos_selection(tx, shift, mask);
		uint3 gridPosone = 2 * gridPos;




		uint indone = gridPosone.x  + (gridPosone.y * ((mask.x) * 2)) + (gridPosone.z * ((mask.x) * 2) * ((mask.y) * 2));

		float x0 = d_selection2[indone];
		float x1 = d_selection2[indone + 1];
		float y1 = d_selection2[indone + ((mask.x) * 2)];
		float z1 = d_selection2[indone + (((mask.x) * 2) * ((mask.y) * 2))];
		float xy = d_selection2[indone + ((mask.x) * 2) + 1];
		float xz = d_selection2[indone + (((mask.x) * 2) * ((mask.y) * 2)) + 1];
		float yz = d_selection2[indone + (((mask.x) * 2) * ((mask.y) * 2)) + ((mask.x) * 2)];
		float xyz = d_selection2[indone + (((mask.x) * 2) * ((mask.y) * 2)) + ((mask.x) * 2) + 1];

		if(x0 == -1.0)
		{
			d_selection[tx] = -1.0;
		
		}
		else if( x0 == 1)
		{
			d_selection[tx] = 1.0;
		}

	

		__syncthreads();

		if((xx < (Nx -1)))
		{
			if(fabs(x1) == 1)
			{
			
				int vx = d_vol_one[indone + 1].val;
				int vx_n = d_vol_one[indone + 2].val;

				if((vx > isoval) && (vx_n > isoval))
				{
					
					if(x1 == -1.0)
					{
						d_selection[tx + 1] = -1.0;
						
					}	
					else if(x1 == 1.0)
					{
						d_selection[tx + 1] = 1.0;
						
					}
				}
				else
				{
					if(x1 == -1.0)
					{
						d_selection[tx] = -1.0;

					}	
					else if(x1 == 1.0)
					{
						d_selection[tx] = 1.0;
				
					}
				}
			}
		}

		__syncthreads();

		if((yy < (Ny -1)))
		{
			if(fabs(y1) == 1)
			{
				
				int vy = d_vol_one[indone + (mask.x) * 2].val;
				int vy_n = d_vol_one[indone + (mask.x) * 2 + (mask.x) * 2].val;

				if((vy > isoval) && (vy_n > isoval))
				{
					
					if(y1 == -1.0)
					{
						d_selection[tx + (mask.x)] = -1.0;
						
					}	
					else if(y1 == 1.0)
					{
						d_selection[tx + (mask.x)] = 1.0;
						
					}
				}
				else
				{
					
					if(y1 == -1.0)
					{
						d_selection[tx ] = -1.0;
						
					}	
					else if(y1 == 1.0)
					{
						d_selection[tx ] = 1.0;
						
					}
				}
			}
		}

		__syncthreads();

		if((zz < (Nz -1 )))
		{
			if(fabs(z1) == 1)
			{
				float vz = d_vol_one[indone + (((mask.x) * 2) * ((mask.y) * 2))].val;
				float vz_n = d_vol_one[indone + (((mask.x) * 2) * ((mask.y) * 2)) + (((mask.x) * 2) * ((mask.y) * 2))].val;

				if((vz > isoval) && (vz_n > isoval))
				{
					
					if(z1 == -1.0)
					{
						d_selection[tx + (((mask.x)) * ((mask.y)))] = -1.0;
						
					}	
					else if(z1 == 1.0)
					{
						d_selection[tx + (((mask.x)) * ((mask.y)))] = 1.0;
						
					}
				}

				else
				{
					
					if(z1 == -1.0)
					{
						d_selection[tx] = -1.0;
						
					}	
					else if(z1 == 1.0)
					{
						d_selection[tx] = 1.0;
						
					}
				}
			}
		}

		__syncthreads();

		if((xx < (Nx -1)) && (yy < (Ny -1)))
		{
			if(fabs(xy) == 1)
			{
				float vxy = d_vol_one[indone + ((mask.x) * 2) + 1].val;
				float vxy_n = d_vol_one[indone + (((mask.x) * 2) * 2) + 2].val;


				if((vxy > isoval) && (vxy_n > isoval))
				{
					
					if(xy == -1.0)
					{
						d_selection[tx + (mask.x + 1)] = -1.0;
						
					}	
					else if(xy == 1.0)
					{
						d_selection[tx + (mask.x + 1)] = 1.0;
						
					}
				}

				else
				{
					
					if(xy == -1.0)
					{
						d_selection[tx] = -1.0;
						
					}	
					else if(xy == 1.0)
					{
						d_selection[tx] = 1.0;
						
					}
				}
			}
		}

		__syncthreads();


		if((xx < (Nx -1)) && (zz < (Nz -1)))
		{
			if(fabs(xz) == 1)
			{
				float vxz = d_vol_one[indone + (((mask.x) * 2) * ((mask.y) * 2)) + 1].val;
				float vxz_n = d_vol_one[indone + (((mask.x) * 2) * ((mask.y) * 2)) * 2 + 2].val;

				if((vxz > isoval) && (vxz_n > isoval))
				{
					
					if(xz == -1.0)
					{
						d_selection[tx + (mask.x * mask.y) + 1] = -1.0;
						
					}	
					else if(xz == 1.0)
					{
						d_selection[tx + (mask.x * mask.y) + 1] = 1.0;
						
					}
				}

				else
				{
					
					if(xz == -1.0)
					{
						d_selection[tx] = -1.0;
						
					}	
					else if(xz == 1.0)
					{
						d_selection[tx] = 1.0;
						
					}
				}
			}
		}

		__syncthreads();


		if((yy < (Ny -1)) && (zz < (Nz -1)))
		{
			if(fabs(yz) == 1)
			{
				float vyz = d_vol_one[indone + (((mask.x) * 2) * ((mask.y) * 2)) + ((mask.x) * 2)].val;
				float vyz_n = d_vol_one[indone + (((mask.x) * 2) * ((mask.y) * 2)) * 2 + ((mask.x) * 2) * 2].val;

				if((vyz > isoval) && (vyz_n > isoval))
				{
					
					if(yz == -1.0)
					{
						d_selection[tx + (mask.x * mask.y) + mask.x] = -1.0;
						
					}	
					else if(yz == 1.0)
					{
						d_selection[tx + (mask.x * mask.y) + mask.x] = 1.0;
						
					}
				}

				else
				{
					
					if(yz == -1.0)
					{
						d_selection[tx] = -1.0;
					}	
					else if(yz == 1.0)
					{
						d_selection[tx] = 1.0;
						
					}
				}
			}
		}

		__syncthreads();

		if((yy < (Ny -1)) && (zz < (Nz -1)) && (xx < (Nx -1)))
		{
			if(fabs(xyz) == 1)
			{
				float vxyz = d_vol_one[indone + (((mask.x) * 2) * ((mask.y) * 2)) + ((mask.x) * 2) + 1].val;
				float vxyz_n = d_vol_one[indone + (((mask.x) * 2) * ((mask.y) * 2)) *2 + ((mask.x) * 2) * 2  + 2].val;

				if((vxyz > isoval) && (vxyz_n > isoval))
				{
					
					if(xyz == -1.0)
					{
						d_selection[tx + (mask.x * mask.y) + mask.x + 1] = -1.0;
						
					}	
					else if(xyz == 1.0)
					{
						d_selection[tx + (mask.x * mask.y) + mask.x + 1] = 1.0;
						
					}
				}

				else
				{
					
					if(xyz == -1.0)
					{
						d_selection[tx] = -1.0;
					}	
					else if(xyz == 1.0)
					{
						d_selection[tx] = 1.0;
						
					}
				}
			}
		}

		__syncthreads();


	}
	
}

void Selection::apply_to_lower(REAL *d_selection, REAL *d_selection2, grid_points *d_vol_one, int Nx, int Ny, int Nz, uint3 gridSizeMask, uint3 gridSizeShift, float isoval)
{
	dim3 grids(ceil( (Nx*Ny*Nz)/ 1024.0),1,1);
    dim3 tids(1024,1,1);
	apply_to_lower_kernel<<<grids,tids>>>(d_selection,d_selection2,d_vol_one,Nx,Ny,Nz,gridSizeMask, gridSizeShift,isoval);
	cudaDeviceSynchronize();
}