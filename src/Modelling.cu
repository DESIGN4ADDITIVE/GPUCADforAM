#include "Modelling.h"
#include "helper_math.h"
#include <iostream>
#include <float.h>




float Modelling::a;
float Modelling::b;


Modelling::Modelling(int Nx, int Ny, int Nz)
{
    Modelling::gridsize = Nx*Ny*Nz;
	Modelling::a = 0.0;
	Modelling::b = 1.0;
}
Modelling::~Modelling()
{
    
}


__global__ void concentric_cylinder_kernel(float* data_1,float3 center,float radius, float thickness_radial, float thickness_axial, int Nx,int Ny, int Nz)
{
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int size = Nx*Ny*Nz;
    int zz = tx/(Nx*Ny);
    int yy = (tx%(Nx*Ny))/Ny;
    int xx = tx%Nx;
    float field_val1;
	float mean_x = (Nx-1)/2.0;
	float mean_y = (Ny-1)/2.0;
	float mean_z = (Nz-1)/2.0;
	xx = xx - mean_x;
	yy = yy - mean_y;
	zz = zz - mean_z;

    if(tx < size)
    {
        field_val1 = powf((xx - yy),2) + powf((yy - zz),2) + powf((zz - xx),2) - powf(3*radius,2);
        data_1[tx] = field_val1;
      
    }

}

void Modelling::concentric_cylinder(float* data_1,float3 center,float radius_1,float thickness_radial, float thickness_axial, int Nx,int Ny, int Nz)
{
    dim3 grids(ceil( (Nx*Ny*Nz)/ 1024.0),1,1);
    dim3 tids(1024,1,1);
    concentric_cylinder_kernel<<<grids,tids>>>(data_1,center,radius_1,thickness_radial,thickness_axial,Nx,Ny,Nz);
    cudaDeviceSynchronize();

}

void concentirc_square()
{

}

 __global__ void GPUScalar_normalise_kernel(float2 *d_result,float *d_vec1,int n)
{
	int tx = threadIdx.x;
	int ind = blockIdx.x*blockDim.x+tx;
	__shared__ float cc[1024];
	__shared__ float dd[1024];
	
	if (ind <n)
	{
		cc[tx] = d_vec1[ind];
		dd[tx] = d_vec1[ind];
	
	}
	
	__syncthreads();

	
	for(int stride = blockDim.x/2; stride>0; stride/=2)
	{
		
		if(tx < stride)
		{
			
			cc[tx] = min(cc[tx],cc[tx+stride]);
			dd[tx] = max(dd[tx],dd[tx+stride]);

		}
		__syncthreads();
	}
	

	if (tx ==0)
	{
		d_result[blockIdx.x].x = cc[tx];

		d_result[blockIdx.x].y = dd[tx];
		
	}

	__syncthreads();

}

__global__ void Min_reduction(float2 *d_DataIn,int block_num)
{
	__shared__ float2 sdata_min[1024];

	unsigned int tid = threadIdx.x;
	sdata_min[tid].x = 0;
	sdata_min[tid].y = 0;
	int index;
	int e;
	e = (block_num/1024) + (!(block_num%1024)?0:1);
	
	float c ;
	float d ;

	for (int k = 0; k< e;k++)
	{
		index = tid + k*1024;
		if(index < block_num)
		{
			c = sdata_min[tid].x;
			d = sdata_min[tid].y;		

			sdata_min[tid] = d_DataIn[index];
			
			sdata_min[tid].x = min(c,sdata_min[tid].x);
			sdata_min[tid].y = max(d,sdata_min[tid].y);
		}
	
	}

	
	__syncthreads();

	

	for(unsigned int s=blockDim.x/2; s>0;s/=2) 
	{
		
		
		if (tid < s) 
		{
			
			sdata_min[tid].x = min(sdata_min[tid].x,sdata_min[tid + s].x);
			sdata_min[tid].y = max(sdata_min[tid].y,sdata_min[tid + s].y);
			
		}
		__syncthreads();
	}

	
	if (tid == 0) 
	{
		d_DataIn[0] = sdata_min[0];
		
	}
	
}


__global__ void device_buffer_dual(float *dataone,float *datatwo,float *datathree, float a, float b,int NX, int NY, int NZ, float isovalue)
{
	
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int size = NX*NY*NZ;
	float j;
	float k;
	float l;
	float n;


	if(tx < size)
	{
		j = dataone[tx];
		

		l = datatwo[tx];

		n = datathree[tx];
	

		k = (j - a)/(b-a);

		
	
			if((j >= (isovalue)))
			{
				
				l = 1.0;
				n = k;
				
			}

			else
			{
				l = 0.0;
				n = k;
			}
			
	
		datatwo[tx] = l;
		datathree[tx] = n;
	
	}
}

void Modelling::GPU_buffer_normalise_dual(float *d_vec1,float *d_vec2,float *d_vec3, size_t size, int Nx, int Ny, int Nz, float isovalue)
{
	dim3 grids(ceil((size)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	int block_num = grids.x;


	float2 gb;
	
	float2 *d_Reduction_min;
	cudaMalloc((void **)&d_Reduction_min, sizeof(float2)* (block_num));

	cudaMemset(d_Reduction_min, 0.0, sizeof(float2)* (block_num));

	GPUScalar_normalise_kernel<<<grids,tids>>>(d_Reduction_min,d_vec1,size);
	cudaDeviceSynchronize();

	unsigned int  x_grid = 1;
	unsigned int  x_thread = 1024;
	
	Min_reduction<<<x_grid, x_thread>>>(d_Reduction_min,block_num);

	cudaDeviceSynchronize();

	cudaMemcpy(&gb, d_Reduction_min, sizeof(float2), cudaMemcpyDeviceToHost);

	float a = gb.x;
	float b = gb.y;
	cudaFree(d_Reduction_min);

	device_buffer_dual<<<grids,tids>>>(d_vec1,d_vec2,d_vec3,a,b, Nx, Ny, Nz, isovalue);
	cudaDeviceSynchronize();

}



__global__ void distance_from_line_kernel(float* data_1,float3 center,float3 axis,float radius, float thickness_radial, float thickness_axial, int Nx,int Ny, int Nz, bool cylind_disc_selected)
{
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int size = Nx*Ny*Nz;
    int zz = tx/(Nx*Ny);
    int yy = (tx%(Nx*Ny))/Ny;
    int xx = tx%Nx;
  
	float mean_x = (Nx-1)/2.0;
	float mean_y = (Ny-1)/2.0;
	float mean_z = (Nz-1)/2.0;

	__syncthreads();
	float x_1 = xx - mean_x;
	float y_1 = yy - mean_y;
	float z_1 = zz - mean_z;
    float3 field_vec = {x_1 - center.x,y_1 - center.y ,z_1 - center.z};

	float t_diff = thickness_radial/2.0;
	float t_diff_ax = thickness_axial/2.0;
	float axis_mag  = sqrtf(powf(axis.x,2)  + powf(axis.y,2)  + powf(axis.z,2));
	axis.x = (axis.x/axis_mag);
	axis.y = (axis.y/axis_mag);
	axis.z = (axis.z/axis_mag);
	float3 end = axis + center ;
	float fld_1, fld_2 = 0.0;
    if(tx < size)
    {
		
		float3 w1 = field_vec - center;
		float3 w2 = field_vec - end;
		float3 w3 = end - center;
	
		float3 d = cross(w1,w2);
		float e  = sqrtf(powf(d.x,2)  + powf(d.y,2)  + powf(d.z,2));
		float dis = (sqrtf(powf(w3.x,2) + powf(w3.y,2) + powf(w3.z,2)));
		float f  = e / dis;
		float3 cosines = {axis.x,axis.y,axis.z};
		float g = ((x_1 - center.x)*cosines.x + (y_1 - center.y)*cosines.y + (z_1 - center.z)*cosines.z);
		fld_1 = max(g - t_diff_ax, (g + t_diff_ax)*-1) ;

		if(cylind_disc_selected)
		{
			fld_2 = max((f - (radius + t_diff)),(f - (radius - t_diff))*-1.0);
		}
		else
		{
			fld_2 =(f - (radius ));
		}
	
		data_1[tx] = max(fld_1,fld_2);
	
		
    }

}

void Modelling::distance_from_line(float* data_1,float3 center,float3 axis,float radius_1,float thickness_radial,float thickness_axial,int Nx,int Ny, int Nz,bool cylind_disc_selected)
{
    dim3 grids(ceil( (Nx*Ny*Nz)/ 1024.0),1,1);
    dim3 tids(1024,1,1);
    distance_from_line_kernel<<<grids,tids>>>(data_1,center,axis,radius_1,thickness_radial,thickness_axial,Nx,Ny,Nz,cylind_disc_selected);
    cudaDeviceSynchronize();

}


__global__ void implicit_sphere_kernel(float* data_1,float3 center,float radius, float thickness_radial, int Nx,int Ny, int Nz, bool sphere_shell_selected)
{
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int size = Nx*Ny*Nz;
    int zz = tx/(Nx*Ny);
    int yy = (tx%(Nx*Ny))/Ny;
    int xx = tx%Nx;
  
	float mean_x = (Nx-1)/2.0;
	float mean_y = (Ny-1)/2.0;
	float mean_z = (Nz-1)/2.0;

	__syncthreads();
	float x_1 = xx - mean_x;
	float y_1 = yy - mean_y;
	float z_1 = zz - mean_z;
    float3 field_vec = {x_1 - center.x,y_1 - center.y ,z_1 - center.z};

	float t_diff = thickness_radial/2.0;

    if(tx < size)
    {

		if(sphere_shell_selected)
		{
			float fld_1 = powf(field_vec.x,2) + powf(field_vec.y,2) + powf(field_vec.z,2) - powf((radius - t_diff),2);

			
			float fld_2 = powf(field_vec.x,2) + powf(field_vec.y,2) + powf(field_vec.z,2) - powf((radius + t_diff),2);
		
			data_1[tx] = max(fld_1 * -1.0,fld_2);
		}

		else
		{
			float fld = powf(field_vec.x,2) + powf(field_vec.y,2) + powf(field_vec.z,2) - powf((radius),2);
			data_1[tx] = fld;
		}


	
		
    }

}



void Modelling::sphere_with_center(float* data_1,float3 center,float radius_1,float thickness_wall,int Nx,int Ny, int Nz,bool sphere_shell_selected)
{
    dim3 grids(ceil( (Nx*Ny*Nz)/ 1024.0),1,1);
    dim3 tids(1024,1,1);
    implicit_sphere_kernel<<<grids,tids>>>(data_1,center,radius_1,thickness_wall,Nx,Ny,Nz,sphere_shell_selected);
    cudaDeviceSynchronize();

}



__global__ void implicit_cuboid_kernel(float* data_1,float3 center,float3 angles,float x_width,float y_width,float z_width, int Nx, int Ny, int Nz)
{

	int tx = blockIdx.x*blockDim.x + threadIdx.x;

    int zz = tx/(Nx*Ny);
    int yy = (tx%(Nx*Ny))/Ny;
    int xx = tx%Nx;

	float mean_x = (Nx-1)/2.0;
	float mean_y = (Ny-1)/2.0;
	float mean_z = (Nz-1)/2.0;

	float x_wid = x_width/2.0;
	float y_wid = y_width/2.0;
	float z_wid = z_width/2.0;


	__syncthreads();
	
	float x_1 = xx - mean_x;
	float y_1 = yy - mean_y;
	float z_1 = zz - mean_z;

	float3 field_vec = {x_1 - center.x,y_1 - center.y ,z_1 - center.z};

	float3 pl_x = {(cosf(angles.x) * cosf(angles.y)),(sinf(angles.x) * cosf(angles.y)),(-1.0f * sinf(angles.y))};

	float3 pl_y = {((cosf(angles.x)) * sinf(angles.y) * sinf(angles.z)) - (sinf(angles.x) * cosf(angles.z)),
	(sinf(angles.x) * sinf(angles.y) * sinf(angles.z)) + (cosf(angles.x) * cosf(angles.z)),cosf(angles.y) * sinf(angles.z)};

	float3 pl_z = {(cosf(angles.x) * sinf(angles.y) * cosf(angles.z)) + (sinf(angles.x) * sinf(angles.z)),
	(sinf(angles.x) * sinf(angles.y) * cosf(angles.z)) - (cosf(angles.x) * sinf(angles.z)), cosf(angles.y) * cosf(angles.z)};


	float fld_1 = field_vec.x*pl_x.x +field_vec.y*pl_x.y + field_vec.z*pl_x.z ;
	fld_1 = max(fld_1 - x_wid,(fld_1 + x_wid)*-1.0);
	float fld_2 = field_vec.x*pl_y.x +field_vec.y*pl_y.y + field_vec.z*pl_y.z;
	fld_2 = max(fld_2 - y_wid,(fld_2 + y_wid)*-1.0);
	float fld_3 = field_vec.x*pl_z.x +field_vec.y*pl_z.y + field_vec.z*pl_z.z;
	fld_3 = max(fld_3 - z_wid,(fld_3 + z_wid)*-1.0);

	float fld = max(max(fld_1,fld_2),fld_3);

	data_1[tx] = fld;

}

void Modelling::cuboid(float* data_1,float3 center,float3 angles, float x_width,float y_width,float z_width, int Nx,int Ny, int Nz)
{
	dim3 grids(ceil( (Nx*Ny*Nz)/ 1024.0),1,1);
    dim3 tids(1024,1,1);

	implicit_cuboid_kernel<<<grids,tids>>>(data_1,center,angles,x_width,y_width,z_width,Nx,Ny,Nz);
    cudaDeviceSynchronize();

}



__global__ void implicit_cuboid_shell_kernel(float* data_1,float3 center,float3 angles,float x_width,float y_width,float z_width,float thickness, int Nx, int Ny, int Nz)
{

	int tx = blockIdx.x*blockDim.x + threadIdx.x;
    
    int zz = tx/(Nx*Ny);
    int yy = (tx%(Nx*Ny))/Ny;
    int xx = tx%Nx;

	float mean_x = (Nx-1)/2.0;
	float mean_y = (Ny-1)/2.0;
	float mean_z = (Nz-1)/2.0;

	float x_wid = x_width/2.0;
	float y_wid = y_width/2.0;
	float z_wid = z_width/2.0;


	__syncthreads();
	
	float x_1 = xx - mean_x;
	float y_1 = yy - mean_y;
	float z_1 = zz - mean_z;

	float3 field_vec = {x_1 - center.x,y_1 - center.y ,z_1 - center.z};

	float3 pl_x = {(cosf(angles.x) * cosf(angles.y)),(sinf(angles.x) * cosf(angles.y)),(-1.0f * sinf(angles.y))};

	float3 pl_y = {((cosf(angles.x)) * sinf(angles.y) * sinf(angles.z)) - (sinf(angles.x) * cosf(angles.z)),
	(sinf(angles.x) * sinf(angles.y) * sinf(angles.z)) + (cosf(angles.x) * cosf(angles.z)),cosf(angles.y) * sinf(angles.z)};

	float3 pl_z = {(cosf(angles.x) * sinf(angles.y) * cosf(angles.z)) + (sinf(angles.x) * sinf(angles.z)),
	(sinf(angles.x) * sinf(angles.y) * cosf(angles.z)) - (cosf(angles.x) * sinf(angles.z)), cosf(angles.y) * cosf(angles.z)};


	float fld_1 = field_vec.x*pl_x.x +field_vec.y*pl_x.y + field_vec.z*pl_x.z ;
	float fld_11 = abs(fld_1) - x_wid;
	float fld_12 = abs(fld_1) - (x_wid - thickness);
	
	float fld_2 = field_vec.x*pl_y.x +field_vec.y*pl_y.y + field_vec.z*pl_y.z;
	float fld_21 = abs(fld_2) - y_wid;
	float fld_22 = abs(fld_2) - (y_wid - thickness);


	float fld_3 = field_vec.x*pl_z.x +field_vec.y*pl_z.y + field_vec.z*pl_z.z;
	fld_3 = abs(fld_3) - z_wid;


	float fld = max(max(max(fld_11,fld_21),(max(fld_12,fld_22))*-1.0),fld_3);

	data_1[tx] = fld;

}





void Modelling::cuboid_shell(float* data_1,float3 center,float3 angles, float x_width,float y_width,float z_width,float thickness, int Nx,int Ny, int Nz)
{
	dim3 grids(ceil( (Nx*Ny*Nz)/ 1024.0),1,1);
    dim3 tids(1024,1,1);

	implicit_cuboid_shell_kernel<<<grids,tids>>>(data_1,center,angles,x_width,y_width,z_width,thickness,Nx,Ny,Nz);
    cudaDeviceSynchronize();

}


__global__ void implicit_torus_kernel(float* data_1,float3 center,float3 angles,float torus_radius, float torus_circle_radius, int Nx,int Ny, int Nz)
{
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int size = Nx*Ny*Nz;
    int zz = tx/(Nx*Ny);
    int yy = (tx%(Nx*Ny))/Ny;
    int xx = tx%Nx;
  
	float mean_x = (Nx-1)/2.0;
	float mean_y = (Ny-1)/2.0;
	float mean_z = (Nz-1)/2.0;

	__syncthreads();
	float x_1 = xx - mean_x;
	float y_1 = yy - mean_y;
	float z_1 = zz - mean_z;
    float3 field_vec = {x_1 - center.x,y_1 - center.y ,z_1 - center.z};

	float3 vec_field;

    if(tx < size)
    {

		float3 pl_x = {(cosf(angles.x) * cosf(angles.y)),(sinf(angles.x) * cosf(angles.y)),(-1.0f * sinf(angles.y))};

		float3 pl_y = {((cosf(angles.x)) * sinf(angles.y) * sinf(angles.z)) - (sinf(angles.x) * cosf(angles.z)),
		(sinf(angles.x) * sinf(angles.y) * sinf(angles.z)) + (cosf(angles.x) * cosf(angles.z)),cosf(angles.y) * sinf(angles.z)};

		float3 pl_z = {(cosf(angles.x) * sinf(angles.y) * cosf(angles.z)) + (sinf(angles.x) * sinf(angles.z)),
		(sinf(angles.x) * sinf(angles.y) * cosf(angles.z)) - (cosf(angles.x) * sinf(angles.z)), cosf(angles.y) * cosf(angles.z)};

		vec_field.x = field_vec.x*pl_x.x +field_vec.y*pl_x.y + field_vec.z*pl_x.z ;
		vec_field.y = field_vec.x*pl_y.x +field_vec.y*pl_y.y + field_vec.z*pl_y.z ;
		vec_field.z = field_vec.x*pl_z.x +field_vec.y*pl_z.y + field_vec.z*pl_z.z ;

		float side = (torus_radius - sqrtf(powf(vec_field.x,2) + powf(vec_field.y,2)));
		float fld_1 = powf(vec_field.z,2) - powf(torus_circle_radius,2) + powf(side,2);
		
	
		data_1[tx] = fld_1;
	
		
    }

}



void Modelling::torus_with_center(float* data_1,float3 center,float3 angles,float torus_radius,float torus_circle_radius,int Nx,int Ny, int Nz)
{
    dim3 grids(ceil( (Nx*Ny*Nz)/ 1024.0),1,1);
    dim3 tids(1024,1,1);
    implicit_torus_kernel<<<grids,tids>>>(data_1,center,angles,torus_radius,torus_circle_radius,Nx,Ny,Nz);
    cudaDeviceSynchronize();

}



__global__ void implicit_cone_kernel(float* data_1,float3 center,float3 angles,float base_radius, float cone_height, int Nx,int Ny, int Nz)
{
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int size = Nx*Ny*Nz;
    int zz = tx/(Nx*Ny);
    int yy = (tx%(Nx*Ny))/Ny;
    int xx = tx%Nx;
  
	float mean_x = (Nx-1)/2.0;
	float mean_y = (Ny-1)/2.0;
	float mean_z = (Nz-1)/2.0;

	__syncthreads();
	float x_1 = xx - mean_x;
	float y_1 = yy - mean_y;
	float z_1 = zz - mean_z;
    float3 field_vec = {x_1 - center.x,y_1 - center.y ,z_1 - center.z};

	float3 vec_field;

	float fld_2 = powf((cone_height/base_radius),2);


    if(tx < size)
    {

		

		///////////////////////////////////////////////////////////////
		float3 pl_x = {(cosf(angles.y) * cosf(angles.z)),(-1.0f*cosf(angles.y)*sinf(angles.z)),(sinf(angles.y))};

		float3 pl_y = {((sinf(angles.x)) * sinf(angles.y) * cosf(angles.z)) + (cosf(angles.x) * sinf(angles.z)),
		(-1.0f * sinf(angles.x) * sinf(angles.y) * sinf(angles.z)) + (cosf(angles.x) * cosf(angles.z)),-1.0f*sinf(angles.x) * cosf(angles.y)};

		float3 pl_z = {(-1.0f* cosf(angles.x) * sinf(angles.y) * cosf(angles.z)) + (sinf(angles.x) * sinf(angles.z)),
		(cosf(angles.x) * sinf(angles.y) * sinf(angles.z)) + (sinf(angles.x) * cosf(angles.z)), cosf(angles.x) * cosf(angles.y)};

		vec_field.x = field_vec.x*pl_x.x +field_vec.y*pl_x.y + field_vec.z*pl_x.z ;
		vec_field.y = field_vec.x*pl_y.x +field_vec.y*pl_y.y + field_vec.z*pl_y.z ;
		vec_field.z = field_vec.x*pl_z.x +field_vec.y*pl_z.y + field_vec.z*pl_z.z ;
		/////////////////////////////////////////////////////////////////////


		float g = (field_vec.x*pl_y.x + field_vec.y*pl_y.y + field_vec.z*pl_y.z);
		float h = max(((g-(cone_height/2.0)) - (cone_height/2.0)) *100,((g-(cone_height/2.0)) + (cone_height/2.0)) *100 * -1.0);
		float fld_1 = ((powf((vec_field.x),2) + powf((vec_field.z),2))*fld_2) - powf((vec_field.y - cone_height),2);
		
	
		data_1[tx] = max(fld_1,h);
    }

}



void Modelling::cone_with_base_radius_height(float* data_1,float3 center,float3 angles,float base_radius,float cone_height,int Nx,int Ny, int Nz)
{
    dim3 grids(ceil( (Nx*Ny*Nz)/ 1024.0),1,1);
    dim3 tids(1024,1,1);
    implicit_cone_kernel<<<grids,tids>>>(data_1,center,angles,base_radius,cone_height,Nx,Ny,Nz);
    cudaDeviceSynchronize();

}


__device__ float updated_grid_val(int ind,float b, float a, float iso, float b1, float a1)
{

	int v1 = 0;
	int v2 = 0;
	if((b < iso) && (b1 < iso))
	{
		v1 = 1;
	}
	
	if(((b >= iso) && (b1 < iso)) || ((b < iso) && (b1 >= iso)))
	{
		v1 = 1;
	}



	if(((a >= iso) && (a1 < iso)) || ((a < iso) && (a1 >= iso)))
	{

		v2 = 1;
	}




	if(((a < iso) && (a1 < iso)))
	{
		v2  = 1;
	}

	if(v1 && !v2)
	{
		return b;
	}

	else if (v2 && !v1)
	{
		return a;
	}

	else if(v1 && v2 )
	{
		return min(b,a);
		
	}

	else 
	{
		return min(b,a);
	}


	return FLT_MAX;

}


__global__ void retain_boundary_kernel(float* data_1, float* data_2,float* data_3,int Nx,int Ny, int Nz)
{
	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	int size = Nx*Ny*Nz;

	int xx = tx%Nx;
	int yy = (tx%(Nx*Ny))/Ny;
	int zz = tx/(Nx*Ny);

	float a;
	float x_1;
	float x_2;
	float y_1;
	float y_2;
	float z_1;
	float z_2;

	float xx_1;
	float xx_2;
	float yy_1;
	float yy_2;
	float zz_1;
	float zz_2;

	int ind;

	float b;
	float c;
	c = FLT_MAX;

	if(tx < size)
	{
		a = data_2[tx];
		b = data_1[tx];

		if(b == FLT_MAX)
		{
			data_3[tx] = a;
			return;
		}

		if(c == FLT_MAX)
		{
			
			ind = (xx - 1) + yy *Nx + zz *(Ny*Nx);
		
			if((ind >= 0 ) && (ind < size))
			{
				
				x_1 = data_2[ind];
				xx_1 = data_1[ind];
				
				c = updated_grid_val(ind,b,a,0.0,xx_1,x_1);
			
			}
		}
		
	

		if(c == FLT_MAX)
		{
			ind = (xx + 1) + yy *Nx + zz *(Ny*Nx);
		
			if((ind >= 0 ) && (ind < size))
			{
				x_2 = data_2[ind];
				xx_2 = data_1[ind];
				c = updated_grid_val(ind,b,a,0.0,xx_2,x_2);

			}
			
		}

	

		if(c == FLT_MAX)
		{
			ind = xx + (yy - 1) *Nx + zz *(Ny*Nx);
			if((ind >= 0 ) && (ind < size))
			{
				y_1 = data_2[ind];
				yy_1 = data_1[ind];
				c = updated_grid_val(ind,b,a,0.0,yy_1,y_1);

			}
		}

	
	

		if( c == FLT_MAX)
		{
			ind = xx + (yy + 1) *Nx + zz *(Ny*Nx);
			if((ind >= 0 ) && (ind < size))
			{
				y_2 = data_2[ind];
				yy_2 = data_1[ind];
				c = updated_grid_val(ind,b,a,0.0,yy_2,y_2);

			}
		}

	
		if( c == FLT_MAX)
		{
			ind = xx + yy *Nx + (zz - 1) *(Ny*Nx);
			if((ind >= 0 ) && (ind < size))
			{
				z_1 = data_2[ind];
				zz_1 = data_1[ind];
				c = updated_grid_val(ind,b,a,0.0,zz_1,z_1);

			}

		}


		if( c == FLT_MAX)
		{
			ind = xx + yy *Nx + (zz + 1) *(Ny*Nx);
			if((ind >= 0 ) && (ind < size))
			{
				z_2 = data_2[ind];
				zz_2 = data_1[ind];
				c = updated_grid_val(ind,b,a,0.0,zz_2,z_2);
		
			}
		}


	}

	__syncthreads();

	if(c != FLT_MAX)
	{
		
		data_3[tx] = c;
		
	}



}


void Modelling::retain_boundary(float* data_1, float* data_2,float* data_3,int Nx,int Ny, int Nz)
{
	
	dim3 grids(ceil( (Nx*Ny*Nz)/ 1024.0),1,1);
    dim3 tids(1024,1,1);
	
	retain_boundary_kernel<<<grids,tids>>>(data_1,data_2,data_3,Nx,Ny,Nz);
	cudaDeviceSynchronize();

}






__global__ void device_bufferfour(float *dataone,float *datatwo,float *datathree, float a, float b,int NX, int NY, int NZ, float isoval_1)
{
	
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 

	int size = NX*NY*NZ;
	float k;
	float m;

	

	if(tx < size)
	{
		k = dataone[tx];

			if((k >= (isoval_1)))
			{
				m = 1.0;
			}
			else
			{
				m = 0.0;
			}
		


		k = (k - a)/(b-a);

		datatwo[tx] = m;
		datathree[tx] = k;
	}
}





void Modelling::GPU_buffer_normalise_four(float *d_vec1,float *d_vec2,float *d_vec3, size_t size, int Nx, int Ny, int Nz, float isoval_1)
{
	dim3 grids(ceil((size)/float(1024)),1,1);
	dim3 tids(1024,1,1);

	int block_num = grids.x;


	float2 gb;
	
	float2 *d_Reduction_min;
	cudaMalloc((void **)&d_Reduction_min, sizeof(float2)* (block_num));

	cudaMemset(d_Reduction_min, 0.0, sizeof(float2)* (block_num));

	GPUScalar_normalise_kernel<<<grids,tids>>>(d_Reduction_min,d_vec1,size);
	cudaDeviceSynchronize();

	unsigned int  x_grid = 1;
	unsigned int  x_thread = 1024;
	
	Min_reduction<<<x_grid, x_thread>>>(d_Reduction_min,block_num);

	cudaDeviceSynchronize();

	cudaMemcpy(&gb, d_Reduction_min, sizeof(float2), cudaMemcpyDeviceToHost);

	float a = gb.x;
	float b = gb.y;
	cudaFree(d_Reduction_min);

	device_bufferfour<<<grids,tids>>>(d_vec1,d_vec2,d_vec3,a,b, Nx, Ny, Nz, isoval_1);
	cudaDeviceSynchronize();

}

__global__ void init_final_boundary_kernel(float* data_1,int NX,int NY, int NZ)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 
	int size = NX*NY*NZ;

	if(tx < size)
	{
		data_1[tx] = FLT_MAX;
	}


}

void Modelling::init_final_boundary(float* data_1,int Nx,int Ny, int Nz)
{
	
	dim3 grids(ceil( (Nx*Ny*Nz)/ 1024.0),1,1);
    dim3 tids(1024,1,1);
	
	init_final_boundary_kernel<<<grids,tids>>>(data_1,Nx,Ny,Nz);
	cudaDeviceSynchronize();

}





