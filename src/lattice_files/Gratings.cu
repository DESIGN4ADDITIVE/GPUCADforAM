
#include "Gratings.h"
#include <helper_cuda.h> 
#include <iostream>
#include <math.h>
#include <vector>
#include <cuComplex.h>

using namespace std;

cudaTextureObject_t     texObj;
cudaExtent array_extent;
static cudaArray *array = NULL;
extern size_t tPitch;
extern cudaExtent extend;



Gratings::Gratings()
{

}

 Gratings::~ Gratings()
{
  
}




__global__ void Reduction_lattice(float *d_DataIn, float *d_DataOut, int block_num)
{
	__shared__ float sdata[1024];

	for (int j=threadIdx.x; j<1024; j+=32*blockDim.x)  
	
	sdata[j]=0;

	unsigned int tid = threadIdx.x;

	int index;
	int e;
	e = (block_num/1024) + (!(block_num%1024)?0:1);
	
	float c = 0.0;

	for (int k = 0; k< e;k++)
	{
		index = tid + k*1024;
		if(index < block_num)
		{
			sdata[tid] = d_DataIn[index];
		
			c += sdata[tid];			
		}
		
	
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


__global__ void VecSMultAddKernel_lattice(float *d_v, const float a1, float *d_w, const float a2, const int NX, const int NY, const int NZ)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    
	int n = NX*NY*NZ;
	if(tx<n)
	{
	
		float V = d_v[tx];
		float W = d_w[tx];
		float result;
		//x(i+1) = 1.0*x(i) + 'alpha'*P(i)
		result = a1*V + a2*W;
		//x(i) = x(i+1) for the next iteration
		d_v[tx] = result;
		
		
	}

	
}


__global__ void finding_phi_kernel(float *d_phi, float *d_period,int x_dim,int y_dim,int z_dim,int i ,int j, int k,
float dx, float dy, float dz,  char latticetype_one, int uniform_type, float const_peirod, float x_period, float y_period, float z_period, float lconn, float lconn_1, bool sinewave_zaxis)
{


	int tx = blockIdx.x * blockDim.x + threadIdx.x; 

	float theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8;
	float per_1,per_2,per_3,per_4,per_5,per_6;
	float kx1,kx2,ky1,ky2,kz1,kz2;

	int n = x_dim*y_dim*z_dim;

	float a1, a2 , b1, b2, c1, c2;
	float xx,yy,zz;
	float mean_x,mean_y,mean_z;

	int x,y,z;
	int x1, x2, y1, y2, z1, z2;


	if (tx < n)
	{
		
		x = int(floorf((tx) % x_dim)); 
 
		
		y = int(floorf(((tx) % (x_dim * y_dim)) / y_dim));
	

		z = int(floorf(((tx) / (x_dim * y_dim))));
	

		if(x == 0)
		{
			a1 = -1;
			a2 = -0.5;
			x1 = x;
			x2 = x+1;
		}

		else if (x == 1)
		{   
			a1 = 1;
			a2 = -0.5;
			x1 = x-1;
			x2 = x+1 ;
		}

		else if (x == x_dim-2)
		{   
			a1 = 0.5;
			a2 = -1.0;
			x1 = x-1;
			x2 = x+1 ;
		}

		else if (x == x_dim -1)
		{   
			a1 = 0.5;
			a2 = 1;
			x1 = x-1;
			x2 = x ;
		}

		else 
		{   
			a1 = 0.5;
			a2 = -0.5;
			x1 = x-1;
			x2 = x+1 ;
		}


		if(y == 0)
		{
			b1 = -1;
			b2 = -0.5;
			y1 = y;
			y2 = y+1;
		}

		else if (y == 1)
		{   
			b1 = 1;
			b2 = -0.5;
			y1 = y-1;
			y2 = y+1 ;
		}

		else if (y == y_dim-2)
		{   
			b1 = 0.5;
			b2 = -1.0;
			y1 = y-1;
			y2 = y+1 ;
		}

		else if (y == y_dim -1)
		{   
			b1 = 0.5;
			b2 = 1;
			y1 = y-1;
			y2 = y ;
		}

		else 
		{   
			b1 = 0.5;
			b2 = -0.5;
			y1 = y-1;
			y2 = y+1 ;
		}



		if(z == 0)
		{
			c1 = -1;
			c2 = -0.5;
			z1 = z;
			z2 = z+1;
		}

		else if (z == 1)
		{   
			c1 = 1;
			c2 = -0.5;
			z1 = z-1;
			z2 = z+1 ;
		}

		else if (z == z_dim-2)
		{   
			c1 = 0.5;
			c2 = -1.0;
			z1 = z-1;
			z2 = z+1 ;
		}

		else if (z == z_dim -1)
		{   
			c1 = 0.5;
			c2 = 1;
			z1 = z-1;
			z2 = z ;
		}

		else 
		{   
			c1 = 0.5;
			c2 = -0.5;
			z1 = z-1;
			z2 = z+1 ;
		}

	
		float con = 1.0;
		float angl = 0.0;

		if(latticetype_one == 'n')
		{
		
			theta1 = 0.0;
			theta2 = 0.0;
			theta3 = 0.0;
			theta4 = 0.0;
		}
		else
		{
			
			if(latticetype_one == 'r')
			{
				mean_x = ((x_dim+1)/2.0f);
				mean_y = ((y_dim+1)/2.0f);
			

				con = 8.0;
				angl = 0.0;
				yy = ((y+1) - mean_y)*dx;
				float xx1 = ((x1+1) - mean_x)*dx;
				theta1 = atan2f(yy,xx1);
				float xx2 = ((x2+1) - mean_x)*dx;
				theta2 = atan2f(yy,xx2);


				xx = ((x+1) - mean_x) *dx;
				float yy1 = ((y1+1) - mean_y) * dx;
				theta3 = atan2f(yy1,xx);
				float yy2 = ((y2+1) - mean_y) * dx;
				theta4 = atan2f(yy2,xx);
			}
			else if(latticetype_one == 'b')
			{
				mean_x = 0;
				mean_y = 0;
				yy = ((y+1) - mean_y)*dx;
				float xx1 = ((x1+1) - mean_x)*dx;
				theta1 = atan2f(yy,xx1);
				float xx2 = ((x2+1) - mean_x)*dx;
				theta2 = atan2f(yy,xx2);


				xx = ((x+1) - mean_x) *dx;
				float yy1 = ((y1+1) - mean_y) * dx;
				theta3 = atan2f(yy1,xx);
				float yy2 = ((y2+1) - mean_y) * dx;
				theta4 = atan2f(yy2,xx);
			
			}
			else if(latticetype_one == 's')
			{
				mean_x = 0;
				mean_y = 0;
				mean_z = 0;

				float lcon = lconn;
				float lcon_1 = float(lconn_1);
				yy = ((y+1) - mean_y)*dx;
				float xx1 = ((x1+1) - mean_x)*dx;
				float zz1 = ((z1+1) - mean_z)*dz;
				theta1 = lcon*sinf(6.28 * lcon_1 * xx1);
				theta5 = lcon*sinf(6.28 * lcon_1 * zz1);
				float xx2 = ((x2+1) - mean_x)*dx;
				float zz2 = ((z2+1) - mean_z)*dz;
				theta2 = lcon*sinf(6.28 * lcon_1 * xx2);
				theta6 = lcon*sinf(6.28 * lcon_1 * zz2);


				xx = ((x+1) - mean_x) *dx;
				zz = ((z+1) - mean_z) *dz;
				
				theta3 = lcon * sinf(6.28 * lcon_1 * xx);
				theta7 = lcon * sinf(6.28 * lcon_1 * zz);
				
				theta4 = lcon * sinf(6.28 * lcon_1 * xx);
				theta8 = lcon * sinf(6.28 * lcon_1 * zz);
			}
			
			
	
		}
			
		
	
		if(uniform_type == 0)
		{
			per_1 = per_2 = per_3 = per_4 = per_5 = per_6 = const_peirod;
		}

		else if(uniform_type == 1)
		{
			per_1 = x_period;
			per_2 = x_period;
			per_3 = y_period;
			per_4 = y_period;
			per_5 = z_period;
			per_6 = z_period;
			
		}

		else if(uniform_type == 2)
		{
			int ind1 = x1 + y*x_dim + z*(x_dim *y_dim);
			int ind2 = x2 + y*x_dim + z*(x_dim *y_dim);
			int ind3 = x + y1*x_dim + z*(x_dim *y_dim);
			int ind4 = x + y2*x_dim + z*(x_dim *y_dim);
			int ind5 = x + y*x_dim + z1*(x_dim * y_dim);
			int ind6 = x + y*x_dim + z2*(x_dim * y_dim);

			per_1 = d_period[ind1];
			per_2 = d_period[ind2];

			per_3 = d_period[ind3];
			per_4 = d_period[ind4];


			per_5 = d_period[ind5];
			per_6 = d_period[ind6];
		}

		
		kx1 = ((2*M_PI)/per_1)*(i*cosf(theta1) - j *sinf(theta1));
		kx2 = ((2*M_PI)/per_2)*(i*cosf(theta2) - j *sinf(theta2));

		ky1 = ((2*M_PI)/per_3)*(i*sinf(theta3) + j *cosf(theta3));
		ky2 = ((2*M_PI)/per_4)*(i*sinf(theta4) + j *cosf(theta4));

		kz1 = ((2*M_PI)/per_5)*k;
		kz2 = ((2*M_PI)/per_6)*k;

		float phii = (a1*kx1 + a2 * kx2) + (b1 * ky1 + b2 * ky2 ) + (c1 *kz1 + c2 *kz2);

		if(latticetype_one == 's')
		{
			if(sinewave_zaxis)
			{
				kz1 = ((2*M_PI)/per_5)*(j*sinf(theta5) + k *cosf(con * theta5 - angl));
				kz2 = ((2*M_PI)/per_6)*(j*sinf(theta6) + k *cosf(con * theta6 - angl));

				ky1 = ((2*M_PI)/per_3)*(j*cosf(theta7) - k *sinf(con * theta7 + angl));
				ky2 = ((2*M_PI)/per_4)*(j*cosf(theta8) - k *sinf(con * theta8 + angl));

				kx1 = ((2*M_PI)/per_1)*i;
				kx2 = ((2*M_PI)/per_2)*i;

				phii += (a1*kx1 + a2 * kx2) + (b1 * ky1 + b2 * ky2 ) + (c1 *kz1 + c2 *kz2);
			}
		}
	
		__syncthreads();
	
		d_phi[tx] = phii;
		

	}

}


__global__ void GPUMatvec_lattice_kernel(float *d_d,float *d_q, int x_dim, int y_dim , int z_dim)
{

	int tx = blockIdx.x * blockDim.x + threadIdx.x; 

	int x, y, z;
	float x1,x2,x3,y1,y2,y3,z1,z2,z3;
	float phi1, phi2, phi3;
	float A_X;

	if (tx < x_dim * y_dim * z_dim)

	{
	
		phi1 = d_d[tx];
		x = int(floorf((tx)%x_dim));
		y = int(floorf(((tx)%(x_dim*y_dim)/y_dim)));
		z = int(floorf((tx)/(x_dim*y_dim)));
    
		if(x == 0)
		{
			phi2 = d_d[tx+1];
			phi3 = d_d[tx+2];
			
			x1 = (-1 * -1) + (-0.5 * -0.5);
			x2 = (-1 * 1) * phi2;
			x3 = (-0.5 * 0.5) * phi3;
		}

		else if(x == 1)
		{
			phi2 = d_d[tx-1];
			phi3 = d_d[tx+2];

			x1 = (1 * 1) + (-0.5 * -0.5);
			x2 = (-1 * 1) * phi2;
			x3 = (-0.5 * 0.5) * phi3;
		}

		else if(x == x_dim - 2 )
		{
			phi2 = d_d[tx-2];
			phi3 = d_d[tx+1];

			x1 = (0.5 * 0.5) + (-1 * -1);
			x2 = (-0.5 * 0.5) * phi2;
			x3 = (-1 * 1) * phi3;
		}

		else if(x == x_dim - 1 )
		{
			phi2 = d_d[tx-2];
			phi3 = d_d[tx-1];

			x1 = (0.5 * 0.5) + (1 * 1);
			x2 = (0.5 * -0.5) * phi2;
			x3 = (1 * -1) * phi3;
		}

		else
		{
			phi2 = d_d[tx-2];
			phi3 = d_d[tx+2];

			x1 = (0.5 * 0.5) + (-0.5 * -0.5);
			x2 = (0.5 * -0.5) * phi2;
			x3 = (-0.5 * 0.5) * phi3;
		}

		if(y == 0)
		{
			phi2 = d_d[tx+ x_dim];
			phi3 = d_d[tx+(2*x_dim)];
			
			y1 = (-1 * -1) + (-0.5 * -0.5);
			y2 = (-1 * 1) * phi2;
			y3 = (-0.5 * 0.5) * phi3;
		}

		else if(y == 1)
		{
			phi2 = d_d[tx-x_dim];
			phi3 = d_d[tx+(2*x_dim)];

			y1 = (1 * 1) + (-0.5 * -0.5);
			y2 = (-1 * 1) * phi2;
			y3 = (-0.5 * 0.5) * phi3;
		}

		else if(y == y_dim - 2 )
		{
			phi2 = d_d[tx-(2*x_dim)];
			phi3 = d_d[tx+(x_dim)];

			y1 = (0.5 * 0.5) + (-1 * -1);
			y2 = (-0.5 * 0.5) * phi2;
			y3 = (-1 * 1) * phi3;
		}

		else if(y == y_dim - 1 )
		{
			phi2 = d_d[tx-(2*x_dim)];
			phi3 = d_d[tx- x_dim];

			y1 = (0.5 * 0.5) + (1 * 1);
			y2 = (0.5 * -0.5) * phi2;
			y3 = (1 * -1) * phi3;
		}

		else
		{
			phi2 = d_d[tx-(2*x_dim)];
			phi3 = d_d[tx+(2*x_dim)];

			y1 = (0.5 * 0.5) + (-0.5 * -0.5);
			y2 = (0.5 * -0.5) * phi2;
			y3 = (-0.5 * 0.5) * phi3;
		}

		int x_y = x_dim*y_dim;

		if(z == 0)
		{
			phi2 = d_d[tx+(x_y)];
			phi3 = d_d[tx+(2*(x_y))];
			
			z1 = (-1 * -1) + (-0.5 * -0.5);
			z2 = (-1 * 1) * phi2;
			z3 = (-0.5 * 0.5) * phi3;
		}

		else if(z == 1)
		{
			phi2 = d_d[tx - x_y];
			phi3 = d_d[tx + 2*x_y];

			z1 = (1 * 1) + (-0.5 * -0.5);
			z2 = (-1 * 1) * phi2;
			z3 = (-0.5 * 0.5) * phi3;
		}

		else if(z == z_dim - 2 )
		{
			phi2 = d_d[tx - 2*(x_y)];
			phi3 = d_d[tx + x_y];

			z1 = (0.5 * 0.5) + (-1 * -1);
			z2 = (-0.5 * 0.5) * phi2;
			z3 = (-1 * 1) * phi3;
		}

		else if(z == z_dim - 1 )
		{
			phi2 = d_d[tx - 2*x_y];
			phi3 = d_d[tx - x_y];

			z1 = (0.5 * 0.5) + (1 * 1);
			z2 = (0.5 * -0.5) * phi2;
			z3 = (1 * -1) * phi3;
		}

		else
		{
			phi2 = d_d[tx-2*x_y];
			phi3 = d_d[tx+2*x_y];

			z1 = (0.5 * 0.5) + (-0.5 * -0.5);
			z2 = (0.5 * -0.5) * phi2;
			z3 = (-0.5 * 0.5) * phi3;
		}

		A_X = (x1+y1+z1) * phi1 + (x2) + (x3)  + (y2) + (y3) + (z2) + (z3);
		d_q[tx] = A_X;
	}

	__syncthreads();

}


__global__ void GPUScalar_lattice_kernel(float *d_result,float *d_vec1,float *d_vec2, int n)
{
	int tx = threadIdx.x;
	int ind = blockIdx.x*blockDim.x+tx;
	__shared__ float cc[1024];

	float a,b;
	
	float c = 0.0;

	
	if (ind <n)
	{
		a = d_vec1[ind];
		b = d_vec2[ind];
		c = a*b;
	
	}
	else
	{	
		c = 0.0;
	}
	


	cc[tx] = c;
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
	

	if (tx ==0)
	{
		d_result[blockIdx.x] = cc[tx];
		
	}

	__syncthreads();

}

__global__ void grating_kernel(float2 *dvol,int NX2,int NY2,int NZ2,float dx, float dy, float dz, cudaTextureObject_t texObj)
{
	
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z * blockDim.z + threadIdx.z;

	int indx = tx + ty*(NX2) + tz *(NX2*NY2);

	float x = tx*dx;
	float y = ty*dy;
	float z = tz*dz;

	

	if (tz < NZ2)
	{

		if(ty < NY2)
		{
			if (tx < NX2)
			{
				
				float b = tex3D<float>(texObj, (float)(x+0.5),(float)(y+0.5),(float)(z+0.5));
				
				dvol[indx].x = cosf(b);
				dvol[indx].y = sinf(b);

				__syncthreads();
				
			}
		}
	}

}

__global__ void refine_kernel(float *dvol,int NX2,int NY2,int NZ2,float dx, float dy, float dz, cudaTextureObject_t texObj)
{
	
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z * blockDim.z + threadIdx.z;

	int indx = tx + ty*(NX2) + tz *(NX2*NY2);

	float x = tx*dx;
	float y = ty*dy;
	float z = tz*dz;

	

	if (tz < NZ2)
	{

		if(ty < NY2)
		{
			if (tx < NX2)
			{
				
				float b = tex3D<float>(texObj, (float)(x+0.5),(float)(y+0.5),(float)(z+0.5));
				
				dvol[indx] = b;
			
				__syncthreads();
				
			}
		}
	}

}

__global__ void svl_kernel(float *d_svl,float2 *d_grating, int NX, int NY,int NZ,int indxx , float2 *data_fft)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;

	float2 a ;
	
	float2 c;

	if(tx < NX*NY*NZ)
	{
		a = d_grating[tx];
		float b = d_svl[tx];
		c = data_fft[indxx];

		float d = (a.x * c.x) - (a.y * (c.y));
		//float d = a.x;
		__syncthreads();
	

		d_svl[tx] =  b + d;
		//d_svl[tx] +=  d;


		__syncthreads();


	}

}


__global__ void fillfrac_kernel(float *d_svl,float *d_fillfrac,int NX, int NY,int NZ)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;

	float a ,b;
	float c;
	if(tx < NX*NY*NZ)
	{
		a = d_fillfrac[tx];
		b = d_svl[tx];
		c = a*b;
		d_svl[tx] = c;

	}

}




__global__ void set_theta_kernel(float *d_theta,int NX,int NY,int NZ,float dx,float dy, float dz, float mean_x,float mean_y,
float mean_z,char axis)
{

	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int size = NX*NY*NZ;
	int xx = tx%NX;
	int yy = (tx%(NX*NY))/NY;
	int zz = tx/(NX*NY);

	float x = (xx-mean_x)*dx;
	float y = (yy-mean_y)*dy;
	float z = (zz-mean_z)*dz;


	float angle = 0.0;
	if(tx < size)
	{

		if(axis == 'z')
		{
			angle = atan2f(y+1,x+1);

		}
		else if(axis == 'y')
		{
			angle = atan2f(z+1,x+1);
		}

		else
		{
			angle = atan2f(y+1,x+1);
		}

		__syncthreads();

		d_theta[tx] = angle;

	}

}


__global__ void set_period_kernel(float *d_period,int NX,int NY,int NZ,float dx,float dy, float dz,float mean_x, float mean_y, float mean_z,
char axis)
{

	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int size = NX*NY*NZ;
	int xx = tx%NX;
	int yy = (tx%(NX*NY))/NY;
	int zz = tx/(NX*NY);
	float period ;

	float x = (xx-mean_x)*dx;
	float y = (yy-mean_y)*dy;
	float z = (zz-mean_z)*dz;


	if(tx < size)
	{
		
		if(axis == 'z')
		{
			period = sqrtf(powf(x+1,2) + powf(y+1,2)) ;
	
		
		}
		else if(axis == 'y')
		{
			period = sqrtf(powf(x+1,2) + powf(z+1,2));
		}

		else
		{
			period = sqrtf(powf(z+1,2) + powf(y+1,2));
		}


		__syncthreads();

		d_period[tx] = period;

	}

}



__global__ void copytotexture_kernel(float * d_phi, cudaPitchedPtr data_ptr, int NX,int NY, int NZ)
{


	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z * blockDim.z + threadIdx.z;

	int indx = tx + ty*(NX) + tz *(NX*NY);

	char* devPtr = (char *) data_ptr.ptr;
	size_t pitch = data_ptr.pitch;
	size_t slicePitch = pitch * NY;

	if(tz < NZ)
	{
		char* slice = devPtr + tz * slicePitch;
		if(ty < NY)
		{

			float* row = (float*)(slice + ty * pitch);
			if (tx < NX)
			{
				float a = d_phi[indx];
				row[tx] = a ;
					
			}
		}
	}


}


__global__ void copytotexture_kernelone(float * d_phi, cudaPitchedPtr data_ptr, int NX,int NY, int NZ)
{


	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z * blockDim.z + threadIdx.z;

	int indx = tx + ty*(NX) + tz *(NX*NY);

	char* devPtr = (char *) data_ptr.ptr;
	size_t pitch = data_ptr.pitch;
	size_t slicePitch = pitch * NY;

	if(tz < NZ)
	{
		
		char* slice = devPtr + tz * slicePitch;
		if(ty < NY)
		{

			float* row = (float*)(slice + ty * pitch);
			if (tx < NX)
			{
				float a = d_phi[indx];
				row[tx] = a ;
					
			}
		}
	}


}




void Gratings::VecSMultAdd_lattice(float *d_v, float a1, float *d_w, const float a2, const int NX, const int NY, const int NZ)
{

	dim3 grids(ceil((NX*NY*NZ)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	VecSMultAddKernel_lattice<<<grids,tids>>>(d_v, a1, d_w, a2, NX,  NY,  NZ);
	cudaDeviceSynchronize();

}



void Gratings::GPUCG_lattice(float *d_phi,const int iter, const int OptIter, const float EndRes, int &FinalIter, float &FinalRes)
{
	
	unsigned int grid_size = NX*NY*NZ;
	int block_num = (grid_size/1024) + (!(grid_size % 1024) ? 0:1);
	float *d_d, *d_q, *d_res,*d_ResReduction_lattice;

	cudaMalloc((void **)&d_d, sizeof(float)* (grid_size));
	cudaMalloc((void **)&d_q, sizeof(float)* (grid_size));
	cudaMalloc((void **)&d_res, sizeof(float)* (grid_size));
	cudaMalloc((void **)&d_ResReduction_lattice, sizeof(float)* (block_num));

	cudaMemset(d_ResReduction_lattice, 0.0, sizeof(float)* (block_num));
	cudaMemset(d_q, 0.0, sizeof(float)* (grid_size));
	cudaMemset(d_res, 0.0, sizeof(float)* (grid_size));
	cudaMemset(d_d, 0.0, sizeof(float)* (grid_size));
	int iCounter = 1;


	cudaMemcpy(d_d, d_phi, sizeof(float)* (grid_size), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_res, d_phi, sizeof(float)* (grid_size), cudaMemcpyDeviceToDevice);
	cudaMemset(d_phi, 0.0, sizeof(float)* (grid_size));

	
	// computing r^t * r
	GPUScalar_lattice(d_ResReduction_lattice,d_res,d_res,grid_size,block_num);
	

	
	float g_ResBest;
	cudaMemcpy(&g_ResBest, d_ResReduction_lattice, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemset(d_ResReduction_lattice, 0.0, sizeof(float)* (block_num));
	g_ResBest = sqrt(g_ResBest);
	
	
// 	//compute r^T * r
	GPUScalar_lattice(d_ResReduction_lattice,d_res,d_d,grid_size,block_num);
	

	float g_delta_new;
	cudaMemcpy(&g_delta_new, d_ResReduction_lattice, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemset(d_ResReduction_lattice, 0.0, sizeof(float)* (block_num));
	const float term = EndRes*EndRes;
	

	while(iCounter < iter && g_delta_new > term)
	{
		//q=Ad
		GPUMatvec_lattice(d_d,d_q,NX,NY,NZ);

		float g_temp;

		GPUScalar_lattice(d_ResReduction_lattice, d_d, d_q,grid_size,block_num);

		

		cudaMemcpy(&g_temp, d_ResReduction_lattice, sizeof(float), cudaMemcpyDeviceToHost);

		cudaMemset(d_ResReduction_lattice, 0.0, sizeof(float)* (block_num));

		float g_alpha = g_delta_new/g_temp;

		VecSMultAdd_lattice(d_phi, 1.0, d_d, g_alpha, NX,  NY,  NZ);
	
		{
			VecSMultAdd_lattice(d_res, 1.0, d_q, -1.0*g_alpha, NX,  NY,  NZ);
		}

		float g_delta_old = g_delta_new;
		
		// //r(i+1)^T * r(i+1)
		GPUScalar_lattice(d_ResReduction_lattice, d_res, d_res,NX*NY*NZ,block_num);
	


		// // //g_delta_new = r(i+1)^T * r(i+1)
		cudaMemcpy(&g_delta_new, d_ResReduction_lattice, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemset(d_ResReduction_lattice, 0.0, sizeof(float)* (block_num));
		cudaDeviceSynchronize();

		// //'beta'(i) = (r(i+1)^T * r(i+1))/(r(i)^T*r(i))
		float g_beta = g_delta_new/g_delta_old;
		// // //P(i+1) = 'beta'(i)*d_d + 1.0* r(i+1)
		VecSMultAdd_lattice(d_d, g_beta, d_res, 1.0, NX,  NY,  NZ);
		cudaDeviceSynchronize();
		iCounter++;

	}
	
	FinalIter = iCounter;
	
	FinalRes = sqrt(g_delta_new);
	cudaFree(d_d);
	cudaFree(d_q);
	cudaFree(d_res);
	cudaFree(d_ResReduction_lattice);

	getLastCudaError("GPUCG failed");
	
}

void Gratings::grating(float2 *dvol,int NX2, int NY2, int NZ2,float dx, float dy, float dz)
{
	dim3 grids(ceil((NX2)/float(16)),ceil((NY2)/float(16)),ceil((NZ2)/float(4)));
	dim3 tids(16,16,4);
	grating_kernel<<<grids,tids>>>(dvol,NX2,NY2,NZ2,dx,dy,dz,texObj);
	cudaDeviceSynchronize();
}

void Gratings::refine(float *dvol,int NX2, int NY2, int NZ2,float dx, float dy, float dz)
{
	dim3 grids(ceil((NX2)/float(16)),ceil((NY2)/float(16)),ceil((NZ2)/float(4)));
	dim3 tids(16,16,4);
	refine_kernel<<<grids,tids>>>(dvol,NX2,NY2,NZ2,dx,dy,dz,texObj);
	cudaDeviceSynchronize();
}

void Gratings::svl(float *d_svl,float2 *d_grating,int NX, int NY, int NZ,int indxx, float2 *data_fft)
{
	dim3 grids(ceil((NX*NY*NZ)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	svl_kernel<<<grids,tids>>>(d_svl,d_grating,NX,NY,NZ,indxx,data_fft);
	cudaDeviceSynchronize();
}


void Gratings::fillfrac(float *d_svl, float *d_fillfrac, int NX ,int NY, int NZ)
{
	dim3 grids(ceil((NX*NY*NZ)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	fillfrac_kernel<<<grids,tids>>>(d_svl,d_fillfrac,NX,NY,NZ);
	cudaDeviceSynchronize();
}


void Gratings::GPUCleanUp()
{
	cudaFree(d_ResReduction);
}

void Gratings::finding_phi(float *d_phi ,float *d_period, int x_dim ,int y_dim, int z_dim, int i, int j, int k, float dx, float dy, float dz,
char latticetype_one , int unform_type, float const_peirod, float x_period, float y_period, float z_period, float lcon, float lcon_1, bool sinewave_zaxis)
{
	dim3 grids(ceil((x_dim*y_dim*z_dim)/float(1024)),1,1);
	dim3 tids(1024,1,1);

	finding_phi_kernel<<<grids,tids>>>(d_phi, d_period, x_dim, y_dim, z_dim, i, j, k, dx, dy, dz, latticetype_one,
	unform_type, const_peirod, x_period, y_period, z_period, lcon, lcon_1, sinewave_zaxis);
	cudaDeviceSynchronize();
};


void Gratings::GPUMatvec_lattice(float *d_d,float *d_q, int x_dim, int y_dim, int z_dim)
{
	dim3 grids(ceil((x_dim*y_dim*z_dim)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	GPUMatvec_lattice_kernel<<<grids,tids>>>(d_d,d_q,x_dim,y_dim,z_dim);
	cudaDeviceSynchronize();
}

void Gratings::GPUScalar_lattice(float *d_result,float *d_vec1,float *d_vec2,int n, int block_num)
{
	dim3 grids(ceil((n)/float(1024)),1,1);
	dim3 tids(1024,1,1);

	GPUScalar_lattice_kernel<<<grids,tids>>>(d_result,d_vec1,d_vec2,n);
	cudaDeviceSynchronize();

	unsigned int  x_grid = 1;
	unsigned int  x_thread = 1024;
	
	Reduction_lattice<<<x_grid, x_thread>>>(d_result, d_result,block_num);

	cudaDeviceSynchronize();
}


void Gratings::setupTexture(int x, int y ,int z)
{

    array_extent = make_cudaExtent(x, y, z);
                          
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    cudaMalloc3DArray(&array,&desc, array_extent);
    getLastCudaError("cudaMalloc failed ");

    cudaResourceDesc            texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));
    
    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = array;

    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL));
    
    
}


void Gratings::updateTexture(cudaPitchedPtr data_ptr)
{
    cudaMemcpy3DParms params ={0};
    params.srcPtr = data_ptr;
    params.dstArray = array;
    params.extent = array_extent;
    params.kind = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&params));
}

void Gratings::deleteTexture()
{
    checkCudaErrors(cudaDestroyTextureObject(texObj));
    checkCudaErrors(cudaFreeArray(array));
}

void Gratings::copytotexture(float *d_phi,cudaPitchedPtr data_ptr,int NX,int NY,int NZ)
{
	
	dim3 grids(ceil((NX)/float(16)),ceil((NY)/float(16)),ceil((NZ)/float(4)));
	dim3 tids(16,16,4);
	copytotexture_kernel<<<grids,tids>>>(d_phi,data_ptr,NX,NY,NZ);
	cudaDeviceSynchronize();
    getLastCudaError("copytotexture failed");
}

__global__ void device_buffer(float *dataone,float *datatwo, int size,float a, float b)
{
	
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 

	float k;

	if(tx < size)
	{
		k = dataone[tx];

		k = (k - a)/(b-a);
	
		datatwo[tx] = k;
		
	}
}


__global__ void device_bufferthree(float *dataone,float *datatwo, int size,float a, float b, float a1, float b1)
{
	
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 

	float k;

	if(tx < size)
	{
		k = dataone[tx];

		k = (k - a)/(b-a); 
	
		datatwo[tx] = a1 + (b1 * k );
		
	}
}

__global__ void device_bufferfour(float *dataone,float *datatwo,float *datathree, float a, float b,int NX, int NY, int NZ, float isoval_1,float isoval_2)
{
	
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 

	int xx = tx%NX;
	int yy = (tx%(NX*NY))/NY;
	int zz = tx/(NX*NY);
	int size = NX*NY*NZ;
	float k;
	float m;


	if(tx < size)
	{
		k = dataone[tx];


		k = (k - a)/(b-a);

		
	
		if ((xx == 0) || (xx == (NX-1)) || (yy == 0) || (yy == (NY-1)) || (zz == 0) || (zz == (NZ-1)))
		{
			
			m = 0.0;
			k = 0.0;

		}

		else
		{

			if((k >= (isoval_1)) && (k <= (isoval_2)))
			{
				m = 1.0;
			}
			else
			{
				m = 0.0;
			}
		
		}

		datatwo[tx] = m;
		datathree[tx] = k;

	}
}


__global__ void device_buffer_dual(float *dataone,float *datatwo,float *datathree,float *datafour, float a, float b,int NX, int NY, int NZ, float isovalue, float iso1, 
float iso2,bool union_val, bool inside, bool outside)
{
	
	int tx = blockIdx.x * blockDim.x + threadIdx.x;


	int xx = tx%NX;
	int yy = (tx%(NX*NY))/NY;
	int zz = tx/(NX*NY);
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


		if((k >= (iso1)) && (k <= (iso2)))
		{
			
				
			if ((xx == 0) || (xx == (NX-1)) || (yy == 0) || (yy == (NY-1)) || (zz == 0) || (zz == (NZ-1)))
			{
				
				l = 0;
				n = 0;
				k = 0;
		

			}
			
			else
			{
				l = 2.0;
				n = k;
			
			}
			
			
		}

		if(union_val == false)
		{
			if(inside && (k < (iso1)))
			{
				l = 0.0;
			
				
			}

			if(outside && (k > (iso2)))
			{
				l = 0.0;
				
			}
		}

		
		datatwo[tx] = l;
		datathree[tx] = n;
		datafour[tx] = k;

	}
}

void Gratings::normalise_buffer(float *dataone, float *datatwo, size_t size)
{
	float *h_B;
	h_B = (float *)malloc((size) * sizeof(*dataone));
	cudaMemcpy(h_B, dataone, (size) * sizeof(*dataone), cudaMemcpyDeviceToHost);
	float a,b;

	for (int i=0;i<size;i++)
	{
		if(i==0)
		{
			a = h_B[i];
			b = h_B[i];
		}
		
		a = min(a,h_B[i]);
		b = max(b,h_B[i]);
	}

	cudaMemcpy(datatwo, h_B, (size) * sizeof(*h_B), cudaMemcpyHostToDevice);
	free(h_B);
	dim3 grids(ceil((size)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	device_buffer<<<grids,tids>>>(dataone,datatwo,size,a,b);
	cudaDeviceSynchronize();
	
}




void Gratings::normalise_bufferone(float *dataone, float *datatwo, size_t size,int NX, int NY,int NZ)
{
	float *h_B;
	h_B = (float *)malloc((size) * sizeof(*dataone));
	cudaMemcpy(h_B, dataone, (size) * sizeof(*dataone), cudaMemcpyDeviceToHost);
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
	int c = (size/NZ);
	int d = size - (size/NZ);
	for(int i=0;i<size;i++)
	{
		if ((i < c ) || (i > d))
		{
			h_B[i] = 0.0;
 		}
		else if((i%c < NX) || (i%c > c - NX))
		{
			h_B[i] = 0.0;
		}
		else if(((i+1)%NX == 0) || ((i+1)%NX ==1))
		{
			h_B[i] = 0.0;
		}
		else
		{
			h_B[i] = (h_B[i] - a)/(b-a);
		}
	}

	cudaMemcpy(datatwo, h_B, (size) * sizeof(*h_B), cudaMemcpyHostToDevice);
	free(h_B);
}


void Gratings::normalise_buffertwo(float *dataone, float *datatwo, size_t size)
{
	float *h_B;
	h_B = (float *)malloc((size) * sizeof(*dataone));
	cudaMemcpy(h_B, dataone, (size) * sizeof(*dataone), cudaMemcpyDeviceToHost);
	float a,b,c;
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
		c = (h_B[i] - a)/(b-a);
		c = 1.0 - c;
		h_B[i] = 0.05 + (0.5 * c ); 

	}

	cudaMemcpy(datatwo, h_B, (size) * sizeof(*h_B), cudaMemcpyHostToDevice);
	free(h_B);
}


void Gratings::normalise_bufferthree(float *dataone, float *datatwo, size_t size, float a1, float b1)
{
	float *h_B;
	h_B = (float *)malloc((size) * sizeof(*dataone));
	cudaMemcpy(h_B, dataone, (size) * sizeof(*dataone), cudaMemcpyDeviceToHost);
	float a,b,c;
	for (int i=0;i<size;i++)
	{
		if(i==0)
		{
			a = h_B[i];
			b = h_B[i];
		
		}
		
		a = min(a,h_B[i]);
		b = max(b,h_B[i]);
		
		
	}
	for(int i=0;i<size;i++)
	{
		c = (h_B[i] - a)/(b-a);

		h_B[i] = a1 + (b1 * c ); 
	
	}

	cudaMemcpy(datatwo, h_B, (size) * sizeof(*h_B), cudaMemcpyHostToDevice);
	free(h_B);
}





 void Gratings::angle_data(float *d_theta,int NX, int NY,int NZ,float dx,float dy,float dz,float mean_x,float mean_y,
 float mean_z,char axis)
 {
	if((axis == 'x') || (axis == 'y') || (axis || 'z'))
	{
		dim3 grids(ceil((NX*NY*NZ)/float(1024)),1,1);
		dim3 tids(1024,1,1);
		set_theta_kernel<<<grids,tids>>>(d_theta,NX,NY,NZ,dx,dy,dz,mean_x,mean_y,mean_z,axis);
		
		cudaDeviceSynchronize();
	}
	else
	{
		printf(" X and Y axis rotation needs to defined \n");
	}
 }


 void Gratings::period_data(float *d_period,int NX ,int NY, int NZ,float dx, float dy, float dz,float mean_x,float mean_y,
 float mean_z, char axis)
 {
	if((axis == 'x') || (axis == 'y') || (axis || 'z'))
	{
		dim3 grids(ceil((NX*NY*NZ)/float(1024)),1,1);
		dim3 tids(1024,1,1);
		set_period_kernel<<<grids,tids>>>(d_period,NX,NY,NZ,dx,dy,dz,mean_x,mean_y,mean_z,axis);
		
		cudaDeviceSynchronize();
	}
	else
	{
		printf("Error Undefined Axis \n");
	}
 }


 __global__ void GPUScalar_normalise_kernel_lattice(float2 *d_result,float *d_vec1,int n)
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


__global__ void Min_reduction_lattice(float2 *d_DataIn,int block_num)
{
	__shared__ float2 sdata_min[1024];

	unsigned int tid = threadIdx.x;

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




void Gratings::GPU_buffer_normalise_buffer(float *d_vec1,float *d_vec2, int n)
{
	dim3 grids(ceil((n)/float(1024)),1,1);
	dim3 tids(1024,1,1);

	int block_num = grids.x;


	float2 gb;
	
	float2 *d_Reduction_min;
	cudaMalloc((void **)&d_Reduction_min, sizeof(float2)* (block_num));

	cudaMemset(d_Reduction_min, 0.0, sizeof(float2)* (block_num));

	GPUScalar_normalise_kernel_lattice<<<grids,tids>>>(d_Reduction_min,d_vec1,n);
	cudaDeviceSynchronize();

	unsigned int  x_grid = 1;

	unsigned int  x_thread = 1024;
	
	Min_reduction_lattice<<<x_grid, x_thread>>>(d_Reduction_min,block_num);

	cudaDeviceSynchronize();

	cudaMemcpy(&gb, d_Reduction_min, sizeof(float2), cudaMemcpyDeviceToHost);

	float a = gb.x;
	float b = gb.y;

	cudaFree(d_Reduction_min);

	device_buffer<<<grids,tids>>>(d_vec1,d_vec2,n,a,b);

	cudaDeviceSynchronize();

}

void Gratings::GPU_buffer_normalise_three(float *dataone, float *datatwo, size_t size, float a1, float b1)
{
	dim3 grids(ceil((size)/float(1024)),1,1);
	dim3 tids(1024,1,1);

	int block_num = grids.x;

	float2 gb;
	
	float2 *d_Reduction_min;
	cudaMalloc((void **)&d_Reduction_min, sizeof(float2)* (block_num));

	cudaMemset(d_Reduction_min, 0.0, sizeof(float2)* (block_num));

	GPUScalar_normalise_kernel_lattice<<<grids,tids>>>(d_Reduction_min,dataone,size);
	cudaDeviceSynchronize();

	unsigned int  x_grid = 1;
	unsigned int  x_thread = 1024;
	
	Min_reduction_lattice<<<x_grid, x_thread>>>(d_Reduction_min,block_num);

	cudaDeviceSynchronize();

	cudaMemcpy(&gb, d_Reduction_min, sizeof(float2), cudaMemcpyDeviceToHost);

	cudaFree(d_Reduction_min);

	
	device_bufferthree<<<grids,tids>>>(dataone,datatwo,size,gb.x,gb.y,a1,b1);

	cudaDeviceSynchronize();

}






void Gratings::GPU_buffer_normalise_four(float *d_vec1,float *d_vec2,float *d_vec3, size_t size, int Nx, int Ny, int Nz, float isoval_1,float isoval_2)
{
	dim3 grids(ceil((size)/float(1024)),1,1);

	dim3 tids(1024,1,1);

	int block_num = grids.x;

	float2 gb;
	
	float2 *d_Reduction_min;

	cudaMalloc((void **)&d_Reduction_min, sizeof(float2)* (block_num));

	cudaMemset(d_Reduction_min, 0.0, sizeof(float2)* (block_num));

	GPUScalar_normalise_kernel_lattice<<<grids,tids>>>(d_Reduction_min,d_vec1,size);
	cudaDeviceSynchronize();

	unsigned int  x_grid = 1;
	unsigned int  x_thread = 1024;
	
	Min_reduction_lattice<<<x_grid, x_thread>>>(d_Reduction_min,block_num);

	cudaDeviceSynchronize();

	cudaMemcpy(&gb, d_Reduction_min, sizeof(float2), cudaMemcpyDeviceToHost);

	float a = gb.x;

	float b = gb.y;

	cudaFree(d_Reduction_min);

	device_bufferfour<<<grids,tids>>>(d_vec1,d_vec2,d_vec3,a,b, Nx, Ny, Nz, isoval_1,isoval_2);

	cudaDeviceSynchronize();

}



void Gratings::GPU_buffer_normalise_dual(float *d_vec1,float *d_vec2,float *d_vec3,float *d_vec4, size_t size, int Nx, int Ny, int Nz, float isovalue,float& iso1, float& iso2,
float& thickness,float radius, bool union_val, bool inside, bool outside)
{
	dim3 grids(ceil((size)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	int block_num = grids.x;


	float2 gb;
	
	float2 *d_Reduction_min;
	cudaMalloc((void **)&d_Reduction_min, sizeof(float2)* (block_num));

	cudaMemset(d_Reduction_min, 0.0, sizeof(float2)* (block_num));

	GPUScalar_normalise_kernel_lattice<<<grids,tids>>>(d_Reduction_min,d_vec1,size);
	cudaDeviceSynchronize();

	unsigned int  x_grid = 1;
	unsigned int  x_thread = 1024;
	
	Min_reduction_lattice<<<x_grid, x_thread>>>(d_Reduction_min,block_num);

	cudaDeviceSynchronize();

	cudaMemcpy(&gb, d_Reduction_min, sizeof(float2), cudaMemcpyDeviceToHost);

	float a = gb.x;
	float b = gb.y;
	cudaFree(d_Reduction_min);

	float c1 = pow(((radius) - thickness), 2) - pow(((radius)), 2);
	float c2 = pow(((radius) + thickness), 2) - pow(((radius)), 2);

	iso1 = (c1 - a)/(b - a);
	iso2 = (c2 - a)/(b - a);

	device_buffer_dual<<<grids,tids>>>(d_vec1,d_vec2,d_vec3,d_vec4,a,b, Nx, Ny, Nz, isovalue,iso1,iso2,union_val,inside,outside);
	cudaDeviceSynchronize();

}




__global__ void topo_field_kernel(float *topo_field, float *isourf, float volfrac, uint size)
{
	int tx = threadIdx.x;
	int ind = blockIdx.x*blockDim.x+tx;
	float a = 0.0;
	if(ind < size )
	{
		a = topo_field[ind];
		if(a < volfrac)
		{
			isourf[ind] = 0.0;
		}

	}

}



void Gratings::topo_field(float *topo_field,float *isosurf,float volfrac, int NX ,int NY, int NZ)
{
	uint sizee = NX*NY*NZ;
	dim3 grids(ceil((sizee)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	topo_field_kernel<<<grids,tids>>>(topo_field,isosurf,volfrac,sizee);
	cudaDeviceSynchronize();
}


__global__ void primitive_field_kernel(grid_points *primitive_field,float *primitive_active, float *isourf, float isoval, uint size, bool fixed, bool dynamic)
{
	int tx = threadIdx.x;
	int ind = blockIdx.x*blockDim.x+tx;
	float a = 0.0;
	float b = 0.0;
	
	if(ind < size )
	{	
		if(fixed)
		{
			a = primitive_field[ind].val;
			if(a > -1)
			{
				isourf[ind] = __FLT_MAX__ ;
			}
		}

		else if(dynamic)
		{
			b = primitive_active[ind];
			if(b >= 0)
			{
				isourf[ind] = __FLT_MAX__ ;
			}
			
		}

	}

}

void Gratings::primitive_field(grid_points *primitive_field,float *primitive_active, float *isosurf,float isoval,bool fixed, bool active, int NX ,int NY, int NZ)
{
	uint sizee = NX*NY*NZ;
	dim3 grids(ceil((sizee)/float(1024)),1,1);
	dim3 tids(1024,1,1);
	primitive_field_kernel<<<grids,tids>>>(primitive_field,primitive_active,isosurf,isoval,sizee,fixed, active);
	cudaDeviceSynchronize();
}





