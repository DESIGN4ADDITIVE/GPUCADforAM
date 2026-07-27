
#include "Interpolations.h"

#include "cuda_runtime_api.h"

cudaExtent array_extent;
static cudaArray *array = NULL;



cudaArray_t array_3d;
cudaExtent array_extent3d;



void Interpolations::deleteTexture()
{
    checkCudaErrors(cudaDestroyTextureObject(texObj));
    checkCudaErrors(cudaFreeArray(array));
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


void Interpolations::copytotexture(float *d_phi,cudaPitchedPtr data_ptr,int NX,int NY,int NZ)
{
	
	dim3 grids(ceil((NX)/float(16)),ceil((NY)/float(16)),ceil((NZ)/float(4)));
	dim3 tids(16,16,4);
	copytotexture_kernel<<<grids,tids>>>(d_phi,data_ptr,NX,NY,NZ);
	cudaDeviceSynchronize();
    getLastCudaError("copytotexture failed");
}


void Interpolations::updateTexture(cudaPitchedPtr data_ptr)
{
    cudaMemcpy3DParms params ={0};
    params.srcPtr = data_ptr;
    params.dstArray = array;
    params.extent = array_extent;
    params.kind = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&params));
}


void Interpolations::setupTexture(int x, int y ,int z)
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



__global__ void copytotexture_results_kernel(float3 * d_displacemnt, cudaPitchedPtr data_ptr, int NX,int NY, int NZ, bool x_result, bool y_result, bool z_result)
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
				
				
				if(x_result)
				{
					float a = d_displacemnt[indx].x;
					row[tx] = a;
				}
				else if(y_result)
				{
					float b = d_displacemnt[indx].y;
					row[tx] = b ;
				}
				else if(z_result)
				{
					float c = d_displacemnt[indx].z;
					row[tx] = c ;
				}
				else
				{
					float3 disp = d_displacemnt[indx];
					float mag = sqrt(pow(disp.x,2) + pow(disp.y,2) + pow(disp.z,2));
					row[tx] = mag ;
				}

					
			}
		}
	}


}


void Interpolations::copytotexture_results(float3 *d_displacement,cudaPitchedPtr data_ptr,int NX,int NY,int NZ, bool x_result, bool y_result, bool z_result)
{
	
	dim3 grids(ceil((NX)/float(16)),ceil((NY)/float(16)),ceil((NZ)/float(4)));
	dim3 tids(16,16,4);
	copytotexture_results_kernel<<<grids,tids>>>(d_displacement,data_ptr,NX,NY,NZ, x_result, y_result, z_result);
	cudaDeviceSynchronize();
    getLastCudaError("copytotexture result failed");
}




__global__ void copytotexture_3d_results_kernel(float3 * d_displacemnt, cudaPitchedPtr data_ptr, int NX,int NY, int NZ)
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

			float4* row = (float4*)(slice + ty * pitch);
			if (tx < NX)
			{
				
				float3 aa = d_displacemnt[indx];
				row[tx] = make_float4(aa.x,aa.y,aa.z,1.0f);
					
			}
		}
	}


}



void Interpolations::copytotexture_3d_results(float3 *d_displacement,cudaPitchedPtr data_ptr,int NX,int NY,int NZ)
{
	
	dim3 grids(ceil((NX)/float(16)),ceil((NY)/float(16)),ceil((NZ)/float(4)));
	dim3 tids(16,16,4);
	copytotexture_3d_results_kernel<<<grids,tids>>>(d_displacement,data_ptr,NX,NY,NZ);
	cudaDeviceSynchronize();
    getLastCudaError("copytotexture_3d result failed");
}

//////////////////////////////////////////////////////////////////////////////////////////////





void Interpolations::delete_3dTexture()
{
    checkCudaErrors(cudaDestroyTextureObject(texObj3d));
    checkCudaErrors(cudaFreeArray(array_3d));
}

void Interpolations::setup_3DTexture(int x, int y ,int z)
{

    array_extent3d = make_cudaExtent(x, y, z);
                          
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();

    cudaMalloc3DArray(&array_3d,&desc, array_extent3d);
    getLastCudaError("cudaMalloc failed ");

    cudaResourceDesc            resDesc;
    memset(&resDesc,0,sizeof(cudaResourceDesc));
    resDesc.resType            = cudaResourceTypeArray;
    resDesc.res.array.array    = array_3d;


    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModeLinear;

	texDescr.addressMode[0] = cudaAddressModeClamp; 
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&texObj3d, &resDesc, &texDescr, NULL));
    
    
}


void Interpolations::update_3dTexture(cudaPitchedPtr data_ptr)
{
    cudaMemcpy3DParms params3d ={0};
    params3d.srcPtr = data_ptr;
    params3d.dstArray = array_3d;
    params3d.extent = array_extent3d;
    params3d.kind = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&params3d));
}


__global__ void refine_3d_kernel(float4 *disp,int NX2,int NY2,int NZ2,float dx, float dy, float dz,
	float *d_result,bool x_val, bool y_val, bool z_val, cudaTextureObject_t texObj3d)
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
				
				float4 b = tex3D<float4>(texObj3d, (float)(x+0.5),(float)(y+0.5),(float)(z+0.5));
				if(x_val)
				{
					d_result[indx] = b.x;
				}
				else if(y_val)
				{
					d_result[indx] = b.y;
				}
				else if(z_val)
				{
					d_result[indx] = b.z;
				}
				else
				{
					d_result[indx] =  sqrt(pow(b.x - x,2) + pow(b.y - y,2) + pow(b.z - z,2));
				}
				
				disp[indx] = b;
				
				__syncthreads();
				
			}
		}
	}

}


void Interpolations::refine_3d(float4 *disp,int NX2, int NY2, int NZ2,float dx, float dy, float dz,
float *d_result, bool x_val, bool y_val, bool z_val)
{
	dim3 grids(ceil((NX2)/float(16)),ceil((NY2)/float(16)),ceil((NZ2)/float(4)));
	dim3 tids(16,16,4);
	refine_3d_kernel<<<grids,tids>>>(disp,NX2,NY2,NZ2,dx,dy,dz,d_result,x_val,y_val,z_val,texObj3d);
	cudaDeviceSynchronize();
	getLastCudaError("refine 3D failed");
}




 __global__ void GPUScalar_normalise_kernel_interpolate(float *d_result_max,float *d_result_min,float4 *disp,int n)
{
	int tx = threadIdx.x;
	int ind = blockIdx.x*blockDim.x+tx;
	__shared__ float3 cc[1024];
	__shared__ float3 dd[1024];
	
	float4 d_disp;
	float3 s_disp;

	
	if (ind < n)
	{
		d_disp = disp[ind];
		s_disp = {d_disp.x,d_disp.y,d_disp.z};
		cc[tx] = s_disp;
		dd[tx] = s_disp;
	
	}
	
	__syncthreads();

	
	for(int stride = blockDim.x/2; stride>0; stride/=2)
	{
		
		if(tx < stride)
		{
			
			cc[tx].x = min(cc[tx].x,cc[tx+stride].x);
			cc[tx].y = min(cc[tx].y,cc[tx+stride].y);
			cc[tx].z = min(cc[tx].z,cc[tx+stride].z);

			dd[tx].x = max(dd[tx].x,dd[tx+stride].x);
			dd[tx].y = max(dd[tx].y,dd[tx+stride].y);
			dd[tx].z = max(dd[tx].z,dd[tx+stride].z);

		}
		__syncthreads();
	}
	

	if (tx == 0)
	{
		d_result_min[blockIdx.x] = min(cc[tx].x,min(cc[tx].y,cc[tx].z));

		d_result_max[blockIdx.x] = max(dd[tx].x,max(dd[tx].y,dd[tx].z));
		
	}

	__syncthreads();

}


__global__ void Min_reduction_interpolate(float *d_Data_min,float *d_Data_max,int block_num)
{
	__shared__ float sdata_min[1024];
	__shared__ float sdata_max[1024];

    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        sdata_min[i] = 0.0f;
		sdata_max[i] = 0.0f;
    }
    __syncthreads(); 

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
			c = sdata_min[tid];
			d = sdata_max[tid];		

			sdata_min[tid] = d_Data_min[index];
			sdata_max[tid] = d_Data_max[index];
			
			sdata_min[tid] = min(c,sdata_min[tid]);

			sdata_max[tid] = max(d,sdata_max[tid]);
		
		}

	}

	__syncthreads();

	for(unsigned int s=blockDim.x/2; s>0;s/=2) 
	{
		
		if (tid < s) 
		{
			
			sdata_min[tid] = min(sdata_min[tid],sdata_min[tid + s]);
		

			sdata_max[tid] = max(sdata_max[tid],sdata_max[tid + s]);
			
			
		}
		__syncthreads();
	}

	
	if (tid == 0) 
	{
		d_Data_min[0] = sdata_min[0];
		d_Data_max[0] = sdata_max[0];
	}
	
}

__global__ void device_buffer_interpolate(float4 *dataone,float4 *datatwo, float *d_selection, int Nx, int Ny, int Nz,float a, float b)
{
	
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 

	int size = Nx * Ny * Nz;

	float4 k = {0.0f,0.0f,0.0f,0.0f};

	float w;

	if(tx < size)
	{
		k = dataone[tx];
		w = d_selection[tx];

		if(w == -1.0f)
		{
			k = {0.0f,0.0f,0.0f,w};
		}
		else
		{
			
			k.x = (2 * ((k.x - a)/(b - a))) - 1.0f;
			k.y = (2 * ((k.y - a)/(b - a))) - 1.0f;
			k.z = (2 * ((k.z - a)/(b - a))) - 1.0f;
			k.w = 0.0f;
		}

		datatwo[tx] = k;
		
	}
}


__global__ void grid_displacement_kernel(float3 *d_grid_pos, float3 *grid_disp, int Nx, int Ny, int Nz, bool disp_grid)
{
	uint n = Nx*Ny*Nz;


	int tx = threadIdx.x;
	int ind = blockIdx.x*blockDim.x+tx;

	uint z = ind/(Nx *Ny);
	uint y = (ind%(Nx *Ny))/Nx;
	uint x = (ind%(Nx *Ny))%Nx;


	
	float3 grid_pos = {0.0f,0.0f,0.0f};

	if(ind < n)
	{
		float3 disp_val = grid_disp[ind];
		if(disp_grid)
		{
			
			grid_pos.x = x + (disp_val.x);
			grid_pos.y = y + (disp_val.y);
			grid_pos.z = z + (disp_val.z);

		}
		else
		{
			grid_pos.x = x;
			grid_pos.y = y;
			grid_pos.z = z;
		}

		d_grid_pos[ind] = grid_pos;
	}
}



void Interpolations::grid_displacement(float3 *d_grid_pos, float3 *grid_disp, int Nx, int Ny, int Nz, bool disp_grid)
{
	dim3 grids(ceil((Nx * Ny * Nz)/float(1024)),1,1);

	dim3 tids(1024,1,1);

	grid_displacement_kernel<<<grids,tids>>>(d_grid_pos,grid_disp,Nx,Ny,Nz,disp_grid);

	cudaDeviceSynchronize();
	
	getLastCudaError("grid_displacement Failed ");
}




 __global__ void displacement_max_min_kernel(float *d_result_max,float *d_result_min,float3 *disp,int n)
{
	int tx = threadIdx.x;
	int ind = blockIdx.x*blockDim.x+tx;
	__shared__ float3 cc[1024];
	__shared__ float3 dd[1024];
	
	float3 s_disp;

	
	if (ind < n)
	{
		s_disp = disp[ind];
		cc[tx] = s_disp;
		dd[tx] = s_disp;
	
	}
	
	__syncthreads();

	
	for(int stride = blockDim.x/2; stride>0; stride/=2)
	{
		
		if(tx < stride)
		{
			
			cc[tx].x = min(cc[tx].x,cc[tx+stride].x);
			cc[tx].y = min(cc[tx].y,cc[tx+stride].y);
			cc[tx].z = min(cc[tx].z,cc[tx+stride].z);

			dd[tx].x = max(dd[tx].x,dd[tx+stride].x);
			dd[tx].y = max(dd[tx].y,dd[tx+stride].y);
			dd[tx].z = max(dd[tx].z,dd[tx+stride].z);

		}
		__syncthreads();
	}
	

	if (tx == 0)
	{
		d_result_min[blockIdx.x] = min(cc[tx].x,min(cc[tx].y,cc[tx].z));

		d_result_max[blockIdx.x] = max(dd[tx].x,max(dd[tx].y,dd[tx].z));
		
	}

	__syncthreads();

}

int NumDigits(int x)  
{  

    return (x < 10 ? 1 :   (x < 100 ? 2 :   (x < 1000 ? 3 :   
        (x < 10000 ? 4 :   (x < 100000 ? 5 :   (x < 1000000 ? 6 :   
        (x < 10000000 ? 7 :  (x < 100000000 ? 8 :  (x < 1000000000 ? 9 :  
        10)))))))));  
}  


__global__ void limit_displacement_kernel(float3 *d_grid_pos, float3 *grid_disp, int Nx, int Ny, int Nz, bool disp_grid, float factor)
{
	uint n = Nx*Ny*Nz;


	int tx = threadIdx.x;
	int ind = blockIdx.x*blockDim.x+tx;

	uint z = ind/(Nx *Ny);
	uint y = (ind%(Nx *Ny))/Nx;
	uint x = (ind%(Nx *Ny))%Nx;

	float3 grid_pos = {0.0f,0.0f,0.0f};

	if(ind < n)
	{
		float3 disp_val = grid_disp[ind];
		
		if(disp_grid)
		{
			
			grid_pos.x = x + (disp_val.x * factor);
			grid_pos.y = y + (disp_val.y * factor);
			grid_pos.z = z + (disp_val.z * factor);

		}
		else
		{
			grid_pos.x = x;
			grid_pos.y = y;
			grid_pos.z = z;
		}

		d_grid_pos[ind] = grid_pos;
	}
}


void Interpolations::limit_displacement(float3 *d_xyz, float3 *disp, int Nx, int Ny, int Nz,bool disp_grid, uint magnify)
{
	dim3 grids(ceil((Nx * Ny * Nz)/float(1024)),1,1);
	dim3 tids(1024,1,1);

	int block_num = grids.x;

	int n = Nx*Ny*Nz;

	float min_val;
	float max_val;
	
	float *d_Reduction_max;
	float *d_Reduction_min;

	cudaMalloc((void **)&d_Reduction_max, sizeof(float)* (block_num));
	cudaMalloc((void **)&d_Reduction_min, sizeof(float)* (block_num));

	cudaMemset(d_Reduction_max, 0.0, sizeof(float) * (block_num));
	cudaMemset(d_Reduction_min, 0.0, sizeof(float) * (block_num));

	displacement_max_min_kernel<<<grids,tids>>>(d_Reduction_max,d_Reduction_min,disp,n);
	cudaDeviceSynchronize();

	unsigned int  x_grid = 1;

	unsigned int  x_thread = 1024;
	
	Min_reduction_interpolate<<<x_grid, x_thread>>>(d_Reduction_min,d_Reduction_max,block_num);

	cudaDeviceSynchronize();

	cudaMemcpy(&min_val, d_Reduction_min, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&max_val, d_Reduction_max, sizeof(float), cudaMemcpyDeviceToHost);
	
	int digits_num = NumDigits(max(abs(min_val),abs(max_val)));

	float fact_val  =  magnify * (10.0f/pow(10,digits_num));

	limit_displacement_kernel<<<grids,tids>>>(d_xyz,disp,Nx,Ny,Nz,disp_grid,fact_val);

	cudaFree(d_Reduction_min);

	cudaFree(d_Reduction_max);

	cudaDeviceSynchronize();

	getLastCudaError("Interpolations.cu - limit displacement function failed");

}