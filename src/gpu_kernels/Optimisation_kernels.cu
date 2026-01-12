

#include "Optimisation_kernels.h"



#define BLOCKFX 32
#define BLOCKFY 16
#define p2 512

#define INDEX(i,j,j_off)  (i +__mul24(j,j_off))
#define IOFF 1
#define JOFF (BLOCKFX+6)
#define KOFF ((BLOCKFX+6)*(BLOCKFY+6))



__global__ void Reduction(float *d_DataIn, float *d_DataOut, int block_num)
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



__global__ void  Update_den_GPU_kernel(REAL3 *d_u, REAL *d_den,REAL VolFrac, REAL *d_grad, float *d_volume, int NX, int NY, int NZ,const REAL lmid,const REAL move,const REAL MinDens,
REAL *d_new_den,REAL *d_new_den_result)
{

  int tx = threadIdx.x;
  int index = tx + blockDim.x*blockIdx.x;
  int zz = (index/(NY*NX));
  int yy = ((index%(NY*NX))/NY);
  int xx = ((index%(NY*NX))%NY);
  int index2 = xx + (yy * NX )+ zz * (NX*NY);
  __shared__ float cc[1024];
  REAL a1,a2,MyGrad;
  REAL a3 = 0.0;
  a1 = d_grad[index2];
  a2 = d_den[index2];
  if((xx < (NX ) )&& (yy < (NY )) && (zz < (NZ)))
  {
    
    
    MyGrad = max((REAL)-1.0*a1, (REAL) 0.0);
    a3 = max(MinDens, max(a2-move, min((REAL)1.0, min(a2+move, a2*sqrtf(MyGrad/lmid)))));
    d_new_den[index2] = a3;


  }


  

  cc[tx] = a3;
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
		d_new_den_result[blockIdx.x] = cc[tx];
		
	}

	__syncthreads();


}






__global__ void GPUMeshFilter_kernel(REAL3 *d_u,REAL *d_den, REAL rmin, REAL *d_grad, const int pitchX, int NX,int NY,int NZ, REAL *d_gradone)
{

  __shared__ float2 sh_den[(BLOCKFX + 6)*(BLOCKFY + 6)*7];
  

  int indg;

	int ind;
	
	int ind1_h;

  int ind2_h;

  int ind3_h;

  int indg1_h;

  int indg12_h;

  int indg13_h;

  int indg2_h;

  int indg22_h;

  int indg23_h;

  int indg3_h;

  int indg32_h;

  int indg33_h;




  int ind_k = threadIdx.x + threadIdx.y*(BLOCKFX);

  int i,j;

  int halo1 = ind_k < 2*(BLOCKFX+BLOCKFY+2);

  int halo2 = ind_k < 2*(BLOCKFX+BLOCKFY+6);

  int halo3 = ind_k < 2*(BLOCKFX+BLOCKFY+10);

  int kk;

  if(halo1)
	{
		
    if(ind_k < 2*(BLOCKFY+2))
    {
      kk = threadIdx.x + threadIdx.y * BLOCKFX;
      i = floorf(kk/(BLOCKFY+2))*(BLOCKFX+1)-1;
      j = kk%(BLOCKFY+2) - 1;
    }

    else
    {
      kk = (threadIdx.x + threadIdx.y * BLOCKFX) - (2*(BLOCKFY+2));
      
      j = floorf(kk/BLOCKFX)*(BLOCKFY+1) - 1;
      i = kk%BLOCKFX ;
    }   
    
		
		ind1_h = INDEX(i+3,j+3,(BLOCKFX+6)) + ((BLOCKFX+6)*(BLOCKFY+6)*3);
   
		i = INDEX(i, blockIdx.x, BLOCKFX);
		j = INDEX(j, blockIdx.y, BLOCKFY);
		indg1_h = INDEX(i, j, pitchX);
    indg12_h = INDEX(i, j, pitchX) + (NY*pitchX);
    indg13_h = INDEX(i, j, pitchX) + 2*(NY*pitchX);



		halo1 = (i>=0) && (i<NX) && (j>=0) && (j<NY);

	}
  
  if(halo2)
	{
		
    if(ind_k < 2*(BLOCKFY+4))
    {
      kk = threadIdx.x + threadIdx.y * BLOCKFX;
      i = floorf(kk/(BLOCKFY+4))*(BLOCKFX+3)-2;
      j = kk%(BLOCKFY+4) - 2;
    }

    else
    {
      kk = (threadIdx.x + threadIdx.y * BLOCKFX) - (2*(BLOCKFY+4));
      
      j = floorf(kk/(BLOCKFX+2))*(BLOCKFY+3) - 2;
      i = kk%(BLOCKFX+2) - 1 ;
    }   
    
		
		ind2_h = INDEX(i+3,j+3,(BLOCKFX+6)) + ((BLOCKFX+6)*(BLOCKFY+6)*3);
   
		i = INDEX(i, blockIdx.x, BLOCKFX);
		j = INDEX(j, blockIdx.y, BLOCKFY);
		indg2_h = INDEX(i, j, pitchX);
    indg22_h = INDEX(i, j, pitchX) + (NY*pitchX);
    indg23_h = INDEX(i, j, pitchX) + 2*(NY*pitchX);

  

		halo2 = (i>=0) && (i<NX) && (j>=0) && (j<NY);
	}

  if(halo3)
	{
		
    if(ind_k < 2*(BLOCKFY+6))
    {
      kk = threadIdx.x + threadIdx.y * BLOCKFX;
      i = floorf(kk/(BLOCKFY+6))*(BLOCKFX+5)-3;
      j = kk%(BLOCKFY+6) - 3;
    }

    else
    {
      kk = (threadIdx.x + threadIdx.y * BLOCKFX) - (2*(BLOCKFY+6));
      
      j = floorf(kk/(BLOCKFX+4))*(BLOCKFY+5) - 3;
      i = kk%(BLOCKFX+4) - 2 ;
    }   
    
		
		ind3_h = INDEX(i+3,j+3,(BLOCKFX+6)) + ((BLOCKFX+6)*(BLOCKFY+6)*3);
		i = INDEX(i, blockIdx.x, BLOCKFX);
		j = INDEX(j, blockIdx.y, BLOCKFY);
		indg3_h = INDEX(i, j, pitchX);
    indg32_h = INDEX(i, j, pitchX) + (NY*pitchX);
    indg33_h = INDEX(i, j, pitchX) + 2*(NY*pitchX);



		halo3 = (i>=0) && (i<NX) && (j>=0) && (j<NY);
	}

  i = threadIdx.x;
	j = threadIdx.y;
	ind = INDEX(i+3,j+3,(BLOCKFX+6)) + ((BLOCKFX+6)*(BLOCKFY+6)*3);
	i = INDEX(i, blockIdx.x, BLOCKFX);
	j = INDEX(j, blockIdx.y, BLOCKFY);

	indg = INDEX(i,j,pitchX);
  int indg2 = indg + (NY*pitchX);
  int indg3 = indg + 2*(NY*pitchX);


  __syncthreads();



	const int active = (i<NX) && (j<NY);

  
	if(active) 
  {
    ////4th(by index) XY plane in shared memory
    sh_den[ind+((BLOCKFX+6)*(BLOCKFY+6))].x = d_den[indg];
    ////5th(by index) XY plane in shared memory
    sh_den[ind+((BLOCKFX+6)*(BLOCKFY+6)*2)].x = d_den[indg2];
    ////6th(by index) XY plane in shared memory
    sh_den[ind+((BLOCKFX+6)*(BLOCKFY+6)*3)].x = d_den[indg3];

     ////4th(by index) XY plane in shared memory
    sh_den[ind+((BLOCKFX+6)*(BLOCKFY+6))].y = d_grad[indg];

    ////5th(by index) XY plane in shared memory
    sh_den[ind+((BLOCKFX+6)*(BLOCKFY+6)*2)].y = d_grad[indg2];
    ////6th(by index) XY plane in shared memory
    sh_den[ind+((BLOCKFX+6)*(BLOCKFY+6)*3)].y = d_grad[indg3];
  }

  __syncthreads();
  
  if(halo1) 
	{
		

    
    sh_den[ind1_h+((BLOCKFX+6)*(BLOCKFY+6))].x = d_den[indg1_h];
    sh_den[ind1_h+((BLOCKFX+6)*(BLOCKFY+6)) * 2].x = d_den[indg12_h];
    sh_den[ind1_h+((BLOCKFX+6)*(BLOCKFY+6)) * 3].x = d_den[indg13_h];

		sh_den[ind1_h+((BLOCKFX+6)*(BLOCKFY+6))].y = d_grad[indg1_h];
		sh_den[ind1_h+((BLOCKFX+6)*(BLOCKFY+6)) * 2].y = d_grad[indg12_h];
		sh_den[ind1_h+((BLOCKFX+6)*(BLOCKFY+6)) * 3].y = d_grad[indg13_h];

	}
  __syncthreads();
  if(halo2) 
	{
		sh_den[ind2_h+((BLOCKFX+6)*(BLOCKFY+6))].x = d_den[indg2_h];
    sh_den[ind2_h+((BLOCKFX+6)*(BLOCKFY+6)) * 2].x = d_den[indg22_h];
    sh_den[ind2_h+((BLOCKFX+6)*(BLOCKFY+6)) * 3].x = d_den[indg23_h];

		sh_den[ind2_h+((BLOCKFX+6)*(BLOCKFY+6))].y = d_grad[indg2_h];
		sh_den[ind2_h+((BLOCKFX+6)*(BLOCKFY+6)) * 2].y = d_grad[indg22_h];
		sh_den[ind2_h+((BLOCKFX+6)*(BLOCKFY+6)) * 3].y = d_grad[indg23_h];
	}
  __syncthreads();
  if(halo3) 
	{
		sh_den[ind3_h+((BLOCKFX+6)*(BLOCKFY+6))].x = d_den[indg3_h];
    sh_den[ind3_h+((BLOCKFX+6)*(BLOCKFY+6)) * 2].x = d_den[indg32_h];
    sh_den[ind3_h+((BLOCKFX+6)*(BLOCKFY+6)) * 3].x = d_den[indg33_h];


		sh_den[ind3_h+((BLOCKFX+6)*(BLOCKFY+6))].y = d_grad[indg3_h];
		sh_den[ind3_h+((BLOCKFX+6)*(BLOCKFY+6)) * 2].y = d_grad[indg32_h];
		sh_den[ind3_h+((BLOCKFX+6)*(BLOCKFY+6)) * 3].y = d_grad[indg33_h];
	}
  
  __syncthreads();
	const int NZM1 = NZ-3;
  int indg_cur;
  
  for(int k=0;k<NZ;k++)
	{
    
    if(active)
		{
			indg_cur = indg;
      indg = INDEX(indg,NY,pitchX);
      

      // 0th XY plane in shared memory
      sh_den[ind-((BLOCKFX+6)*(BLOCKFY+6))*3] = sh_den[ind-((BLOCKFX+6)*(BLOCKFY+6))*2];
      sh_den[ind-((BLOCKFX+6)*(BLOCKFY+6))*2] = sh_den[ind-((BLOCKFX+6)*(BLOCKFY+6))];

			sh_den[ind-((BLOCKFX+6)*(BLOCKFY+6))] = sh_den[ind];
      sh_den[ind] = sh_den[ind+((BLOCKFX+6)*(BLOCKFY+6))];

      sh_den[ind + ((BLOCKFX+6)*(BLOCKFY+6))] = sh_den[ind + ((BLOCKFX+6)*(BLOCKFY+6))*2];
      sh_den[ind + ((BLOCKFX+6)*(BLOCKFY+6))*2] = sh_den[ind + ((BLOCKFX+6)*(BLOCKFY+6))*3];
			// 1st XY plane in shared memory
			
			if(k<NZM1) 
      {
        indg3 = INDEX(indg3, NY, pitchX);
        sh_den[ind + ((BLOCKFX+6)*(BLOCKFY+6))*3].x = d_den[indg3];
        sh_den[ind + ((BLOCKFX+6)*(BLOCKFY+6))*3].y = d_grad[indg3];
      }
		
			
		}
		if(halo1)
		{
			sh_den[ind1_h - ((BLOCKFX+6)*(BLOCKFY+6))*3] = sh_den[ind1_h - ((BLOCKFX+6)*(BLOCKFY+6))*2];
			sh_den[ind1_h - ((BLOCKFX+6)*(BLOCKFY+6))*2] = sh_den[ind1_h - ((BLOCKFX+6)*(BLOCKFY+6))];

			sh_den[ind1_h - ((BLOCKFX+6)*(BLOCKFY+6))] = sh_den[ind1_h];
			sh_den[ind1_h] = sh_den[ind1_h+ ((BLOCKFX+6)*(BLOCKFY+6))];

      sh_den[ind1_h + ((BLOCKFX+6)*(BLOCKFY+6))] = sh_den[ind1_h + ((BLOCKFX+6)*(BLOCKFY+6))*2];
      sh_den[ind1_h + ((BLOCKFX+6)*(BLOCKFY+6))*2] = sh_den[ind1_h + ((BLOCKFX+6)*(BLOCKFY+6))*3];

			if(k<NZM1) 
      {
        indg13_h = INDEX(indg13_h, NY, pitchX);
        sh_den[ind1_h + ((BLOCKFX+6)*(BLOCKFY+6))*3].x = d_den[indg13_h];
        sh_den[ind1_h + ((BLOCKFX+6)*(BLOCKFY+6))*3].y = d_grad[indg13_h];
      }
			
		}

    if(halo2)
		{
			sh_den[ind2_h - ((BLOCKFX+6)*(BLOCKFY+6))*3] = sh_den[ind2_h - ((BLOCKFX+6)*(BLOCKFY+6))*2];
      sh_den[ind2_h - ((BLOCKFX+6)*(BLOCKFY+6))*2] = sh_den[ind2_h - ((BLOCKFX+6)*(BLOCKFY+6))];
			
			sh_den[ind2_h - ((BLOCKFX+6)*(BLOCKFY+6))] = sh_den[ind2_h];
			sh_den[ind2_h] = sh_den[ind2_h+ ((BLOCKFX+6)*(BLOCKFY+6))];

      sh_den[ind2_h + ((BLOCKFX+6)*(BLOCKFY+6))] = sh_den[ind2_h + ((BLOCKFX+6)*(BLOCKFY+6))*2];
      sh_den[ind2_h + ((BLOCKFX+6)*(BLOCKFY+6))*2] = sh_den[ind2_h + ((BLOCKFX+6)*(BLOCKFY+6))*3];

			if(k<NZM1) 
      {
        indg23_h = INDEX(indg23_h, NY, pitchX);
        sh_den[ind2_h + ((BLOCKFX+6)*(BLOCKFY+6))*3].x = d_den[indg23_h];
        sh_den[ind2_h + ((BLOCKFX+6)*(BLOCKFY+6))*3].y = d_grad[indg23_h];
      }
			
		}

    if(halo3)
		{
			sh_den[ind3_h - ((BLOCKFX+6)*(BLOCKFY+6))*3] = sh_den[ind3_h - ((BLOCKFX+6)*(BLOCKFY+6))*2];
      sh_den[ind3_h - ((BLOCKFX+6)*(BLOCKFY+6))*2] = sh_den[ind3_h - ((BLOCKFX+6)*(BLOCKFY+6))];
			
			sh_den[ind3_h - ((BLOCKFX+6)*(BLOCKFY+6))] = sh_den[ind3_h];
			sh_den[ind3_h] = sh_den[ind3_h+ ((BLOCKFX+6)*(BLOCKFY+6))];

      sh_den[ind3_h + ((BLOCKFX+6)*(BLOCKFY+6))] = sh_den[ind3_h + ((BLOCKFX+6)*(BLOCKFY+6))*2];
      sh_den[ind3_h + ((BLOCKFX+6)*(BLOCKFY+6))*2] = sh_den[ind3_h + ((BLOCKFX+6)*(BLOCKFY+6))*3];

			if(k<NZM1) 
      {
        indg33_h = INDEX(indg33_h, NY, pitchX);
        sh_den[ind3_h + ((BLOCKFX+6)*(BLOCKFY+6))*3].x = d_den[indg33_h];
        sh_den[ind3_h + ((BLOCKFX+6)*(BLOCKFY+6))*3].y = d_grad[indg33_h];
      }
			
		}

    __syncthreads();



    ///voxel element values
    const int active1 = (i<NX-1) && (j<NY-1) && (k < NZ-1);
    if(active1 )
    {
      
      float sum = 0.0;
      float new_grad = 0.0;
      
      for(int ek = max(k-(int)rmin, 0);ek<=min(k+(int)rmin, (int)(NZ-1));ek++)
      {
        
        for(int ej = max(j-(int)rmin, 0);ej<=min(j+(int)rmin, (int)(NY-1));ej++)
        {
          for(int ei = max(i-(int)rmin, 0);ei<=min(i+(int)rmin, (int)(NX-1));ei++)
          {
            const REAL fac = rmin-sqrtf((k-ek)*(k-ek)+(j-ej)*(j-ej)+(i-ei)*(i-ei));
            sum += max((REAL)0.0, fac);
            
        
            new_grad += max((REAL)0.0, fac)*sh_den[ind+(ei-i)*IOFF+(ej-j)*JOFF+(ek-k)*(KOFF)].x*sh_den[ind+(ei-i)*IOFF+(ej-j)*JOFF+(ek-k)*(KOFF)].y;
   
          }
        }
      }
      


      new_grad = new_grad/(sh_den[ind].x*sum);

      d_gradone[indg_cur] = new_grad;


    }    

    __syncthreads();

    

  }
  

}



__global__ void init_d_den_kernel(REAL *d_den, REAL volfrac, int size)
{
  int index = threadIdx.x + (blockIdx.x)*(blockDim.x);
  REAL x1;
  if(index < size)
  {
    x1 =volfrac;
    d_den[index] = x1;
  }
}



void Optimisation_kernels::GPUMeshFilter(REAL3 *d_u,REAL *d_den, REAL rmin, REAL *d_grad,const int pitchX, int NX,int NY,int NZ)
{
  
  REAL *d_gradone;
  checkCudaErrors(cudaMalloc((void **)&d_gradone, sizeof(REAL)*NX * NY*NZ));
  cudaMemset(d_gradone, 0.0, sizeof(REAL)*pitchX* NY* NZ);
  
  dim3 tids(BLOCKFX,BLOCKFY,1);
  dim3 grids(ceil((NX)/(BLOCKFX*1.0)),ceil((NY)/(BLOCKFY*1.0)),1);
  GPUMeshFilter_kernel<<<grids,tids>>>(d_u,d_den,rmin,d_grad,pitchX,NX,NY,NZ,d_gradone);
  cudaDeviceSynchronize();
  cudaMemcpy(d_grad, d_gradone, (pitchX*NY*NZ) * sizeof(REAL), cudaMemcpyDeviceToDevice);
  cudaFree(d_gradone);

}




void Optimisation_kernels::Update_den_GPU(REAL3 *d_u,REAL *d_den, REAL VolFrac, REAL *d_grad, float *d_volume,int NX, int NY, int NZ,const REAL lmid,const REAL move,const REAL MinDens, REAL *d_new_den, REAL *d_new_den_result,int block_num)
{
  
  dim3 tids(1024,1,1);
  dim3 grids(ceil((NX*NY*NZ)/float(1024)),1,1);
  Update_den_GPU_kernel<<<grids,tids>>>(d_u, d_den, VolFrac,d_grad,d_volume, NX, NY, NZ,lmid,move,MinDens,d_new_den,d_new_den_result);
  cudaDeviceSynchronize();

  unsigned int  x_grid = 1;
	unsigned int  x_thread = 1024;
	Reduction<<<x_grid, x_thread>>>(d_new_den_result,d_new_den_result,block_num);
  cudaDeviceSynchronize();

}




void Optimisation_kernels::Update_s_one(REAL3 *d_u,REAL *d_den, REAL VolFrac,REAL MinDens, REAL *d_grad, float *d_volume,int pitchX, int NX, int NY, int NZ)
{

  double l1 = 0.0;
  double l2 = 1e6;
  int counter = 0;
  const REAL move = 0.2;

  int block_num = ((NX*NY*NZ)/1024) + (!((NX*NY*NZ) % 1024) ? 0:1);
  REAL *d_new_den_result;
  REAL *d_new_den;
  checkCudaErrors(cudaMalloc((void **)&d_new_den_result, sizeof(REAL)*block_num));
  checkCudaErrors(cudaMalloc((void **)&d_new_den, sizeof(REAL)*(NX*NY*NZ)));
  cudaMemset(d_new_den_result, 0.0, sizeof(REAL)*block_num);
  cudaMemset(d_new_den, 0.0, sizeof(REAL)*(NX*NY*NZ));

  while(((l2-l1) > 1e-4) && (counter < 1e3))
  {

    counter++;
    
    const double lmid = 0.5*(l2+l1);
    Update_den_GPU(d_u,d_den,VolFrac,d_grad,d_volume,NX,NY,NZ,lmid,move,MinDens,d_new_den,d_new_den_result,block_num);


    float sum;
    cudaMemcpy(&sum, d_new_den_result, sizeof(float), cudaMemcpyDeviceToHost);


    if((sum - (VolFrac*(NX )*(NY)*(NZ))) > 0)
    {
      l1 = lmid;
    }
    else
    {
      l2 = lmid;
    }


  }

  
  cudaMemcpy(d_den,d_new_den,sizeof(REAL)*(NX*NY*NZ),cudaMemcpyDeviceToDevice);

  cudaMemcpy(d_volume,d_new_den,sizeof(REAL)*(NX*NY*NZ),cudaMemcpyDeviceToDevice);

  cudaFree(d_new_den);
  cudaFree(d_new_den_result);



}






void Optimisation_kernels::init_d_den(REAL *d_den, REAL volfrac, int size)
{
  dim3 tids(1024,1,1);
  dim3 grids(ceil(size/float(1024)),1,1);
  init_d_den_kernel<<<grids,tids>>>(d_den,volfrac,size);
  cudaDeviceSynchronize();
}















