

#include "Thermalsim.h"
#include "../general/DataTypes.h"
#include <iostream>
#include "../general/topopt_defines.h"


using namespace std;


__constant__ REAL GPU_EleStiffone_t[8][8];


__global__ void MatVecKernel_t(const int NX, const int NY, const int NZ, const int pitchX, REAL3 *d_u,REAL *d_den, REAL *d_selection,REAL3 *res, REAL gpupexp)
{

	int indg;
	
	
	int ind;
	

	int indg_h;
	
	int ind_h;
	

	__shared__ REAL3 s_u[3 * (BLOCKSX + 2) * (BLOCKSY + 2)];
	__shared__ REAL s_d[3 * (BLOCKSX + 2) * (BLOCKSY + 2)];

	
	int k = threadIdx.x + threadIdx.y * BLOCKSX;

	int i,j;
	
	
	int halo = k < 2 * ( BLOCKSX + BLOCKSY + 2 );

	REAL select =  0.0;
	
	if(halo)
	{
		
		if(threadIdx.y < 2)
		{
			
			i = threadIdx.x;
			
			j = threadIdx.y*(BLOCKSY + 1) - 1;
		}
		
		else
		{
			
			i = (k%2)*(BLOCKSX + 1) - 1;
			
			j = k/2 - BLOCKSX - 1;
		}
		
		ind_h = INDEX(i+1,j+1,(BLOCKSX + 2)) + (BLOCKSX + 2) * (BLOCKSY + 2);
		
		i = INDEX(i, blockIdx.x, BLOCKSX);
		j = INDEX(j, blockIdx.y, BLOCKSY);

		indg_h = INDEX(i, j, pitchX);
		
		halo = (i>=0) && (i<NX) && (j>=0) && (j<NY);
	}
	

	i = threadIdx.x;
	j = threadIdx.y;
	
	ind = INDEX(i+1,j+1,(BLOCKSX + 2)) + (BLOCKSX + 2) * (BLOCKSY + 2);
	i = INDEX(i, blockIdx.x, BLOCKSX);
	j = INDEX(j, blockIdx.y, BLOCKSY);

	indg = INDEX(i,j,pitchX);
	
	
	const int active = (i<NX) && (j<NY);
	
	
	if(active) 
	
	{
		s_u[ind + (BLOCKSX + 2) * (BLOCKSY + 2)] = d_u[indg];
		s_d[ind + (BLOCKSX + 2) * (BLOCKSY + 2)] = d_den[indg];
	}

	if(halo) 
	{
		s_u[ind_h + (BLOCKSX + 2) * (BLOCKSY + 2)] = d_u[indg_h];
		s_d[ind_h + (BLOCKSX + 2) * (BLOCKSY + 2)] = d_den[indg_h];
	}
	

	int indg_cur;

	const int NZM1 = NZ-1;
	
	REAL3 MyRes;
	
	for(k=0;k<NZ;k++)
	{
		if(active)
		{
			
			indg_cur = indg;
			
			indg = INDEX(indg, NY, pitchX);
		
			s_u[ind - (BLOCKSX + 2) * (BLOCKSY + 2)] = s_u[ind];
			s_u[ind] = s_u[ind + (BLOCKSX + 2) * (BLOCKSY + 2)];

			s_d[ind - (BLOCKSX + 2) * (BLOCKSY + 2)] = s_d[ind];
			s_d[ind] = s_d[ind + (BLOCKSX + 2) * (BLOCKSY + 2)];

			if(k<NZM1) 
			{
				s_u[ind + (BLOCKSX + 2) * (BLOCKSY + 2)] = d_u[indg];
				s_d[ind + (BLOCKSX + 2) * (BLOCKSY + 2)] = d_den[indg];
			}

			select = d_selection[indg_cur];
		}
		if(halo)
		{
			indg_h = INDEX(indg_h, NY, pitchX);

			s_u[ind_h - (BLOCKSX + 2) * (BLOCKSY + 2)] = s_u[ind_h];
			s_u[ind_h] = s_u[ind_h + (BLOCKSX + 2) * (BLOCKSY + 2)];

			s_d[ind_h - (BLOCKSX + 2) * (BLOCKSY + 2)] = s_d[ind_h];
			s_d[ind_h] = s_d[ind_h + (BLOCKSX + 2) * (BLOCKSY + 2)];



			if(k<NZM1) 
			{
				s_u[ind_h + (BLOCKSX + 2) * (BLOCKSY + 2)] = d_u[indg_h];
				s_d[ind_h + (BLOCKSX + 2) * (BLOCKSY + 2)] = d_den[indg_h];

			}
		}
		
	
		__syncthreads();
		
	
		if(active)
		{
		
			MyRes.x = 0.0;
			MyRes.y = 0.0;
			MyRes.z = 0.0;
		

			if(select == -1.0)
			{
				MyRes.x = 0.0;
				
			}


			else
			{
				for(int ek1=0;ek1<2;ek1++)
				{
					const int EIDK = k-ek1;
					if(EIDK >= 0 && EIDK < NZ-1)
					{
						for(int ej1=0;ej1<2;ej1++)
						{
							const int EIDJ = j-ej1;
							if(EIDJ >= 0 && EIDJ < NY-1)
							{
								for(int ei1=0;ei1<2;ei1++)
								{
									const int EIDI = i-ei1;
									if(EIDI >= 0 && EIDI < NX-1)
									{
										const REAL Dens = pow(s_d[ind-ei1*IOFF-ej1*(BLOCKSX + 2) -ek1*(BLOCKSX + 2) * (BLOCKSY + 2)], gpupexp);
										
										const int LID1 = ei1+ej1*2+ek1*4;
										for(int ek2=0;ek2<2;ek2++)
										{
											for(int ej2=0;ej2<2;ej2++)
											{
												for(int ei2=0;ei2<2;ei2++)
												{
													const int LID2 = ei2+ej2*2+ek2*4;
													const int idiff = ei2-ei1;
													const int jdiff = ej2-ej1;
													const int kdiff = ek2-ek1;
													const REAL3 MyU = s_u[ind+idiff*IOFF+jdiff*(BLOCKSX + 2)+kdiff*(BLOCKSX + 2) * (BLOCKSY + 2)];
													
													MyRes.x += (0.001+0.999*Dens)*(GPU_EleStiffone_t[LID1][LID2]*MyU.x);
													
												}
											}
										}//end e2 loops
									}//if i okay
								}//i-loop
							}//if j okay
						}//j-loop
					}//if k okay
				}//k-loop
			}//end if not dirichlet
			//store results
		
			res[indg_cur] = MyRes;

		}//end if active
		__syncthreads();
	}
}

__global__ void ResidualKernel_t(const int NX, const int NY, const int NZ, const int pitchX, REAL3 *d_u, REAL *d_den, REAL *d_selection, REAL3 *res, REAL gpupexp)
{
	
	int indg;

	int ind;
	
	int indg_h;
	
	int ind_h;
	
	__shared__ REAL3 s_u[3*(BLOCKSX + 2) * (BLOCKSY + 2)];

	__shared__ REAL s_d[3*(BLOCKSX + 2) * (BLOCKSY + 2)];
	

	int k = threadIdx.x + threadIdx.y * BLOCKSX;

	int i,j;

	int halo = k < 2*(BLOCKSX + BLOCKSY +2);

	REAL select = 0.0;
	
	if(halo)
	{
		
		if(threadIdx.y<2)
		{
		
			i = threadIdx.x;
			
			j = threadIdx.y*(BLOCKSY + 1) - 1;
			
		}
	
		else
		{
			
			i = (k%2)*(BLOCKSX + 1)-1;
			
			j = k/2 - BLOCKSX - 1;
		}
		
		ind_h = INDEX(i+1,j+1,(BLOCKSX + 2)) + (BLOCKSX + 2) * (BLOCKSY + 2);

		i = INDEX(i, blockIdx.x, BLOCKSX);
		j = INDEX(j, blockIdx.y, BLOCKSY);
		
		indg_h = INDEX(i, j, pitchX);
		halo = (i>=0) && (i<NX) && (j>=0) && (j<NY);
		
		
	}
	
	i = threadIdx.x;

	j = threadIdx.y;

	ind = INDEX(i+1,j+1,(BLOCKSX + 2)) + (BLOCKSX + 2) * (BLOCKSY + 2);



	////////////////////////////indg and active//////////////////////////////////////////
	i = INDEX(i, blockIdx.x, BLOCKSX);
	j = INDEX(j, blockIdx.y, 28);

	indg = INDEX(i,j,pitchX);
	

	const int active = (i<NX) && (j<NY);
	//////////////////////////////////////////////////////////////////////

	if(active) 
	{
		s_u[ind + (BLOCKSX + 2) * (BLOCKSY + 2) ] = d_u[indg];
		s_d[ind + (BLOCKSX + 2) * (BLOCKSY + 2) ] = d_den[indg];
	
	}

	
	if(halo) 
	{
		s_u[ind_h + (BLOCKSX + 2) * (BLOCKSY + 2)] = d_u[indg_h];
		s_d[ind_h + (BLOCKSX + 2) * (BLOCKSY + 2)] = d_den[indg_h];
	}
	

	int indg_cur;
	
	const int NZM1 = NZ-1;

	REAL3 MyRes;

	for(k=0;k<NZ;k++)
	{

		if(active)
		{
			
			indg_cur = indg;
			
			indg = INDEX(indg, NY, pitchX);
			
			s_u[ind - (BLOCKSX + 2) * (BLOCKSY + 2) ] = s_u[ind];
			s_d[ind - (BLOCKSX + 2) * (BLOCKSY + 2)] = s_d[ind];

			s_u[ind] = s_u[ind+(BLOCKSX + 2) * (BLOCKSY + 2)];
			s_d[ind] = s_d[ind+(BLOCKSX + 2) * (BLOCKSY + 2)];

	
			if(k<NZM1) 
			{
				s_u[ind + (BLOCKSX + 2) * (BLOCKSY + 2)] = d_u[indg];
				s_d[ind + (BLOCKSX + 2) * (BLOCKSY + 2)] = d_den[indg];

			}

			select = d_selection[indg_cur];
		}
		if(halo)
		{
			indg_h = INDEX(indg_h, NY, pitchX);
			s_u[ind_h - (BLOCKSX + 2) * (BLOCKSY + 2)] = s_u[ind_h];
			s_u[ind_h] = s_u[ind_h + (BLOCKSX + 2) * (BLOCKSY + 2)];

			s_d[ind_h - (BLOCKSX + 2) * (BLOCKSY + 2) ] = s_d[ind_h];
			s_d[ind_h] = s_d[ind_h + (BLOCKSX + 2) * (BLOCKSY + 2) ];


			if(k<NZM1) 
			{
				s_u[ind_h + (BLOCKSX + 2) * (BLOCKSY + 2) ] = d_u[indg_h];
				s_d[ind_h + (BLOCKSX + 2) * (BLOCKSY + 2) ] = d_den[indg_h];
			}
		}
		
	
		__syncthreads();
		

		if(active)
		{
			
			MyRes.x = 0.0;
			MyRes.y = 0.0;
			MyRes.z = 0.0;
		

			if(select == -1.0)
			{
				MyRes.x  = 0.0;
			}

			else 
			{
				MyRes.x = 0.01;
				for(int ek1=0;ek1<2;ek1++)
				{
					const int EIDK = k-ek1;
					if(EIDK >= 0 && EIDK < NZ-1)
					{
						for(int ej1=0;ej1<2;ej1++)
						{
							const int EIDJ = j-ej1;
							if(EIDJ >= 0 && EIDJ < NY-1)
							{
								for(int ei1=0;ei1<2;ei1++)
								{
									const int EIDI = i-ei1;
									if(EIDI >= 0 && EIDI < NX-1)
									{
										const REAL Dens = pow(s_d[ind - ei1 * IOFF - ej1 * (BLOCKSX + 2) - ek1 * (BLOCKSX + 2) * (BLOCKSY + 2)], gpupexp);
										const int LID1 = ei1+ej1*2+ek1*4;
										for(int ek2=0;ek2<2;ek2++)
										{
											for(int ej2=0;ej2<2;ej2++)
											{
												for(int ei2=0;ei2<2;ei2++)
												{
													const int LID2 = ei2+ej2*2+ek2*4;
													const int idiff = ei2-ei1;
													const int jdiff = ej2-ej1;
													const int kdiff = ek2-ek1;
													const REAL3 MyU = s_u[ind + idiff * IOFF + jdiff * 34 + kdiff * (BLOCKSX + 2) * (BLOCKSY + 2)];
													//B-AX
													MyRes.x -= (0.001+0.999*Dens)*(GPU_EleStiffone_t[LID1][LID2]*MyU.x) ;
													
												}
											}
										}//end e2 loops
									}//if i okay
								}//i-loop
							}//if j okay
						}//j-loop
					}//if k okay
				}//k-loop
				//end if not dirichlet
				//store results
			}
			res[indg_cur] = MyRes;
	
		}//end if active
		__syncthreads();
	}
}


__global__ void GPUEvalGradone_t(const int NX, const int NY, const int NZ, const int u_pitchX, const int grad_pitchX, REAL3 *d_u,REAL *d_den, REAL *d_grad, REAL *d_ObjValue, float gpupexp)
{

	int indg;
	
	int ind;

	int indg_h;

	int ind_h;

	__shared__ REAL3 s_u[2*((BLOCKGX+1)*(BLOCKGY+1))];
	__shared__ REAL s_d[2*((BLOCKGX+1)*(BLOCKGY+1))];


	int k = threadIdx.x + threadIdx.y*BLOCKGX;

	int i,j;
	
	
	bool halo = k < (BLOCKGX + BLOCKGY +1);

	
	if(halo)
	{
	
		if(threadIdx.y==0)
		{
			i = threadIdx.x;
			j = BLOCKGY;
		}
		
		else
		{
			i = BLOCKGX;
			j = k - BLOCKGX;
		}
		
		ind_h = INDEX(i,j,BLOCKGX+1) + ((BLOCKGX+1)*(BLOCKGY+1));
		
		
		i = INDEX(i, blockIdx.x, BLOCKGX);
		j = INDEX(j, blockIdx.y, BLOCKGY);
		indg_h = INDEX(i, j, u_pitchX);
		
		halo = (i>=0) && (i<NX) && (j>=0) && (j<NY);
	}	

	i = threadIdx.x;
	j = threadIdx.y;
	
	ind = INDEX(i,j,BLOCKGX+1) + ((BLOCKGX+1)*(BLOCKGY+1));

	i = INDEX(i, blockIdx.x, BLOCKGX);
	j = INDEX(j, blockIdx.y, BLOCKGY);
	
	indg = INDEX(i,j,u_pitchX);

	const int NXM1 = NX-1;
	const int NYM1 = NY-1;
	const int NZM1 = NZ-1;

	const bool ActiveMemory = (i<NX) && (j<NY);
	const bool ActiveCompute = (i<NXM1) && (j<NYM1);
	

	if(ActiveMemory)
	{
		s_u[ind] = d_u[indg];
		s_d[ind] = d_den[indg];
	}
	
	if(halo)
	{
	  	s_u[ind_h] = d_u[indg_h];
		s_d[ind_h] = d_den[indg_h];
	}
	
	for(k=0;k<NZ;k++)
	{
		
		i = threadIdx.x;
		j = threadIdx.y;
		if(ActiveMemory)
		{
			
			indg = INDEX(indg, NY, u_pitchX);
	
			s_u[ind-((BLOCKGX+1)*(BLOCKGY+1))] = s_u[ind];
			s_d[ind-((BLOCKGX+1)*(BLOCKGY+1))] = s_d[ind];

			if(k<NZM1)
			{
			  	s_u[ind] = d_u[indg];
				s_d[ind] = d_den[indg];
			}
		}
		if(halo)
		{
			indg_h = INDEX(indg_h, NY, u_pitchX);

			s_u[ind_h-((BLOCKGX+1)*(BLOCKGY+1))] = s_u[ind_h];
			s_d[ind_h-((BLOCKGX+1)*(BLOCKGY+1))] = s_d[ind_h];

			if(k<NZM1)
			{
				s_u[ind_h] = d_u[indg_h];
				s_d[ind_h] = d_den[indg_h];
			}
		}
		
	
		__syncthreads();
		
	
		if(ActiveCompute)
		{
			
			REAL3 te0, te1, te2, te3, te4, te5, te6, te7;
			te0.x = 0.0;
			te0.y = 0.0;
			te0.z = 0.0;

			te1.x = 0.0;
			te1.y = 0.0;
			te1.z = 0.0;

			te2.x = 0.0;
			te2.y = 0.0;
			te2.z = 0.0;

			te3.x = 0.0;
			te3.y = 0.0;
			te3.z = 0.0;

			te4.x = 0.0;
			te4.y = 0.0;
			te4.z = 0.0;

			te5.x = 0.0;
			te5.y = 0.0;
			te5.z = 0.0;

			te6.x = 0.0;
			te6.y = 0.0;
			te6.z = 0.0;

			te7.x = 0.0;
			te7.y = 0.0;
			te7.z = 0.0;
		

			
			for(int ek1=0;ek1<2;ek1++)
			{
				for(int ej1=0;ej1<2;ej1++)
				{
					for(int ei1=0;ei1<2;ei1++)
					{
						const int LocID = General_Kernels::GPUGetLocalID(ei1, ej1, ek1);
						
						REAL3 MyU = s_u[INDEX(i+ei1, j+ej1, BLOCKGX+1) + ek1*((BLOCKGX+1)*(BLOCKGY+1))];
						

						te0.x += GPU_EleStiffone_t[0][LocID]*MyU.x;

						te1.x += GPU_EleStiffone_t[1][LocID]*MyU.x;

						te2.x += GPU_EleStiffone_t[3][LocID]*MyU.x;
				
						te3.x += GPU_EleStiffone_t[2][LocID]*MyU.x;

						te4.x += GPU_EleStiffone_t[4][LocID]*MyU.x;
					
						te5.x += GPU_EleStiffone_t[5][LocID]*MyU.x;
					
						te6.x += GPU_EleStiffone_t[7][LocID]*MyU.x;
						
						te7.x += GPU_EleStiffone_t[6][LocID]*MyU.x;
					
					} //end ei1-loop
				} //end ej1-loop
			} //end ek1-loop
			
			REAL Grad_e = 0.0;
			
			REAL3 MyU = s_u[INDEX(i+0, j+0, BLOCKGX+1) + 0*((BLOCKGX+1)*(BLOCKGY+1))];
			REAL Myden = s_d[INDEX(i+0, j+0, BLOCKGX+1) + 0*((BLOCKGX+1)*(BLOCKGY+1))];
		
			const REAL dDens = gpupexp*pow(Myden, gpupexp-(REAL)1.0);
			
			const REAL Dens = pow(Myden, gpupexp);

			Grad_e += MyU.x*te0.x ;
			
			MyU = s_u[INDEX(i+1, j+0, BLOCKGX+1) + 0*((BLOCKGX+1)*(BLOCKGY+1))];
			Grad_e += MyU.x*te1.x;
		
			MyU = s_u[INDEX(i+1, j+1, BLOCKGX+1) + 0*((BLOCKGX+1)*(BLOCKGY+1))];
			Grad_e += MyU.x*te2.x ;
			
			MyU = s_u[INDEX(i+0, j+1, BLOCKGX+1) + 0*((BLOCKGX+1)*(BLOCKGY+1))];
			Grad_e += MyU.x*te3.x ;
			
			MyU = s_u[INDEX(i+0, j+0, BLOCKGX+1) + 1*((BLOCKGX+1)*(BLOCKGY+1))];
			Grad_e += MyU.x*te4.x ;
		
			MyU = s_u[INDEX(i+1, j+0, BLOCKGX+1) + 1*((BLOCKGX+1)*(BLOCKGY+1))];
			Grad_e += MyU.x*te5.x ;
		
			MyU = s_u[INDEX(i+1, j+1, BLOCKGX+1) + 1*((BLOCKGX+1)*(BLOCKGY+1))];
			Grad_e += MyU.x*te6.x ;
	
			MyU = s_u[INDEX(i+0, j+1, BLOCKGX+1) + 1*((BLOCKGX+1)*(BLOCKGY+1))];
			Grad_e += MyU.x*te7.x ;
		

			const REAL ObjValue = (0.001 + (0.999 * Dens))*Grad_e;

			Grad_e *= -1 * 0.999 * dDens;

			i = INDEX(i, blockIdx.x, BLOCKGX);
			j = INDEX(j, blockIdx.y, BLOCKGY);

			const int MyStoreID = INDEX3D(i,j,k, grad_pitchX, NY);

			d_grad[MyStoreID] = Grad_e;

			d_ObjValue[MyStoreID] = ObjValue;
			
		}//end if active
		__syncthreads();
	}
}



void Thermalsim::InitGPU(REAL EleStiffone[8][8])
{
	cout << endl << "Initialize GPU" << endl;

	cudaMalloc((void**)&d_ResReduction_t, p2*sizeof(REAL));

	cout << "Copy reference stiffness" << endl;

	cudaMemcpyToSymbol(GPU_EleStiffone_t, EleStiffone, 8*8*sizeof(REAL));

	cout << "GPU initialized" << endl;

	cout << endl;

	GPUInitialized = true;
}


void Thermalsim::GPUMatVec(REAL3 *d_u, REAL *d_den, REAL *d_selection, REAL3 *d_res, size_t pitch_bytes,REAL pexp)
{
	dim3 threads(BLOCKSX,BLOCKSY,1);
        
    dim3 grids(ceil((NX)/ float(threads.x)), ceil((NY)/ float(threads.y)), 1);
	
	const int pitch = pitch_bytes/sizeof(REAL3);

	MatVecKernel_t<<<grids, threads>>>(NX, NY, NZ, pitch, d_u, d_den, d_selection, d_res, pexp);

	cudaDeviceSynchronize();
	
	

}


//computes the residual b-Ax inside the GPU
void Thermalsim::GPURes(REAL3 *d_u, REAL *d_den, REAL *d_selection, REAL3 *d_res, size_t pitch_bytes, REAL pexp)
{
	dim3 threads(BLOCKSX,BLOCKSY,1);
        
    dim3 grids(ceil((NX)/ float(threads.x)), ceil((NY)/ float(threads.y)), 1);
	
	const int pitch = pitch_bytes/sizeof(REAL3);

	ResidualKernel_t<<<grids, threads>>>(NX, NY, NZ, pitch, d_u, d_den,d_selection, d_res, pexp);

	cudaDeviceSynchronize();

}



void Thermalsim::GPUCG(REAL3 *d_u, REAL *d_den, REAL *d_selection, const int iter, const int OptIter, const REAL EndRes, int &FinalIter, REAL &FinalRes, REAL pexp)
{


	size_t pitch_bytes = sizeof(REAL3)* NX;
	REAL3 *d_res, *d_d, *d_q;
	cudaMalloc((void **)&d_res, sizeof(REAL3)*NX*NY*NZ);
	cudaMalloc((void **)&d_d, sizeof(REAL3)*NX*NY*NZ);
	cudaMalloc((void **)&d_q, sizeof(REAL3)*NX*NY*NZ);

	cudaMemset(d_ResReduction_t, 0.0, p2*sizeof(REAL));

	int iCounter = 0;

	GPURes(d_u, d_den, d_selection, d_res, pitch_bytes,pexp);


	//computing r^t * r
	General_Kernels::GPUScalarone(d_ResReduction_t, d_res, d_res, pitch_bytes, NX ,NY , NZ);

	cudaMemcpy2D(d_d, pitch_bytes, d_res, pitch_bytes, sizeof(REAL3)*NX, NY*NZ, cudaMemcpyDeviceToDevice);
	
	REAL g_ResBest;

	cudaMemcpy(&g_ResBest, d_ResReduction_t, sizeof(REAL), cudaMemcpyDeviceToHost);
	g_ResBest = sqrt(g_ResBest);

	cudaDeviceSynchronize();
	//compute r^T * r
	General_Kernels::GPUScalarone(d_ResReduction_t, d_res, d_d, pitch_bytes, NX ,NY, NZ);
	cudaDeviceSynchronize();
	REAL g_delta_new;
	
	cudaMemcpy(&g_delta_new, d_ResReduction_t, sizeof(REAL), cudaMemcpyDeviceToHost);

	
	const REAL term = EndRes*EndRes;
	
	while(iCounter < iter && g_delta_new > term)
	
	{
	
		GPUMatVec(d_d,d_den, d_selection, d_q, pitch_bytes,pexp);
		
		REAL g_temp;
		
		General_Kernels::GPUScalarone(d_ResReduction_t, d_d, d_q, pitch_bytes, NX , NY, NZ);
		cudaDeviceSynchronize();
		
	
		cudaMemcpy(&g_temp, d_ResReduction_t, sizeof(REAL), cudaMemcpyDeviceToHost);
		
		REAL g_alpha =g_delta_new/g_temp;
		
		General_Kernels::VecSMultAdd(d_u, 1.0, d_d, g_alpha, pitch_bytes, NX, NY, NZ);

		{
			General_Kernels::VecSMultAdd(d_res, 1.0, d_q, -1.0*g_alpha, pitch_bytes, NX, NY, NZ);
		
		}

		//r(i)^T * r(i) = d_delta_old
		REAL g_delta_old = g_delta_new;
		//r(i+1)^T * r(i+1)
		General_Kernels::GPUScalarone(d_ResReduction_t, d_res, d_res, pitch_bytes, NX, NY, NZ);
		cudaDeviceSynchronize();
		//g_delta_new = r(i+1)^T * r(i+1)
		cudaMemcpy(&g_delta_new, d_ResReduction_t, sizeof(REAL), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		REAL g_beta = g_delta_new/g_delta_old;
	 	General_Kernels::VecSMultAdd(d_d, g_beta, d_res, 1.0, pitch_bytes, NX, NY, NZ);
		cudaDeviceSynchronize();
		iCounter++;
	}


	FinalIter = iCounter;
	FinalRes = sqrt(g_delta_new);

	cudaFree(d_res);
	cudaFree(d_d);
	cudaFree(d_q);

}


void Thermalsim::GPUCompGrad(REAL3 *d_u, REAL *d_den, REAL *d_grad, REAL &Obj, REAL &Vol, const size_t u_pitch_bytes, size_t &grad_pitch_bytes, REAL pexp )
{
	dim3 threads(BLOCKGX,BLOCKGY,1);
	dim3 grids(ceil((NX)/ float(threads.x)), ceil((NY)/ float(threads.y)), 1);
	

	REAL *d_ObjValue;
	cudaMalloc((void **)&d_ObjValue,sizeof(REAL)* NX * NY* NZ);

	const int grad_pitchX = grad_pitch_bytes/sizeof(REAL);
	const int u_pitchX = u_pitch_bytes/sizeof(REAL3);
	cudaMemset(d_grad, (REAL)0.0, grad_pitchX* NY* NZ*sizeof(REAL));
	cudaMemset(d_ObjValue, (REAL)0.0, grad_pitchX* NY* NZ*sizeof(REAL));
	GPUEvalGradone_t<<<grids, threads>>>(NX, NY, NZ, u_pitchX, grad_pitchX, d_u, d_den, d_grad, d_ObjValue, pexp);
	cudaDeviceSynchronize();
	
	
	General_Kernels::GPUSum(d_ResReduction_t, d_ObjValue, grad_pitch_bytes, NX, NY ,NZ);
	cudaDeviceSynchronize();
	cudaMemcpy(&Obj, d_ResReduction_t, sizeof(REAL), cudaMemcpyDeviceToHost);

	
	General_Kernels::GPUVolume(d_ResReduction_t, d_den, u_pitch_bytes, NX, NY, NZ);
	cudaDeviceSynchronize();
	
	cudaMemcpy(&Vol, d_ResReduction_t, sizeof(REAL), cudaMemcpyDeviceToHost);
	cudaFree(d_ObjValue);

}



void Thermalsim::GPUCleanUp()
{
	if(d_ResReduction_t)
	{
		cudaFree(d_ResReduction_t);
	}
}
