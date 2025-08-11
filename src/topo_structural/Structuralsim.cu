

#include "Structuralsim.h"
#include "../general/DataTypes.h"
#include <iostream>
#include "../general/topopt_defines.h"
#include "../gpu_kernels/general_kernels.h"

using namespace std;



__constant__ REAL GPU_EleStiff[24][24];


__global__ void MatVecKernel(const int NX, const int NY, const int NZ, const int pitchX, REAL3 *d_u,REAL *d_den, REAL *d_selection, REAL3 *res, REAL gpupexp)
{

	int indg;

	int ind;
	
	int indg_h;

	int ind_h;
	
	const int mem_val = 3*(BLOCKSX+2)*(BLOCKSY+2);

	__shared__ REAL3 s_u[mem_val];
	__shared__ REAL s_d[mem_val];
	
	int k = threadIdx.x + threadIdx.y*(blockDim.x);

	int i,j;
	
	int halo = k < 2*(blockDim.x+blockDim.y+2);

	REAL select = 0.0;

	if(halo)
	{
		
		if(threadIdx.y < 2)
		{
			i = threadIdx.x;
			j = threadIdx.y*(blockDim.y+1) - 1;
		}
	
		else
		{
		
			i = (k%2)*(blockDim.x+1)-1;
			j = k/2 - blockDim.x - 1;
		}
		
		ind_h = INDEX(i+1,j+1,(BLOCKSX + 2)) + (blockDim.x + 2) * (blockDim.y  + 2);
		i = INDEX(i, blockIdx.x, blockDim.x);
		j = INDEX(j, blockIdx.y, blockDim.y);
		indg_h = INDEX(i, j, pitchX);
		halo = (i>=0) && (i<NX) && (j>=0) && (j<NY);
	}
	

	i = threadIdx.x;
	j = threadIdx.y;
	ind = INDEX(i+1,j+1,(BLOCKSX + 2)) + (blockDim.x + 2) * (blockDim.y  + 2);

	i = INDEX(i, blockIdx.x, blockDim.x);
	j = INDEX(j, blockIdx.y, blockDim.y);
	indg = INDEX(i,j,pitchX);
	
	const int active = (i<NX) && (j<NY);
	
	if(active) 
	{
		s_u[ind+(blockDim.x + 2) * (blockDim.y  + 2)] = d_u[indg];
		s_d[ind+(blockDim.x + 2) * (blockDim.y  + 2)] = d_den[indg];

	}

	if(halo) 
	{
		s_u[ind_h+(blockDim.x + 2) * (blockDim.y  + 2)] = d_u[indg_h];
		s_d[ind_h+(blockDim.x + 2) * (blockDim.y  + 2)] = d_den[indg_h];
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
			s_u[ind - (blockDim.x + 2) * (blockDim.y  + 2)] = s_u[ind];
			s_u[ind] = s_u[ind+(blockDim.x + 2) * (blockDim.y  + 2)];

			s_d[ind - (blockDim.x + 2) * (blockDim.y  + 2)] = s_d[ind];
			s_d[ind] = s_d[ind+(blockDim.x + 2) * (blockDim.y  + 2)];

			if(k<NZM1) 
			{
				s_u[ind + (blockDim.x + 2) * (blockDim.y  + 2)] = d_u[indg];
				s_d[ind + (blockDim.x + 2) * (blockDim.y  + 2)] = d_den[indg];

			}


			select = d_selection[indg_cur];

		}
		if(halo)
		{
			indg_h = INDEX(indg_h, NY, pitchX);
			s_u[ind_h-(blockDim.x + 2) * (blockDim.y  + 2)] = s_u[ind_h];
			s_u[ind_h] = s_u[ind_h + (blockDim.x + 2) * (blockDim.y  + 2)];
			
			s_d[ind_h -(blockDim.x + 2) * (blockDim.y  + 2)] = s_d[ind_h];
			s_d[ind_h] = s_d[ind_h + (blockDim.x + 2) * (blockDim.y  + 2)];

			if(k<NZM1) 
			{
				s_u[ind_h+(blockDim.x + 2) * (blockDim.y  + 2)] = d_u[indg_h];
				s_d[ind_h+(blockDim.x + 2) * (blockDim.y  + 2)] = d_den[indg_h];
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
				MyRes.y = 0.0;
				MyRes.z = 0.0;
			}
		
		
			// if(k==NZM1)
			// {
			// 	MyRes.x = 0.0;
			// 	MyRes.y = 0.0;
			// 	MyRes.z = 0.0;
			// }
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
										const REAL Dens = pow(s_d[ind-ei1*IOFF-ej1*(BLOCKSX + 2)-ek1*(blockDim.x + 2) * (blockDim.y  + 2)], gpupexp);
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
													const REAL3 MyU = s_u[ind+idiff*IOFF+jdiff*(BLOCKSX + 2)+kdiff*(blockDim.x + 2) * (blockDim.y  + 2)];
													MyRes.x += Dens*(GPU_EleStiff[LID1][LID2]*MyU.x + GPU_EleStiff[LID1][LID2+8]*MyU.y + GPU_EleStiff[LID1][LID2+16]*MyU.z);
													MyRes.y += Dens*(GPU_EleStiff[LID1+8][LID2]*MyU.x + GPU_EleStiff[LID1+8][LID2+8]*MyU.y + GPU_EleStiff[LID1+8][LID2+16]*MyU.z);
													MyRes.z += Dens*(GPU_EleStiff[LID1+16][LID2]*MyU.x + GPU_EleStiff[LID1+16][LID2+8]*MyU.y + GPU_EleStiff[LID1+16][LID2+16]*MyU.z);
										
										
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
	
			res[indg_cur] = MyRes;
		}//end if active
		__syncthreads();
	}
}





__global__ void ResidualKernel(const int NX, const int NY, const int NZ, const int pitchX, REAL3 *d_u,REAL *d_den, REAL *d_selection, REAL3 *res, REAL gpupexp)
{

	int indg;

	int ind;

	int indg_h;

	int ind_h;
	
	static const int mem_val = 3*(BLOCKSX + 2) * (BLOCKSY  + 2);
	///area of gridblock = (BLOCKX + 2) * (BLOCKY +2)
	__shared__ REAL3 s_u[mem_val];
	__shared__ REAL s_d[mem_val];
	

	int k = threadIdx.x + threadIdx.y*blockDim.x;

	int i,j;

	REAL select = 0.0;
	////	perimeter of gridblock = 2 * (BLOCKX + BLOCKY + 2)
	int halo = k < 2*(blockDim.x+blockDim.y+2);

	if(halo)
	{

		if(threadIdx.y<2)
		{
			i = threadIdx.x;
			///Minimum and maximum in Y direction
			j = threadIdx.y*(blockDim.y+1) - 1;			
		}
	
		else
		{
			///minimum and maximum in  X direction
			i = (k%2)*(blockDim.x+1)-1;
			j = k/2 - blockDim.x - 1;
			
		}

		ind_h = INDEX(i+1,j+1,(BLOCKSX + 2)) + (blockDim.x + 2) * (blockDim.y  + 2);
	
		i = INDEX(i, blockIdx.x, blockDim.x);
		j = INDEX(j, blockIdx.y, blockDim.y);

		indg_h = INDEX(i, j, pitchX);
		
		halo = (i>=0) && (i<NX) && (j>=0) && (j<NY);
		
	}

	i = threadIdx.x;
	j = threadIdx.y;
	ind = INDEX(i+1,j+1,(BLOCKSX + 2)) + (blockDim.x + 2) * (blockDim.y  + 2);
	i = INDEX(i, blockIdx.x, blockDim.x);
	j = INDEX(j, blockIdx.y, blockDim.y);

	indg = INDEX(i,j,pitchX);
	const int active = (i<NX) && (j<NY);
	////2nd XY plane in shared memory
	if(active) 
	{
		s_u[ind + (blockDim.x + 2) * (blockDim.y  + 2)] = d_u[indg];
		s_d[ind + (blockDim.x + 2) * (blockDim.y  + 2)] = d_den[indg];
	}
	
	if(halo) 
	{
		s_u[ind_h + (blockDim.x + 2) * (blockDim.y  + 2)] = d_u[indg_h];
		s_d[ind_h + (blockDim.x + 2) * (blockDim.y  + 2)] = d_den[indg_h];
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

			// 0th XY plane in shared memory
			s_u[ind - (blockDim.x + 2) * (blockDim.y  + 2)] = s_u[ind];
			s_d[ind - (blockDim.x + 2) * (blockDim.y  + 2)] = s_d[ind];
			// 1st XY plane in shared memory
			s_u[ind] = s_u[ind+(blockDim.x + 2) * (blockDim.y  + 2)];
			s_d[ind] = s_d[ind+(blockDim.x + 2) * (blockDim.y  + 2)];

			if(k<NZM1) 
			{
				s_u[ind+(blockDim.x + 2) * (blockDim.y  + 2)] = d_u[indg];
				s_d[ind+(blockDim.x + 2) * (blockDim.y  + 2)] = d_den[indg];

			}

			select = d_selection[indg_cur];

			
		}
		if(halo)
		{
			
			indg_h = INDEX(indg_h, NY, pitchX);
			s_u[ind_h-(blockDim.x + 2) * (blockDim.y  + 2)] = s_u[ind_h];
			s_u[ind_h] = s_u[ind_h + (blockDim.x + 2) * (blockDim.y  + 2)];

			s_d[ind_h - (blockDim.x + 2) * (blockDim.y  + 2)] = s_d[ind_h];
			s_d[ind_h] = s_d[ind_h + (blockDim.x + 2) * (blockDim.y  + 2)];


			if(k<NZM1) 
			{
				s_u[ind_h+(blockDim.x + 2) * (blockDim.y  + 2)] = d_u[indg_h];
				s_d[ind_h+(blockDim.x + 2) * (blockDim.y  + 2)] = d_den[indg_h];
			}
			
		}
		
	
		__syncthreads();
		

		if(active)
		{
			
			MyRes.x = 0.0;
			MyRes.y = 0.0;
			MyRes.z = 0.0;
		
			
			if(select == 1.0)
			{
				MyRes.y = -1.0;
			}

			if(select == -1.0)
			{
				MyRes.x = 0.0f;
				MyRes.y = 0.0f;
				MyRes.z = 0.0f;
			}

			// if(k == 0 && j == 0)
			// {
			// 	MyRes.y = -1.0;
			// }

			// if(k==NZM1)
			// {
			// 	MyRes.x = 0.0;
			// 	MyRes.y = 0.0;
			// 	MyRes.z = 0.0;
				
			// }
		
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
										const REAL Dens = pow(s_d[ind-ei1*IOFF-ej1*(BLOCKSX + 2)-ek1*(blockDim.x + 2) * (blockDim.y  + 2)], gpupexp);
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
													const REAL3 MyU = s_u[ind+idiff*IOFF+jdiff*(BLOCKSX + 2)+kdiff*(blockDim.x + 2) * (blockDim.y  + 2)];
													MyRes.x -= Dens*(GPU_EleStiff[LID1][LID2]*MyU.x + GPU_EleStiff[LID1][LID2+8]*MyU.y + GPU_EleStiff[LID1][LID2+16]*MyU.z);
													MyRes.y -= Dens*(GPU_EleStiff[LID1+8][LID2]*MyU.x + GPU_EleStiff[LID1+8][LID2+8]*MyU.y + GPU_EleStiff[LID1+8][LID2+16]*MyU.z);
													MyRes.z -= Dens*(GPU_EleStiff[LID1+16][LID2]*MyU.x + GPU_EleStiff[LID1+16][LID2+8]*MyU.y + GPU_EleStiff[LID1+16][LID2+16]*MyU.z);
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
			res[indg_cur] = MyRes;
		}//end if active
		__syncthreads();
	}
}




__global__ void GPUEvalGrad(const int NX, const int NY, const int NZ, const int u_pitchX, const int grad_pitchX, REAL3 *d_u,REAL *d_den, REAL *d_grad, REAL *d_ObjValue, REAL gpupexp)
{

	int indg;
	
	int ind;

	int indg_h;

	int ind_h;
	const int sh_mem = 2*((BLOCKGX + 1) * (BLOCKGY + 1));
	__shared__ REAL3 s_u[sh_mem];
	__shared__ REAL s_d[sh_mem];
	int k = threadIdx.x + threadIdx.y*blockDim.x;

	int i,j;
	
	bool halo = k < (blockDim.x + blockDim.y + 1);
	
	if(halo)
	{

		if(threadIdx.y==0)
		{
			i = threadIdx.x;
			j = blockDim.y;
		}
		
		else
		{
			i = blockDim.x;
			j = k - blockDim.x;
		}

		ind_h = INDEX(i,j,blockDim.x + 1) + ((blockDim.x + 1)*(blockDim.y+1));
		i = INDEX(i, blockIdx.x, blockDim.x);
		j = INDEX(j, blockIdx.y, blockDim.y);
		indg_h = INDEX(i, j, u_pitchX);
		
		halo = (i>=0) && (i<NX) && (j>=0) && (j<NY);
	}	

	i = threadIdx.x;
	j = threadIdx.y;

	ind = INDEX(i,j,blockDim.x+1) + ((blockDim.x+1)*(blockDim.y+1));
	
	i = INDEX(i, blockIdx.x, blockDim.x);
	j = INDEX(j, blockIdx.y, blockDim.y);

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
	
	for(k=0;k<NZM1;k++)
	{
		
		i = threadIdx.x;
		j = threadIdx.y;
		if(ActiveMemory)
		{

			indg = INDEX(indg, NY, u_pitchX);
			s_u[ind-((blockDim.x+1)*(blockDim.y+1))] = s_u[ind];
			s_d[ind-((blockDim.x+1)*(blockDim.y+1))] = s_d[ind];
			if(k<NZM1)
			{
			  	s_u[ind] = d_u[indg];
				s_d[ind] = d_den[indg];
			}
		}
		if(halo)
		{
			indg_h = INDEX(indg_h, NY, u_pitchX);
			s_u[ind_h-((blockDim.x+1)*(blockDim.y+1))] = s_u[ind_h];
			s_d[ind_h-((blockDim.x+1)*(blockDim.y+1))] = s_d[ind_h];
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
						REAL3 MyU = s_u[INDEX(i+ei1, j+ej1, blockDim.x+1) + ek1*((blockDim.x+1)*(blockDim.y+1))];
						te0.x += GPU_EleStiff[0][LocID]*MyU.x;
						te0.x += GPU_EleStiff[0][LocID+8]*MyU.y;
						te0.x += GPU_EleStiff[0][LocID+16]*MyU.z;
						te1.x += GPU_EleStiff[1][LocID]*MyU.x;
						te1.x += GPU_EleStiff[1][LocID+8]*MyU.y;
						te1.x += GPU_EleStiff[1][LocID+16]*MyU.z;
						te2.x += GPU_EleStiff[2][LocID]*MyU.x;
						te2.x += GPU_EleStiff[2][LocID+8]*MyU.y;
						te2.x += GPU_EleStiff[2][LocID+16]*MyU.z;
						te3.x += GPU_EleStiff[3][LocID]*MyU.x;
						te3.x += GPU_EleStiff[3][LocID+8]*MyU.y;
						te3.x += GPU_EleStiff[3][LocID+16]*MyU.z;
						te4.x += GPU_EleStiff[4][LocID]*MyU.x;
						te4.x += GPU_EleStiff[4][LocID+8]*MyU.y;
						te4.x += GPU_EleStiff[4][LocID+16]*MyU.z;
						te5.x += GPU_EleStiff[5][LocID]*MyU.x;
						te5.x += GPU_EleStiff[5][LocID+8]*MyU.y;
						te5.x += GPU_EleStiff[5][LocID+16]*MyU.z;
						te6.x += GPU_EleStiff[6][LocID]*MyU.x;
						te6.x += GPU_EleStiff[6][LocID+8]*MyU.y;
						te6.x += GPU_EleStiff[6][LocID+16]*MyU.z;
						te7.x += GPU_EleStiff[7][LocID]*MyU.x;
						te7.x += GPU_EleStiff[7][LocID+8]*MyU.y;
						te7.x += GPU_EleStiff[7][LocID+16]*MyU.z;

						te0.y += GPU_EleStiff[8][LocID]*MyU.x;
						te0.y += GPU_EleStiff[8][LocID+8]*MyU.y;
						te0.y += GPU_EleStiff[8][LocID+16]*MyU.z;
						te1.y += GPU_EleStiff[9][LocID]*MyU.x;
						te1.y += GPU_EleStiff[9][LocID+8]*MyU.y;
						te1.y += GPU_EleStiff[9][LocID+16]*MyU.z;
						te2.y += GPU_EleStiff[10][LocID]*MyU.x;
						te2.y += GPU_EleStiff[10][LocID+8]*MyU.y;
						te2.y += GPU_EleStiff[10][LocID+16]*MyU.z;
						te3.y += GPU_EleStiff[11][LocID]*MyU.x;
						te3.y += GPU_EleStiff[11][LocID+8]*MyU.y;
						te3.y += GPU_EleStiff[11][LocID+16]*MyU.z;
						te4.y += GPU_EleStiff[12][LocID]*MyU.x;
						te4.y += GPU_EleStiff[12][LocID+8]*MyU.y;
						te4.y += GPU_EleStiff[12][LocID+16]*MyU.z;
						te5.y += GPU_EleStiff[13][LocID]*MyU.x;
						te5.y += GPU_EleStiff[13][LocID+8]*MyU.y;
						te5.y += GPU_EleStiff[13][LocID+16]*MyU.z;
						te6.y += GPU_EleStiff[14][LocID]*MyU.x;
						te6.y += GPU_EleStiff[14][LocID+8]*MyU.y;
						te6.y += GPU_EleStiff[14][LocID+16]*MyU.z;
						te7.y += GPU_EleStiff[15][LocID]*MyU.x;
						te7.y += GPU_EleStiff[15][LocID+8]*MyU.y;
						te7.y += GPU_EleStiff[15][LocID+16]*MyU.z;

						te0.z += GPU_EleStiff[16][LocID]*MyU.x;
						te0.z += GPU_EleStiff[16][LocID+8]*MyU.y;
						te0.z += GPU_EleStiff[16][LocID+16]*MyU.z;
						te1.z += GPU_EleStiff[17][LocID]*MyU.x;
						te1.z += GPU_EleStiff[17][LocID+8]*MyU.y;
						te1.z += GPU_EleStiff[17][LocID+16]*MyU.z;
						te2.z += GPU_EleStiff[18][LocID]*MyU.x;
						te2.z += GPU_EleStiff[18][LocID+8]*MyU.y;
						te2.z += GPU_EleStiff[18][LocID+16]*MyU.z;
						te3.z += GPU_EleStiff[19][LocID]*MyU.x;
						te3.z += GPU_EleStiff[19][LocID+8]*MyU.y;
						te3.z += GPU_EleStiff[19][LocID+16]*MyU.z;
						te4.z += GPU_EleStiff[20][LocID]*MyU.x;
						te4.z += GPU_EleStiff[20][LocID+8]*MyU.y;
						te4.z += GPU_EleStiff[20][LocID+16]*MyU.z;
						te5.z += GPU_EleStiff[21][LocID]*MyU.x;
						te5.z += GPU_EleStiff[21][LocID+8]*MyU.y;
						te5.z += GPU_EleStiff[21][LocID+16]*MyU.z;
						te6.z += GPU_EleStiff[22][LocID]*MyU.x;
						te6.z += GPU_EleStiff[22][LocID+8]*MyU.y;
						te6.z += GPU_EleStiff[22][LocID+16]*MyU.z;
						te7.z += GPU_EleStiff[23][LocID]*MyU.x;
						te7.z += GPU_EleStiff[23][LocID+8]*MyU.y;
						te7.z += GPU_EleStiff[23][LocID+16]*MyU.z;
					} //end ei1-loop
				} //end ej1-loop
			} //end ek1-loop
			//compute u_e^T * te = u_e^T*K*u_e
			REAL Grad_e = 0.0;

			REAL3 MyU = s_u[INDEX(i+0, j+0, blockDim.x+1) + 0*((blockDim.x+1)*(blockDim.y+1))];
			REAL Myden = s_d[INDEX(i+0, j+0, blockDim.x+1) + 0*((blockDim.x+1)*(blockDim.y+1))];
			const REAL dDens = gpupexp*pow(Myden, gpupexp-(REAL)1.0);
			const REAL Dens = pow(Myden, gpupexp);

			Grad_e += MyU.x*te0.x + MyU.y*te0.y + MyU.z*te0.z;
	
			MyU = s_u[INDEX(i+1, j+0, blockDim.x+1) + 0*((blockDim.x+1)*(blockDim.y+1))];
			Grad_e += MyU.x*te1.x + MyU.y*te1.y + MyU.z*te1.z;
	
			MyU = s_u[INDEX(i+0, j+1, blockDim.x+1) + 0*((blockDim.x+1)*(blockDim.y+1))];
			Grad_e += MyU.x*te2.x + MyU.y*te2.y + MyU.z*te2.z;
	
			MyU = s_u[INDEX(i+1, j+1, blockDim.x+1) + 0*((blockDim.x+1)*(blockDim.y+1))];
			Grad_e += MyU.x*te3.x + MyU.y*te3.y + MyU.z*te3.z;

			MyU = s_u[INDEX(i+0, j+0, blockDim.x+1) + 1*((blockDim.x+1)*(blockDim.y+1))];
			Grad_e += MyU.x*te4.x + MyU.y*te4.y + MyU.z*te4.z;

			MyU = s_u[INDEX(i+1, j+0, blockDim.x+1) + 1*((blockDim.x+1)*(blockDim.y+1))];
			Grad_e += MyU.x*te5.x + MyU.y*te5.y + MyU.z*te5.z;

			MyU = s_u[INDEX(i+0, j+1, blockDim.x+1) + 1*((blockDim.x+1)*(blockDim.y+1))];
			Grad_e += MyU.x*te6.x + MyU.y*te6.y + MyU.z*te6.z;

			MyU = s_u[INDEX(i+1, j+1, blockDim.x+1) + 1*((blockDim.x+1)*(blockDim.y+1))];
			Grad_e += MyU.x*te7.x + MyU.y*te7.y + MyU.z*te7.z;

			const REAL ObjValue = Dens*Grad_e;

			Grad_e *= -1.0*dDens;

			i = INDEX(i, blockIdx.x, blockDim.x);
			j = INDEX(j, blockIdx.y, blockDim.y);

			const int MyStoreID = INDEX3D(i,j,k, grad_pitchX, NY);

			d_grad[MyStoreID] = Grad_e;

			d_ObjValue[MyStoreID] = ObjValue;

		
			
		}//end if active
		__syncthreads();
	}
}

void Structuralsim::InitGPU(REAL EleStiff[24][24])
{
	std::cout << endl << "Initialize GPU" << endl;
	
	cudaMalloc((void**)&d_ResReduction, p2*sizeof(REAL));
	std::cout << "Copy reference stiffness" << endl;
	cudaMemcpyToSymbol(GPU_EleStiff, EleStiff, 24*24*sizeof(REAL));
	std::cout << "GPU initialized" << endl;
	std::cout << endl;
	GPUInitialized = true;
}

//computes the scalar product Ax inside the GPU
void Structuralsim::GPUMatVec(REAL3 *d_u, REAL *d_den, REAL *d_selection, REAL3 *d_res, size_t pitch_bytes, const REAL pexp)
{
	

	
	
	dim3 threads(BLOCKSX,BLOCKSY,1);
        
    dim3 grids(ceil((NX)/ float(threads.x)), ceil((NY)/ float(threads.y)), 1);
	const int pitch = pitch_bytes/sizeof(REAL3);
	MatVecKernel<<<grids,threads>>>( NX,  NY,  NZ, pitch, d_u, d_den,d_selection, d_res, pexp);
	cudaDeviceSynchronize();




}

//computes the residual b-Ax inside the GPU
void Structuralsim::GPURes(REAL3 *d_u,REAL *d_den, REAL *d_selection, REAL3 *d_res, size_t pitch_bytes, const REAL pexp)
{
	

		
	
	
	dim3 threads(BLOCKSX,BLOCKSY,1);
        
    dim3 grids(ceil((NX)/ float(threads.x)), ceil((NY)/ float(threads.y)), 1);
	const int pitch = pitch_bytes/sizeof(REAL3);

	ResidualKernel<<<grids,threads>>>( NX,  NY,  NZ, pitch, d_u, d_den,d_selection, d_res, pexp);

	cudaDeviceSynchronize();

	
}




void Structuralsim::GPUCG(REAL3 *d_u,REAL *d_den, REAL *d_selection, const int iter, const int OptIter, const REAL EndRes, int &FinalIter, REAL &FinalRes, const REAL pexp)
{

	size_t pitch_bytes = sizeof(REAL3)* NX;
	REAL3 *d_res, *d_d, *d_q, *d_val;
	cudaMalloc((void **)&d_res, sizeof(REAL3)* NX * NY* NZ);
	cudaMalloc((void **)&d_d, sizeof(REAL3)* NX * NY* NZ);
	cudaMalloc((void **)&d_q, sizeof(REAL3)* NX * NY* NZ);

	cudaMalloc((void **)&d_val, sizeof(REAL3)* NX * NY* NZ);
	
	

	cudaMemset(d_ResReduction, 0.0, p2*sizeof(REAL));

	int iCounter = 1;
	//r = b - Ax
	GPURes(d_u,d_den,d_selection, d_res, pitch_bytes,pexp);
	cudaMemcpy(d_val,d_res, sizeof(REAL3) *NX*NY*NZ, cudaMemcpyDeviceToDevice);
	// p = r
	cudaMemcpy(d_d,d_res, sizeof(REAL3)* NX * NY* NZ, cudaMemcpyDeviceToDevice);

	//computing r^t * r
	General_Kernels::GPUScalar(d_ResReduction, d_res, d_res, pitch_bytes, NX , NY, NZ);

	
	REAL g_ResBest;
	cudaMemcpy(&g_ResBest, d_ResReduction, sizeof(REAL), cudaMemcpyDeviceToHost);
	g_ResBest = sqrt(g_ResBest);

	cudaDeviceSynchronize();
	//compute r^T * r
	General_Kernels::GPUScalar(d_ResReduction, d_res, d_d, pitch_bytes, NX , NY, NZ);
	cudaDeviceSynchronize();
	REAL g_delta_new;
	cudaMemcpy(&g_delta_new, d_ResReduction, sizeof(REAL), cudaMemcpyDeviceToHost);
	const REAL term = EndRes*EndRes;
	
	while(iCounter < iter && g_delta_new > term)
	{
	
		//q=Ad
		GPUMatVec(d_d,d_den,d_selection, d_q, pitch_bytes,pexp);

		REAL g_temp;

		General_Kernels::GPUScalar(d_ResReduction, d_d, d_q, pitch_bytes, NX , NY, NZ);

		cudaDeviceSynchronize();
		cudaMemcpy(&g_temp, d_ResReduction, sizeof(REAL), cudaMemcpyDeviceToHost);

		REAL g_alpha = g_delta_new/g_temp;
		General_Kernels::VecSMultAdd(d_u, 1.0, d_d, g_alpha, pitch_bytes,  NX,  NY,  NZ);

		{
			General_Kernels::VecSMultAdd(d_res, 1.0, d_q, -1.0*g_alpha, pitch_bytes,  NX,  NY,  NZ);
		}

		REAL g_delta_old = g_delta_new;

		General_Kernels::GPUScalar(d_ResReduction, d_res, d_res, pitch_bytes, NX ,NY , NZ);
		
		cudaDeviceSynchronize();
		//g_delta_new = r(i+1)^T * r(i+1)
		cudaMemcpy(&g_delta_new, d_ResReduction, sizeof(REAL), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
		//'beta'(i) = (r(i+1)^T * r(i+1))/(r(i)^T*r(i))
		REAL g_beta = g_delta_new/g_delta_old;
		//P(i+1) = 'beta'(i)*d_d + 1.0* r(i+1)
		General_Kernels::VecSMultAdd(d_d, g_beta, d_res, 1.0, pitch_bytes,  NX,  NY,  NZ);
		cudaDeviceSynchronize();
		iCounter++;
	
	}


	FinalIter = iCounter;
	FinalRes = sqrt(g_delta_new);
	cudaFree(d_res);
	cudaFree(d_d);
	cudaFree(d_q);
	cudaFree(d_val);


}

void Structuralsim::GPUCompGrad(REAL3 *d_u,REAL *d_den, REAL *d_grad, REAL &Obj, REAL &Vol, const size_t u_pitch_bytes, size_t &grad_pitch_bytes, const REAL pexp)
{



	dim3 threads(BLOCKGX,BLOCKGY,1);
        
    dim3 grids(ceil((NX)/ float(threads.x)), ceil((NY)/ float(threads.y)), 1);

	REAL *d_ObjValue;
	cudaMalloc((void **)&d_ObjValue,sizeof(REAL)* NX * NY* NZ);

	const int grad_pitchX = NX;
	const int u_pitchX = NX;
	cudaMemset(d_grad, (REAL)0.0, grad_pitchX* NY* NZ*sizeof(REAL));
	cudaMemset(d_ObjValue, (REAL)0.0, grad_pitchX* NY* NZ*sizeof(REAL));
	cudaDeviceSynchronize();
	GPUEvalGrad<<<grids, threads>>>( NX,  NY,  NZ, u_pitchX, grad_pitchX, d_u,d_den, d_grad, d_ObjValue, pexp);
	cudaDeviceSynchronize();

	General_Kernels::GPUSum(d_ResReduction, d_ObjValue, grad_pitch_bytes, NX, NY, NZ);

	cudaDeviceSynchronize();
	cudaMemcpy(&Obj, d_ResReduction, sizeof(REAL), cudaMemcpyDeviceToHost);

	General_Kernels::GPUVolume(d_ResReduction,d_den, u_pitch_bytes, NX, NY, NZ);
	cudaDeviceSynchronize();

	cudaMemcpy(&Vol, d_ResReduction, sizeof(REAL), cudaMemcpyDeviceToHost);
	cudaFree(d_ObjValue);

}

void Structuralsim::GPUCleanUp()
{
	if(d_ResReduction)
	{
		cudaFree(d_ResReduction);
	}
}







