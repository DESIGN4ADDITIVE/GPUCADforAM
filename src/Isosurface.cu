#include "Isosurface.h"


Isosurface::Isosurface()
{
    
}

Isosurface::~Isosurface()
{

}


void Isosurface::copy_parameter(uint *voxel_verts, uint * voxel_vertsscan, float isoValue,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,uint numVoxels,
    uint *activeVoxels,uint *d_compVoxelArray, grid_points *vol_one,float* vol_two,float *vol_lattice,bool fixed, bool dynamic,float iso1, float iso2, uint *totalverts_1, bool obj_union, bool obj_diff, bool obj_intersect)
    
    {
          
        dim3 grid(ceil(numVoxels/float(1024)), 1, 1);
        dim3 threads(1024,1,1);

        classify_copy_Voxel_lattice(grid,threads,voxel_verts,vol_one,vol_two, vol_lattice, fixed, dynamic, iso1, iso2,
                            gridSize, gridSizeShift, gridSizeMask,numVoxels,voxelSize,isoValue,obj_union, obj_diff, obj_intersect);
        
    }

void Isosurface::computeIsosurface(float4* pos , float4* norm, float isoValue,
    uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
    uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, grid_points  *primitive_fixed,float *primitive_dynamic, float *topo_field,float *lattice_field, 
    float iso1,float iso2, bool obj_union, bool obj_diff, bool obj_intersect , bool primitive, bool topo, bool compute_lattice, bool fixed, bool dynamic)
    
    {
        
        dim3 grid(ceil(numVoxels/float(1024)), 1, 1);
        dim3 threads(1024,1,1);
      
        if (grid.x > 65535)
        {
            grid.y = grid.x / 32768;
            grid.x = 32768;
        }
    
        classifyVoxel_lattice(grid,threads,
                            d_voxelVerts,d_voxelOccupied,primitive_fixed,primitive_dynamic,topo_field,lattice_field,
                            gridSize, gridSizeShift, gridSizeMask,numVoxels, iso1, iso2, voxelSize, isoValue, obj_union, obj_diff, obj_intersect, primitive, topo, compute_lattice,fixed,dynamic);
 
        ////// Numbering active voxels ///////
        
        ThrustScanWrapper_lattice(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);
    
        {
            uint lastElement, lastScanElement;
            checkCudaErrors(cudaMemcpy((void *) &lastElement,
                                    (void *)(d_voxelOccupied + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy((void *) &lastScanElement,
                                    (void *)(d_voxelOccupiedScan + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            *activeVoxels = lastElement + lastScanElement;
           
        }

        if (*activeVoxels == 0)
        {
            *totalVerts = 0;
            return;
        }
    
        dim3 gids(ceil((numVoxels)/float(1024)), 1, 1);
        dim3 tids(1024,1,1);
    
    
        compactVoxels_lattice(gids, tids, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);
        getLastCudaError("compactVoxels failed");

        /////////Finding totalverts ////////////

        ThrustScanWrapper_lattice(d_voxelVertsScan, d_voxelVerts, numVoxels);

        
    
        {
            uint totalverts_1;
            uint lastElement_1, lastScanElement_1;
            
            checkCudaErrors(cudaMemcpy((void *) &lastElement_1,
                                    (void *)(d_voxelVerts + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy((void *) &lastScanElement_1,
                                    (void *)(d_voxelVertsScan + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));

            totalverts_1 = lastElement_1 + lastScanElement_1;
            
            *totalVerts = totalverts_1;
            
        }
       
        checkCudaErrors(cudaMemset(pos,0,maxVerts));
        checkCudaErrors(cudaMemset(norm,0,maxVerts));

        dim3 grid2((int) ceil(*activeVoxels / (float)NTHREADS), 1, 1);
       
        dim3 tids2(NTHREADS,1,1);

        generateTriangles_lattice(grid2, tids2, pos, norm,d_compVoxelArray,
                                d_voxelVertsScan,
                                gridSize, gridSizeShift, gridSizeMask,
                                voxelSize,gridcenter, isoValue,*activeVoxels,
                                maxVerts,*totalVerts,primitive_fixed,primitive_dynamic,topo_field,lattice_field,iso1,iso2,d_voxelVerts,obj_union, obj_diff, obj_intersect,
                                primitive, topo, compute_lattice,fixed,dynamic);

    }




    void Isosurface::computeIsosurface_2(float4* pos , float4* norm, float isoValue,
    uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
    uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, float* vol_one, float isovalue1)
    {
        
        dim3 grid(ceil(numVoxels/float(1024)), 1, 1);
        dim3 threads(1024,1,1);
    
        classifyVoxel_lattice_2(grid,threads,
                            d_voxelVerts, d_voxelOccupied, vol_one,
                            gridSize, gridSizeShift, gridSizeMask,
                            numVoxels, voxelSize, isoValue);
 
        ////// Numbering active voxels ///////
        
        ThrustScanWrapper_lattice(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);
    
        {
            uint lastElement, lastScanElement;
            checkCudaErrors(cudaMemcpy((void *) &lastElement,
                                    (void *)(d_voxelOccupied + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy((void *) &lastScanElement,
                                    (void *)(d_voxelOccupiedScan + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            *activeVoxels = lastElement + lastScanElement;
        }

        if (*activeVoxels == 0)
        {
            *totalVerts = 0;
            return;
        }
    
        dim3 gids(ceil((numVoxels)/float(1024)), 1, 1);
        dim3 tids(1024,1,1);
    
    
        compactVoxels_lattice(gids, tids, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);
        getLastCudaError("compactVoxels failed");

        /////////Finding totalverts ////////////

        ThrustScanWrapper_lattice(d_voxelVertsScan, d_voxelVerts, numVoxels);

        
    
        {
            uint totalverts_1;
            uint lastElement_1, lastScanElement_1;
            
            checkCudaErrors(cudaMemcpy((void *) &lastElement_1,
                                    (void *)(d_voxelVerts + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy((void *) &lastScanElement_1,
                                    (void *)(d_voxelVertsScan + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));

            totalverts_1 = lastElement_1 + lastScanElement_1;
            
            *totalVerts = totalverts_1;
            
        }

        dim3 grid2((int) ceil(*activeVoxels / (float)NTHREADS), 1, 1);
       
        dim3 tids2(NTHREADS,1,1);
       
        generateTriangles_lattice_2(grid2, tids2, pos, norm,
                                d_compVoxelArray,
                                d_voxelVertsScan,
                                gridSize, gridSizeShift, gridSizeMask,
                                voxelSize,gridcenter, isoValue, *activeVoxels,
                                maxVerts,*totalVerts,vol_one,isovalue1);        
    }


    void Isosurface::computeIsosurface_lattice(float* vol, float4* pos , float4* norm, float &isoValue,
    uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
    uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, float* vol_one,float* vol_two, 
    float isovalue1,float isovalue2,float iso1, float iso2)
    {
        
        dim3 grid(ceil(numVoxels/float(1024)), 1, 1);
        dim3 threads(1024,1,1);
      
        if (grid.x > 65535)
        {
            grid.y = grid.x / 32768;
            grid.x = 32768;
        }
    
        classifyVoxel_lattice_new(grid,threads,
                            d_voxelVerts, d_voxelOccupied, vol,
                            gridSize, gridSizeShift, gridSizeMask,
                            numVoxels, voxelSize, isoValue);
 
        ////// Numbering active voxels ///////
        
        ThrustScanWrapper_lattice(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);
    
        {
            uint lastElement, lastScanElement;
            checkCudaErrors(cudaMemcpy((void *) &lastElement,
                                    (void *)(d_voxelOccupied + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy((void *) &lastScanElement,
                                    (void *)(d_voxelOccupiedScan + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            *activeVoxels = lastElement + lastScanElement;
         
        }

        if (*activeVoxels == 0)
        {
            *totalVerts = 0;
            return;
        }
    
        dim3 gids(ceil((numVoxels)/float(1024)), 1, 1);
        dim3 tids(1024,1,1);
    
    
        compactVoxels_lattice(gids, tids, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);
        getLastCudaError("compactVoxels failed");

        /////////Finding totalverts ////////////

        ThrustScanWrapper_lattice(d_voxelVertsScan, d_voxelVerts, numVoxels);

        
    
        {
            uint totalverts_1;
            uint lastElement_1, lastScanElement_1;
            
            checkCudaErrors(cudaMemcpy((void *) &lastElement_1,
                                    (void *)(d_voxelVerts + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy((void *) &lastScanElement_1,
                                    (void *)(d_voxelVertsScan + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));

            totalverts_1 = lastElement_1 + lastScanElement_1;
            
            *totalVerts = totalverts_1;
            
        }

       
        dim3 grid2((int) ceil(*activeVoxels / (float)NTHREADS), 1, 1);
       
        dim3 tids2(NTHREADS,1,1);
       
        generateTriangles_lattice_new(grid2, tids2, pos, norm,
                                d_compVoxelArray,
                                d_voxelVertsScan, vol,
                                gridSize, gridSizeShift, gridSizeMask,
                                voxelSize,gridcenter, isoValue, *activeVoxels,
                                maxVerts,*totalVerts,vol_one,vol_two,isovalue1,isovalue2,iso1,iso2);

    }

    void Isosurface::computeIsosurface_latticeone(float* vol, float4* pos , float4* norm, float &isoValue,
    uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
    uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, float* vol_one,
    float isovalue1,float isovalue2)
    {
        
        dim3 grid(ceil(numVoxels/float(1024)), 1, 1);
        dim3 threads(1024,1,1);
      
        if (grid.x > 65535)
        {
            grid.y = grid.x / 32768;
            grid.x = 32768;
        }
    
        classifyVoxel_lattice_new(grid,threads,
                            d_voxelVerts, d_voxelOccupied, vol,
                            gridSize, gridSizeShift, gridSizeMask,
                            numVoxels, voxelSize, isoValue);
 
        ////// Numbering active voxels ///////
        
        ThrustScanWrapper_lattice(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);
    
        {
            uint lastElement, lastScanElement;
            checkCudaErrors(cudaMemcpy((void *) &lastElement,
                                    (void *)(d_voxelOccupied + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy((void *) &lastScanElement,
                                    (void *)(d_voxelOccupiedScan + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            *activeVoxels = lastElement + lastScanElement;
          
        }

        if (*activeVoxels == 0)
        {
            *totalVerts = 0;
            return;
        }
    
        dim3 gids(ceil((numVoxels)/float(1024)), 1, 1);
        dim3 tids(1024,1,1);
    
    
        compactVoxels_lattice(gids, tids, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);
        getLastCudaError("compactVoxels failed");

        /////////Finding totalverts ////////////

        ThrustScanWrapper_lattice(d_voxelVertsScan, d_voxelVerts, numVoxels);

        
    
        {
            uint totalverts_1;
            uint lastElement_1, lastScanElement_1;
            
            checkCudaErrors(cudaMemcpy((void *) &lastElement_1,
                                    (void *)(d_voxelVerts + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy((void *) &lastScanElement_1,
                                    (void *)(d_voxelVertsScan + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));

            totalverts_1 = lastElement_1 + lastScanElement_1;
            
            *totalVerts = totalverts_1;
            
        }
       
        dim3 grid2((int) ceil(*activeVoxels / (float)NTHREADS), 1, 1);
       
        dim3 tids2(NTHREADS,1,1);

        generateTriangles_lattice_newone(grid2, tids2, pos, norm,
                                d_compVoxelArray,
                                d_voxelVertsScan, vol,
                                gridSize, gridSizeShift, gridSizeMask,
                                voxelSize,gridcenter, isoValue, *activeVoxels,
                                maxVerts,*totalVerts,vol_one,isovalue1,isovalue2); 
        
    }



__global__ void patch_grid_kernel(float *d_vec1, int Nx, int Ny, int Nz, float isoval)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 

	int xx = tx%Nx;
	int yy = (tx%(Nx*Ny))/Ny;
	int zz = tx/(Nx*Ny);

    float k;

	if(tx < (Nx * Ny * Nz))
	{

		k = d_vec1[tx];
		if ((xx == 0) || (xx == (Nx-1)) || (yy == 0) || (yy == (Ny-1)) || (zz == 0) || (zz == (Nz-1)))
		{
			if(k > isoval)
            {
			    k = 0.0f;
            }

		}

        d_vec1[tx] = k;

    }


}


void Isosurface::patch_grid(float *d_vec1 , int Nx, int Ny, int Nz, float isoval )
{
    dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);

	dim3 tids(1024,1,1);

    patch_grid_kernel<<<grids,tids>>>(d_vec1,Nx,Ny,Nz,isoval);

    cudaDeviceSynchronize();

}