#include "Isosurface.h"


Isosurface::Isosurface()
{
    
}

Isosurface::~Isosurface()
{

}


void Isosurface::copy_parameter( uint *voxel_verts, float isoValue,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize,uint numVoxels,
    grid_points *vol_one,float* vol_two,float *vol_lattice,bool fixed, bool dynamic,float iso1, float iso2, bool obj_union, bool obj_diff, bool obj_intersect)
    
    {
          
        dim3 grid(ceil((gridSize.x*gridSize.y*gridSize.z)/float(1024)), 1, 1);
        dim3 threads(1024,1,1);

        classify_copy_Voxel_lattice(grid,threads,voxel_verts,vol_one,vol_two, vol_lattice, fixed, dynamic, iso1, iso2,
                            gridSize, gridSizeShift, gridSizeMask,numVoxels,voxelSize,isoValue,obj_union, obj_diff, obj_intersect);
        
    }


void Isosurface::copy_regions( uint *voxel_verts, float isoValue,
uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize,uint numVoxels,grid_points *vol_topo,
grid_points *vol_one, float* vol_two,float *vol_lattice,bool fixed, bool dynamic,float iso1, float iso2, bool obj_union, bool obj_diff, bool obj_intersect)

{
        
    dim3 grid(ceil(numVoxels/float(1024)), 1, 1);
    dim3 threads(1024,1,1);

    classify_copy_regions(grid,threads,voxel_verts,vol_topo,vol_one,vol_two, vol_lattice, fixed, dynamic, iso1, iso2,
                        gridSize, gridSizeShift, gridSizeMask,numVoxels,voxelSize,isoValue,obj_union, obj_diff, obj_intersect);
    
}

void Isosurface::computeIsosurface(float *vol, uint3 raster_grid, float4* pos , float4* norm, float isoValue,
    uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
    uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, grid_points  *primitive_fixed,float *primitive_dynamic, float *topo_field,float *lattice_field, 
    float iso1,float iso2, bool obj_union, bool obj_diff, bool obj_intersect , bool primitive, bool topo, bool compute_lattice, bool fixed, bool dynamic,
    bool make_region, size_t *nfacets)
    
    {
        
        dim3 grid(ceil(numVoxels/float(1024)), 1, 1);
        dim3 threads(1024,1,1);
      
        if (grid.x > 65535)
        {
            grid.y = grid.x / 32768;
            grid.x = 32768;
        }
        
        
        classifyVoxel_lattice(grid,threads,vol,raster_grid,
                            d_voxelVerts,d_voxelOccupied,primitive_fixed,primitive_dynamic,topo_field,lattice_field,
                            gridSize, gridSizeShift, gridSizeMask,numVoxels, iso1, iso2, voxelSize, isoValue, obj_union, obj_diff, obj_intersect, primitive, topo, compute_lattice,fixed,dynamic,
                            make_region);
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
            *nfacets = *totalVerts/3;
            
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
                                primitive, topo, compute_lattice,fixed,dynamic,make_region);

    }


    void Isosurface::compute_solid_voxels(float isoValue,uint numVoxels,uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, grid_points  *primitive_fixed,
    float *d_solid_field )
    
    {
        
        dim3 grid(ceil(numVoxels/float(1024)), 1, 1);
        dim3 threads(1024,1,1);
      
        if (grid.x > 65535)
        {
            grid.y = grid.x / 32768;
            grid.x = 32768;
        }
        
        
        classify_solid_voxels(grid,threads,primitive_fixed,gridSize, gridSizeShift, gridSizeMask,numVoxels,voxelSize, isoValue, d_solid_field);
      

    }

    void Isosurface::computeIsosurface_region( float4* pos , float4* norm, float isoValue,
    uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
    uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, grid_points * vol_topo, grid_points  *primitive_fixed,float *primitive_dynamic, float *topo_field,float *lattice_field, 
    float iso1,float iso2, bool obj_union, bool obj_diff, bool obj_intersect , bool primitive, bool topo, bool compute_lattice, bool fixed, bool dynamic,
    bool make_region, bool show_region, bool show_domain, triangle_metadata *triangle_data)
    
    {
        
        dim3 grid(ceil(numVoxels/float(1024)), 1, 1);
        dim3 threads(1024,1,1);
      
        if (grid.x > 65535)
        {
            grid.y = grid.x / 32768;
            grid.x = 32768;
        }
        
        
        classifyVoxel_region(grid,threads,
                            d_voxelVerts,d_voxelOccupied,vol_topo ,primitive_fixed,primitive_dynamic,topo_field,lattice_field,
                            gridSize, gridSizeShift, gridSizeMask,numVoxels, iso1, iso2, voxelSize, isoValue, obj_union, obj_diff, obj_intersect, primitive, topo, compute_lattice,fixed,dynamic,
                            make_region, show_region, show_domain);
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

        generateTriangles_region(grid2, tids2, pos, norm,d_compVoxelArray,
                                d_voxelVertsScan,
                                gridSize, gridSizeShift, gridSizeMask,
                                voxelSize,gridcenter, isoValue,*activeVoxels,
                                maxVerts,*totalVerts,vol_topo,primitive_fixed,primitive_dynamic,topo_field,lattice_field,iso1,iso2,d_voxelVerts,obj_union, obj_diff, obj_intersect,
                                primitive, topo, compute_lattice,fixed,dynamic,make_region,show_region, show_domain, triangle_data);

}



    void Isosurface::computeIsosurface_2(float4* pos , float4* norm, float isoValue,
    uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
    uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, grid_points *vol_topo, 
    grid_points *vol_one, float* vol_two, float* d_solid, float isovalue1,float *d_result, triangle_metadata *triangle_data) 
    {
        
        dim3 grid(ceil(numVoxels/float(1024)), 1, 1);
        dim3 threads(1024,1,1);
    
        classifyVoxel_lattice_2(grid,threads,
                            d_voxelVerts, d_voxelOccupied, vol_topo,vol_one,vol_two,d_solid,
                            gridSize, gridSizeShift, gridSizeMask,
                            numVoxels, voxelSize, isoValue, isovalue1);
 
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
        printf("active_voxel  %u  \n",*activeVoxels);
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
                                maxVerts,*totalVerts,vol_topo,vol_one,vol_two,d_solid,isovalue1,d_result,triangle_data);        
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


__device__
uint3 calcGridPos_field(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
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
void fillVolume(float *data, uint3 p, uint3 gridSize, float minDens)
{
    p.x = min(p.x, gridSize.x);
    p.y = min(p.y, gridSize.y);
    p.z = min(p.z, gridSize.z);
    uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
    data[i] = minDens;
}


__global__ void patch_topo_field_kernel(float *d_vec1, int Nx, int Ny, int Nz, float isoval, float *solid_field, float minDens, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 

    float k;

	if(tx < ((Nx-1) * (Ny - 1) * (Nz - 1)))
	{
        

		k = solid_field[tx];

        if(k < 1)
        {
            uint3 gridPos = calcGridPos_field(tx, gridSizeShift, gridSizeMask);

            fillVolume(d_vec1, gridPos, gridSize, minDens);
            fillVolume(d_vec1, gridPos + make_uint3(1, 0, 0), gridSize,minDens);
            fillVolume(d_vec1, gridPos + make_uint3(1, 1, 0), gridSize,minDens);
            fillVolume(d_vec1, gridPos + make_uint3(0, 1, 0), gridSize,minDens);
            fillVolume(d_vec1, gridPos + make_uint3(0, 0, 1), gridSize,minDens);
            fillVolume(d_vec1, gridPos + make_uint3(1, 0, 1), gridSize,minDens);
            fillVolume(d_vec1, gridPos + make_uint3(1, 1, 1), gridSize,minDens);
            fillVolume(d_vec1, gridPos + make_uint3(0, 1, 1), gridSize,minDens);

        }
	

    }


}




void Isosurface::patch_topo_field(float *d_vec1 , int Nx, int Ny, int Nz, float isoval ,float *solid_field,float minDens, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask)
{
    dim3 grids(ceil((Nx*Ny*Nz)/float(1024)),1,1);

	dim3 tids(1024,1,1);

    patch_topo_field_kernel<<<grids,tids>>>(d_vec1,Nx,Ny,Nz,isoval,solid_field,minDens,gridSize,gridSizeShift,gridSizeMask);

    cudaDeviceSynchronize();

}