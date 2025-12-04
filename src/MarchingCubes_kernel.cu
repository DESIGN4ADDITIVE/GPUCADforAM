/*

Reference - https://paulbourke.net/geometry/polygonise/

Reference - https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/marchingCubes

*/


#include <stdio.h>
#include <string.h>
#include <helper_cuda.h>    
#include <helper_math.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include "tables.h"
#include "MarchingCubes_kernel.h"

cudaTextureObject_t triTex_s;
cudaTextureObject_t numVertsTex_s;


void MarchingCubeCuda::allocateTextures_s(uint **d_triTable,  uint **d_numVertsTable)
{

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    checkCudaErrors(cudaMalloc((void **) d_triTable, 256*16*sizeof(uint)));
    checkCudaErrors(cudaMemcpy((void *)*d_triTable, (void *)triTable, 256*16*sizeof(uint), cudaMemcpyHostToDevice));

    cudaResourceDesc            texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType                = cudaResourceTypeLinear;
    texRes.res.linear.devPtr      = *d_triTable;
    texRes.res.linear.sizeInBytes = 256*16*sizeof(uint);
    texRes.res.linear.desc        = channelDesc;

    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&triTex_s, &texRes, &texDescr, NULL));

    checkCudaErrors(cudaMalloc((void **) d_numVertsTable, 256*sizeof(uint)));
    checkCudaErrors(cudaMemcpy((void *)*d_numVertsTable, (void *)numVertsTable, 256*sizeof(uint), cudaMemcpyHostToDevice));

    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType                = cudaResourceTypeLinear;
    texRes.res.linear.devPtr      = *d_numVertsTable;
    texRes.res.linear.sizeInBytes = 256*sizeof(uint);
    texRes.res.linear.desc        = channelDesc;

    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&numVertsTex_s, &texRes, &texDescr, NULL));
}


void MarchingCubeCuda::destroyAllTextureObjects()
{
    checkCudaErrors(cudaDestroyTextureObject(triTex_s));
    checkCudaErrors(cudaDestroyTextureObject(numVertsTex_s));
}


__device__
float sampleVolume(float *data, uint3 p, uint3 gridSize)
{
    p.x = min(p.x, gridSize.x);
    p.y = min(p.y, gridSize.y);
    p.z = min(p.z, gridSize.z);
    uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
    return (float) data[i];
}

__device__
int sampleVolume_2(grid_points *data, uint3 p, uint3 gridSize, int j)
{
    p.x = min(p.x, gridSize.x);
    p.y = min(p.y, gridSize.y);
    p.z = min(p.z, gridSize.z);
   
    uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;

    return (float) data[i].val;
}

__device__
grid_points sampleVolume_3(grid_points *data, uint3 p, uint3 gridSize)
{
    p.x = min(p.x, gridSize.x);
    p.y = min(p.y, gridSize.y);
    p.z = min(p.z, gridSize.z);
    uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
    return data[i];
}

__device__
uint3 calcGridPos(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
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
uint3 calcGridPos_one(uint i, uint3 gridSize)
{
    uint3 gridPos;
    
    uint z_quo = i / (gridSize.x * gridSize.y);
    uint z_rem = i % (gridSize.x * gridSize.y);
    uint y_quo = (z_rem)/gridSize.y;
    uint x_rem = (z_rem) % gridSize.y;

    gridPos.x = x_rem;
    gridPos.y = y_quo;
    gridPos.z = z_quo; 

    return gridPos;
}



__global__ void
classify_copy_Voxel(uint3 raster_grid, uint *voxel_verts,  grid_points *vol_one, float *volume_two,float *vol_lattice,bool fixed, bool dynamic,float iso1, float iso2,
              uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
              float3 voxelSize, float isoValue, cudaTextureObject_t numVertsTex, bool obj_union, bool obj_diff, bool obj_intersect)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;

    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    float isoVal = isoValue;

    if (i < ((gridSize.x * gridSize.y * gridSize.z) - 1))
    {
        uint3 gridPos = calcGridPos_one(i, gridSize);

        grid_points vox_points = vol_one[i];

        float v , v_lat, x , y , z = 0.0;

        v = sampleVolume(volume_two, gridPos, gridSize);
        v_lat = sampleVolume(vol_lattice, gridPos, gridSize);

        if(obj_union)
        {
            
            if (dynamic)
            {
               
               if( (uint(v_lat > iso1) & uint(v_lat < iso2)) | uint(vox_points.val < (isoVal)))
                {
                    vox_points.val = -1;
                }
                else
                {
                    vox_points.val = 1;
                }
            }
            else
            {
                if(uint(v < (isoVal)) | uint(vox_points.val < (isoVal)))
                {
                    vox_points.val = -1;
                }
                else
                {
                    vox_points.val = 1;
                }
            }
        }
        else if (obj_diff)
        {


            if (dynamic)
            {
                if( (uint(v_lat > iso1) & uint(v_lat < iso2)) & uint(vox_points.val >= (isoVal)))
               {

                     vox_points.val = -1;
                }
                else
                {
                    vox_points.val = 1;
                }
            }
            else
            {
                if(uint(v >= (isoVal)) & uint(vox_points.val < (isoVal)))
                {
                    vox_points.val = -1;
                }
                else
                {
                    vox_points.val = 1;
                }
            }
        }
        else if (obj_intersect)
        {
            
             if (dynamic)
            {
                if( (uint(v_lat > iso1) & uint(v_lat < iso2)) & uint(vox_points.val < (isoVal)))
               {

                     vox_points.val = -1;
                }
                else
                {
                    vox_points.val = 1;
                }
            }
            else
            {
                if(uint(v < (isoVal)) & uint(vox_points.val < (isoVal)))
                {
                    vox_points.val = -1;
                }
                else
                {
                    vox_points.val = 1;
                }
            }
        }
        
        

        if(gridPos.x < (gridSize.x - 1) )
        {
            if(dynamic)
            {
                x = sampleVolume(vol_lattice, gridPos + make_uint3(1, 0, 0), gridSize);

                if((uint(x < (iso1)) && uint(v_lat >= (iso1))) || (uint(x >= (iso1)) && uint(v_lat < (iso1))))
                {

                    float t = (iso1 - v_lat) / (x - v_lat);

                    if(vox_points.t_x > 0)
                    {
                        vox_points.t_x = (vox_points.t_x + t) *0.5;
                    }
                    else
                    {
                   
                        vox_points.t_x = t;
                    }
                }
                else if((uint(x < (iso2)) && uint(v_lat >= (iso2))) || (uint(x >= (iso2)) && uint(v_lat < (iso2))))
                {

                    float t = (iso2 - v_lat) / (x - v_lat);

                    if(vox_points.t_x > 0)
                    {
                        vox_points.t_x = (vox_points.t_x + t) *0.5;
                    }
                    else
                    {
                        vox_points.t_x = t;
                    }
                }
            }
            else
            {
                x = sampleVolume(volume_two, gridPos + make_uint3(1, 0, 0), gridSize);

                if((uint(x < (isoVal)) && uint(v >= (isoVal))) || (uint(x >= (isoVal)) && uint(v < (isoVal))))
                {

                    float t = (isoVal - v) / (x - v);

                    if(vox_points.t_x > 0)
                    {
                        vox_points.t_x = (vox_points.t_x + t) *0.5;
                    }
                    else
                    {
                        vox_points.t_x = t;
                    }
                }
            }

        }

        if(gridPos.y < (gridSize.y - 1))
        {
     
            if(dynamic)
            {
                y = sampleVolume(vol_lattice, gridPos + make_uint3(0, 1, 0), gridSize);

                if((uint(y < (iso1)) && uint(v_lat >= (iso1))) || (uint(y >= (iso1)) && uint(v_lat < (iso1))))
                {

                    float t = (iso1 - v_lat) / (y - v_lat);

                    if(vox_points.t_y > 0)
                    {
                        vox_points.t_y = (vox_points.t_y + t) *0.5;
                    }
                    else
                    {
                        vox_points.t_y = t;
                    }
                }
                else if((uint(y < (iso2)) && uint(v_lat >= (iso2))) || (uint(y >= (iso2)) && uint(v_lat < (iso2))))
                {

                    float t = (iso2 - v_lat) / (y - v_lat);

                    if(vox_points.t_y > 0)
                    {
                        vox_points.t_y = (vox_points.t_y + t) *0.5;
                    }
                    else
                    {
                        vox_points.t_y = t;
                    }
                }
            }
            
            else
            {
                y = sampleVolume(volume_two, gridPos + make_uint3(0, 1, 0), gridSize);

                if((uint(y < (isoVal)) && uint(v >= (isoVal))) || (uint(y >= (isoVal)) && uint(v < (isoVal))))
                {


                    float t = (isoVal - v) / (y - v);

                    if(vox_points.t_y > 0)
                    {
                        vox_points.t_y = (vox_points.t_y + t) *0.5;
                    }
                    else
                    {
                        vox_points.t_y = t;
                    }
                }
            }
            
        }


        if(gridPos.z < (gridSize.z - 1))
        {
            if(dynamic)
            {
                z = sampleVolume(vol_lattice, gridPos + make_uint3(0, 0, 1), gridSize);

                if((uint(z < (iso1)) && uint(v_lat >= (iso1))) || (uint(z >= (iso1)) && uint(v_lat < (iso1))))
                {

                    float t = (iso1 - v_lat) / (z - v_lat);

                    if(vox_points.t_z > 0)
                    {
                        vox_points.t_z = (vox_points.t_z + t) *0.5;
                    }
                    else
                    {
                        vox_points.t_z = t;
                    }
                }
                else if((uint(z < (iso2)) && uint(v_lat >= (iso2))) || (uint(z >= (iso2)) && uint(v_lat < (iso2))))
                {

                    float t = (iso2 - v_lat) / (z - v_lat);

                    if(vox_points.t_z > 0)
                    {
                        vox_points.t_z = (vox_points.t_z + t) *0.5;
                    }
                    else
                    {
                        vox_points.t_z = t;
                    }
                }
            }
            
            else
            {
                z = sampleVolume(volume_two, gridPos + make_uint3(0, 0, 1), gridSize);

                if((uint(z < (isoVal)) && uint(v >= (isoVal))) || (uint(z >= (isoVal)) && uint(v < (isoVal))))
                {

                    float t = (isoVal - v) / (z - v);

                    if(vox_points.t_z > 0)
                    {
                        vox_points.t_z = (vox_points.t_z + t) *0.5;
                    }
                    else
                    {
                        vox_points.t_z = t;
                    }
                }
            }
        }

        vol_one[i] = vox_points;

     
    }
  
 
}

void MarchingCubeCuda::classify_copy_Voxel_lattice(dim3 grid, dim3 threads, uint3 raster_grid,  uint *voxel_verts, grid_points *vol_one,float *volume_two,float *vol_lattice,bool fixed, bool dynamic,float iso1, float iso2,
                     uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
                     float3 voxelSize, float isoValue, bool obj_union, bool obj_diff, bool obj_intersect)
{

 
    classify_copy_Voxel<<<grid, threads>>>(raster_grid, voxel_verts, vol_one, volume_two,vol_lattice,fixed, dynamic,iso1, iso2,
                                     gridSize, gridSizeShift, gridSizeMask,
                                     numVoxels, voxelSize, isoValue, numVertsTex_s, obj_union, obj_diff, obj_intersect);
    cudaDeviceSynchronize();

    getLastCudaError("classifyVoxel failed");

   
}



__global__ void
classifyVoxel(float *vol,uint3 raster_grid, uint *voxelVerts, uint *voxelOccupied, grid_points  *primitive_fixed,float *primitive_dynamic, float *topo_field,float *lattice_field, 
              uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,float iso1, float iso2,
              float3 voxelSize, float isoValue, cudaTextureObject_t numVertsTex , bool obj_union, bool obj_diff, bool obj_intersect, bool primitive, bool topo, bool compute_lattice, bool fixed, bool dynamic)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;

    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
  
    if (i < numVoxels)
    {
        uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);
    
        float field_1[8];
        
        float field_2[8];

        float field_3[8];


        field_1[0] = sampleVolume_2(primitive_fixed, gridPos, gridSize,i);
        field_1[1] = sampleVolume_2(primitive_fixed, gridPos + make_uint3(1, 0, 0), gridSize,i);
        field_1[2] = sampleVolume_2(primitive_fixed, gridPos + make_uint3(1, 1, 0), gridSize,i);
        field_1[3] = sampleVolume_2(primitive_fixed, gridPos + make_uint3(0, 1, 0), gridSize,i);
        field_1[4] = sampleVolume_2(primitive_fixed, gridPos + make_uint3(0, 0, 1), gridSize,i);
        field_1[5] = sampleVolume_2(primitive_fixed, gridPos + make_uint3(1, 0, 1), gridSize,i);
        field_1[6] = sampleVolume_2(primitive_fixed, gridPos + make_uint3(1, 1, 1), gridSize,i);
        field_1[7] = sampleVolume_2(primitive_fixed, gridPos + make_uint3(0, 1, 1), gridSize,i);

        field_2[0] = sampleVolume(primitive_dynamic, gridPos, gridSize);
        field_2[1] = sampleVolume(primitive_dynamic, gridPos + make_uint3(1, 0, 0), gridSize);
        field_2[2] = sampleVolume(primitive_dynamic, gridPos + make_uint3(1, 1, 0), gridSize);
        field_2[3] = sampleVolume(primitive_dynamic, gridPos + make_uint3(0, 1, 0), gridSize);
        field_2[4] = sampleVolume(primitive_dynamic, gridPos + make_uint3(0, 0, 1), gridSize);
        field_2[5] = sampleVolume(primitive_dynamic, gridPos + make_uint3(1, 0, 1), gridSize);
        field_2[6] = sampleVolume(primitive_dynamic, gridPos + make_uint3(1, 1, 1), gridSize);
        field_2[7] = sampleVolume(primitive_dynamic, gridPos + make_uint3(0, 1, 1), gridSize);

        field_3[0] = sampleVolume(lattice_field, gridPos, gridSize);
        field_3[1] = sampleVolume(lattice_field, gridPos + make_uint3(1, 0, 0), gridSize);
        field_3[2] = sampleVolume(lattice_field, gridPos + make_uint3(1, 1, 0), gridSize);
        field_3[3] = sampleVolume(lattice_field, gridPos + make_uint3(0, 1, 0), gridSize);
        field_3[4] = sampleVolume(lattice_field, gridPos + make_uint3(0, 0, 1), gridSize);
        field_3[5] = sampleVolume(lattice_field, gridPos + make_uint3(1, 0, 1), gridSize);
        field_3[6] = sampleVolume(lattice_field, gridPos + make_uint3(1, 1, 1), gridSize);
        field_3[7] = sampleVolume(lattice_field, gridPos + make_uint3(0, 1, 1), gridSize);


        float isoVal = isoValue;
     
        uint cubeindex;

        if(obj_union)
        {
            
            if(fixed)
            {
                cubeindex =  (uint(field_2[0] < isoVal) | (uint(field_3[0] > iso1) & uint(field_3[0] < iso2)));
                cubeindex += (uint(field_2[1] < isoVal) | (uint(field_3[1] > iso1) & uint(field_3[1] < iso2))) *2;
                cubeindex += (uint(field_2[2] < isoVal) | (uint(field_3[2] > iso1) & uint(field_3[2] < iso2))) *4;
                cubeindex += (uint(field_2[3] < isoVal) | (uint(field_3[3] > iso1) & uint(field_3[3] < iso2))) *8;
                cubeindex += (uint(field_2[4] < isoVal) | (uint(field_3[4] > iso1) & uint(field_3[4] < iso2))) *16;
                cubeindex += (uint(field_2[5] < isoVal) | (uint(field_3[5] > iso1) & uint(field_3[5] < iso2))) *32;
                cubeindex += (uint(field_2[6] < isoVal) | (uint(field_3[6] > iso1) & uint(field_3[6] < iso2))) *64;
                cubeindex += (uint(field_2[7] < isoVal) | (uint(field_3[7] > iso1) & uint(field_3[7] < iso2))) *128;
            }

            else if(dynamic)
            {
                cubeindex =  (uint(field_1[0] < isoVal) | (uint(field_3[0] > iso1) & uint(field_3[0] < iso2)));
                cubeindex += (uint(field_1[1] < isoVal) | (uint(field_3[1] > iso1) & uint(field_3[1] < iso2))) *2;
                cubeindex += (uint(field_1[2] < isoVal) | (uint(field_3[2] > iso1) & uint(field_3[2] < iso2))) *4;
                cubeindex += (uint(field_1[3] < isoVal) | (uint(field_3[3] > iso1) & uint(field_3[3] < iso2))) *8;
                cubeindex += (uint(field_1[4] < isoVal) | (uint(field_3[4] > iso1) & uint(field_3[4] < iso2))) *16;
                cubeindex += (uint(field_1[5] < isoVal) | (uint(field_3[5] > iso1) & uint(field_3[5] < iso2))) *32;
                cubeindex += (uint(field_1[6] < isoVal) | (uint(field_3[6] > iso1) & uint(field_3[6] < iso2))) *64;
                cubeindex += (uint(field_1[7] < isoVal) | (uint(field_3[7] > iso1) & uint(field_3[7] < iso2))) *128;
            }
            
            else
            {
                cubeindex =  (uint(field_1[0] < isoVal) | uint(field_2[0] < isoVal));
                cubeindex += (uint(field_1[1] < isoVal) | uint(field_2[1] < isoVal)) *2;
                cubeindex += (uint(field_1[2] < isoVal) | uint(field_2[2] < isoVal)) *4;
                cubeindex += (uint(field_1[3] < isoVal) | uint(field_2[3] < isoVal)) *8;
                cubeindex += (uint(field_1[4] < isoVal) | uint(field_2[4] < isoVal)) *16;
                cubeindex += (uint(field_1[5] < isoVal) | uint(field_2[5] < isoVal)) *32;
                cubeindex += (uint(field_1[6] < isoVal) | uint(field_2[6] < isoVal)) *64;
                cubeindex += (uint(field_1[7] < isoVal) | uint(field_2[7] < isoVal)) *128;
            }

        }
        else if(obj_diff)
        {


            if(fixed)
            {
                cubeindex =  (uint(field_2[0] >= isoVal) & (uint(field_3[0] > iso1) & uint(field_3[0] < iso2)));
                cubeindex += (uint(field_2[1] >= isoVal) & (uint(field_3[1] > iso1) & uint(field_3[1] < iso2))) *2;
                cubeindex += (uint(field_2[2] >= isoVal) & (uint(field_3[2] > iso1) & uint(field_3[2] < iso2))) *4;
                cubeindex += (uint(field_2[3] >= isoVal) & (uint(field_3[3] > iso1) & uint(field_3[3] < iso2))) *8;
                cubeindex += (uint(field_2[4] >= isoVal) & (uint(field_3[4] > iso1) & uint(field_3[4] < iso2))) *16;
                cubeindex += (uint(field_2[5] >= isoVal) & (uint(field_3[5] > iso1) & uint(field_3[5] < iso2))) *32;
                cubeindex += (uint(field_2[6] >= isoVal) & (uint(field_3[6] > iso1) & uint(field_3[6] < iso2))) *64;
                cubeindex += (uint(field_2[7] >= isoVal) & (uint(field_3[7] > iso1) & uint(field_3[7] < iso2))) *128;
            }

            else if(dynamic)
            {
                cubeindex =  (uint(field_1[0] < isoVal) & (uint(field_3[0] < iso1) | uint(field_3[0] > iso2)));
                cubeindex += (uint(field_1[1] < isoVal) & (uint(field_3[1] < iso1) | uint(field_3[1] > iso2))) *2;
                cubeindex += (uint(field_1[2] < isoVal) & (uint(field_3[2] < iso1) | uint(field_3[2] > iso2))) *4;
                cubeindex += (uint(field_1[3] < isoVal) & (uint(field_3[3] < iso1) | uint(field_3[3] > iso2))) *8;
                cubeindex += (uint(field_1[4] < isoVal) & (uint(field_3[4] < iso1) | uint(field_3[4] > iso2))) *16;
                cubeindex += (uint(field_1[5] < isoVal) & (uint(field_3[5] < iso1) | uint(field_3[5] > iso2))) *32;
                cubeindex += (uint(field_1[6] < isoVal) & (uint(field_3[6] < iso1) | uint(field_3[6] > iso2))) *64;
                cubeindex += (uint(field_1[7] < isoVal) & (uint(field_3[7] < iso1) | uint(field_3[7] > iso2))) *128;
            }
            
            else
            {
                cubeindex =  (uint(field_2[0] >= isoVal) & uint(field_1[0] < isoVal));
                cubeindex += (uint(field_2[1] >= isoVal) & uint(field_1[1] < isoVal)) *2;
                cubeindex += (uint(field_2[2] >= isoVal) & uint(field_1[2] < isoVal)) *4;
                cubeindex += (uint(field_2[3] >= isoVal) & uint(field_1[3] < isoVal)) *8;
                cubeindex += (uint(field_2[4] >= isoVal) & uint(field_1[4] < isoVal)) *16;
                cubeindex += (uint(field_2[5] >= isoVal) & uint(field_1[5] < isoVal)) *32;
                cubeindex += (uint(field_2[6] >= isoVal) & uint(field_1[6] < isoVal)) *64;
                cubeindex += (uint(field_2[7] >= isoVal) & uint(field_1[7] < isoVal)) *128;
            }
        }
        else if(obj_intersect)
        {
            if(fixed)
            {
                cubeindex =  (uint(field_2[0] < isoVal) & (uint(field_3[0] > iso1) & uint(field_3[0] < iso2)));
                cubeindex += (uint(field_2[1] < isoVal) & (uint(field_3[1] > iso1) & uint(field_3[1] < iso2))) *2;
                cubeindex += (uint(field_2[2] < isoVal) & (uint(field_3[2] > iso1) & uint(field_3[2] < iso2))) *4;
                cubeindex += (uint(field_2[3] < isoVal) & (uint(field_3[3] > iso1) & uint(field_3[3] < iso2))) *8;
                cubeindex += (uint(field_2[4] < isoVal) & (uint(field_3[4] > iso1) & uint(field_3[4] < iso2))) *16;
                cubeindex += (uint(field_2[5] < isoVal) & (uint(field_3[5] > iso1) & uint(field_3[5] < iso2))) *32;
                cubeindex += (uint(field_2[6] < isoVal) & (uint(field_3[6] > iso1) & uint(field_3[6] < iso2))) *64;
                cubeindex += (uint(field_2[7] < isoVal) & (uint(field_3[7] > iso1) & uint(field_3[7] < iso2))) *128;
            }
            else if(dynamic)
            {
                cubeindex =  (uint(field_1[0] < isoVal) & (uint(field_3[0] > iso1) & uint(field_3[0] < iso2)));
                cubeindex += (uint(field_1[1] < isoVal) & (uint(field_3[1] > iso1) & uint(field_3[1] < iso2))) *2;
                cubeindex += (uint(field_1[2] < isoVal) & (uint(field_3[2] > iso1) & uint(field_3[2] < iso2))) *4;
                cubeindex += (uint(field_1[3] < isoVal) & (uint(field_3[3] > iso1) & uint(field_3[3] < iso2))) *8;
                cubeindex += (uint(field_1[4] < isoVal) & (uint(field_3[4] > iso1) & uint(field_3[4] < iso2))) *16;
                cubeindex += (uint(field_1[5] < isoVal) & (uint(field_3[5] > iso1) & uint(field_3[5] < iso2))) *32;
                cubeindex += (uint(field_1[6] < isoVal) & (uint(field_3[6] > iso1) & uint(field_3[6] < iso2))) *64;
                cubeindex += (uint(field_1[7] < isoVal) & (uint(field_3[7] > iso1) & uint(field_3[7] < iso2))) *128;
            }
            else
            {
                cubeindex =  (uint(field_1[0] < isoVal) & uint(field_2[0] < isoVal));
                cubeindex += (uint(field_1[1] < isoVal) & uint(field_2[1] < isoVal)) *2;
                cubeindex += (uint(field_1[2] < isoVal) & uint(field_2[2] < isoVal)) *4;
                cubeindex += (uint(field_1[3] < isoVal) & uint(field_2[3] < isoVal)) *8;
                cubeindex += (uint(field_1[4] < isoVal) & uint(field_2[4] < isoVal)) *16;
                cubeindex += (uint(field_1[5] < isoVal) & uint(field_2[5] < isoVal)) *32;
                cubeindex += (uint(field_1[6] < isoVal) & uint(field_2[6] < isoVal)) *64;
                cubeindex += (uint(field_1[7] < isoVal) & uint(field_2[7] < isoVal)) *128;
            }
        }

        uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);
     
        voxelVerts[i] = numVerts;

        voxelOccupied[i] =  numVerts > 0;

 
    }
  
 
}

void MarchingCubeCuda::classifyVoxel_lattice(dim3 grid, dim3 threads, float *vol,uint3 raster_grid,uint *voxelVerts, uint *voxelOccupied,grid_points  *primitive_fixed,float *primitive_dynamic, float *topo_field,float *lattice_field, 
                     uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels, float iso1, float iso2,
                     float3 voxelSize, float isoValue, bool obj_union, bool obj_diff, bool obj_intersect, bool primitive, bool topo, bool compute_lattice, bool fixed, bool dynamic)
{

 
    classifyVoxel<<<grid, threads>>>(vol,raster_grid,voxelVerts, voxelOccupied, primitive_fixed, primitive_dynamic, topo_field, lattice_field, 
                                     gridSize, gridSizeShift, gridSizeMask,numVoxels,iso1, iso2, voxelSize, isoValue, numVertsTex_s, obj_union, obj_diff, obj_intersect,primitive, topo, compute_lattice, fixed, dynamic);
    cudaDeviceSynchronize();

    getLastCudaError("classifyVoxel failed");

   
}





__global__ void
classifyVoxel_2(uint *voxelVerts, uint *voxelOccupied, float *volume_one,
              uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
              float3 voxelSize, float isoValue, cudaTextureObject_t numVertsTex)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    if (i < numVoxels)
    {
        uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);
    
        float field[8];
        field[0] = sampleVolume(volume_one, gridPos, gridSize);
        field[1] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 0), gridSize);
        field[2] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 0), gridSize);
        field[3] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 0), gridSize);
        field[4] = sampleVolume(volume_one, gridPos + make_uint3(0, 0, 1), gridSize);
        field[5] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 1), gridSize);
        field[6] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 1), gridSize);
        field[7] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 1), gridSize);


        float isoVal = isoValue;
     
        uint cubeindex;
        cubeindex =  uint(field[0] < (isoVal));
        cubeindex += uint(field[1] < (isoVal))*2;
        cubeindex += uint(field[2] < (isoVal))*4;
        cubeindex += uint(field[3] < (isoVal))*8;
        cubeindex += uint(field[4] < (isoVal))*16;
        cubeindex += uint(field[5] < (isoVal))*32;
        cubeindex += uint(field[6] < (isoVal))*64;
        cubeindex += uint(field[7] < (isoVal))*128;
        uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);


        voxelVerts[i] = numVerts;

        voxelOccupied[i] = (numVerts > 0);

    }
  
 
}


void MarchingCubeCuda::classifyVoxel_lattice_2(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, float *volume_one,
                     uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
                     float3 voxelSize, float isoValue)
{

    classifyVoxel_2<<<grid, threads>>>(voxelVerts, voxelOccupied, volume_one,
                                     gridSize, gridSizeShift, gridSizeMask,
                                     numVoxels, voxelSize, isoValue, numVertsTex_s);
    cudaDeviceSynchronize();

    getLastCudaError("classifyVoxel failed");

   
}



__global__ void
classifyVoxel_3(uint *voxelVerts, uint *voxelOccupied, float *volume_one,
              uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
              float3 voxelSize, float isoValue, cudaTextureObject_t numVertsTex)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    if (i < numVoxels)
    {
        uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);
    
        float field[8];
        field[0] = sampleVolume(volume_one, gridPos, gridSize);
        field[1] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 0), gridSize);
        field[2] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 0), gridSize);
        field[3] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 0), gridSize);
        field[4] = sampleVolume(volume_one, gridPos + make_uint3(0, 0, 1), gridSize);
        field[5] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 1), gridSize);
        field[6] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 1), gridSize);
        field[7] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 1), gridSize);


        float isoVal = isoValue;
     
        uint cubeindex;
        cubeindex =  uint(field[0] < (isoVal)) && uint(field[0] > (0.0f));
        cubeindex += (uint(field[1] < (isoVal)) && uint(field[1] > (0.0f))) *2;
        cubeindex += (uint(field[2] < (isoVal)) && uint(field[2] > (0.0f))) *4;
        cubeindex += (uint(field[3] < (isoVal)) && uint(field[3] > (0.0f))) *8;
        cubeindex += (uint(field[4] < (isoVal)) && uint(field[4] > (0.0f))) *16;
        cubeindex += (uint(field[5] < (isoVal)) && uint(field[5] > (0.0f))) *32;
        cubeindex += (uint(field[6] < (isoVal)) && uint(field[6] > (0.0f))) *64;
        cubeindex += (uint(field[7] < (isoVal)) && uint(field[7] > (0.0f))) *128;
        uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

        voxelVerts[i] = numVerts;

        voxelOccupied[i] = (numVerts > 0);

    }
  
 
}


void MarchingCubeCuda::classifyVoxel_lattice_3(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, float *volume_one,
                     uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
                     float3 voxelSize, float isoValue)
{


    classifyVoxel_3<<<grid, threads>>>(voxelVerts, voxelOccupied, volume_one,
                                     gridSize, gridSizeShift, gridSizeMask,
                                     numVoxels, voxelSize, isoValue, numVertsTex_s);
    cudaDeviceSynchronize();

    getLastCudaError("classifyVoxel failed");

   
}


__global__ void
compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    if(i < numVoxels)
    {
        if (voxelOccupied[i])
        {
            compactedVoxelArray[ voxelOccupiedScan[i] ] = i;
        }
    
    }
}

void MarchingCubeCuda::compactVoxels_lattice(dim3 grid, dim3 threads, uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels)
{
    compactVoxels<<<grid, threads>>>(compactedVoxelArray, voxelOccupied,
                                     voxelOccupiedScan, numVoxels);
    getLastCudaError("compactVoxels failed");
}



__device__
float3  vertexInterp4(float isolevelone,float3 p0, float3 p1, float f0, float f1)
{
    
    float t = 0.0;
    

    if (fabs(isolevelone-f0) < 0.0005)
    {
        return(p0);
    }
    if (fabs(isolevelone-f1) < 0.0005)
    {
        return(p1);
    }
    if (fabs(f1-f0) < 0.0005)
    {
        return(p0);
    }

    t = (isolevelone - f0) / (f1 - f0);

 


    return lerp(p0, p1, t);
}


__device__
float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{
    float3 edge0 = *v1 - *v0;
    float3 edge1 = *v2 - *v0;
    return cross(edge0, edge1);
}


__device__
float3  vertexInterp_primitive(float isolevel,float3 p0, float3 p1, float f0, float f1, float edge_id)
{
    
    
    float t1 = edge_id;
    float t2 = 0.0 ;

    float t = 0;



    if(((f1 >= isolevel) && (f0 <= isolevel)) || ((f0 >= isolevel) && (f1 <= isolevel)))
    {
        t2 = (isolevel - f0) / (f1 - f0);
    }

    if( (t1 > 0) && (t2 > 0) )
    {
        t = (t1 + t2) * 0.5;
    }
    else if ((t1 > 0) && (t2 == 0))
    {
        t = t1;
    } 
    else if ((t2 > 0) && (t1 == 0))
    {
        t = t2;
    }

    
    return lerp(p0, p1, t);
}


__device__
float3  vertexInterp_primitive_one(float isolevelone,float isoleveltwo,float3 p0, float3 p1, float f0, float f1, float edge_id)
{
    
    
    float t1 = edge_id;
    float t2 = 0.0 ;
    float t3 = 0.0 ;
    float t = 0;



    if(((f1 >= isolevelone) && (f0 <= isolevelone)) || ((f0 >= isolevelone) && (f1 <= isolevelone)))
    {
        t2 = (isolevelone - f0) / (f1 - f0);
    }

    if( (t1 > 0) && (t2 > 0) )
    {
        t = (t1 + t2) * 0.5;
    }
    else if ((t1 > 0) && (t2 == 0))
    {
        t = t1;
    } 
    else if ((t2 > 0) && (t1 == 0))
    {
        t = t2;
    }

    if(((f1 >= isoleveltwo) && (f0 <= isoleveltwo)) || ((f0 >= isoleveltwo) && (f1 <= isoleveltwo)))
    {
        t3 = (isoleveltwo - f0) / (f1 - f0);
    }

    if( (t1 > 0) && (t3 > 0) )
    {
        t = (t1 + t3) * 0.5;
    }
    else if ((t1 > 0) && (t3 == 0))
    {
        t = t1;
    } 
    else if ((t3 > 0) && (t1 == 0))
    {
        t = t3;
    }
    
    return lerp(p0, p1, t);
}


__device__
float3  vertexInterp_new(float isoval, float isolevelone,float isoleveltwo, float3 p0, float3 p1, float f0, float f1,float f2, float f3)
{
    
    
    float t1 = 0.0f;
    float t2 = 0.0f;
    float t3 = 0.0f;

    float t = 0.0f;


 
    if(((f0 < isoval) && (f1 >= isoval)) || ((f1 < isoval) && (f0 >= isoval)))
    {
        t1 = (isoval - f0)/(f1 - f0);
    }

    if(((f2 < isolevelone) && (f3 >= isolevelone)) || ((f3 < isolevelone) && (f2 >= isolevelone)))
    {
        t2 = (isolevelone - f2)/(f3 - f2);
    }

    if(((f2 < isoleveltwo) && (f3 >= isoleveltwo)) || ((f3 < isoleveltwo) && (f2 >= isoleveltwo)))
    {
        t3 = (isoleveltwo - f2)/(f3 - f2);
    }

    

    if((t1 > 0.0f) && (t2 > 0.0f) && (t3 == 0.0))
    {   
   
        t = (t1 + t2) * 0.5;
    
    }

    else if((t1 > 0.0f) && (t3 > 0.0f) && (t2 == 0.0f))
    {   
   
        t = (t1 + t3) * 0.5;
    
    }
    else if((t1 > 0.0) && (t2 == 0.0) && (t3 == 0.0))
    {
        t = t1;
    }
    else if((t2 > 0.0) && (t1 == 0.0) && (t3 == 0.0))
    {
        t = t2;
    }
    else if((t3 > 0.0) && (t1 == 0.0) && (t2 == 0.0))
    {
        t = t3;
    }

    return lerp(p0, p1, t);
}

__global__ void
generateTriangles_lattice_kernel(float4 *pos, float4 *norm, uint *compactVoxelArray,uint *numVertsScanned,
                   uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                   float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts,
                   cudaTextureObject_t triTex, cudaTextureObject_t numVertsTex,uint totalverts_1, grid_points  *primitive_fixed,float *primitive_dynamic, float *topo_field,float *lattice_field, float iso1, float iso2,
                   uint *voxerl_verts, bool obj_union, bool obj_diff, bool obj_intersect, bool primitive, bool topo, bool compute_lattice, bool fixed, bool dynamic)
{
    
    
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    
  

    if (i < activeVoxels)
    {
        uint voxel = compactVoxelArray[i];

        uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

        float3 p;

        p.x = (gridPos.x - gridcenter.x) *voxelSize.x ;
        p.y = (gridPos.y - gridcenter.y) *voxelSize.y ;
        p.z = (gridPos.z - gridcenter.z) *voxelSize.z ;
        
        float3 v[8];
        v[0] = p;
        v[1] = p + make_float3(voxelSize.x, 0, 0);
        v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
        v[3] = p + make_float3(0, voxelSize.y, 0);
        v[4] = p + make_float3(0, 0, voxelSize.z);
        v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
        v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
        v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

     
        float field_1[8];
     
        float field_3[8];

        grid_points field_2[8];

            field_3[0] = sampleVolume(lattice_field, gridPos, gridSize);
            field_3[1] = sampleVolume(lattice_field, gridPos + make_uint3(1, 0, 0), gridSize);
            field_3[2] = sampleVolume(lattice_field, gridPos + make_uint3(1, 1, 0), gridSize);
            field_3[3] = sampleVolume(lattice_field, gridPos + make_uint3(0, 1, 0), gridSize);
            field_3[4] = sampleVolume(lattice_field, gridPos + make_uint3(0, 0, 1), gridSize);
            field_3[5] = sampleVolume(lattice_field, gridPos + make_uint3(1, 0, 1), gridSize);
            field_3[6] = sampleVolume(lattice_field, gridPos + make_uint3(1, 1, 1), gridSize);
            field_3[7] = sampleVolume(lattice_field, gridPos + make_uint3(0, 1, 1), gridSize);
     
     
            field_1[0] = sampleVolume(primitive_dynamic, gridPos, gridSize);
            field_1[1] = sampleVolume(primitive_dynamic, gridPos + make_uint3(1, 0, 0), gridSize);
            field_1[2] = sampleVolume(primitive_dynamic, gridPos + make_uint3(1, 1, 0), gridSize);
            field_1[3] = sampleVolume(primitive_dynamic, gridPos + make_uint3(0, 1, 0), gridSize);
            field_1[4] = sampleVolume(primitive_dynamic, gridPos + make_uint3(0, 0, 1), gridSize);
            field_1[5] = sampleVolume(primitive_dynamic, gridPos + make_uint3(1, 0, 1), gridSize);
            field_1[6] = sampleVolume(primitive_dynamic, gridPos + make_uint3(1, 1, 1), gridSize);
            field_1[7] = sampleVolume(primitive_dynamic, gridPos + make_uint3(0, 1, 1), gridSize);
        

            field_2[0] = sampleVolume_3(primitive_fixed, gridPos, gridSize);
            field_2[1] = sampleVolume_3(primitive_fixed, gridPos + make_uint3(1, 0, 0), gridSize);
            field_2[2] = sampleVolume_3(primitive_fixed, gridPos + make_uint3(1, 1, 0), gridSize);
            field_2[3] = sampleVolume_3(primitive_fixed, gridPos + make_uint3(0, 1, 0), gridSize);
            field_2[4] = sampleVolume_3(primitive_fixed, gridPos + make_uint3(0, 0, 1), gridSize);
            field_2[5] = sampleVolume_3(primitive_fixed, gridPos + make_uint3(1, 0, 1), gridSize);
            field_2[6] = sampleVolume_3(primitive_fixed, gridPos + make_uint3(1, 1, 1), gridSize);
            field_2[7] = sampleVolume_3(primitive_fixed, gridPos + make_uint3(0, 1, 1), gridSize);
        

        float isoVal = isoValue; 
     
        uint cubeindex;

        if(obj_union)
        {
            
            if(fixed)
            {
      
                cubeindex =  (uint(field_1[0] < isoVal) | (uint(field_3[0] > iso1) & uint(field_3[0] < iso2)));
                cubeindex += (uint(field_1[1] < isoVal) | (uint(field_3[1] > iso1) & uint(field_3[1] < iso2))) *2;
                cubeindex += (uint(field_1[2] < isoVal) | (uint(field_3[2] > iso1) & uint(field_3[2] < iso2))) *4;
                cubeindex += (uint(field_1[3] < isoVal) | (uint(field_3[3] > iso1) & uint(field_3[3] < iso2))) *8;
                cubeindex += (uint(field_1[4] < isoVal) | (uint(field_3[4] > iso1) & uint(field_3[4] < iso2))) *16;
                cubeindex += (uint(field_1[5] < isoVal) | (uint(field_3[5] > iso1) & uint(field_3[5] < iso2))) *32;
                cubeindex += (uint(field_1[6] < isoVal) | (uint(field_3[6] > iso1) & uint(field_3[6] < iso2))) *64;
                cubeindex += (uint(field_1[7] < isoVal) | (uint(field_3[7] > iso1) & uint(field_3[7] < iso2))) *128;

            }
            else if(dynamic)
            {
                cubeindex =  (uint(field_2[0].val < isoVal) | (uint(field_3[0] > iso1) & uint(field_3[0] < iso2)));
                cubeindex += (uint(field_2[1].val < isoVal) | (uint(field_3[1] > iso1) & uint(field_3[1] < iso2))) *2;
                cubeindex += (uint(field_2[2].val < isoVal) | (uint(field_3[2] > iso1) & uint(field_3[2] < iso2))) *4;
                cubeindex += (uint(field_2[3].val < isoVal) | (uint(field_3[3] > iso1) & uint(field_3[3] < iso2))) *8;
                cubeindex += (uint(field_2[4].val < isoVal) | (uint(field_3[4] > iso1) & uint(field_3[4] < iso2))) *16;
                cubeindex += (uint(field_2[5].val < isoVal) | (uint(field_3[5] > iso1) & uint(field_3[5] < iso2))) *32;
                cubeindex += (uint(field_2[6].val < isoVal) | (uint(field_3[6] > iso1) & uint(field_3[6] < iso2))) *64;
                cubeindex += (uint(field_2[7].val < isoVal) | (uint(field_3[7] > iso1) & uint(field_3[7] < iso2))) *128;
            }
            else
            {
                cubeindex =  (uint(field_1[0] < isoVal) | uint(field_2[0].val < isoVal));
                cubeindex += (uint(field_1[1] < isoVal) | uint(field_2[1].val < isoVal)) *2;
                cubeindex += (uint(field_1[2] < isoVal) | uint(field_2[2].val < isoVal)) *4;
                cubeindex += (uint(field_1[3] < isoVal) | uint(field_2[3].val < isoVal)) *8;
                cubeindex += (uint(field_1[4] < isoVal) | uint(field_2[4].val < isoVal)) *16;
                cubeindex += (uint(field_1[5] < isoVal) | uint(field_2[5].val < isoVal)) *32;
                cubeindex += (uint(field_1[6] < isoVal) | uint(field_2[6].val < isoVal)) *64;
                cubeindex += (uint(field_1[7] < isoVal) | uint(field_2[7].val < isoVal)) *128;
            }
        }
        else if(obj_diff)
        {
            if(fixed)
            {
                cubeindex =  (uint(field_1[0] >= isoVal) & (uint(field_3[0] > iso1) & uint(field_3[0] < iso2)));
                cubeindex += (uint(field_1[1] >= isoVal) & (uint(field_3[1] > iso1) & uint(field_3[1] < iso2))) *2;
                cubeindex += (uint(field_1[2] >= isoVal) & (uint(field_3[2] > iso1) & uint(field_3[2] < iso2))) *4;
                cubeindex += (uint(field_1[3] >= isoVal) & (uint(field_3[3] > iso1) & uint(field_3[3] < iso2))) *8;
                cubeindex += (uint(field_1[4] >= isoVal) & (uint(field_3[4] > iso1) & uint(field_3[4] < iso2))) *16;
                cubeindex += (uint(field_1[5] >= isoVal) & (uint(field_3[5] > iso1) & uint(field_3[5] < iso2))) *32;
                cubeindex += (uint(field_1[6] >= isoVal) & (uint(field_3[6] > iso1) & uint(field_3[6] < iso2))) *64;
                cubeindex += (uint(field_1[7] >= isoVal) & (uint(field_3[7] > iso1) & uint(field_3[7] < iso2))) *128;
            }
            else if(dynamic)
            {
                cubeindex =  (uint(field_2[0].val < isoVal) & (uint(field_3[0] < iso1) | uint(field_3[0] > iso2)));
                cubeindex += (uint(field_2[1].val < isoVal) & (uint(field_3[1] < iso1) | uint(field_3[1] > iso2))) *2;
                cubeindex += (uint(field_2[2].val < isoVal) & (uint(field_3[2] < iso1) | uint(field_3[2] > iso2))) *4;
                cubeindex += (uint(field_2[3].val < isoVal) & (uint(field_3[3] < iso1) | uint(field_3[3] > iso2))) *8;
                cubeindex += (uint(field_2[4].val < isoVal) & (uint(field_3[4] < iso1) | uint(field_3[4] > iso2))) *16;
                cubeindex += (uint(field_2[5].val < isoVal) & (uint(field_3[5] < iso1) | uint(field_3[5] > iso2))) *32;
                cubeindex += (uint(field_2[6].val < isoVal) & (uint(field_3[6] < iso1) | uint(field_3[6] > iso2))) *64;
                cubeindex += (uint(field_2[7].val < isoVal) & (uint(field_3[7] < iso1) | uint(field_3[7] > iso2))) *128;
            }
            
            else
            {
                cubeindex =  (uint(field_1[0] >= isoVal) & uint(field_2[0].val < isoVal));
                cubeindex += (uint(field_1[1] >= isoVal) & uint(field_2[1].val < isoVal)) *2;
                cubeindex += (uint(field_1[2] >= isoVal) & uint(field_2[2].val < isoVal)) *4;
                cubeindex += (uint(field_1[3] >= isoVal) & uint(field_2[3].val < isoVal)) *8;
                cubeindex += (uint(field_1[4] >= isoVal) & uint(field_2[4].val < isoVal)) *16;
                cubeindex += (uint(field_1[5] >= isoVal) & uint(field_2[5].val < isoVal)) *32;
                cubeindex += (uint(field_1[6] >= isoVal) & uint(field_2[6].val < isoVal)) *64;
                cubeindex += (uint(field_1[7] >= isoVal) & uint(field_2[7].val < isoVal)) *128;
            }
        
        }
        else if(obj_intersect)
        {
            if(fixed)
            {
                cubeindex =  (uint(field_1[0] < isoVal) & (uint(field_3[0] > iso1) & uint(field_3[0] < iso2)));
                cubeindex += (uint(field_1[1] < isoVal) & (uint(field_3[1] > iso1) & uint(field_3[1] < iso2))) *2;
                cubeindex += (uint(field_1[2] < isoVal) & (uint(field_3[2] > iso1) & uint(field_3[2] < iso2))) *4;
                cubeindex += (uint(field_1[3] < isoVal) & (uint(field_3[3] > iso1) & uint(field_3[3] < iso2))) *8;
                cubeindex += (uint(field_1[4] < isoVal) & (uint(field_3[4] > iso1) & uint(field_3[4] < iso2))) *16;
                cubeindex += (uint(field_1[5] < isoVal) & (uint(field_3[5] > iso1) & uint(field_3[5] < iso2))) *32;
                cubeindex += (uint(field_1[6] < isoVal) & (uint(field_3[6] > iso1) & uint(field_3[6] < iso2))) *64;
                cubeindex += (uint(field_1[7] < isoVal) & (uint(field_3[7] > iso1) & uint(field_3[7] < iso2))) *128;
            }
            else if(dynamic)
            {
                cubeindex =  (uint(field_2[0].val < isoVal) & (uint(field_3[0] > iso1) & uint(field_3[0] < iso2)));
                cubeindex += (uint(field_2[1].val < isoVal) & (uint(field_3[1] > iso1) & uint(field_3[1] < iso2))) *2;
                cubeindex += (uint(field_2[2].val < isoVal) & (uint(field_3[2] > iso1) & uint(field_3[2] < iso2))) *4;
                cubeindex += (uint(field_2[3].val < isoVal) & (uint(field_3[3] > iso1) & uint(field_3[3] < iso2))) *8;
                cubeindex += (uint(field_2[4].val < isoVal) & (uint(field_3[4] > iso1) & uint(field_3[4] < iso2))) *16;
                cubeindex += (uint(field_2[5].val < isoVal) & (uint(field_3[5] > iso1) & uint(field_3[5] < iso2))) *32;
                cubeindex += (uint(field_2[6].val < isoVal) & (uint(field_3[6] > iso1) & uint(field_3[6] < iso2))) *64;
                cubeindex += (uint(field_2[7].val < isoVal) & (uint(field_3[7] > iso1) & uint(field_3[7] < iso2))) *128;
            }
            else
            {
                cubeindex =  (uint(field_1[0] < isoVal) & uint(field_2[0].val < isoVal));
                cubeindex += (uint(field_1[1] < isoVal) & uint(field_2[1].val < isoVal)) *2;
                cubeindex += (uint(field_1[2] < isoVal) & uint(field_2[2].val < isoVal)) *4;
                cubeindex += (uint(field_1[3] < isoVal) & uint(field_2[3].val < isoVal)) *8;
                cubeindex += (uint(field_1[4] < isoVal) & uint(field_2[4].val < isoVal)) *16;
                cubeindex += (uint(field_1[5] < isoVal) & uint(field_2[5].val < isoVal)) *32;
                cubeindex += (uint(field_1[6] < isoVal) & uint(field_2[6].val < isoVal)) *64;
                cubeindex += (uint(field_1[7] < isoVal) & uint(field_2[7].val < isoVal)) *128;
            }
        }

        uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);
    
        if(numVerts > 0)
        {

            __shared__ float3 vertlist[12*NTHREADS];

            if(fixed)
            {
                vertlist[threadIdx.x] = vertexInterp_new(isoVal, iso1, iso2, v[0], v[1],field_1[0], field_1[1], field_3[0], field_3[1]);
                vertlist[NTHREADS+threadIdx.x] = vertexInterp_new(isoVal, iso1, iso2, v[1], v[2],field_1[1], field_1[2], field_3[1], field_3[2]);
                vertlist[(NTHREADS*2)+threadIdx.x] = vertexInterp_new(isoVal, iso1, iso2, v[2], v[3],field_1[2], field_1[3], field_3[2], field_3[3]);
                vertlist[(NTHREADS*3)+threadIdx.x] = vertexInterp_new(isoVal, iso1, iso2, v[3], v[0],field_1[3], field_1[0], field_3[3], field_3[0]);
                vertlist[(NTHREADS*4)+threadIdx.x] = vertexInterp_new(isoVal, iso1, iso2, v[4], v[5],field_1[4], field_1[5], field_3[4], field_3[5]);
                vertlist[(NTHREADS*5)+threadIdx.x] = vertexInterp_new(isoVal, iso1, iso2, v[5], v[6],field_1[5], field_1[6], field_3[5], field_3[6]);
                vertlist[(NTHREADS*6)+threadIdx.x] = vertexInterp_new(isoVal, iso1, iso2, v[6], v[7],field_1[6], field_1[7], field_3[6], field_3[7]);
                vertlist[(NTHREADS*7)+threadIdx.x] = vertexInterp_new(isoVal, iso1, iso2, v[7], v[4],field_1[7], field_1[4], field_3[7], field_3[4]);
                vertlist[(NTHREADS*8)+threadIdx.x] = vertexInterp_new(isoVal, iso1, iso2, v[0], v[4],field_1[0], field_1[4], field_3[0], field_3[4]);
                vertlist[(NTHREADS*9)+threadIdx.x] = vertexInterp_new(isoVal, iso1, iso2, v[1], v[5],field_1[1], field_1[5], field_3[1], field_3[5]);
                vertlist[(NTHREADS*10)+threadIdx.x] = vertexInterp_new(isoVal, iso1, iso2, v[2], v[6],field_1[2], field_1[6], field_3[2], field_3[6]);
                vertlist[(NTHREADS*11)+threadIdx.x] = vertexInterp_new(isoVal, iso1, iso2, v[3], v[7],field_1[3], field_1[7], field_3[3], field_3[7]);
             
                
            }
            else if(dynamic)
            {
                vertlist[threadIdx.x] = vertexInterp_primitive_one(iso1,iso2, v[0], v[1], field_3[0], field_3[1],field_2[0].t_x);
                vertlist[NTHREADS+threadIdx.x] = vertexInterp_primitive_one(iso1,iso2, v[1], v[2], field_3[1], field_3[2],field_2[1].t_y);
                vertlist[(NTHREADS*2)+threadIdx.x] = vertexInterp_primitive_one(iso1,iso2, v[3], v[2], field_3[3], field_3[2],field_2[3].t_x);
                vertlist[(NTHREADS*3)+threadIdx.x] = vertexInterp_primitive_one(iso1,iso2, v[0], v[3], field_3[0], field_3[3],field_2[0].t_y);
                vertlist[(NTHREADS*4)+threadIdx.x] = vertexInterp_primitive_one(iso1,iso2, v[4], v[5], field_3[4], field_3[5],field_2[4].t_x);
                vertlist[(NTHREADS*5)+threadIdx.x] = vertexInterp_primitive_one(iso1,iso2, v[5], v[6], field_3[5], field_3[6],field_2[5].t_y);
                vertlist[(NTHREADS*6)+threadIdx.x] = vertexInterp_primitive_one(iso1,iso2, v[7], v[6], field_3[7], field_3[6],field_2[7].t_x);
                vertlist[(NTHREADS*7)+threadIdx.x] = vertexInterp_primitive_one(iso1,iso2, v[4], v[7], field_3[4], field_3[7],field_2[4].t_y);
                vertlist[(NTHREADS*8)+threadIdx.x] = vertexInterp_primitive_one(iso1,iso2, v[0], v[4], field_3[0], field_3[4],field_2[0].t_z);
                vertlist[(NTHREADS*9)+threadIdx.x] = vertexInterp_primitive_one(iso1,iso2, v[1], v[5], field_3[1], field_3[5],field_2[1].t_z);
                vertlist[(NTHREADS*10)+threadIdx.x] = vertexInterp_primitive_one(iso1,iso2, v[2], v[6], field_3[2], field_3[6],field_2[2].t_z);
                vertlist[(NTHREADS*11)+threadIdx.x] = vertexInterp_primitive_one(iso1,iso2, v[3], v[7], field_3[3], field_3[7],field_2[3].t_z);
            }

            else
            {
                vertlist[threadIdx.x] = vertexInterp_primitive(isoVal, v[0], v[1], field_1[0], field_1[1],field_2[0].t_x);
                vertlist[NTHREADS+threadIdx.x] = vertexInterp_primitive(isoVal, v[1], v[2], field_1[1], field_1[2],field_2[1].t_y);
                vertlist[(NTHREADS*2)+threadIdx.x] = vertexInterp_primitive(isoVal, v[3], v[2], field_1[3], field_1[2],field_2[3].t_x);
                vertlist[(NTHREADS*3)+threadIdx.x] = vertexInterp_primitive(isoVal, v[0], v[3], field_1[0], field_1[3],field_2[0].t_y);
                vertlist[(NTHREADS*4)+threadIdx.x] = vertexInterp_primitive(isoVal, v[4], v[5], field_1[4], field_1[5],field_2[4].t_x);
                vertlist[(NTHREADS*5)+threadIdx.x] = vertexInterp_primitive(isoVal, v[5], v[6], field_1[5], field_1[6],field_2[5].t_y);
                vertlist[(NTHREADS*6)+threadIdx.x] = vertexInterp_primitive(isoVal, v[7], v[6], field_1[7], field_1[6],field_2[7].t_x);
                vertlist[(NTHREADS*7)+threadIdx.x] = vertexInterp_primitive(isoVal, v[4], v[7], field_1[4], field_1[7],field_2[4].t_y);
                vertlist[(NTHREADS*8)+threadIdx.x] = vertexInterp_primitive(isoVal, v[0], v[4], field_1[0], field_1[4],field_2[0].t_z);
                vertlist[(NTHREADS*9)+threadIdx.x] = vertexInterp_primitive(isoVal, v[1], v[5], field_1[1], field_1[5],field_2[1].t_z);
                vertlist[(NTHREADS*10)+threadIdx.x] = vertexInterp_primitive(isoVal, v[2], v[6], field_1[2], field_1[6],field_2[2].t_z);
                vertlist[(NTHREADS*11)+threadIdx.x] = vertexInterp_primitive(isoVal, v[3], v[7], field_1[3], field_1[7],field_2[3].t_z);
            }



            for (int j = 0; j<numVerts; j += 3)
            {
                uint index;
                
                index = numVertsScanned[voxel] + j;
                
                float3 *v[3];

                uint edge;

                edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j);

                
                v[0] = &vertlist[(edge*NTHREADS)+threadIdx.x];

                edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j + 1);

                
                v[1] = &vertlist[(edge*NTHREADS)+threadIdx.x];
                

                edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j + 2);
                
                v[2] = &vertlist[(edge*NTHREADS)+threadIdx.x];

                float3 n = (calcNormal(v[0], v[1], v[2]));

            
                if (index < (maxVerts - 3))
                {
            
                    pos[index] = make_float4(*v[0], 1.0f);
                    norm[index] = make_float4(n, 1.0f);

                    pos[index+1] = make_float4(*v[1], 1.0f);
                    norm[index+1] = make_float4(n, 1.0f);

                    pos[index+2] = make_float4(*v[2], 1.0f);
                    norm[index+2] = make_float4(n, 1.0f);    
                
                }
            }
    
        }
    }

    
}

void MarchingCubeCuda::generateTriangles_lattice(dim3 grid, dim3 threads,float4 *pos, float4 *norm,uint *compactVoxelArray,
                           uint *numVertsScanned,uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                          float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts, uint totalverts_1,grid_points  *primitive_fixed,float *primitive_dynamic, float *topo_field,float *lattice_field, 
                          float iso1,float iso2,uint *voxel_verts, bool obj_union, bool obj_diff, bool obj_intersect, bool primitive, bool topo, bool compute_lattice, bool fixed, bool dynamic)
{
    
    generateTriangles_lattice_kernel<<<grid, threads>>>(pos, norm,compactVoxelArray,
                                           numVertsScanned,
                                           gridSize, gridSizeShift, gridSizeMask,
                                           voxelSize,gridcenter, isoValue, activeVoxels,
                                           maxVerts, triTex_s, numVertsTex_s, totalverts_1, primitive_fixed, primitive_dynamic, topo_field, lattice_field, iso1,iso2,voxel_verts, obj_union, obj_diff, obj_intersect,
                                            primitive, topo, compute_lattice, fixed, dynamic);
    cudaDeviceSynchronize();
    getLastCudaError("generateTriangles failed");
    cudaError_t err = cudaGetLastError();
    
}







__global__ void
generateTriangles_lattice_kernel_2(float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, 
                   uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                   float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts,
                   cudaTextureObject_t triTex, cudaTextureObject_t numVertsTex,uint totalverts, float *volume_one, float isoValue1)
{
    
    
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
    
    if (i < activeVoxels)
    {
             
        uint voxel = compactedVoxelArray[i];
        
        uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

        float3 p;

        p.x = (gridPos.x - gridcenter.x) *voxelSize.x ;
        p.y = (gridPos.y - gridcenter.y) *voxelSize.y ;
        p.z = (gridPos.z - gridcenter.z) *voxelSize.z ;
        
        float3 v[8];
        v[0] = p;
        v[1] = p + make_float3(voxelSize.x, 0, 0);
        v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
        v[3] = p + make_float3(0, voxelSize.y, 0);
        v[4] = p + make_float3(0, 0, voxelSize.z);
        v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
        v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
        v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);



        float field_1[8];
        field_1[0] = sampleVolume(volume_one, gridPos, gridSize);
        field_1[1] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 0), gridSize);
        field_1[2] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 0), gridSize);
        field_1[3] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 0), gridSize);
        field_1[4] = sampleVolume(volume_one, gridPos + make_uint3(0, 0, 1), gridSize);
        field_1[5] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 1), gridSize);
        field_1[6] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 1), gridSize);
        field_1[7] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 1), gridSize);


        float isoVal = isoValue; 
        
        uint cubeindex;
        cubeindex =  uint(field_1[0] < isoVal);
        cubeindex += uint(field_1[1] < isoVal)*2;
        cubeindex += uint(field_1[2] < isoVal)*4;
        cubeindex += uint(field_1[3] < isoVal)*8;
        cubeindex += uint(field_1[4] < isoVal)*16;
        cubeindex += uint(field_1[5] < isoVal)*32;
        cubeindex += uint(field_1[6] < isoVal)*64;
        cubeindex += uint(field_1[7] < isoVal)*128;

        uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

        if(numVerts > 0)
        {
            __shared__ float3 vertlist[12*NTHREADS];
            vertlist[threadIdx.x] = vertexInterp4(isoValue1, v[0], v[1], field_1[0], field_1[1]);
            vertlist[NTHREADS+threadIdx.x] = vertexInterp4(isoValue1, v[1], v[2], field_1[1], field_1[2]);
            vertlist[(NTHREADS*2)+threadIdx.x] = vertexInterp4(isoValue1, v[2], v[3], field_1[2], field_1[3]);
            vertlist[(NTHREADS*3)+threadIdx.x] = vertexInterp4(isoValue1, v[3], v[0], field_1[3], field_1[0]);
            vertlist[(NTHREADS*4)+threadIdx.x] = vertexInterp4(isoValue1, v[4], v[5], field_1[4], field_1[5]);
            vertlist[(NTHREADS*5)+threadIdx.x] = vertexInterp4(isoValue1, v[5], v[6], field_1[5], field_1[6]);
            vertlist[(NTHREADS*6)+threadIdx.x] = vertexInterp4(isoValue1, v[6], v[7], field_1[6], field_1[7]);
            vertlist[(NTHREADS*7)+threadIdx.x] = vertexInterp4(isoValue1, v[7], v[4], field_1[7], field_1[4]);
            vertlist[(NTHREADS*8)+threadIdx.x] = vertexInterp4(isoValue1, v[0], v[4], field_1[0], field_1[4]);
            vertlist[(NTHREADS*9)+threadIdx.x] = vertexInterp4(isoValue1, v[1], v[5], field_1[1], field_1[5]);
            vertlist[(NTHREADS*10)+threadIdx.x] = vertexInterp4(isoValue1, v[2], v[6], field_1[2], field_1[6]);
            vertlist[(NTHREADS*11)+threadIdx.x] = vertexInterp4(isoValue1, v[3], v[7], field_1[3], field_1[7]);
            

            for (int j = 0; j<numVerts; j += 3)
            {
                uint index;
                
                index = numVertsScanned[voxel] + j;
                
                float3 *v[3];

                uint edge;

                edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j);

                
                v[0] = &vertlist[(edge*NTHREADS)+threadIdx.x];

                edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j + 1);

                
                v[1] = &vertlist[(edge*NTHREADS)+threadIdx.x];
                

                edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j + 2);
                
                v[2] = &vertlist[(edge*NTHREADS)+threadIdx.x];


                

                float3 n = calcNormal(v[0], v[1], v[2]);
            
                if (index < (maxVerts - 3))
                {
            
                    pos[index] = make_float4(*v[0], 1.0f);
                    norm[index] = make_float4(n, 0.0f);

                    pos[index+1] = make_float4(*v[1], 1.0f);
                    norm[index+1] = make_float4(n, 0.0f);

                    pos[index+2] = make_float4(*v[2], 1.0f);
                    norm[index+2] = make_float4(n, 0.0f);    
                
                }
            }
        }

      
    }


}

void MarchingCubeCuda::generateTriangles_lattice_2(dim3 grid, dim3 threads,
                          float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
                          uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                          float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts, uint totalverts,float *volume_one,float isovalue1)
{
    
    generateTriangles_lattice_kernel_2<<<grid, threads>>>(pos, norm,
                                           compactedVoxelArray,
                                           numVertsScanned,
                                           gridSize, gridSizeShift, gridSizeMask,
                                           voxelSize,gridcenter, isoValue, activeVoxels,
                                           maxVerts, triTex_s, numVertsTex_s, totalverts, volume_one,isovalue1);
    cudaDeviceSynchronize();
    getLastCudaError("generateTriangles failed");
    cudaError_t err = cudaGetLastError();
    
}




__global__ void
generateTriangles_lattice_kernel_3(float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, 
                   uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                   float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts,
                   cudaTextureObject_t triTex, cudaTextureObject_t numVertsTex,uint totalverts, float *volume_one, float isoValue1)
{
    
    
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
    
    if (i < activeVoxels)
    {
             
        uint voxel = compactedVoxelArray[i];
        
        uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

        float3 p;

        p.x = (gridPos.x - gridcenter.x) *voxelSize.x ;
        p.y = (gridPos.y - gridcenter.y) *voxelSize.y ;
        p.z = (gridPos.z - gridcenter.z) *voxelSize.z ;
        
        float3 v[8];
        v[0] = p;
        v[1] = p + make_float3(voxelSize.x, 0, 0);
        v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
        v[3] = p + make_float3(0, voxelSize.y, 0);
        v[4] = p + make_float3(0, 0, voxelSize.z);
        v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
        v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
        v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);



        float field_1[8];
        field_1[0] = sampleVolume(volume_one, gridPos, gridSize);
        field_1[1] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 0), gridSize);
        field_1[2] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 0), gridSize);
        field_1[3] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 0), gridSize);
        field_1[4] = sampleVolume(volume_one, gridPos + make_uint3(0, 0, 1), gridSize);
        field_1[5] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 1), gridSize);
        field_1[6] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 1), gridSize);
        field_1[7] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 1), gridSize);


        float isoVal = isoValue; 
        
        uint cubeindex;

        cubeindex =  uint(field_1[0] < (isoVal)) && uint(field_1[0] > (0.0f));
        cubeindex += (uint(field_1[1] < (isoVal)) && uint(field_1[1] > (0.0f))) *2;
        cubeindex += (uint(field_1[2] < (isoVal)) && uint(field_1[2] > (0.0f))) *4;
        cubeindex += (uint(field_1[3] < (isoVal)) && uint(field_1[3] > (0.0f))) *8;
        cubeindex += (uint(field_1[4] < (isoVal)) && uint(field_1[4] > (0.0f))) *16;
        cubeindex += (uint(field_1[5] < (isoVal)) && uint(field_1[5] > (0.0f))) *32;
        cubeindex += (uint(field_1[6] < (isoVal)) && uint(field_1[6] > (0.0f))) *64;
        cubeindex += (uint(field_1[7] < (isoVal)) && uint(field_1[7] > (0.0f))) *128;

        uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

        if(numVerts > 0)
        {
            __shared__ float3 vertlist[12*NTHREADS];
            vertlist[threadIdx.x] = vertexInterp4(isoValue1, v[0], v[1], field_1[0], field_1[1]);
            vertlist[NTHREADS+threadIdx.x] = vertexInterp4(isoValue1, v[1], v[2], field_1[1], field_1[2]);
            vertlist[(NTHREADS*2)+threadIdx.x] = vertexInterp4(isoValue1, v[2], v[3], field_1[2], field_1[3]);
            vertlist[(NTHREADS*3)+threadIdx.x] = vertexInterp4(isoValue1, v[3], v[0], field_1[3], field_1[0]);
            vertlist[(NTHREADS*4)+threadIdx.x] = vertexInterp4(isoValue1, v[4], v[5], field_1[4], field_1[5]);
            vertlist[(NTHREADS*5)+threadIdx.x] = vertexInterp4(isoValue1, v[5], v[6], field_1[5], field_1[6]);
            vertlist[(NTHREADS*6)+threadIdx.x] = vertexInterp4(isoValue1, v[6], v[7], field_1[6], field_1[7]);
            vertlist[(NTHREADS*7)+threadIdx.x] = vertexInterp4(isoValue1, v[7], v[4], field_1[7], field_1[4]);
            vertlist[(NTHREADS*8)+threadIdx.x] = vertexInterp4(isoValue1, v[0], v[4], field_1[0], field_1[4]);
            vertlist[(NTHREADS*9)+threadIdx.x] = vertexInterp4(isoValue1, v[1], v[5], field_1[1], field_1[5]);
            vertlist[(NTHREADS*10)+threadIdx.x] = vertexInterp4(isoValue1, v[2], v[6], field_1[2], field_1[6]);
            vertlist[(NTHREADS*11)+threadIdx.x] = vertexInterp4(isoValue1, v[3], v[7], field_1[3], field_1[7]);
            

            for (int j = 0; j<numVerts; j += 3)
            {
                uint index;
                
                index = numVertsScanned[voxel] + j;
                
                float3 *v[3];

                uint edge;

                edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j);

                
                v[0] = &vertlist[(edge*NTHREADS)+threadIdx.x];

                edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j + 1);

                
                v[1] = &vertlist[(edge*NTHREADS)+threadIdx.x];
                

                edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j + 2);
                
                v[2] = &vertlist[(edge*NTHREADS)+threadIdx.x];


                

                float3 n = calcNormal(v[0], v[1], v[2]);
            
                if (index < (maxVerts - 3))
                {
            
                    pos[index] = make_float4(*v[0], 1.0f);
                    norm[index] = make_float4(n, 0.0f);

                    pos[index+1] = make_float4(*v[1], 1.0f);
                    norm[index+1] = make_float4(n, 0.0f);

                    pos[index+2] = make_float4(*v[2], 1.0f);
                    norm[index+2] = make_float4(n, 0.0f);    
                
                }
            }
        }

      
    }


}

void MarchingCubeCuda::generateTriangles_lattice_3(dim3 grid, dim3 threads,
                          float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
                          uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                          float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts, uint totalverts,float *volume_one,float isovalue1)
{
    
    generateTriangles_lattice_kernel_3<<<grid, threads>>>(pos, norm,
                                           compactedVoxelArray,
                                           numVertsScanned,
                                           gridSize, gridSizeShift, gridSizeMask,
                                           voxelSize,gridcenter, isoValue, activeVoxels,
                                           maxVerts, triTex_s, numVertsTex_s, totalverts, volume_one,isovalue1);
    cudaDeviceSynchronize();
    getLastCudaError("generateTriangles failed");
    cudaError_t err = cudaGetLastError();
    
}









void MarchingCubeCuda::ThrustScanWrapper_lattice(unsigned int *output, unsigned int *input, unsigned int numElements)
{
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input),
                           thrust::device_ptr<unsigned int>(input + numElements),
                           thrust::device_ptr<unsigned int>(output));
}



__global__ void
classifyVoxel_new(uint *voxelVerts, uint *voxelOccupied, float *volume,
              uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
              float3 voxelSize, float isoValue, cudaTextureObject_t numVertsTex)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    if (i < numVoxels)
    {
        uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);
    
        float field[8];
        field[0] = sampleVolume(volume, gridPos, gridSize);
        field[1] = sampleVolume(volume, gridPos + make_uint3(1, 0, 0), gridSize);
        field[2] = sampleVolume(volume, gridPos + make_uint3(1, 1, 0), gridSize);
        field[3] = sampleVolume(volume, gridPos + make_uint3(0, 1, 0), gridSize);
        field[4] = sampleVolume(volume, gridPos + make_uint3(0, 0, 1), gridSize);
        field[5] = sampleVolume(volume, gridPos + make_uint3(1, 0, 1), gridSize);
        field[6] = sampleVolume(volume, gridPos + make_uint3(1, 1, 1), gridSize);
        field[7] = sampleVolume(volume, gridPos + make_uint3(0, 1, 1), gridSize);

        float isoVal = isoValue;
     
        uint cubeindex;
        cubeindex =  uint(field[0] < (isoVal));
        cubeindex += uint(field[1] < (isoVal))*2;
        cubeindex += uint(field[2] < (isoVal))*4;
        cubeindex += uint(field[3] < (isoVal))*8;
        cubeindex += uint(field[4] < (isoVal))*16;
        cubeindex += uint(field[5] < (isoVal))*32;
        cubeindex += uint(field[6] < (isoVal))*64;
        cubeindex += uint(field[7] < (isoVal))*128;
        uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

        voxelVerts[i] = numVerts;

        voxelOccupied[i] = (numVerts > 0);
 
    }
  
 
}

void MarchingCubeCuda::classifyVoxel_lattice_new(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, float *volume,
                     uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
                     float3 voxelSize, float isoValue)
{

   
    classifyVoxel_new<<<grid, threads>>>(voxelVerts, voxelOccupied, volume,
                                     gridSize, gridSizeShift, gridSizeMask,
                                     numVoxels, voxelSize, isoValue, numVertsTex_s);
    cudaDeviceSynchronize();

    getLastCudaError("classifyVoxel failed");

   
}



__device__
float3  vertexInterp3_new(float isolevel,float isolevelone,float isoleveltwo, float3 p0, float3 p1, float f0, float f1,float f2, float f3, float iso1, float iso2,float f_id0, float f_id1)
{
    
    
    float t ;// = 0.0;

  

    if (((f_id0  == 1) && (f_id1 == 0)) || ((f_id0  == 0) && (f_id1 == 1)))
    {
        
        if (f1 < f0)
        {
            float3 temp;
            temp = p1;
            p1 = p0;
            p0 = temp;    

            float tm;
            tm = f1;
            f1 = f0;
            f0 = tm;
        }

        
        if((f1 >= isolevelone) && (f0 <= isolevelone))
        {
            if (fabs(isolevelone-f0) < 0.0005)
            {
                return(p0);
            }
            if (fabs(isolevelone-f1) < 0.0005)
            {
                return(p1);
            }
            if (fabs(f1-f0) < 0.0005)
            {
                return(p0);
            }

            t = (isolevelone - f0) / (f1 - f0);
        }



        else if((f1 >= isoleveltwo) && (f0 <= isoleveltwo))
        {
        

            if (fabs(isoleveltwo -f0) < 0.0005)
            {
                return(p0);
            }
            if (fabs(isoleveltwo -f1) < 0.0005)
            {
                return(p1);
            }
            if (fabs(f1-f0) < 0.0005)
            {
                return(p0);
            }

            t = (isoleveltwo - f0) / (f1 - f0);
        }

        else if((f1  == f0) && (p0.z == 0.0))
        {
            t = 1;
        }
        else if((f1  == f0) )
        {
            t = 0;
        }

    }


    if(((f_id0  == 2) && (f_id1 == 0)) || ((f_id1  == 2) && (f_id0 == 0)))
    {
        
        if (f3 < f2)
        {
            float3 temp;
            temp = p1;
            p1 = p0;
            p0 = temp;    

            float tm;
            tm = f3;
            f3 = f2;
            f2 = tm;
        }
        
        
        if((f3 >= iso1) && (f2 <= iso1))
        {
            if (fabs(iso1-f2) < 0.0005)
            {
                return(p0);
            }
            if (fabs(iso1-f3) < 0.0005)
            {
                return(p1);
            }
            if (fabs(f3-f2) < 0.0005)
            {
                return(p0);
            }

            t = (iso1 - f2) / (f3 - f2);
        }


        else if((f3 >= iso2) && (f2 <= iso2))
        {
        

            if (fabs(iso2-f2) < 0.0005)
            {
                return(p0);
            }
            if (fabs(iso2-f3) < 0.0005)
            {
                return(p1);
            }
            if (fabs(f3-f2) < 0.0005)
            {
                return(p0);
            }

            t = (iso2 - f2) / (f3 - f2);

        }

        else if((f3  == f2) && (p0.z == 0.0))
        {
            t = 1;
        }
        else if((f3  == f2))
        {
            t = 0;
        }
  
    }
    
    return lerp(p0, p1, t);
}







__global__ void
generateTriangles_lattice_kernel_new(float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, float *volume,
                   uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                   float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts,
                   cudaTextureObject_t triTex, cudaTextureObject_t numVertsTex,uint totalverts, float *volume_one,float *volume_two,float isoValue1,float isoValue2, float iso1 ,float iso2)
{
    
    
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
    
    if (i < activeVoxels)
    {
             
        uint voxel = compactedVoxelArray[i];
        
        uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

        float3 p;

        p.x = (gridPos.x - gridcenter.x) *voxelSize.x ;
        p.y = (gridPos.y - gridcenter.y) *voxelSize.y ;
        p.z = (gridPos.z - gridcenter.z) *voxelSize.z ;
        
        float3 v[8];
        v[0] = p;
        v[1] = p + make_float3(voxelSize.x, 0, 0);
        v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
        v[3] = p + make_float3(0, voxelSize.y, 0);
        v[4] = p + make_float3(0, 0, voxelSize.z);
        v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
        v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
        v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);


        
        float field_id[8];
        field_id[0] = sampleVolume(volume, gridPos, gridSize);
        field_id[1] = sampleVolume(volume, gridPos + make_uint3(1, 0, 0), gridSize);
        field_id[2] = sampleVolume(volume, gridPos + make_uint3(1, 1, 0), gridSize);
        field_id[3] = sampleVolume(volume, gridPos + make_uint3(0, 1, 0), gridSize);
        field_id[4] = sampleVolume(volume, gridPos + make_uint3(0, 0, 1), gridSize);
        field_id[5] = sampleVolume(volume, gridPos + make_uint3(1, 0, 1), gridSize);
        field_id[6] = sampleVolume(volume, gridPos + make_uint3(1, 1, 1), gridSize);
        field_id[7] = sampleVolume(volume, gridPos + make_uint3(0, 1, 1), gridSize);

        float isoVal = isoValue; 
        
        uint cubeindex;
        cubeindex =  uint(field_id[0] < isoVal);
        cubeindex += uint(field_id[1] < isoVal)*2;
        cubeindex += uint(field_id[2] < isoVal)*4;
        cubeindex += uint(field_id[3] < isoVal)*8;
        cubeindex += uint(field_id[4] < isoVal)*16;
        cubeindex += uint(field_id[5] < isoVal)*32;
        cubeindex += uint(field_id[6] < isoVal)*64;
        cubeindex += uint(field_id[7] < isoVal)*128;
        

        float field_1[8];
        field_1[0] = sampleVolume(volume_one, gridPos, gridSize);
        field_1[1] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 0), gridSize);
        field_1[2] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 0), gridSize);
        field_1[3] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 0), gridSize);
        field_1[4] = sampleVolume(volume_one, gridPos + make_uint3(0, 0, 1), gridSize);
        field_1[5] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 1), gridSize);
        field_1[6] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 1), gridSize);
        field_1[7] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 1), gridSize);


        float field_2[8];
        field_2[0] = sampleVolume(volume_two, gridPos, gridSize);
        field_2[1] = sampleVolume(volume_two, gridPos + make_uint3(1, 0, 0), gridSize);
        field_2[2] = sampleVolume(volume_two, gridPos + make_uint3(1, 1, 0), gridSize);
        field_2[3] = sampleVolume(volume_two, gridPos + make_uint3(0, 1, 0), gridSize);
        field_2[4] = sampleVolume(volume_two, gridPos + make_uint3(0, 0, 1), gridSize);
        field_2[5] = sampleVolume(volume_two, gridPos + make_uint3(1, 0, 1), gridSize);
        field_2[6] = sampleVolume(volume_two, gridPos + make_uint3(1, 1, 1), gridSize);
        field_2[7] = sampleVolume(volume_two, gridPos + make_uint3(0, 1, 1), gridSize);


        __shared__ float3 vertlist[12*NTHREADS];
        vertlist[threadIdx.x] = vertexInterp3_new(isoValue,isoValue1,isoValue2, v[0], v[1], field_1[0], field_1[1],field_2[0], field_2[1],iso1,iso2,field_id[0], field_id[1]);
        vertlist[NTHREADS+threadIdx.x] = vertexInterp3_new(isoValue,isoValue1,isoValue2, v[1], v[2], field_1[1], field_1[2],field_2[1], field_2[2],iso1,iso2,field_id[1], field_id[2]);
        vertlist[(NTHREADS*2)+threadIdx.x] = vertexInterp3_new(isoValue,isoValue1,isoValue2, v[2], v[3], field_1[2], field_1[3],field_2[2], field_2[3],iso1,iso2,field_id[2], field_id[3]);
        vertlist[(NTHREADS*3)+threadIdx.x] = vertexInterp3_new(isoValue,isoValue1,isoValue2, v[3], v[0], field_1[3], field_1[0],field_2[3], field_2[0],iso1,iso2,field_id[3], field_id[0]);
        vertlist[(NTHREADS*4)+threadIdx.x] = vertexInterp3_new(isoValue,isoValue1,isoValue2, v[4], v[5], field_1[4], field_1[5],field_2[4], field_2[5],iso1,iso2,field_id[4], field_id[5]);
        vertlist[(NTHREADS*5)+threadIdx.x] = vertexInterp3_new(isoValue,isoValue1,isoValue2, v[5], v[6], field_1[5], field_1[6],field_2[5], field_2[6],iso1,iso2,field_id[5], field_id[6]);
        vertlist[(NTHREADS*6)+threadIdx.x] = vertexInterp3_new(isoValue,isoValue1,isoValue2, v[6], v[7], field_1[6], field_1[7],field_2[6], field_2[7],iso1,iso2,field_id[6], field_id[7]);
        vertlist[(NTHREADS*7)+threadIdx.x] = vertexInterp3_new(isoValue,isoValue1,isoValue2, v[7], v[4], field_1[7], field_1[4],field_2[7], field_2[4],iso1,iso2,field_id[7], field_id[4]);
        vertlist[(NTHREADS*8)+threadIdx.x] = vertexInterp3_new(isoValue,isoValue1,isoValue2, v[0], v[4], field_1[0], field_1[4],field_2[0], field_2[4],iso1,iso2,field_id[0], field_id[4]);
        vertlist[(NTHREADS*9)+threadIdx.x] = vertexInterp3_new(isoValue,isoValue1,isoValue2, v[1], v[5], field_1[1], field_1[5],field_2[1], field_2[5],iso1,iso2,field_id[1], field_id[5]);
        vertlist[(NTHREADS*10)+threadIdx.x] = vertexInterp3_new(isoValue,isoValue1,isoValue2, v[2], v[6], field_1[2], field_1[6],field_2[2], field_2[6],iso1,iso2,field_id[2], field_id[6]);
        vertlist[(NTHREADS*11)+threadIdx.x] = vertexInterp3_new(isoValue,isoValue1,isoValue2, v[3], v[7], field_1[3], field_1[7],field_2[3], field_2[7],iso1,iso2,field_id[3], field_id[7]);
        


        uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

        for (int j =0; j<numVerts; j += 3)
        {
            uint index;
            
            index = numVertsScanned[voxel] + j;
            
            float3 *v[3];

            uint edge;

            edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j);

            
            v[0] = &vertlist[(edge*NTHREADS)+threadIdx.x];

            edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j + 1);

            
            v[1] = &vertlist[(edge*NTHREADS)+threadIdx.x];
            

            edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j + 2);
            
            v[2] = &vertlist[(edge*NTHREADS)+threadIdx.x];
            

            float3 n = calcNormal(v[0], v[1], v[2]);
        
            if (index < (maxVerts - 3))
            {
        
                pos[index] = make_float4(*v[0], 1.0f);
                norm[index] = make_float4(n, 0.0f);

                pos[index+1] = make_float4(*v[1], 1.0f);
                norm[index+1] = make_float4(n, 0.0f);

                pos[index+2] = make_float4(*v[2], 1.0f);
                norm[index+2] = make_float4(n, 0.0f);    
            
            }
        }
        
    }


    
}

void MarchingCubeCuda::generateTriangles_lattice_new(dim3 grid, dim3 threads,
                          float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, float *volume,
                          uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                          float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts, uint totalverts,float *volume_one,float *volume_two,float isovalue1,float isovalue2, float iso1, float iso2)
{
    
    generateTriangles_lattice_kernel_new<<<grid, threads>>>(pos, norm,
                                           compactedVoxelArray,
                                           numVertsScanned, volume,
                                           gridSize, gridSizeShift, gridSizeMask,
                                           voxelSize,gridcenter, isoValue, activeVoxels,
                                           maxVerts, triTex_s, numVertsTex_s, totalverts, volume_one,volume_two,isovalue1,isovalue2,iso1,iso2);
    cudaDeviceSynchronize();
    getLastCudaError("generateTriangles failed");
    cudaError_t err = cudaGetLastError();
    
}




__device__
float3  vertexInterp2_new(float isolevelone,float isoleveltwo, float3 p0, float3 p1, float f0, float f1,float f_id0, float f_id1)
{
    
    
    float t ;// = 0.0;

  

    if (((f_id0  == 1) && (f_id1 == 0)) || ((f_id0  == 0) && (f_id1 == 1)))
    {
        
        if (f1 < f0)
        {
            float3 temp;
            temp = p1;
            p1 = p0;
            p0 = temp;    

            float tm;
            tm = f1;
            f1 = f0;
            f0 = tm;
        }

        
        
        
        if((f1 >= isolevelone) && (f0 <= isolevelone))
        {
            if (fabs(isolevelone-f0) < 0.0005)
            {
                return(p0);
            }
            if (fabs(isolevelone-f1) < 0.0005)
            {
                return(p1);
            }
            if (fabs(f1-f0) < 0.0005)
            {
                return(p0);
            }

            t = (isolevelone - f0) / (f1 - f0);
        }



        else if((f1 >= isoleveltwo) && (f0 <= isoleveltwo))
        {
        

            if (fabs(isoleveltwo -f0) < 0.0005)
            {
                return(p0);
            }
            if (fabs(isoleveltwo -f1) < 0.0005)
            {
                return(p1);
            }
            if (fabs(f1-f0) < 0.0005)
            {
                return(p0);
            }

            t = (isoleveltwo - f0) / (f1 - f0);
        }

        else if((f1  == f0) && (p0.z == 0.0))
        {
            t = 1;
        }
        else if((f1  == f0) )
        {
            t = 0;
        }

    }

    
    return lerp(p0, p1, t);
}



__global__ void
generateTriangles_lattice_kernel_newone(float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, float *volume,
                   uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                   float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts,
                   cudaTextureObject_t triTex, cudaTextureObject_t numVertsTex,uint totalverts, float *volume_one,float isoValue1,float isoValue2)
{
    
    
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
 
  
 
    if (i < activeVoxels)
    {
     
        uint voxel = compactedVoxelArray[i];
        
        uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

        float3 p;

        p.x = (gridPos.x - gridcenter.x) *voxelSize.x ;
        p.y = (gridPos.y - gridcenter.y) *voxelSize.y ;
        p.z = (gridPos.z - gridcenter.z) *voxelSize.z ;
        
        float3 v[8];
        v[0] = p;
        v[1] = p + make_float3(voxelSize.x, 0, 0);
        v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
        v[3] = p + make_float3(0, voxelSize.y, 0);
        v[4] = p + make_float3(0, 0, voxelSize.z);
        v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
        v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
        v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

     
        
        float field_id[8];
        field_id[0] = sampleVolume(volume, gridPos, gridSize);
        field_id[1] = sampleVolume(volume, gridPos + make_uint3(1, 0, 0), gridSize);
        field_id[2] = sampleVolume(volume, gridPos + make_uint3(1, 1, 0), gridSize);
        field_id[3] = sampleVolume(volume, gridPos + make_uint3(0, 1, 0), gridSize);
        field_id[4] = sampleVolume(volume, gridPos + make_uint3(0, 0, 1), gridSize);
        field_id[5] = sampleVolume(volume, gridPos + make_uint3(1, 0, 1), gridSize);
        field_id[6] = sampleVolume(volume, gridPos + make_uint3(1, 1, 1), gridSize);
        field_id[7] = sampleVolume(volume, gridPos + make_uint3(0, 1, 1), gridSize);

        float isoVal = isoValue; 
        
        uint cubeindex;
        cubeindex =  uint(field_id[0] < isoVal);
        cubeindex += uint(field_id[1] < isoVal)*2;
        cubeindex += uint(field_id[2] < isoVal)*4;
        cubeindex += uint(field_id[3] < isoVal)*8;
        cubeindex += uint(field_id[4] < isoVal)*16;
        cubeindex += uint(field_id[5] < isoVal)*32;
        cubeindex += uint(field_id[6] < isoVal)*64;
        cubeindex += uint(field_id[7] < isoVal)*128;
        

        float field_1[8];
        field_1[0] = sampleVolume(volume_one, gridPos, gridSize);
        field_1[1] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 0), gridSize);
        field_1[2] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 0), gridSize);
        field_1[3] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 0), gridSize);
        field_1[4] = sampleVolume(volume_one, gridPos + make_uint3(0, 0, 1), gridSize);
        field_1[5] = sampleVolume(volume_one, gridPos + make_uint3(1, 0, 1), gridSize);
        field_1[6] = sampleVolume(volume_one, gridPos + make_uint3(1, 1, 1), gridSize);
        field_1[7] = sampleVolume(volume_one, gridPos + make_uint3(0, 1, 1), gridSize);

  
        __shared__ float3 vertlist[12*NTHREADS];
        vertlist[threadIdx.x] = vertexInterp2_new(isoValue1,isoValue2, v[0], v[1], field_1[0], field_1[1],field_id[0], field_id[1]);
        vertlist[NTHREADS+threadIdx.x] = vertexInterp2_new(isoValue1,isoValue2, v[1], v[2], field_1[1], field_1[2],field_id[1], field_id[2]);
        vertlist[(NTHREADS*2)+threadIdx.x] = vertexInterp2_new(isoValue1,isoValue2, v[2], v[3], field_1[2], field_1[3],field_id[2], field_id[3]);
        vertlist[(NTHREADS*3)+threadIdx.x] = vertexInterp2_new(isoValue1,isoValue2, v[3], v[0], field_1[3], field_1[0],field_id[3], field_id[0]);
        vertlist[(NTHREADS*4)+threadIdx.x] = vertexInterp2_new(isoValue1,isoValue2, v[4], v[5], field_1[4], field_1[5],field_id[4], field_id[5]);
        vertlist[(NTHREADS*5)+threadIdx.x] = vertexInterp2_new(isoValue1,isoValue2, v[5], v[6], field_1[5], field_1[6],field_id[5], field_id[6]);
        vertlist[(NTHREADS*6)+threadIdx.x] = vertexInterp2_new(isoValue1,isoValue2, v[6], v[7], field_1[6], field_1[7],field_id[6], field_id[7]);
        vertlist[(NTHREADS*7)+threadIdx.x] = vertexInterp2_new(isoValue1,isoValue2, v[7], v[4], field_1[7], field_1[4],field_id[7], field_id[4]);
        vertlist[(NTHREADS*8)+threadIdx.x] = vertexInterp2_new(isoValue1,isoValue2, v[0], v[4], field_1[0], field_1[4],field_id[0], field_id[4]);
        vertlist[(NTHREADS*9)+threadIdx.x] = vertexInterp2_new(isoValue1,isoValue2, v[1], v[5], field_1[1], field_1[5],field_id[1], field_id[5]);
        vertlist[(NTHREADS*10)+threadIdx.x] = vertexInterp2_new(isoValue1,isoValue2, v[2], v[6], field_1[2], field_1[6],field_id[2], field_id[6]);
        vertlist[(NTHREADS*11)+threadIdx.x] = vertexInterp2_new(isoValue1,isoValue2, v[3], v[7], field_1[3], field_1[7],field_id[3], field_id[7]);
        


        uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

        for (int j =0; j<numVerts; j += 3)
        {
            uint index;
            
            index = numVertsScanned[voxel] + j;
          
            float3 *v[3];

            uint edge;

            edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j);

            
            v[0] = &vertlist[(edge*NTHREADS)+threadIdx.x];

            edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j + 1);

            
            v[1] = &vertlist[(edge*NTHREADS)+threadIdx.x];
            

            edge = tex1Dfetch<uint>(triTex, (cubeindex*16) + j + 2);
            
            v[2] = &vertlist[(edge*NTHREADS)+threadIdx.x];
          

            float3 n = calcNormal(v[0], v[1], v[2]);
        
            if (index < (maxVerts - 3))
        
            {
                
                pos[index] = make_float4(*v[0], 1.0f);
                norm[index] = make_float4(n, 0.0f);

                pos[index+1] = make_float4(*v[1], 1.0f);
                norm[index+1] = make_float4(n, 0.0f);

                pos[index+2] = make_float4(*v[2], 1.0f);
                norm[index+2] = make_float4(n, 0.0f);    
             
            
            }
        }
        
    }


    
}


void MarchingCubeCuda::generateTriangles_lattice_newone(dim3 grid, dim3 threads,
                          float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, float *volume,
                          uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                          float3 voxelSize, float3 gridcenter, float isoValue, uint activeVoxels, uint maxVerts, uint totalverts,float *volume_one,float isovalue1,float isovalue2)
{
    
    generateTriangles_lattice_kernel_newone<<< grid, threads>>>(pos, norm,
                                           compactedVoxelArray,
                                           numVertsScanned, volume,
                                           gridSize, gridSizeShift, gridSizeMask,
                                           voxelSize,gridcenter, isoValue, activeVoxels,
                                           maxVerts, triTex_s, numVertsTex_s, totalverts, volume_one,isovalue1,isovalue2);
    cudaDeviceSynchronize();
    getLastCudaError("generateTriangles failed");
    cudaError_t err = cudaGetLastError();
    
}
