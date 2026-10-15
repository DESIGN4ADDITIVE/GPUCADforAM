#pragma once
#ifndef _APP_MODULE_H_
#define _APP_MODULE_H_

#include <sys/time.h>

#include <sys/stat.h>

#include <errno.h>

#include <unistd.h>

#include "VulkanBaseApp.h"

#include <iomanip>

#define UseGPU

#include <cstdlib>

#include <iostream>

#include <fstream>

#include <cmath>

#include <algorithm>

#include <map>

using namespace std;

#include "./general/DataTypes.h"

#include <cufft.h> 

#include <helper_cuda.h>

#include "Camera_settings.hpp"


#include "./topo_structural/Structuralsim.h"
//////////thermal/////////////////////////

#include "./topo_thermal/Thermalsim.h"
///////////////////////////////////////////
#include "./gpu_kernels/Optimisation_kernels.h"
//////////////////////////////////////////////////////////////////////////

/////////////////LATTICE//////////////////////////////////////////////////
#include "./lattice_files/Fft_lattice.h"
#include "./lattice_files/Gratings.h"
//////////////////////////////////////////////////////////////////////////

////////////////ISOSURFACE EXTRACTION/////////////////////////////////////
#include "Isosurface.h"
//////////////////////////////////////////////////////////////////////////

/////////////////EXPORT DATA//////////////////////////////////////////////
#include "File_output.h"
//////////////////////////////////////////////////////////////////////////

///////////////////////OBJECT PICKING/////////////////////////////////////
#include "Selection.h"
///////////////////////////////////////////////////////////////////////////

/////////////////////INSTNACE RENDERING/////////////////////////////////////
#include "Instances.h"
///////////////////////////////////////////////////////////////////////////

#include <chrono>




class App_Module : public VulkanBaseApp, Modelling
{
    
    public:

    App_Module(bool validation);

    ~App_Module();

    int compute_n_visualise();


    private:
    
    typedef struct UniformBufferObject_st {
        mat4x4 modelViewProj[1];
    } UniformBufferObject;



    VkBuffer v_volumeBuffer_s, v_volumeBuffer_t,v_raster,
    latticeonevol,latticethreevol,v_volume_twice,
    vpos_one,vnorm_one,vpos_two,vnorm_two,
    v_xyzBuffer_s,v_indexBuffer_s,
    v_xyzlatticeBuffer,v_indexlatticeBuffer,
    v_xyzBufferthree,v_indexBufferthree,
    v_loadicon,v_supporticon,v_loadinstance_buffer, v_supportinstance_buffer;

    VkDeviceMemory v_volumeMemory_s,v_volumeMemory_t,v_rasterMemory,
    latticeoneVolMemory,latticethreeVolMemory, v_volume_twice_memory,
    vposMemory_one,vnormMemory_one,
    vposMemory_two,vnormMemory_two,
    v_xyzMemory_s,v_indexMemory_s,
    v_xyzlatticeMemory,v_indexlatticeMemory,
    v_xyzMemorythree,v_indexMemorythree,
    v_loadicon_memory,v_supporticon_memory,v_loadinstance_memory, v_supportinstance_memory;

    UniformBufferObject ubo;

    VkSemaphore v_vkWaitSemaphore, v_vkSignalSemaphore;
 
    Isosurface isosurf;

    Gratings lattice;

    Fft_lattice fftlattice;

    File_output output_file;

    Selection selectt;

    Structuralsim structure;

    Thermalsim thermal;

    Optimisation_kernels opt_kernel;

    cudaExternalSemaphore_t m_cudaWaitSemaphore, m_cudaSignalSemaphore, m_cudaTimelineSemaphore;
    cudaExternalMemory_t m_cudaVertMem_s,m_cudaVertMem_t,m_cudarasterMem,
    m_cudaVertMemone, m_cudaVertMemthree,
    m_cuda_volume_twiceMem,
    m_cudaPos_s,m_cudaNorm_s,
    m_cudaPosone,m_cudaNormone,
    m_cudaPostwo,m_cudaNormtwo,
    m_loadicon_Mem, m_supporticon_Mem,
    m_xyzBuffer_Mem;

    float *d_volume_s, *d_volume_t, *d_volumeone,*d_volumeone_one, *d_volumethree, 

    *d_volumethree_one,*d_volumethree_two, *d_raster, *d_solid, *d_solid_field,  *d_volume_twice;


    InstanceData *d_loadinstance_buffer, *d_supportinstance_buffer;
    
    cudaExternalMemory_t m_loadinstance_Mem, m_supportinstance_Mem;

    std::vector<float*> d_cudastorageBuffers;

    std::vector<cudaExternalMemory_t> d_cudastorageMemory;
   
    float4 *d_posone,*d_normalone,
    *d_postwo, *d_normaltwo, 
    *disp_two;
    

    float3 *d_xyzBuffer_s;

    uint3 gridSize, gridSizeMask ,gridSizeShift;

    float3 voxelSize, gridcenter;

    uint numVoxels, activeVoxels, totalVerts, solid_voxels;

    float dx, dy, dz;

    uint3 gridSizeShifttwo, gridSizetwo, gridSizeMasktwo;
 
    float3 voxelSizetwo, gridcentertwo;

    uint numVoxelstwo, activeVoxelstwo, totalVertstwo;

    uint *d_voxelVertstwo, *d_voxelVertsScantwo, *d_voxelOccupiedtwo, *d_voxelOccupiedScantwo, *d_compVoxelArraytwo, *d_facet_active, *d_facet_active_scan;


    uint3 gridSizeShiftone, gridSizeone, gridSizeMaskone;
 
    float3 voxelSizeone, gridcenterone;

    uint numVoxelsone, activeVoxelsone, totalVertsone;

    uint *d_voxelVertsone, *d_voxelVertsScanone, *d_voxelOccupiedone, *d_voxelOccupiedScanone, *d_compVoxelArrayone;

    int NumX, NumY, NumZ ;

    size_t maxmemverts;

    int volsize;

    float dist2;

    float angle2;

    size_t pitch_bytes, grad_pitch_bytes;

  
    float *d_boundary = NULL;

    float *d_latt_field = NULL;

    grid_points *vol_one = NULL;

    grid_points *vol_topo = NULL;

    triangle_metadata *triangle_data = NULL;

    REAL3 *d_us;

    REAL *d_den, *d_grads, *d_result;

    REAL *d_selection;

    REAL3 *d_loadgroup;

    REAL *d_selection2;

    int FinalIter_s;

    REAL FinalRes_s, Obj_s, Obj_old_s, Vol_s;

    int OptIter;
  
    int FinalIter_t;

    REAL FinalRes_t;

    uint Nxu, Nyu, Nzu;

    uint NumX2, NumY2, NumZ2 ;

    float dx2, dy2, dz2;

    uint size, sizeone, size2;

    size_t maxmemvertstwo, maxmemvertsone, max_nfacets, nfacets;

    float dist1, dist3, distone;

    int indi_range; 

    float *d_phi = NULL;
   
    float *d_theta = NULL;

    float *d_period = NULL;
  
    float2 *d_ga = NULL;

    float *d_svl = NULL;

    char latticetype_one;

    char latticetype_two;

    int range_st ;

    float iso1, iso2;

    int FinalIter_l;

    float FinalRes_l;

    bool load_selection;

    bool boundary_selection;

    bool delete_selection;

    IconData *d_loadicon;

    IconData *d_supporticon;

    int3 loadicon_count;

    int3 support_count;

    



    void destroy_buffers_n_memory();

    void destroy_lattice_buffers();

    void getVertexDescriptions(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc);

    void getVertexDescriptions_u_lattice(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc);

    void getVertexDescriptionsone(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc);

    void getVertexDescriptions_instance(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc);

    void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait, std::vector< VkPipelineStageFlags>& waitStages) const;

    void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;

    void importCudaExternalMemory(void **cudaPtr, cudaExternalMemory_t& cudaMem, VkDeviceMemory& vkMem, VkDeviceSize size, VkExternalMemoryHandleTypeFlagBits handleType);

    void importCudaExternalSemaphore(cudaExternalSemaphore_t& cudaSem, VkSemaphore& vkSem, VkExternalSemaphoreHandleTypeFlagBits handleType);

    void fillRenderingCommandBuffer(VkCommandBuffer& commandBuffer) override;

    void fillRenderingCommandBuffer_subpass1(VkCommandBuffer& commandBuffer);

    void fillRenderingCommandBuffer_unit_lattice(VkCommandBuffer& commandBuffer);

    void fillRenderingCommandBuffer_unit_lattice_subpass1(VkCommandBuffer& commandBuffer);

    void fillRenderingCommandBuffer_spatial_lattice(VkCommandBuffer& commandBuffer);

    void fillRenderingCommandBuffer_spatial_lattice_subpass1(VkCommandBuffer& commandBuffer);

    void fillRenderingCommandBufferone(VkCommandBuffer& commandBuffer);

    void fillRenderingCommandBufferone_subpass1(VkCommandBuffer& buffer) ;

    void fillRenderingCommandBuffer_instance(VkCommandBuffer& commandBuffer);

    void fillRenderingCommandBuffertwo(VkCommandBuffer& commandBuffer);

    void fillRenderingCommandBuffertwo_subpass1(VkCommandBuffer& buffer) ;

    void fillRenderingCommandBufferthree(VkCommandBuffer& commandBuffer);

    void fillRenderingCommandBufferthree_subpass1(VkCommandBuffer& buffer) ;

    void fillRenderingCommandBufferfour(VkCommandBuffer& buffer) ;

    void fillRenderingCommandBufferfour_subpass1(VkCommandBuffer& buffer) ;

    void fillRenderingCommandBuffer_instance_subpass1(VkCommandBuffer& buffer) ;

    VkDeviceSize getUniformSize() const;

    void updateStorageBuffer_two(uint32_t imageIndex, bool load_selection, bool boundary_selection, bool delete_selection);

    void updateStorageBuffer_three(uint32_t imageIndex, bool load_selection, bool boundary_selection, bool delete_selection);

    void update_inputevents();

    void updateUniformBuffer(uint32_t imageIndex,bool shift);

    void drawFrame(bool shift);

    void init_textures();

    void initMC_unitlattice();

    void initMC();

    void initMC_two();

    int3 init_icons(std::string icon_path, VkDevice v_device, VkBuffer& icon_buffer , VkDeviceMemory& icon_memory, IconData *icon_data, cudaExternalMemory_t icon_Mem);


    std::vector<const char *> getRequiredExtensions() const;

    std::vector<const char *> getRequiredDeviceExtensions() const;

    void init_selection();

    void initVulkanCuda_semaphores();

    void createStorageBuffers();

    void re_createStorageBuffers();

    void erase_primitive_data();

    void erase_topo_data();

    void erase_lattice_data();

    void vulkan_create_topo_buffers();

    void vulkan_create_lattice_buffers();

    void update_topo_grid_data();

    int topoinit_struct();

    int toprun_struct();

    int topoinit_thermal();

    int toprun_thermal();

    void init_Boundary();

    void show_model();

    void update_lattice_type();

    int fft_init();

    int lattice_init();

    void init_svl();

    int unit_lattice();

    int unit_latticeone();

    int spatial_lattice_run();

    void check_lattice();

    void check_unit_lattice();

    void display_unit_lattice();

    int mainloopone(bool shift);

    void cleanup_Boundary();

    void cleanup_cuda_storage_buffer_handle();

    void cleanup_svl();

    void cleanup_textures();

    void cleanup_fft_data();

    void cleanup_isosurf();

    void cleanup_isosurf_unit_lattice();

    void cleanup_isosurf_lattice();

    void cleanup_topo();

    void cleanup_selection();

    void free_fft_resource();

    template<typename T> 
    void fill_storage_buffer_one(VkDevice device,std::vector<VkBuffer> buffers,const size_t nfacets,float val);

    void fill_instance_buffer(VkDevice device,const size_t instance_count);

    template <typename T>
    void fill_buffer_xyz(VkDevice device,VkBuffer buffer,const size_t nVerts,int Nx,int Ny, int Nz, float dx, float dy , float dz);

    template <typename T>
    void fill_buffer_indices(VkDevice device,VkBuffer buffer,const size_t nVerts,int Nx,int Ny, int Nz);

    template <typename T>
    void fill_buffer_val(VkDevice device,VkBuffer buffer,const size_t nVerts,int Nx,int Ny, int Nz,float val);

    template <typename T>
    void fill_buffer_vecfield(VkDevice device,VkBuffer buffer,const size_t nVerts,int Nx,int Ny, int Nz);

    template <typename T>
    void fill_buffer_posnorm(VkDevice device,VkBuffer buffer,const size_t maximumverts);

};

#endif //_APP_MODULE_H_