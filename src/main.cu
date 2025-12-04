

#include <sys/time.h>

#include <sys/stat.h>

#include <errno.h>

#include <unistd.h>

#include "VulkanBaseApp.h"

#include <iomanip>

#include "linmath.h"

#define UseGPU

#include <cstdlib>

#include <iostream>

#include <fstream>

#include <cmath>

#include <algorithm>

#include <map>

using namespace std;

#include "../general/DataTypes.h"

#include <cufft.h> 

#include <helper_cuda.h>

///////////OPTIMISATION//////////////////////////////////////////////////
#include "Gauss.h"
///////////structural/////////////////////
#include "../topo_structural/Stiffness_structure.h"
#include "../topo_structural/Structuralsim.h"
//////////thermal/////////////////////////
#include "../topo_thermal/Stiffness_thermal.h"
#include "../topo_thermal/Thermalsim.h"
///////////////////////////////////////////
#include "../gpu_kernels/Optimisation_kernels.h"
//////////////////////////////////////////////////////////////////////////

/////////////////LATTICE//////////////////////////////////////////////////
#include "../lattice_files/Fft_lattice.h"
#include "../lattice_files/Gratings.h"
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

#include <chrono>

struct timeval t1, t2;
double tottime;


cufftHandle planr2c;
cufftHandle planc2r;
cufftHandle planc2c;

////////Marching cube tables//////////////
uint *d_numVertsTable = 0;

uint *d_edgeTable = 0;

uint *d_triTable = 0;
/////////////////////////////////////////

float a_1,b_1,c_1;

static float angle = 0.0;

float select_val;

////////////lattice/////////////////////
cudaPitchedPtr devPitchedPtr;
size_t tPitch = 0;
size_t slicepitch =0;
cudaExtent extend;


static float2 *fft_data = NULL;

static float2 *fft_data_compute = NULL;

static float2 *fft_data_compute_fill = NULL;

static float *fft_gratings = NULL;

static float2 *lattice_data = NULL;
///////////////////////////////////////

class Multitopo : public VulkanBaseApp, Modelling
{

    typedef struct UniformBufferObject_st {
        mat4x4 modelViewProj[1];
    } UniformBufferObject;



    VkBuffer v_structure_s,v_volumeBuffer_s, v_volumeBuffer_t,v_raster,
    latticeonevol,latticethreevol,
    v_pos_s,v_norm_s,vpos_one,vnorm_one,vpos_two,vnorm_two,
    v_xyzBuffer_s,v_indexBuffer_s,
    v_xyzlatticeBuffer,v_indexlatticeBuffer,
    v_xyzBufferthree,v_indexBufferthree;

    VkDeviceMemory v_structureMemory_s,v_volumeMemory_s,v_volumeMemory_t,v_rasterMemory,
    latticeoneVolMemory,latticethreeVolMemory,
    v_posmemory_s,v_normmemory_s,
    vposMemory_one,vnormMemory_one,
    vposMemory_two,vnormMemory_two,
    v_xyzMemory_s,v_indexMemory_s,
    v_xyzlatticeMemory,v_indexlatticeMemory,
    v_xyzMemorythree,v_indexMemorythree;

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
    cudaExternalMemory_t m_cudavulkan_s,m_cudaVertMem_s,m_cudaVertMem_t,m_cudarasterMem,
    m_cudaVertMemone, m_cudaVertMemthree,
    m_cudaPos_s,m_cudaNorm_s,
    m_cudaPosone,m_cudaNormone,
    m_cudaPostwo,m_cudaNormtwo;

    float *d_volume_s, *d_volume_t, *d_volumeone,*d_volumeone_one, *d_volumethree, 

    *d_volumethree_one,*d_volumethree_two, *d_raster;

    std::vector<float*> d_cudastorageBuffers;

    std::vector<cudaExternalMemory_t> d_cudastorageMemory;

    float4 *d_pos , *d_normal,
    *d_posone,
    *d_normalone,
    *d_postwo,
    *d_normaltwo;

    uint3 gridSize, gridSizeMask ,gridSizeShift;

    float3 voxelSize, gridcenter;

    uint numVoxels, activeVoxels, totalVerts;

    float dx, dy, dz;

    uint *d_voxelVerts, *d_voxelVertsScan, *d_voxelOccupied, *d_voxelOccupiedScan, *d_compVoxelArray;


    uint3 gridSizeShifttwo, gridSizetwo, gridSizeMasktwo;
 
    float3 voxelSizetwo, gridcentertwo;

    uint numVoxelstwo, activeVoxelstwo, totalVertstwo;

    uint *d_voxelVertstwo, *d_voxelVertsScantwo, *d_voxelOccupiedtwo, *d_voxelOccupiedScantwo, *d_compVoxelArraytwo;


    uint3 gridSizeShiftone, gridSizeone, gridSizeMaskone;
 
    float3 voxelSizeone, gridcenterone;

    uint numVoxelsone, activeVoxelsone, totalVertsone;

    uint *d_voxelVertsone, *d_voxelVertsScanone, *d_voxelOccupiedone, *d_voxelOccupiedScanone, *d_compVoxelArrayone;

    int NumX, NumY, NumZ ;

    size_t maxmemverts;

    int volsize;

    int dist2;

    float angle2;

    size_t pitch_bytes, grad_pitch_bytes;

    REAL3 *d_cudavulkan_s;
  
    float *d_boundary = NULL;

    float *d_latt_field = NULL;

    grid_points *vol_one = NULL;

    REAL3 *d_us;

    REAL *d_den, *d_grads;

    float *d_volume_twice;

    REAL *d_selection;

    int FinalIter_s;

    REAL FinalRes_s, Obj_s, Obj_old_s, Vol_s;

    int OptIter;
  
    int FinalIter_t;

    REAL FinalRes_t;

    uint Nxu, Nyu, Nzu;

    uint NumX2, NumY2, NumZ2 ;

    float dx2, dy2, dz2;

    uint size, sizeone, size2;

    size_t maxmemvertstwo, maxmemvertsone;
    
    int dist1, dist3, distone;

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

    public:
    

    Multitopo(bool VULKAN_VALIDATION) :

        VulkanBaseApp("GPUCADforAM", VULKAN_VALIDATION),

        Modelling(VulkanBaseApp::grid_value,VulkanBaseApp::grid_value,VulkanBaseApp::grid_value),

        v_structure_s(VK_NULL_HANDLE),

        v_volumeBuffer_s(VK_NULL_HANDLE),

        v_volumeBuffer_t(VK_NULL_HANDLE),

        v_raster(VK_NULL_HANDLE),

        latticeonevol(VK_NULL_HANDLE),

        latticethreevol(VK_NULL_HANDLE),

        v_pos_s(VK_NULL_HANDLE),

        vpos_one(VK_NULL_HANDLE),

        vpos_two(VK_NULL_HANDLE),

        v_norm_s(VK_NULL_HANDLE),

        vnorm_one(VK_NULL_HANDLE),

        vnorm_two(VK_NULL_HANDLE),

        v_xyzBuffer_s(VK_NULL_HANDLE),

        v_indexBuffer_s(VK_NULL_HANDLE),

        v_xyzlatticeBuffer(VK_NULL_HANDLE),

        v_indexlatticeBuffer(VK_NULL_HANDLE),

        v_xyzBufferthree(VK_NULL_HANDLE),

        v_indexBufferthree(VK_NULL_HANDLE),

        v_structureMemory_s(VK_NULL_HANDLE),

        v_volumeMemory_s(VK_NULL_HANDLE),

        v_volumeMemory_t(VK_NULL_HANDLE),

        v_rasterMemory(VK_NULL_HANDLE),

        latticeoneVolMemory(VK_NULL_HANDLE),

        latticethreeVolMemory(VK_NULL_HANDLE),

        v_posmemory_s(VK_NULL_HANDLE),

        v_normmemory_s(VK_NULL_HANDLE),

        v_xyzMemory_s(VK_NULL_HANDLE),

        v_indexMemory_s(VK_NULL_HANDLE),

        v_xyzlatticeMemory(VK_NULL_HANDLE),

        v_indexlatticeMemory(VK_NULL_HANDLE),
        
        v_xyzMemorythree(VK_NULL_HANDLE),

        v_indexMemorythree(VK_NULL_HANDLE),
     
        ubo(),

        isosurf(),
   
        lattice(),

        fftlattice(),

        output_file(),

        selectt(),

        structure(),

        opt_kernel(),
     
        v_vkWaitSemaphore(VK_NULL_HANDLE),

        v_vkSignalSemaphore(VK_NULL_HANDLE),

        m_cudaWaitSemaphore(),

        m_cudaSignalSemaphore(),

        m_cudavulkan_s(),
   
        m_cudaVertMem_s(),

        m_cudaVertMem_t(),
        
        m_cudarasterMem(),

        m_cudaPos_s(),

        m_cudaNorm_s(),

        m_cudaPosone(),

        m_cudaNormone(),

        m_cudaPostwo(),

        m_cudaNormtwo(),

        d_cudastorageBuffers(),

        d_cudastorageMemory(),
       
        d_cudavulkan_s(nullptr),

        d_volume_s(nullptr),

        d_volume_t(nullptr),

        d_raster(nullptr),
     
        d_pos(nullptr),

        d_normal(nullptr),

        d_posone(nullptr),

        d_normalone(nullptr),

        d_postwo(nullptr),

        d_normaltwo(nullptr),

        gridSize(),

        gridSizeMask(),

        voxelSize(),

        numVoxels(0),

        activeVoxels(0),

        totalVerts(0),

        dx(),

        dy(),

        dz(),
    
        gridcenter(),

        d_voxelVerts(),

        d_voxelVertsScan(),

        d_voxelOccupied(),

        d_voxelOccupiedScan(),

        d_compVoxelArray(),

        gridSizeShifttwo(),

        gridSizetwo(),

        gridSizeMasktwo(),

        voxelSizetwo(),

        gridcentertwo(),

        numVoxelstwo(0),

        activeVoxelstwo(0),

        totalVertstwo(0),

        d_voxelVertstwo(nullptr),

        d_voxelVertsScantwo(nullptr),

        d_voxelOccupiedtwo(nullptr),

        d_voxelOccupiedScantwo(nullptr),

        d_compVoxelArraytwo(nullptr),

        gridSizeShiftone(),

        gridSizeone(),

        gridSizeMaskone(),

        voxelSizeone(),

        gridcenterone(),

        numVoxelsone(0),

        activeVoxelsone(0),

        totalVertsone(0),

        d_voxelVertsone(nullptr),

        d_voxelVertsScanone(nullptr),

        d_voxelOccupiedone(nullptr),

        d_voxelOccupiedScanone(nullptr),

        d_compVoxelArrayone(nullptr),

        NumX(),

        NumY(),

        NumZ(),
     
        volsize(),

        maxmemverts(),
 
        dist2(),

        angle2(1.0),

        d_boundary(nullptr),

        d_latt_field(nullptr),

        vol_one(nullptr),

        OptIter(0),
    
        FinalIter_s(-1),

        FinalRes_s(-1.0),

        FinalIter_t(-1),

        FinalRes_t(-1.0),

        Nxu(61),

        Nyu(61),

        Nzu(61),

        NumX2(),

        NumY2(),

        NumZ2(),

        sizeone(Nxu*Nyu*Nzu),
        
        size(),

        size2(),

        d_volumeone(nullptr),

        d_volumeone_one(nullptr),

        d_volumethree(nullptr),

        d_volumethree_one(nullptr),

        d_volumethree_two(nullptr),

        maxmemvertsone(),

        maxmemvertstwo(),

        dist1(),

        dist3(),

        distone(),

        indi_range(5),

        range_st(floor(indi_range/2.0)),
        
        iso1(0.0),

        iso2(0.0),

        FinalIter_l(-1),

        FinalRes_l(-1.0),

        load_selection(false),

        boundary_selection(false),

        delete_selection(false)

        //////////////////////////////////////////////////////////////// 
    {
           
        char aone[] = "../src/shaders/structure_write_grid.vert.spv";
        char atwo[] = "../src/shaders/structure_write_grid.geom.spv";
        char athree[]="../src/shaders/structure_write_grid.frag.spv";
        
        char * write_vertex_shader_path = &aone[0];
        char * write_geometry_shader_path = &atwo[0];
        char * write_fragment_shader_path = &athree[0];
    
        shaderFiles.push_back(std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, write_vertex_shader_path));
        shaderFiles.push_back(std::make_pair(VK_SHADER_STAGE_GEOMETRY_BIT, write_geometry_shader_path));
        shaderFiles.push_back(std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, write_fragment_shader_path));


        char bone[] = "../src/shaders/structure_read_grid.vert.spv";
        char btwo[]="../src/shaders/structure_read_grid.geom.spv";
        char bthree[]="../src/shaders/structure_read_grid.frag.spv";
        
        char * read_vertex_shader_path = &bone[0];
        char * read_geometry_shader_path = &btwo[0];
        char * read_fragment_shader_path = &bthree[0];
        shaderFilesread.push_back(std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, read_vertex_shader_path));
        shaderFilesread.push_back(std::make_pair(VK_SHADER_STAGE_GEOMETRY_BIT, read_geometry_shader_path));
        shaderFilesread.push_back(std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, read_fragment_shader_path));


        char cone[] = "../src/shaders/structure_write_mesh.vert.spv";
        char ctwo[]="../src/shaders/structure_write_mesh.geom.spv";
        char cthree[]="../src/shaders/structure_write_mesh.frag.spv";
        
        char * write_bmesh_vertex_shader_path = &cone[0];
        char * write_bmesh_geometry_shader_path = &ctwo[0];
        char * write_bmesh_fragment_shader_path = &cthree[0];
        shaderFilesone.push_back(std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, write_bmesh_vertex_shader_path));
        shaderFilesone.push_back(std::make_pair(VK_SHADER_STAGE_GEOMETRY_BIT, write_bmesh_geometry_shader_path));
        shaderFilesone.push_back(std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, write_bmesh_fragment_shader_path));

        char done[] = "../src/shaders/structure_read_mesh.vert.spv";
        char dtwo[]="../src/shaders/structure_read_mesh.geom.spv";
        char dthree[]="../src/shaders/structure_read_mesh.frag.spv";
        
        char * read_bmesh_vertex_shader_path = &done[0];
        char * read_bmesh_geometry_shader_path = &dtwo[0];
        char * read_bmesh_fragment_shader_path = &dthree[0];
        shaderFilesoneread.push_back(std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, read_bmesh_vertex_shader_path));
        shaderFilesoneread.push_back(std::make_pair(VK_SHADER_STAGE_GEOMETRY_BIT, read_bmesh_geometry_shader_path));
        shaderFilesoneread.push_back(std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, read_bmesh_fragment_shader_path));

    }


    ~Multitopo() 
    {

        if (v_vkSignalSemaphore != VK_NULL_HANDLE) {
            
            checkCudaErrors(cudaDestroyExternalSemaphore(m_cudaSignalSemaphore));
            vkDestroySemaphore(device, v_vkSignalSemaphore, nullptr);
        }
        if (v_vkWaitSemaphore != VK_NULL_HANDLE) {
            
            checkCudaErrors(cudaDestroyExternalSemaphore(m_cudaWaitSemaphore));
            vkDestroySemaphore(device, v_vkWaitSemaphore, nullptr);
        }

    }

     void destroy_buffers_n_memory()
    {
         
        if (d_cudavulkan_s) {
            checkCudaErrors(cudaDestroyExternalMemory(m_cudavulkan_s));
            checkCudaErrors(cudaFree(d_cudavulkan_s));
        }


        if (d_volume_s) {
            checkCudaErrors(cudaDestroyExternalMemory(m_cudaVertMem_s));
            checkCudaErrors(cudaFree(d_volume_s));
        }

        if (d_volume_t) {
            checkCudaErrors(cudaDestroyExternalMemory(m_cudaVertMem_t));
            checkCudaErrors(cudaFree(d_volume_t));
        }


        if (d_raster) {
            checkCudaErrors(cudaDestroyExternalMemory(m_cudarasterMem));
            checkCudaErrors(cudaFree(d_raster));
        }



        if (d_pos) {
           
            checkCudaErrors(cudaDestroyExternalMemory(m_cudaPos_s));
            checkCudaErrors(cudaFree(d_pos));
        }


        if (d_normal) {
            checkCudaErrors(cudaDestroyExternalMemory(m_cudaNorm_s));
            checkCudaErrors(cudaFree(d_normal));
        }


         
        if (v_xyzBuffer_s != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, v_xyzBuffer_s, nullptr);
        }

            
        if (v_structure_s != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, v_structure_s, nullptr);
        }


        if (v_volumeBuffer_s != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, v_volumeBuffer_s, nullptr);
        }

        if (v_volumeBuffer_t != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, v_volumeBuffer_t, nullptr);
        }

        if (v_raster != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, v_raster, nullptr);
        }

        if (v_pos_s != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, v_pos_s, nullptr);
        }


        if (v_norm_s != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, v_norm_s, nullptr);
        }

        
        if (v_indexBuffer_s != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, v_indexBuffer_s, nullptr);
        }


        if (v_xyzMemory_s != VK_NULL_HANDLE) {
                vkFreeMemory(device, v_xyzMemory_s, nullptr);
            }

     
        if (v_structureMemory_s != VK_NULL_HANDLE) {
            vkFreeMemory(device, v_structureMemory_s, nullptr);
        }


        if (v_volumeMemory_s != VK_NULL_HANDLE) {
            vkFreeMemory(device, v_volumeMemory_s, nullptr);
        }

        if (v_volumeMemory_t != VK_NULL_HANDLE) {
            vkFreeMemory(device, v_volumeMemory_t, nullptr);
        }

        if (v_raster != VK_NULL_HANDLE) {
            vkFreeMemory(device, v_rasterMemory, nullptr);
        }

        if (v_posmemory_s != VK_NULL_HANDLE) {
            vkFreeMemory(device, v_posmemory_s, nullptr);
        }


        if (v_normmemory_s != VK_NULL_HANDLE) {
            vkFreeMemory(device, v_normmemory_s, nullptr);
        }


        if (v_indexMemory_s != VK_NULL_HANDLE) {
            vkFreeMemory(device, v_indexMemory_s, nullptr);
        }

    }

    void destroy_lattice_buffers()
    {
        if (d_volumeone) {
            checkCudaErrors(cudaDestroyExternalMemory(m_cudaVertMemone));
            checkCudaErrors(cudaFree(d_volumeone));
        }

        if (d_volumethree) {
            checkCudaErrors(cudaDestroyExternalMemory(m_cudaVertMemthree));
            checkCudaErrors(cudaFree(d_volumethree));
        }

        if (d_posone) {
           
            checkCudaErrors(cudaDestroyExternalMemory(m_cudaPosone));
            checkCudaErrors(cudaFree(d_posone));
        }

        if (d_normalone) {
            checkCudaErrors(cudaDestroyExternalMemory(m_cudaNormone));
            checkCudaErrors(cudaFree(d_normalone));
        }

        if (d_postwo) {
           
            checkCudaErrors(cudaDestroyExternalMemory(m_cudaPostwo));
            checkCudaErrors(cudaFree(d_postwo));
        }

        if (d_normaltwo) {
            checkCudaErrors(cudaDestroyExternalMemory(m_cudaNormtwo));
            checkCudaErrors(cudaFree(d_normaltwo));
        }

        if (latticeonevol != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, latticeonevol, nullptr);
        }


        if (latticethreevol != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, latticethreevol, nullptr);
        }

        if (vpos_two != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, vpos_two, nullptr);
        }

        if (vnorm_two != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, vnorm_two, nullptr);
        }

        if (vpos_one != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, vpos_one, nullptr);
        }

        if (vnorm_one != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, vnorm_one, nullptr);
        }

        if (v_xyzlatticeBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, v_xyzlatticeBuffer, nullptr);
        }

         if (v_indexlatticeBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, v_indexlatticeBuffer, nullptr);
        }

        if (v_xyzBufferthree != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, v_xyzBufferthree, nullptr);
        }

        if (v_indexBufferthree != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, v_indexBufferthree, nullptr);
        }

        if (latticeoneVolMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, latticeoneVolMemory, nullptr);
        }

        
        if (latticethreeVolMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, latticethreeVolMemory, nullptr);
        }

        if (vposMemory_one != VK_NULL_HANDLE) {
            vkFreeMemory(device, vposMemory_one, nullptr);
        }

        if (vnormMemory_one != VK_NULL_HANDLE) {
            vkFreeMemory(device, vnormMemory_one, nullptr);
        }

        if (vposMemory_two != VK_NULL_HANDLE) {
            vkFreeMemory(device, vposMemory_two, nullptr);
        }

        if (vnormMemory_two != VK_NULL_HANDLE) {
            vkFreeMemory(device, vnormMemory_two, nullptr);
        }

        if (v_xyzlatticeMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, v_xyzlatticeMemory, nullptr);
        }
        
        if (v_indexlatticeMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, v_indexlatticeMemory, nullptr);
        }
   
        if (v_xyzMemorythree != VK_NULL_HANDLE) {
            vkFreeMemory(device, v_xyzMemorythree, nullptr);
        }
        
        if (v_indexMemorythree != VK_NULL_HANDLE) {
            vkFreeMemory(device, v_indexMemorythree, nullptr);
        }

         

    }

 

    void getVertexDescriptions(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc) {
        bindingDesc.resize(3);
        attribDesc.resize(3);

        bindingDesc[0].binding = 0;
        bindingDesc[0].stride = sizeof(REAL);
        bindingDesc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        bindingDesc[1].binding = 1;
        bindingDesc[1].stride = sizeof(vec3);
        bindingDesc[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        bindingDesc[2].binding = 2;
        bindingDesc[2].stride = sizeof(REAL);
        bindingDesc[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;


        attribDesc[0].binding = 0;
        attribDesc[0].location = 0;
        attribDesc[0].format = VK_FORMAT_R32_SFLOAT;
        attribDesc[0].offset = 0;

        attribDesc[1].binding = 1;
        attribDesc[1].location = 1;
        attribDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribDesc[1].offset = 0;

        attribDesc[2].binding = 2;
        attribDesc[2].location = 2;
        attribDesc[2].format = VK_FORMAT_R32_SFLOAT;
        attribDesc[2].offset = 0;

    }


        
    void getVertexDescriptionsone(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc) {
        bindingDesc.resize(2);
        attribDesc.resize(2);

        bindingDesc[0].binding = 0;
        bindingDesc[0].stride = sizeof(vec4);
        bindingDesc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        bindingDesc[1].binding = 1;
        bindingDesc[1].stride = sizeof(vec4);
        bindingDesc[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        attribDesc[0].binding = 0;
        attribDesc[0].location = 0;
        attribDesc[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attribDesc[0].offset = 0;

        attribDesc[1].binding = 1;
        attribDesc[1].location = 1;
        attribDesc[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attribDesc[1].offset = 0;

    }

  

    void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info) {
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        info.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        info.primitiveRestartEnable = VK_FALSE;
    }


    void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait, std::vector< VkPipelineStageFlags>& waitStages) const {
        if (currentFrame != 0) {
            wait.push_back(v_vkWaitSemaphore);
            waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }
    }

    void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const {
        signal.push_back(v_vkSignalSemaphore);
    }

    
    
    void importCudaExternalMemory(void **cudaPtr, cudaExternalMemory_t& cudaMem, VkDeviceMemory& vkMem, VkDeviceSize size, VkExternalMemoryHandleTypeFlagBits handleType) {
        cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};


        if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) 
        {
            externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        }
        else 
        {
            throw std::runtime_error("Unknown handle type requested!");
        }

        externalMemoryHandleDesc.size = size;

      
        externalMemoryHandleDesc.handle.fd = (int)(uintptr_t)getMemHandle(vkMem, handleType);
       

        checkCudaErrors(cudaImportExternalMemory(&cudaMem, &externalMemoryHandleDesc));

        cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
        externalMemBufferDesc.offset = 0;
        externalMemBufferDesc.size = size;
        externalMemBufferDesc.flags = 0;
        checkCudaErrors(cudaExternalMemoryGetMappedBuffer(cudaPtr, cudaMem, &externalMemBufferDesc));
    }


    void importCudaExternalSemaphore(cudaExternalSemaphore_t& cudaSem, VkSemaphore& vkSem, VkExternalSemaphoreHandleTypeFlagBits handleType) {
        cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};

    
        if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
            externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
        }

        else 
        {
            throw std::runtime_error("Unknown handle type requested!");
        }

        externalSemaphoreHandleDesc.handle.fd = (int)(uintptr_t)getSemaphoreHandle(vkSem, handleType);
       
        externalSemaphoreHandleDesc.flags = 0;

        checkCudaErrors(cudaImportExternalSemaphore(&cudaSem, &externalSemaphoreHandleDesc));
    }

    void fillRenderingCommandBuffer(VkCommandBuffer& commandBuffer) 
    {
   
    
        if(ImguiApp::thermal)
        {
            VkBuffer vertexBuffers[] = { v_volumeBuffer_t,v_xyzBuffer_s,v_raster};

            VkDeviceSize offsets[] = { 0, 0, 0};

            vkCmdPushConstants(commandBuffer,pipelineLayout,VK_SHADER_STAGE_GEOMETRY_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);

            vkCmdBindVertexBuffers(commandBuffer, 0, 3, vertexBuffers, offsets);

            vkCmdBindIndexBuffer(commandBuffer, v_indexBuffer_s, 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(commandBuffer, (uint32_t)(NumX * NumY * NumZ), 1, 0, 0, 0);
        }
        else
        {
            VkBuffer vertexBuffers[] = { v_volumeBuffer_s,v_xyzBuffer_s, v_raster};
            
            VkDeviceSize offsets[] = { 0, 0, 0};

            vkCmdPushConstants(commandBuffer,pipelineLayout,VK_SHADER_STAGE_GEOMETRY_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);

            vkCmdBindVertexBuffers(commandBuffer, 0, 3, vertexBuffers, offsets);

            vkCmdBindIndexBuffer(commandBuffer, v_indexBuffer_s, 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(commandBuffer, (uint32_t)(NumX * NumY * NumZ), 1, 0, 0, 0);
        }
    
    
    }

    void fillRenderingCommandBuffer_subpass1(VkCommandBuffer& commandBuffer) 
    {
        if(ImguiApp::thermal)
        {
            VkBuffer vertexBuffers[] = { v_volumeBuffer_t,v_xyzBuffer_s, v_raster};

            VkDeviceSize offsets[] = { 0, 0, 0};

            vkCmdPushConstants(commandBuffer,pipelineLayoutread,VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);

            vkCmdBindVertexBuffers(commandBuffer, 0, 3, vertexBuffers, offsets);

            vkCmdBindIndexBuffer(commandBuffer, v_indexBuffer_s, 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(commandBuffer, (uint32_t)(NumX * NumY * NumZ), 1, 0, 0, 0);
        }
        else
        {
            VkBuffer vertexBuffers[] = { v_volumeBuffer_s,v_xyzBuffer_s, v_raster};

            VkDeviceSize offsets[] = { 0, 0, 0};

            vkCmdPushConstants(commandBuffer,pipelineLayoutread,VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);

            vkCmdBindVertexBuffers(commandBuffer, 0, 3, vertexBuffers, offsets);

            vkCmdBindIndexBuffer(commandBuffer, v_indexBuffer_s, 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(commandBuffer, (uint32_t)(NumX * NumY * NumZ), 1, 0, 0, 0);
        }
    }


    void fillRenderingCommandBuffer_unit_lattice(VkCommandBuffer& commandBuffer)
    {
        VkBuffer vertexBuffers[] = { latticeonevol,v_xyzlatticeBuffer,v_raster};

        VkDeviceSize offsets[] = { 0, 0};

        vkCmdPushConstants(commandBuffer,pipelineLayout,VK_SHADER_STAGE_GEOMETRY_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);

        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);

        vkCmdBindIndexBuffer(commandBuffer, v_indexlatticeBuffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(commandBuffer, (uint32_t)(Nxu * Nyu * Nzu), 1, 0, 0, 0);
    }

    void fillRenderingCommandBuffer_unit_lattice_subpass1(VkCommandBuffer& commandBuffer)
    {
        VkBuffer vertexBuffers[] = { latticeonevol,v_xyzlatticeBuffer,v_raster};

        VkDeviceSize offsets[] = { 0, 0};

        vkCmdPushConstants(commandBuffer,pipelineLayoutread,VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);

        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);

        vkCmdBindIndexBuffer(commandBuffer, v_indexlatticeBuffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(commandBuffer, (uint32_t)(Nxu * Nyu * Nzu), 1, 0, 0, 0);
    }



   void fillRenderingCommandBuffer_spatial_lattice(VkCommandBuffer& commandBuffer)
    {
        VkBuffer vertexBuffers[] = { latticethreevol,v_xyzBufferthree, v_raster };

        VkDeviceSize offsets[] = { 0, 0};

        vkCmdPushConstants(commandBuffer,pipelineLayout,VK_SHADER_STAGE_GEOMETRY_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);

        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);

        vkCmdBindIndexBuffer(commandBuffer, v_indexBufferthree, 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(commandBuffer, (uint32_t)(NumX2 * NumY2 * NumZ2), 1, 0, 0, 0);
    }

    void fillRenderingCommandBuffer_spatial_lattice_subpass1(VkCommandBuffer& commandBuffer)
    {
        VkBuffer vertexBuffers[] = { latticethreevol,v_xyzBufferthree, v_raster};

        VkDeviceSize offsets[] = { 0, 0};

        vkCmdPushConstants(commandBuffer,pipelineLayoutread,VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);

        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);

        vkCmdBindIndexBuffer(commandBuffer, v_indexBufferthree, 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(commandBuffer, (uint32_t)(NumX2 * NumY2 * NumZ2), 1, 0, 0, 0);
    }

    void fillRenderingCommandBufferone(VkCommandBuffer& commandBuffer) {
   
        if((!show_topo_lattice) )
        {
            VkBuffer vertexBuffers[] = {v_pos_s, v_norm_s};

            VkDeviceSize offsets[] = { 0, 0 };

            vkCmdPushConstants(commandBuffer,pipelineLayout,VK_SHADER_STAGE_GEOMETRY_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);
            
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
            
            vkCmdDraw(commandBuffer, (uint32_t)(totalVerts), 1, 0, 0);

        }
        if(show_topo_lattice )
        {
            VkBuffer vertexBuffers[] = {vpos_two, vnorm_two};

            VkDeviceSize offsets[] = { 0, 0 };

            vkCmdPushConstants(commandBuffer,pipelineLayout,VK_SHADER_STAGE_GEOMETRY_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);
           
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);

            vkCmdDraw(commandBuffer, (uint32_t)(totalVertstwo), 1, 0, 0);
        }
    }


    void fillRenderingCommandBuffertwo(VkCommandBuffer& commandBuffer) {
   
        if(show_lattice_data)
        {
            VkBuffer vertexBuffers[] = {vpos_two, vnorm_two};

            VkDeviceSize offsets[] = { 0, 0 };

            vkCmdPushConstants(commandBuffer,pipelineLayout,VK_SHADER_STAGE_GEOMETRY_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);
            
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);

            vkCmdDraw(commandBuffer, (uint32_t)(totalVertstwo), 1, 0, 0);
        }
    }

    void fillRenderingCommandBufferthree(VkCommandBuffer& commandBuffer) {
   

        VkBuffer vertexBuffers[] = {vpos_two, vnorm_two};

        VkDeviceSize offsets[] = { 0, 0 };

        vkCmdPushConstants(commandBuffer,pipelineLayout,VK_SHADER_STAGE_GEOMETRY_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);
        
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
        
        vkCmdDraw(commandBuffer, (uint32_t)(totalVertstwo), 1, 0, 0);
    
    }



    void fillRenderingCommandBufferfour(VkCommandBuffer& commandBuffer) 
    {
   
    
        VkBuffer vertexBuffers[] = {vpos_one, vnorm_one};

        VkDeviceSize offsets[] = { 0, 0 };

        vkCmdPushConstants(commandBuffer,pipelineLayout,VK_SHADER_STAGE_GEOMETRY_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);
        
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
        
        vkCmdDraw(commandBuffer, (uint32_t)(totalVertsone), 1, 0, 0);
        
    }


    void fillRenderingCommandBufferone_subpass1(VkCommandBuffer& commandBuffer) 
    {

        if((!show_topo_lattice))
        {
            VkBuffer vertexBuffers[] = {v_pos_s, v_norm_s};
            
            VkDeviceSize offsets[] = { 0, 0 };
            
            vkCmdPushConstants(commandBuffer,pipelineLayoutread,VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);
            
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
            
            vkCmdDraw(commandBuffer, (uint32_t)(totalVerts), 1, 0, 0);
        }

        if(show_topo_lattice )
        {
            VkBuffer vertexBuffers[] = {vpos_two, vnorm_two};
            
            VkDeviceSize offsets[] = { 0, 0 };
            
            vkCmdPushConstants(commandBuffer,pipelineLayoutread,VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);
            
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
            
            vkCmdDraw(commandBuffer, (uint32_t)(totalVertstwo), 1, 0, 0);
        }

    }


    void fillRenderingCommandBuffertwo_subpass1(VkCommandBuffer& commandBuffer) 
    {

        if(show_lattice_data)
        {
            VkBuffer vertexBuffers[] = {vpos_two, vnorm_two};
            
            VkDeviceSize offsets[] = { 0, 0 };
            
            vkCmdPushConstants(commandBuffer,pipelineLayoutread,VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);
            
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
            
            vkCmdDraw(commandBuffer, (uint32_t)(totalVertstwo), 1, 0, 0);
        }

    }

    void fillRenderingCommandBufferthree_subpass1(VkCommandBuffer& commandBuffer) 
    {

        VkBuffer vertexBuffers[] = {vpos_two, vnorm_two};
        
        VkDeviceSize offsets[] = { 0, 0 };
        
        vkCmdPushConstants(commandBuffer,pipelineLayoutread,VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);
        
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
        
        vkCmdDraw(commandBuffer, (uint32_t)(totalVertstwo), 1, 0, 0);
        

    }

    void fillRenderingCommandBufferfour_subpass1(VkCommandBuffer& commandBuffer) 
    {

     
            VkBuffer vertexBuffers[] = {vpos_one, vnorm_one};
            
            VkDeviceSize offsets[] = { 0, 0 };
            
            vkCmdPushConstants(commandBuffer,pipelineLayoutread,VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT ,0,sizeof(ImguiApp::push_constants),&push_constants);
            
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
            
            vkCmdDraw(commandBuffer, (uint32_t)(totalVertsone), 1, 0, 0);
     

    }


    VkDeviceSize getUniformSize() const {
        return sizeof(UniformBufferObject);
    }

    void updateStorageBuffer(uint32_t imageIndex, bool load_selection, bool boundary_selection, bool delete_selection)
    {
        
        if(imageIndex == 0)
        {
            
            selectt.vertex_selection_two(d_cudastorageBuffers[0],d_cudastorageBuffers[1],NumX,NumY,NumZ, load_selection, boundary_selection, delete_selection);
        }
        else if(imageIndex == 1)
        {
          
            selectt.vertex_selection_two(d_cudastorageBuffers[1],d_cudastorageBuffers[0],NumX,NumY,NumZ, load_selection, boundary_selection, delete_selection);
        }
        getLastCudaError("Failed in updating storage buffer in cuda \n");
       
    }

    void update_inputevents()
    {

        ImVec2 mouse_pos = ImGui::GetMousePos();

        VulkanBaseApp::push_constants.mouse_x = mouse_pos.x;

        VulkanBaseApp::push_constants.mouse_y = mouse_pos.y;


        if((VulkanBaseApp::push_constants.pix_delta <= 100.0)  &&  (VulkanBaseApp::push_constants.pix_delta >= 1.0))
        {
            if(ImGui::IsKeyPressed(ImGuiKey_E))
            {
                select_val += 1.0;

                VulkanBaseApp::push_constants.pix_delta  = max(1.0,min(100.0,select_val));

            }
            else if(ImGui::IsKeyPressed(ImGuiKey_R))
            {
                select_val -= 1.0;

                VulkanBaseApp::push_constants.pix_delta = max(1.0,min(100.0,select_val));;
            }
        }
      
        
        if (ImGui::IsKeyReleased(ImGuiKey_Q) || ImGui::IsMouseReleased(ImGuiMouseButton_Right))
        {
            VulkanBaseApp::push_constants.support = 0 ;

            VulkanBaseApp::push_constants.mouse_click = 0 ;

            load_selection = false;
        }
        
        if (ImGui::IsKeyReleased(ImGuiKey_W) || ImGui::IsMouseReleased(ImGuiMouseButton_Left))
        {
            VulkanBaseApp::push_constants.support = 0 ;

            VulkanBaseApp::push_constants.mouse_click = 0 ;

            boundary_selection = false;
        }

        if (ImGui::IsKeyReleased(ImGuiKey_D))
        {
            VulkanBaseApp::push_constants.support = 0 ;

            VulkanBaseApp::push_constants.mouse_click = 0 ;

            delete_selection = false;


        }


        if(ImGui::IsKeyDown(ImGuiKey_W) && ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImguiApp::select_support_node)
        {
            VulkanBaseApp::push_constants.support = -1 ;

            VulkanBaseApp::push_constants.mouse_click = 2 ;

            boundary_selection = true;
        }
        else if(ImGui::IsKeyDown(ImGuiKey_Q) && ImGui::IsMouseDown(ImGuiMouseButton_Right) && ImguiApp::select_load_node)
        
        {
            VulkanBaseApp::push_constants.support = 1 ;

            VulkanBaseApp::push_constants.mouse_click = 1 ;

            load_selection = true;


        }

        if (ImGui::IsKeyDown(ImGuiKey_D))
        {
            VulkanBaseApp::push_constants.support = -1 ;

            VulkanBaseApp::push_constants.mouse_click = -1 ;

            delete_selection = true;

        }

    }


    void updateUniformBuffer(uint32_t imageIndex, bool shift) {
        {

            dist2 = ImguiApp::zoom_value * (MAX(NumZ,MAX(NumX,NumY))); 

            distone =ImguiApp::zoom_value *  (MAX(Nxu,MAX(Nyu,Nzu)));

            
            if(show_unit_lattice_data && (ImguiApp::lattice || ImguiApp::show_topo_lattice))
            {

                if ((VulkanBaseApp::shift))
                {
                    
                    if(view_top || view_bottom)
                    {
                        a_1 = (distone*1.2f)*sin(angle) + (Nxu-1)/2.0f;

                        b_1 = (distone*1.2f)*cos(angle) + (Nyu-1)/2.0f;

                        c_1 = (Nzu-1)/2.0f;
                    }
                    else
                    {
                        a_1 = (distone*1.2f)*sin(angle) + (Nxu-1)/2.0f;

                        b_1 = (Nyu-1)/2.0f;

                        c_1 = (distone*1.2f)*cos(angle) + (Nzu-1)/2.0f;
                    }
                    angle += 0.001;
                }

                else if(VulkanBaseApp::view_front)
                {
                    
                    a_1 = (Nxu-1)/2.0f;

                    b_1 = (Nyu-1)/2.0f;

                    c_1 = (1.0*distone*1.2f);
                
                }

                else if(VulkanBaseApp::view_back)
                {
                    
                    a_1 = (Nxu-1)/2.0f;

                    b_1 = (Nyu-1)/2.0f;

                    c_1 = (-1.0*distone*1.2f);
                }

                else if(VulkanBaseApp::view_top)
                {
                    
                    a_1 = (Nxu-1)/2.0f;

                    b_1 = (1.0*distone*2.5f);

                    c_1 = (Nzu-1)/2.0f;
                }

                else if(VulkanBaseApp::view_bottom)
                {
                    
                    a_1 = (Nxu-1)/2.0f;

                    b_1 = (-1.0*distone*2.5f);

                    c_1 = (Nzu-1)/2.0f;
                }

                else if(VulkanBaseApp::view_right)
                {
                    
                    a_1 = (1.0*distone*1.2f);

                    b_1 = (Nyu-1)/2.0f;

                    c_1 = (Nzu-1)/2.0f;
                }

                else if(VulkanBaseApp::view_left)
                {
                    
                    a_1 = (-1.0*distone*1.2f);

                    b_1 = (Nyu-1)/2.0f;

                    c_1 = (Nzu-1)/2.0f;
                }

                else
                {
                    a_1 = (distone*1.2f)*sin(angle) + (Nxu-1)/2.0f;

                    b_1 = (Nyu-1)/2.0f;

                    c_1 = (distone*1.2f)*cos(angle) + (Nzu-1)/2.0f;
                }
            }
            else
            {
                show_unit_lattice_data = false;

                if ((VulkanBaseApp::shift))
                {
                    
                    if(view_top || view_bottom)
                    {
                        a_1 = (NumX-1)/2.0f;

                        b_1 = (dist2*1.2f)*cos(angle) + (NumY-1)/2.0f;

                        c_1 = (dist2*1.2f)*sin(angle) + (NumZ-1)/2.0f;
                    }
                    else
                    {
                    
                        a_1 = (dist2*1.2f)*sin(angle) + (NumX-1)/2.0f;

                        b_1 = (NumY-1)/2.0f;

                        c_1 = (dist2*1.2f)*cos(angle) + (NumZ-1)/2.0f;
                    }
                    angle += 0.001;
                    
                }

                else if(VulkanBaseApp::view_front)
                {
                    
                    a_1 = (NumX-1)/2.0f;

                    b_1 = (NumY-1)/2.0f;

                    c_1 = (1.0*dist2*1.2f);
                
                }

                else if(VulkanBaseApp::view_back)
                {
                    
                    a_1 = (NumX-1)/2.0f;

                    b_1 = (NumY-1)/2.0f;

                    c_1 = (-1.0*dist2*1.2f);
                }

                else if(VulkanBaseApp::view_top)
                {
                    
                    a_1 = (NumX-1)/2.0f;

                    b_1 = (1.0*dist2*2.5f);

                    c_1 = (NumZ-1)/2.0f;

                
                }

                else if(VulkanBaseApp::view_bottom)
                {
                    
                    a_1 = (NumX-1)/2.0f;

                    b_1 = (-1.0*dist2*2.5f);

                    c_1 = (NumZ-1)/2.0f;
                }

                else if(VulkanBaseApp::view_right)
                {
                    
                    a_1 = (1.0*dist2*1.2f);

                    b_1 = (NumY-1)/2.0f;

                    c_1 = (NumZ-1)/2.0f;
                }

                else if(VulkanBaseApp::view_left)
                {
                    
                    a_1 = (-1.0*dist2*1.2f);

                    b_1 = (NumY-1)/2.0f;

                    c_1 = (NumZ-1)/2.0f;
                }

                else
                {
                    a_1 = (dist2*1.2f)*sin(angle) + (NumX-1)/2.0f;

                    b_1 = (NumY-1)/2.0f;

                    c_1 = (dist2*1.2f)*cos(angle) + (NumZ-1)/2.0f;
                }
            
            }

            mat4x4 view[1], proj[1];

            vec3 eye[1] = {{a_1,b_1,c_1}};

            VulkanBaseApp::push_constants.eyes[0] = a_1*3;
            VulkanBaseApp::push_constants.eyes[1] = b_1;
            VulkanBaseApp::push_constants.eyes[2] = c_1*3;
            VulkanBaseApp::push_constants.eyes[3] = 1.0;
            
            vec3 center[1];

            if(show_unit_lattice_data)
            {
                center[0][0] = float((Nxu-1)/2.0);
                center[0][1] = float((Nyu-1)/2.0);
                center[0][2] =  float((Nzu-1)/2.0);
            }
            else
            {
                center[0][0] = float((NumX-1)/2.0);
                center[0][1] = float((NumY-1)/2.0);
                center[0][2] =  float((NumZ-1)/2.0);
            }
            
            vec3 up[1] ;

            float r_l ;
            float t_b ;
            float n_f ;

            if(show_unit_lattice_data)
            {
                r_l = distone*0.7f;
                t_b = distone*0.7f;
                n_f = distone*3.5f;

                if(view_top || view_bottom)
                {
                    up[0][0] =  0.0f;
                    up[0][1] =  0.0f;
                    up[0][2] =  1.0f;
                  
                    mat4x4_ortho(proj[0],-(r_l),(r_l),-(t_b),(t_b),-n_f,n_f);
              
                }
                else
                {
                  
                    up[0][0] =  0.0f;
                    up[0][1] =  1.0f;
                    up[0][2] =  0.0f;

                    mat4x4_ortho(proj[0],-(r_l),(r_l),-(t_b),(t_b),-n_f,n_f);
                    proj[0][1][1] *= -1.0f; // Flip y axis
                }
            }
            else
            {
                r_l = dist2*0.7f;
                t_b = dist2*0.7f;
                n_f = dist2*3.5f;
            
                if(view_top || view_bottom)
                {
                    up[0][0] =  0.0f;
                    up[0][1] =  0.0f;
                    up[0][2] =  1.0f;

                    mat4x4_ortho(proj[0],-(r_l),(r_l),-(t_b),(t_b),-n_f,n_f);
                
                }
                else
                {
                    up[0][0] =  0.0f;
                    up[0][1] =  1.0f;
                    up[0][2] =  0.0f;

                    mat4x4_ortho(proj[0],-(r_l),(r_l),-(t_b),(t_b),-n_f,n_f);
                    proj[0][1][1] *= -1.0f; // Flip y axis
                }
            
            }
        

            mat4x4_look_at(view[0], eye[0], center[0], up[0]);
           
            mat4x4_mul(ubo.modelViewProj[0], proj[0], view[0]);
       
        }
      
        void *data;
        vkMapMemory(device, uniformMemory[imageIndex], 0, getUniformSize(), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformMemory[imageIndex]);
    }

    std::vector<const char *> getRequiredExtensions() const {
        std::vector<const char *> extensions;
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
        return extensions;
    }

    std::vector<const char *> getRequiredDeviceExtensions() const {
        std::vector<const char *> extensions;
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
   
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
 
        return extensions;
    }


    void drawFrame(bool shift)
    {
        
        cudaExternalSemaphoreWaitParams waitParams = {};
        waitParams.flags = 0;
        waitParams.params.fence.value = 0;

        cudaExternalSemaphoreSignalParams signalParams = {};
        signalParams.flags = 0;
        signalParams.params.fence.value = 0;
        // Signal vulkan to continue with the updated buffers
        checkCudaErrors(cudaSignalExternalSemaphoresAsync(&m_cudaSignalSemaphore, &signalParams, 1));

        VulkanBaseApp::drawFrame(shift);

        checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaWaitSemaphore, &waitParams, 1));
        
        
    }


    void init_textures()
    {
        isosurf.allocateTextures_s(&d_triTable, &d_numVertsTable);
    }

    void initMC_unitlattice()
    {
        gridSizeone = make_uint3(Nxu, Nyu, Nzu);

        gridSizeMaskone = make_uint3(Nxu-1, Nyu-1, Nzu-1);

        gridSizeShiftone = make_uint3(1,Nxu-1,(Nxu-1)*(Nyu-1));

        numVoxelsone = gridSizeMaskone.x*gridSizeMaskone.y*gridSizeMaskone.z;

        voxelSizeone = make_float3(1.0,1.0,1.0);

        gridcenterone = make_float3(0.0,0.0,0.0);

        unsigned int memSizeone = sizeof(uint) * numVoxelsone ;
    
        checkCudaErrors(cudaMalloc((void **) &d_voxelVertsone,            memSizeone));

        checkCudaErrors(cudaMalloc((void **) &d_voxelVertsScanone,        memSizeone));

        checkCudaErrors(cudaMalloc((void **) &d_voxelOccupiedone,         memSizeone));

        checkCudaErrors(cudaMalloc((void **) &d_voxelOccupiedScanone,     memSizeone));

        checkCudaErrors(cudaMalloc((void **) &d_compVoxelArrayone,   memSizeone));
    }


    void initMC()
    {
      
        gridSize = make_uint3(NumX, NumY, NumZ);

        gridSizeMask = make_uint3(NumX-1, NumY-1, NumZ-1);
      
        gridSizeShift = make_uint3(1,NumX-1,(NumX-1)*(NumY-1));
    
        numVoxels = gridSizeMask.x*gridSizeMask.y*gridSizeMask.z;
  
        voxelSize = make_float3(dx,dy,dz);
  
        gridcenter = make_float3(0.0,0.0,0.0);

        unsigned int memSize = sizeof(uint) * numVoxels ;
    
        checkCudaErrors(cudaMalloc((void **) &d_voxelVerts,            memSize));

        checkCudaErrors(cudaMalloc((void **) &d_voxelVertsScan,        memSize));

        checkCudaErrors(cudaMalloc((void **) &d_voxelOccupied,         memSize));

        checkCudaErrors(cudaMalloc((void **) &d_voxelOccupiedScan,     memSize));

        checkCudaErrors(cudaMalloc((void **) &d_compVoxelArray,   memSize));

    
    }

    void initMC_two()
    {
      
        gridSizetwo = make_uint3(NumX2, NumY2, NumZ2);

        gridSizeMasktwo = make_uint3(NumX2-1, NumY2-1, NumZ2-1);
      
        gridSizeShifttwo = make_uint3(1,NumX2-1,(NumX2-1)*(NumY2-1));
    
        numVoxelstwo = gridSizeMasktwo.x*gridSizeMasktwo.y*gridSizeMasktwo.z;
  
        voxelSizetwo = make_float3(dx2,dy2,dz2);

        gridcentertwo = make_float3(0.0,0.0,0.0);

        unsigned int memSizetwo = sizeof(uint) * numVoxelstwo ;
    
        checkCudaErrors(cudaMalloc((void **) &d_voxelVertstwo,            memSizetwo));

        checkCudaErrors(cudaMalloc((void **) &d_voxelVertsScantwo,        memSizetwo));

        checkCudaErrors(cudaMalloc((void **) &d_voxelOccupiedtwo,         memSizetwo));

        checkCudaErrors(cudaMalloc((void **) &d_voxelOccupiedScantwo,     memSizetwo));

        checkCudaErrors(cudaMalloc((void **) &d_compVoxelArraytwo,   memSizetwo));

    
    }

    void init_selection()
    {
        checkCudaErrors(cudaMalloc((void **)&d_selection, sizeof(REAL)*NumX * NumY*NumZ));

        checkCudaErrors(cudaMemset(d_selection, 0.0, sizeof(REAL)*NumX * NumY * NumZ));
    }
    
 

    void initVulkanCuda_semaphores() 
    {

        createExternalSemaphore(v_vkSignalSemaphore, getDefaultSemaphoreHandleType());
      
        createExternalSemaphore(v_vkWaitSemaphore, getDefaultSemaphoreHandleType());
      
        importCudaExternalSemaphore(m_cudaWaitSemaphore, v_vkSignalSemaphore, getDefaultSemaphoreHandleType());
      
        importCudaExternalSemaphore(m_cudaSignalSemaphore, v_vkWaitSemaphore, getDefaultSemaphoreHandleType());

    }


    void createStorageBuffers(size_t nVerts)
    {
        storageBuffers.resize(swapChainImages.size());

        storageMemory.resize(swapChainImages.size());

        d_cudastorageBuffers.resize(swapChainImages.size());

        d_cudastorageMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < storageBuffers.size(); i++) 
        {
          
            createExternalBuffer(nVerts * sizeof(float),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             getDefaultMemHandleType(),
                             storageBuffers[i], storageMemory[i]);
        
            importCudaExternalMemory((void **)&d_cudastorageBuffers[i], d_cudastorageMemory[i], storageMemory[i], nVerts * sizeof(float), getDefaultMemHandleType());
            getLastCudaError("Cuda External Memory - Storage Buffer \n");
        }
        
    }

    void erase_primitive_data()
    {
        checkCudaErrors(cudaMemset(d_postwo, 0.0, maxmemvertstwo * sizeof(*d_postwo)));
        checkCudaErrors(cudaMemset(d_normaltwo, 0.0, maxmemvertstwo * sizeof(*d_postwo)));
        checkCudaErrors(cudaMemset(d_volume_twice,0.0,sizeof(float)*NumX2 * NumY2*NumZ2));
        checkCudaErrors(cudaMemset(d_boundary, 0.0, (NumX2 *NumY2 * NumZ2) * sizeof(*d_boundary)));
        checkCudaErrors(cudaMemset(vol_one, 0.0, (NumX2 *NumY2 * NumZ2) * sizeof(*vol_one)));
        checkCudaErrors(cudaMemset(d_raster, 0.0, (NumX *NumY * NumZ) * sizeof(*d_raster)));
        checkCudaErrors(cudaMemset(d_latt_field, 0.0, NumX2*NumY2*NumZ2 * sizeof(*d_latt_field)));
        

    }

    void erase_topo_data()
    {
        checkCudaErrors(cudaMemset(d_postwo, 0.0, maxmemvertstwo * sizeof(*d_postwo)));
        checkCudaErrors(cudaMemset(d_normaltwo, 0.0, maxmemvertstwo * sizeof(*d_postwo)));
        checkCudaErrors(cudaMemset(d_volume_twice,0.0,sizeof(float)*NumX2 * NumY2*NumZ2));

        checkCudaErrors(cudaMemset(d_pos, 0.0, (maxmemverts) * sizeof(*d_pos)));
        checkCudaErrors(cudaMemset(d_normal, 0.0, (maxmemverts) * sizeof(*d_normal)));
        checkCudaErrors(cudaMemset(d_volume_s, 0.0, (NumX *NumY * NumZ) * sizeof(*d_volume_s)));
        checkCudaErrors(cudaMemset(d_volume_t, 0.0, (NumX *NumY * NumZ) * sizeof(*d_volume_t)));
        checkCudaErrors(cudaMemset(d_raster, 0.0, (NumX *NumY * NumZ) * sizeof(*d_raster)));

        checkCudaErrors(cudaMemset(d_selection, 0.0, sizeof(*d_selection)*NumX * NumY * NumZ));
        checkCudaErrors(cudaMemset(d_cudastorageBuffers[0], 0.0, (NumX *NumY * NumZ) * sizeof(*d_cudastorageBuffers[0])));
        checkCudaErrors(cudaMemset(d_cudastorageBuffers[1], 0.0, (NumX *NumY * NumZ) * sizeof(*d_cudastorageBuffers[1])));
     

    }

    void erase_lattice_data()
    {
        checkCudaErrors(cudaMemset(d_volumethree, 0.0, NumX2*NumY2*NumZ2 * sizeof(float)));
        checkCudaErrors(cudaMemset(d_volumethree_one, 0.0, NumX2*NumY2*NumZ2 * sizeof(float)));
        checkCudaErrors(cudaMemset(d_volumethree_two, 0.0, NumX2*NumY2*NumZ2 * sizeof(float)));
        checkCudaErrors(cudaMemset(d_postwo, 0.0, maxmemvertstwo * sizeof(*d_postwo)));
        checkCudaErrors(cudaMemset(d_normaltwo, 0.0, maxmemvertstwo * sizeof(*d_postwo)));
    }

    template <typename T>
    void fill_buffer_xyz(VkDevice device,VkBuffer buffer,const size_t nVerts,int Nx,int Ny, int Nz, float dx, float dy , float dz)
    {
        
        void *stagingBase;
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        VkDeviceSize stagingSz = nVerts * sizeof(T);
        createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

        vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);

         T *stagedVert = (T *)stagingBase;

        uint cou = 0;

        for (size_t z =0; z<Nz; z++){
            for (size_t y = 0; y < Ny; y++) {
                for (size_t x = 0; x < Nx; x++) {

                   
                        stagedVert[cou][0] = x * dx;
                        stagedVert[cou][1] = y * dy;
                        stagedVert[cou][2] = z * dz;
                    
                    cou++;
                }
            }
        }

        copyBuffer(buffer, stagingBuffer,0, nVerts * sizeof(T));
        vkUnmapMemory(device, stagingMemory);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
    }

    template <typename T>
    void fill_buffer_indices(VkDevice device,VkBuffer buffer,const size_t nVerts,int Nx,int Ny, int Nz)
    {
       
        void *stagingBase;
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        VkDeviceSize stagingSz = nVerts * sizeof(T);
        createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

        vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);
        T *indices = (T *)stagingBase;
        uint cou = 0;
        for (size_t z =0; z<Nz; z++){
            for (size_t y = 0; y < Ny; y++) {
                for (size_t x = 0; x < Nx; x++) {
    
                    indices[cou] = cou;
                    cou++;
                }
            }
        }

        copyBuffer(buffer, stagingBuffer,0, nVerts * sizeof(T));
        vkUnmapMemory(device, stagingMemory);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
    }

    template <typename T>
    void fill_buffer_val(VkDevice device,VkBuffer buffer,const size_t nVerts,int Nx,int Ny, int Nz,float val)
    {
       
        void *stagingBase;
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        VkDeviceSize stagingSz = nVerts * sizeof(T);
        createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

        vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);
        T *indices = (T *)stagingBase;
        uint cou = 0;
        for (size_t z =0; z<Nz; z++){
            for (size_t y = 0; y < Ny; y++) {
                for (size_t x = 0; x < Nx; x++) {
    
                    indices[cou] = val;
                    cou++;
                }
            }
        }

        copyBuffer(buffer, stagingBuffer,0, nVerts * sizeof(T));
        vkUnmapMemory(device, stagingMemory);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
    }

    template <typename T>
    void fill_buffer_vecfield(VkDevice device,VkBuffer buffer,const size_t nVerts,int Nx,int Ny, int Nz)
    {
        
        void *stagingBase;
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        VkDeviceSize stagingSz = nVerts * sizeof(T);
        createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

        vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);

         T *stagedVert = (T *)stagingBase;

        uint cou = 0;

        for (size_t z =0; z<Nz; z++){
            for (size_t y = 0; y < Ny; y++) {
                for (size_t x = 0; x < Nx; x++) {

                
                    stagedVert[cou].x = 0.0;
                    stagedVert[cou].y = 0.0;
                    stagedVert[cou].z = 0.0;

                    cou++;
                }
            }
        }

        copyBuffer(buffer, stagingBuffer,0, nVerts * sizeof(T));
        vkUnmapMemory(device, stagingMemory);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
    }


    template <typename T>
    void fill_buffer_posnorm(VkDevice device,VkBuffer buffer,const size_t maximumverts)
    {
        
        void *stagingBase;
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        VkDeviceSize stagingSz = maximumverts * sizeof(T);
        createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

        vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);

         T *stagedVert = (T *)stagingBase;

   

        for (size_t i = 0; i < maximumverts; i++){
           
            stagedVert[i].x = 0.0;
            stagedVert[i].y = 0.0;
            stagedVert[i].z = 0.0;
            stagedVert[i].w = 0.0;
           
        }

        copyBuffer(buffer, stagingBuffer,0, maximumverts* sizeof(T));
        vkUnmapMemory(device, stagingMemory);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
    }

    template<typename T> 
    void fill_storage_buffer(VkDevice device,std::vector<VkBuffer> buffers,const size_t nVerts,int Nx,int Ny, int Nz, float val)
    {
        void *stagingBase;
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        VkDeviceSize stagingSz = nVerts * sizeof(float);
        createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

        vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);
        
        uint cou = 0;
        float *heightval = (float *)stagingBase;

        for (size_t z =0; z<Nz; z++){
            for (size_t y = 0; y < Ny; y++) {
                for (size_t x = 0; x < Nx; x++) {
                    
                    heightval[cou] = val;
                    
                    cou++;
                }
            }
        }
        for(int i =0 ; i < swapChainImages.size(); i++)
        {
            copyBuffer(buffers[i], stagingBuffer,0, nVerts * sizeof(float));
        }

        vkUnmapMemory(device, stagingMemory);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
    }


    void vulkan_create_topo_buffers()
    {
    
        const size_t nVerts = (NumX)*(NumY)*(NumZ);
        const size_t nVolume = (NumX)*(NumY)*(NumZ);
        const size_t nInds =  (NumX)*(NumY)*(NumZ);

        uint max_dim = max(NumZ,(max(NumX,NumY)));

        size = NumX*NumY*NumZ;
        volsize = ((NumX)*(NumY)*(NumZ));
        maxmemverts = (NumX*NumY*(max_dim*4));

   

        createExternalBuffer(nVerts * sizeof(REAL3),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             getDefaultMemHandleType(),
                             v_structure_s, v_structureMemory_s);



        createExternalBuffer(nVolume * sizeof(REAL),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             getDefaultMemHandleType(),
                             v_volumeBuffer_s, v_volumeMemory_s);

        createExternalBuffer(nVolume * sizeof(REAL),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             getDefaultMemHandleType(),
                             v_volumeBuffer_t, v_volumeMemory_t);

        createExternalBuffer((nVerts) * sizeof(REAL),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        getDefaultMemHandleType(),
                        v_raster, v_rasterMemory);

        
        createExternalBuffer(maxmemverts * sizeof(float4),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT  | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             getDefaultMemHandleType(),
                             v_pos_s, v_posmemory_s);


        createExternalBuffer(maxmemverts * sizeof(float4),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             getDefaultMemHandleType(),
                             v_norm_s, v_normmemory_s);


        createBuffer(nVerts * sizeof(vec3),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     v_xyzBuffer_s, v_xyzMemory_s);


        createBuffer(nInds * sizeof(uint32_t),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     v_indexBuffer_s, v_indexMemory_s);

        importCudaExternalMemory((void **)&d_cudavulkan_s, m_cudavulkan_s, v_structureMemory_s, nVerts * sizeof(REAL3), getDefaultMemHandleType());
        
        importCudaExternalMemory((void **)&d_volume_s, m_cudaVertMem_s, v_volumeMemory_s, nVolume * sizeof(REAL), getDefaultMemHandleType());
        
        importCudaExternalMemory((void **)&d_volume_t, m_cudaVertMem_t, v_volumeMemory_t, nVolume * sizeof(REAL), getDefaultMemHandleType());

        importCudaExternalMemory((void **)&d_raster, m_cudarasterMem, v_rasterMemory, (nVerts) * sizeof(REAL), getDefaultMemHandleType());
        
        importCudaExternalMemory((void **)&d_pos, m_cudaPos_s,v_posmemory_s, maxmemverts * sizeof(*d_pos), getDefaultMemHandleType());
        
        importCudaExternalMemory((void **)&d_normal, m_cudaNorm_s,v_normmemory_s, maxmemverts * sizeof(*d_normal), getDefaultMemHandleType());
    

        fill_buffer_xyz<vec3>(device,v_xyzBuffer_s,nVerts,NumX,NumY,NumZ,dx,dy,dz);
        
        fill_buffer_indices<uint32_t>(device,v_indexBuffer_s,nInds,NumX,NumY,NumZ);
        
        fill_storage_buffer<float>(device,storageBuffers,nVerts,NumX,NumY,NumZ,0.0);
        ///////////////////////////////float ///////////////////////////////////////
       
        fill_buffer_val<float>(device,v_volumeBuffer_s,nVerts,NumX,NumY,NumZ,0.0f);

        fill_buffer_val<float>(device,v_volumeBuffer_t,nVerts,NumX,NumY,NumZ,0.0f);

        fill_buffer_val<float>(device,v_raster,nVerts,NumX,NumY,NumZ,0.0f);
        
        ////////////////////////////////////////////////////////////////////////////

        fill_buffer_vecfield<REAL3>(device,v_structure_s,nVerts,NumX,NumY,NumZ);
  
        fill_buffer_posnorm<float4>(device,v_pos_s,maxmemverts);

        fill_buffer_posnorm<float4>(device,v_norm_s,maxmemverts);

        ImguiApp::vulkan_buffer_created = true;

    }

    void vulkan_create_lattice_buffers()
    {
        size_t nVertsone = (Nxu)*(Nyu)*(Nzu);

        size_t nVertsthree = NumX2*NumY2*NumZ2;

        checkCudaErrors(cudaMalloc((void **)&d_volume_twice, sizeof(float)*NumX2 * NumY2*NumZ2));

        cudaMemset(d_volume_twice, 0.0, sizeof(float)*NumX2 * NumY2 * NumZ2);

        createExternalBuffer(nVertsone * sizeof(float),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             getDefaultMemHandleType(),
                             latticeonevol, latticeoneVolMemory);

        createExternalBuffer(nVertsthree *sizeof(float),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             getDefaultMemHandleType(),
                             latticethreevol, latticethreeVolMemory);

        createExternalBuffer(maxmemvertsone * sizeof(float4),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    getDefaultMemHandleType(),
                    vpos_one, vposMemory_one);

        createExternalBuffer(maxmemvertsone * sizeof(float4),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        getDefaultMemHandleType(),
                        vnorm_one, vnormMemory_one);

        createExternalBuffer(maxmemvertstwo * sizeof(float4),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        getDefaultMemHandleType(),
                        vpos_two, vposMemory_two);

        createExternalBuffer(maxmemvertstwo * sizeof(float4),
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            getDefaultMemHandleType(),
                            vnorm_two, vnormMemory_two);

        createBuffer(nVertsone * sizeof(vec3),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     v_xyzlatticeBuffer, v_xyzlatticeMemory);

        createBuffer(nVertsone * sizeof(uint32_t),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     v_indexlatticeBuffer, v_indexlatticeMemory);

        createBuffer(nVertsthree * sizeof(vec3),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     v_xyzBufferthree, v_xyzMemorythree);

        createBuffer(nVertsthree * sizeof(uint32_t),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     v_indexBufferthree, v_indexMemorythree);

        importCudaExternalMemory((void **)&d_volumeone, m_cudaVertMemone, latticeoneVolMemory, nVertsone * sizeof(float), getDefaultMemHandleType());
        
        importCudaExternalMemory((void **)&d_volumethree, m_cudaVertMemthree, latticethreeVolMemory, nVertsthree * sizeof(float), getDefaultMemHandleType());  
        
        importCudaExternalMemory((void **)&d_postwo, m_cudaPostwo,vposMemory_two, maxmemvertstwo * sizeof(*d_postwo), getDefaultMemHandleType());
    
        importCudaExternalMemory((void **)&d_normaltwo, m_cudaNormtwo,vnormMemory_two, maxmemvertstwo * sizeof(*d_normaltwo), getDefaultMemHandleType());
       
        importCudaExternalMemory((void **)&d_posone, m_cudaPosone,vposMemory_one, maxmemvertsone * sizeof(*d_posone), getDefaultMemHandleType());
    
        importCudaExternalMemory((void **)&d_normalone, m_cudaNormone,vnormMemory_one, maxmemvertsone * sizeof(*d_normalone), getDefaultMemHandleType());
       
        fill_buffer_val<float>(device,latticethreevol,nVertsthree,NumX2,NumY2,NumZ2,0.0f);
       
        fill_buffer_val<float>(device,latticeonevol,nVertsone,Nxu,Nyu,Nzu,0.0f);

        fill_buffer_posnorm<float4>(device,vpos_two,maxmemvertstwo);

        fill_buffer_posnorm<float4>(device,vnorm_two,maxmemvertstwo);
      
        fill_buffer_posnorm<float4>(device,vpos_one,maxmemvertsone);

        fill_buffer_posnorm<float4>(device,vnorm_one,maxmemvertsone);

        fill_buffer_xyz<vec3>(device,v_xyzlatticeBuffer,nVertsone,Nxu,Nyu,Nzu, 1.0, 1.0, 1.0);
       
        fill_buffer_indices<uint32_t>(device,v_indexlatticeBuffer,nVertsone,Nxu,Nyu,Nzu);
       
        fill_buffer_xyz<vec3>(device,v_xyzBufferthree,nVertsthree,NumX2,NumY2,NumZ2,dx2,dy2,dz2);
       
        fill_buffer_indices<uint32_t>(device,v_indexBufferthree,nVertsthree,NumX2,NumY2,NumZ2);

        ImguiApp::lattice_buffer_created = true;

    }


   

    void update_topo_grid_data()
    {
        NumX = NumY = NumZ = ImguiApp::grid_value;
        
        dx = dy = dz = 1.0;
        
        dist2 = (MAX(NumZ,MAX(NumX,NumY)));
        
        maxmemverts = max((NumX*NumY*NumZ*4),300000);

        NumX2 = NumY2 = NumZ2 = 2*(NumX);
        
        dx2 = 0.5 *dx;
        
        dy2 = 0.5 *dy;
        
        dz2 = 0.5 *dz;
        
        maxmemvertstwo = max((NumX2*NumY2*NumZ2*4),300000);
        
        maxmemvertsone = max((Nxu*Nyu*Nzu*4),300000);
        
        dist3 = (MAX(NumZ2,MAX(NumX2,NumY2)));
    
        size = NumX*NumY*NumZ;
        
        sizeone = Nxu*Nyu*Nzu;
        
        size2 = NumX2*NumY2*NumZ2;
    }

 

    int topoinit_struct(){

        OptIter = 0;

        REAL RefStiff_s[3][8][3][8];
        
        MakeRefStiff_s(RefStiff_s);
        
        register REAL EleStiff_s[24][24];
        
        //local stiffness matrix develops here 
        MakeEleStiffness_s(EleStiff_s, RefStiff_s,Topopt_val::Youngs_Modulus,Topopt_val::poisson);
        
        structure.InitGPU(EleStiff_s);

        getLastCudaError("Initialisation of Stiffness matrix in GPU failed");

        checkCudaErrors(cudaMalloc((void **)&d_us, sizeof(REAL3)*NumX * NumY*NumZ));

        checkCudaErrors(cudaMalloc((void **)&d_den, sizeof(REAL)*NumX * NumY*NumZ));

        checkCudaErrors(cudaMalloc((void **)&d_grads, sizeof(REAL)*NumX * NumY*NumZ));

        
        pitch_bytes = sizeof(REAL3)* NumX;

        grad_pitch_bytes = sizeof(REAL)* NumX;

        cudaMemset(d_grads, 0.0, sizeof(REAL)*NumX * NumY * NumZ);
        
        cudaMemset(d_den, 0.0, sizeof(REAL)*NumX * NumY * NumZ);
        
        cudaMemset(d_us, 0.0, sizeof(REAL3)*NumX * NumY * NumZ);
        
        opt_kernel.init_d_den(d_den,Topopt_val::VolumeFraction,NumX * NumY * NumZ);
    
        gettimeofday(&t1, 0);
        
        structure.GPUCG(d_us,d_den,d_selection, Topopt_val::iter, 0, Topopt_val::EndRes, FinalIter_s, FinalRes_s,Topopt_val::pexp);
        
        gettimeofday(&t2, 0);

        tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

        printf("Time to generate GPUCG:  %3.1f ms \n\n", tottime);

 
        cout << "CG-Residual_s: " << FinalRes_s << " Final iter "<<FinalIter_s<<" \n\n"<< endl;

        gettimeofday(&t1, 0);

        structure.GPUCompGrad(d_us,d_den, d_grads, Obj_s, Vol_s, pitch_bytes, grad_pitch_bytes, Topopt_val::pexp);

        cout << "After Iter "<<OptIter<<": Compliance = " << Obj_s  << " Vol = "<<Vol_s<<"\n"<< endl;

        gettimeofday(&t2, 0);

        tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

        printf("Time to generate GPUCompGrad:  %3.1f ms \n\n", tottime);

        printf("Initialisation Completed Successfully \n");
       
        return 0;
    }

    int topoinit_thermal()
    {
        
        OptIter = 0;
        
        REAL RefStiff_t[8][8];

        MakeRefStiff_t(RefStiff_t);

        REAL EleStiff_t[8][8];

        MakeEleStiff_t(Topopt_val::conductivity,EleStiff_t, RefStiff_t);

        thermal.InitGPU(EleStiff_t);

        checkCudaErrors(cudaMalloc((void **)&d_us, sizeof(REAL3)*NumX * NumY*NumZ));

        checkCudaErrors(cudaMalloc((void **)&d_den, sizeof(REAL)*NumX * NumY*NumZ));

        checkCudaErrors(cudaMalloc((void **)&d_grads, sizeof(REAL)*NumX * NumY*NumZ));

        
        pitch_bytes = sizeof(REAL3)* NumX;

        grad_pitch_bytes = sizeof(REAL)* NumX;

        cudaMemset(d_grads, 0.0, sizeof(REAL)*NumX * NumY * NumZ);
        
        cudaMemset(d_den, 0.0, sizeof(REAL)*NumX * NumY * NumZ);
        
        cudaMemset(d_us, 0.0, sizeof(REAL3)*NumX * NumY * NumZ);
     
        opt_kernel.init_d_den(d_den,Topopt_val::VolumeFraction,NumX * NumY * NumZ);
        
        getLastCudaError("Initialisation of density buffer failed ");
 
        gettimeofday(&t1, 0);
       
        thermal.GPUCG(d_us,d_den,d_selection, Topopt_val::iter, 0, Topopt_val::EndRes, FinalIter_t,FinalRes_t, Topopt_val::pexp);
        
        getLastCudaError("Thermal GPUCG failed");
        
        gettimeofday(&t2, 0);

        tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

        printf("Time to generate GPUCG:  %3.1f ms \n\n", tottime);
 
        cout << "CG-Residual_t: " << FinalRes_t << "  Final iter "<<FinalIter_t<< endl;

        gettimeofday(&t1, 0);
      
        thermal.GPUCompGrad(d_us,d_den, d_grads, Obj_s, Vol_s, pitch_bytes, grad_pitch_bytes,Topopt_val::pexp);
        
        gettimeofday(&t2, 0);

        tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

        printf("Time to generate GPUCompGrad:  %3.1f ms \n\n", tottime);

        printf("Initialisation Completed Successfully \n");
       
        return 0;

    }

    int toprun_struct()
    {
       
        while(OptIter < Topopt_val::MaxOptIter)
        {
            
            cout<<"OptIter "<<OptIter<<endl;
        
            gettimeofday(&t1, 0);
            const int grad_pitchX = grad_pitch_bytes/sizeof(REAL);
            
            opt_kernel.GPUMeshFilter(d_us,d_den,Topopt_val::FilterRadius,d_grads,grad_pitchX,NumX,NumY,NumZ);

            getLastCudaError("GPU MeshFilter  failed");

            gettimeofday(&t2, 0);

            tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

            printf("Time to generate MeshFIlter:  %3.1f ms \n\n", tottime);

            gettimeofday(&t1, 0);

            opt_kernel.Update_s_one(d_us,d_den,Topopt_val::VolumeFraction,Topopt_val::MinDens,d_grads,d_volume_s,grad_pitchX,NumX,NumY,NumZ);

            gettimeofday(&t2, 0);

            tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

            printf("Time to generate Update_density:  %3.1f ms \n\n", tottime);

            gettimeofday(&t1, 0);
            
            isosurf.computeIsosurface_2(d_pos,d_normal,0.31,numVoxels,d_voxelVerts,d_voxelVertsScan,
            d_voxelOccupied,d_voxelOccupiedScan,gridSize,gridSizeShift,gridSizeMask,voxelSize,gridcenter,
            &activeVoxels,&totalVerts,d_compVoxelArray,maxmemverts,d_volume_s,0.31);

            gettimeofday(&t2, 0);

            tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

            printf("Time to generate compute_isosurface:  %3.1f ms \n\n", tottime);

            cudaExternalSemaphoreWaitParams waitParams = {};
            waitParams.flags = 0;
            waitParams.params.fence.value = 0;

            cudaExternalSemaphoreSignalParams signalParams = {};
            signalParams.flags = 0;
            signalParams.params.fence.value = 0;
                            
            checkCudaErrors(cudaSignalExternalSemaphoresAsync(&m_cudaSignalSemaphore, &signalParams, 1));
        
            VulkanBaseApp::drawFrame(shift);
            
            checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaWaitSemaphore, &waitParams, 1));

            gettimeofday(&t1, 0);
            
            structure.GPUCG(d_us,d_den,d_selection, Topopt_val::iter, OptIter, Topopt_val::EndRes, FinalIter_s, FinalRes_s,Topopt_val::pexp);
            
            gettimeofday(&t2, 0);

            tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

            printf("Time to generate GPUCG:  %3.1f ms \n\n", tottime);  
            
            cout << "CG-Residual_s: " << FinalRes_s << "  Final iter "<<FinalIter_s<<"\n\n"<< endl;

            Obj_old_s = Obj_s;

            gettimeofday(&t1, 0);

            structure.GPUCompGrad(d_us,d_den, d_grads, Obj_s, Vol_s, pitch_bytes, grad_pitch_bytes,Topopt_val::pexp);
            
            getLastCudaError("GPU CompGrad  failed");

            cout << "After Iter "<<OptIter<<": Compliance = " << Obj_s  << " Vol = "<<Vol_s<<"\n"<< endl;

            gettimeofday(&t2, 0);

            tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

            printf("Time to generate GPUCompGrad:  %3.1f ms \n\n", tottime);

            OptIter++;
        }
        
        return 0;
    }

    int toprun_thermal()
    {
       
        VulkanBaseApp::shift = false;

        while(OptIter < Topopt_val::MaxOptIter)
        {

            VulkanBaseApp::shift = false;

            cout<<"OptIter "<<OptIter<<endl;

            gettimeofday(&t1, 0);

            const int grad_pitchX = grad_pitch_bytes/sizeof(REAL);
            
            opt_kernel.GPUMeshFilter(d_us,d_den,Topopt_val::FilterRadius,d_grads,grad_pitchX,NumX,NumY,NumZ);

            getLastCudaError("GPU MeshFilter  failed");
            
            gettimeofday(&t2, 0);

            tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

            printf("Time to generate MeshFIlter:  %3.1f ms \n\n", tottime);

            ///////////////////////////////////////////////////////////////////////////////
            gettimeofday(&t1, 0);

            opt_kernel.Update_s_one(d_us,d_den,Topopt_val::VolumeFraction,Topopt_val::MinDens,d_grads,d_volume_t,grad_pitchX,NumX,NumY,NumZ);

            gettimeofday(&t2, 0);

            tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

            printf("Time to generate Update_density:  %3.1f ms \n\n", tottime);

            gettimeofday(&t1, 0);
    
            isosurf.computeIsosurface_2(d_pos,d_normal,0.3,numVoxels,d_voxelVerts,d_voxelVertsScan,
            d_voxelOccupied,d_voxelOccupiedScan,gridSize,gridSizeShift,gridSizeMask,voxelSize,gridcenter,
            &activeVoxels,&totalVerts,d_compVoxelArray,maxmemverts,d_volume_t,0.3);

            gettimeofday(&t2, 0);

            tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

            printf("Time to generate compute_isosurface:  %3.1f ms \n\n", tottime);
        
            cudaExternalSemaphoreWaitParams waitParams = {};
            waitParams.flags = 0;
            waitParams.params.fence.value = 0;

            cudaExternalSemaphoreSignalParams signalParams = {};
            signalParams.flags = 0;
            signalParams.params.fence.value = 0;
                            
            checkCudaErrors(cudaSignalExternalSemaphoresAsync(&m_cudaSignalSemaphore, &signalParams, 1));
        
            VulkanBaseApp::drawFrame(shift);

            checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaWaitSemaphore, &waitParams, 1));

            gettimeofday(&t1, 0);
            
            thermal.GPUCG(d_us,d_den, d_selection, Topopt_val::iter, OptIter, Topopt_val::EndRes, FinalIter_s, FinalRes_s,Topopt_val::pexp);
            
            gettimeofday(&t2, 0);

            tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

            printf("Time to generate GPUCG:  %3.1f ms \n\n", tottime);

            cout << "CG-Residual_s: " << FinalRes_s << "  Final iter \n\n"<<FinalIter_s<< endl;
   
            Obj_old_s = Obj_s;

            gettimeofday(&t1, 0);

            thermal.GPUCompGrad(d_us,d_den, d_grads, Obj_s, Vol_s, pitch_bytes, grad_pitch_bytes,Topopt_val::pexp);

            gettimeofday(&t2, 0);

            tottime = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

            printf("Time to generate GPUCompGrad:  %3.1f ms \n\n", tottime);

            OptIter++;
        }
        
        return 0;
    }


    void init_Boundary()
    {
        checkCudaErrors(cudaMalloc((void **)&d_boundary, sizeof(float) * (NumX2*NumY2*NumZ2)));

        checkCudaErrors(cudaMalloc((void **)&d_latt_field, sizeof(float) * (NumX2*NumY2*NumZ2)));

        checkCudaErrors(cudaMalloc((void **)&vol_one, sizeof(*vol_one) * (NumX2*NumY2*NumZ2)));

        cudaMemset(d_latt_field,0,sizeof(float) * size2);

        cudaMemset(d_boundary,0,sizeof(float) * size2);

        cudaMemset(vol_one,0,sizeof(*vol_one) *size2 );

        ImguiApp::boundary_buffers = true;

    }


    

    void show_model()
    {
          
        if(ImguiApp::retain)
        {
            isosurf.copy_parameter(gridSize,d_voxelVertstwo,d_voxelVertsScantwo,0.0,gridSizetwo,gridSizeShifttwo,gridSizeMasktwo,voxelSizetwo,gridcentertwo,numVoxelstwo,&activeVoxelstwo,d_compVoxelArraytwo,vol_one,d_boundary,d_volumethree,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo,
            &totalVertstwo, obj_union, obj_diff, obj_intersect);
            
            checkCudaErrors(cudaMemset(d_raster, 0.0, (size) * sizeof(*d_raster)));
            selectt.raster_update(0.0,0.0,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo,d_raster,vol_one,d_boundary,d_volumethree,d_latt_field,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic,NumX,NumY,NumZ);
            isosurf.computeIsosurface(d_raster, gridSize,d_postwo,d_normaltwo,0.0,numVoxelstwo,d_voxelVertstwo,d_voxelVertsScantwo,
            d_voxelOccupiedtwo,d_voxelOccupiedScantwo,gridSizetwo,gridSizeShifttwo,gridSizeMasktwo,voxelSizetwo,gridcentertwo,
            &activeVoxelstwo,&totalVertstwo,d_compVoxelArraytwo,maxmemvertstwo,vol_one,d_boundary,d_volume_twice,d_volumethree,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo, obj_union, obj_diff, obj_intersect,ImguiApp::primitives,ImguiApp::structural,ImguiApp::lattice,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic);
            

            ImguiApp::retain = false;
            
        }


        if(ImguiApp::calculate)
        {
            bool compute_iso = false;

             if(cylind_selected || cylind_disc_selected)
            {
                
                distance_from_line(d_boundary, ImguiApp::center,ImguiApp::axis, 2 * ImguiApp::radius, 2 * ImguiApp::thickness_radial, 2 * ImguiApp::thickness_axial,NumX2,NumY2,NumZ2,cylind_disc_selected);
                
                compute_iso = true;
            }

            if(sphere_selected || sphere_shell_selected)
            {
                sphere_with_center(d_boundary,center, 2 * ImguiApp::sphere_radius, 2 * ImguiApp::sphere_thickness,NumX2,NumY2,NumZ2, ImguiApp::sphere_shell_selected);

                compute_iso = true;
            }
            if(cuboid_selected)
            {
                cuboid(d_boundary,center,angles, 2 * ImguiApp::cuboid_x, 2 * ImguiApp::cuboid_y, 2 * ImguiApp::cuboid_z,NumX2,NumY2,NumZ2);
                
                compute_iso = true;
            }
            if(cuboid_shell_selected)
            {
                cuboid_shell(d_boundary,center,angles, 2 * ImguiApp::cuboid_x, 2 * ImguiApp::cuboid_y, 2 * ImguiApp::cuboid_z, 2 *ImguiApp::cu_sh_thick,NumX2,NumY2,NumZ2);
            
                compute_iso = true;
            }
            if(torus_selected)
            {
                torus_with_center(d_boundary,center,angles,  2 * ImguiApp::torus_radius, 2 * ImguiApp::torus_circle_radius,NumX2,NumY2,NumZ2);
            
                compute_iso = true;
            }
            if(cone_selected)
            {
                
                cone_with_base_radius_height(d_boundary,center,angles, 2 * ImguiApp::base_radius, 2 * ImguiApp::cone_height,NumX2,NumY2,NumZ2);
            
                compute_iso = true;
            }
            
        

            if(compute_iso)
            {
                
                if(ImguiApp::lattice_fixed || ImguiApp::lattice_dynamic)
                {   
                    cudaMemcpy(d_volumethree,d_latt_field,size2 * sizeof(float),cudaMemcpyDeviceToDevice);

                    lattice.primitive_field(vol_one,d_boundary,d_volumethree,0.0,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic,NumX2,NumY2,NumZ2);
                }
                
                
                checkCudaErrors(cudaMemset(d_raster, 0.0, (size) * sizeof(*d_raster)));
                selectt.raster_update(0.0,0.0,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo,d_raster,vol_one,d_boundary,d_volumethree,d_latt_field,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic,NumX,NumY,NumZ);
                isosurf.computeIsosurface(d_raster, gridSize,d_postwo,d_normaltwo,0.0,numVoxelstwo,d_voxelVertstwo,d_voxelVertsScantwo,
                d_voxelOccupiedtwo,d_voxelOccupiedScantwo,gridSizetwo,gridSizeShifttwo,gridSizeMasktwo,voxelSizetwo,gridcentertwo,
                &activeVoxelstwo,&totalVertstwo,d_compVoxelArraytwo,maxmemvertstwo,vol_one,d_boundary,d_volume_twice,d_volumethree,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo, obj_union, obj_diff, obj_intersect,ImguiApp::primitives,ImguiApp::structural,ImguiApp::lattice,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic);
                
            }

        }

    }

    void update_lattice_type()
    {
          if(ImguiApp::lattice_type_index == 0)
        {
            latticetype_one = 'n';
        }
        else if (ImguiApp::lattice_type_index == 1)
        {
            latticetype_one = 'b';
        }

        else if (ImguiApp::lattice_type_index == 2)
        {
            latticetype_one = 'r';
        }

        else if (ImguiApp::lattice_type_index == 3)
        {
            latticetype_one = 's';
        }

        if(ImguiApp::lattice_size_index == 0)
        {
            latticetype_two = 'u';
        }
        else if (ImguiApp::lattice_size_index == 1)
        {
            latticetype_two = 'v';
        }

        
    }


    int fft_init()
    {
        checkCudaErrors(cudaMalloc((void **)&fft_data, sizeof(float2) * (((Nxu/2)+1) * Nyu * Nzu )));

        checkCudaErrors(cudaMalloc((void **)&fft_data_compute, sizeof(float2) * (((Nxu/2)+1) * Nyu * Nzu )));

        checkCudaErrors(cudaMalloc((void **)&fft_data_compute_fill, sizeof(float2) * (Nxu * Nyu * Nzu)));

        cudaMemset(fft_data,0,sizeof(float2) * (Nxu * Nyu * ((Nzu/2)+1)));

        cudaMemset(fft_data_compute,0,sizeof(float2) * (Nxu * Nyu * ((Nzu/2)+1)));

        cudaMemset(fft_data_compute_fill,0,sizeof(float2) * (Nxu * Nyu * Nzu));

        return 0;

    }


    int lattice_init(){

        checkCudaErrors(cufftPlan3d(&planr2c, Nxu, Nyu, Nzu, CUFFT_R2C));

        checkCudaErrors(cufftPlan3d(&planc2r, Nxu, Nyu, Nzu, CUFFT_C2R));

        checkCudaErrors(cufftPlan3d(&planc2c, Nxu, Nyu, Nzu, CUFFT_C2C));
        
        checkCudaErrors(cudaMalloc((void **)&fft_gratings, sizeof(float) * (Nxu * Nyu * Nzu)));

        checkCudaErrors(cudaMalloc((void **)&lattice_data, sizeof(float2) * (indi_range*indi_range*indi_range)));

        checkCudaErrors(cudaMalloc((void **)&d_volumeone_one, sizeof(float) * (Nxu*Nyu*Nzu)));
        
        cudaMemset(fft_gratings,0,sizeof(float) * (Nxu * Nyu * Nzu));

        cudaMemset(lattice_data,0,sizeof(float2) * (indi_range*indi_range*indi_range));

        cudaMemset(d_volumeone_one,0,sizeof(float) * (Nxu*Nyu*Nzu));

  
        return 0;
    }

     void init_svl()
    {
        
        checkCudaErrors(cudaMalloc((void **)&d_volumethree_one, sizeof(float) * (NumX2*NumY2*NumZ2)));

        cudaMemset(d_volumethree_one,0,sizeof(float) * (NumX2*NumY2*NumZ2));

        checkCudaErrors(cudaMalloc((void **)&d_volumethree_two, sizeof(float) * (NumX2*NumY2*NumZ2)));

        cudaMemset(d_volumethree_two,0,sizeof(float) * (NumX2*NumY2*NumZ2));

        checkCudaErrors(cudaMalloc((void **)&d_phi, sizeof(float) * (NumX*NumY*NumZ)));

        printf("NumX %d NumY %d NumZ %d \n",NumX,NumY,NumZ);

        getLastCudaError("phi allocation failed");
       
        checkCudaErrors(cudaMalloc((void **)&d_ga, sizeof(float2) * (NumX2*NumY2*NumZ2)));

        checkCudaErrors(cudaMalloc((void **)&d_svl, sizeof(float) * (NumX2*NumY2*NumZ2)));

        checkCudaErrors(cudaMalloc((void **)&d_theta, sizeof(float) * (NumX*NumY*NumZ)));

        checkCudaErrors(cudaMalloc((void **)&d_period, sizeof(float) * (NumX*NumY*NumZ)));

        cudaMemset(d_phi,0,sizeof(float) * size);

        cudaMemset(d_svl,0,sizeof(float) * size2);
     
        size_t d_width = NumX, d_height = NumY, d_depth = NumZ;

        extend = make_cudaExtent(d_width*sizeof(float),d_height, d_depth);

        printf("extend x %lu y %lu z %lu \n",extend.width,extend.height,extend.depth);

        cudaMalloc3D(&devPitchedPtr, extend);

        getLastCudaError("cudaMalloc3D failed");

        tPitch = devPitchedPtr.pitch;

        slicepitch = tPitch*d_height;
    
        cudaMemcpy3DParms params ={0};

        params.srcPtr = make_cudaPitchedPtr(d_phi,d_width*sizeof(float),d_width,d_height);

        params.dstPtr = devPitchedPtr;

        params.extent = extend;

        params.kind = cudaMemcpyDeviceToDevice;

        cudaMemcpy3D( &params );

        getLastCudaError("cudaMemcpy3D failed ");

        lattice.setupTexture(NumX,NumY,NumZ);

        getLastCudaError("set up texture failed ");

        printf("Lattice svl Initialisation Completed Successfully \n\n");

        ImguiApp::svl_data = true;
    }

    int unit_lattice()
    {

        fftlattice.create_lattice(d_volumeone,Nxu,Nyu,Nzu,sizeone,lattice_index_type);
        
        checkCudaErrors(cudaMemcpy2D(fft_data,(Nxu+1)*sizeof(float), d_volumeone,(Nxu*sizeof(float)), (Nxu*sizeof(float)),Nyu*Nzu, cudaMemcpyDeviceToDevice));
        
        getLastCudaError("Error in 'unit_lattice function' \n ");

        fftlattice.fft_func(fft_data);

        checkCudaErrors(cudaMemcpy(fft_data_compute, fft_data, (((Nxu/2)+1) * Nyu * Nzu) * sizeof(float2), cudaMemcpyDeviceToDevice));

        fftlattice.fft_scalar(fft_data_compute,sizeone,(((Nxu/2)+1) * Nyu * Nzu));

        uint mid_index = floor((Nxu*Nyu*Nzu)/2.0);

        fftlattice.fft_fill(fft_data_compute,fft_data_compute_fill,Nxu,Nyu,Nzu,mid_index);

        fftlattice.ifft_func(fft_data);
        
        float2 *h_dat;
        
        h_dat = (float2 *)malloc((Nxu*Nyu*Nzu) * sizeof(float2));

        checkCudaErrors(cudaMemcpy(h_dat, fft_data_compute_fill, (Nxu * Nyu * Nzu) * sizeof(float2), cudaMemcpyDeviceToHost));

        float2 *h_lattice_dat;

        h_lattice_dat = (float2 *)malloc((indi_range*indi_range*indi_range) * sizeof(float2));

        int l_count = 0;

        int midd1 = floor((Nxu*Nyu*Nzu)/2.0);
        
        int midd2 = floor((indi_range*indi_range*indi_range)/2.0);

        for(int k = -range_st;k <(range_st+1);k++)
        {
            for(int j = -range_st;j <(range_st+1);j++)
            {
                for(int i = -range_st;i <(range_st+1);i++)
                {
                    
                    if((i == 0) && (j == 0) && (k == 0))
                    {
                        h_lattice_dat[midd2] = h_dat[0];
                    
                        l_count++;

                        continue;
                    }
                    
                    int u,s,t;

                    int mid_idx = floor(Nxu/2.0);

                    int mid_idy = floor(Nyu/2.0);

                    int mid_idz = floor(Nzu/2.0);

                    u = mid_idx + i;

                    s = mid_idy + j;

                    t = mid_idz + k;

                    if( u == mid_idx)
                    {
                        u = 0;
                    }
                    else if( u > mid_idx)
                    {
                        u -= mid_idx;
                    }
                    else
                    {
                        u += (mid_idx+1);
                    }

                    if( s == mid_idy)
                    {
                        s = 0;
                    }
                    else if( s > mid_idy)
                    {
                        s -= mid_idy;
                    }
                    else
                    {
                        s += (mid_idy + 1);
                    }

                    if( t == mid_idz)
                    {
                        t = 0;
                    }
                    else if( t > (mid_idz))
                    {
                        t -= mid_idz;
                    }
                    else
                    {
                        t += (mid_idz + 1);
                    }
        
                    int indx2 = u + s * Nxu + t * (Nxu*Nyu);

                    h_lattice_dat[l_count] = h_dat[indx2];
                 
                    l_count++;
                
                }
                
            }

        }   

  

        checkCudaErrors(cudaMemcpy2D(fft_gratings,Nxu * sizeof(float), fft_data,(((Nxu/2)+1)*2)*sizeof(float), (Nxu*sizeof(float)),Nyu*Nzu, cudaMemcpyDeviceToDevice));
   
        lattice.GPU_buffer_normalise_buffer(fft_gratings,fft_gratings,Nxu*Nyu*Nzu);
     
        lattice.GPU_buffer_normalise_four(fft_gratings,d_volumeone_one,fft_gratings,Nxu*Nyu*Nzu,Nxu,Nyu,Nzu,ImguiApp::bound_isoValone, ImguiApp::bound_isoValtwo);
        
        checkCudaErrors(cudaMemcpy(lattice_data,h_lattice_dat,sizeof(float2)*(indi_range*indi_range*indi_range),cudaMemcpyHostToDevice));
        
        free(h_lattice_dat);

        free(h_dat);

        free_fft_resource();

        return 0;
    }


     int unit_latticeone()
    {
       
        fftlattice.create_lattice(d_volumeone,Nxu,Nyu,Nzu,sizeone,lattice_index_type);

        lattice.GPU_buffer_normalise_buffer(d_volumeone,d_volumeone,Nxu*Nyu*Nzu);

        if(ImguiApp::approx_unit_lattice)
        {
            cudaMemset(fft_gratings,0,sizeof(float) * (Nxu * Nyu * Nzu));

            float2* individual_grating = NULL;

            checkCudaErrors(cudaMalloc((void **)&individual_grating, sizeof(float2) * (Nxu*Nyu*Nzu)));

            float* individual_grating_one = NULL;

            checkCudaErrors(cudaMalloc((void **)&individual_grating_one, sizeof(float) * (Nxu*Nyu*Nzu)));

            cudaMemset(individual_grating_one,0,sizeof(float) * (Nxu * Nyu * Nzu));
        
            checkCudaErrors(cudaMemcpy2D(fft_data,(Nxu+1)*sizeof(float), d_volumeone,(Nxu*sizeof(float)), (Nxu*sizeof(float)),Nyu*Nzu, cudaMemcpyDeviceToDevice));
            
            getLastCudaError("fft data copy failed \n ");
            
            fftlattice.fft_func(fft_data);
            
            checkCudaErrors(cudaMemcpy(fft_data_compute, fft_data, (((Nxu/2)+1) * Nyu * Nzu) * sizeof(float2), cudaMemcpyDeviceToDevice));
            
            fftlattice.fft_scalar(fft_data_compute,sizeone,(((Nxu/2)+1) * Nyu * Nzu));

            uint mid_index = floor((Nxu*Nyu*Nzu)/2.0);

            fftlattice.fft_fill(fft_data_compute,fft_data_compute_fill,Nxu,Nyu,Nzu,mid_index);

            fftlattice.ifft_func(fft_data);
            
            float2 *h_dat;
            
            h_dat = (float2 *)malloc((Nxu*Nyu*Nzu) * sizeof(float2));
            
            checkCudaErrors(cudaMemcpy(h_dat, fft_data_compute_fill, (Nxu * Nyu * Nzu) * sizeof(float2), cudaMemcpyDeviceToHost));

            float2 *h_lattice_dat;
            
            h_lattice_dat = (float2 *)malloc((indi_range*indi_range*indi_range) * sizeof(float2));

            int l_count = 0;

            int r_count = 0;

            int midd1 = floor((Nxu*Nyu*Nzu)/2.0);
            
            int midd2 = floor((indi_range*indi_range*indi_range)/2.0) ;

            for(int k = -range_st;k <(range_st+1);k++)
            {
                    for(int j = -range_st;j <(range_st+1);j++)
                    {
                    
                            for(int i = -range_st;i <(range_st+1);i++)
                            {
                            
                                if(l_count < midd2)
                                {
                                    cudaMemset(individual_grating,0,sizeof(float2) * (Nxu * Nyu * Nzu));

                                    int u,s,t;

                                    int mid_idx = floor(Nxu/2.0);

                                    int mid_idy = floor(Nyu/2.0);

                                    int mid_idz = floor(Nzu/2.0);

                                    u = mid_idx + i;

                                    s = mid_idy + j;

                                    t = mid_idz + k;

                                    if( u == mid_idx)
                                    {
                                        u = 0;
                                    }
                                    else if( u > mid_idx)
                                    {
                                        u -= mid_idx;
                                    }
                                    else
                                    {
                                        u += (mid_idx+1);
                                    }

                                    if( s == mid_idy)
                                    {
                                        s = 0;
                                    }
                                    else if( s > mid_idy)
                                    {
                                        s -= mid_idy;
                                    }
                                    else
                                    {
                                        s += (mid_idy + 1);
                                    }

                                    if( t == mid_idz)
                                    {
                                        t = 0;
                                    }
                                    else if( t > (mid_idz))
                                    {
                                        t -= mid_idz;
                                    }
                                    else
                                    {
                                        t += (mid_idz + 1);
                                    }
                        
                                    int indx2 = u + s * Nxu + t * (Nxu*Nyu);

                                    h_lattice_dat[l_count] = h_dat[indx2];

                                    fftlattice.add_fft_constants(individual_grating, h_dat[indx2],indx2);

                                    fftlattice.ifft_func_complex(individual_grating);

                                    fftlattice.indiviual_grating_sum(individual_grating,individual_grating_one,Nxu,Nyu,Nzu);
                                    
                                    lattice.GPU_buffer_normalise_buffer(individual_grating_one,fft_gratings,Nxu*Nyu*Nzu);

                                    display_unit_lattice();

                                    ImguiApp::show_unit_lattice_data = true;

                                    ImguiApp::show_lattice_data = false;

                                    cudaExternalSemaphoreWaitParams waitParams = {};
                                    waitParams.flags = 0;
                                    waitParams.params.fence.value = 0;

                                    cudaExternalSemaphoreSignalParams signalParams = {};
                                    signalParams.flags = 0;
                                    signalParams.params.fence.value = 0;
                                                    
                                    checkCudaErrors(cudaSignalExternalSemaphoresAsync(&m_cudaSignalSemaphore, &signalParams, 1));
                            
                                    VulkanBaseApp::drawFrame(shift);

                                    checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaWaitSemaphore, &waitParams, 1));

                                    sleep(0.1);

                                    r_count++;

                                }

                                l_count++;
                            
                            }
                        
                    }

            }   

            checkCudaErrors(cudaMemcpy(fft_gratings, individual_grating_one, (Nxu * Nyu * Nzu) * sizeof(float), cudaMemcpyDeviceToDevice));

            checkCudaErrors(cudaFree(individual_grating));

            checkCudaErrors(cudaFree(individual_grating_one));

            lattice.GPU_buffer_normalise_buffer(fft_gratings,fft_gratings,Nxu*Nyu*Nzu);

            lattice.GPU_buffer_normalise_four(fft_gratings,d_volumeone_one,fft_gratings,Nxu*Nyu*Nzu,Nxu,Nyu,Nzu,ImguiApp::bound_isoValone, ImguiApp::bound_isoValtwo);
            
            checkCudaErrors(cudaMemcpy(lattice_data,h_lattice_dat,sizeof(float2)*(indi_range*indi_range*indi_range),cudaMemcpyHostToDevice));
            
            free(h_lattice_dat);

            free(h_dat);

            free_fft_resource();
        }

        return 0;
    }


    
     int spatial_lattice_run()
    {
        
        float2 *h_lattice_dat;

        h_lattice_dat = (float2 *)malloc((indi_range*indi_range*indi_range) * sizeof(float2));

        checkCudaErrors(cudaMemcpy(h_lattice_dat,lattice_data,sizeof(float2)*(indi_range*indi_range*indi_range),cudaMemcpyDeviceToHost));
            
        float mn_x = 0;

        float mn_y = 0;

        float mn_z = 0;

        if((latticetype_one == 'r'))
        {
            mn_x = (NumX/2.0);

            mn_y = (NumY/2.0);

            mn_z = (NumZ/2.0);
        }
        lattice.angle_data(d_theta,NumX,NumY,NumZ,dx,dy,dz,mn_x,mn_y,mn_z,'z');

        lattice.period_data(d_period,NumX,NumY,NumZ,dx,dy,dz,mn_x,mn_y,mn_z,'z');

        lattice.GPU_buffer_normalise_three(d_period,d_period,size,NumX/10,NumX/4);
       
        if (indi_range%2 == 0)
        {
            printf("Truncated Fft Matrix should have odd range. Exiting! \n");
            return -1;
        }
        
        int half_range = floor((indi_range * indi_range *indi_range)/2.0) ;
        
        int mycount = 0;

        int p0 = floor(Nxu/2.0);

        int q0 = floor(Nyu/2.0);

        int r0 = floor(Nzu/2.0);

        for(int k = -range_st;k <(range_st+1);k++)
        {
            for(int j = -range_st;j <(range_st+1);j++)
            {
                for(int i = -range_st;i <(range_st+1);i++)
                {
                    
                    if(mycount < (half_range) )
                    {
                    
                        lattice.finding_phi(d_phi,d_period,NumX,NumY,NumZ,i,j,k,dx,dy,dz,latticetype_one,
                        period_type,period_of_grating,x_period,y_period,z_period, ImguiApp::lcon, ImguiApp::lcon_1,ImguiApp::sinewave_zaxis);
                    
                        lattice.GPUCG_lattice(d_phi,500,1,0.01, FinalIter_l, FinalRes_l);
                        
                        lattice.copytotexture(d_phi,devPitchedPtr,NumX,NumY,NumZ);
                        
                        lattice.updateTexture(devPitchedPtr);

                        lattice.grating(d_ga,NumX2,NumY2,NumZ2,dx2,dy2,dz2);
                        
                        lattice.svl(d_svl,d_ga,NumX2,NumY2,NumZ2,mycount,lattice_data);
                    
                        lattice.GPU_buffer_normalise_four(d_svl,d_volumethree_one,d_volumethree,NumX2*NumY2*NumZ2,NumX2,NumY2,NumZ2,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo);

                        if(ImguiApp::primitives)
                        {
                            cudaMemcpy(d_latt_field,d_volumethree,size2 * sizeof(float),cudaMemcpyDeviceToDevice);

                            lattice.primitive_field(vol_one,d_boundary,d_volumethree,0.0,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic,NumX2,NumY2,NumZ2);
                       
                        }
                        else if(ImguiApp::structural || ImguiApp::thermal)
                        {
                            lattice.topo_field(d_volume_twice,d_volumethree_one,Topopt_val::VolumeFraction,NumX2,NumY2,NumZ2);
                        }

                        iso1 = 0.0;

                        iso2 = 0.0;

                        if(ImguiApp::primitives)
                        {
                            

                            checkCudaErrors(cudaMemset(d_raster, 0.0, (size) * sizeof(*d_raster)));

                            selectt.raster_update(0.0,0.0,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo,d_raster,vol_one,d_boundary,d_volumethree,d_latt_field,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic,NumX,NumY,NumZ);
                           
                            isosurf.computeIsosurface(d_raster, gridSize,d_postwo,d_normaltwo,0.0,numVoxelstwo,d_voxelVertstwo,d_voxelVertsScantwo,
                            d_voxelOccupiedtwo,d_voxelOccupiedScantwo,gridSizetwo,gridSizeShifttwo,gridSizeMasktwo,voxelSizetwo,gridcentertwo,
                            &activeVoxelstwo,&totalVertstwo,d_compVoxelArraytwo,maxmemvertstwo,vol_one,d_boundary,d_volume_twice,d_volumethree,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo, obj_union, obj_diff, obj_intersect,
                            ImguiApp::primitives,ImguiApp::structural,ImguiApp::lattice,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic);
                        
            
                        }
                        else
                        {
                            isosurf.computeIsosurface_lattice(d_volumethree_one,d_postwo,d_normaltwo,ImguiApp::bound_isoVal,
                            numVoxelstwo,d_voxelVertstwo,d_voxelVertsScantwo, d_voxelOccupiedtwo,d_voxelOccupiedScantwo,
                            gridSizetwo,gridSizeShifttwo,gridSizeMasktwo, voxelSizetwo,gridcentertwo,
                            &activeVoxelstwo, &totalVertstwo, d_compVoxelArraytwo,maxmemvertstwo,
                            d_volumethree,d_volumethree_two,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo,iso1,iso2);
                        }
                     
                        cudaExternalSemaphoreWaitParams waitParams = {};
                        waitParams.flags = 0;
                        waitParams.params.fence.value = 0;

                        cudaExternalSemaphoreSignalParams signalParams = {};
                        signalParams.flags = 0;
                        signalParams.params.fence.value = 0;
                                        
                        checkCudaErrors(cudaSignalExternalSemaphoresAsync(&m_cudaSignalSemaphore, &signalParams, 1));
                
                        VulkanBaseApp::drawFrame(shift);

                        checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaWaitSemaphore, &waitParams, 1));
                        
                    }   
                    mycount++;

                }
            }

        }
        free(h_lattice_dat);

        return 0;
    }

    void free_fft_resource()
    {
        checkCudaErrors(cudaFree(fft_data));

        checkCudaErrors(cudaFree(fft_data_compute));

        checkCudaErrors(cudaFree(fft_data_compute_fill));
        
    }

    void check_lattice()
    {

        cudaMemset(d_theta,0,sizeof(float) * size);

        cudaMemset(d_period,0,sizeof(float) * size);

        cudaMemset(d_phi,0,sizeof(float) * size);

        cudaMemset(d_svl,0,sizeof(float) * size2);

        cudaMemset(fft_gratings,0,sizeof(float) * (Nxu * Nyu * Nzu));

        cudaMemset(lattice_data,0,sizeof(float2) * (indi_range*indi_range*indi_range));

        cudaMemset(d_volumeone_one,0,sizeof(float) * (Nxu*Nyu*Nzu));

        fft_init();

        update_lattice_type();

        unit_lattice();

        ImguiApp::show_unit_lattice_data = false;
        if(!ImguiApp::primitive_lattice_options)
        {
            ImguiApp::show_lattice_data = true;
        }
    }

    void check_unit_lattice()
    {

        cudaMemset(fft_gratings,0,sizeof(float) * (Nxu * Nyu * Nzu));

        cudaMemset(lattice_data,0,sizeof(float2) * (indi_range*indi_range*indi_range));

        cudaMemset(d_volumeone_one,0,sizeof(float) * (Nxu*Nyu*Nzu));

        fft_init();

        update_lattice_type();

        unit_latticeone();

        display_unit_lattice();

        ImguiApp::show_unit_lattice_data = true;

        ImguiApp::show_lattice_data = false;
    
    }




    void display_unit_lattice()
    {
        cudaMemset(d_volumeone_one,0,sizeof(float) * (Nxu*Nyu*Nzu));

        if(ImguiApp::real_unit_lattice)
        {

            lattice.GPU_buffer_normalise_four(d_volumeone,d_volumeone_one,d_volumeone,Nxu*Nyu*Nzu,Nxu,Nyu,Nzu,ImguiApp::bound_isoValone, ImguiApp::bound_isoValtwo);
      
         
            isosurf.computeIsosurface_latticeone(d_volumeone_one,d_posone,d_normalone,ImguiApp::bound_isoVal,
                    numVoxelsone,d_voxelVertsone,d_voxelVertsScanone, d_voxelOccupiedone,d_voxelOccupiedScanone,
                    gridSizeone,gridSizeShiftone,gridSizeMaskone, voxelSizeone,gridcenterone,
                    &activeVoxelsone, &totalVertsone, d_compVoxelArrayone,maxmemvertsone,
                    d_volumeone,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo);

            getLastCudaError("display unit lattice failed");



        }
        if(ImguiApp::approx_unit_lattice)
        {
            lattice.GPU_buffer_normalise_four(fft_gratings,d_volumeone_one,fft_gratings,Nxu*Nyu*Nzu,Nxu,Nyu,Nzu,ImguiApp::bound_isoValone, ImguiApp::bound_isoValtwo);
           

            isosurf.computeIsosurface_latticeone(d_volumeone_one,d_posone,d_normalone,ImguiApp::bound_isoVal,
                    numVoxelsone,d_voxelVertsone,d_voxelVertsScanone, d_voxelOccupiedone,d_voxelOccupiedScanone,
                    gridSizeone,gridSizeShiftone,gridSizeMaskone, voxelSizeone,gridcenterone,
                    &activeVoxelsone, &totalVertsone, d_compVoxelArrayone,maxmemvertsone,
                    fft_gratings,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo);

            getLastCudaError("display unit lattice failed");
          
        }
    }

    void mainloopone(bool shift)
    {
        
        VulkanBaseApp::push_constants.mouse_click = 0;

        GaussInit();

        gettimeofday(&t1, 0);

        while (!glfwWindowShouldClose(window) && !(ImguiApp::initialise_grid) ) 
        {
           
            glfwPollEvents();

            drawFrame(shift);
          
            uint32_t FrmIdx = (currentFrame - 1) % ( swapChainImages.size());
          
            if(ImguiApp::initialise_grid)
            {
                
                if(ImguiApp::grid_value_check)
                {
                    update_topo_grid_data();

                    structure.NX = NumX;

                    structure.NY = NumY;

                    structure.NZ = NumZ;

                    thermal.NX = NumX;

                    thermal.NY = NumY;

                    thermal.NZ = NumZ;

                    opt_kernel.NX = NumX;

                    opt_kernel.NY = NumY;

                    opt_kernel.NZ = NumZ;

                    lattice.NX = NumX;

                    lattice.NY = NumY;

                    lattice.NZ = NumZ;

                    vulkan_create_topo_buffers();

                    vulkan_create_lattice_buffers();

                    init_Boundary();

                    init_selection();

                    lattice_init();

                    init_svl();

                    init_textures();

                    initMC();

                    initMC_unitlattice();

                    initMC_two();

                    printf("Initialisation Completed \n");
                    
                }

                ImguiApp::initialise_grid = false;
               
            }

            if((vulkan_buffer_created) && ((VulkanBaseApp::push_constants.mouse_click == 1) || (VulkanBaseApp::push_constants.mouse_click == 2) || (VulkanBaseApp::push_constants.mouse_click == -1)))
            {
                
                uint32_t currentFrmIdx = (currentFrame - 1) % ( swapChainImages.size());

                updateStorageBuffer(currentFrmIdx, load_selection, boundary_selection, delete_selection);

            }
            
            if(vulkan_buffer_created && (!initialise_grid) )
            {
                
                if(ImguiApp::structural && ImguiApp::update_load && ImguiApp::update_support  && ImguiApp::checkpoint == 0 && ImguiApp::execute_topo_data)
                {
                    
                    uint32_t currentFrmIdx = (currentFrame - 1) % ( swapChainImages.size());

                    cudaMemcpy(d_selection,d_cudastorageBuffers[currentFrmIdx],sizeof(REAL)*NumX*NumY*NumZ,cudaMemcpyDeviceToDevice);
               
                    ImguiApp::checkpoint = 1;

                    topoinit_struct();

              
                    VulkanBaseApp::topo_data = true;

                    ImguiApp::update_load = false;

                    ImguiApp::update_support = false;
                   
                    
                }
                else if(ImguiApp::thermal && ImguiApp::update_source && ImguiApp::update_sink && ImguiApp::checkpoint == 0 && ImguiApp::execute_topo_data)
                {
                        uint32_t currentFrmIdx = (currentFrame - 1) % ( swapChainImages.size());

                        cudaMemcpy(d_selection,d_cudastorageBuffers[currentFrmIdx],sizeof(REAL)*NumX*NumY*NumZ,cudaMemcpyDeviceToDevice);
                       
                        topoinit_thermal();

                        ImguiApp::checkpoint = 2;
                        
                        VulkanBaseApp::topo_data = true;

                        ImguiApp::update_source = false;

                        ImguiApp::update_sink = false;
                        
                }


            }

       
            if(ImguiApp::execute_primitive_lattice)
            {                
            
                check_lattice();

                spatial_lattice_run();

                ImguiApp::execute_primitive_lattice = false;
              
            }

            if(ImguiApp::execute_lattice_data)
            {
                check_lattice();

                spatial_lattice_run();
            
                ImguiApp::execute_lattice_data = false;
                
            }

            if(ImguiApp::view_lattice)
            {
                check_lattice();

                spatial_lattice_run();

                ImguiApp::view_lattice = false;

            }

             if(ImguiApp::view_unit_lattice_data)
            {
                ImguiApp::view_lattice = false;

                check_unit_lattice();

                ImguiApp::view_unit_lattice_data = false;

            }

            if(ImguiApp::show_lattice_data && ImguiApp::update_isorange  && ImguiApp::lattice)
            {
                lattice.GPU_buffer_normalise_four(d_svl,d_volumethree_one,d_volumethree,NumX2*NumY2*NumZ2,NumX2,NumY2,NumZ2,VulkanBaseApp::bound_isoValone,VulkanBaseApp::bound_isoValtwo);
            
                isosurf.computeIsosurface_lattice(d_volumethree_one,d_postwo,d_normaltwo,ImguiApp::bound_isoVal,
                numVoxelstwo,d_voxelVertstwo,d_voxelVertsScantwo, d_voxelOccupiedtwo,d_voxelOccupiedScantwo,
                gridSizetwo,gridSizeShifttwo,gridSizeMasktwo, voxelSizetwo,gridcentertwo,
                &activeVoxelstwo, &totalVertstwo, d_compVoxelArraytwo,maxmemvertstwo,
                d_volumethree,d_volumethree_two,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo,iso1,iso2);

                ImguiApp::update_isorange = false;

            }


            if(ImguiApp::primitive_lattice_options && ImguiApp::update_isorange  && ImguiApp::primitives)
            {
                     
                lattice.primitive_field(vol_one,d_boundary,d_volumethree,0.0,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic,NumX2,NumY2,NumZ2);
                       
                iso1 = 0.0;

                iso2 = 0.0;

                
                checkCudaErrors(cudaMemset(d_raster, 0.0, (size) * sizeof(*d_raster)));

                selectt.raster_update(0.0,0.0,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo,d_raster,vol_one,d_boundary,d_volumethree,d_latt_field,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic,NumX,NumY,NumZ);

                isosurf.computeIsosurface(d_raster, gridSize,d_postwo,d_normaltwo,0.0,numVoxelstwo,d_voxelVertstwo,d_voxelVertsScantwo,
                d_voxelOccupiedtwo,d_voxelOccupiedScantwo,gridSizetwo,gridSizeShifttwo,gridSizeMasktwo,voxelSizetwo,gridcentertwo,
                &activeVoxelstwo,&totalVertstwo,d_compVoxelArraytwo,maxmemvertstwo,vol_one,d_boundary,d_volume_twice,d_volumethree,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo, obj_union, obj_diff, obj_intersect,
                ImguiApp::primitives,ImguiApp::structural,ImguiApp::lattice,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic);
            
                ImguiApp::update_isorange = false;

            }

            if(ImguiApp::cad_bool  && ImguiApp::primitives)
            {
                
                checkCudaErrors(cudaMemset(d_raster, 0.0, (size) * sizeof(*d_raster)));

                selectt.raster_update(0.0,0.0,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo,d_raster,vol_one,d_boundary,d_volumethree,d_latt_field,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic,NumX,NumY,NumZ);
                
                isosurf.computeIsosurface(d_raster,gridSize,d_postwo,d_normaltwo,0.0,numVoxelstwo,d_voxelVertstwo,d_voxelVertsScantwo,
                d_voxelOccupiedtwo,d_voxelOccupiedScantwo,gridSizetwo,gridSizeShifttwo,gridSizeMasktwo,voxelSizetwo,gridcentertwo,
                &activeVoxelstwo,&totalVertstwo,d_compVoxelArraytwo,maxmemvertstwo,vol_one,d_boundary,d_volume_twice,d_volumethree,ImguiApp::bound_isoValone,ImguiApp::bound_isoValtwo, obj_union, obj_diff, obj_intersect,
                ImguiApp::primitives,ImguiApp::structural,ImguiApp::lattice,ImguiApp::lattice_fixed,ImguiApp::lattice_dynamic);
            }

            if(ImguiApp::show_unit_lattice_data && ImguiApp::update_unit_isorange)
            {
                
                display_unit_lattice();
                
                ImguiApp::update_unit_isorange = false;
                
            }

            if(ImguiApp::primitives && ImguiApp::show_model)
            {
                
                if((!ImguiApp::show_primitive_lattice))
                {
                    show_model();

                }   
                if(boundary_buffers && ImguiApp::primitive_done_lattice_do && ImguiApp::svl_data)
                {
                    
                        ImguiApp::show_primitive_lattice = true;

                        check_lattice();

                        spatial_lattice_run();

                        printf("Lattice generated  \n");

                        ImguiApp::primitive_done_lattice_do = false;

                        ImguiApp::show_model = false;

                }
            }

        
    
            if((ImguiApp::execute_topo_data) && (ImguiApp::vulkan_buffer_created))
            {
          
                if((OptIter < Topopt_val::MaxOptIter) )
                {
                    if (ImguiApp::checkpoint == 1)
                    {
                        toprun_struct();

                        isosurf.patch_grid(d_volume_s,NumX,NumY,NumZ, 0.31);
                        
                        lattice.copytotexture(d_volume_s,devPitchedPtr,NumX,NumY,NumZ);

                        lattice.updateTexture(devPitchedPtr);

                        lattice.refine(d_volume_twice,NumX2,NumY2,NumZ2,dx2,dy2,dz2);
                         
                        isosurf.computeIsosurface_2(d_postwo,d_normaltwo,0.31,numVoxelstwo,d_voxelVertstwo,d_voxelVertsScantwo,
                        d_voxelOccupiedtwo,d_voxelOccupiedScantwo,gridSizetwo,gridSizeShifttwo,gridSizeMasktwo,voxelSizetwo,
                        gridcentertwo,&activeVoxelstwo,&totalVertstwo,d_compVoxelArraytwo,maxmemvertstwo,d_volume_twice,0.31);

                        ImguiApp::checkpoint = 3;

                       
                    }
                    else if (ImguiApp::checkpoint == 2)
                    {
                        toprun_thermal();

                        isosurf.patch_grid(d_volume_t,NumX,NumY,NumZ, 0.31);

                        lattice.copytotexture(d_volume_t,devPitchedPtr,NumX,NumY,NumZ);

                        lattice.updateTexture(devPitchedPtr);

                        lattice.refine(d_volume_twice,NumX2,NumY2,NumZ2,dx2,dy2,dz2);

                        isosurf.computeIsosurface_2(d_postwo,d_normaltwo,0.31,numVoxelstwo,d_voxelVertstwo,d_voxelVertsScantwo,
                        d_voxelOccupiedtwo,d_voxelOccupiedScantwo,gridSizetwo,gridSizeShifttwo,gridSizeMasktwo,voxelSizetwo,
                        gridcentertwo,&activeVoxelstwo,&totalVertstwo,d_compVoxelArraytwo,maxmemvertstwo,d_volume_twice,0.31);

                        ImguiApp::checkpoint = 4;
                        
                    }

                    ImguiApp::topo_done_lattice_do = true;
                }

                ImguiApp::execute_topo_data = false;
                
            }

            if((ImguiApp::checkpoint == 3) || (ImguiApp::checkpoint == 4))
            {
                if(ImguiApp::generate_topo_lattice)
                {
                    
                    ImguiApp::generate_topo_lattice = false;
                    
                    ImguiApp::show_topo_lattice = true;

                    check_lattice();
                    
                    if(ImguiApp::checkpoint == 3)
                    {
                    
                        lattice.copytotexture(d_volume_s,devPitchedPtr,NumX,NumY,NumZ);
                    }
                    if(ImguiApp::checkpoint == 4)
                    {
                   
                        lattice.copytotexture(d_volume_t,devPitchedPtr,NumX,NumY,NumZ);
                    }

                    lattice.updateTexture(devPitchedPtr);

                    lattice.refine(d_volume_twice,NumX2,NumY2,NumZ2,dx2,dy2,dz2);

                    spatial_lattice_run();
                
                }
         
            }

            if(ImguiApp::export_data_primitive)
            {
                
                int check_folder = mkdir("../Results_Modelling", 0777);

                if ( (check_folder != 0) && (errno != EEXIST))
                {
                    throw std::runtime_error("Failed to create 'Results_Modelling' folder ");
                }

                else
                {
                    const char *filenameshape;

                    if(!show_primitive_lattice)
                    {
                        filenameshape = "../Results_Modelling/Shape.obj";
                    }   
                    else
                    {
                        filenameshape = "../Results_Modelling/Shape_Lattice.obj";
                    }

                    output_file.file_write(d_postwo,totalVertstwo,filenameshape, true);

                
                }

                ImguiApp::export_data_primitive = false;
            }

            if(ImguiApp::export_data_optimise)
            {
                int check_folder = mkdir("../Results_Optimise", 0777);

                if ( (check_folder != 0) && (errno != EEXIST))
                {
                    throw std::runtime_error("Failed to create 'Results_Optimise' folder ");
                }

                else
                {

                    const char *filenameoptimise;

                    if(topo_done_lattice_do && !show_lattice_data)
                    {
                        filenameoptimise = "../Results_Optimise/Optimise_shape.obj";
                    }
                    else
                    {
                        filenameoptimise = "../Results_Optimise/Optimise_Lattice_shape.obj";
                    }

                    output_file.file_write(d_postwo,totalVertstwo,filenameoptimise, false);

                }

                ImguiApp::export_data_optimise = false;
            }


            if(ImguiApp::export_data_lattice)
            {
                int check_folder = mkdir("../Results_Lattice", 0777);

                if ( (check_folder != 0) && (errno != EEXIST))
                {
                    throw std::runtime_error("Failed to create 'Results_Lattice' folder ");
                }
                else
                {
                
                    const char *filenameone = "../Results_Lattice/Unit_Lattice.obj";

                    output_file.file_write(d_posone,totalVertsone,filenameone, false);

                    const char *filenametwo = "../Results_Lattice/Spatial_Lattice.obj";

                    output_file.file_write(d_postwo,totalVertstwo,filenametwo, false);

                }

                ImguiApp::export_data_lattice = false;
            }

            if(ImguiApp::execute_done && ImguiApp::vulkan_buffer_created )
            {

                ImguiApp::show_model = false;

                ImguiApp::show_primitive_lattice = false;

                ImguiApp::show_topo_lattice = false;

                ImguiApp::show_lattice_data = false;

                ImguiApp::show_unit_lattice_data = false;

                ImguiApp::calculate = true;
    
                vkDeviceWaitIdle(device);

                if(ImguiApp::boundary_buffers)
                {
                    cleanup_Boundary();

                    ImguiApp::boundary_buffers = false;
                }
                if(VulkanBaseApp::topo_data)
                {
                    cleanup_topo();

                    VulkanBaseApp::topo_data = false;
                }
                
                if(ImguiApp::svl_data)
                {
                    cleanup_svl();

                    cleanup_fft_data();

                    ImguiApp::svl_data = false;
                }
               
                if(ImguiApp::checkpoint == 3)
                {
                    structure.GPUCleanUp();

                    ImguiApp::checkpoint = 0;
                }
                if(ImguiApp::checkpoint == 4)
                {
                    thermal.GPUCleanUp();

                    ImguiApp::checkpoint = 0;
                }
             
                if(ImguiApp::lattice_buffer_created)
                {
                    destroy_lattice_buffers();

                    cleanup_isosurf_lattice();

                    cleanup_isosurf_unit_lattice();

                    lattice.GPUCleanUp();

                    ImguiApp::lattice_buffer_created = false;
                }
                if(ImguiApp::vulkan_buffer_created)
                {
                    destroy_buffers_n_memory();

                    cleanup_isosurf();

                    cleanup_selection();

                    ImguiApp::vulkan_buffer_created = false;
                }


                std::cout<<"Exiting \n"<<std::endl;
                
                ImguiApp::execute_done = false;
             
            }
          
        }
        std::cout<<"Mainloop Terminated \n"<<std::endl;     

        vkDeviceWaitIdle(device);


        if(ImguiApp::boundary_buffers)
        {
            cleanup_Boundary();

            ImguiApp::boundary_buffers = false;
        }


        if(VulkanBaseApp::topo_data)
        {
       
            cleanup_topo();

            VulkanBaseApp::topo_data = false;
        }

        if(ImguiApp::checkpoint == 3)
        {
            structure.GPUCleanUp();

            ImguiApp::checkpoint = 0;
        }
        if(ImguiApp::checkpoint == 4)
        {
            thermal.GPUCleanUp();

            ImguiApp::checkpoint = 0;
        }

        if(ImguiApp::svl_data)
        {
            cleanup_svl();

            cleanup_fft_data();
         
        }

        if(ImguiApp::lattice_buffer_created)
        {
            destroy_lattice_buffers();

            cleanup_isosurf_lattice();

            cleanup_isosurf_unit_lattice();

            lattice.GPUCleanUp();
        }
       
        if(ImguiApp::vulkan_buffer_created)
        {
            
            cleanup_isosurf();

            destroy_buffers_n_memory(); 

            cleanup_selection();
        }

        cleanup_cuda_storage_buffer_handle();
  
    }


    void cleanup_Boundary()
    {
        if(d_boundary)
        {
            checkCudaErrors(cudaFree(d_boundary));
        }

        if(d_latt_field)
        {
            checkCudaErrors(cudaFree(d_latt_field));
        }


        if(vol_one)
        {
            checkCudaErrors(cudaFree(vol_one));
        }

    }

    void cleanup_cuda_storage_buffer_handle()
    {
        for (size_t i = 0; i < d_cudastorageBuffers.size(); i++) 
        {
            checkCudaErrors(cudaDestroyExternalMemory(d_cudastorageMemory[i]));
            
            checkCudaErrors(cudaFree(d_cudastorageBuffers[i]));
        }
    }


    void cleanup_svl()
    {
            
        lattice.GPUCleanUp();  

        lattice.deleteTexture();

        if(devPitchedPtr.ptr)
        {
            checkCudaErrors(cudaFree(devPitchedPtr.ptr));
        }
        if(d_phi)
        {
            checkCudaErrors(cudaFree(d_phi));
        }
    
        if(d_ga)
        {
            checkCudaErrors(cudaFree(d_ga));
        }

        if(d_svl)
        {
            checkCudaErrors(cudaFree(d_svl));
        }
    
        if(d_theta)
        {
            checkCudaErrors(cudaFree(d_theta));
        }

        if(d_period)
        {
            checkCudaErrors(cudaFree(d_period));
        }

        if(d_volumethree_one)
        {
            checkCudaErrors(cudaFree(d_volumethree_one));
        }

         if(d_volumethree_two)
        {
            checkCudaErrors(cudaFree(d_volumethree_two));
        }

        if(d_volume_twice != nullptr)
        {
            checkCudaErrors(cudaFree(d_volume_twice));
        }
   
    }

    void cleanup_fft_data()
    {
   
        if(d_volumeone_one)
        {
            checkCudaErrors(cudaFree(d_volumeone_one));
        }
    
        if(fft_gratings)
        {
            checkCudaErrors(cudaFree(fft_gratings));
        }

        if(lattice_data)
        {
            checkCudaErrors(cudaFree(lattice_data));
        }

        checkCudaErrors(cufftDestroy(planc2r));

        checkCudaErrors(cufftDestroy(planr2c));

        checkCudaErrors(cufftDestroy(planc2c));
    }


    void  cleanup_isosurf()
    {
        isosurf.destroyAllTextureObjects();
    
        checkCudaErrors(cudaFree(d_edgeTable));

        checkCudaErrors(cudaFree(d_triTable));

        checkCudaErrors(cudaFree(d_numVertsTable));

        checkCudaErrors(cudaFree(d_voxelVerts));

        checkCudaErrors(cudaFree(d_voxelVertsScan));

        checkCudaErrors(cudaFree(d_voxelOccupied));

        checkCudaErrors(cudaFree(d_voxelOccupiedScan));

        checkCudaErrors(cudaFree(d_compVoxelArray));

    }

      void cleanup_isosurf_unit_lattice()
    {
      
        checkCudaErrors(cudaFree(d_voxelVertsone));

        checkCudaErrors(cudaFree(d_voxelVertsScanone));

        checkCudaErrors(cudaFree(d_voxelOccupiedone));

        checkCudaErrors(cudaFree(d_voxelOccupiedScanone));

        checkCudaErrors(cudaFree(d_compVoxelArrayone));

       
    }

    void cleanup_isosurf_lattice()
    {
      
        checkCudaErrors(cudaFree(d_voxelVertstwo));


        checkCudaErrors(cudaFree(d_voxelVertsScantwo));


        checkCudaErrors(cudaFree(d_voxelOccupiedtwo));


        checkCudaErrors(cudaFree(d_voxelOccupiedScantwo));


        checkCudaErrors(cudaFree(d_compVoxelArraytwo));

       
    }

  
    void cleanup_topo()
    {
        
        checkCudaErrors(cudaFree(d_us));

        checkCudaErrors(cudaFree(d_grads));

        checkCudaErrors(cudaFree(d_den));

    }

    void cleanup_selection()
    {
        if(d_selection != nullptr)
        {
            checkCudaErrors(cudaFree(d_selection));
        }
    }


    int compute_n_visualise()
    {
        VulkanBaseApp::init();

        mainloopone(shift);

        printf("Exiting the Application \n");

        return 0;
    }

};



int main(int argc, char** argv)
{
    bool validation = true;

    std::string validate_data ="";

    if(argc == 2)
    {
        validate_data = argv[1];

        if(validate_data == "false")
        {
            validation = false;
        }

        else if(validate_data == "true")
        {
            validation = true;
        }
        else
        {
            throw std::runtime_error("Unknow bool type ");
        }
       
    }
    else if (argc == 1 && (validate_data ==""))
    {
        validation = false;
        
    }


    Multitopo app(validation);

    app.compute_n_visualise();
    
    return 0;

}












