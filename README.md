# GPUCADforAM

This software application helps you in creating implicit CAD models (currently primitves), topology optimised structure (currently structural and thermal ) , lattice structures that can rotate, bend and change its size and shape spatially and export the data in '.obj' format.

## Implicit Modelling 
Uses mathematical equation to create field. Using Isosurface Extraction (Marching Cube Alogrithm) user intended geometry is extracted. 

## Topology Optimisation

Finding the optimal material layout based the physics of the process.

## Spatially Varying Lattice

Distributing lattice structure in 3D space by changing it shape and size while maintaining the topology.



This software had complied and ran in Ubuntu 22.04.4 LTS (jammy). 

Users should have a GPU device with atleast 8 gb memory. It may run on lower end GPU's for low grid resolution but have not tested.


## Prerequisite
1. Install CUDA Toolkit (Cuda - GPU programming)
2. Install GLFW (Window Manager)
3. Install Vulkan (New generation Cross platform API for 3D graphics)


For more details please check DEPENDENCIES.md file.

## Download and Compilation 
1. Select a folder or directory
2. Use ' git clone https://github.com/DESIGN4ADDITIVE/GPUCADforAM.git' or download the "zip" folder and extract it in the folder(directory) selected.
3. Select and open the  directory 'GPUCADforAM'.
4. Open a terminal inside 'GPUCADforAM' folder.
5. In the ' Makefile ' provide the location of ' nvcc ' compiler in CUDA_PATH. (usually in ' /usr/local/cuda ')
6. In the ' Makefile ' provide the compute capability value in 'CURRENT_SM ' (here 8.9 so '89'). You can either use the link ' https://developer.nvidia.com/cuda-gpus ' or the command ' nvidia-smi --query-gpu=gpu_name,compute_cap --format=csv '
7. Next type ' make all '
8. Check for an output you get in the terminal  message is ' Compilation Completed! ' for a successful compilation.
9. Got to 'GPUCADforAM/bin' folder. Open a command terminal within the folder and type './GPUCADforAM' or double click on the executable 'GPUCADforAM'.

## Installation Demo 

[![IMAGE ALT TEXT](http://img.youtube.com/vi/VFgrwhIZFUg/0.jpg)](http://www.youtube.com/watch?v=VFgrwhIZFUg "GPUCADforAM")
