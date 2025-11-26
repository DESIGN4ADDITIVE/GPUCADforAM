# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

TARGET_SIZE := $(shell getconf LONG_BIT)

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)

# Common includes and paths for CUDA
INCLUDES  := -I./src/commons
LIBRARIES :=

################################################################################

# MakeFile include to help find Vulkan SDK and dependencies
VULKAN_HEADER := /usr/include
VULKAN_SDK_LIB := /usr/lib/x86_64-linux-gnu 
# Vulkan specific libraries
ifeq ($(TARGET_OS),linux)

 LIBRARIES += -L$(VULKAN_SDK_LIB)
 LIBRARIES += `pkg-config --static --libs glfw3` -lvulkan
 INCLUDES  += `pkg-config --static --cflags glfw3` -I$(VULKAN_HEADER)
 
endif
CURRENT_SM := 89

GENCODE_FLAGS += -gencode arch=compute_$(CURRENT_SM),code=compute_$(CURRENT_SM)

ALL_CCFLAGS += --std=c++14 --threads 0 
LIBRARIES += -lcufft

BUILDDIR = build
BINDIR = bin

PRIMITIVE = Results_Modelling
LATTICE = Results_Lattice
OPTIMISE = Results_Optimise

################################################################################
msg1 := "Compiling Shaders  - Step 1 of 23"

msg2 := "Compiling imgui_widgets File 'imgui_folder/imgui_widgets.cpp' - Step 2 of 23"

msg3 := "Compiling imgui_tables File 'imgui_folder/imgui_tables.cpp' - Step 3 of 23"

msg4 := "Compiling imgui_impl_glfw File 'imgui_folder/imgui_impl_glfw.cpp' - Step 4 of 23"

msg5 := "Compiling imgui_draw File 'imgui_folder/imgui_draw.cpp' - Step 5 of 23"

msg6 := "Compiling imgui_demo File 'imgui_folder/imgui_demo.cpp' - Step 6 of 23"

msg7 := "Compiling imgui File 'imgui_folder/imgui.cpp' - Step 7 of 23"

msg8 := "Compiling imgui_impl_vulkan File 'imgui_folder/imgui_impl_vulkan.cpp' - Step 8 of 23"

msg9 := "Compiling ImguiApp File 'ImguiApp.cpp' - Step 9 of 23"

msg10 := "Compiling general_kernels File 'gpu_kernels/general_kernels.cu' - Step 10 of 23"

msg11 := "Compiling Optimisation_kernels File 'gpu_kernels/Optimisation_kernels.cu' - Step 11 of 23"

msg12 := "Compiling Selection File 'Selection.cu' - Step 12 of 23"

msg13 := "Compiling File_ouput File 'File_ouput.cu' - Step 13 of 23"

msg14 := "Compiling Fft_lattice File 'lattice_files/Fft_lattice.cu' - Step 14 of 23"

msg15 := "Compiling Gratings File 'lattice_files/Gratings.cu' - Step 15 of 23"

msg16 := "Compiling Structuralsim File 'topo_structural/Structuralsim.cu' - Step 16 of 23"

msg17 := "Compiling Thermalsim File 'topo_thermal/Thermalsim.cu' - Step 17 of 23"

msg18 := "Compiling Modelling File 'Modelling.cu' - Step 18 of 23"

msg19 := "Compiling Isosurface File 'Isosurface.cu' - Step 19 of 23"

msg20 := "Compiling MarchingCubes_kernel File 'MarchingCubes_kernel.cu' - Step 20 of 23"

msg21 := "Compiling VulkanBaseApp File 'VulkanBaseApp.cpp' - Step 21 of 23"

msg22 := "Compiling main File 'main.cu' - Step 22 of 23"

msg23 := "Building executable 'GPUCADforAM' - Step 23 of 23"

msg24 := "Compilation Completed !"
################################################################################


DIRS := build bin

GLSLC_PATH := $(shell which glslc)
msg0 := ""glslc found at: $(GLSLC_PATH)""

# Target rules

shader:
ifeq ($(GLSLC_PATH),)
	$(error "Error: glslc executable PATH not found.")
else
	@echo $(msg0)
	@echo $(msg1)
	@$(shell ./src/compile_shaders.sh)
endif


all: build

build: $(shell mkdir -p $(DIRS)) shader $(BINDIR)/GPUCADforAM

shader:
	@echo $(msg1)
	@$(shell ./src/compile_shaders.sh)


$(BUILDDIR)/imgui_widgets.o:src/imgui_folder/imgui_widgets.cpp
	@echo $(msg2)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<


$(BUILDDIR)/imgui_tables.o:src/imgui_folder/imgui_tables.cpp
	@echo $(msg3)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/imgui_impl_glfw.o:src/imgui_folder/imgui_impl_glfw.cpp
	@echo $(msg4)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<


$(BUILDDIR)/imgui_draw.o:src/imgui_folder/imgui_draw.cpp
	@echo $(msg5)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/imgui_demo.o:src/imgui_folder/imgui_demo.cpp
	@echo $(msg6)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/imgui.o:src/imgui_folder/imgui.cpp
	@echo $(msg7)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<


$(BUILDDIR)/imgui_impl_vulkan.o:src/imgui_folder/imgui_impl_vulkan.cpp
	@echo $(msg8)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<


$(BUILDDIR)/ImguiApp.o:src/ImguiApp.cpp
	@echo $(msg9)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<


$(BUILDDIR)/general_Kernels.o:src/gpu_kernels/general_kernels.cu
	@echo $(msg10)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<


$(BUILDDIR)/Optimisation_Kernels.o:src/gpu_kernels/Optimisation_kernels.cu
	@echo $(msg11)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/Selection.o:src/Selection.cu
	@echo $(msg12)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/File_output.o:src/File_output.cu
	@echo $(msg13)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<


$(BUILDDIR)/Fft_lattice.o:src/lattice_files/Fft_lattice.cu
	@echo $(msg14)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/Gratings.o:src/lattice_files/Gratings.cu
	@echo $(msg15)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/Structuralsim.o:src/topo_structural/Structuralsim.cu
	@echo $(msg16)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/Thermalsim.o:src/topo_thermal/Thermalsim.cu
	@echo $(msg17)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/Modelling.o:src/Modelling.cu
	@echo $(msg18)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/Isosurface.o:src/Isosurface.cu
	@echo $(msg19)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/MarchingCubes_kernel.o:src/MarchingCubes_kernel.cu
	@echo $(msg20)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/VulkanBaseApp.o:src/VulkanBaseApp.cpp
	@echo $(msg21)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/main.o:src/main.cu
	@echo $(msg22)
	@$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

# $(info    LIBRARIES is $(LIBRARIES))


OBJS = $(addprefix $(BUILDDIR)/,imgui_widgets.o imgui_tables.o imgui_impl_glfw.o imgui_draw.o imgui_demo.o imgui.o imgui_impl_vulkan.o ImguiApp.o general_Kernels.o Optimisation_Kernels.o Selection.o File_output.o Fft_lattice.o Gratings.o Structuralsim.o Thermalsim.o Modelling.o Isosurface.o MarchingCubes_kernel.o VulkanBaseApp.o main.o)

$(BINDIR)/GPUCADforAM: $(OBJS)
	@echo $(msg23)
	@$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

	@echo $(msg24)
	

run: build
	./bin/GPUCADforAM



clean:
	rm -f src/shaders/*.spv

	@if [ -d "$(BUILDDIR)" ]; then \
	echo "Removing  '$(BUILDDIR)' folder "; \
	rm -rf "$(BUILDDIR)"; fi

	@if [ -d "$(BINDIR)" ]; then \
	echo "Removing  '$(BINDIR)' folder "; \
	rm -rf "$(BINDIR)"; fi

	@if [ -d "$(PRIMITIVE)" ]; then \
	echo "Removing  '$(PRIMITIVE)' folder "; \
	rm -rf "$(PRIMITIVE)"; fi

	@if [ -d "$(LATTICE)" ]; then \
	echo "Removing  '$(LATTICE)' folder "; \
	rm -rf "$(LATTICE)"; fi

	@if [ -d "$(OPTIMISE)" ]; then \
	echo "Removing  '$(OPTIMISE)' folder "; \
	rm -rf "$(OPTIMISE)"; fi




