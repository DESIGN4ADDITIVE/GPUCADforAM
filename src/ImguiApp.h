#pragma once


#ifndef __IMGUIAPP_H__
#define __IMGUIAPP_H__


#include"../src/general/DataTypes.h"

#include "imgui_folder/imgui.h"
#include "imgui_folder/imgui_impl_glfw.h"
#include "imgui_folder/imgui_impl_vulkan.h"

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>




class ImguiApp 
{
    

    protected:

    static bool vulkan_buffer_created;
    static int grid_value;
    static bool grid_value_check;
    static bool initialise_grid;
    static bool execute_signal;
    static bool execute_done;
    static bool execute_primitive_lattice;
    static bool execute_lattice_data;
    static bool export_data_primitive;
    static bool export_data_optimise;
    static bool export_data_lattice;
  
    static bool view_settings;
    static bool grid_settings;
    static bool background_color;

    static bool execute_optimize;

    static bool execute_primitive;
    static bool execute_lattice;
    static bool execute_topo_data;

   
    static bool view_front;
    static bool view_back;
    static bool view_top;
    static bool view_bottom;
    static bool view_right;
    static bool view_left;

    static bool reset_load_button;
    static bool reset_support_button;

    static bool reset_source_button;
    static bool reset_sink_button;

    static bool structural;
    static bool thermal;
    static bool primitives;
    static bool lattice;

    static bool show_model;
      
    static bool topo_done_lattice_do;
    static bool generate_topo_lattice;
    static bool show_topo_lattice;

    static bool primitive_done_lattice_do;
    static bool show_primitive_lattice;
    static bool show_lattice_data;

    static bool view_lattice;
    static bool view_unit_lattice_data;
    static bool show_unit_lattice_data;

    static bool update_load;
    static bool update_support;
    static bool update_source;
    static bool update_sink;

    static bool spatial_angle_window;
    static bool spatial_period_window;

    static bool x_axis;
    static bool y_axis;
    static bool z_axis;

    
    static bool select_load_node;
    static bool select_support_node;

    static bool fea_settings;
    static bool fea_settings_set;

    static bool cg_solver_settings;
    static bool solver_settings_set;

    static bool optimisation_settings;
    static bool optimisation_settings_set;

    static uint lattice_type_index;
    static uint lattice_size_index;

    static uint lattice_index_type;

    static bool unit_lattice_settings;

    static bool svl_data;

    static float bound_isoVal;
    static float bound_isoValone;
    static float bound_isoValtwo;

    static bool update_isorange;
    static bool update_unit_isorange;


    static bool retain;
    static bool calculate;
    static bool undoo;

    public:

    static bool cylind_selected;
    static bool cylind_disc_selected;
    static bool cuboid_selected ;
    static bool cuboid_shell_selected;
    static bool sphere_selected ;
    static bool sphere_shell_selected;
    static bool torus_selected ;
    static bool cone_selected ;

    static float radius;
    static float thickness_radial;
    static float thickness_axial;

    static float sphere_radius;
    static float sphere_thickness;

    static float cuboid_x;
    static float cuboid_y;
    static float cuboid_z;
    static float cu_sh_thick;

    static float torus_radius;
    static float torus_circle_radius;

    static float cone_height;
    static float base_radius;

    static bool lattice_buffer_created;
    static bool boundary_buffers;

    static bool debug_window;

    static bool real_unit_lattice;
    static bool approx_unit_lattice;

    std::vector<bool*> window_bools;
    std::vector<bool*> view_bools;
    std::vector<bool*> physics_bools;

    static float3 center;
    static float3 axis;
    static float3 angles;

    /////////////////////////////////////////////////////////////
    
    static int period_type;
    static float period_of_grating;
    static float x_period;
    static float y_period;
    static float z_period;

    ////////////////////////////////////////////////////////////

    static float zoom_value;
    //////////////////////////////////////////////////////////////

    static int checkpoint;

    /////////////////////////////////////////////////////////////

    static bool export_settings;


    static ImVec2 window_extent;

    static ImVec4 clear_color;

    ImguiApp();
    ~ImguiApp();
    static void show_view_settings(bool *view_setting, bool *shift, bool *reset,bool *show_grid, bool *show_mesh ,float *f1,float *f2, float *f3, float *f4);
    static void show_execute_topo(bool *execute_setting, bool *execute_signal, bool *execute_done );
    static void show_execute_lattice(bool *execute_lattice, bool *execute_signal, bool *execute_done );
    static void show_grid_settings(bool *grid_setting, bool vulkan_buffer_created, ImVec4 clear_color);
    static void show_background_color_settings(bool *background_color, ImVec4& clear_color);
    static void show_select_load_structure(bool *window);
    static void show_select_support_structure(bool *window);
    static void show_select_load_thermal();
    static void show_select_support_thermal();
    static void show_selected_primitive();
    void make_inactive(std::vector<bool*> window_bools,bool* active);
    void make_all_inactive(std::vector<bool*> window_bools);
    static void show_fea_settings();
    static void show_cg_solver_settings();
    static void show_optimisation_settings();
    static void show_unit_lattice_settings();
    static void show_spatial_period_settings();
    static void show_spatial_angle_settings();
    static void show_export_settings();
    static void show_debugging_window();
};

class Topopt_val
{
    public:
    static REAL Youngs_Modulus; //Young Modulus
    static REAL poisson; //Poisson's Ratio
    static REAL conductivity;


    static REAL pexp ; //penalty exponent
    static REAL VolumeFraction ;//0.2;
    static REAL FilterRadius ;//2.8;//2.2;
    static int iter;
    static REAL EndRes;
    static int MaxOptIter;
    static REAL MinDens;
 

    Topopt_val();
    ~Topopt_val();

    
};

#endif