
#include "ImguiApp.h"
#include "../src/general/topopt_defines.h"


int ImguiApp::grid_value;

int ImguiApp::checkpoint = 0;

bool ImguiApp::vulkan_buffer_created = false;

bool ImguiApp::svl_data = false;

bool ImguiApp::retain = false;
bool ImguiApp::calculate = true;
bool ImguiApp::undoo = false;

uint ImguiApp::lattice_type_index = 0;
uint ImguiApp::lattice_size_index = 0;

float ImguiApp::sphere_radius = 5;
float ImguiApp::sphere_thickness = 2.0;

float ImguiApp::radius = 5;
float ImguiApp::thickness_radial = 2.0;
float ImguiApp::thickness_axial = 2.0;

float ImguiApp::cuboid_x = 5.0;
float ImguiApp::cuboid_y = 2.0;
float ImguiApp::cuboid_z = 4.0;

float ImguiApp::cu_sh_thick = 2.0;

float ImguiApp::torus_radius = 5.0;
float ImguiApp::torus_circle_radius = 2.0;

float ImguiApp::cone_height = 6;
float ImguiApp::base_radius = 3;

int ImguiApp::period_type = 0;

float ImguiApp::period_of_grating = 30.0;
float ImguiApp::x_period = 30.0;
float ImguiApp::y_period = 30.0;
float ImguiApp::z_period = 30.0;

bool ImguiApp::update_isorange = false;
bool ImguiApp::update_unit_isorange = false;

float ImguiApp::bound_isoVal = 0.25;
float ImguiApp::bound_isoValone = 0.20;
float ImguiApp::bound_isoValtwo = 0.30;

uint ImguiApp::lattice_index_type = 0;

bool ImguiApp::grid_value_check;
bool ImguiApp::initialise_grid = false;
bool ImguiApp::execute_signal;
bool ImguiApp::execute_done = false;
bool ImguiApp::execute_primitive_lattice = false;
bool ImguiApp::execute_lattice_data = false;

bool ImguiApp::view_lattice = false;


////////////PRIMITVE//////////////////////////
bool ImguiApp::cylind_selected = false;
bool ImguiApp::cylind_disc_selected = false;
bool ImguiApp::cuboid_selected = false;
bool ImguiApp::cuboid_shell_selected = false;
bool ImguiApp::sphere_selected = false;
bool ImguiApp::sphere_shell_selected = false;
bool ImguiApp::torus_selected = false;
bool ImguiApp::cone_selected = false;
//////////////////////////////////////////////

bool ImguiApp::view_settings = false;
bool ImguiApp::grid_settings = false;
bool ImguiApp::background_color = false;
bool ImguiApp::execute_optimize = false;
bool ImguiApp::select_load_node = false;
bool ImguiApp::select_support_node = false;


bool ImguiApp::execute_lattice = false;
bool ImguiApp::execute_primitive = false;
bool ImguiApp::execute_topo_data = false;

bool ImguiApp::update_load = false;
bool ImguiApp::update_support = false;
bool ImguiApp::update_source = false;
bool ImguiApp::update_sink = false;

bool ImguiApp::spatial_angle_window = false;
bool ImguiApp::spatial_period_window = false;

bool ImguiApp::fea_settings = false;
bool ImguiApp::fea_settings_set = false;


bool ImguiApp::cg_solver_settings = false;
bool ImguiApp::solver_settings_set = false;

bool ImguiApp::optimisation_settings = false;
bool ImguiApp::optimisation_settings_set = false;

bool ImguiApp::unit_lattice_settings = false;

bool ImguiApp::x_axis = false;
bool ImguiApp::y_axis = false;
bool ImguiApp::z_axis = false;

bool ImguiApp::view_front = true;
bool ImguiApp::view_back = false;
bool ImguiApp::view_top = false;
bool ImguiApp::view_bottom = false;
bool ImguiApp::view_right = false;
bool ImguiApp::view_left = false;

bool ImguiApp::reset_load_button = false;
bool ImguiApp::reset_support_button = false;

bool ImguiApp::reset_source_button = false;
bool ImguiApp::reset_sink_button = false;

bool ImguiApp::lattice_buffer_created = false;
bool ImguiApp::boundary_buffers = false;

bool ImguiApp::primitive_done_lattice_do = false;
bool ImguiApp::show_primitive_lattice = false;

bool ImguiApp::topo_done_lattice_do = false;
bool ImguiApp::generate_topo_lattice = false;
bool ImguiApp::show_topo_lattice = false;

bool ImguiApp::show_lattice_data = false;

bool ImguiApp::view_unit_lattice_data = false;
bool ImguiApp::show_unit_lattice_data = false;

bool ImguiApp::real_unit_lattice = false;
bool ImguiApp::approx_unit_lattice = false;

float ImguiApp::zoom_value = 1.0;

bool ImguiApp::export_settings = false;

bool ImguiApp::debug_window = false;

bool ImguiApp::show_model = false;

bool ImguiApp::export_data_primitive = false;

bool ImguiApp::export_data_optimise = false;

bool ImguiApp::export_data_lattice = false;

ImVec2 ImguiApp::window_extent = {50,50};

ImVec4 ImguiApp::clear_color = ImVec4(0.148f, 0.148f, 0.148f, 1.00f);

/////////////////////Topopt_val//////////////////////////////
REAL Topopt_val::Youngs_Modulus = 1.0; //Young Modulus
REAL Topopt_val::poisson = 0.3; //Poisson's Ratio
REAL Topopt_val::conductivity = 1.0;


REAL Topopt_val::pexp = 3.0; //penalty exponent
REAL Topopt_val::VolumeFraction = 0.4;//0.2;
REAL Topopt_val::FilterRadius = 3.0;//2.8;//2.2;
int Topopt_val::iter = 10;
REAL Topopt_val::EndRes = 0.009;
int Topopt_val::MaxOptIter = 10;
REAL Topopt_val::MinDens = 0.2;

bool ImguiApp::structural = false;
bool ImguiApp::thermal = false;
bool ImguiApp::primitives = false;
bool ImguiApp::lattice = false;

float3 ImguiApp::center = {0.0,0.0,0.0};
float3 ImguiApp::axis = {0.0,0.0,1.0};
float3 ImguiApp::angles = {0.0,0.0,0.0};

Topopt_val::Topopt_val()
{

}

Topopt_val::~Topopt_val()
{

}
/////////////////////////////////////////////////////////////////////

ImguiApp::ImguiApp()
{
    window_bools = {
    &cylind_selected,
    &cylind_disc_selected,
    &cuboid_selected,
    &cuboid_shell_selected,
    &sphere_selected,
    &sphere_shell_selected,
    &torus_selected,
    &cone_selected,
    &view_settings,
    &grid_settings,
    &background_color,
    &execute_optimize,
    &execute_lattice,
    &execute_primitive,
    &select_load_node,
    &select_support_node,
    &spatial_angle_window,
    &spatial_period_window,
    &fea_settings,
    &optimisation_settings,
    &cg_solver_settings,
    &unit_lattice_settings,
    &export_settings,
    &debug_window
        
    


    };

    view_bools = {
        &view_front,
        &view_back,
        &view_top,
        &view_bottom,
        &view_right,
        &view_left,
        &view_settings
    };
    
    physics_bools = {
        &structural,
        &thermal,
        &primitives,
        &lattice
    };
}

 ImguiApp::~ImguiApp()
 {

 }


 void sphere_settings()
 {

    ////////////////////////////////////////////////////////////////
            ImGui::SeparatorText("Center");
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float c_x = 0.0;
            ImGui::InputFloat("x1", &c_x, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.x = c_x;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float c_y = 3.0;
            ImGui::InputFloat("y1", &c_y, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.y = c_y;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float c_z = 0.0;
            ImGui::InputFloat("z1", &c_z, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.z = c_z;
            }
     


            static bool rad1 = false;
            static bool thik = false;
            static bool thik_ax = false;
            static float rad = 5.0f;
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            ImGui::SliderFloat("Sphere Radius", &rad,2, 100, "%.1f");
            rad1 = ImGui::IsItemActive();

            if(rad1)
            {
                ImguiApp::sphere_radius = rad;
            }


            static float thick = 2.0f;
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            ImGui::SliderFloat("Wall Thickness ", &thick,1, 20, "%.1f");
            thik = ImGui::IsItemActive();  
         
            if(thik && ImguiApp::sphere_shell_selected)
            {
                ImguiApp::sphere_thickness = thick;
          
            }
 }



 void cylinder_settings()
 {
    ////////////////////////////////////////////////////////////////
            ImGui::SeparatorText("Center");
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float c_x = 0.0;
            ImGui::InputFloat("x1", &c_x, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.x = c_x;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float c_y = 3.0;
            ImGui::InputFloat("y1", &c_y, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.y = c_y;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float c_z = 0.0;
            ImGui::InputFloat("z1", &c_z, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.z = c_z;
            }
           
            /////////////////////////////////////////////////////////////
            ImGui::SeparatorText("Axis");
            ImGui::SetNextItemWidth(500);
            static float a_xis[3] = { 0.0f, 0.0f, 1.0f};
            ImGui::DragFloat3("Axis",a_xis,0.01,-1.0,1.0,"%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::axis.x = a_xis[0];
                ImguiApp::axis.y = a_xis[1];
                ImguiApp::axis.z = a_xis[2];
            }
          
           
            ////////////////////////////////////////////////////////////////


            static bool rad1 = false;
            static bool thik = false;
            static bool thik_ax = false;
            static float rad = 5.0f;
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            ImGui::SliderFloat("Radius", &rad,1, 150, "%.1f");
            rad1 = ImGui::IsItemActive();
     
            static float thick_axial = 6.0f;
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            ImGui::SliderFloat("Thickness Axial", &thick_axial,0, 150 , "%.1f");
            thik_ax = ImGui::IsItemActive();  
            if(rad1 || thik_ax )
            {
                ImguiApp::radius = rad;
                
                ImguiApp::thickness_axial = thick_axial;
            }

            if(ImguiApp::cylind_disc_selected)
            {
                static float thick = 2.0f;
                ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
                ImGui::SliderFloat("Thickness Radial", &thick,0, 150, "%.1f");
                thik = ImGui::IsItemActive(); 
                if(thik)
                {
                    ImguiApp::thickness_radial = thick;
                
                } 
            }
 }


  void cuboid_settings()
 {

    ////////////////////////////////////////////////////////////////
            ImGui::SeparatorText("Center");
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float c_x = 0.0;
            ImGui::InputFloat("x1", &c_x, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.x = c_x;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float c_y = 3.0;
            ImGui::InputFloat("y1", &c_y, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.y = c_y;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float c_z = 0.0;
            ImGui::InputFloat("z1", &c_z, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.z = c_z;
            }
     


            static bool xx_roll = false;
            static bool yy_pitch = false;
            static bool zz_yaw = false;

            ImGui::SeparatorText("Angle");
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float x_roll = 0.0;
            ImGui::SliderFloat("roll", &x_roll, -180.0f, 180.0f, "%.1f");
            xx_roll = ImGui::IsItemActive();
      
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float y_pitch = 0.0;
            ImGui::SliderFloat("pitch", &y_pitch, -180.0f, 180.0f, "%.1f");
            yy_pitch = ImGui::IsItemActive();
          
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float z_yaw = 0.0;
            ImGui::SliderFloat("yaw", &z_yaw, -180.0f, 180.0f, "%.1f");
            zz_yaw = ImGui::IsItemActive();
            if(xx_roll || yy_pitch || zz_yaw)
            {
                ImguiApp::angles.x = (x_roll/180)*3.14;
                ImguiApp::angles.y = (y_pitch/180)*3.14;
                ImguiApp::angles.z = (z_yaw/180)*3.14;
            }
     
            ImGui::SeparatorText("Width");
            static bool x_wid = false;
            static bool y_wid = false;
            static bool z_wid = false;

          
            static float x_1 = 5.0f;
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            ImGui::SliderFloat("x_width", &x_1,1, 50, "%.1f");
            x_wid = ImGui::IsItemActive();
            static float y_1 = 2.0f;
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            ImGui::SliderFloat("y_width ", &y_1,1, 50, "%.1f");
            y_wid = ImGui::IsItemActive();  
            static float z_1 = 4.0f;
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            ImGui::SliderFloat("z_width ", &z_1,1, 50, "%.1f");
            z_wid = ImGui::IsItemActive(); 
            if(x_wid || y_wid || z_wid )
            {
                ImguiApp::cuboid_x = x_1;
                ImguiApp::cuboid_y = y_1;
                ImguiApp::cuboid_z = z_1;
          
            }

            if(ImguiApp::cuboid_shell_selected)
            {
                ImGui::SeparatorText("Thickness");
                static bool cu_thick = false;
                static float c_t = 2.0f;
                ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
                ImGui::SliderFloat("thickness", &c_t,1, 10, "%.1f");
                cu_thick = ImGui::IsItemActive();
                if(cu_thick)
                {
                    ImguiApp::cu_sh_thick = c_t;
                }
            }

 }


 void torus_settings()
 {

    ////////////////////////////////////////////////////////////////
            ImGui::SeparatorText("Torus Center");
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float c_x = 0.0;
            ImGui::InputFloat("x1", &c_x, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.x = c_x;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float c_y = 3.0;
            ImGui::InputFloat("y1", &c_y, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.y = c_y;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float c_z = 0.0;
            ImGui::InputFloat("z1", &c_z, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.z = c_z;
            }
     


            static bool xx_roll = false;
            static bool yy_pitch = false;
            static bool zz_yaw = false;

            ImGui::SeparatorText("Angle");
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float x_roll = 0.0;
            ImGui::SliderFloat("roll", &x_roll, -180.0f, 180.0f, "%.1f");
            xx_roll = ImGui::IsItemActive();
      
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float y_pitch = 0.0;
            ImGui::SliderFloat("pitch", &y_pitch, -180.0f, 180.0f, "%.1f");
            yy_pitch = ImGui::IsItemActive();
          
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float z_yaw = 0.0;
            ImGui::SliderFloat("yaw", &z_yaw, -180.0f, 180.0f, "%.1f");
            zz_yaw = ImGui::IsItemActive();
            if(xx_roll || yy_pitch || zz_yaw)
            {
                ImguiApp::angles.x = (x_roll/180)*3.14;
                ImguiApp::angles.y = (y_pitch/180)*3.14;
                ImguiApp::angles.z = (z_yaw/180)*3.14;
            }
     
            ImGui::SeparatorText("Torus Radius");
            static bool T_rad = false;
            static bool C_rad = false;
    
            static float Tor_rad = 5.0f;
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            ImGui::SliderFloat("Torus Radius", &Tor_rad,1, 100, "%.1f");
            T_rad = ImGui::IsItemActive();
            static float Cir_rad = 2.0f;
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            ImGui::SliderFloat("Circle Radius ", &Cir_rad,1, 5, "%.1f");
            C_rad = ImGui::IsItemActive();  
          
            if(T_rad || C_rad )
            {
                ImguiApp::torus_radius = Tor_rad;
                ImguiApp::torus_circle_radius = Cir_rad;
            }

   

 }


  void cone_settings()
 {
      ////////////////////////////////////////////////////////////////
            ImGui::SeparatorText("Cone Center");
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float co_x = 0.0;
            ImGui::InputFloat("x1", &co_x, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.x = co_x;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float co_y = 0.0;
            ImGui::InputFloat("y1", &co_y, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.y = co_y;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float co_z = 0.0;
            ImGui::InputFloat("z1", &co_z, 0.1f, 1.0f, "%.1f");
            if(ImGui::IsItemActive())
            {
                ImguiApp::center.z = co_z;
            }
     


            static bool cone_roll = false;
            static bool cone_pitch = false;
            static bool cone_yaw = false;

            ImGui::SeparatorText("Angle");
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float conx_roll = 0.0;
            ImGui::SliderFloat("roll", &conx_roll, -180.0f, 180.0f, "%.1f");
            cone_roll = ImGui::IsItemActive();
      
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float cony_pitch = 1.0;
            ImGui::SliderFloat("pitch", &cony_pitch, -180.0f, 180.0f, "%.1f");
            cone_pitch = ImGui::IsItemActive();
          
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            static float conz_yaw = 0.0;
            ImGui::SliderFloat("yaw", &conz_yaw, -180.0f, 180.0f, "%.1f");
            cone_yaw = ImGui::IsItemActive();
            if(cone_roll || cone_pitch || cone_yaw)
            {
                ImguiApp::angles.x = (conx_roll/180)*3.14;
                ImguiApp::angles.y = (cony_pitch/180)*3.14;
                ImguiApp::angles.z = (conz_yaw/180)*3.14;
  

            }

            ImGui::SeparatorText("Radius & Height");
            static bool co_rad = false;
            static bool co_hei = false;
    
            static float cone_rad = 3.0f;
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            ImGui::SliderFloat("Base Radius", &cone_rad,1, 100, "%.1f");
            co_rad = ImGui::IsItemActive();
            static float cone_hei = 6.0f;
            ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
            ImGui::SliderFloat("Cone Height ", &cone_hei,1, 100, "%.1f");
            co_hei = ImGui::IsItemActive();  
          
            if(co_rad || co_hei )
            {
                ImguiApp::base_radius = cone_rad;
                ImguiApp::cone_height = cone_hei;
            }

 }

 void ImguiApp::show_execute_topo(bool *execute_setting, bool *execute_signal, bool *execute_done )
 {
    ImGui::Begin("EXECUTE OPTIMISE", execute_setting); 
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);
    static int execute_code_num = 0;
    static int check_num = 0;
    if(ImguiApp::execute_signal)
    {
        ImGui::SeparatorText("EXECUTE");
        execute_code_num = 0;
        
        ImGui::Text("Execute the program");
        if (ImGui::Button("EXECUTE"))
        {
            if(!ImguiApp::fea_settings_set || !ImguiApp::optimisation_settings_set || !ImguiApp::solver_settings_set )
            {
                check_num = 1;
            }

            else if(ImguiApp::structural && ((!ImguiApp::update_load || !ImguiApp::update_support)))
            {
                
                check_num = 2;
                
            }

            else if(ImguiApp::thermal && (!ImguiApp::update_source || !ImguiApp::update_sink) )
            {
               
                check_num = 3;
                
            }

            else
            {
                execute_code_num++;
                check_num = 0;
            }

          
        }
        if(check_num == 1)
        {
            if(!ImguiApp::fea_settings_set)
            {
                ImGui::Text("Please Set FEA Settings");
            }
            if(!ImguiApp::solver_settings_set)
            {
                ImGui::Text("Please Set Solver Settings");
            }
            if(!ImguiApp::optimisation_settings_set)
            {
                ImGui::Text("Please Set Optimisation Settings before proceed!");
            }
        }
        if(check_num == 2)
        {
            if(!ImguiApp::update_load)
            {
                ImGui::Text("Please Apply Load ");
            }

            if(!ImguiApp::update_support)
            {
                ImGui::Text("Please Apply Support ");
            }


        }

        if(check_num == 3)
        {
            if(!ImguiApp::update_source)
            {
                ImGui::Text("Please Apply Source ");
            }

            if(!ImguiApp::update_sink)
            {
                ImGui::Text("Please Apply Sink ");
            }


        }
    }
    
    if((execute_code_num == 1) && (execute_signal))
    {
        ImguiApp::execute_topo_data = true;
        *execute_signal = false;
        execute_code_num++;
        
        
    }
    else if ((execute_code_num == 2) && (!(*execute_signal)))
    {
        
     
        if(!ImguiApp::topo_done_lattice_do && (ImguiApp::checkpoint != 0))
        {
            ImGui::Text("Executing the data...");
            ImGui::Text("Please Wait ");
        }
        else if(ImguiApp::topo_done_lattice_do)
        {
            ImGui::Text("Execution Done !");
        }
        
        ImGui::NewLine();
        
        if(ImGui::Button("LATTICE GENERATION ") && ImguiApp::topo_done_lattice_do)
        {
            ImguiApp::generate_topo_lattice = true;
           
        }

        ImGui::NewLine();
        ImGui::NewLine();

        if(ImGui::Button("RE RUN ") && ImguiApp::topo_done_lattice_do)
        {
            ImguiApp::execute_topo_data = true;
            ImguiApp::topo_done_lattice_do = false;
            if(ImguiApp::show_topo_lattice)
            {
                ImguiApp::show_topo_lattice = false;
            }
            ImguiApp::checkpoint = 0;
           
            if(ImguiApp::structural)
            {
                ImguiApp::update_load = true;
                ImguiApp::update_support = true;
                
            }
            else if(ImguiApp::thermal)
            {
                ImguiApp::update_source = true;
                ImguiApp::update_sink = true;
            }
        }
        else
        {
            ImguiApp::execute_topo_data = false;
        }

        ImGui::NewLine();
        ImGui::NewLine();
        ImGui::SeparatorText("CLEAR DATA");
        if (ImGui::Button("CLEAR OPTIMISATION DATA"))
        {

            *execute_done = true;
            execute_code_num++;
        
            
        }

 

      
        
    }
    else if ((execute_code_num > 2) && (!(*execute_signal)))
    {
        *execute_done = false;
        ImGui::Text("Optimisation Data Cleared!");
        execute_code_num++;
        check_num = 0;
        ImguiApp::fea_settings_set = false;
        ImguiApp::optimisation_settings_set = false;
        ImguiApp::solver_settings_set = false;

        *execute_setting = false;
        
    }

 
 
    ImGui::End();
 }



 void ImguiApp::show_execute_lattice(bool *execute_lattice, bool *execute_signal, bool *execute_done )
 {
    ImGui::Begin("EXECUTE SETTINGS", execute_lattice); 
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);
    static int execute_lattice_num = 0;

     if(ImguiApp::execute_signal)
    {
        ImGui::SeparatorText("EXECUTE");
        execute_lattice_num = 0;
        ImGui::Text("Execute the program");
        if (ImGui::Button("EXECUTE LATTICE"))
        {
            
            execute_lattice_num++;
        }
    }
    
    if((execute_lattice_num == 1) && (execute_signal))
    {
        ImguiApp::execute_lattice_data = true;
        *execute_signal = false;
        execute_lattice_num++;
    }
    else if ((execute_lattice_num == 2) && (!(*execute_signal)))
    {
        
        ImGui::SeparatorText("CLEAR DATA");
        ImGui::Text("Execute Done !");
        ImGui::NewLine();
        if (ImGui::Button("CLEAR LATTICE DATA"))
        {

            *execute_done = true;
            execute_lattice_num++;
            
        
        }

        ImguiApp::execute_lattice_data = false;
        
    }
    else if ((execute_lattice_num > 2) && (!(*execute_signal)))
    {
        *execute_done = false;
        ImGui::Text("Lattice Data Cleared!");
        *execute_lattice = false;
    }
 
    ImGui::End();
 }


//  void ImguiApp::show_execute_primitive(bool *execute_primitive, bool *execute_signal, bool *execute_code, bool *execute_done  )
//  {
//     ImGui::Begin("CLEAR PRIMITIVE DATA", execute_primitive); 
//     ImGui::SetWindowPos(ImVec2(5,50));
//     ImGui::SetWindowSize(window_extent);
//     static int execute_lattice_num = 0;

//      if(ImguiApp::execute_signal && ImguiApp::show_model)
//     {
//         ImGui::SeparatorText("CLEAR");
//         execute_lattice_num = 0;
//         ImGui::Text("CLEAR DATA");
//         if (ImGui::Button("CLEAR PRIMITIVE"))
//         {
            
//             execute_lattice_num++;
//             *execute_signal = false;
//             *execute_done = true;
         
//         }
//     }

//     else if ((execute_lattice_num > 1) && (!(*execute_signal)))
//     {
//         *execute_done = false;
//         ImGui::Text("Primitive Data Cleared!");

//         *execute_primitive = false;
//     }
 
//     ImGui::End();
//  }


void ImguiApp::show_grid_settings(bool *grid_setting, bool vulkan_buffer_created, ImVec4 clear_color)
{
    ImGui::Begin("GRID SETTINGS", grid_setting); 
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);
    ImGui::SetWindowCollapsed(true,2);
    ImGui::Text("3D GRID SPECIFICATIONS");

    ImGui::SeparatorText("Enter Grid Dimension");
    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
    static int i0 = 16;

    ImGui::InputInt("Enter Grid Dimension ", &i0,1,32);
    grid_value = i0;

    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
    static float f0 = 1.0f;
    ImGui::InputFloat("Grid Spacing", &f0, 0.1f, 1.0f, "%.2f");

    static int initialise_grid_num = 0;

    if((initialise_grid_num == 5) && (!(execute_done)) && (!(vulkan_buffer_created)))
    {
        
        if (ImGui::Button("REINITIALISE"))
        {
         
            if(initialise_grid_num == 5)
            {
                initialise_grid_num = 1;

                reset_load_button = true;
                reset_support_button = true;

                reset_source_button = true;
                reset_sink_button = true;
            }
        }
    }

    else if((initialise_grid_num == 0 || initialise_grid_num == 2) && !(execute_done) )
    {
        if (ImGui::Button("INITIALISE"))
        {
         
            if(initialise_grid_num == 0)
            {
                initialise_grid_num++;
            }
            else if(initialise_grid_num == 2)
            {
                initialise_grid_num = 1;
            }

            
        }
    }



    if (initialise_grid_num == 1)
    {
        ImGui::SameLine();
        
            
        if ((ImguiApp::grid_value < 16 ) || (ImguiApp::grid_value > 150))
        {
            ImguiApp::grid_value_check = false;
        }
        else
        {
            ImguiApp::grid_value_check = true;
        }
        

        if(ImguiApp::grid_value_check && (initialise_grid_num == 1))
        {
            ImGui::Text("Initialising 3D grid!");
            ImguiApp::initialise_grid = true;
            initialise_grid_num = 5;
            ImguiApp::execute_signal = true;
            ImguiApp::execute_done = false;
            
        }

        else if ((!ImguiApp::grid_value_check))
        {
            
            initialise_grid_num = 2;
            

        }

    }
    else if(initialise_grid_num == 2)
    {
        ImGui::SameLine();
        ImGui::Text("Grid dimension shoulbe in between 16 and 150");
        
    }

    else if((initialise_grid_num == 5) && (ImguiApp::execute_signal))
    {
        
        
        ImGui::Text("Initialisation Done!");
        
    }

    ImGui::End();
}


void ImguiApp::show_background_color_settings(bool *background_color, ImVec4& clear_color)
{
    ImGui::Begin("BACKGROUND COLOR", background_color); 
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);
    ImGui::SetWindowCollapsed(true,2);
    ImGui::Text("BACKGROUND COLOR");

    
    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.8);
    ImGui::NewLine();
    static float  r0 = 0.148f;
    ImGui::SliderFloat("Red ", &r0, 0.0f, 1.0f, "%.3f");
    clear_color.x = r0;
    
    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.8);
    ImGui::NewLine();
    static float  g0 = 0.148f;
    ImGui::SliderFloat("Green ", &g0, 0.0f, 1.0f, "%.3f");
    clear_color.y = g0;
    
    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.8);
    ImGui::NewLine();
    static float  b0 = 0.148f;
    ImGui::SliderFloat("Blue ", &b0, 0.0f, 1.0f, "%.3f");
    clear_color.z = b0;
    
    // ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.35);
    // ImGui::NewLine();
    // static float  al_0 = 0.392f;
    // ImGui::SliderFloat("Alpha ", &al_0, 0.0f, 1.0f, "%.3f");
    // clear_color.w = al_0;
    

    ImGui::End();
}
void ImguiApp::show_view_settings(bool *view_setting, bool *shift, bool *reset, bool *show_grid, bool *show_mesh, float *f1,float *f2, float *f3, float *f4)
{
    
    ImGui::Begin("VIEW SETTINGS", view_setting); 
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);
    ImGui::SetWindowCollapsed(true,2);
    ImGui::NewLine();
    ImGui::Checkbox("ROTATE",shift);

    ImGui::NewLine();
    static float f00 = 1.00f;
    ImGui::SliderFloat("Zoom", &f00, 0.01f, 1.00f, "%.2f");
    ImguiApp::zoom_value = f00;

    ImGui::NewLine();
    ImGui::NewLine();

    ImGui::Checkbox("Show 3D Grid", show_grid); 

    static float f1_1 = 0.0f;
    ImGui::SliderFloat("Point Size1", &f1_1, 0.0f, 10.0f, "%.1f");
    static float f1_2 = 0.0f;
    ImGui::SliderFloat("Point Size2", &f1_2, 0.0f, 10.0f, "%.1f");
    static float f1_3 = 0.0f;
    ImGui::SliderFloat("Point Size3", &f1_3, 0.0f, 10.0f, "%.1f");
    static float f1_4 = (ImguiApp::structural || ImguiApp::thermal) ? 15.0 : 3.0;
    ImGui::SliderFloat("Point Size4", &f1_4, 0.0f, 15.0f, "%.1f");

    ImGui::Checkbox("Show 3D Mesh", show_mesh);

    *f1 = f1_1;
    *f2 = f1_2;
    *f3 = f1_3;
    *f4 = f1_4;

    ImGui::End();
}


void ImguiApp::show_select_load_structure(bool* window)
{
    
    ImGui::Begin("STRUCTURAL LOAD",window); 
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);
    ImGui::SetWindowCollapsed(true,2);

    static int execute_load = 0;

    if(ImguiApp::reset_load_button)
    {
        execute_load = 0;
        reset_load_button = false;
    }

    ImGui::NewLine();
    if(execute_load == 0)
    {
        
        ImGui::Checkbox("X",&x_axis);
        ImGui::Checkbox("Y",&y_axis);
        ImGui::Checkbox("Z",&z_axis);
        
        if(ImGui::Button("APPLY LOAD"))
        {
         
            if(select_load_node)
            {
                
                ImguiApp::update_load = true;
                execute_load++;
            }

        }
    }
    if(execute_load == 1)
    {
        
        ImGui::Text("Loads Applied ! ");
       
    }
    


    ImGui::End();
}


void ImguiApp::show_select_load_thermal()
{
    
    ImGui::Begin("THERMAL"); 
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);
    ImGui::SetWindowCollapsed(true,2);

    static int execute_source = 0;

    if(ImguiApp::reset_source_button)
    {
    execute_source = 0;
    reset_source_button = false;

    }

    ImGui::NewLine();
    if(execute_source == 0)
    {
        if(ImGui::Button("APPLY SOURCE"))
        {
            if(select_load_node)
            {
                ImguiApp::update_source = true;
                execute_source++;
            }

        }
    }
    if(execute_source == 1)
    {
        
        ImGui::Text("Source Applied ! ");

    }
    
    ImGui::End();
}


void ImguiApp::show_select_support_structure(bool *window)
{
    
    ImGui::Begin("STRUCTURAL SUPPORT",window); 
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);
    ImGui::SetWindowCollapsed(true,2);

    static int execute_support = 0;

    if(ImguiApp::reset_support_button)
    {
        execute_support = 0;
        reset_support_button = false;
        
    }

    ImGui::NewLine();
    if(execute_support == 0)
    {
        if(ImGui::Button("APPLY SUPPORT"))
        {
            if(select_support_node)
            {
                update_support = true;
                execute_support++;
            }

        }
    }
    if(execute_support == 1)
    {
        ImGui::Text("Support Applied ! ");
     
        
    }
    
    ImGui::End();
}


void ImguiApp::show_select_support_thermal()
{
    
    ImGui::Begin("THERMAL"); 
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);
    ImGui::SetWindowCollapsed(true,2);

    static int execute_sink = 0;

    if(ImguiApp::reset_sink_button)
    {
        execute_sink = 0;
        reset_sink_button = false;
        
    }
    
    ImGui::NewLine();
    if(execute_sink == 0)
    {
        if(ImGui::Button("APPLY SINK"))
        {
            if(select_support_node)
            {
                update_sink = true;
                execute_sink++;
            }

        }
    }
    if(execute_sink == 1)
    {
        ImGui::Text("Sink Applied ! ");
    }
    


    ImGui::End();
}


void ImguiApp::show_selected_primitive()
{

            if(ImguiApp::cylind_selected)
            {
                ImGui::Begin("CYLINDER PARAMETERS", &ImguiApp::cylind_selected ); 
            }

            if(ImguiApp::cylind_disc_selected)
            {
                ImGui::Begin("CYLINDER DISC PARAMETERS", &ImguiApp::cylind_disc_selected ); 
            }

            if(ImguiApp::sphere_selected)
            {
              ImGui::Begin("SPHERE PARAMETERS", &ImguiApp::sphere_selected ); 
            }
            if(ImguiApp::sphere_shell_selected)
            {
                ImGui::Begin("SPHERE SHELL PARAMETERS", &ImguiApp::sphere_shell_selected ); 
            }

            if(ImguiApp::cuboid_selected)
            {
                ImGui::Begin("CUBOID PARAMETERS", &ImguiApp::cuboid_selected ); 
            }

            if(ImguiApp::cuboid_shell_selected)
            {
                ImGui::Begin("CUBOID SHELL PARAMETERS", &ImguiApp::cuboid_shell_selected ); 
            }

            if(ImguiApp::torus_selected)
            {
                ImGui::Begin("TORUS PARAMETERS",&ImguiApp::torus_selected);
            }

            if(ImguiApp::cone_selected)
            {
                ImGui::Begin("CONE PARAMETERS",&ImguiApp::cone_selected);
            }

            ImGui::SetWindowPos(ImVec2(5,50));
            ImGui::SetWindowSize(window_extent);
            ImGui::SetWindowCollapsed(true,2);


            ///////////////////////////////////////////////////////////////////////////////////////////////
            if(ImguiApp::cylind_selected || ImguiApp::cylind_disc_selected)
            {
                cylinder_settings();
            }

            if(ImguiApp::sphere_selected || ImguiApp::sphere_shell_selected)
            {
                sphere_settings();
            }

            if(ImguiApp::cuboid_selected || ImguiApp::cuboid_shell_selected)
            {
                cuboid_settings();
            }

            if(ImguiApp::torus_selected)
            {
                torus_settings();
            }
            if(ImguiApp::cone_selected)
            {
                cone_settings();
            }
        
            if(ImguiApp::boundary_buffers)
            {
                ImGui::NewLine();
                if (ImGui::Button("RETAIN"))
                {
        
                    ImguiApp::retain = true;
                    ImguiApp::calculate = false;
                    ImguiApp::undoo = false;
                }

                ImGui::SameLine();
                ImGui::Text("  ");
                ImGui::SameLine();
                if(ImGui::Button("UNDO"))
                {
                    ImguiApp::undoo = true;
                    ImguiApp::calculate = false;
                    ImguiApp::retain = false;
                }

                ImGui::NewLine();
                if(ImGui::Button("CONTINUE"))
                {
                    ImguiApp::calculate = true;
                    ImguiApp::retain = false;
                    ImguiApp::undoo = false;
                    ImguiApp::show_model = true;
                    ImguiApp::show_primitive_lattice = false;
                }
                ImGui::NewLine();
                ImGui::NewLine();
                if(!show_primitive_lattice)
                {
                    if(ImGui::Button("GENERATE LATTICE"))
                    {
                        if(!ImguiApp::retain)
                        {
                            ImguiApp::retain = true;
                            ImguiApp::calculate = false;
                            ImguiApp::undoo = false;
                        }
                        ImguiApp::primitive_done_lattice_do = true;
                        
                    }
                }
            }
            else
            {
                ImguiApp::debug_window = true;
            }
            
            ImGui::End();
}


void ImguiApp::show_fea_settings()
{
    ImGui::Begin("FEA SETTINGS",&ImguiApp::fea_settings);
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);

    ImGui::NewLine();
    if(ImguiApp::structural)
    {
        ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
        static float fea_0 = 1.0f;
        ImGui::InputFloat("Young's Modulus ", &fea_0, 1.0f, 10000000.0f,"%.2f");
        Topopt_val::Youngs_Modulus = fea_0;

        ImGui::NewLine();

        ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
        static float fea_1 = 0.3f;
        ImGui::InputFloat("Poisson's Ratio ", &fea_1,0.0f,1.0f,"%.2f");
        Topopt_val::poisson = fea_1;
    
        ImGui::NewLine();
    }
    if(ImguiApp::thermal)
    {
        ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
        static float fea_t = 1.0f;
        ImGui::InputFloat("Conductivity ", &fea_t, 1.0f, 10.0f,"%.2f");
        Topopt_val::conductivity = fea_t;
    }
    if(!ImguiApp::fea_settings_set)
    {
        if(ImGui::Button("Done "))
        {
            ImguiApp::fea_settings_set = true;
        }
    }

    ImGui::End();
}


void ImguiApp::show_cg_solver_settings()
{
    ImGui::Begin("CG Solver",&ImguiApp::cg_solver_settings);
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);

    ImGui::NewLine();

    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
    static int cg_0 = 500;
    ImGui::InputInt("CG Iteration ", &cg_0, 1, 500);
    Topopt_val::iter = cg_0;

    ImGui::NewLine();

    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
    static float cg_1 = 0.01f;
    ImGui::InputFloat("End Residual ", &cg_1,0.0f,1.0f,"%.2f");
    Topopt_val::EndRes = cg_1;
   
    ImGui::NewLine();

    if(!ImguiApp::solver_settings_set)
    {
        if(ImGui::Button("Done "))
        {
            ImguiApp::solver_settings_set = true;
        }
    }

    ImGui::End();
}


void ImguiApp::show_optimisation_settings()
{
    ImGui::Begin("Optimisation Settings",&ImguiApp::optimisation_settings);
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);

    ImGui::NewLine();

    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
    static int opint_0 = (ImguiApp::thermal) ? 40 : 15;
    ImGui::InputInt("Maximum Iteration ", &opint_0,1,15);
    Topopt_val::MaxOptIter = opint_0;

    ImGui::NewLine();
    
    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
    static float op_0 = 3.0f;
    ImGui::InputFloat("Penality Exponent ", &op_0,2.0f,5.0f,"%.2f");
    Topopt_val::pexp = op_0;

    ImGui::NewLine();

    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
    static float op_1 = 0.3f;
    ImGui::InputFloat("Volume Fraction  ", &op_1,0.05f,0.95f,"%.2f");
    Topopt_val::VolumeFraction = op_1;


    ImGui::NewLine();

    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
    static float op_2 = (ImguiApp::thermal) ? 1.4 : 3;
    ImGui::InputFloat("Filter Radius  ", &op_2,1.1f,0.0f,"%.2f");
    if(op_2 > 3)
    {
        op_2 = 3.0f;
    }
    Topopt_val::FilterRadius = op_2;


    ImGui::NewLine();

    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
    static float op_3 = (ImguiApp::thermal) ? 0.001 : 0.01;
    ImGui::InputFloat("Minimum Density  ", &op_3,0.001f,0.0f,"%.3f");
    Topopt_val::MinDens = op_3;

    ImGui::NewLine();

    if(!ImguiApp::optimisation_settings_set)
    {
        if(ImGui::Button("Done "))
        {
            ImguiApp::optimisation_settings_set = true;
        }
    }

    ImGui::End();

}

void ImguiApp::show_unit_lattice_settings()
{
    ImGui::Begin("Unit Lattice Settings",&ImguiApp::unit_lattice_settings);
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);

    ImGui::NewLine();

    ImGui::SeparatorText("LATTICE TYPE");

    ImGui::NewLine();

    static ImGuiComboFlags flags_four = ImGuiComboFlags_WidthFitPreview;
    const char* itemsfour[] = { "Gyroid","Schwarz P","FRD","IWP","Type4","Schwarz D"};
    static int item_current_idxfour = 0; 

    const char* combo_preview_valuefour = itemsfour[item_current_idxfour];

    if (ImGui::BeginCombo("Select Lattice Type ", combo_preview_valuefour, flags_four))
    {
        for (int n = 0; n < IM_ARRAYSIZE(itemsfour); n++)
        {
            const bool is_selectedfour = (item_current_idxfour == n);
            if (ImGui::Selectable(itemsfour[n], is_selectedfour))
                item_current_idxfour = n;

            if (is_selectedfour)
                ImGui::SetItemDefaultFocus();
        }
        ImguiApp::lattice_index_type = item_current_idxfour;
        
        ImGui::EndCombo();
    }
    
    if(ImguiApp::lattice)
    {
        ImGui::NewLine();
        ImGui::SeparatorText("EXACT OR APPROXIMATED LATTICE");
        ImGui::NewLine();
        static int l_m = 0;
        ImGui::RadioButton("Approximation ", &l_m, 0); 
        ImGui::RadioButton("Real ", &l_m, 1);

        if(l_m == 1)
        {
            ImguiApp::real_unit_lattice = true;
            ImguiApp::approx_unit_lattice = false;
        }
        else
        {
            ImguiApp::approx_unit_lattice = true;
            ImguiApp::real_unit_lattice = false;
        }
        
        ImGui::NewLine();
        static bool iso_ubool = false;
        static bool iso1_ubool = false;
        static bool iso2_ubool = false;
        ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
        static float is_1 = 0.25f;
        ImGui::SliderFloat("IsoValue", &is_1,0.02, 0.98, "%.3f");
        iso_ubool = ImGui::IsItemActive();
        ImguiApp::bound_isoVal = is_1;
        ImGui::NewLine();

        ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
        static float is_2 = 0.20f;
    
        ImGui::SliderFloat("IsoRange -  ", &is_2,0.01f,is_1 - 0.01,"%.3f");
        iso1_ubool = ImGui::IsItemActive();
        is_2 = std::max(std::min(is_2,is_1 - 0.01f),0.01f);
        ImguiApp::bound_isoValone = is_2;

        ImGui::NewLine();
        ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
        static float is_3 = 0.30f;
        ImGui::SliderFloat("IsoRange +  ", &is_3,is_1 + 0.01 ,0.99f,"%.3f");
        iso2_ubool = ImGui::IsItemActive();
        is_3 = std::max(std::min(is_3,0.99f),is_1 + 0.01f);
        ImguiApp::bound_isoValtwo = is_3;

        if(iso1_ubool || iso2_ubool )
    {
            if( ImguiApp::show_unit_lattice_data)
            {
                ImguiApp::update_unit_isorange = true;
            }

    }


        if(ImGui::Button("View Unit Lattice") && !ImguiApp::view_unit_lattice_data )
        {
            if(ImguiApp::lattice_buffer_created)
            {
                ImguiApp::view_unit_lattice_data = true;
            }
            else
            {
                ImguiApp::debug_window = true;
            }
        }
    }
    ImGui::End();
}

void ImguiApp::show_spatial_angle_settings()
{
    ImGui::Begin("ANGLE", &spatial_angle_window ); 
    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);
    ImGui::SetWindowCollapsed(false,2);
    ImGui::Text("ANGLE PARAMETERS");

        static ImGuiComboFlags flags = ImGuiComboFlags_WidthFitPreview;

    const char* items[] = { "Normal","Bend","Round","Sinewave"};
    static int item_current_idx = 0; 

    const char* combo_preview_value = items[item_current_idx];

    if (ImGui::BeginCombo("Lattice Type", combo_preview_value, flags))
    {
        for (int n = 0; n < IM_ARRAYSIZE(items); n++)
        {
            const bool is_selected = (item_current_idx == n);
            if (ImGui::Selectable(items[n], is_selected))
                item_current_idx = n;
                ImguiApp::lattice_type_index = item_current_idx;
      
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        
        ImGui::EndCombo();
    }

 

    ImGui::NewLine();
        
    if(ImGui::Button("View Lattice") && !ImguiApp::view_lattice)
    {
        
        if(ImguiApp::lattice_buffer_created)
        {
            ImguiApp::view_lattice = true;
        }   
        else
        {
            ImguiApp::debug_window = true;
        }
    }

    ImGui::End();
}

void ImguiApp::show_spatial_period_settings()
{
    ImGui::Begin("PERIOD", &spatial_period_window ); 

    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);
    ImGui::SetWindowCollapsed(false,2);
    ImGui::SeparatorText("PERIOD PARAMETERS");
    ImGui::Text("Period over the Grid");

    static int e = 0;
    ImGui::RadioButton("Constant", &e, 0); 
    ImGui::RadioButton("Variable Axis", &e, 1); ImGui::SameLine();
    ImGui::RadioButton("Variable Grid", &e, 2);
    ImguiApp::period_type = e;
    
    if(e == 0)
    {
        ImGui::NewLine();
        ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
        static float fp_0 = 40.0f;
        ImGui::InputFloat("Period of Grating ", &fp_0, 0.1f, 100.0f, "%.2f");
        ImguiApp::period_of_grating = fp_0;

    }
    else if(e == 1)
    {
        ImGui::NewLine();
        ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
        static float fx_0 = 10.0f;
        ImGui::InputFloat("Period X AXIS ", &fx_0, 0.1f, 100.0f, "%.2f");
        ImguiApp::x_period = fx_0;

        ImGui::NewLine();
        ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
        static float fy_0 = 10.0f;
        ImGui::InputFloat("Period Y AXIS ", &fy_0, 0.1f, 100.0f, "%.2f");
        ImguiApp::y_period = fy_0;

        ImGui::NewLine();
        ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
        static float fz_0 = 10.0f;
        ImGui::InputFloat("Period Z AXIS ", &fz_0, 0.1f, 100.0f, "%.2f");
        ImguiApp::z_period = fz_0;


    }

    ImGui::NewLine();

    static bool iso_bool = false;
    static bool iso1_bool = false;
    static bool iso2_bool = false;

    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
    static float is_1 = 0.25f;
    ImGui::SliderFloat("IsoValue", &is_1,0.02, 0.98, "%.3f");
    iso_bool = ImGui::IsItemActive();
    ImguiApp::bound_isoVal = is_1;
    ImGui::NewLine();

    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
    static float is_2 = 0.20f;
  
    ImGui::SliderFloat("IsoRange -  ", &is_2,0.01f,is_1 - 0.01,"%.3f");
    iso1_bool = ImGui::IsItemActive();
    is_2 = std::max(std::min(is_2,is_1 - 0.01f),0.01f);
    ImguiApp::bound_isoValone = is_2;

    ImGui::NewLine();
    ImGui::SetNextItemWidth(ImguiApp::window_extent.x* 0.265);
    static float is_3 = 0.30f;
    ImGui::SliderFloat("IsoRange +  ", &is_3,is_1 + 0.01 ,0.99f,"%.3f");
    iso2_bool = ImGui::IsItemActive();
    is_3 = std::max(std::min(is_3,0.99f),is_1 + 0.01f);
    ImguiApp::bound_isoValtwo = is_3;

   if(iso1_bool || iso2_bool )
   {
        if( ImguiApp::show_lattice_data)
        {
            ImguiApp::update_isorange = true;
        }

   }


    ImGui::NewLine();
    
    if(ImGui::Button("View Lattice") && !ImguiApp::view_lattice)
    {
        ImguiApp::view_lattice = true;
    }

    ImGui::End();
}


void ImguiApp::show_export_settings()
{
    ImGui::Begin("Export Data", &export_settings ); 

    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(window_extent);
    ImGui::SetWindowCollapsed(false,2);
    ImGui::SeparatorText("Export Settings");

    ImGui::NewLine();

    static int ex = 0;

    ImGui::RadioButton("None ", &ex, 0); 

    ImGui::NewLine();

    ImGui::RadioButton("Primitive ", &ex, 1); 

    if(ex == 1)
    {
       if(ImguiApp::show_primitive_lattice || ImguiApp::show_model)
       {
            ImGui::NewLine();
            if(ImGui::Button("EXPORT PRIMITVE"))
            {
                ImguiApp::export_data_primitive = true;
                
            }
            ImGui::NewLine();
       }
       else
       {
            ImGui::Text("Generate Primitive Data ");
       }
    }

    ImGui::RadioButton("Optimise ", &ex, 2); 

    if(ex == 2)
    {
       if(ImguiApp::topo_done_lattice_do)
       {
            ImGui::NewLine();
            if(ImGui::Button("EXPORT OPTIMISE"))
            {
                ImguiApp::export_data_optimise = true;
                
                if(ImguiApp::show_topo_lattice)
                {
                    ImguiApp::topo_done_lattice_do = false;
                }

            }
            ImGui::NewLine();
       }
       else
       {
          
            ImGui::Text("Generate Optimise Data ");
          
       }
    }

    ImGui::RadioButton("Lattice ", &ex, 3);

    if(ex == 3)
    {
       if(ImguiApp::show_lattice_data)
       {
            ImGui::NewLine();
            if(ImGui::Button("EXPORT LATTICE"))
            {   
                ImguiApp::export_data_lattice = true;
            }
            ImGui::NewLine();
       }
       else
       {
            ImGui::Text("Generate Lattice Data ");
       }
    }






    ImGui::End();
}

void ImguiApp::show_debugging_window()
{
    ImGui::Begin("Debug Window", &debug_window); 

    ImGui::SetWindowPos(ImVec2(5,50));
    ImGui::SetWindowSize(ImVec2(350,200));
    ImGui::SetWindowCollapsed(false,2);
    ImGui::NewLine();
    ImGui::Text("Intialise Grid First");
    ImGui::NewLine();
    ImGui::Text("in 'Edit/Grid Settings' ");
    ImGui::NewLine();
    ImGui::End();
}

void ImguiApp::make_inactive(std::vector<bool*> window_bools, bool* active)
{
   
 
    for (auto it = window_bools.begin(); it != window_bools.end(); ++it) 
    {
        if(*it != active)
        {
          
            **it = false;
            
        }
     
    }
}

void ImguiApp::make_all_inactive(std::vector<bool*> window_bools)
{
   
 
    for (auto it = window_bools.begin(); it != window_bools.end(); ++it) 
    {
      
        **it = false;
     
    }
}


