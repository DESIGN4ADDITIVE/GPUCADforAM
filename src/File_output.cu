
#include "File_output.h"


void File_output::file_write_obj(float4 *d_pos, uint totalVerts, const char *filename)
{
     
    float4 *latttice_data;
    latttice_data = (float4 *)malloc((totalVerts) * sizeof(*d_pos));
    cudaMemcpy(latttice_data, d_pos, (totalVerts) * sizeof(*d_pos), cudaMemcpyDeviceToHost);

    ofstream mfile_latttice ;
    mfile_latttice.open(filename,ios::out);

    mfile_latttice<<"##Sample latttice new Obj \n";
    mfile_latttice<<"o Solid \n";
    // ///////////////////////////////////////////////////////////////////////////
    std::vector<float> vec_flot;
    typedef std::map<std::vector<float>, int> VectorMap;
    VectorMap vertex_check;
    std::vector<uint> faces;
    int index = 0;
    for (int i=0;i<totalVerts;i++)
    {
    
        float v_x = int(latttice_data[i].x * 1000) * 0.001;
        float v_y = int(latttice_data[i].y * 1000) * 0.001;
        float v_z = int(latttice_data[i].z * 1000) * 0.001;
        
        vec_flot =  {v_x,v_y,v_z};

        if (vertex_check.count(vec_flot) == 0)
        {
            
            
            index++;
            vertex_check[vec_flot] = index;
            faces.push_back(index);

            mfile_latttice<<"v "<< v_x <<" "<< v_y <<" "<< v_z <<"\n";
            
        }   
        else
        {
            
            faces.push_back(vertex_check[vec_flot]);
            
        }
    
    }

    mfile_latttice<<"\n";

    std::vector<uint> face_flot;
    typedef std::map<std::vector<uint>, int> FaceMap;
    FaceMap face_check;

    mfile_latttice<<"\n";
    
    for (int i=0;i<faces.size();i=i+3)
    {
            
        if((faces[i] != faces[i+1]) && (faces[i] != faces[i+2]) && (faces[i+1] != faces[i+2] ))
        {
            face_flot =  {faces[i],faces[i+1],faces[i+2]};
                
                
            if(face_check.count(face_flot) == 0)
            {
                face_check[face_flot] = 1;
                ///// Orientation Order is different - issue with Marching cube table and coordinate system used in the application /////
                mfile_latttice<<" f  "<<faces[i]<<" "<<faces[i + 2]<<" "<<faces[i + 1]<<"\n";
               
            }
        }
    }

    mfile_latttice.close();
    free(latttice_data);                    
           
}



void File_output::file_read_obj(const std::string& filename, IconData *loadicon, int face_count, int vertex_count, int normal_count)
{

    std::string s = filename;
    size_t len = s.length();
    std::string extension = s.substr(len - 4,len);

    printf("extension is %s \n",extension.c_str());

    float3 *h_vertex, *h_normals;
    h_vertex = (float3 *)malloc(vertex_count * sizeof(float3));
    h_normals = (float3 *)malloc(normal_count * sizeof(float3));

    IconData *h_icon;
    h_icon = (IconData *)malloc(face_count * 3 *  sizeof(IconData));

    if(extension == ".obj")
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            
        }
        std::string line;

        std::string delims = "//"; // list of delimiters

        int count = 0;
        int count_one = 0;
        int count_two = 0;

        while (std::getline(file, line)) 
        {
            std::istringstream iss(line);

            std::string type;
            iss >> type;
            if (type == "v") {

                float x, y, z;

                iss >> x >> y >> z;

                h_vertex[count].x = x;
                h_vertex[count].y = y;
                h_vertex[count].z = z;

                count++;

            } 

            if (type == "vn") {

                float x1, y1, z1;

                iss >> x1 >> y1 >> z1;

                h_normals[count_one].x = x1;
                h_normals[count_one].y = y1;
                h_normals[count_one].z = z1;

                count_one++;

            } 
            
            if (type == "f") 
            {
                // Replace all delimiters with a space
                for (char &c : line) {
                    if (delims.find(c) != std::string::npos) 
                    {
                        c = ' ';
                    }
                }

                std::istringstream iss_e(line);
                int v1, v2, v3, v4 , v5 , v6;
                char ch;
                iss_e >> ch >> v1 >> v2 >> v3 >> v4 >> v5 >> v6 ;

                int buffer_count = 3*count_two;

                h_icon[buffer_count].pos = h_vertex[v1 - 1];
                h_icon[buffer_count + 1].pos = h_vertex[v3 - 1];
                h_icon[buffer_count + 2].pos = h_vertex[v5 - 1];

                h_icon[buffer_count].normal = h_normals[v2 - 1];
                h_icon[buffer_count + 1].normal = h_normals[v4 - 1];
                h_icon[buffer_count + 2].normal = h_normals[v6 - 1];

                count_two++;
            }
        }

    }
    else 
    {

         std::cerr << "Failed to load model." << std::endl;
  
    }


    cudaMemcpy(loadicon,h_icon,face_count * 3 * sizeof(IconData),cudaMemcpyHostToDevice);
    
    free(h_vertex);
    free(h_normals);
    free(h_icon);

}

int3 File_output::get_vertex_size(const std::string& filename)
{
    std::ifstream in_file(filename,std::ifstream::in | std::ifstream::binary);

    in_file.seekg(0);

    std::string line;

    int count_f = 0;

    int count_v = 0;

    int count_n = 0;

    while (std::getline(in_file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        if(type == "v")
        {
            count_v++;
        }
        if(type == "vn")
        {
            count_n++;
        }
        if(type == "f")
        {
            count_f++;
        }

    }

    printf("Total faces %d count , Total vertices are %d and data size is %lu \n",count_f,count_f*3,count_f*3*sizeof(float3));

    printf("Vertex data count  %d \n",count_v);

    printf("Normal data count  %d \n",count_n);

    return {count_f,count_v,count_n};

}


