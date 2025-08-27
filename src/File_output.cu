
#include "File_output.h"
#include <fstream>
#include <vector>
#include <map>

using namespace std;

void File_output::file_write(float4 *d_pos, uint totalVerts, const char *filename)
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
                mfile_latttice<<" f  "<<faces[i]<<" "<<faces[i+1]<<" "<<faces[i+2]<<"\n";
                    
            }
        }
    }

    mfile_latttice.close();
    free(latttice_data);                    
           
}


