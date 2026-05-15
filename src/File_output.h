#pragma once
#ifndef _FILE_OUTPUT_H_
#define _FILE_OUTPUT_H_

#include <fstream>
#include <vector>
#include <map>

#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>

#include "Instances.h"

using namespace std;


// Structure to represent a 3D vertex
struct Vertex {
    float x, y, z;
    Vertex(float x=0, float y=0, float z=0) : x(x), y(y), z(z) {}
};
// Structure to represent a face of a 3D object
struct Face {
    int v1, v2, v3;
    Face(int v1, int v2, int v3) : v1(v1), v2(v2), v3(v3) {}
};


class File_output
{
    public:


    void file_write_obj(float4 *d_pos,uint totalVerts, const char* filename);

    void file_read_obj(const std::string& filename, IconData *pos,int face_count, int vertex_count, int normal_count);

    int3 get_vertex_size(const std::string& filename);
    

};


#endif //_FILE_OUTPUT_H_
