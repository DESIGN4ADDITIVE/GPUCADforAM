#pragma once
#ifndef _INSTANCES_H_
#define _INSTANCES_H_


#include "vector" 
#include "linmath.h"

struct IconData {
    float3 pos;
    float3 normal;
};

struct InstanceData {
    float3 pos;
    float3 normal;
    float3 load_dir;
    float val;
    int load_id;
};



#endif //_INSTANCES_H_