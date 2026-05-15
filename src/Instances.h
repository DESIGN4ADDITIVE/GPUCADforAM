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
    float val;

};


#endif //_INSTANCES_H_