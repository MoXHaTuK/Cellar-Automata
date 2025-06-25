#pragma once
#include "Cell.hpp"
#include "cuda_runtime.h"

template<int R, typename T>
struct __align__(16) Window
{
    __device__ const T& get(int dx, int dy) const {
        return buf[threadIdx.y + R + dy][threadIdx.x + R + dx];
    }
    T buf[blockDim.y + 2 * R][blockDim.x + 2 * R];
};
