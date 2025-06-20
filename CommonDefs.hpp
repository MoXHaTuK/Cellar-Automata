#pragma once
#include <cstdint>
#include "cuda_runtime.h"

constexpr int LIGHTS_TICKS = 5;

#define FOR_EACH_DEPTH(OP)                       \
    if(bitDepth == 8)  { using DT = uint8_t ; OP } \
    else               { using DT = uint16_t; OP }

template<int R> struct TileDim { 
    static constexpr int pad = 2 * R; 
};

template <typename T>
__device__ __host__ constexpr T hd_min(T a, T b) { return (a < b) ? a : b; }

template <typename T>
__device__ __host__ constexpr T hd_max(T a, T b) { return (a > b) ? a : b; }

template<int R, typename T>
__device__ inline void loadTile(const T* __restrict__ gSrc, T* __restrict__ sTile, int W, int H)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;

    const int TILE_W = blockDim.x + 2 * R;
    const int TILE_H = blockDim.y + 2 * R;

    for (int dy = -R; dy <= R; ++dy)
    {
        int sy = threadIdx.y + R + dy;
        int gyClamped = min(max(gy + dy, 0), H - 1);

        for (int dx = -R; dx <= R; ++dx)
        {
            int sx = threadIdx.x + R + dx;
            int gxClamped = min(max(gx + dx, 0), W - 1);

            sTile[sy * TILE_W + sx] = gSrc[gyClamped * W + gxClamped];
        }
    }
    __syncthreads();
}
