#pragma once
#include "Cell.hpp"
#include <curand_kernel.h>

template<int R> struct TileDim { static constexpr int pad = 2 * R; };

template<int R, typename T>
class ConvolveWindows
{
public:
    __device__ explicit ConvolveWindows(T* sharedTile,
        int bx, int by, int W, int H,
        curandStatePhilox4_32_10_t* rng)
        : buf(sharedTile), blockX(bx), blockY(by), W(W), H(H), rng(rng) {
    }

    /* чтение с безопасными границами */
    __device__ T* getPtr(int dx, int dy) const
    {
        int lx = threadIdx.x + R + dx;
        int ly = threadIdx.y + R + dy;
        return &buf[ly * (blockDim.x + TileDim<R>::pad) + lx];
    }
    __device__ const T& get(int dx, int dy) const { return *getPtr(dx, dy); }

    /* поворот (0∞ или 90∞) Ч просто мен€ем dx,dy местами */
    template<int DEG>
    __device__ const T& getRot(int dx, int dy) const
    {
        if constexpr (DEG == 0)   return get(dx, dy);
        else                    return get(dy, -dx);     // 90∞ по часовой
    }

    /* RNG */
    __device__ float randf() const { return curand_uniform(rng); }

private:
    T* buf;
    int  blockX, blockY, W, H;
    curandStatePhilox4_32_10_t* rng;
};
