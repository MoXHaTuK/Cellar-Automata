#include "Cell.hpp"
#include "CommonDefs.hpp"

#define MAX_SPEED 3

constexpr int R = 1;

template<int R, typename DT>
__device__ void loadTile(DT* in, DT* tile, int W, int H)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    for (int dy = -R; dy <= R; ++dy)
    {
        int y = min(max(gy + dy, 0), H - 1);
        for (int dx = -R; dx <= R; ++dx)
        {
            int x = min(max(gx + dx, 0), W - 1);
            tile[(threadIdx.y + R + dy) * (blockDim.x + 2 * R) + (threadIdx.x + R + dx)] =
                in[y * W + x];
        }
    }
    __syncthreads();
}

template<typename DT>
__global__ void phaseSetSpeed(DT* in, DT* out, int W, int H)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= W || gy >= H) return;
    DT c = in[gy * W + gx];
    if (c.length)
    {
        int dir = (c.speed >= 100) ? 100 : 0;
        int v = c.speed % 100;
        int vmax = MAX_SPEED - (c.length - 1);
        v = hd_min<int>(v + 1, vmax);
        c.speed = dir + v;
    }
    if (c.speed > 0)
    {
        unsigned r = 1664525u * (gx + gy * W) + 1013904223u;
        if ((r >> 24) < 26) c.speed--;
    }
    out[gy * W + gx] = c;
}

extern "C" void launchPhaseSetSpd(uint8_t depth, void* dA, void* dB, int W, int H)
{
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    if (depth == 8)
        phaseSetSpeed<Cell8> << <grid, block >> > ((Cell8*)dA, (Cell8*)dB, W, H);
    else
        phaseSetSpeed<Cell16> << <grid, block >> > ((Cell16*)dA, (Cell16*)dB, W, H);
}
