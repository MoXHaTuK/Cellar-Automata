#include "Cell.hpp"
#include "ConvolveWindows.hpp"

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
__global__ void phaseSetLength(DT* in, DT* out, int W, int H)
{
    __shared__ DT tile[(16 + 1 * 2) * (16 + 1 * 2)];
    loadTile<1>(in, tile, W, H);
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= W || gy >= H) return;

    DT c = tile[(threadIdx.y + 1) * (blockDim.x + 2) + (threadIdx.x + 1)];
    if (c.length == 0 && c.speed != 0) c.length = 1;
    out[gy * W + gx] = c;
}

extern "C" void launchPhaseSetLen(uint8_t depth, void* dA, void* dB, int W, int H)
{
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    if (depth == 8)
        phaseSetLength<Cell8> << <grid, block >> > ((Cell8*)dA, (Cell8*)dB, W, H);
    else
        phaseSetLength<Cell16> << <grid, block >> > ((Cell16*)dA, (Cell16*)dB, W, H);
}
