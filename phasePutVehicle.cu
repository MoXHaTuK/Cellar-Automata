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
__global__ void phasePutVehicle(DT* in, DT* out, int W, int H)
{
    __shared__ DT tile[(16 + 2 * R) * (16 + 2 * R)];
    loadTile<R>(in, tile, W, H);

    curandStatePhilox4_32_10_t rng;
    curand_init(1234, blockIdx.y * gridDim.x + blockIdx.x, 0, &rng);
    ConvolveWindows<R, DT> win(tile, blockIdx.x, blockIdx.y, W, H, &rng);

    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= W || gy >= H) return;

    DT cell = win.get(0, 0);
    if (cell.lane == LANE_WE && cell.length == 0 && win.randf() > 0.9f) {
        cell.speed = 1;
        cell.length = 1;
    }
    out[gy * W + gx] = cell;
}

extern "C" void launchPhasePut(uint8_t depth, void* dA, void* dB, int W, int H)
{
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    if (depth == 8)
        phasePutVehicle<Cell8> << <grid, block >> > ((Cell8*)dA, (Cell8*)dB, W, H);
    else
        phasePutVehicle<Cell16> << <grid, block >> > ((Cell16*)dA, (Cell16*)dB, W, H);
}
