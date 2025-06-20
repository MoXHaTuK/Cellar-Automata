#include "Cell.hpp"
#include "CommonDefs.hpp"

constexpr int R = 1;

template<typename DT>
__global__ void phaseTurn(DT* in, DT* out, int W, int H)
{
    __shared__ DT tile[(16 + 2 * R) * (16 + 2 * R)];
    loadTile<R>(in, tile, W, H);

    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= W || gy >= H) return;

    DT c = tile[(threadIdx.y + R) * (blockDim.x + 2 * R) + (threadIdx.x + R)];
    if (c.lane == LANE_WE && c.speed > 0 && tile[(threadIdx.y + R - 1) * (blockDim.x + 2 * R) + (threadIdx.x + R)].lane == LANE_WE) {
        
        if ((threadIdx.x + threadIdx.y + clock()) % 10 == 0) {
            int upIdx = (gy - 1) * W + gx;
            out[upIdx] = c;                // перенос в соседнюю полосу
            c.length = 0; c.speed = 0;
        }
    }
    out[gy * W + gx] = c;
}

extern "C" void launchPhaseTurn(uint8_t depth, void* A, void* B, int W, int H)
{
    dim3 block(16, 16), grid((W + 15) / 16, (H + 15) / 16);
    if (depth == 8)  phaseTurn<Cell8 > << <grid, block >> > ((Cell8*)A, (Cell8*)B, W, H);
    else          phaseTurn<Cell16> << <grid, block >> > ((Cell16*)A, (Cell16*)B, W, H);
}
