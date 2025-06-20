#include "Cell.hpp"
#include "CommonDefs.hpp"

template<typename DT>
__global__ void phaseMove(DT* in, DT* out, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    DT c = in[idx];

    if (c.speed > 0) {
        int nx = min(x + c.speed, W - 1);
        int nidx = y * W + nx;
        out[nidx] = c;
        DT z{}; out[idx] = z;
    }
    else {
        out[idx] = c;
    }
}

extern "C" void launchPhaseMove(uint8_t depth, void* A, void* B, int W, int H)
{
    dim3 block(16, 16), grid((W + 15) / 16, (H + 15) / 16);
    if (depth == 8)  phaseMove<Cell8 > << <grid, block >> > ((Cell8*)A, (Cell8*)B, W, H);
    else          phaseMove<Cell16> << <grid, block >> > ((Cell16*)A, (Cell16*)B, W, H);
}
