#include "Cell.hpp"
#include "CommonDefs.hpp"

template<typename DT>
__global__ void phaseLights(DT* in, DT* out, int W, int H)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gy * W + gx;
    if (gx >= W || gy >= H) return;

    DT c = in[idx];
    out[idx] = c;

    if (c.lane != LANE_TL)
        return;

    int state = c.speed % 200;
    if (state == 201 || state == 202) {
        state = (state == 201) ? 100 + LIGHTS_TICKS : LIGHTS_TICKS;
    }
    else if (state == 0 || state == 100) {
        state = (state == 0) ? 201 : 202;
    }
    else {
        state--;
    }
    out[idx].speed = state;
}

extern "C" void launchPhaseLights(uint8_t depth, void* A, void* B, int W, int H)
{
    dim3 block(16, 16), grid((W + 15) / 16, (H + 15) / 16);
    if (depth == 8)
        phaseLights<Cell8> << <grid, block >> > ((Cell8*)A, (Cell8*)B, W, H);
    else
        phaseLights<Cell16> << <grid, block >> > ((Cell16*)A, (Cell16*)B, W, H);
}
