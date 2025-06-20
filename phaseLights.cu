#include "Cell.hpp"
#include "CommonDefs.hpp"

template<typename DT>
__global__ void phaseLights(DT* in, DT* out, int W, int H)
{
    int idx = (blockIdx.y * blockDim.y + threadIdx.y) * W +
        (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= W * H) return;

    DT c = in[idx];

    /* светофор хранит Ђspeedї как код состо€ни€:
       0-99  Ц зелЄный дл€ WE,   t = 99-counter
       100-199 Ц зелЄный дл€ SN, t = 199-counter
       201 / 202 Ц жЄлтый */

    if (c.lane != LANE_TL) { out[idx] = c; return; }

    int state = c.speed;
    if (state == 201 || state == 202) {                 // жЄлтый
        state = (state == 201) ? 100 + LIGHTS_TICKS : LIGHTS_TICKS;
    }
    else if (state == 0 || state == 100) {          // момент переключени€
        state = (state == 0) ? 201 : 202;
    }
    else {
        state--;
    }
    c.speed = state;
    out[idx] = c;
}

extern "C" void launchPhaseLights(uint8_t depth, void* A, void* B, int W, int H)
{
    dim3 block(16, 16), grid((W + 15) / 16, (H + 15) / 16);
    if (depth == 8)
        phaseLights<Cell8> << <grid, block >> > ((Cell8*)A, (Cell8*)B, W, H);
    else
        phaseLights<Cell16> << <grid, block >> > ((Cell16*)A, (Cell16*)B, W, H);
}
