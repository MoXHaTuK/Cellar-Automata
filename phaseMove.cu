#include "Cell.hpp"
#include "CommonDefs.hpp"

template<typename DT>
__global__ void phaseMove(DT* in, DT* out, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y * W + x;
    DT car = in[idx];

    out[idx] = car;

    if (car.speed == 0 || car.length == 0) return;

    int nx = x, ny = y;
    bool onCross = (car.lane == LANE_CROSS);

    int v = car.speed % 100;

    if (car.lane == LANE_WE || (onCross && car.speed < 100))
    {
        nx = min(x + v, W - 1);
    }
    else if (car.lane == LANE_SN || (onCross && car.speed >= 100))
    {
        ny = min(y + v, H - 1);
    }
    else
        return;

    int nidx = ny * W + nx;

    DT head = car;

    if (!onCross && head.lane == LANE_CROSS)
    {
        if (car.lane == LANE_SN) head.speed = 100 + v;
    }
    if (onCross && head.lane != LANE_CROSS)
    {
        if (head.speed >= 100) head.speed -= 100;
    }

    out[nidx] = head;

    out[idx].speed = 0;
    out[idx].length = 0;
}

extern "C" void launchPhaseMove(uint8_t depth, void* A, void* B, int W, int H)
{
    dim3 block(16, 16), grid((W + 15) / 16, (H + 15) / 16);
    if (depth == 8)
        phaseMove<Cell8 > << <grid, block >> > ((Cell8*)A, (Cell8*)B, W, H);
    else
        phaseMove<Cell16 > << <grid, block >> > ((Cell16*)A, (Cell16*)B, W, H);
}
