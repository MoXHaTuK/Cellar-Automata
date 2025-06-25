#include "Cell.hpp"
#include "CommonDefs.hpp"

constexpr int R = 1;

template<typename DT>
__device__ __forceinline__
DT& tileAt(DT* tile, int x, int y, int pitch)
{
    return tile[y * pitch + x];
}

template<typename DT>
__global__ void phaseTurn(DT* in, DT* out, int W, int H)
{
    extern __shared__ unsigned char shmem[];
    DT* tile = (DT*)shmem;               

    const int bx = blockDim.x;
    const int by = blockDim.y;
    const int gx = blockIdx.x * bx + threadIdx.x;
    const int gy = blockIdx.y * by + threadIdx.y;
    const int pitch = bx + 2 * R;       

    loadTile<R>(in, tile, W, H);          
    __syncthreads();

    if (gx >= W || gy >= H) return;

    int lx = threadIdx.x + R;         
    int ly = threadIdx.y + R;
    DT c = tileAt(tile, lx, ly, pitch); 

    out[gy * W + gx] = c;

    if (c.lane != LANE_WE || c.speed == 0 || c.length == 0) return;

    if (gy == 0) return;


    DT above = tileAt(tile, lx, ly - 1, pitch);
    bool laneOK = (above.lane == LANE_WE);
    bool cellFree = (above.length == 0);

    unsigned rnd = 1103515245u * (gx + gy * W) + 12345u;
    bool doTurn = (rnd >> 28) < 1;      

    if (laneOK && cellFree && doTurn)
    {
        int upIdx = (gy - 1) * W + gx;

        DT dst = out[upIdx];
        dst.speed = c.speed % 100;
        dst.length = c.length;
        out[upIdx] = dst;

        out[gy * W + gx].speed = 0;
        out[gy * W + gx].length = 0;
    }
}

extern "C"
void launchPhaseTurn(uint8_t depth, void* A, void* B, int W, int H)
{
    dim3 block(16, 16), grid((W + 15) / 16, (H + 15) / 16);
    size_t shBytes = (16 + 2 * R) * (16 + 2 * R) * sizeof(Cell8);

    if (depth == 8)
        phaseTurn<Cell8 > << <grid, block, shBytes >> > ((Cell8*)A, (Cell8*)B, W, H);
    else
        phaseTurn<Cell16> << <grid, block, shBytes >> > ((Cell16*)A, (Cell16*)B, W, H);
}
