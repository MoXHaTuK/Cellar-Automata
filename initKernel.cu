#include "Cell.hpp"
#include "CommonDefs.hpp"
#include <curand_kernel.h>

constexpr int BLOCK_IMP = 15;
constexpr int ROAD_LANES = 3;
constexpr int STEP = BLOCK_IMP + ROAD_LANES;

template<typename DT>
__global__ void initKernel(DT* g, int W, int H, unsigned seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int localX = x % STEP;
    int localY = y % STEP;

    bool onVerRoad = localX < ROAD_LANES; 
    bool onHorRoad = localY < ROAD_LANES;
    bool isCross = onVerRoad && onHorRoad;

    DT c{};

    if (isCross)
        c.lane = LANE_CROSS;
    else if (onVerRoad)
        c.lane = LANE_SN;
    else if (onHorRoad)
        c.lane = LANE_WE;
    else
        c.lane = LANE_IMP;

    if (!isCross)
    {
        bool northTL = onVerRoad && (localY == ROAD_LANES - 1);
        bool westTL = onHorRoad && (localX == ROAD_LANES - 1);
        if (northTL || westTL) c.lane = LANE_TL;
    }

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, y * W + x, 0, &rng);

    if (c.lane == LANE_WE || c.lane == LANE_SN) {
        if (curand_uniform(&rng) > 0.97f) {   // ~3 %
            c.speed = curand(&rng) % 4;      // 0…3
            c.length = 1;
        }
    }

    g[y * W + x] = c;
}

extern "C"
void launchInit(uint8_t depth, void* dBuf, int W, int H, unsigned seed)
{
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    if (depth == 8)
        initKernel<Cell8 > << <grid, block >> > ((Cell8*)dBuf, W, H, seed);
    else
        initKernel<Cell16> << <grid, block >> > ((Cell16*)dBuf, W, H, seed);
}
