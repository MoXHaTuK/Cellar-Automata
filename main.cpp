#include "HostSide.hpp"
#include "CommonDefs.hpp"
#include "Cell.hpp"
#include "PhaseLaunchers.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

int main()
{
    constexpr int  W = 512, H = 512;
    constexpr int  STEPS = 10;
    const     uint8_t bitDepth = 8;            // 8-битна€ симул€ци€

    size_t bytes = size_t(W) * H *
        (bitDepth == 8 ? sizeof(Cell8) : sizeof(Cell16));

    void* d_A, * d_B;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);

    launchInit(bitDepth, d_A, W, H, 12345);
    auto swapAB = [&] { std::swap(d_A, d_B); };

    for (int s = 0; s < STEPS; ++s) {
        launchPhasePut(bitDepth, d_A, d_B, W, H); swapAB();
        launchPhaseSetLen(bitDepth, d_A, d_B, W, H); swapAB();
        launchPhaseSetSpd(bitDepth, d_A, d_B, W, H); swapAB();
        launchPhaseLights(bitDepth, d_A, d_B, W, H); swapAB();
        launchPhaseTurn(bitDepth, d_A, d_B, W, H); swapAB();
        launchPhaseMove(bitDepth, d_A, d_B, W, H); swapAB();
    }

    std::vector<Cell8>  buf8;
    std::vector<Cell16> buf16;

    if (bitDepth == 8)  buf8.resize(bytes / sizeof(Cell8));
    else                buf16.resize(bytes / sizeof(Cell16));

    void* hostPtr = (bitDepth == 8) ? static_cast<void*>(buf8.data())
        : static_cast<void*>(buf16.data());

    cudaMemcpy(hostPtr, d_A, bytes, cudaMemcpyDeviceToHost);

    uint8_t minLane = 255, maxLane = 0;
    for (const Cell8& c : buf8) {
        minLane = std::min(minLane, (uint8_t)c.lane);
        maxLane = std::max(maxLane, (uint8_t)c.lane);
    }
    std::cout << "lane range = [" << +minLane << ", " << +maxLane << "]\n";

    const char* suffix[4] = { "lane", "speed", "length", "tmp" };

    Frame laneFrame;

    for (int ch = 0; ch < 4; ++ch)
    {
        Frame f;  f.w = W; f.h = H; f.channels = 1; f.bitDepth = bitDepth;

        if (bitDepth == 8)  f.data8.resize(W * H);
        else                f.data16.resize(W * H);

        for (size_t i = 0; i < size_t(W) * H; ++i)
        {
            if (bitDepth == 8) {
                const Cell8& c = buf8[i];
                uint8_t val =
                    (ch == 0) ? (c.lane+2)*64 :
                    (ch == 1) ? c.speed*64 :
                    (ch == 2) ? c.length*85 : c.tmp;
                f.data8[i] = val;
            }
            else {
                const Cell16& c = buf16[i];
                uint16_t val =
                    (ch == 0) ? c.lane :
                    (ch == 1) ? c.speed :
                    (ch == 2) ? c.length : c.tmp;
                f.data16[i] = val;
            }
        }

        std::string name = "traffic_" + std::string(suffix[ch]) + ".tiff";
        if (!saveTiff(name, f))
            std::cerr << "Failed to save " << name << '\n';
        else
            std::cout << "Saved " << name << '\n';

        if (ch == 0)
            laneFrame = f;
    }
    initGL(W, H);
    updateFrame(laneFrame);
    while (renderFrame()) { /* loop */ }

    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
