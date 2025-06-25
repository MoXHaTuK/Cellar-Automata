#include "HostSide.hpp"
#include "CommonDefs.hpp"
#include "Cell.hpp"
#include "PhaseLaunchers.hpp"
#include "RenderUtils.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>


int main()
{
    constexpr int  W = 512;
    constexpr int  H = 512;
    constexpr int  STEPS = 100;
    constexpr int  DELAY = 1000;
    const     uint8_t bitDepth = 8;

    size_t bytes = size_t(W) * H * (bitDepth == 8 ? sizeof(Cell8) : sizeof(Cell16));
    void* d_A, * d_B;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);

    launchInit(bitDepth, d_A, W, H, 12345);
    launchPhasePut(bitDepth, d_A, d_B, W, H);
    std::swap(d_A, d_B);

    std::vector<uint8_t>  host8;
    std::vector<uint16_t> host16;
    if (bitDepth == 8)  host8.resize(bytes);
    else                host16.resize(bytes / 2);

    Frame disp{ W, H, 4, bitDepth };
    if (bitDepth == 8)  disp.data8.resize(W * H * 4);
    else                disp.data16.resize(W * H * 4);

    initGL(W, H, disp.channels, disp.bitDepth);
    
    auto last = std::chrono::steady_clock::now();
    int  step = 0;

    while (renderFrame())
    {
        auto now = std::chrono::steady_clock::now();
        if (step < STEPS &&
            std::chrono::duration_cast<std::chrono::milliseconds>(now-last).count() >= DELAY)
        {
            launchPhaseSetLen(bitDepth, d_A, d_B, W, H); std::swap(d_A, d_B);
            launchPhaseSetSpd(bitDepth, d_A, d_B, W, H); std::swap(d_A, d_B);
            launchPhaseTurn(bitDepth, d_A, d_B, W, H); std::swap(d_A, d_B);
            launchPhaseLights(bitDepth, d_A, d_B, W, H); std::swap(d_A, d_B);
            launchPhaseMove(bitDepth, d_A, d_B, W, H); std::swap(d_A, d_B);

            void* hostPtr = (bitDepth == 8) ? static_cast<void*>(host8.data())
                : static_cast<void*>(host16.data());
            cudaMemcpy(hostPtr, d_A, bytes, cudaMemcpyDeviceToHost);

            if (bitDepth == 8)
            {
                const Cell8* src = reinterpret_cast<const Cell8*>(host8.data());
                buildRGBAFrame(src, disp, W, H);
            }
            else
            {
                const Cell16* src = reinterpret_cast<const Cell16*>(host16.data());
                buildRGBAFrame(src, disp, W, H);
            }
            updateFrame(disp);

            last = now;  
            ++step;
        }
    }

    const char* name[4] = { "lane","speed","length","tmp" };

    if (bitDepth == 8)
    {
        const Cell8* src = reinterpret_cast<const Cell8*>(host8.data());
        for (int ch = 0; ch < 4; ++ch)
        {
            Frame f{ W, H, 1, 8 }; f.data8.resize(W * H);
            for (size_t i = 0; i < size_t(W) * H; ++i)
            {
                const Cell8& c = src[i];
                f.data8[i] =
                    (ch == 0) ? uint8_t((c.lane + 2) * 64) :
                    (ch == 1) ? uint8_t(c.speed * 64) :
                    (ch == 2) ? uint8_t(c.length * 85) : c.tmp;
            }
            saveTiff(std::string("traffic_") + name[ch] + ".tiff", f);
        }
    }
    else
    {
        const Cell16* src = reinterpret_cast<const Cell16*>(host16.data());
        for (int ch = 0; ch < 4; ++ch)
        {
            Frame f{ W, H, 1, 16 }; f.data16.resize(W * H);
            for (size_t i = 0; i < size_t(W) * H; ++i)
            {
                const Cell16& c = src[i];
                f.data16[i] =
                    (ch == 0) ? static_cast<uint16_t>(c.lane) :
                    (ch == 1) ? static_cast<uint16_t>(c.speed) :
                    (ch == 2) ? static_cast<uint16_t>(c.length) : c.tmp;
            }
            saveTiff(std::string("traffic_") + name[ch] + ".tiff", f);
        }
    }

    cudaFree(d_A); 
    cudaFree(d_B);
    return 0;
}