#pragma once
#include "Cell.hpp"
#include "HostSide.hpp"

static inline uint8_t laneGray8(int8_t lane)
{
    switch (lane) {
    case LANE_IMP:   return  16;
    case LANE_WE:    return  80;
    case LANE_SN:    return 144;
    case LANE_CROSS: return 208;
    case LANE_TL:    return 255;
    default:         return   0;
    }
}

static inline uint16_t laneGray16(int16_t lane)
{
    switch (lane) {
    case LANE_IMP:   return  16u * 257u;
    case LANE_WE:    return  80u * 257u;
    case LANE_SN:    return 144u * 257u;
    case LANE_CROSS: return 208u * 257u;
    case LANE_TL:    return 65535u;
    default:         return     0u;
    }
}

static void buildRGBAFrame(const Cell8* src, Frame& out, int W, int H)
{
    constexpr float A = 0.70f;
    constexpr float B = 1.0f - A;

    for (int i = 0; i < W * H; ++i)
    {
        float r = laneGray8(src[i].lane);
        float g = r, b = r;

        if (src[i].length)
        {
            uint8_t lr = 0, lg = 0, lb = 0;

            bool vertical = (src[i].lane == LANE_SN);
            if (vertical)
            {
                lr = 200; lg = 0; lb = 200; // фиолетовый
            }
            else
            {
                switch (src[i].length) {
                case 1: lg = 255; break; // зелёный
                case 2: lb = 255; break; // синий
                default: lr = 255; break; // красный
                }
            }
            r = B * r + A * lr;
            g = B * g + A * lg;
            b = B * b + A * lb;
        }
        else if (src[i].speed)
        {
            uint8_t sr = 0, sg = 0, sb = 0;
            switch (src[i].speed) {
            case 0: sr = sg = sb = 60; break; // тёмно-серый
            case 1: sr = sg = 80; sb = 255; break; // голубой
            case 2: sr = 160; sg = 80; sb = 160; break; // лиловый
            default:sr = 255; sg = 160; sb = 40; break; // оранжевый
            }
            r = B * r + A * sr;
            g = B * g + A * sg;
            b = B * b + A * sb;
        }

        out.data8[i * 4 + 0] = static_cast<uint8_t>(r);
        out.data8[i * 4 + 1] = static_cast<uint8_t>(g);
        out.data8[i * 4 + 2] = static_cast<uint8_t>(b);
        out.data8[i * 4 + 3] = 255;
    }
}

static void buildRGBAFrame(const Cell16* src, Frame& out, int W, int H)
{
    constexpr float A = 0.70f;
    constexpr float B = 1.0f - A;

    for (int i = 0; i < W * H; ++i)
    {
        float r = laneGray16(src[i].lane);
        float g = r, b = r;

        if (src[i].length)
        {
            uint16_t lr = 0, lg = 0, lb = 0;

            bool vertical = (src[i].lane == LANE_SN);
            if (vertical)
            {
                lr = 32768u; lg = 0; lb = 32768u;
            }
            else
            {
                switch (src[i].length) {
                case 1: lg = 65535u; break;
                case 2: lb = 65535u; break;
                default: lr = 65535u; break;
                }
            }
            r = B * r + A * lr;
            g = B * g + A * lg;
            b = B * b + A * lb;
        }
        else if (src[i].speed)
        {
            uint16_t sr = 0, sg = 0, sb = 0;
            switch (src[i].speed) {
            case 0: sr = sg = sb = 15420u; break;
            case 1: sr = sg = 20560u; sb = 65535u; break;
            case 2: sr = 41120u; sg = 20560u; sb = 41120u; break;
            default:sr = 65535u; sg = 41120u; sb = 10280u; break;
            }
            r = B * r + A * sr;
            g = B * g + A * sg;
            b = B * b + A * sb;
        }

        out.data16[i * 4 + 0] = static_cast<uint16_t>(r);
        out.data16[i * 4 + 1] = static_cast<uint16_t>(g);
        out.data16[i * 4 + 2] = static_cast<uint16_t>(b);
        out.data16[i * 4 + 3] = 65535u;
    }
}
