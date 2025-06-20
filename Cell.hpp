#pragma once
#include <cstdint>

struct Cell8 { int8_t  lane, speed, length, tmp; };
struct Cell16 { int16_t lane, speed, length, tmp; };

enum : int8_t {
    LANE_IMP = -1, LANE_WE = 0, LANE_SN = 1,
    LANE_CROSS = 2, LANE_TL = 3
};
