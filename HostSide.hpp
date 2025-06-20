#pragma once
#include <vector>
#include <string>
#include <cstdint>

struct Frame {
    int  w = 0, h = 0;
    int  channels = 1;            // 1 (gray) или 3 (RGB)
    int  bitDepth = 8;            // 8 или 16

    std::vector<uint8_t > data8;  // если bitDepth == 8
    std::vector<uint16_t> data16; // если bitDepth == 16

    inline void* data() {
        return bitDepth == 8 ? (void*)data8.data()
            : (void*)data16.data();
    }
    inline const void* data() const {
        return bitDepth == 8 ? (const void*)data8.data()
            : (const void*)data16.data();
    }

    inline size_t byteSize() const {
        return size_t(w) * h * channels * (bitDepth / 8);
    }
};

bool loadTiff(const std::string& file, Frame& f);
bool saveTiff(const std::string& file, const Frame& f);

void  initGL(int w, int h);
void  updateFrame(const Frame& f);
bool  renderFrame();
