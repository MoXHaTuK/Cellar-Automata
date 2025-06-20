#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

	void launchInit(uint8_t depth, void* dBuf, int W, int H, unsigned seed);
	void launchPhasePut(uint8_t depth, void* in, void* out, int W, int H);
	void launchPhaseSetLen(uint8_t depth, void* in, void* out, int W, int H);
	void launchPhaseSetSpd(uint8_t depth, void* in, void* out, int W, int H);
	void launchPhaseLights(uint8_t depth, void* in, void* out, int W, int H);
	void launchPhaseTurn(uint8_t depth, void* in, void* out, int W, int H);
	void launchPhaseMove(uint8_t depth, void* in, void* out, int W, int H);

#ifdef __cplusplus
}
#endif
