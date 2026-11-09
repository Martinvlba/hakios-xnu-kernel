/*
 * The IOKit half of drivers/core/hakios.stubs.c: what the rest of the kernel calls in a driver that is left out of it,
 * for the calls that are C++ (docs/drivers-config.md).
 */
#include "MiniIOSurface.h"

#if !defined(HAKIOS_ADRENO) && !defined(HAKIOS_AKITA_MALI)
/* the display's driver asks whether the GPU is done with a surface before it shows it: with no GPU nothing draws into one
 * (the Adreno's driver and the Pixel 8a's Mali service, hakios.akita_gpu.cpp, have their own) */
bool
MiniIOSurfaceDrawn(__unused MiniIOSurface * surface, __unused uint64_t milliseconds)
{
	return true;
}
#endif
