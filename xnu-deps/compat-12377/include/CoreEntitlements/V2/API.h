/*
 * HakiOS compat: CoreEntitlements "V2", which xnu-12377 includes but Apple does not publish.
 * Only the names the kernel's sources use, shaped by their call sites. HakiOS registers no
 * CoreEntitlements implementation (AMFI is ours, libkern/amfi/miniamfi.c), so libCoreEntitlements
 * stays NULL and none of these functions is ever called; they only have to compile.
 */
#ifndef HAKIOS_COMPAT_CE_V2_API_H
#define HAKIOS_COMPAT_CE_V2_API_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <CoreEntitlements/CoreEntitlements.h>

typedef enum {
	kCEReturnSuccess = 0,
	kCEReturnError = 1,
	kCEReturnV1NotEligibleForAcceleration = 2,
} CEReturn_t;

typedef enum {
	kCEContextTypeUnknown = 0,
	kCEContextTypeEntitlements = 1,
	kCEContextTypeProvisioningProfile = 2,
	kCEContextTypeProvisioningProfileEntitlements = 3,
} CEContextType_t;

/* opaque storage, large enough for what the kernel embeds by value */
typedef struct { uint64_t opaque[16]; } CEContext_t;
typedef struct { uint64_t opaque[8]; } CEElement_t;
typedef struct { const uint8_t *data; size_t length; } CEBuffer_t;
#endif
