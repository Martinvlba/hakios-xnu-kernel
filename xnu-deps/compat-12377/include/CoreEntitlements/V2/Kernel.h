/* HakiOS compat: the kernel's table of CoreEntitlements V2 functions (see API.h). */
#ifndef HAKIOS_COMPAT_CE_V2_KERNEL_H
#define HAKIOS_COMPAT_CE_V2_KERNEL_H
#include <CoreEntitlements/V2/API.h>

typedef struct {
	uint64_t version;
	CEReturn_t (*contextValueForKey)(const CEContext_t *ctx, const char *key, CEElement_t *out);
	CEReturn_t (*elementMatchBool)(const CEElement_t *element, bool value);
	CEReturn_t (*elementGetCEBuffer)(const CEElement_t *element, CEBuffer_t *out);
	CEReturn_t (*contextInitWithTypeLegacy)(const uint8_t *der, size_t length, CEContextType_t type,
	    CEContext_t *storage, ...);
	CEReturn_t (*contextGetLegacyContext)(const CEContext_t *ctx, CEQueryContext_t *out);
	CEReturn_t (*contextCheckAcceleration)(const CEContext_t *ctx);
	CEReturn_t (*contextEvaluateAcceleration)(const CEContext_t *ctx, size_t *length);
	CEReturn_t (*contextAccelerate)(const CEContext_t *ctx);
} CEKernelAPI_t;
#endif
