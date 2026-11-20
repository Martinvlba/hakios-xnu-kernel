/* HakiOS compat: <CodeSignature/Entitlements.h>, included by xnu-12377 but not published.
 * Names are added here as the build asks for them; the strings are the ones iOS 27's own
 * binaries carry (found in its dyld shared cache). */
#ifndef HAKIOS_COMPAT_CODESIGNATURE_ENTITLEMENTS_H
#define HAKIOS_COMPAT_CODESIGNATURE_ENTITLEMENTS_H
#define kCSWebBrowserHostEntitlement       "com.apple.developer.web-browser-engine.host"
#define kCSWebBrowserGPUEntitlement        "com.apple.developer.web-browser-engine.rendering"
#define kCSWebBrowserNetworkEntitlement    "com.apple.developer.web-browser-engine.networking"
#define kCSWebBrowserWebContentEntitlement "com.apple.developer.web-browser-engine.webcontent"
#endif
