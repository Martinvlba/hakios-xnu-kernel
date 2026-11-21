/*
 * Stand-in for iig output (xnu-12377's IOExtensiblePaniclog.iig; the iig shim copies this file by name). Only the two
 * DriverKit RPC dispatchers the class's vtables reference: user-space DriverKit is non-functional, as in OSObject.iig.cpp.
 */
#include <IOKit/IOReturn.h>
#include <mach/kern_return.h>

extern "C" {
kern_return_t __iig_paniclog_stub_0(void) __asm__("__ZN20IOExtensiblePaniclog8DispatchE5IORPC");
kern_return_t __iig_paniclog_stub_0(void) { return kIOReturnUnsupported; }
kern_return_t __iig_paniclog_stub_1(void) __asm__("__ZN20IOExtensiblePaniclog9MetaClass8DispatchE5IORPC");
kern_return_t __iig_paniclog_stub_1(void) { return kIOReturnUnsupported; }
}
