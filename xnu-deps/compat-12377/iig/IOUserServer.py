#!/usr/bin/env python3
# IOUserServer.py SDK-HEADER OUT: the macOS 26.1 SDK's generated IOUserServer.h with Panic() added, which
# xnu-12377.121's IOUserServer.iig has and the SDK predates. Shaped like Exit; its RPC id is made up (HakiOS runs
# no DriverKit extensions). Run by the iig stand-in (xnu-deps/host/bin/iig) when XNU_IIG_OVERRIDE names this folder.
import sys
s = open(sys.argv[1]).read()
def rep(a, b):
    global s
    if a not in s: sys.exit('IOUserServer.py: the SDK header is not the one expected (%r)' % a[:50])
    s = s.replace(a, b, 1)
rep('#include <DriverKit/IOService.h>  /* .iig include */\n',
    '#include <DriverKit/IOService.h>  /* .iig include */\n\n/* HakiOS: Panic() added (xnu-deps/compat-12377/iig/IOUserServer.py) */\n#define kIOUserServrMaxPanicReasonLength    1024\n')
rep('#define IOUserServer_Exit_ID            0xe949d58832ebe980ULL\n',
    '#define IOUserServer_Exit_ID            0xe949d58832ebe980ULL\n#define IOUserServer_Panic_ID            0x48616b694f530001ULL\n')
rep('#define IOUserServer_Exit_Args \\\n        const char * reason\n',
    '#define IOUserServer_Exit_Args \\\n        const char * reason\n\n#define IOUserServer_Panic_Args \\\n        const char * reason\n')
rep('    kern_return_t\\\n    Exit(\\\n        const char * reason,\\\n        OSDispatchMethod supermethod = NULL);\\\n',
    '    kern_return_t\\\n    Exit(\\\n        const char * reason,\\\n        OSDispatchMethod supermethod = NULL);\\\n\\\n    kern_return_t\\\n    Panic(\\\n        const char * reason,\\\n        OSDispatchMethod supermethod = NULL);\\\n')
rep('    kern_return_t\\\n    Exit_Impl(IOUserServer_Exit_Args);\\\n',
    '    kern_return_t\\\n    Exit_Impl(IOUserServer_Exit_Args);\\\n\\\n    kern_return_t\\\n    Panic_Impl(IOUserServer_Panic_Args);\\\n')
rep('    Exit_Invoke(const IORPC rpc,\\\n        OSMetaClassBase * target,\\\n        Exit_Handler func);\\\n',
    '    Exit_Invoke(const IORPC rpc,\\\n        OSMetaClassBase * target,\\\n        Exit_Handler func);\\\n\\\n    typedef kern_return_t (*Panic_Handler)(OSMetaClassBase * target, IOUserServer_Panic_Args);\\\n    static kern_return_t\\\n    Panic_Invoke(const IORPC rpc,\\\n        OSMetaClassBase * target,\\\n        Panic_Handler func);\\\n')
open(sys.argv[2], 'w').write(s)
