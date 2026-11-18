/*
 * libhakios.media.dylib: code of ours inside mediaserverd (see inject.c for how it gets there).
 *
 * mediaserverd is the audio server, and also the camera's: the capture stack of CMCapture lives
 * in the same process. Soon after it starts, that stack "prewarms" the Core Image filters of the
 * portrait mode, for which Core Image wants an OpenGL ES context; there is no GPU, the context
 * has no functions behind it, and the daemon jumps to address 0 -- every time launchd restarts
 * it. Nobody is going to take a portrait here: the prewarming is skipped.
 *
 * What the daemon is wanted for meanwhile is neither sound nor the camera: it is where
 * VideoToolbox decodes for everybody else (the service com.apple.coremedia.decompressionsession),
 * and a HEIC picture is HEVC inside. Without the daemon ImageIO has "no way to decode a HEIC",
 * and the stock wallpaper of Darwin userland 16, whose layers are four such pictures, is its gradient alone.
 * There is no hardware decoder here; VideoToolbox takes its software one (VCPHEVC.videocodec).
 *
 * The flashlight (only where the kernel has one: torch=3, /dev/hakios-torch): Control Center's
 * button and AVFlashlight reach FigFlashlight in here, which asks the capture stack's device vendor
 * (+[BWFigCaptureDeviceVendor sharedCaptureDeviceVendor]) for the camera device and its back
 * camera's stream, and sets the stream's "TorchLevel" (0 to 1). Without a camera there is no
 * vendor (it is nil), every message to it does nothing, and FigFlashlight says it has set the level
 * when nothing has been. So where there is no vendor, FigFlashlight -- and only FigFlashlight,
 * by where it calls from -- is given one of ours, with a device and a stream whose torch level goes
 * to the kernel (see flashlight_*() below). Everybody else still gets none, as before.
 *
 * The camera (only where the kernel has one: cam=5, /dev/hakios-camera): the vendor is made for
 * everybody, with the create function of our capture device (camera_device.c) in place of the
 * one H10ISP.mediacapture would give for the iPhone's ISP; the capture graph is built around it and
 * its frames reach Camera.app's preview (see camera_device.c). Its stream takes the torch level too,
 * so FigFlashlight would need nothing of its own then (not tried yet).
 *
 * What the capture stack asks of hardware this phone has none of is refused here: there is no ISP
 * processing session, so there can be no video stabilization (see video_stabilization_method below).
 *
 * CMCapture is not loaded yet when this library is: the methods are replaced when its image
 * arrives.
 */
#include <CoreFoundation/CoreFoundation.h>
#include <CoreMedia/CoreMedia.h>
#include <CoreVideo/CoreVideo.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <mach/mach_time.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/sysctl.h>
#include <unistd.h>

static void
say(const char *format, ...)
{
	char line[256];
	va_list ap;
	int n;

	va_start(ap, format);
	n = vsnprintf(line, sizeof(line), format, ap);
	va_end(ap);
	if (n > (int)sizeof(line) - 1) {
		n = sizeof(line) - 1;
	}
	write(2, line, (size_t)n);
}

extern const void *_dyld_get_shared_cache_range(size_t *length);

#define CACHE_BASE 0x180000000UL                /* where the cache was built to be, before the slide */
/* an address of the shared cache as each build this is made for has it: Darwin userland 16.5 (20F66), 16.7.16 (20H392:
 * the same symbols, found by name in the cache's local symbols -- tools/dsc-symbols.py). ios_build.h is written by
 * build.sh from sysimage/build.conf. */
#include "ios_build.h"
#if defined(HAKIOS_IOS_20G81)
#define AT(f66, h392, g81) (g81)
#elif defined(HAKIOS_IOS_20H392)
#define AT(f66, h392, g81) (h392)
#else
#define AT(f66, h392, g81) (f66)
#endif

static uintptr_t slide;
#if HAKIOS_CAMERA_DEBUG
static void patch_signal_error(void);
static void hook_session(void);
#endif

/* +[BWCoreImageFilterRenderer prewarmPortraitFilterVersion:semanticStyleFilters:] */
static void
prewarm_portrait_filter(__unused id self, __unused SEL selector, __unused int version, __unused BOOL styles)
{
}

/*
 * Two more portrait prewarms (11-17, Darwin userland 16.6.1 / 20G81 -- the iPhone 13 mini's, d16ap, whose camera does
 * portrait VIDEO: the iPhone X's never reached these). FigPreloadShaders, once the daemon is up, builds the depth effect's
 * Metal pipelines ahead of use: Portrait.framework's disparity post-processing (+[PTDisparityPostProcessing
 * prewarmForMediaserverd]) and the SDOF renderer, the principal class of SDOFRenderingV5.bundle, FigSDOFRendering, whose
 * -init makes CCPortrait's kernels. Both need what this Metal does not give yet -- the compute pipelines' REFLECTION
 * (CCPortrait asserts self.arguments in -[UniKernelInternal initWithName:...]), argument buffers, imageblocks -- and an
 * assert there is an abort of the whole daemon: launchd restarts it every ~5 s, and with it go the sound and the
 * vibration. Nobody takes a portrait video here; the prewarming is skipped, as the portrait filters' above is.
 */
static void
prewarm_disparity(__unused id self, __unused SEL selector)
{
}

/* the SDOF renderer's -init: none (the caller takes nil as no renderer); see above */
static id
sdof_init(id self, __unused SEL selector)
{
	((void (*)(id, SEL))objc_msgSend)(self, sel_registerName("release"));
	return nil;
}

/* ---- the flashlight ------------------------------------------------------------------------------- */

static int torch = -1;                  /* /dev/hakios-torch */
static bool camera;                     /* /dev/hakios-camera is there */
extern int hakios_camera_device_create(const void *allocator, const void *options, void **out);
static Class vendor_class, device_class, stream_class;
static IMP original_shared_vendor;
static unsigned torch_said;

/*
 * FigFlashlight's functions that ask for the vendor, unslid (20F66): flashlight_IsAvailable,
 * _NotifyForCurrentState and _setupDevice lie together, _teardownDevice apart.
 */
static const struct { uintptr_t from, to; } flashlight_code[] = {
	{ AT(0x1a1fb0830UL, 0x1a20007b0UL, 0x1a3e04c54UL), AT(0x1a1fb0c5cUL, 0x1a2000bdcUL, 0x1a3e050acUL) }, { AT(0x1a1eee5c0UL, 0x1a1f3e540UL, 0x1a3d37df0UL), AT(0x1a1eee6e8UL, 0x1a1f3e668UL, 0x1a3d37f3cUL) },
};

/* +[BWFigCaptureDeviceVendor sharedCaptureDeviceVendor] */
static id
shared_vendor(id self, SEL selector)
{
	uintptr_t caller = (uintptr_t)__builtin_return_address(0) - slide;
	id vendor = ((id (*)(id, SEL))original_shared_vendor)(self, selector);
	static id ours, camera_vendor;
	static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

	if (vendor != NULL) {
		return vendor;
	}
	if (!camera && access("/dev/hakios-camera", F_OK) == 0) {
		camera = true;          /* (should the kernel have made it after mediaserverd started) */
		say("hakios: mediaserverd: the kernel has a camera (/dev/hakios-camera)\n");
	}
	if (camera) {
		pthread_mutex_lock(&lock);
		if (camera_vendor == NULL) {
			id made = ((id (*)(Class, SEL))objc_msgSend)(objc_getClass("BWFigCaptureDeviceVendor"), sel_registerName("alloc"));

			camera_vendor = ((id (*)(id, SEL, void *))objc_msgSend)(made, sel_registerName("initWithDeviceCreateFunction:"),
			    (void *)hakios_camera_device_create);
			say("hakios: mediaserverd: the camera's device vendor %s\n", camera_vendor ? "made (camera_device.c)" :
			    "CANNOT BE MADE");
#if HAKIOS_CAMERA_DEBUG
			patch_signal_error();
			hook_session();
#endif
		}
		pthread_mutex_unlock(&lock);
		return camera_vendor;
	}
	for (size_t i = 0; torch >= 0 && i < sizeof(flashlight_code) / sizeof(flashlight_code[0]); i++) {
		if (caller >= flashlight_code[i].from && caller < flashlight_code[i].to) {
			if (ours == NULL) {
				ours = class_createInstance(vendor_class, 0);
				say("hakios: mediaserverd: no camera, so no device vendor: FigFlashlight gets the flashlight's own\n");
			}
			return ours;
		}
	}
	return NULL;
}

static void
flashlight_set(uint32_t level)
{
	ssize_t written;

	if (torch < 0) {
		torch = open("/dev/hakios-torch", O_WRONLY);    /* (a kernel that makes it after this program started) */
	}
	written = torch >= 0 ? write(torch, &level, sizeof(level)) : -1;

	if (written != (ssize_t)sizeof(level) || torch_said < 40) {
		torch_said++;
		say("hakios: mediaserverd: the flashlight at %u of 1000%s\n", level, written == (ssize_t)sizeof(level) ? "" :
		    " (NOT TAKEN by /dev/hakios-torch)");
	}
}

/* the vendor's: one client, one device, always available to it */
static int
flashlight_register(__unused id self, __unused SEL selector, __unused int pid, __unused id description, __unused int priority,
    __unused BOOL steal, __unused BOOL share, __unused id handler)
{
	return 1;
}

static id
flashlight_copy_device(__unused id self, __unused SEL selector, __unused int client, __unused BOOL inform, int *error)
{
	if (error != NULL) {
		*error = 0;
	}
	return class_createInstance(device_class, 0);
}

static id
flashlight_copy_stream(__unused id self, __unused SEL selector, __unused int position, __unused int type, id device)
{
	return device != NULL && object_getClass(device) == device_class ? class_createInstance(stream_class, 0) : NULL;
}

static void
flashlight_take_back(__unused id self, __unused SEL selector, __unused id device, __unused int client, __unused BOOL inform)
{
	flashlight_set(0);
}

static BOOL
flashlight_active_equals(__unused id self, __unused SEL selector, id device)
{
	return device == NULL || object_getClass(device) == device_class;
}

/* the stream's -supportedProperties: none that FigFlashlight looks for ("HighCurrentTorchEnabled") */
static id
flashlight_stream_properties(__unused id self, __unused SEL selector)
{
	static CFDictionaryRef none;

	if (none == NULL) {
		none = CFDictionaryCreate(NULL, NULL, NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	}
	return (id)none;
}

/* the stream's -setProperty:value: -- "TorchLevel", 0 to 1, as thousandths to the kernel */
static int
flashlight_stream_set(__unused id self, __unused SEL selector, id key, id value)
{
	float level = 0;

	if (key != NULL && CFEqual((CFTypeRef)key, CFSTR("TorchLevel"))) {
		if (value != NULL && CFGetTypeID((CFTypeRef)value) == CFNumberGetTypeID()) {
			CFNumberGetValue((CFNumberRef)value, kCFNumberFloatType, &level);
		}
		level = level < 0 ? 0 : level > 1 ? 1 : level;
		flashlight_set((uint32_t)(level * 1000 + 0.5f));
	}
	return 0;
}

/*
 * Video stabilization, of which this hardware has none: the capture graph's VIS node
 * (-[FigCaptureVISPipeline _newVISNodeWithUpstreamOutput:...]) is made with the ISP's processing
 * session, and this device has no such session, so the node comes back nil, the VIS pipeline with
 * it, the movie file sink's tail pipeline with that, and captureSession_buildGraphWithConfiguration
 * tears the whole graph down -- the preview too, for all that the preview has nothing to do with
 * stabilizing a recording. The connection's stabilization method is where the whole stack reads the
 * wish from (+[FigVideoCaptureConnectionConfiguration videoStabilizationMethods:includeIris:] ->
 * cspc_getVideoStabilizationType -> the connections' FigCaptureVideoStabilizationType), and 0 there
 * means none: the graph is then built without a VIS pipeline anywhere
 * (+[FigCaptureVISPipeline visPipelineIsRequiredForFigCaptureVideoStabilizationType:] says no to
 * every type but 2 and 3), and recordings are as steady as the hand that holds the phone.
 */
static int
video_stabilization_method(id self, SEL selector)
{
	(void)self;
	(void)selector;
	return 0;
}

/*
 * The type the stack acts on. The connection configuration arrives from the application's
 * AVFoundation with the type in it, which is read straight out of the object and not computed from
 * the method above (-[FigCaptureSessionParsedConfiguration _computeVideoStabilizationTypes] only
 * fills in what the client left out), so the wish has to be refused here as well.
 */
static int
video_stabilization_type(id self, SEL selector)
{
	(void)self;
	(void)selector;
	return 0;                       /* kFigCaptureVideoStabilizationType_None */
}

/*
 * The microphone, which the capture stack has none of here. A capture session's sound comes through
 * CoreMedia's own capture audio session -- BWAudioSourceNode's start sets its properties and reads
 * its pull format, all of it through a brokered plug-in of the audio HAL -- and that plug-in is not
 * on this phone: the node's start comes back with 'nope', the HAL's word for "no such thing".
 *
 * Camera.app's session has an audio input in it, because it is a session for photographs and for
 * video both. The graph starts its source nodes late (startDeferredSourceNodesIfNeeded) and takes
 * the first error it gets for the whole graph's, so the session reported AVCaptureSessionRuntimeError
 * to the application, which took the session down and built it again, two seconds later, for ever:
 * the camera's preview came in bursts of five frames with the sensor powering down in between.
 *
 * The node is told that it started. It then produces no sound -- which is all a phone with no
 * capture audio has to offer -- and the rest of the graph, the preview and the photograph, runs.
 * Only that one error is swallowed: a microphone that is there and fails for another reason still
 * takes the session down, as it should. (The kernel's own microphone, adsp=1 mic=4, is a different
 * path: AVAudioRecorder and Voice Memos record through it, and nothing here touches it.)
 */
#define ERROR_NO_SUCH_THING     1852797029      /* 'nope' */

static IMP original_audio_source_start;

static BOOL
audio_source_start(id self, SEL selector, id *error)
{
	BOOL started = ((BOOL (*)(id, SEL, id *))original_audio_source_start)(self, selector, error);
	id made = error != NULL ? *error : NULL;
	static bool said;

	if (started || made == NULL || ((long (*)(id, SEL))objc_msgSend)(made, sel_registerName("code")) != ERROR_NO_SUCH_THING) {
		return started;
	}
	*error = NULL;                  /* (autoreleased by the capture stack: not ours to give back) */
	if (!said) {
		said = true;
		say("hakios: mediaserverd: there is no capture audio on this phone: the session's audio source gives nothing, "
		    "and the camera runs without it\n");
	}
	return true;
}

/*
 * A photograph's quality prioritization, "speed" at the most. Asked for "balanced" (Camera.app's) or
 * "quality", the capture device takes a scene for a low-light one whenever a frame was exposed at
 * least as long as the sensor's still image stabilization threshold
 * (-[BWFigVideoCaptureDevice _isLowLightSceneUsingFrameStatisticsByPortType:]) -- a threshold from
 * the sensor's tuning in CameraSetup.plist, 0 for a sensor it has no tuning for, as ours -- and
 * then asks the stream for a bracket of four frames for SIS fusion (captureType 4, BWSISNode), which
 * this port has nothing to fuse with: the photograph never came. With "speed" it asks for one frame
 * (-_resolveStillImageCaptureTypeFromStatisticsByPortType:..., captureType 1), which the stream
 * gives (camera_device.c, StillImageCaptureNow).
 */
static IMP original_quality_prioritization;

static int
quality_prioritization(id self, SEL selector)
{
	int asked = ((int (*)(id, SEL))original_quality_prioritization)(self, selector);
	static bool said;

	if (asked > 1 && !said) {
		said = true;
		say("hakios: mediaserverd: a photograph's quality prioritization %d taken as 1 (speed): one frame, no SIS bracket\n",
		    asked);
	}
	return asked > 1 ? 1 : asked;
}

/*
 * A photograph's noise reduction (BWGNRNode, between the still image coordinator and the photo
 * encoder) is a FigSampleBufferProcessor of the iPhone's, which is handed the picture and never
 * hands one back here: the photograph stopped there, with no error, and the request with it (the
 * shutter's second press asked for nothing more). The picture goes past it to the node's output as
 * it came, as with no noise reduction to do.
 */
static void
gnr_render(id self, SEL selector, void *sample_buffer, id input)
{
	id output = ((id (*)(id, SEL))objc_msgSend)(self, sel_registerName("output"));
	static bool said;

	(void)selector, (void)input;
	if (!said) {
		said = true;
		say("hakios: mediaserverd: a photograph goes past the noise reduction, which has nothing to do it with here\n");
	}
	if (output != nil) {
		((void (*)(id, SEL, void *))objc_msgSend)(output, sel_registerName("emitSampleBuffer:"), sample_buffer);
	}
}

static bool
is_node(id node, const char *name)
{
	return node != nil && strcmp(object_getClassName(node), name) == 0;
}

/*
 * How long a photograph's encoding took, from the picture handed to the photo encoder node to the one
 * it hands on (HEVC in software, see register_hevc_encoder below: the slowest step of a photograph).
 */
static void
time_photo_encoder(id output, CMSampleBufferRef sample_buffer)
{
	CVImageBufferRef image = CMSampleBufferGetImageBuffer(sample_buffer);
	static uint64_t began;
	static mach_timebase_info_data_t timebase;
	id connection, input;

	if (image == NULL || CVPixelBufferGetWidth(image) < 2000) {
		return;                 /* not a photograph */
	}
	if (is_node(((id (*)(id, SEL))objc_msgSend)(output, sel_registerName("node")), "BWPhotoEncoderNode")) {
		if (began != 0) {
			if (timebase.denom == 0) {
				mach_timebase_info(&timebase);
			}
			say("hakios: mediaserverd: a photograph of %zux%zu encoded in %.0f ms\n", CVPixelBufferGetWidth(image),
			    CVPixelBufferGetHeight(image), (double)(mach_absolute_time() - began) * timebase.numer / timebase.denom / 1e6);
		}
		began = 0;
		return;
	}
	connection = ((id (*)(id, SEL))objc_msgSend)(output, sel_registerName("connection"));
	input = connection != nil ? ((id (*)(id, SEL))objc_msgSend)(connection, sel_registerName("input")) : nil;
	if (input != nil && is_node(((id (*)(id, SEL))objc_msgSend)(input, sel_registerName("node")), "BWPhotoEncoderNode")) {
		began = mach_absolute_time();
	}
}

/*
 * A node's output given no sample buffer at all, which no node takes: the session's audio source
 * (-[BWAudioSourceNode _generateSamples]), whose audio unit never started here (see audio_source_start
 * above), fills the silence by a timer with buffers it fails to make and emits them all the same --
 * NULL. Live Photo's staging node added that to an array (-[BWIrisStagingNode renderSampleBuffer:forInput:])
 * and mediaserverd died of the NSInvalidArgumentException, every half a minute with Live Photo on, the
 * preview black for a second each time. A NULL is not passed on.
 */
static IMP original_emit_sample_buffer;

static void
emit_sample_buffer(id self, SEL selector, CMSampleBufferRef sample_buffer)
{
	static bool said;

	if (sample_buffer == NULL) {
		if (!said) {
			said = true;
			say("hakios: mediaserverd: a node's output was given no sample buffer (the silent audio source's): not passed on\n");
		}
		return;
	}
	time_photo_encoder(self, sample_buffer);
	((void (*)(id, SEL, CMSampleBufferRef))original_emit_sample_buffer)(self, selector, sample_buffer);
}

/*
 * The photograph's encoder. Camera.app asks for HEIC -- HEVC inside a HEIF file, which CMPhoto makes
 * through VideoToolbox -- and VideoToolbox here has no HEVC encoder but the iPhone's hardware ones
 * (com.apple.videotoolbox.videoencoder.hevc and .hevc.gva, behind its AVE): the photo encoder node
 * failed with kVTCouldNotFindVideoEncoderErr (-12908). Apple's software HEVC codec is in the system
 * all the same, VCPHEVC.videocodec, whose decoder VideoToolbox takes here for every HEIC picture (see
 * the top), and there is an encoder in it, HEVCVideoEncoder_CreateInstance, that Darwin never registers:
 * the codec's VCPHEVCRegisterEncoder is an empty function. It is registered here, as the codec
 * registers its decoder (VTRegisterVideoDecoderWithInfo, not hardware accelerated).
 */
static void
register_hevc_encoder(void)
{
	void *codec = dlopen("/System/Library/VideoCodecs/VCPHEVC.videocodec", RTLD_NOW | RTLD_LOCAL);
	void *vt = dlopen("/System/Library/Frameworks/VideoToolbox.framework/VideoToolbox", RTLD_NOW | RTLD_GLOBAL);
	void *create = codec != NULL ? dlsym(codec, "HEVCVideoEncoder_CreateInstance") : NULL;
	int (*register_encoder)(uint32_t, CFDictionaryRef, void *) = vt != NULL ? dlsym(vt, "VTRegisterVideoEncoderWithInfo") : NULL;
	CFStringRef *id_key = vt != NULL ? dlsym(vt, "kVTVideoEncoderList_EncoderID") : NULL;
	CFStringRef *name_key = vt != NULL ? dlsym(vt, "kVTVideoEncoderList_EncoderName") : NULL;
	CFStringRef *hardware_key = vt != NULL ? dlsym(vt, "kVTVideoEncoderList_IsHardwareAccelerated") : NULL;
	int status;

	if (create == NULL || register_encoder == NULL || id_key == NULL || name_key == NULL || hardware_key == NULL) {
		say("hakios: mediaserverd: no software HEVC encoder to register (%s)\n", codec == NULL ? "no VCPHEVC.videocodec" :
		    create == NULL ? "no HEVCVideoEncoder_CreateInstance" : "VideoToolbox without VTRegisterVideoEncoderWithInfo");
		return;
	}
	const void *keys[] = { *id_key, *name_key, *hardware_key };
	const void *values[] = { CFSTR("com.apple.videotoolbox.videoencoder.hevc.vcp"), CFSTR("HEVC (VCPHEVC, software)"),
		kCFBooleanFalse };
	CFDictionaryRef info = CFDictionaryCreate(NULL, keys, values, 3, &kCFTypeDictionaryKeyCallBacks,
	    &kCFTypeDictionaryValueCallBacks);

	status = register_encoder('hvc1', info, create);
	CFRelease(info);
	say("hakios: mediaserverd: the software HEVC encoder of VCPHEVC.videocodec registered for 'hvc1' (%d)\n", status);
}

struct method_spec {
	const char *selector, *types;
	IMP imp;
};

static Class
make_class(const char *name, const struct method_spec *methods, size_t count)
{
	Class cls = objc_allocateClassPair(objc_getClass("NSObject"), name, 0);

	for (size_t i = 0; i < count; i++) {
		class_addMethod(cls, sel_registerName(methods[i].selector), methods[i].imp, methods[i].types);
	}
	objc_registerClassPair(cls);
	return cls;
}

static void
flashlight_classes(void)
{
	static const struct method_spec vendor[] = {
		{ "registerClientWithPID:clientDescription:clientPriority:canStealFromClientsWithSamePriority:"
		  "deviceSharingWithOtherClientsAllowed:deviceAvailabilityChangedHandler:", "i@:i@iBB@", (IMP)flashlight_register },
		{ "copyDeviceForClient:informClientWhenDeviceAvailableAgain:error:", "@@:iB^i", (IMP)flashlight_copy_device },
		{ "copyStreamForFlashlightWithPosition:deviceType:forDevice:", "@@:ii@", (IMP)flashlight_copy_stream },
		{ "takeBackDevice:forClient:informClientWhenDeviceAvailableAgain:", "v@:@iB", (IMP)flashlight_take_back },
		{ "activeDeviceEquals:", "B@:@", (IMP)flashlight_active_equals },
	}, stream[] = {
		{ "supportedProperties", "@@:", (IMP)flashlight_stream_properties },
		{ "setProperty:value:", "i@:@@", (IMP)flashlight_stream_set },
	};

	vendor_class = make_class("HakiosFlashlightVendor", vendor, sizeof(vendor) / sizeof(vendor[0]));
	device_class = make_class("HakiosFlashlightDevice", NULL, 0);
	stream_class = make_class("HakiosFlashlightStream", stream, sizeof(stream) / sizeof(stream[0]));
}

/*
 * Diagnostics of how CMCapture works out its capture sources (FigCaptureSourceUtilities, 20G81): which streams the
 * vendor is asked for by position and device type, what it gives, and how many sources the list ends with. Said only
 * where the kernel has a camera.
 */
static IMP original_copy_streams, original_backings_init;

static const char *
describe(id object)
{
	id text = object ? ((id (*)(id, SEL))objc_msgSend)(object, sel_registerName("description")) : NULL;

	return text ? ((const char *(*)(id, SEL))objc_msgSend)(text, sel_registerName("UTF8String")) : "(nil)";
}

static id
copy_streams_logged(id self, SEL selector, id positions, id types, id device, int *error)
{
	int e = 0;
	id streams = ((id (*)(id, SEL, id, id, id, int *))original_copy_streams)(self, selector, positions, types, device, &e);

	if (camera) {
		char line[300];

		snprintf(line, sizeof(line), "%s", describe(positions));
		for (char *c = line; *c; c++) {
			*c = *c == '\n' ? ' ' : *c;
		}
		say("hakios: mediaserverd: streams asked of the vendor: positions %s", line);
		snprintf(line, sizeof(line), "%s", describe(types));
		for (char *c = line; *c; c++) {
			*c = *c == '\n' ? ' ' : *c;
		}
		say(" device types %s -> %ld stream(s), error %d\n", line,
		    streams ? (long)((unsigned long (*)(id, SEL))objc_msgSend)(streams, sel_registerName("count")) : -1L, e);
	}
	if (error != NULL) {
		*error = e;
	}
	return streams;
}

/*
 * Photographic Styles ("semantic style rendering": the iPhone 13's, not the iPhone X's): with it the preview is built
 * as a FILTERED pipeline with person-segmentation inferences -- -[FigCapturePreviewSinkPipeline
 * _appendFilteredPreviewPipeline:...] uses an inference configuration the path here never made, and mediaserverd
 * crashed (SIGSEGV at 0x10 in objc_msgSend, every time Camera.app started its session; 11-18). There is no ISP nor
 * neural engine to render styles with: the formats say it is not supported, and a preview is never asked to render it.
 */
static bool
answers_no(__unused id self, __unused SEL selector)
{
	return false;
}

static IMP original_copy_device;

static id
copy_device_logged(id self, SEL selector, int client, bool inform, int *error)
{
	int e = 0;
	id device = ((id (*)(id, SEL, int, bool, int *))original_copy_device)(self, selector, client, inform, &e);

	if (camera) {
		say("hakios: mediaserverd: the vendor's device for client %d: %s, error %d\n", client, device ? describe(device) : "NONE", e);
	}
	if (error != NULL) {
		*error = e;
	}
	return device;
}

static IMP original_take_back_streams;

/* who gives streams back (the first three times): a session that could not build its graph does, on its way out */
static void
take_back_streams_logged(id self, SEL selector, id streams, id device)
{
	static int said;

	if (camera && said < 3) {
		void *frames[24];
		int n = backtrace(frames, 24);
		char line[600];
		int at = 0;

		said++;
		for (int i = 1; i < n && at < (int)sizeof(line) - 24; i++) {
			at += snprintf(line + at, sizeof(line) - (size_t)at, " %#lx",
			    (unsigned long)(((uintptr_t)frames[i] & 0x0000007fffffffffUL) - slide));
		}
		say("hakios: mediaserverd: streams taken back, called from (unslid):%s\n", line);
	}
	((void (*)(id, SEL, id, id))original_take_back_streams)(self, selector, streams, device);
}

static id
backings_init_logged(id self, SEL selector, id dictionaries, id common, id hevc, id h264)
{
	if (camera) {
		long n = dictionaries ? (long)((unsigned long (*)(id, SEL))objc_msgSend)(dictionaries, sel_registerName("count")) : -1;

		say("hakios: mediaserverd: CMCapture's capture sources: %ld\n", n);
		for (long i = 0; i < n && i < 12; i++) {
			id one = ((id (*)(id, SEL, unsigned long))objc_msgSend)(dictionaries, sel_registerName("objectAtIndex:"), (unsigned long)i);
			id attributes = ((id (*)(id, SEL, id))objc_msgSend)(one, sel_registerName("objectForKey:"), (id)CFSTR("Attributes"));
			id formats = ((id (*)(id, SEL, id))objc_msgSend)(one, sel_registerName("objectForKey:"), (id)CFSTR("Formats"));
			id name = attributes ? ((id (*)(id, SEL, id))objc_msgSend)(attributes, sel_registerName("objectForKey:"),
			    (id)CFSTR("uniqueID")) : NULL;

			say("hakios: mediaserverd:   source %ld: %s, %ld format(s)\n", i, describe(name),
			    formats ? (long)((unsigned long (*)(id, SEL))objc_msgSend)(formats, sel_registerName("count")) : -1L);
		}
	}
	return ((id (*)(id, SEL, id, id, id, id))original_backings_init)(self, selector, dictionaries, common, hevc, h264);
}

/*
 * ---- DIAGNOSTICS OF THE CAPTURE SESSION (20G81), built only with HAKIOS_CAMERA_DEBUG (build.sh CAMERA_DEBUG=1) ----
 *
 * How a session that never showed a picture was found out (11-18): CMCapture says nothing this image can read -- its
 * log has no reader, its error reporters are reached from inside the shared cache where interposing does not go, and
 * most of its methods are direct (no selector to replace). So: CoreMedia's FigSignalErrorAt and assertion reporters
 * patched at their first instructions; BWGraph's methods wrapped (nodes added, commit, start, dealloc); and CMCapture's
 * C functions entered through trampolines (hook_function), each saying what it returns. That chain led from
 * "startGraph returns -12780" through allocateSharedBufferPools and _bringStreamUpToDate to
 * _verifyColorSpacePropertiesForAllOutputs: the stream's VideoOutputsColorInfo (camera_device.c).
 * All of it costs time and writes over shared-cache text (private copies): never in a build that is not looking.
 */
#if HAKIOS_CAMERA_DEBUG
/*
 * A probe (20G81): what the dimension getters of FigCaptureSourceVideoFormat answer for the formats in use -- the graph
 * builder sizes its outputs by them, and a zero there is a session that never starts. Each selector's answers are said
 * the first few times they differ.
 */
static void
probe_format_dimensions(void)
{
	static const char *const selectors[] = { "previewDimensions", "sensorDimensions", "sensorCropDimensions", "postGDCCropDimensions",
		                                 "nativeDimensions", "preGDCSensorCropDimensions", "geometricDistortionCorrectedNativeDimensions",
		                                 "cropDimensionsForFES", "fesInputCropDimensions", "sourceCropAspectRatio" };
	static bool done;
	Class cls = objc_getClass("FigCaptureSourceVideoFormat");

	if (done || cls == NULL || !camera) {
		return;
	}
	done = true;
	for (size_t i = 0; i < sizeof(selectors) / sizeof(selectors[0]); i++) {
		SEL selector = sel_registerName(selectors[i]);
		Method method = class_getInstanceMethod(cls, selector);
		const char *name = selectors[i];
		IMP original;

		if (method == NULL) {
			say("hakios: mediaserverd: probe: no -[FigCaptureSourceVideoFormat %s]\n", name);
			continue;
		}
		original = method_getImplementation(method);
		method_setImplementation(method, imp_implementationWithBlock((id)^uint64_t (id self) {
			static uint64_t seen[10][6];
			static int count[10];
			uint64_t r = ((uint64_t (*)(id, SEL))original)(self, selector);
			bool known = false;

			for (int k = 0; k < count[i]; k++) {
				known |= seen[i][k] == r;
			}
			if (!known && count[i] < 6) {
				seen[i][count[i]++] = r;
				say("hakios: mediaserverd: probe: %s = %d x %d\n", name, (int)(uint32_t)r, (int)(uint32_t)(r >> 32));
			}
			return r;
		}));
	}
	say("hakios: mediaserverd: probe: the format's dimension getters are watched\n");
}

/*
 * A diagnostic: CoreMedia's FigSignalErrorAt, through which CMCapture signals its failures (they go to a log this image
 * does not show). Each call site's first error is said, with where it came from (the cache's address: tools/dsc-symbols.py
 * names it). Only with the kernel's camera, and only the first 60 sites.
 */
extern int32_t FigSignalErrorAt(int32_t, void *, void *, void *, void *, void *, void *, void *);

static int32_t
signal_error_logged(int32_t error, void *a, void *b, void *c, void *d, void *e, void *f, void *g)
{
	static uintptr_t sites[60];
	static int count;
	/* (arm64e: the return address carries its signature in the top bits) */
	uintptr_t from = ((uintptr_t)__builtin_return_address(0) & 0x0000007fffffffffUL) - slide;

	if (camera && error != 0 && count < 60) {
		bool known = false;

		for (int i = 0; i < count; i++) {
			known |= sites[i] == from;
		}
		if (!known) {
			Dl_info where = { 0 };
			void *raw = (void *)((uintptr_t)__builtin_return_address(0) & 0x0000007fffffffffUL);

			sites[count++] = from;
			if (dladdr(raw, &where) != 0 && where.dli_fname != NULL) {
				say("hakios: mediaserverd: FigSignalErrorAt(%d) from %s + %#lx (%s), cache address %#lx\n", error, where.dli_fname,
				    (unsigned long)((uintptr_t)raw - (uintptr_t)where.dli_fbase), where.dli_sname ? where.dli_sname : "?",
				    (unsigned long)from);
			} else {
				say("hakios: mediaserverd: FigSignalErrorAt(%d) from %#lx\n", error, (unsigned long)from);
			}
		}
	}
	return FigSignalErrorAt(error, a, b, c, d, e, f, g);
}

__attribute__((used)) static struct {
	const void *ours, *theirs;
} signal_error_interpose __attribute__((section("__DATA,__interpose"))) = {
	(const void *)signal_error_logged, (const void *)FigSignalErrorAt
};

/*
 * A diagnostic: what CMCapture writes to the system log (which this image has no reader for). The format string of
 * each os_log call made from CMCapture is said once, with its level; the arguments are not decoded, except that a
 * first argument that is a 32-bit number is said too (an error code, most of the time). At most 400 different lines.
 */
#include <os/log.h>

static void
capture_log_said(void *dso, os_log_type_t type, const char *format, const uint8_t *buf, uint32_t size)
{
	static uint64_t seen[900];
	static int count;
	static const void *wanted[4], *unwanted[48];
	static int n_wanted, n_unwanted;
	char line[420];
	int at;
	uint64_t hash = 1469598103934665603ULL;
	bool ours = false;

	if (!camera || format == NULL || count >= 900) {
		return;
	}
	for (int i = 0; i < n_wanted; i++) {
		ours |= wanted[i] == dso;
	}
	if (!ours) {
		Dl_info where = { 0 };

		for (int i = 0; i < n_unwanted; i++) {
			if (unwanted[i] == dso) {
				return;
			}
		}
		if (dladdr(dso, &where) == 0 || where.dli_fname == NULL ||
		    (strstr(where.dli_fname, "CMCapture") == NULL && strstr(where.dli_fname, "CoreMedia.framework") == NULL) ||
		    n_wanted >= 4) {
			if (n_unwanted < 48) {
				unwanted[n_unwanted++] = dso;
			}
			return;
		}
		wanted[n_wanted++] = dso;
	}
	at = snprintf(line, sizeof(line), "(%d) %.120s", (int)type, format);
	/* the buffer: summary, count, then {descriptor, size, bytes} for each argument: strings and 32-bit numbers said */
	if (buf != NULL && size >= 2) {
		uint32_t off = 2;

		for (int i = 0; i < buf[1] && off + 2 <= size && at < (int)sizeof(line) - 40; i++) {
			uint8_t kind = buf[off] >> 4, length = buf[off + 1];

			if (off + 2 + length > size) {
				break;
			}
			if (kind == 2 && length == 8) {
				const char *text = NULL;

				memcpy(&text, buf + off + 2, 8);
				at += snprintf(line + at, sizeof(line) - (size_t)at, " | %.240s", text ? text : "(null)");
			} else if (kind == 0 && length == 4) {
				int32_t v;

				memcpy(&v, buf + off + 2, 4);
				at += snprintf(line + at, sizeof(line) - (size_t)at, " | %d", v);
			}
			off += 2u + length;
		}
	}
	for (const char *c = line; *c != 0; c++) {
		if (*c < '0' || *c > '9') {     /* (numbers left out: the same line with another pointer or time is the same line) */
			hash = (hash ^ (uint8_t)*c) * 1099511628211ULL;
		}
	}
	for (int i = 0; i < count; i++) {
		if (seen[i] == hash) {
			return;
		}
	}
	seen[count++] = hash;
	say("hakios: mediaserverd: capture log %s\n", line);
}

extern void _os_log_impl(void *, os_log_t, os_log_type_t, const char *, uint8_t *, uint32_t);
extern void _os_log_error_impl(void *, os_log_t, os_log_type_t, const char *, uint8_t *, uint32_t);
extern void _os_log_fault_impl(void *, os_log_t, os_log_type_t, const char *, uint8_t *, uint32_t);
extern void _os_log_debug_impl(void *, os_log_t, os_log_type_t, const char *, uint8_t *, uint32_t);

#define LOG_WATCHED(name)                                                                                              \
	static void watched##name(void *dso, os_log_t log, os_log_type_t type, const char *format, uint8_t *buf,       \
	    uint32_t size)                                                                                             \
	{                                                                                                              \
		capture_log_said(dso, type, format, buf, size);                                                        \
		name(dso, log, type, format, buf, size);                                                               \
	}                                                                                                              \
	__attribute__((used)) static struct {                                                                          \
		const void *ours, *theirs;                                                                             \
	} interpose##name __attribute__((section("__DATA,__interpose"))) = { (const void *)watched##name,              \
		                                                             (const void *)name };

/* (with nobody reading the log every level is "not enabled" and the calls above are never made: all enabled, for this) */
static bool
log_type_always_enabled(os_log_t log, os_log_type_t type)
{
	return camera ? true : os_log_type_enabled(log, type);
}

__attribute__((used)) static struct {
	const void *ours, *theirs;
} interpose_log_enabled __attribute__((section("__DATA,__interpose"))) = { (const void *)log_type_always_enabled,
	                                                                   (const void *)os_log_type_enabled };

LOG_WATCHED(_os_log_impl)
LOG_WATCHED(_os_log_error_impl)
LOG_WATCHED(_os_log_fault_impl)
LOG_WATCHED(_os_log_debug_impl)

/*
 * A diagnostic: FigSignalErrorAt as CMCapture calls it. Interposing does not reach a call from one image of the shared
 * cache into another (they are bound when the cache is built), so the function's own first instructions are replaced
 * by a jump to signal_error_patched (a private copy of the page). It says each call site's first error and returns the
 * error as the function does; what the function would have logged is lost, which nobody reads here anyway.
 */
#include <libkern/OSCacheControl.h>
#include <mach/mach.h>

static int32_t
signal_error_patched(int32_t error)
{
	static uintptr_t sites[120];
	static int count;
	uintptr_t from = ((uintptr_t)__builtin_return_address(0) & 0x0000007fffffffffUL) - slide;

	if (error != 0 && count < 120) {
		bool known = false;

		for (int i = 0; i < count; i++) {
			known |= sites[i] == from;
		}
		if (!known) {
			sites[count++] = from;
			say("hakios: mediaserverd: FigSignalErrorAt(%d) at cache address %#lx\n", error, (unsigned long)from);
		}
	}
	return error;
}

/* (CoreMedia's assertion reporters, the same way: who complains, and the numbers it complains with) */
static void
assert_patched(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5)
{
	static uintptr_t sites[160];
	static int count;
	uintptr_t from = ((uintptr_t)__builtin_return_address(0) & 0x0000007fffffffffUL) - slide;

	(void)a0;
	(void)a1;
	if (count < 160) {
		for (int i = 0; i < count; i++) {
			if (sites[i] == from) {
				return;
			}
		}
		sites[count++] = from;
		say("hakios: mediaserverd: FigDebugAssert at cache address %#lx (%ld %ld %ld %ld)\n", (unsigned long)from, (long)(int32_t)a2,
		    (long)(int32_t)a3, (long)(int32_t)a4, (long)(int32_t)a5);
	}
}

static bool
patch_function(uintptr_t at, void *to)
{
	uintptr_t target = (uintptr_t)__builtin_ptrauth_strip(to, 0);
	uint32_t code[4];
	vm_address_t page;

	at += slide;
	code[0] = 0x58000050;           /* ldr x16, #8 */
	code[1] = 0xd61f0200;           /* br x16 */
	memcpy(&code[2], &target, 8);
	page = at & ~(vm_address_t)(vm_page_size - 1);
	if (vm_protect(mach_task_self(), page, 2 * vm_page_size, FALSE, VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY) != KERN_SUCCESS) {
		return false;
	}
	memcpy((void *)at, code, sizeof(code));
	vm_protect(mach_task_self(), page, 2 * vm_page_size, FALSE, VM_PROT_READ | VM_PROT_EXECUTE);
	sys_icache_invalidate((void *)at, sizeof(code));
	return true;
}

static void
patch_signal_error(void)
{
	static bool done;

	if (done || !camera || AT(0UL, 0UL, 1UL) == 0 || slide == 0) {
		return;
	}
	done = true;
	say("hakios: mediaserverd: patched: FigSignalErrorAt %d, figDebugAssertCommon %d, FigDebugAssert2 %d\n",
	    patch_function(AT(0UL, 0UL, 0x1921fc7c4UL), (void *)signal_error_patched),
	    patch_function(AT(0UL, 0UL, 0x19224d238UL), (void *)assert_patched),
	    patch_function(AT(0UL, 0UL, 0x19224d434UL), (void *)assert_patched));
}

/*
 * A diagnostic: the capture graph being built. Each node added, each connection refused, the commit, the start and the
 * error status the graph is given (with who gave it), so that a session that never runs says where it stopped.
 */
static void
said_from(const char *what)
{
	void *frames[14];
	int n = backtrace(frames, 14), at;
	char line[400];

	at = snprintf(line, sizeof(line), "hakios: mediaserverd: graph: %s, called from (cache addresses):", what);
	for (int i = 2; i < n && at < (int)sizeof(line) - 24; i++) {
		at += snprintf(line + at, sizeof(line) - (size_t)at, " %#lx",
		    (unsigned long)(((uintptr_t)frames[i] & 0x0000007fffffffffUL) - slide));
	}
	say("%s\n", line);
}

static void
probe_graph(void)
{
	static bool done;
	Class cls = objc_getClass("BWGraph");
	Method m;

	if (done || cls == NULL || !camera) {
		return;
	}
	done = true;
	if ((m = class_getInstanceMethod(cls, sel_registerName("addNode:error:"))) != NULL) {
		IMP original = method_getImplementation(m);
		SEL selector = method_getName(m);

		method_setImplementation(m, imp_implementationWithBlock((id)^signed char (id self, id node, id *error) {
			static int said;
			id mine = NULL;
			signed char ok = ((signed char (*)(id, SEL, id, id *))original)(self, selector, node, error ? error : &mine);

			if (said < 300 || !ok) {
				said++;
				say("hakios: mediaserverd: graph: node %s%s\n", node ? class_getName(object_getClass(node)) : "(nil)",
				    ok ? "" : ": NOT ADDED");
				if (!ok) {
					say("hakios: mediaserverd: graph:   error %s\n", describe(error ? *error : mine));
					said_from("a node not added");
				}
			}
			return ok;
		}));
	}
	if ((m = class_getInstanceMethod(cls, sel_registerName("connectOutput:toInput:pipelineStage:"))) != NULL) {
		IMP original = method_getImplementation(m);
		SEL selector = method_getName(m);

		method_setImplementation(m, imp_implementationWithBlock((id)^signed char (id self, id output, id input, id stage) {
			signed char ok = ((signed char (*)(id, SEL, id, id, id))original)(self, selector, output, input, stage);

			if (!ok) {
				say("hakios: mediaserverd: graph: NOT CONNECTED: %s to %s\n", describe(output), describe(input));
				said_from("a connection refused");
			}
			return ok;
		}));
	}
	if ((m = class_getInstanceMethod(cls, sel_registerName("commitConfigurationWithID:error:"))) != NULL) {
		IMP original = method_getImplementation(m);
		SEL selector = method_getName(m);

		method_setImplementation(m, imp_implementationWithBlock((id)^signed char (id self, long long number, id *error) {
			static int said;
			id mine = NULL;
			signed char ok = ((signed char (*)(id, SEL, long long, id *))original)(self, selector, number, error ? error : &mine);

			if (said++ < 12) {
				say("hakios: mediaserverd: graph: commit %lld: %s %s\n", number, ok ? "done" : "FAILED",
				    ok ? "" : describe(error ? *error : mine));
			}
			return ok;
		}));
	}
	if ((m = class_getInstanceMethod(cls, sel_registerName("start:"))) != NULL) {
		IMP original = method_getImplementation(m);
		SEL selector = method_getName(m);

		method_setImplementation(m, imp_implementationWithBlock((id)^signed char (id self, id *error) {
			static int said;
			id mine = NULL;
			signed char ok = ((signed char (*)(id, SEL, id *))original)(self, selector, error ? error : &mine);

			if (said++ < 12) {
				say("hakios: mediaserverd: graph: start: %s %s\n", ok ? "done" : "FAILED", ok ? "" : describe(error ? *error : mine));
			}
			return ok;
		}));
	}
	if ((m = class_getInstanceMethod(cls, sel_registerName("setErrorStatus:"))) != NULL) {
		IMP original = method_getImplementation(m);
		SEL selector = method_getName(m);

		method_setImplementation(m, imp_implementationWithBlock((id)^void (id self, int status) {
			static int said;

			if (status != 0 && said++ < 6) {
				char what[60];

				snprintf(what, sizeof(what), "error status %d", status);
				said_from(what);
			}
			((void (*)(id, SEL, int))original)(self, selector, status);
		}));
	}
	if ((m = class_getInstanceMethod(cls, sel_registerName("dealloc"))) != NULL) {
		IMP original = method_getImplementation(m);
		SEL selector = method_getName(m);

		method_setImplementation(m, imp_implementationWithBlock((id)^void (__unsafe_unretained id self) {
			static int said;

			if (said++ < 3) {
				said_from("the graph is deallocated");
			}
			((void (*)(id, SEL))original)(self, selector);
		}));
	}
	say("hakios: mediaserverd: graph: BWGraph is watched\n");
}

/*
 * A diagnostic: the camera source node as the graph makes its configuration live -- which of its outputs are wanted,
 * the names it knows the stream's outputs by, and what it works out for them.
 */
static void
probe_source_node(void)
{
	static bool done;
	Class cls = objc_getClass("BWMultiStreamCameraSourceNode");
	Method m;

	if (done || cls == NULL || !camera) {
		return;
	}
	done = true;
	if ((m = class_getInstanceMethod(cls, sel_registerName("prepareForCurrentConfigurationToBecomeLive"))) != NULL) {
		IMP original = method_getImplementation(m);
		SEL selector = method_getName(m);

		method_setImplementation(m, imp_implementationWithBlock((id)^void (id self) {
			static int said;

			if (said++ < 2) {
				static const char *const flags[] = { "previewOutputEnabled", "videoCaptureOutputEnabled", "stillImageOutputEnabled",
					                             "depthOutputEnabled", "usesFirmwareStillImageOutput",
					                             "geometricDistortionCorrectionOnStreamingOutputsEnabled", "primaryScalerDecoupled",
					                             "videoStabilizationEnabled" };
				id configuration = ((id (*)(id, SEL))objc_msgSend)(self, sel_registerName("configuration"));
				unsigned count = 0;
				Ivar *ivars = class_copyIvarList(object_getClass(self), &count);

				say("hakios: mediaserverd: source node: configuration %s\n",
				    configuration ? class_getName(object_getClass(configuration)) : "(nil)");
				for (size_t i = 0; configuration != NULL && i < sizeof(flags) / sizeof(flags[0]); i++) {
					SEL flag = sel_registerName(flags[i]);

					if (class_getInstanceMethod(object_getClass(configuration), flag) != NULL) {
						say("hakios: mediaserverd: source node:   %s %d\n", flags[i],
						    (int)((signed char (*)(id, SEL))objc_msgSend)(configuration, flag));
					}
				}
				for (unsigned i = 0; i < count; i++) {
					const char *name = ivar_getName(ivars[i]), *type = ivar_getTypeEncoding(ivars[i]);

					if (name != NULL && type != NULL && type[0] == '@' &&
					    (strstr(name, "OutputID") != NULL || strstr(name, "utputs") != NULL || strstr(name, "ormat") != NULL)) {
						say("hakios: mediaserverd: source node:   %s = %.500s\n", name, describe(object_getIvar(self, ivars[i])));
					} else if (name != NULL && type != NULL && (type[0] == 'i' || type[0] == 'B' || type[0] == 'c') &&
					    (strstr(name, "ormatIndex") != NULL || strstr(name, "nabled") != NULL)) {
						int v = 0;

						memcpy(&v, (char *)(void *)self + ivar_getOffset(ivars[i]), type[0] == 'i' ? 4 : 1);
						say("hakios: mediaserverd: source node:   %s = %d\n", name, v);
					}
				}
				free(ivars);
			}
			((void (*)(id, SEL))original)(self, selector);
		}));
		say("hakios: mediaserverd: source node: watched\n");
	}
}

/*
 * A diagnostic: CMCapture's own session functions (C, reached by no selector), each entered through a jump written
 * over its first four instructions, which are kept in a trampoline that goes on into the function: what the session
 * tells its client, the states it goes through and who has the graph torn down.
 */
#include <sys/mman.h>

typedef uintptr_t (*session_function)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t);

static session_function
hook_function(uintptr_t at, void *to)
{
	uint32_t first[4], *trampoline;
	uintptr_t back;

	at += slide;
	memcpy(first, (void *)at, sizeof(first));
	for (int i = 0; i < 4; i++) {           /* (nothing there may depend on where it runs) */
		uint32_t w = first[i];

		if ((w & 0x1f000000) == 0x10000000 || (w & 0x7c000000) == 0x14000000 || (w & 0xff000010) == 0x54000000 ||
		    (w & 0x7e000000) == 0x34000000 || (w & 0x7e000000) == 0x36000000 || (w & 0x3b000000) == 0x18000000) {
			return NULL;
		}
	}
	trampoline = mmap(NULL, 0x4000, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
	if (trampoline == MAP_FAILED) {
		return NULL;
	}
	memcpy(trampoline, first, sizeof(first));
	trampoline[4] = 0x58000050;             /* ldr x16, #8 */
	trampoline[5] = 0xd61f0200;             /* br x16 */
	back = at + 16;
	memcpy(&trampoline[6], &back, 8);
	if (mprotect(trampoline, 0x4000, PROT_READ | PROT_EXEC) != 0) {
		return NULL;
	}
	sys_icache_invalidate(trampoline, 32);
	if (!patch_function(at - slide, to)) {
		return NULL;
	}
	return (session_function)__builtin_ptrauth_sign_unauthenticated((void *)trampoline, 0, 0);
}

static session_function session_post, session_transition, session_teardown, session_start_graph, session_stop_graph;

static uintptr_t
session_post_logged(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5)
{
	static int said;

	if (said++ < 40) {
		say("hakios: mediaserverd: session: notification %.80s %.700s\n", a1 ? describe((id)a1) : "(nil)",
		    a2 ? describe((id)a2) : "");
	}
	return session_post(a0, a1, a2, a3, a4, a5);
}

static uintptr_t
session_transition_logged(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5)
{
	static int said;

	if (said++ < 30) {
		char what[80];

		snprintf(what, sizeof(what), "session: status to %ld (%ld %ld)", (long)(int32_t)a1, (long)(int32_t)a2, (long)(int32_t)a3);
		said_from(what);
	}
	return session_transition(a0, a1, a2, a3, a4, a5);
}

static uintptr_t
session_teardown_logged(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5)
{
	static int said;

	if (said++ < 4) {
		char what[80];

		snprintf(what, sizeof(what), "session: the graph torn down (%ld %ld)", (long)(int32_t)a1, (long)(int32_t)a2);
		said_from(what);
	}
	return session_teardown(a0, a1, a2, a3, a4, a5);
}

static uintptr_t
session_start_graph_logged(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5)
{
	static int said;
	uintptr_t r = session_start_graph(a0, a1, a2, a3, a4, a5);

	if (said++ < 8) {
		say("hakios: mediaserverd: session: startGraph(%ld) returns %ld\n", (long)(int32_t)a1, (long)(int32_t)r);
	}
	return r;
}

static uintptr_t
session_stop_graph_logged(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5)
{
	static int said;

	if (said++ < 4) {
		char what[80];

		snprintf(what, sizeof(what), "session: stopGraph (%ld %ld)", (long)(int32_t)a1, (long)(int32_t)a2);
		said_from(what);
	}
	return session_stop_graph(a0, a1, a2, a3, a4, a5);
}

static session_function session_allowed, session_pools;

static uintptr_t
session_allowed_logged(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5)
{
	static int said;
	uintptr_t r = session_allowed(a0, a1, a2, a3, a4, a5);

	if (said++ < 6) {
		say("hakios: mediaserverd: session: checkClientSessionIsAllowedToRun returns %ld\n", (long)(int32_t)r);
	}
	return r;
}

static uintptr_t
session_pools_logged(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5)
{
	static int said;
	uintptr_t r = session_pools(a0, a1, a2, a3, a4, a5);

	if (said++ < 6) {
		say("hakios: mediaserverd: session: allocateSharedBufferPools returns %ld\n", (long)(int32_t)r);
	}
	return r;
}

#define RETURN_WATCHED(name, text)                                                                                     \
	static session_function name;                                                                                  \
	static uintptr_t name##_logged(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4,           \
	    uintptr_t a5)                                                                                              \
	{                                                                                                              \
		static int said;                                                                                       \
		uintptr_t r = name(a0, a1, a2, a3, a4, a5);                                                            \
		if (said++ < 8) {                                                                                      \
			say("hakios: mediaserverd: session: " text " returns %ld\n", (long)(int32_t)r);                 \
		}                                                                                                      \
		return r;                                                                                              \
	}

RETURN_WATCHED(node_update_outputs, "_updateOutputConfigurations")
RETURN_WATCHED(node_verify_colour, "_verifyColorSpacePropertiesForAllOutputs")
RETURN_WATCHED(node_bring_up, "_bringStreamUpToDate")
RETURN_WATCHED(node_update_metadata, "_updateMetadataConfigurations")
RETURN_WATCHED(node_update_depth, "_updateDepthConfiguration")
RETURN_WATCHED(node_update_detection, "_updateObjectDetectionMetadataRegistration")

static void
hook_session(void)
{
	static bool done;

	if (done || !camera || AT(0UL, 0UL, 1UL) == 0 || slide == 0) {
		return;
	}
	done = true;
	session_post = hook_function(0x1a3d91ca4UL, (void *)session_post_logged);
	session_transition = hook_function(0x1a3d38454UL, (void *)session_transition_logged);
	session_teardown = hook_function(0x1a3d2c928UL, (void *)session_teardown_logged);
	session_start_graph = hook_function(0x1a3d8a098UL, (void *)session_start_graph_logged);
	session_stop_graph = hook_function(0x1a3d1b11cUL, (void *)session_stop_graph_logged);
	session_allowed = hook_function(0x1a3d84c6cUL, (void *)session_allowed_logged);
	session_pools = hook_function(0x1a3d86ad4UL, (void *)session_pools_logged);
	node_update_outputs = hook_function(0x1a3db579cUL, (void *)node_update_outputs_logged);
	node_verify_colour = hook_function(0x1a3da85acUL, (void *)node_verify_colour_logged);
	node_bring_up = hook_function(0x1a3d743f0UL, (void *)node_bring_up_logged);
	node_update_metadata = hook_function(0x1a3d8ce1cUL, (void *)node_update_metadata_logged);
	node_update_depth = hook_function(0x1a3d9a9d4UL, (void *)node_update_depth_logged);
	node_update_detection = hook_function(0x1a3d74314UL, (void *)node_update_detection_logged);
	say("hakios: mediaserverd: session: hooked: outputs %d, colour %d, bring up %d, metadata %d, depth %d, detection %d\n",
	    node_update_outputs != NULL, node_verify_colour != NULL, node_bring_up != NULL, node_update_metadata != NULL,
	    node_update_depth != NULL, node_update_detection != NULL);
	say("hakios: mediaserverd: session: hooked: allowed %d, pools %d\n", session_allowed != NULL, session_pools != NULL);
	say("hakios: mediaserverd: session: hooked: post %d, transition %d, teardown %d, start %d, stop %d\n", session_post != NULL,
	    session_transition != NULL, session_teardown != NULL, session_start_graph != NULL, session_stop_graph != NULL);
}

#endif /* HAKIOS_CAMERA_DEBUG */

/*
 * The class methods replaced, each as soon as its class exists. `expected` is the unslid address
 * of the code in Darwin userland 16.5 (20F66): a method that is not what it was is left alone. `original`, where
 * there is one, keeps what was there (the vendor's: given as it is wherever there is one, and where
 * there is none, ours only with the kernel's camera or flashlight).
 */
static struct replacement {
	const char *class_name, *selector;
	uintptr_t expected;
	IMP replacement, *original;
	bool instance;                  /* an instance method, not a class method */
	bool done;
} replacements[] = {
	{ "BWCoreImageFilterRenderer", "prewarmPortraitFilterVersion:semanticStyleFilters:", AT(0x1a2115c0cUL, 0x1a2165b8cUL, 0x1a3f7b6a0UL),
	  (IMP)prewarm_portrait_filter, NULL, false, false },
	{ "BWFigCaptureDeviceVendor", "sharedCaptureDeviceVendor", AT(0x1a1ef0d04UL, 0x1a1f40c84UL, 0x1a3d3a5f0UL),
	  (IMP)shared_vendor, &original_shared_vendor, false, false },
	{ "BWAudioSourceNode", "start:", AT(0x1a1eefa18UL, 0x1a1f3f998UL, 0x1a3d392d0UL), (IMP)audio_source_start, &original_audio_source_start, true, false },
	{ "FigVideoCaptureConnectionConfiguration", "videoStabilizationMethod", AT(0x1a1e99ef8UL, 0x1a1ee9e78UL, 0x1a3ce12f0UL),
	  (IMP)video_stabilization_method, NULL, true, false },
	{ "FigVideoCaptureConnectionConfiguration", "videoStabilizationType", AT(0x1a20d8f48UL, 0x1a2128ec8UL, 0x1a3f3c34cUL),
	  (IMP)video_stabilization_type, NULL, true, false },
	{ "FigCaptureStillImageSettings", "qualityPrioritization", AT(0x1a1f533dcUL, 0x1a1fa335cUL, 0x1a3da23dcUL),
	  (IMP)quality_prioritization, &original_quality_prioritization, true, false },
	{ "BWGNRNode", "renderSampleBuffer:forInput:", AT(0x1a1ec83b8UL, 0x1a1f18338UL, 0x1a3d1078cUL), (IMP)gnr_render, NULL, true, false },
	{ "BWNodeOutput", "emitSampleBuffer:", AT(0x1a1ea37e0UL, 0x1a1ef3760UL, 0x1a3ceafbcUL), (IMP)emit_sample_buffer, &original_emit_sample_buffer, true, false },
	/* (20G81 only: the iPhone X's builds do not prewarm it -- 0 leaves it alone there) */
	{ "PTDisparityPostProcessing", "prewarmForMediaserverd", AT(0UL, 0UL, 0x1c2d40a44UL), (IMP)prewarm_disparity, NULL, false, false },
	/* (20G81 only: no Photographic Styles -- see answers_no) */
	{ "FigCaptureSourceVideoFormat", "isSemanticStyleRenderingSupported", AT(0UL, 0UL, 0x1a3ce5bacUL), (IMP)answers_no, NULL, true, false },
	/* ... and no filtered preview at all: -[FigCapturePreviewSinkPipeline _buildVideoPreviewSinkPipeline:...] appends the
	 * filter pipeline when the connection says previewFilterRenderingEnabled (answering no for the STYLE alone left a
	 * BWStreamingFilterNode "not configured for semantic style rendering" that was then given a style: an exception) */
	{ "FigVideoCaptureConnectionConfiguration", "previewFilterRenderingEnabled", AT(0UL, 0UL, 0x1a3dbe1f8UL), (IMP)answers_no, NULL, true,
	  false },
	/* (20G81 only: the diagnostics of the capture sources) */
	{ "BWFigCaptureDeviceVendor", "copyStreamsWithPositions:deviceTypes:forDevice:error:", AT(0UL, 0UL, 0x1a3d97f6cUL),
	  (IMP)copy_streams_logged, &original_copy_streams, true, false },
	{ "BWFigCaptureDeviceVendor", "takeBackStreams:device:", AT(0UL, 0UL, 0x1a3d5eb88UL), (IMP)take_back_streams_logged,
	  &original_take_back_streams, true, false },
	{ "BWFigCaptureDeviceVendor", "copyDeviceForClient:informClientWhenDeviceAvailableAgain:error:", AT(0UL, 0UL, 0x1a3d47434UL),
	  (IMP)copy_device_logged, &original_copy_device, true, false },
	{ "FigCaptureSourceBackings", "initWithSourceInfoDictionaries:commonSettings:hevcSettings:h264Settings:",
	  AT(0UL, 0UL, 0x1a3ec0eccUL), (IMP)backings_init_logged, &original_backings_init, true, false },
};

/* -[FigSDOFRendering init] is in a bundle the cache's symbols say nothing of, so it is not told by an address: it is
 * replaced when its class is there and its method is the bundle's own (SDOFRenderingV5.bundle), and on 20G81 only */

/* the renderer class FigPreloadShaders gets from SDOFRenderingV5.bundle (its principal class is FigSDOFRendering; it
 * may ask for another by name): told by where it is, not by its name, and only when its -init is the bundle's own */
static bool
in_sdof_bundle(const void *address)
{
	Dl_info info = { 0 };

	return address != NULL && dladdr(address, &info) != 0 && info.dli_fname != NULL &&
	    strstr(info.dli_fname, "/SDOFRenderingV5.bundle/") != NULL;
}

static void
sdof_replace(Class cls)
{
#if defined(HAKIOS_IOS_20G81)
	Method method;

	if (cls == Nil || !in_sdof_bundle((const void *)cls)) {
		return;
	}
	method = class_getInstanceMethod(cls, sel_registerName("init"));
	if (method != NULL && method_getImplementation(method) == (IMP)sdof_init) {
		return;                 /* (this one done already: FigPreloadShaders asks for several of the bundle's) */
	}
	if (method == NULL || !in_sdof_bundle((const void *)method_getImplementation(method))) {
		say("hakios: mediaserverd: -[%s init] is not SDOFRenderingV5's own; left alone\n", class_getName(cls));
		return;
	}
	method_setImplementation(method, (IMP)sdof_init);
	say("hakios: mediaserverd: -[%s init] replaced (SDOFRenderingV5's renderer: its prewarm is skipped)\n",
	    class_getName(cls));
#else
	(void)cls;
#endif
}

/* FigPreloadShaders has the class from its bundle (+[NSBundle bundleWithPath:], then -principalClass or -classNamed:),
 * which is when it first exists for the runtime -- and before anything allocates one. dyld's add-image call comes too
 * early (the classes not there yet), objc_addLoadImageFunc's under the runtime's lock (objc_getClass() in it traps,
 * 11-17). So the two NSBundle methods are wrapped, in mediaserverd only: each answers as it did, and the first answer
 * that is FigSDOFRendering has its -init replaced before it is returned. */
static IMP original_principal_class, original_class_named;

static Class
bundle_principal_class(id self, SEL selector)
{
	Class cls = ((Class (*)(id, SEL))original_principal_class)(self, selector);

	if (cls != Nil) {
		sdof_replace(cls);
	}
	return cls;
}

static Class
bundle_class_named(id self, SEL selector, id name)
{
	Class cls = ((Class (*)(id, SEL, id))original_class_named)(self, selector, name);

	if (cls != Nil) {
		sdof_replace(cls);
	}
	return cls;
}

static void
bundle_hooks(void)
{
#if defined(HAKIOS_IOS_20G81)
	Class bundle = objc_getClass("NSBundle");
	Method principal = bundle != Nil ? class_getInstanceMethod(bundle, sel_registerName("principalClass")) : NULL;
	Method named = bundle != Nil ? class_getInstanceMethod(bundle, sel_registerName("classNamed:")) : NULL;

	if (principal == NULL || named == NULL) {
		say("hakios: mediaserverd: no -[NSBundle principalClass]/classNamed:: -[FigSDOFRendering init] is not replaced\n");
		return;
	}
	original_principal_class = method_setImplementation(principal, (IMP)bundle_principal_class);
	original_class_named = method_setImplementation(named, (IMP)bundle_class_named);
#endif
}

static void
image_added(__unused const struct mach_header *header, __unused intptr_t image_slide)
{
#if HAKIOS_CAMERA_DEBUG
	probe_format_dimensions();
	probe_graph();
	probe_source_node();
#endif
	for (size_t i = 0; i < sizeof(replacements) / sizeof(replacements[0]); i++) {
		struct replacement *r = &replacements[i];
		Class cls;
		Method method;
		IMP original;

		if (r->done || (cls = objc_getClass(r->class_name)) == NULL) {
			continue;
		}
		r->done = true;
		method = r->instance ? class_getInstanceMethod(cls, sel_registerName(r->selector)) :
		    class_getClassMethod(cls, sel_registerName(r->selector));
		original = method ? method_getImplementation(method) : NULL;
		if (original == NULL && r->expected != 0) {
			r->done = false;        /* (the class is there, its method not yet: looked for again at the next image) */
			continue;
		}
		if (original == NULL || (uintptr_t)original != r->expected + slide) {
			say("hakios: %c[%s %s]: found %#lx, expected %#lx; left alone\n", r->instance ? '-' : '+', r->class_name,
			    r->selector,
			    original ? (uintptr_t)original - slide : 0, r->expected);
			continue;
		}
		if (r->original != NULL) {
			*r->original = original;
		}
		method_setImplementation(method, r->replacement);
		say("hakios: mediaserverd: %c[%s %s] replaced\n", r->instance ? '-' : '+', r->class_name, r->selector);
	}
}

/*
 * CMCapture keeps the list of capture sources it once worked out (FigCaptureSourceUtilities' csu_getBackings: a
 * "CaptureSourceInfo" record -- CreationDate, DeviceModel, Sources, DependentUserDefaults -- in its preferences) and
 * uses it again while the model and those defaults are the same. A volume that first started without a camera keeps a
 * record with no video source: with the kernel's camera there, our device was made, no AVCaptureDevice came of it and
 * Camera.app's preview stayed black (the Pixel 8a, 11-18). Looked for in the domains CMCapture may keep it in (said,
 * where it is found); one without a video source is removed, so that the list is worked out again with our device.
 */
static void
forget_stale_capture_sources(void)
{
	static const char *const domains[] = { "com.apple.cameracapture.volatile", "com.apple.cameracapture", "com.apple.coremedia",
		                               "com.apple.avfoundation", "com.apple.mediaserverd", "com.apple.cmio" };

	for (size_t i = 0; i < sizeof(domains) / sizeof(domains[0]); i++) {
		CFStringRef domain = CFStringCreateWithCString(NULL, domains[i], kCFStringEncodingUTF8);
		CFPropertyListRef record = CFPreferencesCopyValue(CFSTR("CaptureSourceInfo"), domain, kCFPreferencesCurrentUser,
		    kCFPreferencesAnyHost);

		if (record != NULL) {
			CFIndex video = 0, sources = 0;

			if (CFGetTypeID(record) == CFDictionaryGetTypeID()) {
				CFTypeRef list = CFDictionaryGetValue(record, CFSTR("Sources"));

				if (list != NULL && CFGetTypeID(list) == CFArrayGetTypeID()) {
					sources = CFArrayGetCount(list);
					for (CFIndex k = 0; k < sources; k++) {
						CFTypeRef one = CFArrayGetValueAtIndex(list, k), type;

						if (CFGetTypeID(one) != CFDictionaryGetTypeID()) {
							continue;
						}
						type = CFDictionaryGetValue(one, CFSTR("mediaType"));
						video += type == NULL || (CFGetTypeID(type) == CFStringGetTypeID() &&
						    CFStringCompare(type, CFSTR("vide"), 0) == kCFCompareEqualTo);
					}
				}
			}
			say("hakios: mediaserverd: CMCapture's kept capture sources are in %s: %ld source(s), %ld of video%s\n", domains[i],
			    (long)sources, (long)video, video == 0 ? ": removed, to be worked out again with the kernel's camera" : "");
			if (video == 0) {
				CFPreferencesSetValue(CFSTR("CaptureSourceInfo"), NULL, domain, kCFPreferencesCurrentUser, kCFPreferencesAnyHost);
				CFPreferencesSynchronize(domain, kCFPreferencesCurrentUser, kCFPreferencesAnyHost);
			}
			CFRelease(record);
		}
		CFRelease(domain);
	}
}


__attribute__((constructor))
static void
hakios_media_init(void)
{
	const char *name = getprogname();
	size_t length = 0;

	if (name == NULL || strcmp(name, "mediaserverd") != 0) {
		return;
	}
	slide = (uintptr_t)_dyld_get_shared_cache_range(&length) - CACHE_BASE;
	torch = open("/dev/hakios-torch", O_WRONLY);
	camera = access("/dev/hakios-camera", F_OK) == 0;
	{
		/* the flashlight also when its node is not there yet: the loader's flash comes with torch=N, and a kernel may
		 * make the node after mediaserverd starts (the Pixel 8a's, its bus brought up first: 2 s late, 11-11) --
		 * flashlight_set() opens it then */
		char arguments[1024] = "";
		size_t size = sizeof(arguments) - 1;
		const char *t = sysctlbyname("kern.bootargs", arguments, &size, NULL, 0) == 0 ? strstr(arguments, "torch=") : NULL;
		bool asked = t != NULL && (t == arguments || t[-1] == ' ') && t[6] >= '1' && t[6] <= '9';

		if (torch >= 0 || asked) {
			flashlight_classes();
			say("hakios: mediaserverd: the kernel has a flashlight (/dev/hakios-torch%s)\n", torch >= 0 ? "" :
			    ", not there yet: opened when first used");
		}
	}
	if (camera) {
		say("hakios: mediaserverd: the kernel has a camera (/dev/hakios-camera)\n");
		forget_stale_capture_sources();
		register_hevc_encoder();
	}
	_dyld_register_func_for_add_image(image_added);         /* also called for every image already there */
	bundle_hooks();
}
