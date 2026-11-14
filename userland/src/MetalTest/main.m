/*
 * MetalTest.app: Metal on this phone, one thing at a time, full screen. Every scene runs a few seconds and says at
 * the top what it is and what a correct picture looks like; a tap goes to the next one at once, and after the last
 * the first comes again. A scene whose pipeline or texture Metal would not make says so in red with Metal's error and
 * the others go on. Every line the label shows also goes to the console ("MetalTest: ...").
 *
 * All shaders are Metal Shading Language source compiled on the phone (-newLibraryWithSource:), so this needs no Mac.
 * Installed over USB: devices/akita/tools/usbmux-install.py install userland/obj/MetalTest.app
 */
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#import <UIKit/UIKit.h>
#include <fcntl.h>
#include <simd/simd.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <unistd.h>

#define SCENE_SECONDS 5.0

static void
say(NSString *line)
{
	static int fd = -2;
	const char *text = [[NSString stringWithFormat:@"MetalTest: %@\n", line] UTF8String];

	if (fd == -2) {
		fd = open("/dev/console", O_WRONLY);
	}
	(void)write(fd >= 0 ? fd : 2, text, strlen(text));
}

/* what every shader gets at buffer 0 */
typedef struct {
	matrix_float4x4 mvp;
	vector_float4 a, b;
	float time, aspect;
	uint32_t n, pad;
} Uniforms;

static NSString *const kPrelude = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct U { float4x4 mvp; float4 a; float4 b; float time; float aspect; uint n; uint pad; };\n"
"struct VO { float4 pos [[position]]; float4 col; float2 uv; };\n"
"static float2 corner(uint vid) { return float2(float(vid & 1), float((vid >> 1) & 1)); }\n"
"static float3 hue(float h) { return clamp(abs(fract(h + float3(0.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0); }\n"
/* a rectangle: u.a = x, y, width, height in clip space (a triangle strip of 4); uv 0..1, v down */
"vertex VO rect_v(uint vid [[vertex_id]], constant U &u [[buffer(0)]]) {\n"
"  VO o; float2 q = corner(vid); o.pos = float4(u.a.xy + q * u.a.zw, 0.0, 1.0); o.uv = float2(q.x, 1.0 - q.y); o.col = u.b; return o; }\n"
"fragment float4 flat_f(VO in [[stage_in]]) { return in.col; }\n"
"fragment float4 tex_f(VO in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
"  return t.sample(s, in.uv) * in.col; }\n";

static matrix_float4x4
perspective(float fovy, float aspect, float near, float far)
{
	float f = 1.0f / tanf(fovy * 0.5f);

	return (matrix_float4x4){ {
		{ f / aspect, 0, 0, 0 }, { 0, f, 0, 0 }, { 0, 0, far / (near - far), -1 }, { 0, 0, far * near / (near - far), 0 } } };
}

static matrix_float4x4
translate(float x, float y, float z)
{
	matrix_float4x4 m = matrix_identity_float4x4;

	m.columns[3] = (vector_float4){ x, y, z, 1 };
	return m;
}

static matrix_float4x4
rotate(float angle, float x, float y, float z)
{
	vector_float3 a = simd_normalize((vector_float3){ x, y, z });
	float c = cosf(angle), s = sinf(angle), t = 1 - c;

	return (matrix_float4x4){ {
		{ t * a.x * a.x + c, t * a.x * a.y + s * a.z, t * a.x * a.z - s * a.y, 0 },
		{ t * a.x * a.y - s * a.z, t * a.y * a.y + c, t * a.y * a.z + s * a.x, 0 },
		{ t * a.x * a.z + s * a.y, t * a.y * a.z - s * a.x, t * a.z * a.z + c, 0 },
		{ 0, 0, 0, 1 } } };
}

@class Renderer;
typedef void (^SceneDraw)(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t);

@interface Scene : NSObject
@property NSString *name, *expect, *failure;
@property (copy) SceneDraw draw;
@property BOOL overlay;         /* UIKit views are moved over the Metal layer while it shows (the controller's) */
@end
@implementation Scene
@end

@interface Renderer : NSObject
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property NSMutableArray<Scene *> *scenes;
@property MTLPixelFormat format;
@property float aspect;
@property Scene *making;        /* the scene being set up: failures go to it */
@end

@implementation Renderer {
	id<MTLBuffer> _uniforms[3];
	NSUInteger _frame, _used;
	id<MTLSamplerState> _linear, _nearest, _mip;
}

- (void)fail:(NSString *)format, ... NS_FORMAT_FUNCTION(1, 2)
{
	va_list ap;

	va_start(ap, format);
	NSString *text = [[NSString alloc] initWithFormat:format arguments:ap];
	va_end(ap);
	if (_making != nil && _making.failure == nil) {
		_making.failure = text;
	}
	say([NSString stringWithFormat:@"FAILED: %@: %@", _making.name ?: @"?", text]);
}

- (id<MTLLibrary>)library:(NSString *)source
{
	NSError *error = nil;
	id<MTLLibrary> library = [_device newLibraryWithSource:[kPrelude stringByAppendingString:source] options:nil error:&error];

	if (library == nil) {
		[self fail:@"shader source: %@", error.localizedDescription];
	}
	return library;
}

- (id<MTLRenderPipelineState>)pipeline:(MTLRenderPipelineDescriptor *)d
{
	NSError *error = nil;
	id<MTLRenderPipelineState> state = d.vertexFunction != nil && d.fragmentFunction != nil
	    ? [_device newRenderPipelineStateWithDescriptor:d error:&error] : nil;

	if (state == nil) {
		[self fail:@"pipeline %@+%@: %@", d.vertexFunction.name, d.fragmentFunction.name,
		 error != nil ? error.localizedDescription : @"a function is missing"];
	}
	return state;
}

/* the usual pipeline: one colour attachment of FORMAT, optional blending, depth format, samples */
- (id<MTLRenderPipelineState>)pipelineIn:(id<MTLLibrary>)library vertex:(NSString *)v fragment:(NSString *)f
				  format:(MTLPixelFormat)format blend:(BOOL)blend depth:(MTLPixelFormat)depth samples:(NSUInteger)samples
{
	MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];

	if (library == nil) {
		return nil;
	}
	d.vertexFunction = [library newFunctionWithName:v];
	d.fragmentFunction = [library newFunctionWithName:f];
	d.colorAttachments[0].pixelFormat = format;
	if (blend) {
		d.colorAttachments[0].blendingEnabled = YES;
		d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
		d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
		d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
		d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
	}
	d.depthAttachmentPixelFormat = depth;
	if (depth == MTLPixelFormatDepth32Float_Stencil8) {
		d.stencilAttachmentPixelFormat = depth;
	}
	d.rasterSampleCount = samples;
	return [self pipeline:d];
}

- (id<MTLRenderPipelineState>)simple:(id<MTLLibrary>)library vertex:(NSString *)v fragment:(NSString *)f blend:(BOOL)blend
{
	return [self pipelineIn:library vertex:v fragment:f format:_format blend:blend depth:MTLPixelFormatInvalid samples:1];
}

- (id<MTLTexture>)texture:(MTLTextureDescriptor *)d
{
	id<MTLTexture> t = [_device newTextureWithDescriptor:d];

	if (t == nil) {
		[self fail:@"no texture of format %lu, %lux%lu, type %lu", (unsigned long)d.pixelFormat, (unsigned long)d.width,
		 (unsigned long)d.height, (unsigned long)d.textureType];
	}
	return t;
}

- (id<MTLTexture>)target:(MTLPixelFormat)format width:(NSUInteger)w height:(NSUInteger)h samples:(NSUInteger)samples
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];

	d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	if (samples > 1) {
		d.textureType = MTLTextureType2DMultisample;
		d.sampleCount = samples;
		d.storageMode = MTLStorageModePrivate;
	}
	return [self texture:d];
}

/* an RGBA8 checkerboard of CELL-pixel squares in two colours */
- (id<MTLTexture>)checker:(NSUInteger)size cell:(NSUInteger)cell a:(uint32_t)a b:(uint32_t)b mipmapped:(BOOL)mipmapped
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:size
											     height:size mipmapped:mipmapped];
	id<MTLTexture> t = [self texture:d];
	uint32_t *pixels = malloc(size * size * 4);

	for (NSUInteger y = 0; y < size; y++) {
		for (NSUInteger x = 0; x < size; x++) {
			pixels[y * size + x] = ((x / cell + y / cell) & 1) ? a : b;
		}
	}
	[t replaceRegion:MTLRegionMake2D(0, 0, size, size) mipmapLevel:0 withBytes:pixels bytesPerRow:size * 4];
	free(pixels);
	return t;
}

- (MTLRenderPassDescriptor *)passTo:(id<MTLTexture>)target clear:(MTLClearColor)colour
{
	MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor new];

	pass.colorAttachments[0].texture = target;
	pass.colorAttachments[0].loadAction = MTLLoadActionClear;
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	pass.colorAttachments[0].clearColor = colour;
	return pass;
}

/* this draw's uniforms: a piece of the frame's buffer, bound at buffer 0 of both stages */
- (void)uniforms:(Uniforms)u on:(id<MTLRenderCommandEncoder>)e
{
	id<MTLBuffer> buffer = _uniforms[_frame % 3];

	if (_used + 256 > buffer.length) {
		return;
	}
	u.aspect = _aspect;
	memcpy((char *)buffer.contents + _used, &u, sizeof(u));
	[e setVertexBuffer:buffer offset:_used atIndex:0];
	[e setFragmentBuffer:buffer offset:_used atIndex:0];
	_used += 256;
}

/* a rectangle (clip space x, y, w, h) in colour B with the pipeline bound */
- (void)rect:(vector_float4)rect colour:(vector_float4)colour time:(float)t on:(id<MTLRenderCommandEncoder>)e
{
	Uniforms u = { matrix_identity_float4x4, rect, colour, t, 0, 0, 0 };

	[self uniforms:u on:e];
	[e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
}

- (void)add:(NSString *)name expect:(NSString *)expect setup:(SceneDraw (^)(Renderer *r))setup
{
	Scene *scene = [Scene new];

	scene.name = name;
	scene.expect = expect;
	_making = scene;
	@try {
		scene.draw = setup(self);
	} @catch (NSException *exception) {
		[self fail:@"exception in setup: %@: %@", exception.name, exception.reason];
	}
	_making = nil;
	[_scenes addObject:scene];
	say([NSString stringWithFormat:@"scene %lu \"%@\": %@", (unsigned long)_scenes.count, name,
	     scene.failure != nil ? @"NOT SET UP" : @"set up"]);
}

- (instancetype)initWithDevice:(id<MTLDevice>)device format:(MTLPixelFormat)format
{
	self = [super init];
	_device = device;
	_format = format;
	_queue = [device newCommandQueue];
	_scenes = [NSMutableArray new];
	_aspect = 1;
	for (int i = 0; i < 3; i++) {
		_uniforms[i] = [device newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
	}
	MTLSamplerDescriptor *s = [MTLSamplerDescriptor new];
	s.sAddressMode = s.tAddressMode = MTLSamplerAddressModeRepeat;
	_nearest = [device newSamplerStateWithDescriptor:s];
	s.minFilter = s.magFilter = MTLSamplerMinMagFilterLinear;
	_linear = [device newSamplerStateWithDescriptor:s];
	s.mipFilter = MTLSamplerMipFilterLinear;
	_mip = [device newSamplerStateWithDescriptor:s];
	say([NSString stringWithFormat:@"device \"%@\", drawable format %lu", device.name, (unsigned long)format]);
	[self makeScenes];
	return self;
}

- (void)makeScenes
{
	id<MTLSamplerState> linear = _linear, nearest = _nearest, mip = _mip;
	MTLPixelFormat format = _format;

	/* 1 */
	[self add:@"Clear colours" expect:@"the whole screen red, green, blue, white: one a second" setup:^SceneDraw(Renderer *r) {
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			static const MTLClearColor colours[4] = { { 1, 0, 0, 1 }, { 0, 1, 0, 1 }, { 0, 0, 1, 1 }, { 1, 1, 1, 1 } };
			[[cb renderCommandEncoderWithDescriptor:[r passTo:target clear:colours[(int)t & 3]]] endEncoding];
		};
	}];

	/* 2 */
	[self add:@"Triangle, vertex buffer" expect:@"one spinning triangle, corners red green blue, smooth between"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "struct VI { float2 p [[attribute(0)]]; float4 c [[attribute(1)]]; };\n"
		    "vertex VO tri_v(VI in [[stage_in]], constant U &u [[buffer(0)]]) {\n"
		    "  VO o; float c = cos(u.time), s = sin(u.time); float2 p = float2(in.p.x * c - in.p.y * s, in.p.x * s + in.p.y * c);\n"
		    "  o.pos = float4(p.x, p.y * u.aspect, 0.0, 1.0); o.col = in.c; o.uv = float2(0.0); return o; }\n"];
		MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
		MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
		static const float vertices[] = { 0, 0.8f, 1, 0, 0, 1, -0.7f, -0.4f, 0, 1, 0, 1, 0.7f, -0.4f, 0, 0, 1, 1 };
		id<MTLBuffer> buffer = [r.device newBufferWithBytes:vertices length:sizeof(vertices) options:0];

		vd.attributes[0].format = MTLVertexFormatFloat2;
		vd.attributes[0].bufferIndex = 1;
		vd.attributes[1].format = MTLVertexFormatFloat4;
		vd.attributes[1].offset = 8;
		vd.attributes[1].bufferIndex = 1;
		vd.layouts[1].stride = 24;
		d.vertexFunction = [library newFunctionWithName:@"tri_v"];
		d.fragmentFunction = [library newFunctionWithName:@"flat_f"];
		d.vertexDescriptor = vd;
		d.colorAttachments[0].pixelFormat = r.format;
		id<MTLRenderPipelineState> pipeline = [r pipeline:d];
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0.1, 0.1, 0.12, 1)]];
			Uniforms u = { matrix_identity_float4x4, { 0 }, { 0 }, t, 0, 0, 0 };

			[e setRenderPipelineState:pipeline];
			[e setVertexBuffer:buffer offset:0 atIndex:1];
			[r uniforms:u on:e];
			[e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
			[e endEncoding];
		};
	}];

	/* 3 */
	[self add:@"Indexed draws, 16 and 32 bit" expect:@"top half: 8x8 grid of rainbow squares (16-bit indices); bottom half the same (32-bit, base vertex)"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "vertex VO grid_v(uint vid [[vertex_id]], constant U &u [[buffer(0)]], const device packed_float2 *p [[buffer(1)]]) {\n"
		    "  VO o; float2 q = p[vid]; o.pos = float4(u.a.xy + q * u.a.zw, 0.0, 1.0);\n"
		    "  o.col = float4(hue(q.x * 0.5 + q.y * 0.5 + u.time * 0.1), 1.0); o.uv = q; return o; }\n"];
		id<MTLRenderPipelineState> pipeline = [r simple:library vertex:@"grid_v" fragment:@"flat_f" blend:NO];
		enum { N = 8 };
		float positions[(N * N * 4 + 4) * 2];
		uint16_t index16[N * N * 6];
		uint32_t index32[N * N * 6];

		positions[0] = positions[1] = positions[2] = positions[3] = positions[4] = positions[5] = positions[6] = positions[7] = 9;
		for (int i = 0; i < N * N; i++) {
			float x = (float)(i % N) / N, y = (float)(i / N) / N, s = 0.8f / N;
			float *p = &positions[(4 + i * 4) * 2];

			p[0] = x; p[1] = y; p[2] = x + s; p[3] = y; p[4] = x; p[5] = y + s; p[6] = x + s; p[7] = y + s;
			for (int k = 0; k < 6; k++) {
				static const int corner[6] = { 0, 1, 2, 2, 1, 3 };
				index16[i * 6 + k] = (uint16_t)(4 + i * 4 + corner[k]);
				index32[i * 6 + k] = (uint32_t)(i * 4 + corner[k]);     /* (drawn with base vertex 4) */
			}
		}
		id<MTLBuffer> vb = [r.device newBufferWithBytes:positions length:sizeof(positions) options:0];
		id<MTLBuffer> i16 = [r.device newBufferWithBytes:index16 length:sizeof(index16) options:0];
		id<MTLBuffer> i32 = [r.device newBufferWithBytes:index32 length:sizeof(index32) options:0];
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0, 0, 0, 1)]];
			Uniforms u = { matrix_identity_float4x4, { -0.9f, 0.05f, 1.8f, 0.8f }, { 0 }, t, 0, 0, 0 };

			[e setRenderPipelineState:pipeline];
			[e setVertexBuffer:vb offset:0 atIndex:1];
			[r uniforms:u on:e];
			[e drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:N * N * 6 indexType:MTLIndexTypeUInt16 indexBuffer:i16 indexBufferOffset:0];
			u.a.y = -0.85f;
			[r uniforms:u on:e];
			[e drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:N * N * 6 indexType:MTLIndexTypeUInt32 indexBuffer:i32
			       indexBufferOffset:0 instanceCount:1 baseVertex:4 baseInstance:0];
			[e endEncoding];
		};
	}];

	/* 4 */
	[self add:@"Texture, nearest and linear" expect:@"black/yellow checkerboard scrolling: top half sharp squares, bottom half the same but soft-edged"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "fragment float4 scroll_f(VO in [[stage_in]], constant U &u [[buffer(0)]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
		    "  return t.sample(s, in.uv * float2(1.5, 1.5 * u.a.w / u.a.z / u.aspect) + float2(u.time * 0.05, 0.0)); }\n"];
		id<MTLRenderPipelineState> pipeline = [r simple:library vertex:@"rect_v" fragment:@"scroll_f" blend:NO];
		id<MTLTexture> texture = [r checker:16 cell:2 a:0xff00e0ff b:0xff101010 mipmapped:NO];
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0.2, 0, 0.2, 1)]];

			[e setRenderPipelineState:pipeline];
			[e setFragmentTexture:texture atIndex:0];
			[e setFragmentSamplerState:nearest atIndex:0];
			[r rect:(vector_float4){ -1, 0.01f, 2, 0.99f } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			[e setFragmentSamplerState:linear atIndex:0];
			[r rect:(vector_float4){ -1, -1, 2, 0.99f } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			[e endEncoding];
		};
	}];

	/* 5 */
	[self add:@"Mipmaps" expect:@"a floor to the horizon: checkerboard near, then bands red, green, blue, white further away, blending smoothly; right third: generated mipmaps, a fine checkerboard near and plain grey from where the left turns blue, no sparkle"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "vertex VO floor_v(uint vid [[vertex_id]], constant U &u [[buffer(0)]]) {\n"
		    "  VO o; float2 q = corner(vid); o.pos = u.mvp * float4((q.x - 0.5) * 40.0, 0.0, -q.y * 200.0, 1.0);\n"
		    "  o.uv = float2(q.x * 20.0, q.y * 100.0); o.col = float4(1.0); return o; }\n"];
		id<MTLRenderPipelineState> pipeline = [r simple:library vertex:@"floor_v" fragment:@"tex_f" blend:NO];
		id<MTLTexture> coloured = [r checker:256 cell:32 a:0xffffffff b:0xff202020 mipmapped:YES];
		/* (cells of 4 pixels: plain grey from level 3 on, where the left part turns blue) */
		id<MTLTexture> generated = [r checker:256 cell:4 a:0xffffffff b:0xff000000 mipmapped:YES];
		static const uint32_t level_colours[] = { 0, 0xff0000ff, 0xff00ff00, 0xffff0000, 0xffffffff, 0xff00ffff, 0xffff00ff, 0xffffff00, 0xff808080 };

		for (NSUInteger level = 1; level < coloured.mipmapLevelCount; level++) {
			NSUInteger size = MAX(256u >> level, 1u);
			uint32_t *pixels = malloc(size * size * 4);

			for (NSUInteger i = 0; i < size * size; i++) {
				pixels[i] = level_colours[MIN(level, 8u)];
			}
			[coloured replaceRegion:MTLRegionMake2D(0, 0, size, size) mipmapLevel:level withBytes:pixels bytesPerRow:size * 4];
			free(pixels);
		}
		if (generated != nil) {
			id<MTLCommandBuffer> cb = [r.queue commandBuffer];
			id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];

			[blit generateMipmapsForTexture:generated];
			[blit endEncoding];
			[cb commit];
			/* what the generated levels hold: a black and white checkerboard of 4-pixel cells is plain grey from
			 * the level where a cell is under a pixel (3) */
			[cb waitUntilCompleted];
			for (NSUInteger level = 3; level < generated.mipmapLevelCount; level += 3) {
				uint8_t px[4] = { 1, 2, 3, 4 };

				[generated getBytes:px bytesPerRow:4 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:level];
				say([NSString stringWithFormat:@"generated mipmaps %@: level %lu pixel %u %u %u %u (expected 127 or 128 grey; 1 2 3 4: nothing read)",
				     level >= 3 ? (abs((int)px[0] - 128) < 8 && abs((int)px[1] - 128) < 8 ? @"RIGHT" : @"WRONG") : @"-",
				     (unsigned long)level, px[0], px[1], px[2], px[3]]);
			}
		}
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0.3, 0.5, 0.9, 1)]];
			Uniforms u = { simd_mul(perspective(1.0f, r.aspect > 0 ? r.aspect : 1, 0.1f, 400), translate(0, -2.0f, fmodf(t, 4) * 2 - 4)),
				       { 0 }, { 0 }, t, 0, 0, 0 };

			[e setRenderPipelineState:pipeline];
			[e setFragmentSamplerState:mip atIndex:0];
			[r uniforms:u on:e];
			[e setFragmentTexture:coloured atIndex:0];
			[e setScissorRect:(MTLScissorRect){ 0, 0, target.width * 2 / 3, target.height }];
			[e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
			[e setFragmentTexture:generated atIndex:0];
			[e setScissorRect:(MTLScissorRect){ target.width * 2 / 3, 0, target.width - target.width * 2 / 3, target.height }];
			[e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
			[e endEncoding];
		};
	}];

	/* 6 */
	[self add:@"Alpha blending" expect:@"on white: three see-through discs, red green blue, circling; where two overlap the colours mix, edges smooth"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "fragment float4 disc_f(VO in [[stage_in]]) { float d = length(in.uv - 0.5) * 2.0;\n"
		    "  return float4(in.col.rgb, in.col.a * (1.0 - smoothstep(0.96, 1.0, d))); }\n"];
		id<MTLRenderPipelineState> pipeline = [r simple:library vertex:@"rect_v" fragment:@"disc_f" blend:YES];
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(1, 1, 1, 1)]];

			[e setRenderPipelineState:pipeline];
			for (int i = 0; i < 3; i++) {
				float angle = t + (float)i * 2.0944f, w = 1.0f, h = w * r.aspect;
				vector_float4 colour = { i == 0, i == 1, i == 2, 0.5f };

				[r rect:(vector_float4){ cosf(angle) * 0.3f - w / 2, sinf(angle) * 0.3f * r.aspect - h / 2, w, h } colour:colour time:t on:e];
			}
			[e endEncoding];
		};
	}];

	/* 7 */
	[self add:@"Depth test" expect:@"two cubes with six differently coloured faces turning through each other: the cut where they meet is a clean straight edge, no face shows through another"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "struct CV { packed_float3 p; packed_float3 c; };\n"
		    "vertex VO cube_v(uint vid [[vertex_id]], constant U &u [[buffer(0)]], const device CV *v [[buffer(1)]]) {\n"
		    "  VO o; o.pos = u.mvp * float4(float3(v[vid].p), 1.0); o.col = float4(float3(v[vid].c) * u.b.rgb, 1.0); o.uv = float2(0.0); return o; }\n"];
		id<MTLRenderPipelineState> pipeline = [r pipelineIn:library vertex:@"cube_v" fragment:@"flat_f" format:r.format blend:NO
							      depth:MTLPixelFormatDepth32Float samples:1];
		MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
		float vertices[36 * 6];
		static const float colours[6][3] = { { 1, 0.2f, 0.2f }, { 0.2f, 1, 0.2f }, { 0.2f, 0.2f, 1 }, { 1, 1, 0.2f }, { 1, 0.2f, 1 }, { 0.2f, 1, 1 } };

		for (int face = 0; face < 6; face++) {
			int axis = face / 2, a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
			float sign = (face & 1) ? 1 : -1;
			static const float corners[6][2] = { { -1, -1 }, { 1, -1 }, { -1, 1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };

			for (int k = 0; k < 6; k++) {
				float *v = &vertices[(face * 6 + k) * 6];

				v[axis] = sign; v[a1] = corners[k][0]; v[a2] = corners[k][1];
				v[3] = colours[face][0]; v[4] = colours[face][1]; v[5] = colours[face][2];
			}
		}
		id<MTLBuffer> buffer = [r.device newBufferWithBytes:vertices length:sizeof(vertices) options:0];
		dd.depthCompareFunction = MTLCompareFunctionLess;
		dd.depthWriteEnabled = YES;
		id<MTLDepthStencilState> depthState = [r.device newDepthStencilStateWithDescriptor:dd];
		__block id<MTLTexture> depth = nil;
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			if (depth == nil || depth.width != target.width || depth.height != target.height) {
				MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
													     width:target.width height:target.height mipmapped:NO];
				d.usage = MTLTextureUsageRenderTarget;
				d.storageMode = MTLStorageModePrivate;
				depth = [r.device newTextureWithDescriptor:d];
			}
			MTLRenderPassDescriptor *pass = [r passTo:target clear:MTLClearColorMake(0.05, 0.05, 0.08, 1)];
			pass.depthAttachment.texture = depth;
			pass.depthAttachment.loadAction = MTLLoadActionClear;
			pass.depthAttachment.storeAction = MTLStoreActionDontCare;
			pass.depthAttachment.clearDepth = 1;
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
			matrix_float4x4 projection = perspective(0.9f, r.aspect > 0 ? r.aspect : 1, 0.1f, 50);

			[e setRenderPipelineState:pipeline];
			[e setDepthStencilState:depthState];
			[e setVertexBuffer:buffer offset:0 atIndex:1];
			for (int i = 0; i < 2; i++) {
				matrix_float4x4 model = simd_mul(translate(i ? 0.6f : -0.6f, 0, -12), rotate(t * (i ? 0.9f : -0.7f), 0.3f, 1, i ? 0.5f : -0.2f));
				Uniforms u = { simd_mul(projection, model), { 0 }, { 1, 1, 1, 1 }, t, 0, 0, 0 };

				[r uniforms:u on:e];
				[e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:36];
			}
			[e endEncoding];
		};
	}];

	/* 8 */
	[self add:@"Stencil" expect:@"a rainbow visible only inside a disc that moves in a circle; everything outside the disc dark grey"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "fragment float4 mask_f(VO in [[stage_in]]) { if (length(in.uv - 0.5) > 0.5) discard_fragment(); return float4(0.0); }\n"
		    "fragment float4 rainbow_f(VO in [[stage_in]], constant U &u [[buffer(0)]]) { return float4(hue(in.uv.x + in.uv.y + u.time * 0.2), 1.0); }\n"];
		MTLPixelFormat ds = MTLPixelFormatDepth32Float_Stencil8;
		MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
		d.vertexFunction = [library newFunctionWithName:@"rect_v"];
		d.fragmentFunction = [library newFunctionWithName:@"mask_f"];
		d.colorAttachments[0].pixelFormat = r.format;
		d.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
		d.depthAttachmentPixelFormat = d.stencilAttachmentPixelFormat = ds;
		id<MTLRenderPipelineState> maskPipeline = [r pipeline:d];
		id<MTLRenderPipelineState> fillPipeline = [r pipelineIn:library vertex:@"rect_v" fragment:@"rainbow_f" format:r.format blend:NO depth:ds samples:1];
		MTLDepthStencilDescriptor *write = [MTLDepthStencilDescriptor new], *test = [MTLDepthStencilDescriptor new];
		MTLStencilDescriptor *s = [MTLStencilDescriptor new];

		s.stencilCompareFunction = MTLCompareFunctionAlways;
		s.depthStencilPassOperation = MTLStencilOperationReplace;
		write.frontFaceStencil = write.backFaceStencil = s;
		s = [MTLStencilDescriptor new];
		s.stencilCompareFunction = MTLCompareFunctionEqual;
		test.frontFaceStencil = test.backFaceStencil = s;
		id<MTLDepthStencilState> writeState = [r.device newDepthStencilStateWithDescriptor:write];
		id<MTLDepthStencilState> testState = [r.device newDepthStencilStateWithDescriptor:test];
		__block id<MTLTexture> stencil = nil;
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			if (stencil == nil || stencil.width != target.width || stencil.height != target.height) {
				MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:ds width:target.width
													    height:target.height mipmapped:NO];
				d.usage = MTLTextureUsageRenderTarget;
				d.storageMode = MTLStorageModePrivate;
				stencil = [r.device newTextureWithDescriptor:d];
			}
			MTLRenderPassDescriptor *pass = [r passTo:target clear:MTLClearColorMake(0.15, 0.15, 0.15, 1)];
			pass.depthAttachment.texture = pass.stencilAttachment.texture = stencil;
			pass.depthAttachment.loadAction = pass.stencilAttachment.loadAction = MTLLoadActionClear;
			pass.depthAttachment.storeAction = pass.stencilAttachment.storeAction = MTLStoreActionDontCare;
			pass.depthAttachment.clearDepth = 1;
			pass.stencilAttachment.clearStencil = 0;
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
			float w = 1.1f, h = w * r.aspect;

			[e setRenderPipelineState:maskPipeline];
			[e setDepthStencilState:writeState];
			[e setStencilReferenceValue:1];
			[r rect:(vector_float4){ cosf(t) * 0.35f - w / 2, sinf(t) * 0.35f * r.aspect - h / 2, w, h } colour:(vector_float4){ 0 } time:t on:e];
			[e setRenderPipelineState:fillPipeline];
			[e setDepthStencilState:testState];
			[r rect:(vector_float4){ -1, -1, 2, 2 } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			[e endEncoding];
		};
	}];

	/* 9 */
	[self add:@"Instancing, 16384 instances" expect:@"a galaxy: thousands of small coloured squares in spiral arms, turning smoothly"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "vertex VO swarm_v(uint vid [[vertex_id]], uint iid [[instance_id]], constant U &u [[buffer(0)]]) {\n"
		    "  VO o; float f = float(iid) / float(u.n); float radius = sqrt(f) * 0.95; float angle = f * 60.0 + u.time * (0.3 + 0.4 * (1.0 - f));\n"
		    "  float2 centre = float2(cos(angle), sin(angle)) * radius; float2 q = (corner(vid) - 0.5) * 0.012;\n"
		    "  o.pos = float4(centre.x + q.x, (centre.y + q.y) * u.aspect, 0.0, 1.0); o.col = float4(hue(f * 3.0 + u.time * 0.05), 1.0); o.uv = corner(vid); return o; }\n"];
		id<MTLRenderPipelineState> pipeline = [r simple:library vertex:@"swarm_v" fragment:@"flat_f" blend:NO];
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0, 0, 0.03, 1)]];
			Uniforms u = { matrix_identity_float4x4, { 0 }, { 0 }, t, 0, 16384, 0 };

			[e setRenderPipelineState:pipeline];
			[r uniforms:u on:e];
			[e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:16384];
			[e endEncoding];
		};
	}];

	/* 10 */
	[self add:@"Multisampling, 4 samples" expect:@"a white wheel of thin spokes turning slowly: top half smooth edges (4x MSAA resolved), bottom half the same wheel with visibly jagged edges (no MSAA)"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "vertex VO spoke_v(uint vid [[vertex_id]], constant U &u [[buffer(0)]]) {\n"
		    "  VO o; uint spoke = vid / 3; uint k = vid % 3; float angle = float(spoke) * 6.2831853 / float(u.n) + u.time * 0.05;\n"
		    "  float2 p = k == 0 ? float2(0.0) : float2(cos(angle + (k == 1 ? -0.012 : 0.012)), sin(angle + (k == 1 ? -0.012 : 0.012))) * 0.9;\n"
		    "  o.pos = float4(u.a.xy + (p * 0.5 + 0.5) * u.a.zw, 0.0, 1.0); o.col = float4(1.0); o.uv = float2(0.0); return o; }\n"];
		id<MTLRenderPipelineState> msaa = [r pipelineIn:library vertex:@"spoke_v" fragment:@"flat_f" format:r.format blend:NO
							  depth:MTLPixelFormatInvalid samples:4];
		id<MTLRenderPipelineState> plain = [r simple:library vertex:@"spoke_v" fragment:@"flat_f" blend:NO];
		__block id<MTLTexture> samples = nil;
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			if (samples == nil || samples.width != target.width || samples.height != target.height) {
				samples = [r target:r.format width:target.width height:target.height samples:4];
			}
			MTLRenderPassDescriptor *pass = [r passTo:samples clear:MTLClearColorMake(0, 0, 0, 1)];
			pass.colorAttachments[0].resolveTexture = target;
			pass.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
			float w = 1.9f, h = w * r.aspect;            /* a square on the screen */
			Uniforms u = { matrix_identity_float4x4, { -w / 2, 0.5f - h / 2, w, h }, { 0 }, t, 0, 48, 0 };

			[e setRenderPipelineState:msaa];
			[r uniforms:u on:e];
			[e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:48 * 3];
			[e endEncoding];
			pass = [r passTo:target clear:MTLClearColorMake(0, 0, 0, 1)];
			pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
			e = [cb renderCommandEncoderWithDescriptor:pass];
			u.a.y = -0.5f - h / 2;
			[e setRenderPipelineState:plain];
			[r uniforms:u on:e];
			[e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:48 * 3];
			[e endEncoding];
		};
	}];

	/* 11 */
	[self add:@"Render to RGBA16Float, then tonemap" expect:@"a bright sun moving over a dark blue gradient; the sun's core white, a wide soft orange glow around it, no hard square or banding"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "fragment float4 hdr_f(VO in [[stage_in]], constant U &u [[buffer(0)]]) {\n"
		    "  float2 c = float2(0.5 + 0.3 * cos(u.time), 0.5 + 0.3 * sin(u.time * 0.7)); float d = distance(in.uv, c);\n"
		    "  float3 sky = float3(0.02, 0.04, 0.15) * (1.0 - in.uv.y); return float4(sky + float3(40.0, 16.0, 4.0) * exp(-d * 14.0), 1.0); }\n"
		    "fragment float4 tonemap_f(VO in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
		    "  float3 c = t.sample(s, in.uv).rgb; return float4(c / (1.0 + c), 1.0); }\n"];
		id<MTLRenderPipelineState> hdr = [r pipelineIn:library vertex:@"rect_v" fragment:@"hdr_f" format:MTLPixelFormatRGBA16Float blend:NO
							 depth:MTLPixelFormatInvalid samples:1];
		id<MTLRenderPipelineState> tonemap = [r simple:library vertex:@"rect_v" fragment:@"tonemap_f" blend:NO];
		id<MTLTexture> offscreen = [r target:MTLPixelFormatRGBA16Float width:256 height:512 samples:1];
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:offscreen clear:MTLClearColorMake(0, 0, 0, 1)]];

			[e setRenderPipelineState:hdr];
			[r rect:(vector_float4){ -1, -1, 2, 2 } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			[e endEncoding];
			e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(1, 0, 1, 1)]];
			[e setRenderPipelineState:tonemap];
			[e setFragmentTexture:offscreen atIndex:0];
			[e setFragmentSamplerState:linear atIndex:0];
			[r rect:(vector_float4){ -1, -1, 2, 2 } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			[e endEncoding];
		};
	}];

	/* 12 */
	[self add:@"Three render targets in one pass" expect:@"three equal horizontal bands, each a moving stripe pattern: top RED stripes, middle GREEN, bottom BLUE"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "struct Out3 { float4 a [[color(0)]]; float4 b [[color(1)]]; float4 c [[color(2)]]; };\n"
		    "fragment Out3 mrt_f(VO in [[stage_in]], constant U &u [[buffer(0)]]) {\n"
		    "  float s = step(0.5, fract(in.uv.x * 6.0 + u.time * 0.3)); Out3 o; o.a = float4(s, 0.0, 0.0, 1.0); o.b = float4(0.0, s, 0.0, 1.0); o.c = float4(0.0, 0.0, s, 1.0); return o; }\n"];
		MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
		static const MTLPixelFormat formats[3] = { MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA16Float, MTLPixelFormatBGRA8Unorm };
		NSMutableArray<id<MTLTexture>> *targets = [NSMutableArray new];

		d.vertexFunction = [library newFunctionWithName:@"rect_v"];
		d.fragmentFunction = [library newFunctionWithName:@"mrt_f"];
		for (int i = 0; i < 3; i++) {
			id<MTLTexture> t = [r target:formats[i] width:256 height:128 samples:1];

			d.colorAttachments[i].pixelFormat = formats[i];
			if (t != nil) {
				[targets addObject:t];
			}
		}
		id<MTLRenderPipelineState> mrt = [r pipeline:d];
		id<MTLRenderPipelineState> show = [r simple:library vertex:@"rect_v" fragment:@"tex_f" blend:NO];
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor new];

			for (NSUInteger i = 0; i < targets.count; i++) {
				pass.colorAttachments[i].texture = targets[i];
				pass.colorAttachments[i].loadAction = MTLLoadActionClear;
				pass.colorAttachments[i].storeAction = MTLStoreActionStore;
				pass.colorAttachments[i].clearColor = MTLClearColorMake(1, 1, 1, 1);
			}
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
			[e setRenderPipelineState:mrt];
			[r rect:(vector_float4){ -1, -1, 2, 2 } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			[e endEncoding];
			e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0.3, 0.3, 0.3, 1)]];
			[e setRenderPipelineState:show];
			[e setFragmentSamplerState:linear atIndex:0];
			for (NSUInteger i = 0; i < targets.count; i++) {
				[e setFragmentTexture:targets[i] atIndex:0];
				[r rect:(vector_float4){ -1, 1 - (float)(i + 1) * 0.6f - 0.1f, 2, 0.58f } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			}
			[e endEncoding];
		};
	}];

	/* 13 */
	[self add:@"Compute kernel writes a texture" expect:@"a smooth moving plasma of all colours filling the screen (computed by a compute shader, 256x256, shown stretched)"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "kernel void plasma(texture2d<float, access::write> out [[texture(0)]], constant U &u [[buffer(0)]], uint2 id [[thread_position_in_grid]]) {\n"
		    "  if (id.x >= out.get_width() || id.y >= out.get_height()) return;\n"
		    "  float2 p = float2(id) / 256.0; float v = sin(p.x * 10.0 + u.time) + sin(p.y * 12.0 - u.time * 1.3) + sin((p.x + p.y) * 8.0 + u.time * 0.7);\n"
		    "  out.write(float4(hue(v * 0.16 + 0.5), 1.0), id); }\n"];
		NSError *error = nil;
		id<MTLFunction> function = [library newFunctionWithName:@"plasma"];
		id<MTLComputePipelineState> compute = function != nil ? [r.device newComputePipelineStateWithFunction:function error:&error] : nil;
		MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:256 height:256 mipmapped:NO];

		if (compute == nil) {
			[r fail:@"compute pipeline: %@", error != nil ? error.localizedDescription : @"no function"];
		}
		d.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
		id<MTLTexture> texture = [r texture:d];
		id<MTLBuffer> ub = [r.device newBufferWithLength:256 options:MTLResourceStorageModeShared];
		id<MTLRenderPipelineState> show = [r simple:library vertex:@"rect_v" fragment:@"tex_f" blend:NO];
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			Uniforms u = { matrix_identity_float4x4, { 0 }, { 0 }, t, 0, 0, 0 };
			id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];

			memcpy(ub.contents, &u, sizeof(u));
			[c setComputePipelineState:compute];
			[c setTexture:texture atIndex:0];
			[c setBuffer:ub offset:0 atIndex:0];
			[c dispatchThreadgroups:MTLSizeMake(32, 32, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
			[c endEncoding];
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(1, 0, 1, 1)]];
			[e setRenderPipelineState:show];
			[e setFragmentTexture:texture atIndex:0];
			[e setFragmentSamplerState:linear atIndex:0];
			[r rect:(vector_float4){ -1, -1, 2, 2 } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			[e endEncoding];
		};
	}];

	/* 14 */
	[self add:@"ASTC compressed textures" expect:@"a 4x4 grid of 14 flat coloured tiles on grey, every tile ONE even colour and all different (rainbow order): the 14 ASTC block sizes 4x4 .. 12x12; last row's 2 tiles: sRGB 4x4 and 8x8, the same dark brown (0.5 0.35 0.2 decoded to linear: about 55 26 8 on this non-sRGB drawable; a mid grey-orange would be the value NOT decoded)"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@""];
		id<MTLRenderPipelineState> show = [r simple:library vertex:@"rect_v" fragment:@"tex_f" blend:NO];
		static const struct { MTLPixelFormat format; int bw, bh; } kinds[16] = {
			{ MTLPixelFormatASTC_4x4_LDR, 4, 4 }, { MTLPixelFormatASTC_5x4_LDR, 5, 4 }, { MTLPixelFormatASTC_5x5_LDR, 5, 5 },
			{ MTLPixelFormatASTC_6x5_LDR, 6, 5 }, { MTLPixelFormatASTC_6x6_LDR, 6, 6 }, { MTLPixelFormatASTC_8x5_LDR, 8, 5 },
			{ MTLPixelFormatASTC_8x6_LDR, 8, 6 }, { MTLPixelFormatASTC_8x8_LDR, 8, 8 }, { MTLPixelFormatASTC_10x5_LDR, 10, 5 },
			{ MTLPixelFormatASTC_10x6_LDR, 10, 6 }, { MTLPixelFormatASTC_10x8_LDR, 10, 8 }, { MTLPixelFormatASTC_10x10_LDR, 10, 10 },
			{ MTLPixelFormatASTC_12x10_LDR, 12, 10 }, { MTLPixelFormatASTC_12x12_LDR, 12, 12 },
			{ MTLPixelFormatASTC_4x4_sRGB, 4, 4 }, { MTLPixelFormatASTC_8x8_sRGB, 8, 8 } };
		NSMutableArray *textures = [NSMutableArray new];

		for (int i = 0; i < 16; i++) {
			/* a 120x120 texture of void-extent blocks: every block one colour (FC FD FF FF FF FF FF FF, then R G B A, 16 bits each) */
			NSUInteger across = (120 + kinds[i].bw - 1) / kinds[i].bw, down = (120 + kinds[i].bh - 1) / kinds[i].bh;
			uint8_t *blocks = malloc(across * down * 16);
			float h = (float)i / 14.0f, c[3];
			MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kinds[i].format width:120 height:120 mipmapped:NO];

			for (int k = 0; k < 3; k++) {
				float x = fabsf(fmodf(h + (float[]){ 0, 2.0f / 3, 1.0f / 3 }[k], 1.0f) * 6 - 3) - 1;
				c[k] = i < 14 ? fminf(fmaxf(x, 0), 1) : (float[]){ 0.5f, 0.35f, 0.2f }[k];
			}
			for (NSUInteger b = 0; b < across * down; b++) {
				uint8_t *p = blocks + b * 16;
				static const uint8_t head[8] = { 0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };

				memcpy(p, head, 8);
				for (int k = 0; k < 4; k++) {
					uint16_t v = k < 3 ? (uint16_t)(c[k] * 65535.0f) : 0xffff;
					p[8 + k * 2] = (uint8_t)v;
					p[9 + k * 2] = (uint8_t)(v >> 8);
				}
			}
			id<MTLTexture> t = [r texture:d];
			[t replaceRegion:MTLRegionMake2D(0, 0, 120, 120) mipmapLevel:0 withBytes:blocks bytesPerRow:across * 16];
			free(blocks);
			[textures addObject:t ?: (id)[NSNull null]];
		}
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0.25, 0.25, 0.25, 1)]];

			[e setRenderPipelineState:show];
			[e setFragmentSamplerState:nearest atIndex:0];
			for (int i = 0; i < 16; i++) {
				if (textures[i] == (id)[NSNull null]) {
					continue;
				}
				[e setFragmentTexture:textures[i] atIndex:0];
				[r rect:(vector_float4){ -0.95f + (float)(i % 4) * 0.48f, 0.45f - (float)(i / 4) * 0.36f, 0.44f, 0.32f }
				 colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			}
			[e endEncoding];
		};
	}];

	/* 15 */
	[self add:@"Array, cube and 3D textures" expect:@"three bands. Top: four vertical stripes red, green, blue, yellow (array layers). Middle: six colours flowing past as the view turns inside a cube (cube map). Bottom: a colour slowly fading from black to white, tinted by position (3D texture slices)"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "fragment float4 array_f(VO in [[stage_in]], texture2d_array<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
		    "  return t.sample(s, in.uv, uint(in.uv.x * 3.999)); }\n"
		    "fragment float4 cube_f(VO in [[stage_in]], constant U &u [[buffer(0)]], texturecube<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
		    "  float a = u.time * 0.6; float3 d = float3(in.uv.x * 2.0 - 1.0, (in.uv.y * 2.0 - 1.0) * 0.4, 1.0);\n"
		    "  d = float3(d.x * cos(a) + d.z * sin(a), d.y + 0.6 * sin(a * 0.5), d.z * cos(a) - d.x * sin(a)); return t.sample(s, d); }\n"
		    "fragment float4 volume_f(VO in [[stage_in]], constant U &u [[buffer(0)]], texture3d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
		    "  return t.sample(s, float3(in.uv, fract(u.time * 0.2))); }\n"];
		id<MTLRenderPipelineState> arrayP = [r simple:library vertex:@"rect_v" fragment:@"array_f" blend:NO];
		id<MTLRenderPipelineState> cubeP = [r simple:library vertex:@"rect_v" fragment:@"cube_f" blend:NO];
		id<MTLRenderPipelineState> volumeP = [r simple:library vertex:@"rect_v" fragment:@"volume_f" blend:NO];
		static const uint32_t colours[6] = { 0xff0000ff, 0xff00ff00, 0xffff0000, 0xff00ffff, 0xffff00ff, 0xffffff00 };
		uint32_t pixels[16 * 16 * 16];
		MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:16 mipmapped:NO];

		d.textureType = MTLTextureType2DArray;
		d.arrayLength = 4;
		id<MTLTexture> array = [r texture:d];
		for (NSUInteger layer = 0; layer < 4; layer++) {
			for (int i = 0; i < 256; i++) {
				pixels[i] = colours[layer];
			}
			[array replaceRegion:MTLRegionMake2D(0, 0, 16, 16) mipmapLevel:0 slice:layer withBytes:pixels bytesPerRow:64 bytesPerImage:0];
		}
		d = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm size:16 mipmapped:NO];
		id<MTLTexture> cube = [r texture:d];
		for (NSUInteger face = 0; face < 6; face++) {
			for (int i = 0; i < 256; i++) {
				pixels[i] = ((i % 16) == 0 || (i / 16) == 0) ? 0xff000000 : colours[face];
			}
			[cube replaceRegion:MTLRegionMake2D(0, 0, 16, 16) mipmapLevel:0 slice:face withBytes:pixels bytesPerRow:64 bytesPerImage:0];
		}
		d = [MTLTextureDescriptor new];
		d.textureType = MTLTextureType3D;
		d.pixelFormat = MTLPixelFormatRGBA8Unorm;
		d.width = d.height = d.depth = 16;
		id<MTLTexture> volume = [r texture:d];
		for (uint32_t z = 0; z < 16; z++) {
			for (uint32_t y = 0; y < 16; y++) {
				for (uint32_t x = 0; x < 16; x++) {
					pixels[(z * 16 + y) * 16 + x] = 0xff000000 | (z * 17 * (x + 1) / 16) | (z * 17 * (y + 1) / 16) << 8 | (z * 17) << 16;
				}
			}
		}
		[volume replaceRegion:MTLRegionMake3D(0, 0, 0, 16, 16, 16) mipmapLevel:0 slice:0 withBytes:pixels bytesPerRow:64 bytesPerImage:64 * 16];
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0.2, 0.2, 0.2, 1)]];
			id<MTLRenderPipelineState> pipelines[3] = { arrayP, cubeP, volumeP };
			id<MTLTexture> textures[3] = { array, cube, volume };

			for (int i = 0; i < 3; i++) {
				[e setRenderPipelineState:pipelines[i]];
				[e setFragmentTexture:textures[i] atIndex:0];
				[e setFragmentSamplerState:i == 0 ? nearest : linear atIndex:0];
				[r rect:(vector_float4){ -1, 1 - (float)(i + 1) * 0.6f - 0.1f, 2, 0.58f } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			}
			[e endEncoding];
		};
	}];

	/* 16 */
	[self add:@"Texture formats" expect:@"eight horizontal strips, each a smooth ramp dark (left) to bright (right), top to bottom: R8 red, RG8 yellow, R16Float red, RGBA16Float white, RGB10A2 cyan, RG11B10Float magenta, BGRA8 sRGB grey, R32Float red"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@""];
		id<MTLRenderPipelineState> show = [r simple:library vertex:@"rect_v" fragment:@"tex_f" blend:NO];
		NSMutableArray *textures = [NSMutableArray new];
		static const MTLPixelFormat formats[8] = { MTLPixelFormatR8Unorm, MTLPixelFormatRG8Unorm, MTLPixelFormatR16Float, MTLPixelFormatRGBA16Float,
			MTLPixelFormatRGB10A2Unorm, MTLPixelFormatRG11B10Float, MTLPixelFormatBGRA8Unorm_sRGB, MTLPixelFormatR32Float };
		static const NSUInteger sizes[8] = { 1, 2, 2, 8, 4, 4, 4, 4 };

		for (int i = 0; i < 8; i++) {
			enum { W = 64 };
			uint8_t row[W * 8];
			MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:formats[i] width:W height:1 mipmapped:NO];
			id<MTLTexture> t = [r texture:d];

			for (int x = 0; x < W; x++) {
				float v = (float)x / (W - 1);
				uint8_t *p = row + (NSUInteger)x * sizes[i];
				__fp16 half = (__fp16)v, one = (__fp16)1.0f;
				uint32_t word;

				switch (i) {
				case 0: p[0] = (uint8_t)(v * 255); break;
				case 1: p[0] = p[1] = (uint8_t)(v * 255); break;
				case 2: memcpy(p, &half, 2); break;
				case 3: memcpy(p, &half, 2); memcpy(p + 2, &half, 2); memcpy(p + 4, &half, 2); memcpy(p + 6, &one, 2); break;
				case 4: word = 3u << 30 | (uint32_t)(v * 1023) << 20 | (uint32_t)(v * 1023) << 10; memcpy(p, &word, 4); break;   /* B, G; R = 0 */
				case 5: {
					/* 11-bit floats (5 exponent, 6 mantissa) for R and G, 10-bit (5, 5) for B: v as 2^(e-15) * (1 + m/64) */
					uint32_t r11 = 0, b10 = 0;
					if (v > 0) {
						int e;
						float m = frexpf(v, &e);         /* v = m * 2^e, m in [0.5, 1) */
						r11 = (uint32_t)(e - 1 + 15) << 6 | ((uint32_t)((m * 2 - 1) * 64) & 63);
						b10 = (uint32_t)(e - 1 + 15) << 5 | ((uint32_t)((m * 2 - 1) * 32) & 31);
					}
					word = b10 << 22 | r11;
					memcpy(p, &word, 4);
					break;
				}
				case 6: p[0] = p[1] = p[2] = (uint8_t)(v * 255); p[3] = 255; break;
				default: memcpy(p, &v, 4); break;
				}
			}
			[t replaceRegion:MTLRegionMake2D(0, 0, W, 1) mipmapLevel:0 withBytes:row bytesPerRow:W * sizes[i]];
			[textures addObject:t ?: (id)[NSNull null]];
		}
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0, 0, 0.3, 1)]];

			[e setRenderPipelineState:show];
			for (int i = 0; i < 8; i++) {
				if (textures[i] == (id)[NSNull null]) {
					continue;
				}
				[e setFragmentTexture:textures[i] atIndex:0];
				[e setFragmentSamplerState:i == 7 ? nearest : linear atIndex:0];
				[r rect:(vector_float4){ -0.95f, 0.7f - (float)i * 0.21f, 1.9f, 0.18f } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			}
			[e endEncoding];
		};
	}];

	/* 17 */
	[self add:@"Points, lines, strips" expect:@"top: a ring of coloured squares growing in size around it, 4 to 50 pixels (points). Middle: a white zigzag line and a green sine curve (lines, line strip). Bottom: a rainbow ribbon waving (triangle strip)"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "struct PO { float4 pos [[position]]; float4 col; float size [[point_size]]; };\n"
		    "vertex PO point_v(uint vid [[vertex_id]], constant U &u [[buffer(0)]]) {\n"
		    "  PO o; float a = float(vid) / float(u.n) * 6.2831853 + u.time * 0.3; o.pos = float4(cos(a) * 0.6, 0.62 + sin(a) * 0.3 * u.aspect, 0.0, 1.0);\n"
		    "  o.col = float4(hue(float(vid) / float(u.n)), 1.0); o.size = 4.0 + float(vid) * 2.0; return o; }\n"
		    "fragment float4 point_f(PO in [[stage_in]]) { return in.col; }\n"
		    "vertex VO line_v(uint vid [[vertex_id]], constant U &u [[buffer(0)]]) {\n"
		    "  VO o; float x = float(vid) / float(u.n - 1); float y = u.b.x > 0.5 ? sin(x * 12.0 + u.time * 2.0) * 0.1 : ((vid & 1) != 0 ? 0.1 : -0.1);\n"
		    "  o.pos = float4(x * 1.8 - 0.9, u.a.y + y, 0.0, 1.0); o.col = u.b.x > 0.5 ? float4(0.2, 1.0, 0.2, 1.0) : float4(1.0); o.uv = float2(0.0); return o; }\n"
		    "vertex VO ribbon_v(uint vid [[vertex_id]], constant U &u [[buffer(0)]]) {\n"
		    "  VO o; float x = float(vid / 2) / float(u.n / 2 - 1); float y = sin(x * 8.0 + u.time * 1.5) * 0.12 + ((vid & 1) != 0 ? 0.08 : -0.08);\n"
		    "  o.pos = float4(x * 1.8 - 0.9, -0.62 + y, 0.0, 1.0); o.col = float4(hue(x + u.time * 0.1), 1.0); o.uv = float2(0.0); return o; }\n"];
		id<MTLRenderPipelineState> points = [r simple:library vertex:@"point_v" fragment:@"point_f" blend:NO];
		id<MTLRenderPipelineState> lines = [r simple:library vertex:@"line_v" fragment:@"flat_f" blend:NO];
		id<MTLRenderPipelineState> ribbon = [r simple:library vertex:@"ribbon_v" fragment:@"flat_f" blend:NO];
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0.05, 0.05, 0.05, 1)]];
			Uniforms u = { matrix_identity_float4x4, { 0 }, { 0 }, t, 0, 24, 0 };

			[e setRenderPipelineState:points];
			[r uniforms:u on:e];
			[e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:24];
			[e setRenderPipelineState:lines];
			u.n = 40; u.a.y = 0.12f;
			[r uniforms:u on:e];
			[e drawPrimitives:MTLPrimitiveTypeLine vertexStart:0 vertexCount:40];
			u.n = 200; u.a.y = -0.15f; u.b.x = 1;
			[r uniforms:u on:e];
			[e drawPrimitives:MTLPrimitiveTypeLineStrip vertexStart:0 vertexCount:200];
			[e setRenderPipelineState:ribbon];
			u.n = 128;
			[r uniforms:u on:e];
			[e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:128];
			[e endEncoding];
		};
	}];

	/* 18 */
	[self add:@"Viewports and scissor" expect:@"four quarters, each the same rainbow picture squeezed into its quarter (four viewports); over them one sharp-edged square window moving in a circle, inside it everything is inverted to plain white"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "fragment float4 uvhue_f(VO in [[stage_in]], constant U &u [[buffer(0)]]) {\n"
		    "  float ring = step(0.5, fract(length(in.uv - 0.5) * 8.0 - u.time * 0.5)); return float4(hue(in.uv.x) * (0.5 + 0.5 * ring), 1.0); }\n"];
		id<MTLRenderPipelineState> picture = [r simple:library vertex:@"rect_v" fragment:@"uvhue_f" blend:NO];
		id<MTLRenderPipelineState> flat = [r simple:library vertex:@"rect_v" fragment:@"flat_f" blend:NO];
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0, 0, 0, 1)]];
			double w = (double)target.width, h = (double)target.height;
			NSUInteger side = target.width / 3;

			[e setRenderPipelineState:picture];
			for (int i = 0; i < 4; i++) {
				[e setViewport:(MTLViewport){ (i & 1) * w / 2 + 4, (i >> 1) * h / 2 + 4, w / 2 - 8, h / 2 - 8, 0, 1 }];
				[r rect:(vector_float4){ -1, -1, 2, 2 } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			}
			[e setViewport:(MTLViewport){ 0, 0, w, h, 0, 1 }];
			[e setScissorRect:(MTLScissorRect){ (NSUInteger)((w - side) / 2 + cos(t) * w / 4), (NSUInteger)((h - side) / 2 + sin(t) * h / 4), side, side }];
			[e setRenderPipelineState:flat];
			[r rect:(vector_float4){ -1, -1, 2, 2 } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			[e endEncoding];
		};
	}];

	/* 19 */
	[self add:@"Discard and derivatives" expect:@"left half: a blue sheet full of round holes showing orange behind, holes drifting. Right half: thin white grid lines of the SAME thickness near and far on a tilted pattern (fwidth), smooth not broken"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "fragment float4 holes_f(VO in [[stage_in]], constant U &u [[buffer(0)]]) {\n"
		    "  float2 p = fract(in.uv * float2(4.0, 8.0) + float2(u.time * 0.1, 0.0)) - 0.5; if (length(p) < 0.3) discard_fragment(); return float4(0.1, 0.3, 0.9, 1.0); }\n"
		    "fragment float4 grid_f(VO in [[stage_in]], constant U &u [[buffer(0)]]) {\n"
		    "  float2 p = float2(in.uv.x - 0.5, 1.0) / (in.uv.y * 0.9 + 0.1) * 3.0 + float2(0.0, u.time * 0.5);\n"
		    "  float2 g = abs(fract(p) - 0.5) / fwidth(p); float line = 1.0 - clamp(min(g.x, g.y), 0.0, 1.0); return float4(float3(line), 1.0); }\n"];
		id<MTLRenderPipelineState> holes = [r simple:library vertex:@"rect_v" fragment:@"holes_f" blend:NO];
		id<MTLRenderPipelineState> grid = [r simple:library vertex:@"rect_v" fragment:@"grid_f" blend:NO];
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(1, 0.5, 0, 1)]];

			[e setRenderPipelineState:holes];
			[r rect:(vector_float4){ -1, -1, 1, 2 } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			[e setRenderPipelineState:grid];
			[r rect:(vector_float4){ 0, -1, 1, 2 } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			[e endEncoding];
		};
	}];

	/* 20 */
	[self add:@"Blit copy and read back" expect:@"the whole screen GREEN = a rendered texture, copied by the blit encoder and read back by the CPU, held the right pixels. RED = wrong pixels came back (the console says which)"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@""];
		id<MTLRenderPipelineState> flat = [r pipelineIn:library vertex:@"rect_v" fragment:@"flat_f" format:MTLPixelFormatRGBA8Unorm blend:NO
							  depth:MTLPixelFormatInvalid samples:1];
		id<MTLTexture> source = [r target:MTLPixelFormatRGBA8Unorm width:64 height:64 samples:1];
		id<MTLTexture> copy = [r target:MTLPixelFormatRGBA8Unorm width:64 height:64 samples:1];
		__block int verdict = 0, frames = 0;         /* 0: not yet, 1: right, 2: wrong */
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			if (verdict == 0 && flat != nil && source != nil && copy != nil && (frames++ % 30) == 0) {
				id<MTLCommandBuffer> work = [r.queue commandBuffer];
				id<MTLRenderCommandEncoder> e = [work renderCommandEncoderWithDescriptor:[r passTo:source clear:MTLClearColorMake(0, 0, 1, 1)]];

				/* the left half (x 0..31) 25% red 50% green 75% blue, the right half the clear's blue */
				[e setRenderPipelineState:flat];
				[r rect:(vector_float4){ -1, -1, 1, 2 } colour:(vector_float4){ 0.25f, 0.5f, 0.75f, 1 } time:t on:e];
				[e endEncoding];
				id<MTLBlitCommandEncoder> blit = [work blitCommandEncoder];
				[blit copyFromTexture:source sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(64, 64, 1)
					    toTexture:copy destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
				[blit endEncoding];
				[work commit];
				[work waitUntilCompleted];
				uint8_t pixels[64 * 64 * 4];
				[copy getBytes:pixels bytesPerRow:256 fromRegion:MTLRegionMake2D(0, 0, 64, 64) mipmapLevel:0];
				const uint8_t *left = &pixels[(32 * 64 + 8) * 4], *right = &pixels[(32 * 64 + 56) * 4];
				BOOL ok = abs(left[0] - 64) <= 2 && abs(left[1] - 128) <= 2 && abs(left[2] - 191) <= 2 && left[3] == 255 &&
				    right[0] == 0 && right[1] == 0 && right[2] == 255 && right[3] == 255;
				verdict = ok ? 1 : 2;
				say([NSString stringWithFormat:@"blit/readback %@: left pixel %u %u %u %u (expected 64 128 191 255), right %u %u %u %u (expected 0 0 255 255)",
				     ok ? @"RIGHT" : @"WRONG", left[0], left[1], left[2], left[3], right[0], right[1], right[2], right[3]]);
			}
			[[cb renderCommandEncoderWithDescriptor:[r passTo:target clear:verdict == 1 ? MTLClearColorMake(0, 0.7, 0, 1)
								   : verdict == 2 ? MTLClearColorMake(0.8, 0, 0, 1) : MTLClearColorMake(0.3, 0.3, 0.3, 1)]] endEncoding];
		};
	}];

	/* 21 */
	[self add:@"Load: 400 draws, 240000 triangles" expect:@"a field of 400 turning coloured pinwheels in a 16x25 grid, all moving smoothly; the frame rate at the top should stay near the display's"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "vertex VO wheel_v(uint vid [[vertex_id]], constant U &u [[buffer(0)]]) {\n"
		    "  VO o; uint tri = vid / 3; uint k = vid % 3; float a = (float(tri) + (k == 2 ? 0.6 : 0.0)) * 6.2831853 / float(u.n) + u.time * u.b.w;\n"
		    "  float2 p = k == 0 ? float2(0.0) : float2(cos(a), sin(a)); o.pos = float4(u.a.xy + p * u.a.zw, 0.0, 1.0);\n"
		    "  o.col = float4(hue(float(tri) / float(u.n) + u.b.x), 1.0); o.uv = float2(0.0); return o; }\n"];
		id<MTLRenderPipelineState> pipeline = [r simple:library vertex:@"wheel_v" fragment:@"flat_f" blend:NO];
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0, 0, 0, 1)]];

			[e setRenderPipelineState:pipeline];
			for (int i = 0; i < 400; i++) {
				float x = ((float)(i % 16) + 0.5f) / 16 * 2 - 1, y = ((float)(i / 16) + 0.5f) / 25 * 1.8f - 0.95f;
				Uniforms u = { matrix_identity_float4x4, { x, y, 0.058f, 0.058f * r.aspect }, { (float)i / 400, 0, 0, (i & 1) ? 1.0f : -1.0f }, t, 0, 600, 0 };

				[r uniforms:u on:e];
				[e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:600 * 3];
			}
			[e endEncoding];
		};
	}];

	/* 22 */
	[self add:@"Layered rendering, 4 layers in one pass" expect:@"four quarters: red, green (top), blue, yellow (bottom) -- ONE instanced draw sent each instance to its own layer of an array texture ([[render_target_array_index]]); a quarter black or all four the same = the layers were not told apart (the console says each layer's centre pixel)"
	    setup:^SceneDraw(Renderer *r) {
		id<MTLLibrary> library = [r library:@
		    "struct LO { float4 pos [[position]]; float4 col; uint layer [[render_target_array_index]]; };\n"
		    "struct LI { float4 pos [[position]]; float4 col; };\n"
		    "vertex LO layer_v(uint vid [[vertex_id]], uint iid [[instance_id]]) {\n"
		    "  LO o; float2 q = corner(vid); o.pos = float4(q * 2.0 - 1.0, 0.0, 1.0);\n"
		    "  o.col = float4((iid == 0 || iid == 3) ? 1.0 : 0.0, (iid == 1 || iid == 3) ? 1.0 : 0.0, iid == 2 ? 1.0 : 0.0, 1.0);\n"
		    "  o.layer = iid; return o; }\n"
		    "fragment float4 layer_f(LI in [[stage_in]]) { return in.col; }\n"
		    "fragment float4 layers_show_f(VO in [[stage_in]], texture2d_array<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
		    "  uint layer = (in.uv.x < 0.5 ? 0 : 1) + (in.uv.y < 0.5 ? 0 : 2); return t.sample(s, fract(in.uv * 2.0), layer); }\n"];
		MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
		MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:64 height:64 mipmapped:NO];

		if (library != nil) {
			pd.vertexFunction = [library newFunctionWithName:@"layer_v"];
			pd.fragmentFunction = [library newFunctionWithName:@"layer_f"];
		}
		pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
		pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
		id<MTLRenderPipelineState> fill = [r pipeline:pd];
		id<MTLRenderPipelineState> show = [r simple:library vertex:@"rect_v" fragment:@"layers_show_f" blend:NO];
		td.textureType = MTLTextureType2DArray;
		td.arrayLength = 4;
		td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		td.storageMode = MTLStorageModeShared;
		id<MTLTexture> layers = [r texture:td];
		__block BOOL checked = NO;
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			MTLRenderPassDescriptor *pass = [r passTo:layers clear:MTLClearColorMake(0, 0, 0, 1)];

			pass.renderTargetArrayLength = 4;
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
			if (fill != nil) {
				[e setRenderPipelineState:fill];
				[e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:4];
			}
			[e endEncoding];
			e = [cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0.2, 0.2, 0.2, 1)]];
			if (show != nil && layers != nil) {
				[e setRenderPipelineState:show];
				[e setFragmentSamplerState:nearest atIndex:0];
				[e setFragmentTexture:layers atIndex:0];
				[r rect:(vector_float4){ -1, -1, 2, 2 } colour:(vector_float4){ 1, 1, 1, 1 } time:t on:e];
			}
			[e endEncoding];
			if (!checked && layers != nil) {
				checked = YES;
				[cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
					static const uint8_t want[4][3] = { { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 255, 0 } };
					NSMutableString *line = [NSMutableString string];
					BOOL right = YES;

					for (NSUInteger i = 0; i < 4; i++) {
						uint8_t px[4] = { 1, 2, 3, 4 };

						[layers getBytes:px bytesPerRow:4 bytesPerImage:4 fromRegion:MTLRegionMake2D(32, 32, 1, 1) mipmapLevel:0 slice:i];
						right = right && px[0] == want[i][0] && px[1] == want[i][1] && px[2] == want[i][2];
						[line appendFormat:@" layer %lu: %u %u %u", (unsigned long)i, px[0], px[1], px[2]];
					}
					say([NSString stringWithFormat:@"layered rendering %@:%@ (expected 255 0 0, 0 255 0, 0 0 255, 255 255 0)", right ? @"RIGHT" : @"WRONG", line]);
				}];
			}
		};
	}];

	/* 23 */
	[self add:@"UIKit over Metal" expect:@"a slow colour wash (Metal) and over it, drawn by UIKit: a white box sliding left to right and back, smoothly (an animation the render server runs); in it a counter going up every frame (a layer redrawn in place); a spinning grey square (a transform set every frame). Nothing of them left behind, none frozen"
	    setup:^SceneDraw(Renderer *r) {
		return ^(Renderer *r, id<MTLCommandBuffer> cb, id<MTLTexture> target, float t) {
			[[cb renderCommandEncoderWithDescriptor:[r passTo:target clear:MTLClearColorMake(0.25 + 0.2 * sin(t), 0.25 + 0.2 * sin(t * 0.7 + 2), 0.3 + 0.2 * sin(t * 0.4 + 4), 1)]] endEncoding];
		};
	}];
	_scenes.lastObject.overlay = YES;
}

- (void)draw:(Scene *)scene into:(id<CAMetalDrawable>)drawable time:(float)t
{
	id<MTLCommandBuffer> cb = [_queue commandBuffer];

	_frame++;
	_used = 0;
	_aspect = (float)drawable.texture.width / (float)drawable.texture.height;
	if (scene.failure == nil && scene.draw != nil) {
		@try {
			scene.draw(self, cb, drawable.texture, t);
		} @catch (NSException *exception) {
			_making = scene;
			[self fail:@"exception while drawing: %@: %@", exception.name, exception.reason];
			_making = nil;
			scene.draw = nil;
		}
	} else {
		[[cb renderCommandEncoderWithDescriptor:[self passTo:drawable.texture clear:MTLClearColorMake(0.35, 0, 0, 1)]] endEncoding];
	}
	/* a command buffer Metal failed (a pass it would not encode, a fault on the GPU): the scene says so in red from then
	 * on, instead of showing whatever was left on the screen */
	[cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
		if (done.status == MTLCommandBufferStatusError && scene.failure == nil) {
			scene.failure = [NSString stringWithFormat:@"a command buffer failed: %@", done.error.localizedDescription ?: @"no error given"];
			say([NSString stringWithFormat:@"FAILED: %@: %@", scene.name, scene.failure]);
		}
	}];
	[cb presentDrawable:drawable];
	[cb commit];
}
@end

@interface MetalView : UIView
@end
@implementation MetalView
+ (Class)layerClass
{
	return [CAMetalLayer class];
}
@end

@interface TestController : UIViewController
@end

@implementation TestController {
	Renderer *_renderer;
	CAMetalLayer *_layer;
	UILabel *_label;
	NSUInteger _index, _frames;
	CFTimeInterval _started, _fpsFrom;
	double _fps;
	BOOL _announced;
	/* UIKit over Metal (a scene with .overlay) */
	UIView *_box, *_spinner;
	UILabel *_counter;
	NSUInteger _counted;
	/* pacing, a scene at a time: the display link's callbacks, the wait for a drawable, the drawing */
	CFTimeInterval _lastCallback, _gapSum, _gapMin, _gapMax, _waitSum, _waitMax, _drawSum, _drawMax;
	NSUInteger _callbacks, _lateGaps, _noDrawable;
}

- (void)pacingSay
{
	if (_callbacks > 1) {
		say([NSString stringWithFormat:@"pacing of %lu/%lu: %lu callbacks, %.1f ms apart (%.1f .. %.1f; %lu over 12.5 ms), the drawable waited for "
		     "%.2f ms (%.1f at most), drawing and commit %.2f ms (%.1f at most), %lu times no drawable", (unsigned long)_index + 1,
		     (unsigned long)_renderer.scenes.count, (unsigned long)_callbacks, _gapSum / (double)(_callbacks - 1) * 1000, _gapMin * 1000,
		     _gapMax * 1000, (unsigned long)_lateGaps, _waitSum / (double)_callbacks * 1000, _waitMax * 1000,
		     _drawSum / (double)_callbacks * 1000, _drawMax * 1000, (unsigned long)_noDrawable]);
	}
	_callbacks = _lateGaps = _noDrawable = 0;
	_gapSum = _gapMax = _waitSum = _waitMax = _drawSum = _drawMax = 0;
	_gapMin = 1;
	_lastCallback = 0;
}

- (void)overlayShow:(BOOL)show
{
	if (show && _box == nil) {
		CGFloat w = self.view.bounds.size.width, h = self.view.bounds.size.height;

		_box = [[UIView alloc] initWithFrame:CGRectMake(10, h * 0.45, 150, 70)];
		_box.backgroundColor = [UIColor colorWithWhite:1 alpha:0.85];
		_counter = [[UILabel alloc] initWithFrame:CGRectMake(0, 0, 150, 70)];
		_counter.font = [UIFont monospacedDigitSystemFontOfSize:26 weight:UIFontWeightBold];
		_counter.textAlignment = NSTextAlignmentCenter;
		_counter.textColor = UIColor.blackColor;
		[_box addSubview:_counter];
		[self.view addSubview:_box];
		_spinner = [[UIView alloc] initWithFrame:CGRectMake(w / 2 - 40, h * 0.7, 80, 80)];
		_spinner.backgroundColor = [UIColor colorWithWhite:0.5 alpha:0.9];
		[self.view addSubview:_spinner];
		_counted = 0;
		[UIView animateWithDuration:1.5 delay:0 options:UIViewAnimationOptionRepeat | UIViewAnimationOptionAutoreverse |
		 UIViewAnimationOptionCurveLinear animations:^{
			self->_box.frame = CGRectMake(w - 160, h * 0.45, 150, 70);
		} completion:nil];
	} else if (!show && _box != nil) {
		[_box.layer removeAllAnimations];
		[_box removeFromSuperview];
		[_spinner removeFromSuperview];
		_box = _spinner = nil;
		_counter = nil;
	}
}

- (BOOL)prefersStatusBarHidden
{
	return YES;
}

- (BOOL)prefersHomeIndicatorAutoHidden
{
	return YES;
}

- (void)loadView
{
	self.view = [[MetalView alloc] initWithFrame:UIScreen.mainScreen.bounds];
}

- (void)viewDidLoad
{
	[super viewDidLoad];
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();

	_label = [[UILabel alloc] initWithFrame:CGRectZero];
	_label.numberOfLines = 0;
	_label.font = [UIFont monospacedSystemFontOfSize:13 weight:UIFontWeightMedium];
	_label.textColor = UIColor.whiteColor;
	_label.backgroundColor = [UIColor colorWithWhite:0 alpha:0.6];
	[self.view addSubview:_label];
	[self.view addGestureRecognizer:[[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(next)]];
	if (device == nil) {
		say(@"NO METAL DEVICE (MTLCreateSystemDefaultDevice gave nil)");
		self.view.backgroundColor = UIColor.redColor;
		_label.text = @" NO METAL DEVICE ";
		return;
	}
	_layer = (CAMetalLayer *)self.view.layer;
	_layer.device = device;
	_layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
	_layer.framebufferOnly = YES;
	_layer.opaque = YES;
	_renderer = [[Renderer alloc] initWithDevice:device format:_layer.pixelFormat];
	_started = _fpsFrom = CACurrentMediaTime();
	_gapMin = 1;
	{
		/* boot-arg metaltest_from=N: begin at scene N (a test run that is after the later ones) */
		char args[1024] = "";
		size_t size = sizeof(args) - 1;
		const char *from;

		if (sysctlbyname("kern.bootargs", args, &size, NULL, 0) == 0 && (from = strstr(args, "metaltest_from=")) != NULL) {
			int n = atoi(from + 15);

			if (n >= 1 && (NSUInteger)n <= _renderer.scenes.count) {
				_index = (NSUInteger)n - 1;
			}
		}
	}
	CADisplayLink *link = [CADisplayLink displayLinkWithTarget:self selector:@selector(frame:)];
	if (@available(iOS 15.0, *)) {
		link.preferredFrameRateRange = CAFrameRateRangeMake(60, 120, 120);
	}
	[link addToRunLoop:NSRunLoop.mainRunLoop forMode:NSRunLoopCommonModes];
}

- (void)viewDidLayoutSubviews
{
	[super viewDidLayoutSubviews];
	CGFloat scale = UIScreen.mainScreen.nativeScale;
	CGSize size = self.view.bounds.size;

	_layer.contentsScale = scale;
	_layer.drawableSize = CGSizeMake(size.width * scale, size.height * scale);
	[self layoutLabel];
}

- (void)layoutLabel
{
	CGFloat width = self.view.bounds.size.width;
	CGSize fit = [_label sizeThatFits:CGSizeMake(width - 16, CGFLOAT_MAX)];

	_label.frame = CGRectMake(0, 0, width, fit.height + 12);
}

- (void)next
{
	[self pacingSay];
	_index = (_index + 1) % MAX(_renderer.scenes.count, 1u);
	_started = CACurrentMediaTime();
	_announced = NO;
}

- (void)frame:(CADisplayLink *)link
{
	CFTimeInterval now = CACurrentMediaTime();
	Scene *scene;

	if (_renderer.scenes.count == 0) {
		return;
	}
	if (now - _started >= SCENE_SECONDS) {
		[self next];
	}
	scene = _renderer.scenes[_index];
	_frames++;
	if (_lastCallback != 0) {
		CFTimeInterval gap = now - _lastCallback;

		_gapSum += gap;
		_gapMin = MIN(_gapMin, gap);
		_gapMax = MAX(_gapMax, gap);
		_lateGaps += gap > 0.0125;
	}
	_lastCallback = now;
	_callbacks++;
	[self overlayShow:scene.overlay];
	if (scene.overlay) {
		_counter.text = [NSString stringWithFormat:@"%lu", (unsigned long)++_counted];
		_spinner.transform = CGAffineTransformMakeRotation((CGFloat)(now - _started) * 2);
	}
	if (now - _fpsFrom >= 0.5 || !_announced) {
		if (now - _fpsFrom >= 0.5) {
			_fps = (double)_frames / (now - _fpsFrom);
			_frames = 0;
			_fpsFrom = now;
		}
		_label.text = [NSString stringWithFormat:@" %lu/%lu  %@   %.0f fps\n %@", (unsigned long)_index + 1,
			       (unsigned long)_renderer.scenes.count, scene.name, _fps,
			       scene.failure != nil ? [@"FAILED: " stringByAppendingString:scene.failure] : [@"Expect: " stringByAppendingString:scene.expect]];
		_label.textColor = scene.failure != nil ? [UIColor colorWithRed:1 green:0.5 blue:0.5 alpha:1] : UIColor.whiteColor;
		[self layoutLabel];
		if (!_announced) {
			_announced = YES;
			say([NSString stringWithFormat:@"showing %lu/%lu \"%@\"%@ (the last one ran at %.0f fps)", (unsigned long)_index + 1,
			     (unsigned long)_renderer.scenes.count, scene.name, scene.failure != nil ? @" -- FAILED" : @"", _fps]);
		}
	}
	@autoreleasepool {
		CFTimeInterval t0 = CACurrentMediaTime(), t1;
		id<CAMetalDrawable> drawable = [_layer nextDrawable];

		t1 = CACurrentMediaTime();
		_waitSum += t1 - t0;
		_waitMax = MAX(_waitMax, t1 - t0);
		if (drawable != nil) {
			[_renderer draw:scene into:drawable time:(float)(now - _started)];
			t0 = CACurrentMediaTime();
			_drawSum += t0 - t1;
			_drawMax = MAX(_drawMax, t0 - t1);
		} else {
			_noDrawable++;
		}
	}
}
@end

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property (strong) UIWindow *window;
@end

@implementation AppDelegate
- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)options
{
	self.window = [[UIWindow alloc] initWithFrame:UIScreen.mainScreen.bounds];
	self.window.rootViewController = [TestController new];
	[self.window makeKeyAndVisible];
	application.idleTimerDisabled = YES;
	return YES;
}
@end

int
main(int argc, char **argv)
{
	@autoreleasepool {
		return UIApplicationMain(argc, argv, nil, NSStringFromClass([AppDelegate class]));
	}
}
