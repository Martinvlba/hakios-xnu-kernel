/*
 * sb_cube=1 (the tester, 11-10): a cube tumbling in 3D over the home screen, with the panel's frame rate under it --
 * something that moves all the time, to see the GPU's and the display's artifacts and the frame rate without swiping
 * pages. Part of libhakios.sb.dylib (inject_sb.c starts it in SpringBoard).
 *
 * The cube is six layers in a CATransformLayer (each face a colour, a white edge and its number, back faces culled)
 * under a perspective; it turns and floats by CoreAnimation's own animations, which the render server (backboardd)
 * plays: every refresh is drawn there, on the Mali with the GPU path, and SpringBoard is not woken for it -- a stall of
 * SpringBoard's main thread does not stop the cube, a stall of the render server, the GPU or the display does. The
 * layers go into the home screen's window (a sublayer, not a window of their own: no touch is taken from the icons
 * under it).
 *
 * The numbers under it, every half second: the frames that reached the panel a second (kern.hakios.display_flips,
 * hakios.akita_display.c: what was really shown, not what was drawn), the lowest of the last five seconds, and the
 * frames replaced by a newer one before they were shown (kern.hakios.display_dropped). Without those sysctls (another
 * display driver) the line says so.
 */
#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>
#import <UIKit/UIKit.h>
#include <fcntl.h>
#include <mach/mach_time.h>
#include <stdarg.h>
#include <sys/sysctl.h>
#include <unistd.h>

void hakios_cube_start(void);

static int cube_fd = -1;
static CATextLayer *cube_label;
static uint32_t last_flips, last_dropped, window_fps[10];
static unsigned window_at;
static uint64_t last_at;

static void
cube_say(const char *format, ...)
{
	va_list ap;

	if (cube_fd < 0 && (cube_fd = open("/dev/console", O_WRONLY | O_NOCTTY)) < 0) {
		cube_fd = 2;
	}
	va_start(ap, format);
	vdprintf(cube_fd, format, ap);
	va_end(ap);
}

static bool
sysctl_u32(const char *name, uint32_t *value)
{
	size_t size = sizeof(*value);

	return sysctlbyname(name, value, &size, NULL, 0) == 0 && size == sizeof(*value);
}

static NSArray<UIWindow *> *
all_windows(void)
{
	NSMutableArray<UIWindow *> *windows = [NSMutableArray array];

	for (UIScene *scene in UIApplication.sharedApplication.connectedScenes) {
		if ([scene isKindOfClass:UIWindowScene.class]) {
			[windows addObjectsFromArray:((UIWindowScene *)scene).windows];
		}
	}
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	for (UIWindow *window in UIApplication.sharedApplication.windows) {
		if (![windows containsObject:window]) {
			[windows addObject:window];
		}
	}
#pragma clang diagnostic pop
	return windows;
}

static CALayer *
face(CGFloat size, UIColor *color, int number, CATransform3D place, CGFloat scale)
{
	CALayer *layer = [CALayer layer];
	CATextLayer *text = [CATextLayer layer];

	layer.bounds = CGRectMake(0, 0, size, size);
	layer.backgroundColor = [color colorWithAlphaComponent:0.88].CGColor;
	layer.borderColor = UIColor.whiteColor.CGColor;
	layer.borderWidth = 3;
	layer.doubleSided = NO;                 /* (the far faces culled: a convex body needs no depth sorting) */
	layer.transform = place;
	text.string = [NSString stringWithFormat:@"%d", number];
	text.fontSize = size * 0.55;
	text.foregroundColor = UIColor.whiteColor.CGColor;
	text.alignmentMode = kCAAlignmentCenter;
	text.contentsScale = scale;
	text.frame = CGRectMake(0, size * 0.14, size, size * 0.72);
	[layer addSublayer:text];
	return layer;
}

static void
tick(void)
{
	uint32_t flips, dropped;
	uint64_t now = mach_absolute_time();
	static mach_timebase_info_data_t timebase;

	if (timebase.denom == 0) {
		mach_timebase_info(&timebase);
	}
	if (!sysctl_u32("kern.hakios.display_flips", &flips)) {
		cube_label.string = @"no frame counter\n(kern.hakios.display_flips)";
		return;
	}
	if (!sysctl_u32("kern.hakios.display_dropped", &dropped)) {
		dropped = 0;
	}
	if (last_at != 0) {
		double seconds = (double)(now - last_at) * timebase.numer / timebase.denom / 1e9;
		uint32_t fps = seconds > 0 ? (uint32_t)((flips - last_flips) / seconds + 0.5) : 0;
		uint32_t low = fps;

		window_fps[window_at++ % 10] = fps;
		for (unsigned i = 0; i < 10 && i < window_at; i++) {
			low = window_fps[i] < low ? window_fps[i] : low;
		}
		[CATransaction begin];
		[CATransaction setDisableActions:YES];
		cube_label.string = [NSString stringWithFormat:@"%u fps\nlow %u  ·  %u replaced", fps, low, dropped - last_dropped];
		[CATransaction commit];
	}
	last_flips = flips;
	last_dropped = dropped;
	last_at = now;
}

/*
 * A WINDOW OF OUR OWN over everything of SpringBoard's and over applications (11-10): not the home screen's window --
 * SpringBoard snapshots that one for the widgets page's blur, and the snapshot showed the cube and the strip at rest
 * (the counter read 0, the cube flat) for a frame now and then: a flicker that was ours, not the GPU's. Touches pass
 * through (user interaction off), and it is only as big as what it shows.
 */
static UIWindow *cube_window, *strip_window;

static UIWindow *
overlay_window(UIWindow *home, CGRect frame)
{
	UIWindow *w = [[UIWindow alloc] initWithWindowScene:home.windowScene];

	w.frame = frame;
	w.windowLevel = 2500;                   /* (over SpringBoard's alerts, 2000, its control centre and cover sheet) */
	w.userInteractionEnabled = NO;
	w.backgroundColor = UIColor.clearColor;
	w.opaque = NO;
	w.hidden = NO;
	return w;
}

static bool
build_cube(void)
{
	UIWindow *home = nil;
	NSArray<UIWindow *> *windows = all_windows();
	static bool listed;

	for (UIWindow *window in windows) {
		if ([NSStringFromClass(window.class) containsString:@"HomeScreen"]) {
			home = window;
			break;
		}
	}
	if (!listed || home != nil) {
		NSMutableString *names = [NSMutableString string];

		for (UIWindow *window in windows) {
			[names appendFormat:@" %@(%.0f)", NSStringFromClass(window.class), window.windowLevel];
		}
		cube_say("hakios: cube: SpringBoard's windows:%s\n", names.UTF8String);
		listed = true;
	}
	if (home == nil) {
		return false;
	}

	CGRect bounds = home.layer.bounds;
	CGFloat scale = home.screen.scale > 0 ? home.screen.scale : 3, size = floor(MIN(bounds.size.width, 400) * 0.30);
	CALayer *stage = [CALayer layer];
	CATransformLayer *cube = [CATransformLayer layer];
	CATransform3D perspective = CATransform3DIdentity;
	NSArray<UIColor *> *colors = @[ UIColor.systemRedColor, UIColor.systemGreenColor, UIColor.systemBlueColor,
		                        UIColor.systemOrangeColor, UIColor.systemPurpleColor, UIColor.systemTealColor ];
	CATransform3D places[6] = {
		CATransform3DMakeTranslation(0, 0, size / 2),
		CATransform3DTranslate(CATransform3DMakeRotation(M_PI, 0, 1, 0), 0, 0, size / 2),
		CATransform3DTranslate(CATransform3DMakeRotation(M_PI_2, 0, 1, 0), 0, 0, size / 2),
		CATransform3DTranslate(CATransform3DMakeRotation(-M_PI_2, 0, 1, 0), 0, 0, size / 2),
		CATransform3DTranslate(CATransform3DMakeRotation(-M_PI_2, 1, 0, 0), 0, 0, size / 2),
		CATransform3DTranslate(CATransform3DMakeRotation(M_PI_2, 1, 0, 0), 0, 0, size / 2),
	};

	/* a stage the size of the cube and its numbers, in the middle of the screen a little above the centre */
	stage.bounds = CGRectMake(0, 0, size * 2.2, size * 2.2 + 64);
	stage.position = CGPointMake(CGRectGetMidX(bounds), CGRectGetHeight(bounds) * 0.42);
	stage.zPosition = 1000;
	perspective.m34 = -1.0 / (size * 5);
	stage.sublayerTransform = perspective;

	cube.bounds = CGRectMake(0, 0, size, size);
	cube.position = CGPointMake(size * 1.1, size * 1.1);
	for (int i = 0; i < 6; i++) {
		CALayer *side = face(size, colors[i], i + 1, places[i], scale);

		side.position = CGPointMake(size / 2, size / 2);
		[cube addSublayer:side];
	}
	[stage addSublayer:cube];

	/* turning about two axes at once (additive: the two rotations multiply), and floating up and down */
	CABasicAnimation *turn_y = [CABasicAnimation animationWithKeyPath:@"transform"];
	turn_y.valueFunction = [CAValueFunction functionWithName:kCAValueFunctionRotateY];
	turn_y.fromValue = @0;
	turn_y.toValue = @(2 * M_PI);
	turn_y.duration = 4.0;
	turn_y.repeatCount = HUGE_VALF;
	turn_y.additive = YES;
	CABasicAnimation *turn_x = [CABasicAnimation animationWithKeyPath:@"transform"];
	turn_x.valueFunction = [CAValueFunction functionWithName:kCAValueFunctionRotateX];
	turn_x.fromValue = @0;
	turn_x.toValue = @(2 * M_PI);
	turn_x.duration = 6.5;
	turn_x.repeatCount = HUGE_VALF;
	turn_x.additive = YES;
	[cube addAnimation:turn_y forKey:@"hakios.turn.y"];
	[cube addAnimation:turn_x forKey:@"hakios.turn.x"];
	CABasicAnimation *float_y = [CABasicAnimation animationWithKeyPath:@"position.y"];
	float_y.fromValue = @(-size * 0.12);
	float_y.toValue = @(size * 0.12);
	float_y.duration = 2.2;
	float_y.autoreverses = YES;
	float_y.repeatCount = HUGE_VALF;
	float_y.additive = YES;
	float_y.timingFunction = [CAMediaTimingFunction functionWithName:kCAMediaTimingFunctionEaseInEaseOut];
	[cube addAnimation:float_y forKey:@"hakios.float"];

	/* the numbers, on a dark plate under it */
	cube_label = [CATextLayer layer];
	cube_label.frame = CGRectMake(size * 0.1, size * 2.2, size * 2.0, 58);
	cube_label.backgroundColor = [UIColor colorWithWhite:0 alpha:0.55].CGColor;
	cube_label.cornerRadius = 12;
	cube_label.fontSize = 20;
	cube_label.foregroundColor = UIColor.whiteColor.CGColor;
	cube_label.alignmentMode = kCAAlignmentCenter;
	cube_label.contentsScale = scale;
	cube_label.string = @"…";
	[stage addSublayer:cube_label];

	{
		CGSize sz = stage.bounds.size;
		CGPoint at = stage.position;

		cube_window = overlay_window(home, CGRectMake(at.x - sz.width / 2, at.y - sz.height / 2, sz.width, sz.height));
		stage.position = CGPointMake(sz.width / 2, sz.height / 2);
		[cube_window.layer addSublayer:stage];
		cube_say("hakios: cube: in a window of its own (level %.0f) over %s, %.0f points a side, at %.0f,%.0f of "
		    "%.0fx%.0f\n", cube_window.windowLevel, NSStringFromClass(home.class).UTF8String, size, at.x, at.y,
		    bounds.size.width, bounds.size.height);
	}
	tick();
	[NSTimer scheduledTimerWithTimeInterval:0.5 repeats:YES block:^(__unused NSTimer *timer) {
		tick();
	}];
	return true;
}

/* from inject_sb.c (sb_cube=1), on SpringBoard's main queue: tried every two seconds until the home screen is there */
void
hakios_cube_start(void)
{
	static unsigned tries;

	if (build_cube()) {
		return;
	}
	if (++tries < 60) {
		dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
			hakios_cube_start();
		});
	} else {
		cube_say("hakios: cube: no home screen window found in two minutes\n");
	}
}

/*
 * sb_debug=1 (the tester, 11-10): marks on the home screen for slow-motion videos, to tell every frame on the panel apart
 * and put it beside the logs. Under the status bar:
 *
 *   [white][black] b11 b10 ... b1 b0     t 123.45
 *
 * The squares are a frame counter in Gray code that the render server draws (CoreAnimation's discrete keyframe animations on
 * their opacity, bit k a period of 2^(k+1) refreshes, all begun at one epoch): every frame drawn carries its number --
 * a frame on the panel again shows an old number, a frame skipped makes it jump by 2, a frame torn between two shows
 * squares of both; the counter wraps every 4096 refreshes (34.1 s at 120 Hz). The two at the left are always white and
 * always black (the thresholds for reading the others from a video: devices/akita/tools/video-counter.py). The text is
 * the uptime in seconds, the kernel log's clock, from SpringBoard twenty times a second. The epoch is said on the
 * console ("hakios: debug: counter epoch ..."): a frame numbered N was drawn for the refresh at epoch + N / Hz.
 */
static CATextLayer *debug_time;

static bool
build_debug_strip(void)
{
	UIWindow *home = nil;

	for (UIWindow *window in all_windows()) {
		if ([NSStringFromClass(window.class) containsString:@"HomeScreen"]) {
			home = window;
			break;
		}
	}
	if (home == nil) {
		return false;
	}

	CGRect bounds = home.layer.bounds;
	CGFloat scale = home.screen.scale > 0 ? home.screen.scale : 3, side = 16, gap = 3, top = 37;
	char args[1024] = "";
	size_t size = sizeof(args) - 1;
	double hz = sysctlbyname("kern.bootargs", args, &size, NULL, 0) == 0 && strstr(args, "display_hz=60") ? 60 : 120;
	CALayer *strip = [CALayer layer];
	unsigned count = 14;
	CGFloat width = count * side + (count - 1) * gap;

	CGFloat left = (bounds.size.width - width - 88) / 2;             /* (the clock beside it: 82 + 6) */

	strip_window = overlay_window(home, CGRectMake(left, top - 2, width + 88, side + 4));
	strip.frame = CGRectMake(0, 2, width, side);
	strip.zPosition = 1001;
	CFTimeInterval epoch = CACurrentMediaTime() + 0.2;      /* (a common begin for every bit, a little ahead) */
	for (unsigned i = 0; i < count; i++) {
		CALayer *cell = [CALayer layer], *on = [CALayer layer];

		cell.frame = CGRectMake(i * (side + gap), 0, side, side);
		cell.backgroundColor = UIColor.blackColor.CGColor;
		cell.borderColor = UIColor.redColor.CGColor;
		cell.borderWidth = 1;
		on.frame = CGRectInset(cell.bounds, 2, 2);
		on.backgroundColor = UIColor.whiteColor.CGColor;
		if (i == 0) {
			on.opacity = 1;                 /* the white reference */
		} else if (i == 1) {
			on.opacity = 0;                 /* the black reference */
		} else {
			unsigned bit = count - 1 - i;   /* most significant first: b11 at the left, b0 at the right */
			CAKeyframeAnimation *blink = [CAKeyframeAnimation animationWithKeyPath:@"opacity"];

			on.opacity = 0;
			/* GRAY CODE (one square changes a refresh: a camera's frame that spans two refreshes reads as one of them,
			 * not as a third number): bit k high for n mod 2^(k+2) in [2^k, 3 * 2^k); the top bit as binary (cyclic) */
			if (bit == 11) {
				blink.values = @[ @0, @1 ];
				blink.keyTimes = @[ @0, @0.5, @1 ];     /* (discrete: one key time more than values) */
				blink.duration = 4096.0 / hz;
			} else {
				blink.values = @[ @0, @1, @0 ];
				blink.keyTimes = @[ @0, @0.25, @0.75, @1 ];
				blink.duration = (double)(4u << bit) / hz;
			}
			blink.calculationMode = kCAAnimationDiscrete;
			blink.repeatCount = HUGE_VALF;
			blink.beginTime = epoch;
			blink.removedOnCompletion = NO;
			[on addAnimation:blink forKey:@"hakios.counter"];
		}
		[cell addSublayer:on];
		[strip addSublayer:cell];
	}
	[strip_window.layer addSublayer:strip];

	debug_time = [CATextLayer layer];
	debug_time.frame = CGRectMake(CGRectGetMaxX(strip.frame) + 6, 0, 82, side + 4);
	debug_time.zPosition = 1001;
	debug_time.backgroundColor = UIColor.blackColor.CGColor;
	debug_time.foregroundColor = UIColor.whiteColor.CGColor;
	debug_time.font = (__bridge CFTypeRef)@"Menlo-Bold";
	debug_time.fontSize = 14;
	debug_time.alignmentMode = kCAAlignmentCenter;
	debug_time.contentsScale = scale;
	[strip_window.layer addSublayer:debug_time];

	/* the epoch on the kernel log's clock: CACurrentMediaTime is mach_absolute_time in seconds, as the log's stamps */
	cube_say("hakios: debug: counter epoch %.6f s (uptime), %.0f Hz, 12 bits of Gray code (b11..b0 left to right after "
	    "the white and black references), wraps every %.1f s; the strip at %.0f,%.0f, %.0f points a square\n", epoch, hz, 4096 / hz,
	    left, top, side);
	[NSTimer scheduledTimerWithTimeInterval:0.05 repeats:YES block:^(__unused NSTimer *timer) {
		[CATransaction begin];
		[CATransaction setDisableActions:YES];
		debug_time.string = [NSString stringWithFormat:@"t %.2f", CACurrentMediaTime()];
		[CATransaction commit];
	}];
	return true;
}

void hakios_debug_start(void);

/* from inject_sb.c (sb_debug=1), on SpringBoard's main queue: tried every two seconds until the home screen is there */
void
hakios_debug_start(void)
{
	static unsigned tries;

	if (build_debug_strip()) {
		return;
	}
	if (++tries < 60) {
		dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
			hakios_debug_start();
		});
	}
}
