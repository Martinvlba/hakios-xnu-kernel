/*
 * lsreg APP.app ...: an application of this image told to LaunchServices as one of the system's, so that it has an
 * icon and starts -- for applications the image carries itself (share/system-apps: the Metal test application), which
 * are on the boot volume and not in the system volume's /Applications, where nobody can add one. What LaunchServices
 * gets is what it would have read from the bundle there: the path, the identifier, the type "System".
 * lsreg -u IDENTIFIER takes one away again.
 */
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <string.h>

@interface LSApplicationWorkspace : NSObject
+ (instancetype)defaultWorkspace;
- (BOOL)registerApplicationDictionary:(NSDictionary *)dictionary;
- (BOOL)unregisterApplication:(NSURL *)url;
- (BOOL)openApplicationWithBundleID:(NSString *)identifier;
@end
@interface LSApplicationProxy : NSObject
+ (instancetype)applicationProxyForIdentifier:(NSString *)identifier;
@property (readonly) NSURL *bundleURL;
@property (readonly) BOOL isInstalled;
@end

int
main(int argc, char **argv)
{
	@autoreleasepool {
		LSApplicationWorkspace *workspace = [LSApplicationWorkspace defaultWorkspace];
		int failed = 0;

		if (argc == 3 && strcmp(argv[1], "-u") == 0) {
			LSApplicationProxy *proxy = [LSApplicationProxy applicationProxyForIdentifier:@(argv[2])];
			BOOL ok = proxy.bundleURL != nil && [workspace unregisterApplication:proxy.bundleURL];

			printf("lsreg: %s: %s\n", argv[2], ok ? "taken away" : "not registered, or not taken away");
			return !ok;
		}
		if (argc == 3 && strcmp(argv[1], "-o") == 0) {
			BOOL ok = [workspace openApplicationWithBundleID:@(argv[2])];

			printf("lsreg: %s: %s\n", argv[2], ok ? "opened" : "NOT opened");
			return !ok;
		}
		for (int i = 1; i < argc; i++) {
			NSString *path = @(argv[i]);
			NSDictionary *info = [NSDictionary dictionaryWithContentsOfFile:[path stringByAppendingPathComponent:@"Info.plist"]];
			NSString *identifier = info[@"CFBundleIdentifier"];
			LSApplicationProxy *proxy;
			BOOL ok;

			if (identifier == nil) {
				printf("lsreg: %s: no Info.plist with an identifier\n", argv[i]);
				failed++;
				continue;
			}
			proxy = [LSApplicationProxy applicationProxyForIdentifier:identifier];
			if (proxy.isInstalled && [proxy.bundleURL.path isEqualToString:path]) {
				printf("lsreg: %s: registered already (%s)\n", identifier.UTF8String, argv[i]);
				continue;
			}
			ok = [workspace registerApplicationDictionary:@{
				@"ApplicationType": @"System",
				@"CFBundleIdentifier": identifier,
				@"CodeInfoIdentifier": identifier,
				@"CompatibilityState": @0,
				@"IsDeletable": @NO,
				@"IsAdHocSigned": @YES,
				@"Path": path,
				@"_LSBundlePlugins": @{},
			}];
			printf("lsreg: %s at %s: %s\n", identifier.UTF8String, argv[i], ok ? "registered" : "NOT registered");
			failed += !ok;
		}
		return failed != 0;
	}
}
