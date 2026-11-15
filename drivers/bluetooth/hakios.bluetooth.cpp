/* From upstream's branch mtk (RedwoodBluetooth.cpp, 2026-11-10), renamed. */
/*
 * The Bluetooth controller as iOS looks for it in the registry. MobileBluetooth's BTSessionAttachWithQueue (the way BluetoothManager, Settings'
 * Bluetooth pane, gets to bluetoothd) answers error 9 -- and BluetoothManager.init returns nil, the switch stays grey -- unless
 * IOServiceGetMatchingService finds a service named "marconi-bt" or, failing that, "bluetooth" (an iPhone's device tree has one; this
 * phone's controller is in CONNSYS, and the loader's tree has none: run 735, bt_test's "BluetoothManager attach error 9").
 * Made by hakios.bthost.c once the controller is up.
 */
#include <IOKit/IOLib.h>
#include <IOKit/IOPlatformExpert.h>
#include <IOKit/IOService.h>

class HakiosBluetooth : public IOService
{
	OSDeclareDefaultStructors(HakiosBluetooth);
};

#define super IOService
OSDefineMetaClassAndStructors(HakiosBluetooth, IOService);

extern "C" bool HakiosBluetoothNode(void);
extern "C" void HakiosBluetoothAddress(const uint8_t * wifi);

static HakiosBluetooth * node;
static uint8_t address[6];
static bool have_address;

static void
publish_address(void)
{
	if (node != NULL && have_address) {
		OSData * data = OSData::withBytes(address, 6);

		if (data != NULL) {
			node->setProperty("local-mac-address", data);        /* what bluetoothd's FindLocalMacAddress reads */
			data->release();
		}
	}
}

/* The Bluetooth address of this phone, stable from boot to boot: the Wi-Fi radio's, one up (the way a phone's two addresses go). */
void
HakiosBluetoothAddress(const uint8_t * wifi)
{
	memcpy(address, wifi, 6);
	address[5] = (uint8_t)(address[5] + 1);
	have_address = true;
	publish_address();
}

bool
HakiosBluetoothNode(void)
{
	IOService * platform = IOService::getPlatform();
	HakiosBluetooth * made;

	if (node != NULL) {
		return true;
	}
	if (platform == NULL || (made = OSTypeAlloc(HakiosBluetooth)) == NULL) {
		return false;
	}
	if (!made->init() || !made->attach(platform)) {
		made->release();
		return false;
	}
	made->setName("bluetooth");
	if (!made->start(platform)) {
		made->detach(platform);
		made->release();
		return false;
	}
	made->registerService();
	node = made;
	publish_address();
	IOLog("HakiosBluetooth: the service \"bluetooth\" is in the registry\n");
	return true;
}
