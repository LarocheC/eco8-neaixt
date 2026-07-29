#!/usr/bin/env python
"""Put the RT595 into ROM ISP mode over SWD — no SW7 change, no physical access.

WHY
---
Flashing this board was believed to require physical access, because:
  * pyocd/SWD flashing is broken on this part (pyocd types the SRAM alias
    0x10000000-0x10020000 as ROM, so the DFP's FlexSPI algo at RAMstart
    0x1001c000 is rejected; forcing past that hardfaults the algo's pc_init), and
  * the supported route, blhost over the ROM ISP, was thought to need SW7 set to
    Serial ISP (1-ON,2-OFF,3-OFF) + a power cycle.

But the RT500 boot ROM exports `runBootloader()` at offset 0 of its API tree at
0x1302F000, and it takes a boot option whose `mode` field selects ISP. So the
target can be told to enter ISP *from software*. We do not even need to inject a
stub: halt the core, park the argument in SRAM, set R0/SP/PC to the ROM function,
and resume. The ROM takes over and enumerates as an ISP device on the same VCOM.

FLOW
----
    enter_isp.py                 -> target is in ROM ISP
    blhost -p /dev/ttyACM0 -- flash-image build/<app>.bin erase
    reset_target.py              -> boots the newly flashed image

`iap_boot_option_t` bit layout (sdk/drivers/iap3/fsl_iap.h:413-436):
    [31:24] tag = 0xEB   [23:20] mode (1 = ISP)   [19:16] bootInterface
    [15:12] instance     [11:8] bootImageIndex    [7:0] reserved
RT500 bootInterface: 0 = USART, 2 = SPI, 3 = USB HID, 4 = FlexSPI.

CAVEAT: this leaves the target in the bootloader, not running your app. Reset (or
reset_target.py / a power cycle) returns it to normal boot.
"""
import argparse
import sys
import time

ROM_API_TREE = 0x1302F000          # FSL_ROM_API_BASE_ADDR; runBootloader is at offset 0
IAP_BOOT_TAG = 0xEB
MODE_ISP = 1
IFACE = {"usart": 0, "spi": 2, "usbhid": 3}

# High SRAM scratch, clear of the low region the boot ROM uses for itself.
ARG_ADDR = 0x20400000
STACK_TOP = 0x20480000


def boot_option(interface: str) -> int:
    return (IAP_BOOT_TAG << 24) | (MODE_ISP << 20) | (IFACE[interface] << 16)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="mimxrt595sffoc")
    ap.add_argument("--interface", choices=sorted(IFACE), default="usart",
                    help="ROM ISP transport (default: %(default)s -> blhost over the VCOM)")
    ap.add_argument("--frequency", type=int, default=1_000_000)
    ap.add_argument("--dry-run", action="store_true",
                    help="connect and report state, but do not divert the core")
    a = ap.parse_args()

    from pyocd.core.helpers import ConnectHelper

    opt = boot_option(a.interface)
    print(f"boot option = 0x{opt:08X}  (tag 0xEB, mode ISP, interface {a.interface})")

    with ConnectHelper.session_with_chosen_probe(
            target_override=a.target, connect_mode="attach",
            options={"frequency": a.frequency, "resume_on_disconnect": False}) as session:
        t = session.target

        t.halt()
        print(f"halted. PC=0x{t.read_core_register('pc'):08X} "
              f"XPSR=0x{t.read_core_register('xpsr'):08X}")

        run_bootloader = t.read32(ROM_API_TREE)          # offset 0 of bootloader_tree_t
        version = t.read32(ROM_API_TREE + 4)
        print(f"ROM API tree @0x{ROM_API_TREE:08X}: runBootloader=0x{run_bootloader:08X} "
              f"version=0x{version:08X}")

        if run_bootloader in (0x00000000, 0xFFFFFFFF) or not (0x13000000 <= run_bootloader < 0x13100000):
            print("ERROR: runBootloader pointer is not in ROM — refusing to jump there.",
                  file=sys.stderr)
            return 2

        if a.dry_run:
            print("dry run: not diverting the core.")
            return 0

        t.write32(ARG_ADDR, opt)
        if t.read32(ARG_ADDR) != opt:
            print(f"ERROR: SRAM scratch at 0x{ARG_ADDR:08X} did not read back.", file=sys.stderr)
            return 3

        t.write_core_register("r0", ARG_ADDR)            # -> iap_boot_option_t *arg
        t.write_core_register("sp", STACK_TOP)
        t.write_core_register("pc", run_bootloader)
        # Thumb state must be set or the jump faults immediately.
        t.write_core_register("xpsr", t.read_core_register("xpsr") | (1 << 24))

        print(f"diverting: R0=0x{ARG_ADDR:08X} SP=0x{STACK_TOP:08X} PC=0x{run_bootloader:08X}")
        t.resume()
        time.sleep(0.5)
        print("resumed — the ROM should now be in ISP mode.")

    print("\nNext:")
    print("  ~/.venvs/rt595-flash/bin/blhost -p /dev/ttyACM0 -- get-property 1")
    print("  ~/.venvs/rt595-flash/bin/blhost -p /dev/ttyACM0 -- flash-image "
          "deploy/rt595/build/lisennet_se_cm33.bin erase")
    return 0


if __name__ == "__main__":
    sys.exit(main())
