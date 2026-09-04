#!/usr/bin/env python
"""Bracket VDDCORE on-die using the RT595 PMC's voltage detectors, over SWD.

The EVK has no remotely readable ammeter, and the PCA9420 PMIC has no current
telemetry — but the PMC can at least confirm the core RAIL VOLTAGE from inside
the die, which pins the V in any P = V x I estimate:

  * LVDCORE: programmable falling-trip detector, 0.720 V + n x 15 mV, n = 0..15
    (max 0.945 V). Sweeping n and watching the flag gives a lower bracket: if the
    flag never latches even at 0.945 V, the rail is above 0.945 V.
  * HVDCORE: fixed-threshold high detector (~1.2 V). If its flag stays clear, the
    rail is below that.

A healthy default rail (PCA9420 power-on SW1 = 1.000 V; nothing in our firmware
ever writes the PMIC) therefore reads as: LVD never trips, HVD never trips ->
0.945 V < VDDCORE < ~1.2 V. An undervolted or sagging rail shows up immediately
as LVD trips at the top of the sweep.

Safety: CTRL.LVDCORERE / CTRL.HVDCORERE make a trip RESET the chip. They are
cleared for the duration of the sweep and restored afterwards.
"""
import argparse
import sys
import time

PMC        = 0x40135000
PMC_STATUS = PMC + 0x04
PMC_FLAGS  = PMC + 0x08          # LVDCOREF bit20, HVDCOREF bit22 (write-1-to-clear)
PMC_CTRL   = PMC + 0x0C          # LVDCOREIE b20, LVDCORERE b21, HVDCOREIE b22, HVDCORERE b23
PMC_RUNCTRL     = PMC + 0x10
PMC_LVDCORECTRL = PMC + 0x18     # [3:0] trip = 0.720 V + n * 15 mV

LVDCOREF = 1 << 20
HVDCOREF = 1 << 22
RE_IE_MASK = (0xF << 20)         # both detectors' interrupt+reset enables

def lvl_mv(n):
    return 720 + 15 * n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="mimxrt595sffoc")
    ap.add_argument("--frequency", type=int, default=1_000_000)
    ap.add_argument("--settle", type=float, default=0.005,
                    help="seconds between trip-level write and flag read")
    a = ap.parse_args()

    from pyocd.core.helpers import ConnectHelper
    with ConnectHelper.session_with_chosen_probe(
            target_override=a.target, connect_mode="attach",
            options={"frequency": a.frequency, "resume_on_disconnect": False}) as session:
        t = session.target
        t.halt()

        ctrl0 = t.read32(PMC_CTRL)
        lvd0 = t.read32(PMC_LVDCORECTRL)
        print(f"PMC STATUS=0x{t.read32(PMC_STATUS):08X} FLAGS=0x{t.read32(PMC_FLAGS):08X} "
              f"CTRL=0x{ctrl0:08X} RUNCTRL=0x{t.read32(PMC_RUNCTRL):08X} "
              f"LVDCORECTRL=0x{lvd0:08X}")
        if ctrl0 & RE_IE_MASK:
            print(f"note: disabling LVD/HVD reset+irq enables (0x{ctrl0 & RE_IE_MASK:08X}) "
                  f"for the sweep")
        t.write32(PMC_CTRL, ctrl0 & ~RE_IE_MASK)

        try:
            # Stale flags from boot transients mean nothing — clear before judging.
            t.write32(PMC_FLAGS, LVDCOREF | HVDCOREF)
            time.sleep(a.settle)

            highest_quiet = None
            tripped_at = None
            for n in range(16):
                t.write32(PMC_LVDCORECTRL, n)
                t.write32(PMC_FLAGS, LVDCOREF)      # re-arm
                time.sleep(a.settle)
                f = t.read32(PMC_FLAGS)
                trip = bool(f & LVDCOREF)
                print(f"  LVD trip {lvl_mv(n)} mV: {'TRIPPED' if trip else 'quiet'}")
                if trip and tripped_at is None:
                    tripped_at = n
                if not trip:
                    highest_quiet = n

            t.write32(PMC_FLAGS, HVDCOREF)
            time.sleep(a.settle)
            hvd = bool(t.read32(PMC_FLAGS) & HVDCOREF)

        finally:
            t.write32(PMC_LVDCORECTRL, lvd0)
            t.write32(PMC_FLAGS, LVDCOREF | HVDCOREF)
            t.write32(PMC_CTRL, ctrl0)

        print()
        if tripped_at is not None:
            print(f"VDDCORE is BELOW {lvl_mv(tripped_at)} mV (LVD tripped) — "
                  f"rail is NOT at the expected 1.0 V default!")
            return 1
        if highest_quiet == 15 and not hvd:
            print("VDDCORE bracket: > 945 mV (LVD quiet at max trip) and < ~1.2 V "
                  "(HVD quiet) — consistent with the PCA9420 1.000 V power-on default.")
            return 0
        print(f"inconclusive: highest quiet LVD level {highest_quiet}, HVD {'SET' if hvd else 'clear'}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
