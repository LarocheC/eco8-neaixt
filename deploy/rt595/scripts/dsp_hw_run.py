#!/usr/bin/env python
"""Boot the HiFi4 DSP on the RT595 entirely over SWD and read the bench results.

No M33 firmware change, no reflash. Everything BOARD_DSP_Init() does on the M33
(sdk/examples/_boards/evkmimxrt595/dsp_examples/dsp_support.c) is plain MMIO plus
memcpy, and the debugger can do both. The M33 is halted first and left halted, so
the whole sequence runs with the applications processor stopped:

    halt M33 -> stall DSP -> SYSPLL0 PFD1 = /24 (396 MHz) -> DSPCPUCLKSELB = dsp_pll
    -> DSPCPUCLKDIV = /2 (198 MHz) -> DSP_VECT_REMAP = 0x600 -> power up DSP domain
    (PDRUNCFG1 bit25 + all SRAM partitions, PMC apply) -> PSCCTL0 DSP clock enable
    -> reset pulse -> write dsp_{reset,text,data}.bin -> readback verify -> unstall
    -> poll g_bench_result.

Address map facts this depends on (verified against the ELF and the LSP):
  * The ELF links .ResetVector.text at VMA 0 (primary static vector base); the
    DSP_VECT_REMAP value 0x600 makes the DSP fetch vectors from 0x600*0x400 =
    0x180000, which is where we place dsp_reset.bin. Remap MUST be set before
    unstall or the DSP fetches from unmapped address 0.
  * DSP I-side addresses equal M33/physical addresses (text at 0x180400 both).
  * DSP D-side addresses are physical + 0x800000: dram0_0_seg at DSP 0x840000 is
    physical 0x040000. This -0x800000 alias is what the run empirically confirms.
  * g_bench_result sits at the start of .bss (see `xt-nm dsp_bench.elf`), DSP
    0x00904e10 -> physical 0x104e10 -> M33 data bus 0x20104e10.

The PMIC is deliberately not touched: at 198/198 MHz the required VDDCORE tier is
1.0 V (fsl_power.c powerFreqLevel[]), which is the PCA9420's power-on default.

CAVEAT: writing the data blob at 0x20040000 overwrites RAM of the (halted) M33
application. The M33 must be reset before it is trusted again — do not resume it.

Layout of g_bench_result (iss/dsp_bench_mem.cpp): 8 x u32 header
(magic, state, n_frames, arena_used, n_io, n_states, feat_len, ccount_overhead)
then u32 cycles[16], then i32 checksum[16].
"""
import argparse
import struct
import sys
import time

# ---- register map (MIMXRT595S, from sdk/devices/RT/RT500) -------------------
SYSCTL0 = 0x40002000
DSP_VECT_REMAP = SYSCTL0 + 0x004     # [11:0] remap (units of 1 KB), [12] statvec sel
DSPSTALL       = SYSCTL0 + 0x00C     # bit0: 1 = stalled
PDRUNCFG1      = SYSCTL0 + 0x614     # bit25 = DSP power down
PDRUNCFG2      = SYSCTL0 + 0x618     # SRAM partition array power
PDRUNCFG3      = SYSCTL0 + 0x61C
PDRUNCFG1_CLR  = SYSCTL0 + 0x634
PDRUNCFG2_CLR  = SYSCTL0 + 0x638
PDRUNCFG3_CLR  = SYSCTL0 + 0x63C

PMC        = 0x40135000
PMC_STATUS = PMC + 0x4               # bit0 ACTIVEFSM
PMC_CTRL   = PMC + 0xC               # bit0 APPLYCFG, bit1 CLKDIVEN

CLKCTL0     = 0x40001000
PSCCTL0     = CLKCTL0 + 0x010        # bit1 = DSP clock
PSCCTL0_SET = CLKCTL0 + 0x040
SYSPLL0PFD  = CLKCTL0 + 0x218        # PFD1: div [13:8], CLKRDY bit14, CLKGATE bit15

CLKCTL1       = 0x40021000
DSPCPUCLKDIV  = CLKCTL1 + 0x400      # [7:0] div-1, bit29 RESET, bit31 REQFLAG
DSPCPUCLKSELB = CLKCTL1 + 0x434      # 2 = dsp_pll (SYSPLL0 PFD1)

RSTCTL0      = 0x40000000
PRSTCTL0     = RSTCTL0 + 0x10        # bit1 = DSP reset
PRSTCTL0_SET = RSTCTL0 + 0x40
PRSTCTL0_CLR = RSTCTL0 + 0x70

# ---- image + result --------------------------------------------------------
BLOBS = [  # (file, M33 data-bus address)
    ("dsp_reset.bin", 0x20180000),
    ("dsp_text.bin",  0x20180400),
    ("dsp_data.bin",  0x20040000),
]
RESULT_ADDR_DEFAULT = 0x20104E10     # xt-nm g_bench_result (DSP 0x00904e10) - 0x800000 + 0x20000000
N_FRAMES = 16
RESULT_LEN = 4 * (8 + N_FRAMES + N_FRAMES)

SE_BENCH_MAGIC = 0x48465F31          # "HF_1"
STATE_NAMES = {0: "not started", 1: "ST_ENTER", 2: "ST_INIT_OK", 3: "ST_WARMED",
               4: "ST_RUNNING", 5: "ST_DONE", 0xE1: "ST_INIT_FAIL", 0xE2: "ST_FRAME_FAIL"}

DSP_HZ = 198_000_000                 # SYSPLL0 528 * 18/24 = 396, /2
ISS_CYCLES = 1_464_615               # blockdiag_full, ISS, per frame (representative)
ISS_CHECKSUMS = [202378, 240195, 245601, 246605, 247156, 247152, 246046, 245410,
                 241804, 240230, 240753, 241726, 244238, 246238, 248171, 250472]


def poll32(t, addr, mask, want_set, what, timeout=2.0):
    dl = time.time() + timeout
    while True:
        v = t.read32(addr)
        if bool(v & mask) == want_set:
            return v
        if time.time() > dl:
            raise TimeoutError(f"{what}: 0x{addr:08X} stuck at 0x{v:08X} "
                               f"(want bit(s) 0x{mask:08X} {'set' if want_set else 'clear'})")
        time.sleep(0.001)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default="deploy/rt595/build",
                    help="dir holding dsp_{reset,text,data}.bin (default: %(default)s)")
    ap.add_argument("--result-addr", type=lambda s: int(s, 0), default=RESULT_ADDR_DEFAULT,
                    help="M33 address of g_bench_result; re-derive with xt-nm after any "
                         "rebuild (default: 0x%(default)X)" % {"default": RESULT_ADDR_DEFAULT})
    ap.add_argument("--target", default="mimxrt595sffoc")
    ap.add_argument("--frequency", type=int, default=1_000_000)
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="seconds to wait for the done-magic (default: %(default)s)")
    ap.add_argument("--skip-load", action="store_true",
                    help="do not touch the target state; just poll/read the result struct")
    a = ap.parse_args()

    import os
    blobs = []
    if not a.skip_load:
        for name, addr in BLOBS:
            path = os.path.join(a.build_dir, name)
            with open(path, "rb") as f:
                data = f.read()
            blobs.append((name, addr, data))
            print(f"{name}: {len(data)} bytes -> 0x{addr:08X}")

    from pyocd.core.helpers import ConnectHelper
    with ConnectHelper.session_with_chosen_probe(
            target_override=a.target, connect_mode="attach",
            options={"frequency": a.frequency, "resume_on_disconnect": False}) as session:
        t = session.target

        t.halt()
        print(f"M33 halted at PC=0x{t.read_core_register('pc'):08X} (left halted; "
              f"reset the board before trusting the M33 app again)")

        if not a.skip_load:
            print(f"before: DSPSTALL={t.read32(DSPSTALL) & 1} "
                  f"PDRUNCFG1=0x{t.read32(PDRUNCFG1):08X} "
                  f"PRSTCTL0=0x{t.read32(PRSTCTL0):08X} "
                  f"PSCCTL0=0x{t.read32(PSCCTL0):08X} "
                  f"SYSPLL0PFD=0x{t.read32(SYSPLL0PFD):08X}")

            # 0. Hold the DSP stalled while we set everything up.
            t.write32(DSPSTALL, 1)

            # 1. SYSPLL0 PFD1 -> /24 = 396 MHz (mirrors CLOCK_InitSysPfd(kCLOCK_Pfd1, 24)).
            pfd = t.read32(SYSPLL0PFD)
            base = pfd & ~0xBF00                    # clear PFD1 div + CLKGATE, keep the rest
            t.write32(SYSPLL0PFD, base | 0x8000)    # gate PFD1
            t.write32(SYSPLL0PFD, base | (24 << 8)) # div 24, ungated
            poll32(t, SYSPLL0PFD, 1 << 14, True, "PFD1 CLKRDY")
            print(f"PFD1 ready: SYSPLL0PFD=0x{t.read32(SYSPLL0PFD):08X} (396 MHz)")

            # 2. DSP main clock = dsp_pll / 2 = 198 MHz
            #    (CLOCK_AttachClk(kDSP_PLL_to_DSP_MAIN_CLK) + CLOCK_SetClkDiv(.., 2)).
            t.write32(DSPCPUCLKSELB, 2)
            div = t.read32(DSPCPUCLKDIV)
            t.write32(DSPCPUCLKDIV, div | (1 << 29))  # reset divider counter
            t.write32(DSPCPUCLKDIV, 2 - 1)
            poll32(t, DSPCPUCLKDIV, 1 << 31, False, "DSPCPUCLKDIV REQFLAG")
            print(f"DSP clock: SELB={t.read32(DSPCPUCLKSELB)} "
                  f"DIV=0x{t.read32(DSPCPUCLKDIV):08X} -> 198 MHz")

            # 3. Vectors: primary static base (0) remapped to 0x600 KB = 0x180000.
            t.write32(DSP_VECT_REMAP, 0x600)

            # 4. Power up the DSP domain and make sure no SRAM partition is gated
            #    (clearing already-clear bits is a no-op), then the PMC apply dance.
            t.write32(PDRUNCFG1_CLR, 1 << 25)
            t.write32(PDRUNCFG2_CLR, 0xFFFFFFFF)
            t.write32(PDRUNCFG3_CLR, 0xFFFFFFFF)
            t.write32(PMC_CTRL, t.read32(PMC_CTRL) & ~0x2)   # CLKDIVEN off
            poll32(t, PMC_STATUS, 1, False, "PMC ACTIVEFSM (pre)")
            t.write32(PMC_CTRL, t.read32(PMC_CTRL) | 0x1)    # APPLYCFG
            poll32(t, PMC_STATUS, 1, False, "PMC ACTIVEFSM (apply)", timeout=5.0)
            t.write32(PMC_CTRL, t.read32(PMC_CTRL) | 0x2)    # CLKDIVEN back on
            print(f"power applied: PDRUNCFG1=0x{t.read32(PDRUNCFG1):08X}")

            # 5. DSP clock gate + reset pulse (CLOCK_EnableClock, RESET_PeripheralReset).
            t.write32(PSCCTL0_SET, 1 << 1)
            t.write32(PRSTCTL0_SET, 1 << 1)
            poll32(t, PRSTCTL0, 1 << 1, True, "DSP reset assert")
            t.write32(PRSTCTL0_CLR, 1 << 1)
            poll32(t, PRSTCTL0, 1 << 1, False, "DSP reset release")
            print("DSP clocked and out of reset (still stalled)")

            # 6. Place the image, with full readback verification.
            for name, addr, data in blobs:
                t0 = time.time()
                t.write_memory_block8(addr, data)
                rb = bytes(t.read_memory_block8(addr, len(data)))
                dt = time.time() - t0
                if rb != data:
                    bad = next(i for i in range(len(data)) if rb[i] != data[i])
                    print(f"ERROR: {name} readback mismatch at +0x{bad:X} "
                          f"(0x{addr + bad:08X}): wrote {data[bad]:02X} read {rb[bad]:02X}",
                          file=sys.stderr)
                    return 2
                print(f"{name}: wrote+verified {len(data)} B in {dt:.1f}s "
                      f"({len(data) / dt / 1024:.0f} KiB/s)")

            # 7. Invalidate the result struct so nothing stale can satisfy the poll.
            t.write_memory_block8(a.result_addr, b"\x00" * RESULT_LEN)

            # 8. Go.
            t.write32(DSPSTALL, 0)
            print("DSP released — polling for results")

        # 9. Poll magic; report state transitions while waiting.
        deadline = time.time() + a.timeout
        last_state = None
        while True:
            magic = t.read32(a.result_addr)
            state = t.read32(a.result_addr + 4)
            if state != last_state:
                print(f"  state = 0x{state:02X} ({STATE_NAMES.get(state, '?')})")
                last_state = state
            if magic == SE_BENCH_MAGIC:
                break
            if time.time() > deadline:
                print(f"TIMEOUT: magic=0x{magic:08X} state=0x{state:02X} "
                      f"({STATE_NAMES.get(state, '?')})", file=sys.stderr)
                return 4
            time.sleep(0.25)

        raw = bytes(t.read_memory_block8(a.result_addr, RESULT_LEN))
        hdr = struct.unpack_from("<8I", raw, 0)
        magic, state, n_frames, arena, n_io, n_states, feat_len, cc_ovh = hdr
        cycles = struct.unpack_from(f"<{N_FRAMES}I", raw, 32)
        checks = struct.unpack_from(f"<{N_FRAMES}i", raw, 32 + 4 * N_FRAMES)

        print(f"\n=== HiFi4 SILICON RESULT (state {STATE_NAMES.get(state, state)}) ===")
        print(f"model: {n_io} IO, {n_states} states; arena used {arena} B; "
              f"feat_len {feat_len}; ccount overhead {cc_ovh} cyc")
        if state != 5:
            print(f"DSP reported failure state 0x{state:02X} — numbers below are not valid")
            return 3

        n = min(n_frames, N_FRAMES)
        ok = True
        print(f"{'frame':>5} {'cycles':>10} {'ms@198':>7} {'checksum':>9}  vs ISS")
        for i in range(n):
            match = checks[i] == ISS_CHECKSUMS[i]
            ok &= match
            print(f"{i:5d} {cycles[i]:10d} {cycles[i] / DSP_HZ * 1e3:7.2f} {checks[i]:9d}"
                  f"  {'==' if match else f'!= {ISS_CHECKSUMS[i]}'}")
        mean = sum(cycles[:n]) / n
        print(f"\nmean {mean:,.0f} cyc/frame = {mean / DSP_HZ * 1e3:.2f} ms @198 MHz "
              f"({mean / 3_168_000:.2f}x of the 3,168,000 real-time budget)")
        print(f"ISS said {ISS_CYCLES:,} -> silicon/ISS = {mean / ISS_CYCLES:.3f}x")
        print("checksums: ALL 16 MATCH the ISS" if ok else
              "checksums: MISMATCH — computation differs from the ISS, cycles are moot")
        return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
