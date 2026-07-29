# Getting real silicon numbers

Every performance figure in `BENCHMARKS.md` comes from the Cadence ISS. This is what it
takes to replace them with measurements from the actual chip, and what each tier proves.

**One physical action gates everything: re-plug the debug USB cable.** The LPC-Link2's
CMSIS-DAP engine is unresponsive (writes accepted, reads return zero bytes, at every
packet size and timeout, with libusb holding interface 0 cleanly). Its VCOM keeps working
because that is a separate CDC interface on the same LPC4322. No software fix exists —
`USBDEVFS_RESET` resets the USB port, not the LPC4322. See `scripts/flash_linux.sh`.

---

## Tier 1 — M33 on silicon. Ready now, nothing to change.

The firmware in `build/` already carries `blockdiag_full`. After the re-plug:

```bash
scripts/enter_isp.py                                             # ROM ISP, no SW7 change
~/.venvs/rt595-flash/bin/blhost -p /dev/ttyACM0 -- flash-image \
        build/lisennet_se_cm33.bin erase
scripts/flash_linux.sh          # or: remote nRESET + capture /dev/ttyACM0
```

**What it proves.** The M33 is not the deploy target, so its cycle count is not the number
we care about — but the run is still worth doing, for one reason that is easy to miss:

`main.cpp:69` and `iss/iss_bench.cpp:37` compute the **same** checksum,
`(int)(sum(mask) * 1000.0f)`, over the same baked features and the same int8 graph. The
M33 and the HiFi4 execute completely different code (CMSIS-NN vs xa_nnlib) but must agree
bit-for-bit on int8 arithmetic. So the checksums are directly comparable, and matching them
is strong evidence the ISS is faithfully executing the real model.

**Expected per-frame checksums** (HiFi4 ISS, `blockdiag_full`, 16 baked frames):

```
frame :  0       1       2       3       4       5       6       7
chksum:  202378  240195  245601  246605  247156  247152  246046  245410
frame :  8       9       10      11      12      13      14      15
chksum:  241804  240230  240753  241726  244238  246238  248171  250472
```

If the board prints these, the ISS is numerically faithful and the only open question is
timing. If it prints something else, one of the two paths is wrong and that matters far
more than any cycle count. The banner also prints the real core clock, which is the
independent check on the 198 MHz assumed by the 3,168,000 cyc/frame budget.

---

## Tier 2 — HiFi4 DSP on silicon. This is the number that matters.

`blockdiag_full`'s headline 1,464,617 cyc/frame is a **DSP** figure. Measuring it on
hardware needs the M33 to boot the DSP, which is a real piece of work — roughly a day.
NXP ships a working template for this exact board at
`sdk/examples/eiq_examples/common/tflm/dsp/evkmimxrt595_fusionf1/`.

**1. Rebuild the bench against the deployable LSP.** Use `min-rt`, not `gdbio`. min-rt has
no stdio — `printf` is silently dropped by libminrt — so the bench cannot report by
printing. Replace the per-frame `printf` with writes to a shared struct:

```c
typedef struct { uint32_t magic, n_frames, cycles[16]; int32_t checksum[16]; } se_result_t;
```

placed at a fixed address both cores agree on, and set `magic` last as the done-flag.

**2. Split the DSP ELF into the three blobs the M33 loader copies** (recipe from that
example's `Makefile.include`):

```
xt-objcopy -O binary <elf> dsp_reset.bin --only-section=.ResetVector.text
xt-objcopy -O binary <elf> dsp_text.bin  --only-section=.WindowVectors.text \
    --only-section=.Level2InterruptVector.text --only-section=.Level3InterruptVector.literal \
    --only-section=.Level3InterruptVector.text --only-section=.DebugExceptionVector.literal \
    --only-section=.DebugExceptionVector.text --only-section=.NMIExceptionVector.literal \
    --only-section=.NMIExceptionVector.text --only-section=.KernelExceptionVector.text \
    --only-section=.UserExceptionVector.literal --only-section=.UserExceptionVector.text \
    --only-section=.DoubleExceptionVector.text --only-section=.text
xt-objcopy -O binary <elf> dsp_data.bin  --only-section=.rtos.rodata --only-section=.rodata \
    --only-section=.clib.data --only-section=.rtos.percpu.data --only-section=.data
```

Destinations come from that example's `dsp_config.h`: reset `0x00180000`, text `0x00180400`,
data `0x00040000`.

**3. M33 side.** Embed the blobs with the SDK's `incbin.S`, call `BOARD_DSP_Init()` from
`dsp_examples/dsp_support.c` (it copies all three and releases the DSP from reset), then
poll the result struct and print it over the VCOM. The M33 linker script must reserve room
for the embedded image — that example ships `MIMXRT595Sxxxx_cm33_flash.ld` already sized
for it. Note NXP's own DSP blob is only ~26 KB total; ours is ~750 KB because of the
weights, so the reserved region needs enlarging.

**4. Sanity check before trusting any of it.** The DSP-side checksums must match the same
16 values above. Cycles come from `XT_RSR_CCOUNT`, which works identically on silicon and
on the ISS, so a matching checksum plus a differing cycle count is exactly the interesting
result: it would quantify how wrong the ISS timing model is while proving the computation
is right.

**What Tier 2 would settle that the ISS cannot.** Real memory behaviour. The ISS numbers
assume zero-wait-state local RAM and were shown to be extremely sensitive to that
assumption — the same ELFs measured 8.6-18.4x worse under a generic off-core layout. On
silicon, `blockdiag_full`'s 738 KB of weights and arena sit in DSP DRAM, and only hardware
can confirm the access cost.

---

## Not required, contrary to earlier notes

- **SW7 does not need changing.** `scripts/enter_isp.py` enters ROM ISP via the boot ROM's
  `runBootloader()` (API tree offset 0 at `0x1302F000`, `mode = ISP`), so the dip switch can
  stay at flash-boot.
- **Do not bind `usbhid`** to the probe's interface 0. pyocd uses libusb (`pyusb` backend);
  the interface must stay unbound.
