# RT595 results — provenance

Every number this target reports falls into one of three classes. The class is not a
detail: a modelled figure quoted as a measurement is the one failure mode that would
discredit the rest of the table.

| Class | Meaning |
| --- | --- |
| **SILICON** | Read off an MIMXRT595-EVK. A raw capture is committed under `results/`. |
| **ISS** | Cycle count from the Xtensa instruction-set simulator. Faithful to ~0.1 % against silicon where both exist, but it is a simulator. |
| **MODELLED** | Computed from datasheet/application-note constants. No instrument was attached. |

## Measured on silicon

| Artifact | What it is |
| --- | --- |
| `silicon_m33_nsnet2_blockdiag_full.txt` | 16 consecutive frames, NSNet2 `blockdiag_full` int8 on the Cortex-M33 via TFLite-Micro + CMSIS-NN, DWT cycle counter, core at 198 MHz. |

From that capture: **mean 5,319,161 cycles/frame** over 16 frames, spread 0.24 %
(min 5,316,272 / max 5,328,934) — i.e. **26.86 ms**, against a 16 ms hop budget of
3,168,000 cycles. The M33 alone therefore misses real time by **1.68×**. Arena occupancy
is in the same capture: **17,096 of 524,288 B**.

Two caveats travel with this file:

- The banner in the original capture reads `LiSenNet`. It is mislabelled. The payload is
  `nsnet2_blockdiagfull_streaming.tflite` — see the provenance comment at the top of
  `app/model_data.h`. The firmware now prints `MODEL_NAME` from the generated
  `model_io_layout.h`, so the banner can no longer drift from the payload; the historical
  capture is kept verbatim rather than edited.
- `mask_checksum` is a determinism check across backends, computed on a synthetic
  sine/exponential feature pattern. It is **not** an audio quality measurement — no STFT,
  no iSTFT, no speech passed through this path.

The HiFi4 figure quoted elsewhere in this directory (**1,466,196 cycles/frame, 7.41 ms,
meeting real time with 2.16× headroom**) was obtained on silicon over pure SWD, but no raw
capture was retained — it survives only as prose in `BENCHMARKS.md` and
`ONBOARD_MEASUREMENT.md`. Treat it as SILICON with a missing artifact until the run is
repeated with stdout captured; that rerun takes about 40 seconds and is the single largest
evidence gap in this target.

### Closing that gap

The exact DSP image that produced the figure is committed under `results/dsp_image/`
(`dsp_reset.bin` / `dsp_text.bin` / `dsp_data.bin`, with `SHA256SUMS`), so the rerun needs
no rebuild and therefore no Cadence toolchain or licence — only the board. From the repo
root, with the EVK on USB:

```bash
pyocd list          # must print the probe serial ORA2CQKQ before anything else
python deploy/rt595/scripts/dsp_hw_run.py \
       --build-dir deploy/rt595/results/dsp_image \
       2>&1 | tee deploy/rt595/results/silicon_hifi4_nsnet2_blockdiag_full.txt
```

Then update the table above: move the HiFi4 row from "no capture retained" to the committed
filename, and check the 16 per-frame checksums against `ONBOARD_MEASUREMENT.md`'s reference
values — if those match, the rerun is measuring the same graph as the original.

Two things to expect. The script halts the M33 and overwrites its RAM at `0x20040000`, so
the board must be **reset**, not resumed, afterwards. And if `pyocd list` reports
`index out of range`, the probe's CMSIS-DAP engine has wedged: that is **only** fixable by
physically unplugging and replugging the DEBUG USB cable, which de-powers the LPC4322 so it
reboots its firmware. `USBDEVFS_RESET`, a sysfs unbind/bind, and `udevadm` re-triggers all
leave the LPC4322 powered and do not clear it — this has been retried and confirmed. See
`scripts/flash_linux.sh` for the full diagnosis.

## Simulated (ISS)

Everything in `BENCHMARKS.md` not listed above. Two rows need footnotes:

- The **dense NSNet2 baseline** was timed on *synthetic* weights
  (`cp_nsnet2_plain_synth`); its PESQ column comes from a different, real checkpoint. The
  timing is representative of the shape, not of that trained model.
- The **ConvFSENet** row was measured with a dead recurrent state. The corrected figure is
  2,812,605 cycles/frame, not the 2,767,999 printed in the older table.

## Modelled

All power and energy figures — 19–23 mW, 0.14–0.17 mJ/frame — are computed from NXP
AN13657 and the i.MX RT500 datasheet. **No ammeter was attached to this board.** Only two
inputs to that model were measured: VDDCORE at 1.00 V (via the PMC LVD sweep) and a 46.3 %
duty cycle. `POWER.md` is a projection.

## Withdrawn claims

Removed rather than carried forward, because no artifact supports them. Note that
"no artifact in this tree" is not the same as "never measured" — one entry below was
withdrawn on that confusion and has been reinstated:

- ~~A LiSenNet nc24 Cortex-M33 figure of ~21.2 M cycles/frame.~~ **Reinstated 2026-09-04.**
  Withdrawing this was an over-correction. The run did happen: LiSenNet streaming on the M33
  at 198 MHz, observed live over the VCOM console on 2026-07-23 ("SE model: 18 IO, 17 states;
  arena 377,200 / 524,288 B", all 16 frames, ~107 ms/frame ≈ 6.7× over the 16 ms budget).
  That is a *different model and run* from the NSNet2 capture above, which reports 2 IO,
  1 state and a 17,096 B arena. Its correct status is the same as the HiFi4 figure:
  **SILICON with no retained capture** — the console output was watched, not saved. Reproduce
  it the same way (`cat /dev/ttyACM0` at 115200 *before* toggling nRESET; the board prints in
  a burst within milliseconds of reset and then idles, so listening late shows nothing).
- A `dsp_offload` block reporting "13.12× speedup / 8080 µs". These were always labelled
  illustrative in `dsp_offload/README_DSP_OFFLOAD.md` — they show the output format of a
  bench that has never been run — but they read as measurements at a glance, so the block is
  now marked unmistakably. Do not quote them.
- A 9.8 KB dense arena figure, contradicted by the 17 K measured in the silicon capture
  above.

## A correction to the headline

`blockdiag_full` does not straightforwardly beat dense on quality. It wins by 0.010 PESQ in
int8 and *loses* by 0.018 in FP32 — both inside metric noise. The defensible claim is that
it **matches dense within metric noise at 4× fewer parameters**, and that this is what makes
it fit and run in real time on the DSP where dense does not fit at all.
