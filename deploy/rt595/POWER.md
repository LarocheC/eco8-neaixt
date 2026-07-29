# Power measurement on the RT595 EVK

Status 2026-07-29: **voltage and duty cycle are measured; current needs one physical
step** (a meter on the board). Everything that can be prepared remotely has been:
the rail voltage is confirmed on-die, and the board can be put into clean,
reproducible load states over SWD so a meter reading is push-button when hands are
available.

## Why current cannot be measured remotely on this board

- The onboard LPC-Link2 is a plain CMSIS-DAP probe on this EVK — it has no
  energy-measurement circuit (that is an MCU-Link Pro / LPCXpresso feature).
- The PCA9420 PMIC has no output-current telemetry registers; I2C could only read
  back settings we already know.
- There is no current-sense amplifier wired to any target-readable ADC channel.

What the board does offer (from the EVK user guide, UM11287): per-rail
current-measurement headers with shunt resistors (VDDCORE among them) for an
external ammeter — physical access required.

## Measured remotely (done)

**VDDCORE = 1.00 V.** Two independent legs:
- Setting: the PCA9420 powers up at SW1 = 1.000 V and *nothing in this firmware
  ever writes the PMIC* (no PMIC code is linked).
- On-die: `scripts/vddcore_probe.py` sweeps the PMC's programmable low-voltage
  detector to its 0.945 V maximum (never trips) and checks the fixed ~1.2 V
  high detector (clear): **0.945 V < VDDCORE < ~1.2 V**, no sag under load.
  Note: the probe must (and does) temporarily clear `PMC_CTRL.LVDCORERE/HVDCORERE`
  — at boot defaults a detector trip *resets the chip*.

**Duty cycle at 198 MHz: 46.3 %** — 7.41 ms of DSP inference per 16 ms frame
(silicon, `BENCHMARKS.md`). In sustained mode the DSP processes 135 frames/s =
2.16 s of audio per second.

## Reproducible load states (all over SWD, no firmware flashing)

| state | what | how to enter |
|---|---|---|
| S0 idle | M33 halted, DSP unpowered | reset the board, then `pyocd cmd -t mimxrt595sffoc -c halt` (or any of our scripts' halt) — do NOT run dsp_hw_run first |
| S1 DSP sustained inference | M33 halted, HiFi4 running blockdiag_full end to end at 135 f/s, self-checking | `dsp_hw_run.py --build-dir <FOREVER=1 build> --forever` |
| S2 DSP awake, idle spin | M33 halted, DSP parked in a `for(;;)` branch loop | plain `dsp_hw_run.py` (the bench ends in a spin) |
| S3 M33 active spin | M33 firmware after its bench, no debugger | reset, do not attach |

Build the S1 image with `FOREVER=1 bash iss/build_dsp_hw.sh` (see the flag in the
script). The sustained loop re-runs the 16 baked frames with a state reset each
pass and **verifies every pass against the reference checksums** — if it ever
diverged it would park with `ST_LOOP_MISMATCH` rather than keep drawing "inference
power" while computing garbage. `n_frames` in `g_bench_result` counts up
monotonically as the liveness signal; the state survives debugger disconnect.

## The metered protocol (needs hands, ~15 minutes)

Preferred, zero board knowledge: **an inline USB power meter** in the debug USB
line (the board is powered through it). Absolute readings are dominated by the
LPC-Link2 (~100 mA class) — use **deltas between states**, which cancel everything
that does not change:

1. Insert the meter (data passthrough required — the probe must still enumerate;
   note the probe's DAP engine sometimes needs a re-plug anyway, see
   `rt595-probe-wedge-after-powercycle`).
2. Read S0 (idle baseline). 3. Run S1, read. 4. Re-run plain for S2, read.
4. Deltas: **S1−S0 = DSP-domain inference power** (core + local SRAM traffic +
   PFD1/clock tree) — the headline number. S2−S0 = DSP awake-idle floor.
   S1−S2 = the marginal cost of the actual MACs and memory traffic.
   Convert: I_delta × 5 V (USB) × regulator efficiency ≈ rail power; for
   publishable per-rail numbers use the UM11287 VDDCORE shunt header instead and
   multiply by the measured 1.00 V.

Sanity anchors when reading: at 46 % duty the *average* inference power is 0.463 ×
the S1−S0 sustained delta (S1 runs the DSP flat out, no idle gaps). Energy per
16 ms frame = (S1−S0 power) × 7.41 ms.

## Estimated energy per frame (datasheet-anchored, pending the meter)

With VDDCORE = 1.00 V measured and the duty cycle measured, the only unknown is the
active current. The arithmetic, ready for a number:

```
P_active            = 1.00 V × I_DSP@198MHz
E per 16 ms frame   = P_active × 7.41 ms
P_avg (continuous SE, sleep between frames) ≈ 0.463 × P_active + 0.537 × P_idle
```

Datasheet/app-note values for I_DSP are being collected below; the meter protocol
above replaces them with a measurement.

## Not done / possible follow-ups

- **On-die thermal proxy**: the PMC has a temperature sensor (LPADC channel;
  `demo_apps/pmc_temperature_sensor` in the SDK shows the full bring-up incl.
  calibration). An S1-vs-S0 die ΔT divided by the package θJA would give a crude
  fully-remote power number — but expected ΔT is ~1 K at these power levels,
  near the sensor's noise floor, and the LPADC bring-up over SWD is another
  DSP-boot-sized register project. Parked.
- MCU-Link Pro on the external debug header would add real energy measurement
  (and a second, healthier probe).
