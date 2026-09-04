# TFLite-Micro for the RT500 HiFi4 (Fusion-F1) core

Rebuilds `libtflm_rt500.a` — the TFLite-Micro static library, *including the bundled optimized
xa_nnlib int8 kernels*, compiled for the RT595's HiFi4 DSP. Everything under
`deploy/rt595/iss/` links against it.

## Why not the SDK's prebuilt library

The MCUXpresso SDK ships `libtflm.a` built for the **rt685s** core. Linking it for
`nxp_rt500_RI23_11_newlib` fails with *"could not decode instruction; possible configuration
mismatch"* — the RT500 Fusion-F1 is a **narrower** HiFi4 than the rt600 and lacks several
intrinsics (`AE_ADDANDSUB32S`, `AE_MOVBA4`, `AE_MUL32JS`, `AE_ADDANDSUBRNG32`). Hence a
from-source build.

## Build

```bash
export XTENSA_LICENSE_FILE=/path/to/RT500SDK.lic     # not in git; node-locked, see below
export XT=$HOME/toolchains/tools/RI-2023.11-linux/XtensaTools
./build_rt500.sh          # clones cad-audio/tflite-micro, applies tflm.patch, builds
```

Produces `libtflm_rt500.a` (~4.75 MB) beside the script. `resume_build.sh` re-runs `make`
without re-cloning.

## What `tflm.patch` does

Comments out the ndsplib **FFT** source block in `ext_libs/xtensa.inc`. Those sources use the
intrinsics the rt500 core does not have, so the build dies there. None of our models contain
signal/FFT ops — the STFT is computed on the M33 — so excluding them is free.

## Pinned inputs

| | |
|---|---|
| upstream | `github.com/cad-audio/tflite-micro` |
| commit | `54331e9fe42af1dcc8221d693ae85890fb399944` |
| core | `nxp_rt500_RI23_11_newlib` (RI-2023.11, XtensaTools-14.11) |
| make target | `TARGET=xtensa TARGET_ARCH=hifi4 XTENSA_USE_LIBC=true OPTIMIZED_KERNEL_DIR=xtensa` |

## Gotchas

- **numpy must be on `PATH`'s python** — TFLM's `generate_cc_arrays.py` needs it. Put a venv that
  has it first: `PATH="$HOME/.venvs/rt595-export/bin:$XT/bin:$PATH"`.
- **This fork exposes 97 builtin ops.** The NXP eIQ SDK tree exposes a different (larger) set, so
  "passes the op check" is a question about *which tree gets linked*. Always check against the
  tree you are actually linking.
- **`RT500SDK.lic` is deliberately not committed** — it is node-locked to one MAC address and
  expires 2027-07-28. Obtain it from the Cadence/NXP account and point `XTENSA_LICENSE_FILE` at it.
