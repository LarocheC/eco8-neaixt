# Track 1 — stateless-windowed ConvFSENet: host work done, deploy-box handoff

Implements **Track 1** of [EFFICIENCY_REWORK_PLAN.md](EFFICIENCY_REWORK_PLAN.md):
a *stateless windowed* ConvFSENet that removes the per-block FIFO state plumbing
(the `Slice`/`Concat`/`Gather` class that is ConvFSENet's M55 floor) by feeding a
fixed context window and running the BN-folded offline-causal model as **valid
(padding-0) convs**. This document is the bridge between the **host machine**
(training box — implementation, PTQ, PESQ done here) and the **deploy box** (the
one with `stedgeai` + STM32N6570-DK), which runs Gate-0 generate + Phase-4
on-board measurement.

> The host machine has **no `stedgeai` and no board** (and no ST cloud creds), so
> Gate-0 (`generate`) and Phase-4 (on-board) **must run on the deploy box**. All
> ONNX artifacts and exact commands are below.

## What the graph looks like (and why it should beat 4.40 ms)

The windowed graph is **stateless**: input `noisy_mag_window [1, n_freq, L+T]`,
output `mask [1, n_freq, T]`. `L = sum_blocks (K-1)*D = 3·(1+2+4)·2 = 42`
(receptive field `RF = L+1 = 43`). Op histogram (int8, T=1):

```
Conv 29 · Add 10 · Slice 9 · Pow 1 · Sigmoid 1 · QuantizeLinear 49 · DequantizeLinear 98
Gather 0 · Pad 0 · BatchNormalization 0 · state nodes 0   ← the FIFO class is GONE
```

vs the deployed streaming graph, whose per-block FIFO `Gather`/`Slice`/`Concat`
forced Hybrid (M55) epochs (lever-2-closed, ~4.40 ms floor). The 9 `Slice` here
are **static residual crops** (`x[..., trim:]`, pointer offsets), not dynamic
state I/O. The dilated dconvs are **native valid convs over the 43-column
window** — the array-fill (`h:43` not `h:1`) the rework banks on. The input path
is `noisy_mag_window → Add(eps) → Pow → Conv` (FP32 compression prologue, then
quantized) — **no Slice/Gather before quantization**, unlike the streaming model,
so the `--input-data-type` quirk in `scripts/generate.sh` does not apply here.

## Host results (measured on this box)

- **FP32 windowed parity:** bit-exact (< 1e-6) to the offline causal / deployed
  streaming model on frames with full real left context (`tests/test_convfsenet_windowed_parity.py`, 16/16).
- **Cold-start (measured, and fixed host-side — no retrain):** the model was
  trained with a zero-ACTIVATION history before t=0 (offline causal left-pad),
  but a zero-MAGNITUDE ring buffer feeds the frontend *bias* (frontend(0) ≠ 0)
  for the first `< L` frames → an out-of-distribution start-of-clip transient
  (decays to FP noise by ~frame 30; bit-exact after). On short VBD clips this
  cost **~0.045 PESQ** with `coldstart=zero`. **Fix:** initialize the ring buffer
  by **replicating the first frame** (`coldstart=replicate`, now the default) —
  CAUSAL (no look-ahead, so on-device deployable) and in-distribution. Measured
  on 200 VBD utts (FP32-256): zero **2.939** → replicate **2.983** → reflect
  **2.984** (reflect needs look-ahead → not deployable). The on-device C glue
  must seed the ring buffer with the first received magnitude column, not zeros.
- **Host int8 PESQ (full 824-utt VBD test, `coldstart=replicate`, no retrain):**

| variant | n_freq | FP32 PESQ | int8 PESQ | gate ≥2.85 |
|---|---:|---:|---:|:--:|
| streaming reference (deployed) | 257 | 2.931 | 2.911 | — |
| windowed-257 | 257 | 2.923 | 2.904 | ✓ |
| **windowed-256 (deploy target)** | 256 | **2.933** | **2.913** | ✓ |

The windowed-256 int8 **2.913 matches/slightly beats the streaming 2.911** — same
quality, but the graph has no FIFO/state/Pad class. (With `coldstart=zero` the
windowed-256 int8 is only 2.843 — the cold-start fix is load-bearing; the on-device
C glue must seed the ring buffer with the first frame, not zeros.)

## Artifacts to copy to the deploy box

In `cp_convfsenet_win/` (built from the v5 / 2.931-FP32 / 2.911-int8 weights):

| file | purpose |
|---|---|
| `g_best_win256.onnx` | **deploy target** — int8 QDQ, 256-bin windowed (T=1) |
| `g_best_win256_fp32.onnx` | FP32 ref for `validate` on-target cosine |
| `g_best_win257.onnx` | int8, 257-bin (reference / bit-exact-arch check) |
| `g_best_win257_fp32.onnx` | FP32 ref |
| `config.json` | the 192/384 mag_compressed config (sibling, required by tooling) |

Plus `deploy/stm32n6/gate0_artifacts/gate0d_conv2d_probe.onnx` (Track-3 leg-C probe).

Regenerate any of them on the host with:
```bash
.venv/bin/python -m convfsenet.export_onnx --windowed --checkpoint_file cp_convfsenet_win/g_best --emit_T 1 --drop_nyquist   # 256
.venv/bin/python -m convfsenet.quant_windowed --checkpoint_dir cp_convfsenet_win --num_utterances 200 --frames_per_utterance 80 --emit_T 1 --drop_nyquist
```

## Deploy-box procedure (mirrors ONBOARD_MEASUREMENT.md)

`N6DIR=~/stedgeai/install/4.0/scripts/N6_scripts` ; `STEDGEAI=~/stedgeai/install/4.0/Utilities/linux/stedgeai`

### Gate-0A — generate-only de-risk (no board)
The real 256-bin int8 ONNX *is* the Gate-0A artifact (real weights, right
architecture — no dummy stub needed). Generate and read the report:
```bash
cd "$N6DIR"
$STEDGEAI generate -m ~/eco8-neaixt/cp_convfsenet_win/g_best_win256.onnx --target stm32n6 \
  --st-neural-art n6-noextmem@user_neuralart.json \
  --fix-parametric-shapes "{'B':1}" -n network -o /tmp/n6val_win
grep -E "epoch|Hybrid|SW|h:|util" /tmp/n6val_win/network_generate_report.txt
```
**PASS:** 0 pure-SW epochs (or far fewer than streaming), **no `Gather`/`state_*`
nodes**, weights ~1.4 MB on-chip, and the convs show **`h:43`** (array fill), not
`h:1`. This single run replaces the un-reproduced epoch/util claims with a number.

### Gate-0C — confirm the FIFO class is gone
Diff the windowed epoch list against the deployed streaming report:
```bash
diff <(grep -iE "epoch|gather|slice|concat|state" /tmp/n6val_win/network_generate_report.txt) \
     <(grep -iE "epoch|gather|slice|concat|state" /tmp/n6val_int/network_generate_report.txt) || true
```
**PASS:** the per-block FIFO `Gather`/`Slice`/`Concat` epochs present in the
streaming report are absent in the windowed one (leg-A confirmation).

### Gate-0D — Track-3 leg-C probe (only if pursuing the 2-D front-end)
```bash
cd "$N6DIR"
$STEDGEAI generate -m ~/eco8-neaixt/deploy/stm32n6/gate0_artifacts/gate0d_conv2d_probe.onnx \
  --target stm32n6 --st-neural-art n6-noextmem@user_neuralart.json -o /tmp/n6_g0d
```
**PASS:** int8 compiles, maps to a HW/NPU epoch, conv extent `[h>1, w>1]`.
**FAIL** like the NSNet2 `lower_arith_set_in_batch` case → leg C dead → Track 3 cancelled.

### Phase-4 — on-board latency + cosine (the real verdict)
```bash
cd "$N6DIR"
pkill -x ST-LINK_gdbserver 2>/dev/null
python n6_loader.py --config config.json -nf /tmp/n6val_win/network.c -bc N6-DK
$STEDGEAI validate -m ~/eco8-neaixt/cp_convfsenet_win/g_best_win256.onnx --target stm32n6 \
  --st-neural-art n6-noextmem@user_neuralart.json \
  --mode target -d serial:/dev/ttyACM0:921600
# per-epoch HW/SW split + utilization:
RUNNER=~/stedgeai/install/4.0/scripts/ai_runner
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="$RUNNER" \
  /tmp/profenv/bin/python "$RUNNER/examples/npu_profiler.py" \
  -d serial:/dev/ttyACM0:921600 -c /tmp/n6val_win -b 16
```
**Decision gate:** does measured per-frame latency beat ConvFSENet streaming
(4.40 ms) and ideally `monarch_full` (2.13 ms) at mask cos ≥ 0.99? Note the
windowed graph recomputes the full 43-col RF per frame, so the *good* outcome is
that this recompute is absorbed by the idle MAC array (compute-bound, high util)
— if instead it pushes latency up, the array did NOT fill (the leg-C risk) and
Track 1 lands ~streaming speed (still removes the FIFO Hybrid floor).

## Cloud fallback (no local board)
`deploy/stm32n6/cloud/dev_cloud_bench.py` benchmarks on ST's Edge AI Developer
Cloud N6 farm. Needs `cloud/.env` (MyST creds) + a `stm32ai-modelzoo-services`
checkout:
```bash
.venv/bin/python deploy/stm32n6/cloud/dev_cloud_bench.py \
  --model cp_convfsenet_win/g_best_win256.onnx --board STM32N6570-DK
```
This returns macc / cycles / duration_ms / RAM / ROM without a wired board.
