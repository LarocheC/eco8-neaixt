# FSPEN — integration status

FSPEN (Yang *et al.*, [ICASSP 2024](https://ieeexplore.ieee.org/document/10446016)) re-implemented into this framework: an ultra-lightweight full-band + sub-band enhancer predicting a **complex mask** (learned phase — no Griffin-Lim), with dual-path-extension (DPE) blocks whose only time-recurrent state is a grouped inter-frame GRU. This makes it the natural complex-mask counterpart to LiSenNet's magnitude-mask + noisy-phase recipe at the same ~35 k parameter scale.

This file tracks the integration; **training on VoiceBank-DEMAND has not been run yet**, so there are no quality numbers here. Everything below is an implementation-level fact — the parity/quant gates are verified by `tests/test_fspen_*`; the RTF table is a one-off host measurement (not test-enforced).

## Model

| item | value |
| --- | --- |
| parameters | **34,796** (`configs/fspen.json`, the reference architecture config) |
| STFT | n_fft 512 / hop 256 @ 16 kHz, Hann (repo standard; the reference repo profiles with Hamming) |
| sub-band split | 5 groups of the 257 bins — widths 2/3/6/11/20 per band, 8/6/6/6/6 bands = 32 bands |
| DPE | 3 blocks; intra = bidir GRU(16) over the 32 bands, inter = 8-group GRU(16) over time |
| streaming state | 3 tensors — one `(groups=8, B*32 bands, 2)` inter-GRU hidden per DPE block; nothing else |

## Provenance and parity vs the reference implementation

The unofficial reference ([gitwukeyi/FSPEN](https://github.com/gitwukeyi/FSPEN)) has no license file, so the model was re-implemented from the paper with the layer hyperparameters pinned to the reference config. With weights copied across, `enhance_spectrum` is **bit-exact (max abs err 0.0)** to the reference forward once the reference's sub-band decoder indexing bug is replicated:

* the reference never advances `start_idx` in its `SubBandDecoder` loop, so all five decoder groups read DPE-feature bands `[0:bands_g]` and bands 8–31 never reach the sub-band decoders;
* this implementation uses the paper-intended consecutive slices `[0:8], [8:14], …, [26:32]` (documented in `fspen/model.py`).

## Deploy pipeline gates (untrained weights — parities are weight-independent)

| gate | result |
| --- | --- |
| frame-by-frame streaming vs offline | max abs err **1.4e-06** (nn.GRU sequence-vs-step kernel numerics) |
| fp32 ONNX (whole-utterance, dynamic B/T) vs PyTorch | max abs err **1.4e-06** |
| fp32 streaming ONNX (explicit `state_i` I/O) frame loop vs offline | max abs err **1.4e-06** |
| dynamic int8 (weight-only) | >50% of weight elements int8 (ORT leaves GRU/ConvTranspose fp32), output within 0.5 max-abs of fp32 (smoke-tested; quality via `eval_deploy` after training) |

Graphs (opset 17, legacy tracer): whole-utterance 0.30 MiB / 1088 nodes; streaming 0.28 MiB / 995 nodes, 27 GRU nodes, 3 state tensors. Like LiSenNet's GRU variant, these graphs are ORT targets, not Neural-ART targets (GRU + LayerNormalization are NPU compiler blockers); an NPU-mappable variant would be a separate hardening effort.

## Real-time factor (one-off measurement, idle x86 dev box under WSL2, 1 thread)

| path | ms/frame | RTF (16 ms hop) |
| --- | --- | --- |
| streaming fp32 ONNX (ORT, the deploy path) | **0.61** | **0.038** |
| eager PyTorch `FSPENStreamer` (parity reference, not a deploy path) | 5.98 | 0.37 |

Both paths pinned to 1 thread (torch via `measure_rtf`, ORT via `intra_op_num_threads=1`); ORT number is best-of-3 over 2000 frames. The eager streamer pays Python dispatch for ~30 tiny modules per frame; the ORT graph is the number that matters, and it is comfortably real-time. Machine-dependent — re-measure with `python -m fspen.eval_deploy` on the target host.

## Reproducing

```bash
python -m fspen.train --config configs/fspen.json --checkpoint_path cp_fspen --training_epochs 100
python -m fspen.export_onnx --checkpoint_file cp_fspen/g_best              # whole-utterance fp32
python -m fspen.export_onnx --checkpoint_file cp_fspen/g_best --streaming  # frame-by-frame fp32
python -m fspen.quant_onnx --fp32 cp_fspen/g_best_fp32.onnx --mode dynamic
python -m fspen.quant_onnx --fp32 cp_fspen/g_best_fp32.onnx --mode static --config cp_fspen/config.json
python -m fspen.quant_onnx --fp32 cp_fspen/g_best_streaming_fp32.onnx --mode static --streaming \
                           --checkpoint cp_fspen/g_best --config cp_fspen/config.json
python -m fspen.eval_deploy --checkpoint_file cp_fspen/g_best --n_utts 824
uv run pytest tests/test_fspen_smoke.py tests/test_fspen_onnx_parity.py \
              tests/test_fspen_streaming_parity.py tests/test_fspen_streaming_export.py \
              tests/test_fspen_quant_onnx.py
```

## Next steps

1. Train on VoiceBank-DEMAND with the shared CMGAN recipe (`configs/fspen.json`); the paper reports PESQ 2.97 on VBD for its (larger, 79 k) configuration.
2. Run `fspen/eval_deploy.py` for the torch / fp32 / int8-dynamic / int8-static PESQ ladder + RTF.
3. If quality lands in the LiSenNet band: static-int8 study, and evaluate an NPU-hardening pass (the GRUs and LayerNorms would need the same treatment LiSenNet's conv variant got).
