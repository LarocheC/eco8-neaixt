"""Export the NSNet2 streaming wrapper to FP32 ONNX.

CLI mirrors ``inference.py`` (``--checkpoint_file <path>``); writes
``cp_<run>/g_best_fp32.onnx`` by default. Optional ``--output`` overrides.
See ``.planning/phases/02-fp32-onnx-export/02-CONTEXT.md`` for D-13..D-16
(output path, telemetry) and Phase 1 D-05 (variant fallback).
"""

import argparse
import contextlib
from collections import Counter
from pathlib import Path

import onnx
import torch

from nsnet2.streaming import NSNet2Streaming


def _blockdiag_multiply_export(x, weight):
    """ONNX-export-friendly blockdiag_multiply for the 2D streaming path.

    Numerically identical to torch_structured.monarch.BlockdiagMultiply, but
    relies only on constant-target Reshapes (`reshape(-1, k, p)` /
    `reshape(-1, k*q)`) so ORT's symbolic_shape_infer succeeds on the exported
    graph. The streaming wrapper always feeds 2D `(B, n)` into fc_in/fc1/fc2/
    fc_out, so dropping the arbitrary-rank `...` form is safe here.

    Failure modes ruled out by this shape:
      - einops.rearrange (upstream `blockdiag_multiply_reference`): the torch
        backend bakes trace-time leading dims into literal Reshape targets
        that don't generalize.
      - unflatten(-1, ...) / flatten(-2, -1): emit Reshape whose shape input
        is a `Concat(Shape(x)[:-1], const)` chain. ORT's symbolic_shape_infer
        asserts on `is_literal(shape_rank)` for that pattern (it cannot prove
        the rank in advance) and aborts during quant_pre_process.

    x:      (B, n) where n == nblocks * p
    weight: (nblocks, q, p)
    returns:(B, nblocks * q)
    """
    nblocks, q, p = weight.shape
    x_r = x.reshape(-1, nblocks, p)                      # (B, nblocks, p)
    out = torch.einsum('bkp,kqp->bkq', x_r, weight)      # (B, nblocks, q)
    return out.reshape(-1, nblocks * q)                  # (B, nblocks*q)


def _blockdiag_butterfly_multiply_export(x, w1_bfly, w2_bfly):
    """ONNX-export-friendly two-factor Monarch multiply (block-diagonal x
    permutation x block-diagonal) for the 2D streaming path.

    Numerically identical to torch_structured.monarch.BlockdiagButterflyMultiply
    (== blockdiag_butterfly_multiply_reference version 2), but every Reshape
    target uses a `-1` batch placeholder and the intermediate permutation is a
    plain reshape+transpose instead of einops.rearrange, so ORT's
    symbolic_shape_infer succeeds (same rationale as _blockdiag_multiply_export).

    The permutation between the two factors is exactly what makes this a genuine
    Monarch rather than a block-diagonal — it is preserved in the exported graph,
    so the deployed model keeps full cross-channel mixing and the two-factor
    parameter savings.

    x:       (B, n) where n == k * p
    w1_bfly: (k, q, p)
    w2_bfly: (l, s, r)   with l * r == k * q
    returns: (B, s * l)
    """
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    x_r = x.reshape(-1, k, p)                             # b (k p) -> b k p
    out1 = torch.einsum('kqp,bkp->bkq', w1_bfly, x_r)    # b k q
    out1 = out1.reshape(-1, r, l).transpose(1, 2)        # b (r l) -> b l r
    out2 = torch.einsum('lsr,blr->bsl', w2_bfly, out1)   # b s l
    return out2.reshape(-1, s * l)                        # b (s l)


def _butterfly_export_forward(self, input, transpose=False, conjugate=False, subtwiddle=False):
    """Structure-preserving export-friendly replacement for Butterfly.forward.

    Same math as torch_structured.butterfly.multiply.butterfly_multiply_torch
    plus the surrounding pre/post_process, but every Reshape/expand target
    uses a `-1` placeholder for the batch axis instead of a runtime
    `Shape(input)[0]` extract. Without that change the trace produces ~1500
    nodes whose Reshape shape inputs are Concat(Shape(...), const) chains
    that ORT's symbolic_shape_infer cannot propagate ("Incomplete symbolic
    shape inference" → quant_pre_process abort).

    Critically this preserves the structured (twiddle) weight layout in the
    exported graph — the deployed model still gets the parameter savings,
    int8 quant operates on the small twiddle tensors, and downstream
    decomposition tooling sees the butterfly factors rather than a dense
    matrix.

    Restricted to the call path NSNet2's FC layers actually use:
      transpose=False, conjugate=False, subtwiddle=False, complex=False.
    Asserts on the unsupported flags so silent corruption is impossible.

    Input contract: 2D `(B, in_size)` — what the streaming wrapper feeds via
    fc_in/fc1/fc2/fc_out and what each StructuredGRUCell projection sees.
    """
    assert not (transpose or conjugate or subtwiddle), (
        "_butterfly_export_forward only supports the default forward path"
    )
    assert not self.complex, (
        "_butterfly_export_forward only supports real-valued Butterfly"
    )

    twiddle = self.twiddle                                # (nstacks, nblocks, log_n, n//2, 2, 2)
    nstacks = int(self.nstacks)
    nblocks = int(self.nblocks)
    log_n   = int(twiddle.shape[2])
    n       = 1 << log_n

    # --- pre_process (no Shape extracts) ---------------------------------
    input_size = input.size(-1)                           # static at trace
    output = input.reshape(-1, input_size)                # (B, in_size)
    output = output.unsqueeze(1).expand(-1, nstacks, input_size)  # (B, nstacks, in_size)

    # --- butterfly_multiply (math from butterfly_multiply_torch, view -1) -
    if input_size < n:
        output = torch.nn.functional.pad(output, (0, n - input_size))
    elif input_size > n:
        output = output[:, :, :n]
    out_size_intrinsic = self.out_size if nstacks == 1 else n

    cur_inc = self.increasing_stride
    for block in range(nblocks):
        for idx in range(log_n):
            log_stride = idx if cur_inc else log_n - 1 - idx
            stride = 1 << log_stride
            t = twiddle[:, block, idx].view(
                nstacks, n // (2 * stride), stride, 2, 2
            ).permute(0, 1, 3, 4, 2)
            out_r = output.view(-1, nstacks, n // (2 * stride), 1, 2, stride)
            output = (t * out_r).sum(dim=4)
        cur_inc = not cur_inc
    output = output.view(-1, nstacks, n)
    if out_size_intrinsic != n:
        output = output[:, :, :out_size_intrinsic]

    # --- post_process (no Shape extracts) --------------------------------
    output = output.reshape(-1, nstacks * out_size_intrinsic)  # (B, nstacks*o)
    if self.out_size != output.shape[-1]:
        output = output[:, :self.out_size]
    if self.bias is not None:
        output = output + self.bias[:self.out_size]
    return output                                         # (B, out_size)


@contextlib.contextmanager
def _patch_structured_for_export(model):
    """Swap structured-FC custom ops with ONNX-export-friendly replacements
    that PRESERVE structure.

    torch.onnx.export(dynamo=False) cannot trace:
      - torch_structured.monarch.BlockdiagMultiply / BlockdiagButterflyMultiply
        (custom torch.autograd.Functions whose forward calls torch.bmm with the
        out= kwarg — emit prim::PythonOp, then trip
        _jit_pass_onnx_graph_shape_type_inference).
      - torch.ops.torch_structured.butterfly_multiply (custom registered C++ op).

    Strategy per variant:
      - Block-diagonal ("blockdiag"): substitute the module-level
        blockdiag_multiply with _blockdiag_multiply_export (constant-target
        Reshape + Einsum). Preserves the single block-diagonal weight layout.
      - Monarch ("monarch", genuine 2-factor): substitute the module-level
        blockdiag_butterfly_multiply with _blockdiag_butterfly_multiply_export.
        Preserves both factors AND the permutation between them (the thing that
        makes it a real Monarch, not a block-diagonal).
      - Butterfly: substitute Butterfly.forward with _butterfly_export_forward
        — same math as butterfly_multiply_torch + pre/post_process, but every
        Reshape/expand target uses -1 for the batch dim so ORT's symbolic
        shape inference succeeds. Preserves the structured (twiddle) weight
        layout — required for downstream decomposition / structured deploy.

    `model` is walked only to short-circuit the patch when no variant
    instances exist; both substitutions are class/module-level so all
    instances (including those nested inside StructuredGRUCells for the
    structured-GRU configs) pick up the patch automatically.

    Patches are removed in `finally` so subsequent code paths (validation,
    training, runtime inference) keep the fast kernels.
    """
    patches = []

    # --- Block-diagonal: function-level swap on the module namespace ---------
    try:
        from torch_structured.monarch import blockdiag_linear as _bd_mod
        patches.append((_bd_mod, "blockdiag_multiply", _bd_mod.blockdiag_multiply))
        _bd_mod.blockdiag_multiply = _blockdiag_multiply_export
    except ImportError:
        pass

    # --- Monarch (genuine 2-factor): swap blockdiag_butterfly_multiply -------
    try:
        from torch_structured.monarch import monarch_linear as _mon_mod
        patches.append(
            (_mon_mod, "blockdiag_butterfly_multiply", _mon_mod.blockdiag_butterfly_multiply)
        )
        _mon_mod.blockdiag_butterfly_multiply = _blockdiag_butterfly_multiply_export
    except ImportError:
        pass

    # --- Butterfly: class-level swap on Butterfly.forward --------------------
    butterfly_cls = None
    try:
        from torch_structured.butterfly.butterfly import Butterfly as butterfly_cls
        butterfly_cls._original_butterfly_forward = butterfly_cls.forward
        butterfly_cls.forward = _butterfly_export_forward
    except ImportError:
        butterfly_cls = None

    # --- Triton GRULayer instances: flip use_triton=False so the per-step
    # Python body runs instead of the persistent Triton kernel (which goes
    # through a custom autograd.Function and won't trace). The per-step path
    # uses standard PyTorch ops + the structured matmuls above, both of
    # which now lower cleanly to ONNX. -------------------------------------
    use_triton_restore = []
    for sub in model.modules():
        if (
            hasattr(sub, "use_triton")
            and isinstance(getattr(sub, "use_triton"), bool)
            and sub.use_triton
        ):
            use_triton_restore.append(sub)
            sub.use_triton = False

    try:
        yield
    finally:
        for mod, attr, original in patches:
            setattr(mod, attr, original)
        if butterfly_cls is not None and hasattr(butterfly_cls, "_original_butterfly_forward"):
            butterfly_cls.forward = butterfly_cls._original_butterfly_forward
            del butterfly_cls._original_butterfly_forward
        for sub in use_triton_restore:
            sub.use_triton = True


def export_streaming(checkpoint_file, output_path=None):
    """Export the streaming view of a trained NSNet2 to FP32 ONNX.

    checkpoint_file: path to a g_best (or any generator) checkpoint file.
    output_path: optional override; defaults to <ckpt_dir>/g_best_fp32.onnx (D-13).

    Returns the resolved output path (pathlib.Path).
    Structured-variant GRUs (gru_kind in {'butterfly','blockdiag','monarch'})
    flow through NSNet2Streaming._forward_step_structured, which delegates to
    the StructuredGRU's Python time-loop. Each StructuredGRUCell's W_ih and W_hh
    projections are BlockdiagLinear, MonarchLinear, or Butterfly modules — all
    substituted by _patch_structured_for_export, so the trace contains only
    standard ONNX primitives. No separate gru_kind guard needed here.
    """
    streaming = NSNet2Streaming.from_checkpoint(checkpoint_file)

    if output_path is None:
        output_path = Path(checkpoint_file).parent / "g_best_fp32.onnx"          # D-13 default
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read shapes from the constructed model — never hardcode shape literals.
    n_freq = streaming.base.h.n_fft // 2 + 1                                     # (B, n_freq)
    H = streaming.hidden_size                                                    # hidden dim
    L = streaming.num_layers                                                     # num GRU layers
    frame_in = torch.zeros(1, n_freq, dtype=torch.float32)                       # (B=1, n_freq) example input
    states_in = torch.zeros(L, 1, H, dtype=torch.float32)                        # (L, B=1, H)   example input

    streaming.eval()
    with _patch_structured_for_export(streaming):
        torch.onnx.export(
            streaming,
            (frame_in, states_in),
            str(output_path),
            input_names=["frame_in", "states_in"],
            output_names=["mask", "states_out"],
            dynamic_axes={
                "frame_in":   {0: "B"},                                              # batch axis only — EXP-01
                "states_in":  {1: "B"},                                              # (L, B=dyn, H)
                "mask":       {0: "B"},
                "states_out": {1: "B"},
            },
            opset_version=17,                                                        # DET-01 onnx 1.21.x compat
            dynamo=False,                                                            # MANDATORY — Pitfall 10 prevention
        )

    # D-16 telemetry — print() per CONVENTIONS.md.
    model = onnx.load(str(output_path))
    op_hist = Counter(n.op_type for n in model.graph.node)
    sorted_ops = dict(sorted(op_hist.items(), key=lambda kv: -kv[1]))             # deterministic, by-count
    size_mib = output_path.stat().st_size / (1024 * 1024)
    n_gru = op_hist.get("GRU", 0)
    print("Exported FP32 ONNX to {}".format(output_path))
    print("  size : {:.3f} MiB".format(size_mib))
    print("  nodes: {}".format(len(model.graph.node)))
    print("  ops  : {}".format(sorted_ops))
    print("  GRU  : {} (Pitfall 11 closure: must be 0)".format(n_gru))

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export NSNet2 streaming view to FP32 ONNX."
    )
    parser.add_argument(
        "--checkpoint_file", required=True,
        help="Path to checkpoint file (e.g., cp_baseline/g_best). "
             "Sibling config.json is auto-loaded.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .onnx path. Defaults to <checkpoint_dir>/g_best_fp32.onnx.",
    )
    a = parser.parse_args()                                                      # `a` per CONVENTIONS.md
    export_streaming(a.checkpoint_file, a.output)


if __name__ == "__main__":
    main()
