#!/usr/bin/env python
"""Single-op int8 TFLite models for the HiFi4 kernel cost model.

Why this exists: every HiFi4 cycle number this repo has is a whole-model figure on a
dense-MLP graph (NSNet2) or an unrealizable one (ConvFSENet, which did not fit the
1.24 MB DSP data segment). The per-op cost of xa_nnlib's CONV_2D on this core has
never been measured, and it is the number that decides whether a weight-reusing
convolutional enhancer is real-time on the RT595 at all.

Each model here is ONE op with a realistic streaming shape (W = 1 frame), converted
to full int8 so the xa_nnlib int8 kernels are what actually run. Built in Keras
rather than torch on purpose: ai-edge-torch inserts NCHW->NHWC TRANSPOSEs around
convs, which is exactly the cost we are trying to isolate.

    ~/.venvs/rt595-export/bin/python gen_kernel_bench.py --out_dir <work dir>

Writes kernel_bench_models.h (all models as C arrays + their MAC counts) plus the
individual .tflite files for inspection.
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

F_BINS = 257


def _conv(name, h, cin, cout, kh, macs_note=None, depthwise=False, groups=1):
    """(1, h, 1, cin) -> Conv2D (kh, 1) -> (1, h, 1, cout), 'same' padding."""
    inp = tf.keras.Input(batch_size=1, shape=(h, 1, cin), dtype=tf.float32)
    if depthwise:
        out = tf.keras.layers.DepthwiseConv2D((kh, 1), padding="same", use_bias=True)(inp)
        macs = h * kh * cin
    else:
        out = tf.keras.layers.Conv2D(cout, (kh, 1), padding="same", use_bias=True,
                                     groups=groups)(inp)
        macs = h * kh * (cin // groups) * cout
    return name, tf.keras.Model(inp, out), (1, h, 1, cin), macs


def _dense(name, nin, nout):
    inp = tf.keras.Input(batch_size=1, shape=(nin,), dtype=tf.float32)
    out = tf.keras.layers.Dense(nout, use_bias=True)(inp)
    return name, tf.keras.Model(inp, out), (1, nin), nin * nout


def _batch_matmul(name, b, m, k, n):
    """(1, b, m, k) x const (b, k, n) -> BATCH_MATMUL, the blockdiag lowering shape."""
    inp = tf.keras.Input(batch_size=1, shape=(b, m, k), dtype=tf.float32)
    w = tf.constant(np.random.randn(b, k, n).astype("float32") * 0.05)
    out = tf.keras.layers.Lambda(lambda x: tf.matmul(x, w))(inp)
    return name, tf.keras.Model(inp, out), (1, b, m, k), b * m * k * n


def zoo():
    """The shapes that decide our architectures, not a generic sweep."""
    m = []
    # --- FULLY_CONNECTED: NSNet2's own shapes; the only op measured fast so far ---
    m.append(_dense("fc_257x400", 257, 400))
    m.append(_dense("fc_400x400", 400, 400))
    m.append(_dense("fc_600x600", 600, 600))
    # --- CONV_2D 1x1 pointwise over the full spectrum: ConvFSENet's TCM ends ---
    m.append(_conv("pw_F257_C64", F_BINS, 64, 64, 1))
    m.append(_conv("pw_F257_C96", F_BINS, 96, 96, 1))
    m.append(_conv("pw_F257_C192", F_BINS, 192, 192, 1))
    m.append(_conv("pw_F64_C96", 64, 96, 96, 1))
    # --- CONV_2D k x 1 over frequency: the N6Net shape, weights reused across bins ---
    m.append(_conv("conv_k3_F257_C96", F_BINS, 96, 96, 3))
    m.append(_conv("conv_k5_F257_C96", F_BINS, 96, 96, 5))
    m.append(_conv("conv_k3_F64_C96", 64, 96, 96, 3))
    m.append(_conv("conv_k3_F257_C48", F_BINS, 48, 48, 3))
    # --- DEPTHWISE: ConvFSENet's TCM middle ---
    m.append(_conv("dw_k3_F257_C192", F_BINS, 192, 192, 3, depthwise=True))
    # --- grouped conv: the blockdiag lowering that WORKS on Neural-ART ---
    m.append(_conv("grp4_F257_C96", F_BINS, 96, 96, 1, groups=4))
    # --- BATCH_MATMUL: what torch_structured actually emits; the slow path ---
    m.append(_batch_matmul("bmm_b4_m1_64x64", 4, 1, 64, 64))
    m.append(_batch_matmul("bmm_b8_m1_64x64", 8, 1, 64, 64))
    return m


def to_int8(model, in_shape):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8

    def rep():
        for _ in range(32):
            yield [np.random.randn(*in_shape).astype("float32")]
    conv.representative_dataset = rep
    return conv.convert()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="kernel_bench")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    np.random.seed(0)
    tf.random.set_seed(0)

    entries = []
    for name, model, in_shape, macs in zoo():
        try:
            blob = to_int8(model, in_shape)
        except Exception as e:                      # one bad shape must not kill the sweep
            print(f"  SKIP {name}: {type(e).__name__}: {e}")
            continue
        path = os.path.join(a.out_dir, f"{name}.tflite")
        open(path, "wb").write(blob)
        entries.append((name, blob, macs))
        print(f"  {name:<20} {len(blob):>7,} B   {macs:>12,} MAC")

    hdr = os.path.join(a.out_dir, "kernel_bench_models.h")
    with open(hdr, "w") as f:
        f.write("// Generated by host/gen_kernel_bench.py -- do not edit.\n")
        f.write("#pragma once\n#include <stddef.h>\n#include <stdint.h>\n\n")
        for name, blob, _ in entries:
            f.write(f"alignas(16) static const unsigned char kModel_{name}[] = {{\n")
            for i in range(0, len(blob), 16):
                f.write("  " + ",".join(f"0x{b:02x}" for b in blob[i:i + 16]) + ",\n")
            f.write("};\n\n")
        f.write("typedef struct { const char* name; const unsigned char* data;\n"
                "                 size_t size; long long macs; } KernelBenchModel;\n\n")
        f.write("static const KernelBenchModel kKernelBenchModels[] = {\n")
        for name, blob, macs in entries:
            f.write(f'  {{ "{name}", kModel_{name}, sizeof(kModel_{name}), {macs}LL }},\n')
        f.write("};\n")
        f.write(f"static const int kKernelBenchModelCount = {len(entries)};\n")
    print(f"\nWrote {hdr} ({os.path.getsize(hdr)/1024:.0f} KiB, {len(entries)} models)")


if __name__ == "__main__":
    main()
