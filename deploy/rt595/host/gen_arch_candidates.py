#!/usr/bin/env python
"""Candidate RT595-native streaming enhancers, as whole int8 TFLite graphs.

Designed against the measured HiFi4 cost law (iss/kernel_bench.cpp):

    cycles ~ MAC/3.2 + 20.9 * output_elements + 1300 per op

which says three things this architecture obeys:

  * An op is output-bound unless k*Cin > 67, so NO depthwise (k*Cin = 3) and no
    grouped convs (24). Every conv here is dense with k*Cin >= 144.
  * Output elements are as expensive as ~67 MACs each, so the stem strides the
    257 bins down to F rows and every block works at that reduced height.
  * Op count is not free, and the real killer in ConvFSENet's 190-op graph is the
    ~160 non-conv ops. So: ReLU (fuses into the conv, unlike PReLU), no Concat or
    Slice for the FIFO, no Reshape or Transpose anywhere.

The FIFO is the host's job, exactly as model_se_stream.cpp already does states:
block i takes its k_t-wide window as an input tensor and returns its own input
column, and the host memmoves. That keeps the time convolution a single Conv2D.

Built in Keras rather than torch because ai-edge-torch inserts NCHW->NHWC
TRANSPOSEs around every conv -- precisely the op class we are avoiding.

    ~/.venvs/rt595-export/bin/python gen_arch_candidates.py --out_dir <work dir>
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

A, B, COST_OP = 1 / 3.20, 20.9, 1296
BUDGET = 3_168_000


def build(F_in, stride, C, n_blocks, k_t, k_f, full_band=False):
    """Streaming graph: (frame, per-block windows) -> (mask, per-block columns)."""
    F = F_in // stride
    frame = tf.keras.Input(batch_size=1, shape=(F_in, 1, 1), name="frame")
    wins = [tf.keras.Input(batch_size=1, shape=(F, k_t, C), name=f"win{i}")
            for i in range(n_blocks)]

    # Stem: stride over frequency. 'same' keeps F = F_in / stride rows.
    x = tf.keras.layers.Conv2D(C, (2 * stride, 1), strides=(stride, 1), padding="same",
                               activation="relu", name="stem")(frame)
    columns = []
    for i in range(n_blocks):
        columns.append(x)                                     # what the host shifts in
        # Time: one Conv2D over the k_t-wide window -> a single column. No Concat.
        t = tf.keras.layers.Conv2D(C, (1, k_t), padding="valid", activation="relu",
                                   name=f"b{i}_t")(wins[i])
        # Frequency: k_f x 1 over the rows.
        f = tf.keras.layers.Conv2D(C, (k_f, 1), padding="same", activation="relu",
                                   name=f"b{i}_f")(t)
        if full_band and i == 0:
            # Frequency reach, the lesson N6Net v1 paid for: a stem stride of 4 and
            # k_f=3 over n blocks reaches +-(k_f//2)*n rows -- 17 of 256 bins at
            # s4/k_f3/b2, which is v1's failure exactly. A pooled global context
            # broadcast back over the rows fixes it for ~1 % of the MACs: MEAN is
            # F*C reads to C outputs, and the broadcast Add is the only F*C-output
            # op it adds.
            g = tf.keras.layers.Lambda(lambda z: tf.reduce_mean(z, axis=1, keepdims=True),
                                       name=f"b{i}_pool")(f)
            g = tf.keras.layers.Conv2D(C, (1, 1), activation="relu", name=f"b{i}_ctx")(g)
            f = tf.keras.layers.Add(name=f"b{i}_bcast")([f, g])
        x = tf.keras.layers.Add(name=f"b{i}_add")([t, f])
    mask = tf.keras.layers.Conv2D(stride, (3, 1), padding="same", activation="sigmoid",
                                  name="head")(x)
    return tf.keras.Model([frame] + wins, [mask] + columns)


def predict(F_in, stride, C, n_blocks, k_t, k_f):
    F = F_in // stride
    mac = F * (2 * stride) * 1 * C
    out = F * C
    ops = 1
    for _ in range(n_blocks):
        mac += F * k_t * C * C; out += F * C; ops += 1
        mac += F * k_f * C * C; out += F * C; ops += 1
        out += F * C; ops += 1                                # Add
    mac += F * 3 * C * stride; out += F * stride; ops += 2    # head + sigmoid
    return A * mac + B * out + COST_OP * ops, mac


CANDIDATES = [
    # name,            F_in, stride, C,  blocks, k_t, k_f
    ("rt595a_s4_C64_b2", 256, 4, 64, 2, 6, 3),
    ("rt595b_s4_C48_b3", 256, 4, 48, 3, 6, 3),
    ("rt595c_s2_C48_b2", 256, 2, 48, 2, 4, 3),
    # The same nets with full-band frequency reach -- the v2 A/B says this is worth
    # +0.16 PESQ on Neural-ART; the question here is what it costs in cycles.
    ("rt595a_fb_s4_C64_b2", 256, 4, 64, 2, 6, 3, True),
    ("rt595b_fb_s4_C48_b3", 256, 4, 48, 3, 6, 3, True),
]


def to_int8(model, shapes):
    """Quantize, feeding the representative data in the CONVERTER's input order.

    A multi-input Keras model does not necessarily convert with its inputs in the
    order they were declared, and the calibrator matches by position -- feeding the
    declared order silently hands the 1-channel frame to a block's time conv and
    dies with "input_channel % filter_input_channel != 0". So convert FP32 first
    and read the real order back off the flatbuffer.
    """
    fp32 = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    det = tf.lite.Interpreter(model_content=fp32).get_input_details()
    order = [tuple(int(d) for d in x["shape"]) for x in det]
    for s in order:
        assert s in shapes, f"converter input {s} not among declared {shapes}"

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8

    def rep():
        for _ in range(32):
            yield [np.random.randn(*s).astype("float32") for s in order]
    conv.representative_dataset = rep
    return conv.convert()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="arch")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    np.random.seed(0); tf.random.set_seed(0)

    entries = []
    for name, F_in, stride, C, nb, k_t, k_f, *fb in CANDIDATES:
        m = build(F_in, stride, C, nb, k_t, k_f, bool(fb and fb[0]))
        F = F_in // stride
        shapes = [(1, F_in, 1, 1)] + [(1, F, k_t, C)] * nb
        blob = to_int8(m, shapes)
        open(os.path.join(a.out_dir, f"{name}.tflite"), "wb").write(blob)
        cyc, mac = predict(F_in, stride, C, nb, k_t, k_f)
        params = m.count_params()
        entries.append((name, blob, mac))
        print(f"  {name:<20} {len(blob)/1024:>6.0f} KiB  {params:>8,} params  "
              f"{mac:>10,} MAC  predicted {cyc:>10,.0f} cyc ({cyc/BUDGET:.2f}x budget)")

    hdr = os.path.join(a.out_dir, "kernel_bench_models.h")
    with open(hdr, "w") as f:
        f.write("// Generated by host/gen_arch_candidates.py -- do not edit.\n")
        f.write("#pragma once\n#include <stddef.h>\n#include <stdint.h>\n\n")
        for name, blob, _ in entries:
            f.write(f"alignas(16) static const unsigned char kModel_{name}[] = {{\n")
            for i in range(0, len(blob), 16):
                f.write("  " + ",".join(f"0x{b:02x}" for b in blob[i:i + 16]) + ",\n")
            f.write("};\n\n")
        f.write("typedef struct { const char* name; const unsigned char* data;\n"
                "                 size_t size; long long macs; } KernelBenchModel;\n\n")
        f.write("static const KernelBenchModel kKernelBenchModels[] = {\n")
        for name, blob, mac in entries:
            f.write(f'  {{ "{name}", kModel_{name}, sizeof(kModel_{name}), {mac}LL }},\n')
        f.write("};\n")
        f.write(f"static const int kKernelBenchModelCount = {len(entries)};\n")
    print(f"\nWrote {hdr}")


if __name__ == "__main__":
    main()
