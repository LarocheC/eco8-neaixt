"""Phase 2: correct RT-int8 metrics for the conv systems (nc24, relu6-deep).

The windowed deploy artifacts cannot reproduce Table 1's protocol on the host
(zero *feature* frames in the warm-up are not the per-layer zero padding the
offline causal graph uses), so — like Table 1 / eval_deploy — this evaluates
the FULL-UTTERANCE static-int8 graph + noisy phase.

Per system, tries the candidate quantization recipes, evaluates full-split
PESQ for each, picks the one matching the published figure (2.998 / 3.014),
then computes STOI / SI-SDR / DNSMOS on the winner and merges the row into
paper/data/extended_metrics.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/claroche/eco8-neaixt")
sys.path.insert(0, str(REPO))

from common.dataset import Dataset, load_voicebank_demand  # noqa: E402
from common.env import AttrDict  # noqa: E402
from lisennet.export_onnx import _load_from_checkpoint, export_fp32  # noqa: E402
from lisennet.quant_onnx import VBDCalibrationReader, quantize_static_int8  # noqa: E402
from lisennet.eval_metrics_ext import (_session, full_graph_mask,  # noqa: E402
                                       score_system, _pesq_one)
from joblib import Parallel, delayed  # noqa: E402

TMP = REPO / "paper" / "data" / "tmp_quant"
TMP.mkdir(exist_ok=True)

SYSTEMS = {
    "nc24_rt_int8": (REPO / "cp_lisennet_conv_hardened_nc24/g_best", 2.998),
    "relu6deep_rt_int8": (REPO / "cp_lisennet_conv_hardened_deep_relu6/g_best", 3.014),
}
RECIPES = [  # (per_channel, signed)
    ("pt_unsigned", False, False),   # eval_deploy default
    ("pt_signed", False, True),
    ("pc_signed", True, True),       # paper Sec. 4.1 wording
]


@torch.no_grad()
def enhance(model, sess, ds, n):
    refs, wavs = [], []
    for i in range(n):
        clean, noisy = ds[i]
        clean, noisy = clean.unsqueeze(0), noisy.unsqueeze(0)
        length = noisy.shape[-1]
        spec = model.power_compress(model.apply_stft(noisy))
        mag, pha = spec.abs(), spec.angle()
        feat = model.build_features(mag, pha).numpy().astype(np.float32)
        em = torch.from_numpy(full_graph_mask(sess, feat))
        est = torch.complex(em * pha.cos(), em * pha.sin())
        wav = model.apply_istft(model.power_uncompress(est), length=length)
        refs.append(clean.squeeze().numpy())
        wavs.append(wav.squeeze().numpy())
    return refs, wavs


def main():
    n_utts = int(sys.argv[1]) if len(sys.argv) > 1 else 824
    with open(REPO / "cp_lisennet/config.json") as f:
        h = AttrDict(json.load(f))
    ds = Dataset(load_voicebank_demand()["test"], h.segment_size, h.sampling_rate,
                 split=False, shuffle=False, seed=h.seed)
    n = min(n_utts, len(ds))

    out_path = REPO / "paper" / "data" / "extended_metrics.json"
    results = json.load(open(out_path)) if out_path.exists() else {}

    for name, (ckpt, target) in SYSTEMS.items():
        model = _load_from_checkpoint(ckpt)
        fp32 = TMP / f"{name}_fp32.onnx"
        if not fp32.exists():
            export_fp32(model, fp32)
        best = None
        for tag, pc, signed in RECIPES:
            q = TMP / f"{name}_{tag}.onnx"
            if not q.exists():
                try:
                    quantize_static_int8(fp32, q, VBDCalibrationReader(h, 24),
                                         per_channel=pc, signed=signed)
                except Exception as e:
                    print(f"{name} {tag}: quantization failed ({type(e).__name__}: {e})",
                          flush=True)
                    continue
            refs, wavs = enhance(model, _session(q), ds, n)
            pesqs = Parallel(n_jobs=24)(delayed(_pesq_one)(r, e)
                                        for r, e in zip(refs, wavs))
            p = float(np.nanmean(pesqs))
            print(f"{name} {tag}: PESQ {p:.4f} (target {target})", flush=True)
            if best is None or abs(p - target) < abs(best[1] - target):
                best = (tag, p, refs, wavs)
        tag, p, refs, wavs = best
        print(f"--> {name}: using recipe '{tag}' (PESQ {p:.4f})", flush=True)
        results[name] = score_system(name, refs, wavs)
        results[name]["quant_recipe"] = tag
        results[name]["graph"] = "full-utterance int8-static (Table 1 protocol)"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    print(f"updated {out_path}")


if __name__ == "__main__":
    main()
