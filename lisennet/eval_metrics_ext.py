"""Extended metrics (STOI, SI-SDR, DNSMOS) for the paper's key systems.

Protocol matches Table 1 of the paper:
  * RT-int8  = published static-int8 ONNX mask + noisy phase (no Griffin-Lim).
    - GRU repro: full-utterance graph  cp_lisennet/g_best_int8_static.onnx
    - nc24 / relu6-deep: the *windowed* signed-int8 deploy artifacts
      (feat_window -> last-64-frame est_mag), driven as at deployment:
      left zero-pad warm-up, hop 64.
  * FP32 offline = torch model + 2-iteration Griffin-Lim (the trained recipe).

Metrics per system, averaged over the full 824-utt VoiceBank-DEMAND test split:
  PESQ-WB (consistency check vs the paper), STOI, SI-SDR (dB),
  DNSMOS P.835 (SIG/BAK/OVRL, non-personalized) + P.808 MOS.

DNSMOS: official ONNX models + constants from microsoft/DNS-Challenge
dnsmos_local.py (sig_bak_ovr.onnx, model_v8.onnx in ~/.cache/dnsmos/),
reimplemented to take in-memory waveforms.

Run:  python -m lisennet.eval_metrics_ext [n_utts]
Writes paper/data/extended_metrics.json (or *_<n>.json for partial runs).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import torch
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common.dataset import Dataset, load_voicebank_demand  # noqa: E402
from common.env import AttrDict  # noqa: E402
from lisennet.export_onnx import _load_from_checkpoint  # noqa: E402

SR = 16000

# --------------------------------------------------------------------------- DNSMOS
DNSMOS_DIR = Path.home() / ".cache" / "dnsmos"
INPUT_LENGTH = 9.01  # seconds, official constant


class DNSMOS:
    """Official DNSMOS P.835 + P.808, waveform-in (16 kHz float)."""

    def __init__(self, model_dir=DNSMOS_DIR):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.primary = ort.InferenceSession(str(model_dir / "sig_bak_ovr.onnx"),
                                            sess_options=so, providers=["CPUExecutionProvider"])
        self.p808 = ort.InferenceSession(str(model_dir / "model_v8.onnx"),
                                         sess_options=so, providers=["CPUExecutionProvider"])
        # non-personalized polynomial mappings (dnsmos_local.py)
        self.p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        self.p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        self.p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

    def _melspec(self, audio):
        mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=321,
                                             hop_length=160, n_mels=120)
        return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T

    def __call__(self, audio: np.ndarray) -> dict:
        audio = np.asarray(audio, dtype=np.float32)
        len_samples = int(INPUT_LENGTH * SR)
        while len(audio) < len_samples:            # official tiling for short clips
            audio = np.append(audio, audio)
        num_hops = int(np.floor(len(audio) / SR) - INPUT_LENGTH) + 1
        sig, bak, ovr, p808 = [], [], [], []
        for idx in range(num_hops):
            seg = audio[int(idx * SR): int((idx + INPUT_LENGTH) * SR)]
            if len(seg) < len_samples:
                continue
            mel = self._melspec(seg[:-160]).astype("float32")[np.newaxis, :, :]
            p808.append(float(self.p808.run(None, {"input_1": mel})[0][0][0]))
            s, b, o = self.primary.run(None, {"input_1": seg[np.newaxis, :]})[0][0]
            sig.append(float(self.p_sig(s)))
            bak.append(float(self.p_bak(b)))
            ovr.append(float(self.p_ovr(o)))
        return {"SIG": np.mean(sig), "BAK": np.mean(bak),
                "OVRL": np.mean(ovr), "P808": np.mean(p808)}


# --------------------------------------------------------------------------- classic metrics
def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.dot(target, target) + 1e-12) /
                               (np.dot(noise, noise) + 1e-12)))


def _pesq_one(ref, est):
    from pesq import pesq as _pesq
    try:
        return _pesq(SR, ref, est)
    except Exception:
        return np.nan


def _stoi_one(ref, est):
    from pystoi import stoi
    n = min(len(ref), len(est))
    try:
        return float(stoi(ref[:n], est[:n], SR, extended=False))
    except Exception:
        return np.nan


# --------------------------------------------------------------------------- mask backends
def _session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def full_graph_mask(sess, feat_np):
    """feat (1,3,T,F) -> est_mag (1,T,F), full-utterance graph."""
    return sess.run(["est_mag"], {"feat": feat_np})[0]


def windowed_mask(sess, feat_np, window, emit_t=64):
    """Drive the fixed-size deploy graph over a full utterance.

    Window covers [s-ctx, s+emit_t) after ctx frames of left zero-padding
    (deployment warm-up); the graph emits the last emit_t frames. Causal
    model => right-edge zero-padding cannot affect earlier frames; the
    padded tail is truncated.
    """
    ctx = window - emit_t
    t_tot = feat_np.shape[2]
    x = np.pad(feat_np, ((0, 0), (0, 0), (ctx, 0), (0, 0)))
    outs = []
    for s in range(0, t_tot, emit_t):
        win = x[:, :, s: s + window, :]
        if win.shape[2] < window:
            win = np.pad(win, ((0, 0), (0, 0), (0, window - win.shape[2]), (0, 0)))
        m = sess.run(["est_mag"], {"feat_window": win.astype(np.float32)})[0]
        outs.append(m)
    return np.concatenate(outs, axis=1)[:, :t_tot]


# --------------------------------------------------------------------------- systems
SYSTEMS = {
    # name: (checkpoint g_best, int8 onnx or None, windowed size or None, use torch fp32+GL)
    "noisy": None,
    "gru_repro_rt_int8": (REPO / "cp_lisennet/g_best",
                          REPO / "cp_lisennet/g_best_int8_static.onnx", None, False),
    "nc24_rt_int8": (REPO / "cp_lisennet_conv_hardened_nc24/g_best",
                     REPO / "cp_lisennet_conv_hardened_nc24/g_best_windowed_int8_static.onnx",
                     132, False),
    "relu6deep_rt_int8": (REPO / "cp_lisennet_conv_hardened_deep_relu6/g_best",
                          REPO / "cp_lisennet_conv_hardened_deep_relu6/g_best_windowed_int8_static.onnx",
                          260, False),
    "relu6deep_fp32_offline": (REPO / "cp_lisennet_conv_hardened_deep_relu6/g_best",
                               None, None, True),
}


@torch.no_grad()
def enhance_all(n_utts):
    """Return refs, noisy list and dict name -> list of enhanced waveforms (np)."""
    with open(REPO / "cp_lisennet/config.json") as f:
        h = AttrDict(json.load(f))
    hf = load_voicebank_demand()
    ds = Dataset(hf["test"], h.segment_size, h.sampling_rate, split=False,
                 shuffle=False, seed=h.seed)
    n_utts = min(n_utts, len(ds))

    models, sessions = {}, {}
    for name, spec in SYSTEMS.items():
        if spec is None:
            continue
        ckpt, int8, window, _ = spec
        models[name] = _load_from_checkpoint(ckpt)
        if int8 is not None:
            sessions[name] = _session(int8)

    refs, noisies = [], []
    ests = {name: [] for name in SYSTEMS if name != "noisy"}
    t0 = time.time()
    for i in range(n_utts):
        clean, noisy = ds[i]
        clean = clean.unsqueeze(0)
        noisy = noisy.unsqueeze(0)
        length = noisy.shape[-1]
        refs.append(clean.squeeze().numpy())
        noisies.append(noisy.squeeze().numpy())

        for name, spec in SYSTEMS.items():
            if spec is None:
                continue
            ckpt, int8, window, use_torch_gl = spec
            model = models[name]
            spec_c = model.power_compress(model.apply_stft(noisy))
            src_mag, src_pha = spec_c.abs(), spec_c.angle()
            feat = model.build_features(src_mag, src_pha)
            if use_torch_gl:
                est_mag = model.apply_mask(model.predict_mask(feat), src_mag)
                est_pha = model.griffinlim(est_mag, src_pha, length)
            else:
                fn = feat.numpy().astype(np.float32)
                em = (windowed_mask(sessions[name], fn, window) if window
                      else full_graph_mask(sessions[name], fn))
                est_mag = torch.from_numpy(em)
                est_pha = src_pha
            est_spec = torch.complex(est_mag * est_pha.cos(), est_mag * est_pha.sin())
            wav = model.apply_istft(model.power_uncompress(est_spec), length=length)
            ests[name].append(wav.squeeze().numpy())
        if (i + 1) % 50 == 0:
            print(f"  enhanced {i+1}/{n_utts}  ({time.time()-t0:.0f}s)", flush=True)
    return refs, noisies, ests


def score_system(name, refs, wavs, n_jobs=24):
    pesqs = Parallel(n_jobs=n_jobs)(delayed(_pesq_one)(r, e) for r, e in zip(refs, wavs))
    stois = Parallel(n_jobs=n_jobs)(delayed(_stoi_one)(r, e) for r, e in zip(refs, wavs))
    sisdrs = [si_sdr(r, e) for r, e in zip(refs, wavs)]
    dns = DNSMOS()
    dnsmos = Parallel(n_jobs=n_jobs, backend="threading")(delayed(dns)(e) for e in wavs)
    out = {
        "n": len(wavs),
        "PESQ": float(np.nanmean(pesqs)),
        "STOI": float(np.nanmean(stois)),
        "SI-SDR": float(np.mean(sisdrs)),
        "DNSMOS_SIG": float(np.mean([d["SIG"] for d in dnsmos])),
        "DNSMOS_BAK": float(np.mean([d["BAK"] for d in dnsmos])),
        "DNSMOS_OVRL": float(np.mean([d["OVRL"] for d in dnsmos])),
        "P808_MOS": float(np.mean([d["P808"] for d in dnsmos])),
    }
    print(f"{name:26s} " + "  ".join(f"{k}={v:.3f}" for k, v in out.items() if k != "n"),
          flush=True)
    return out


def main():
    n_utts = int(sys.argv[1]) if len(sys.argv) > 1 else 824
    print(f"Enhancing {n_utts} utterances for {len(SYSTEMS)-1} systems ...", flush=True)
    refs, noisies, ests = enhance_all(n_utts)

    results = {}
    results["clean"] = score_system("clean", refs, refs)          # anchors
    results["noisy"] = score_system("noisy", refs, noisies)
    for name, wavs in ests.items():
        results[name] = score_system(name, refs, wavs)

    suffix = "" if n_utts >= 824 else f"_{n_utts}"
    out = REPO / "paper" / "data" / f"extended_metrics{suffix}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
