"""D as a per-utterance confidence gate for a downstream ASR consumer.

The product-shaped question for an N6 audio front-end: the enhancer feeds a
recognizer, and enhancement sometimes *hurts* recognition — especially off
distribution. The UQ study showed member disagreement D detects spectral
novelty the input level cannot see (white noise: AUROC 1.0 vs clip-rate 0.53).
Here D gates a bypass: hand the recognizer the RAW input when the enhancer is
uncertain, its OUTPUT when it is confident.

Protocol (no ground-truth transcripts in VoiceBank-DEMAND-16k): differential
WER with pseudo-references — whisper-tiny.en transcriptions of the CLEAN
signal. All comparisons are relative (noisy vs enhanced vs gated on the same
references), which is exactly the decision the gate makes. Greedy decoding,
EnglishTextNormalizer before scoring.

Conditions: `id` (824) + the study's OOD recipes (200 each): `white5db`
(unseen noise — D fires), `gain-12` (D provably blind — negative control),
`gain+12`. Enhanced = the member-correct MinMax K=4 fold's averaged est_mag +
noisy phase (the deployment path); member stacks come from `sr_gating.py`'s
cache, so run that first (or this script fills the cache itself).

Policies, thresholds FROZEN from the ID distribution:
  D-gate      bypass if utterance-mean D_t > p{90,95,99}(ID)
  level-gate  the two-sided check in the FEATURE-QUANTIZER domain (waveform RMS
              is vacuous here — the dataset pipeline RMS-normalizes every clip):
              hot = clip-rate of the quantized input mag (q > 127) above
              p99(ID)+eps; quiet = mean |q - zp| below p1(ID) — the
              variance-collapse detector for D's quiet-side blindness
  D+level     bypass if either fires
Read-outs per condition: WER of noisy / enhanced / each policy / oracle
(per-utterance min), bypass rates, and Spearman(mean D, per-utt ASR damage).
Cached per-utterance WERs in the results JSON are reused on re-runs, so policy
changes do not re-transcribe.

usage: HF_HUB_OFFLINE=1 ./.venv/bin/python3 sr_asr_gate.py [--n 824] [--n_ood 200]
Results -> sr_asr_gate_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(REPO))

from sr_gating import CACHE, collect_one, wav_for  # noqa: E402

OUT = REPO / "sr_asr_gate_results.json"
SR = 16000
CONDS = ("id", "white5db", "gain-12", "gain+12")


# ------------------------------------------------------------------- whisper
class ASR:
    def __init__(self, batch=16):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny.en").to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.batch = batch

    def transcribe(self, wavs):
        """List of float32 waveforms -> list of transcripts (greedy)."""
        out = []
        t = self.torch
        with t.no_grad():
            for i in range(0, len(wavs), self.batch):
                chunk = [np.asarray(w, np.float32) for w in wavs[i:i + self.batch]]
                feats = self.proc(chunk, sampling_rate=SR,
                                  return_tensors="pt").input_features
                feats = feats.to(self.device)
                if self.device == "cuda":
                    feats = feats.half()
                ids = self.model.generate(feats, do_sample=False, num_beams=1,
                                          max_new_tokens=220)
                out += self.proc.batch_decode(ids, skip_special_tokens=True)
        return out


def wer(refs, hyps):
    import jiwer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    norm = EnglishTextNormalizer({})
    pairs = [(norm(r), norm(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r.strip()]
    return float(jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]))


def per_utt_wer(refs, hyps):
    import jiwer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    norm = EnglishTextNormalizer({})
    out = []
    for r, h in zip(refs, hyps):
        r, h = norm(r), norm(h)
        out.append(float(jiwer.wer(r, h)) if r.strip() else np.nan)
    return np.array(out)


def spearman(a, b):
    def rank(x):
        r = np.empty(len(x))
        r[np.argsort(x)] = np.arange(len(x))
        return r
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(rank(a[m]), rank(b[m]))[0, 1])


# ------------------------------------------------------------------- signals
def input_qparams():
    """(scale, zero_point) of the deployed streaming graph's feat quantizer."""
    import onnx
    from onnx import numpy_helper

    from sr_stream_ensemble import CKPT_DIR, TMP

    for p in (CKPT_DIR / "g_best_streaming_int8_minmax.onnx",
              TMP / "relu6deep_streaming_int8_minmax.onnx"):
        if p.exists():
            break
    m = onnx.load(p)
    inits = {t.name: t for t in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] == "feat":
            return (float(numpy_helper.to_array(inits[n.input[1]])),
                    int(numpy_helper.to_array(inits[n.input[2]])))
    raise RuntimeError("feat quantizer not found")


def utt_signals(idx, cond, s, zp):
    """(mean D_t, hot clip-rate, quiet mean|q-zp|) in the feat-quantizer domain.

    Uses the compressed-magnitude channel (feat[:,0], the one carrying level;
    GD/IFD are bounded). The device computes q for every frame anyway, so both
    flags are running means of quantities it already has.
    """
    z = np.load(CACHE / f"u{idx:04d}_{cond}.npz")
    D_t = z["ems"].var(axis=0).mean(axis=-1)
    q = z["mag"] / s + zp
    hot = float((q > 127).mean())
    quiet = float(np.abs(q - zp).mean())
    return float(D_t.mean()), hot, quiet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=824)
    ap.add_argument("--n_ood", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=24)
    args = ap.parse_args()

    from joblib import Parallel, delayed

    from common.dataset import load_voicebank_demand

    n = min(args.n, len(load_voicebank_demand()["test"]))
    n_ood = min(args.n_ood, n)
    counts = {c: (n if c == "id" else n_ood) for c in CONDS}

    # stacks (idempotent if sr_gating.py already ran)
    for cond in CONDS:
        Parallel(n_jobs=args.jobs)(
            delayed(collect_one)(i, cond) for i in range(counts[cond]))
    print("member stacks ready", flush=True)

    # signals + ID-frozen thresholds (feature-quantizer domain)
    qs, qz = input_qparams()
    sig = {cond: [utt_signals(i, cond, qs, qz) for i in range(counts[cond])]
           for cond in CONDS}
    D_id = np.array([s[0] for s in sig["id"]])
    hot_id = np.array([s[1] for s in sig["id"]])
    quiet_id = np.array([s[2] for s in sig["id"]])
    th = {"D_p90": float(np.percentile(D_id, 90)),
          "D_p95": float(np.percentile(D_id, 95)),
          "D_p99": float(np.percentile(D_id, 99)),
          "hot": float(np.percentile(hot_id, 99) + 1e-4),
          "quiet": float(np.percentile(quiet_id, 1))}
    print("thresholds (ID-frozen):", {k: f"{v:.3e}" for k, v in th.items()},
          flush=True)

    prev = json.load(open(OUT)) if OUT.exists() else {}
    asr = None
    res = {"n": counts, "thresholds": th, "feat_q": [qs, qz], "protocol":
           "differential WER, refs = whisper-tiny.en on clean (greedy)"}
    for cond in CONDS:
        m = counts[cond]
        t0 = time.time()
        cached = prev.get(cond, {}).get("per_utt", {})
        if len(cached.get("wer_noisy", [])) == m:
            w_n = np.array(cached["wer_noisy"])
            w_e = np.array(cached["wer_enh"])
            pooled = (prev[cond].get("wer_noisy_pooled"),
                      prev[cond].get("wer_enh_pooled"))
        else:
            if asr is None:
                asr = ASR()
            cleans, noisys = [], []
            for i in range(m):
                z = np.load(CACHE / f"u{i:04d}_{cond}.npz")
                cleans.append(z["clean"])
                noisys.append(z["noisy"])
            enhs = Parallel(n_jobs=args.jobs)(
                delayed(wav_for)(i, "none", {}, cond) for i in range(m))
            refs = asr.transcribe(cleans)
            hyp_n = asr.transcribe(noisys)
            hyp_e = asr.transcribe(enhs)
            w_n, w_e = per_utt_wer(refs, hyp_n), per_utt_wer(refs, hyp_e)
            pooled = (wer(refs, hyp_n), wer(refs, hyp_e))

        D = np.array([s[0] for s in sig[cond]])
        hot = np.array([s[1] for s in sig[cond]])
        quiet = np.array([s[2] for s in sig[cond]])
        level_flag = (hot > th["hot"]) | (quiet < th["quiet"])

        row = {"wer_noisy": float(np.nanmean(w_n)),
               "wer_enh": float(np.nanmean(w_e)),
               "wer_noisy_pooled": pooled[0], "wer_enh_pooled": pooled[1],
               "mean_D": float(D.mean()),
               "spearman_D_vs_damage": spearman(D, w_e - w_n),
               "per_utt": {"wer_noisy": w_n.tolist(), "wer_enh": w_e.tolist(),
                           "D": D.tolist(), "hot": hot.tolist(),
                           "quiet": quiet.tolist()},
               "policies": {}}
        oracle = np.where(w_e <= w_n, w_e, w_n)
        row["wer_oracle"] = float(np.nanmean(oracle))
        for pol, flag in [
                ("D_p90", D > th["D_p90"]), ("D_p95", D > th["D_p95"]),
                ("D_p99", D > th["D_p99"]), ("level", level_flag),
                ("D_p95+level", (D > th["D_p95"]) | level_flag)]:
            w_mix = np.where(flag, w_n, w_e)
            row["policies"][pol] = {"wer": float(np.nanmean(w_mix)),
                                    "bypass_rate": float(flag.mean())}
        res[cond] = row
        pol_str = "  ".join(f"{p}={r['wer']:.3f}@{r['bypass_rate']:.0%}"
                            for p, r in row["policies"].items())
        print(f"[{cond}] WER noisy {row['wer_noisy']:.3f} | enh "
              f"{row['wer_enh']:.3f} | oracle {row['wer_oracle']:.3f} | "
              f"{pol_str} | rho(D,damage)={row['spearman_D_vs_damage']:+.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
