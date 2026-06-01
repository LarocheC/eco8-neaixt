"""Shared PESQ evaluation helpers.

Used by both model families' training / evaluation paths. Previously these
lived in the NSNet2 model file; they were moved here so the ConvFSENet
package never has to import from the NSNet2 package.
"""
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed


def pesq_score(utts_r, utts_g, h):
    scores = Parallel(n_jobs=30)(delayed(eval_pesq)(
        utts_r[i].squeeze().cpu().numpy(),
        utts_g[i].squeeze().cpu().numpy(),
        h.sampling_rate,
    ) for i in range(len(utts_r)))
    # eval_pesq returns -1 on failure (silent/too-short/NaN utterance). Drop
    # those rather than averaging the sentinel into the mean, which would
    # corrupt the validation score used to select g_best.
    valid = [s for s in scores if s != -1]
    return np.mean(valid) if valid else np.float64("nan")


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        return pesq(sr, clean_utt, esti_utt)
    except Exception:
        return -1
