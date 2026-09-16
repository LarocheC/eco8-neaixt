"""Training entry point for N6Net.

Reuses convfsenet/train.py wholesale — the time-domain objective, the PESQ
metric-GAN, the checkpoint/resume conventions and the g_best selection are all
architecture-agnostic — and swaps only the model builder. Same CLI surface, so
the existing sweep drivers work untouched.
"""

from __future__ import annotations

import argparse
import json

from common.env import AttrDict, build_env
from convfsenet.train import train
from n6net.model import build_causal_model, cost_summary


def main():
    print("Initializing N6Net Training Process..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/n6net_b3.json")
    parser.add_argument("--checkpoint_path", default="cp_n6net")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--training_epochs", default=200, type=int)
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=2000, type=int)
    parser.add_argument("--summary_interval", default=50, type=int)
    parser.add_argument("--validation_interval", default=2000, type=int)
    parser.add_argument("--best_checkpoint_start_epoch", default=5, type=int)
    parser.add_argument("--init_from", default=None)
    a = parser.parse_args()

    with open(a.config) as f:
        h = AttrDict(json.load(f))
    build_env(a.config, "config.json", a.checkpoint_path)

    # The architecture's whole claim is about cost shape, so state it up front.
    c = cost_summary(build_causal_model(h))
    print(f"N6Net: params={c['params']:,}  MAC/frame={c['macs_per_frame']:,}  "
          f"int8 weights={c['weight_bytes_int8']/1024:.0f} KiB  "
          f"arithmetic intensity={c['arithmetic_intensity']:.0f} MAC/B  "
          f"receptive field={c['receptive_field_frames']} frames  "
          f"FIFO state={c['fifo_bytes_int8']/1024:.0f} KiB")

    train(a, h, model_builder=build_causal_model)


if __name__ == "__main__":
    main()
