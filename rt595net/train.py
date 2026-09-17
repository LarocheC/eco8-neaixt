"""Training entry point for RT595Net.

Reuses convfsenet/train.py wholesale — objective, PESQ metric-GAN, checkpoint and
g_best conventions are architecture-agnostic — and swaps only the model builder,
exactly as n6net/train.py does. Same CLI, so the sweep drivers work untouched.
"""

from __future__ import annotations

import argparse
import json

from common.env import AttrDict, build_env
from convfsenet.train import train
from rt595net.model import HIFI4_BUDGET_CYCLES, build_causal_model, cost_summary


def main():
    print("Initializing RT595Net Training Process..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rt595b_fb.json")
    parser.add_argument("--checkpoint_path", default="cp_rt595b_fb")
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

    # The architecture is a claim about the target's cost law; state it up front.
    c = cost_summary(build_causal_model(h))
    print(f"RT595Net: params={c['params']:,}  MAC/frame={c['macs_per_frame']:,}  "
          f"int8 weights={c['weight_bytes_int8']/1024:.0f} KiB  "
          f"predicted HiFi4={c['hifi4_cycles']:,.0f} cyc/frame "
          f"({c['hifi4_budget_fraction']:.2f}x of {HIFI4_BUDGET_CYCLES:,})  "
          f"receptive field={c['receptive_field_frames']} frames  "
          f"frequency reach={c['frequency_reach_bins']}/{c['params'] and 256} bins  "
          f"FIFO={c['fifo_bytes_int8']/1024:.0f} KiB")

    train(a, h, model_builder=build_causal_model)


if __name__ == "__main__":
    main()
