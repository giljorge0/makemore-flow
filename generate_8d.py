"""
Generate names from the 8D conditioned flow model.

Usage:
  python generate_8d.py --compare
  python generate_8d.py --length short --n 20
  python generate_8d.py --length long --guidance 3.0 --n 20

The 8D model has near-perfect reconstruction and much crisper length
conditioning than the 2D version — steering with w=2 already achieves
the target length distribution without degrading name quality.
"""

import argparse
import torch

from model import AutoEncoder, decode
from flow_8d import VelocityField8D, sample_ode_8d, LATENT_DIM
from train_flow_cond import COND_NAMES, NULL_TOKEN

DEVICE = "cpu"
DEFAULT_GUIDANCE = {1: 2.0, 2: 2.0, 3: 2.0}
LENGTH_MAP = {"short": 1, "medium": 2, "long": 3}


def load():
    data = torch.load("checkpoints/latents_8d.pt", weights_only=False)
    mean, std = data["mean"], data["std"]

    ae = AutoEncoder(latent_dim=LATENT_DIM)
    ae.load_state_dict(
        torch.load("checkpoints/autoencoder_8d.pt", weights_only=False)["state_dict"]
    )
    ae.eval()

    v = VelocityField8D()
    v.load_state_dict(
        torch.load("checkpoints/flow_cond_8d.pt", weights_only=False)["state_dict"]
    )
    v.eval()
    return ae, v, mean, std


def generate(ae, v, mean, std, cond_token, n, guidance, seed=42):
    torch.manual_seed(seed)
    z_norm = sample_ode_8d(v, n, cond_token, guidance_scale=guidance, n_steps=100)
    z      = z_norm * std + mean
    with torch.no_grad():
        toks = ae.decoder.sample(z, greedy=False, temperature=0.9)
    return [nm for nm in [decode(t.tolist()) for t in toks] if nm]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--length",   choices=["short", "medium", "long"], default=None)
    p.add_argument("--n",        type=int,   default=20)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--compare",  action="store_true")
    p.add_argument("--seed",     type=int,   default=42)
    args = p.parse_args()

    ae, v, mean, std = load()

    if args.compare or args.length is None:
        print("=== 8D conditioned generation (classifier-free guidance) ===\n")
        col_w = 22
        print(f"{'SHORT (≤4)':{col_w}} {'MEDIUM (5-6)':{col_w}} {'LONG (≥7)':{col_w}}")
        print("-" * (col_w * 3))
        cols = {}
        for lname, ct in LENGTH_MAP.items():
            w     = args.guidance or DEFAULT_GUIDANCE[ct]
            names = generate(ae, v, mean, std, ct, args.n, w, seed=args.seed)
            avg   = sum(len(nm) for nm in names) / max(len(names), 1)
            cols[lname] = (names, avg, w)
        for i in range(args.n):
            row = ""
            for lname in ["short", "medium", "long"]:
                nm = cols[lname][0][i] if i < len(cols[lname][0]) else ""
                row += f"{nm:{col_w}}"
            print(row)
        print()
        for lname, ct in LENGTH_MAP.items():
            nms, avg, w = cols[lname]
            print(f"  {lname:6s}  guidance={w}  avg_len={avg:.1f}")
    else:
        ct    = LENGTH_MAP[args.length]
        w     = args.guidance or DEFAULT_GUIDANCE[ct]
        names = generate(ae, v, mean, std, ct, args.n, w, seed=args.seed)
        avg   = sum(len(nm) for nm in names) / max(len(names), 1)
        print(f"8D conditioned generation  length={args.length}  guidance={w}\n")
        for nm in names:
            print(f"  {nm}")
        print(f"\navg length: {avg:.1f}")


if __name__ == "__main__":
    main()
