"""
Generate names conditioned on length, using classifier-free guidance.

Usage:
  python generate_cond.py --length short --n 20
  python generate_cond.py --length medium --n 20 --guidance 3.0
  python generate_cond.py --length long --n 20 --guidance 4.0
  python generate_cond.py --compare          # show all three side-by-side

Length buckets:
  short   : names with <= 4 characters
  medium  : 5-6 characters
  long    : >= 7 characters

How it works:
  The velocity field v(z, t, c) is trained with a condition token c in
  {null, short, medium, long}. At inference we run:

      v_guided = v_uncond + w * (v_cond - v_uncond)

  where w is --guidance (w=0 ignores the condition, w>1 amplifies it).
  Starting from Gaussian noise, the ODE integrator follows this guided
  field from t=0 to t=1, landing in a region of name-space that matches
  the requested length condition.

  This is "steering the evolution" — you pick a direction and the path
  tilts that way at every step.
"""

import argparse
import torch

from model import AutoEncoder, decode
from train_flow_cond import (
    CondVelocityField, sample_ode_cfg, LATENT_DIM, COND_NAMES,
    NULL_TOKEN,
)

DEVICE = "cpu"
DEFAULT_GUIDANCE = {1: 2.5, 2: 3.0, 3: 4.0}
LENGTH_MAP = {"short": 1, "medium": 2, "long": 3}


def load():
    data = torch.load("checkpoints/latents.pt", weights_only=False)
    mean, std = data["mean"], data["std"]

    ae = AutoEncoder(latent_dim=LATENT_DIM)
    ae.load_state_dict(torch.load("checkpoints/autoencoder.pt", map_location=DEVICE, weights_only=False))
    ae.eval()

    v = CondVelocityField()
    v.load_state_dict(torch.load("checkpoints/flow_cond.pt", map_location=DEVICE, weights_only=False))
    v.eval()

    return ae, v, mean, std


def generate(ae, v, mean, std, cond_token, n, guidance, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    z_norm = sample_ode_cfg(v, n, cond_token, guidance_scale=guidance, n_steps=100)
    z = z_norm * std + mean
    with torch.no_grad():
        toks = ae.decoder.sample(z, greedy=False, temperature=0.9)
    names = [decode(t.tolist()) for t in toks]
    return [nm for nm in names if nm]  # drop empty


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--length", choices=["short", "medium", "long"], default=None)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--compare", action="store_true",
                   help="Show all three conditions side by side")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ae, v, mean, std = load()

    if args.compare or args.length is None:
        print("=== Conditioned generation (classifier-free guidance) ===\n")
        col_w = 22
        header = f"{'SHORT (≤4)':{col_w}} {'MEDIUM (5-6)':{col_w}} {'LONG (≥7)':{col_w}}"
        print(header)
        print("-" * len(header))

        cols = {}
        for lname, ct in LENGTH_MAP.items():
            w = args.guidance or DEFAULT_GUIDANCE[ct]
            names = generate(ae, v, mean, std, ct, args.n, w, seed=args.seed)
            avg = sum(len(nm) for nm in names) / max(len(names), 1)
            cols[lname] = (names, avg, w)

        n_rows = args.n
        for i in range(n_rows):
            row = ""
            for lname in ["short", "medium", "long"]:
                nms, _, _ = cols[lname]
                nm = nms[i] if i < len(nms) else ""
                row += f"{nm:{col_w}}"
            print(row)

        print()
        for lname, ct in LENGTH_MAP.items():
            nms, avg, w = cols[lname]
            print(f"  {lname:6s}  guidance={w}  avg_len={avg:.1f}")

    else:
        ct = LENGTH_MAP[args.length]
        w  = args.guidance or DEFAULT_GUIDANCE[ct]
        names = generate(ae, v, mean, std, ct, args.n, w, seed=args.seed)
        avg = sum(len(nm) for nm in names) / max(len(names), 1)

        print(f"Conditioned generation  length={args.length}  guidance={w}\n")
        for nm in names:
            print(f"  {nm}")
        print(f"\navg length: {avg:.1f}")


if __name__ == "__main__":
    main()
