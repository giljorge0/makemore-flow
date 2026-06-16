"""
Generate new names:
  1. sample z0 ~ N(0, I)               (pure noise, t=0)
  2. integrate dz/dt = v_theta(z, t)   from t=0 to t=1  (the learned "evolution")
  3. un-normalize the resulting z1     -> latent point in name-space
  4. decode z1 -> characters

This is the "make more" loop: instead of directly sampling a discrete next
character, we sample a point in latent space and *evolve* it along a learned
path until it represents a name, then read the name off.
"""

import sys
import torch

from model import AutoEncoder, decode
from train_flow import VelocityField, sample_ode, LATENT_DIM


def load_models(device="cpu"):
    data = torch.load("checkpoints/latents.pt")
    mean, std = data["mean"].to(device), data["std"].to(device)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(device)
    ae.load_state_dict(torch.load("checkpoints/autoencoder.pt", map_location=device))
    ae.eval()

    v_field = VelocityField().to(device)
    v_field.load_state_dict(torch.load("checkpoints/flow.pt", map_location=device))
    v_field.eval()

    return ae, v_field, mean, std


def main(n=20, n_steps=100, temperature=1.0, seed=None):
    device = "cpu"
    if seed is not None:
        torch.manual_seed(seed)
    ae, v_field, mean, std = load_models(device)

    z_norm = sample_ode(v_field, n_samples=n, n_steps=n_steps, device=device)
    z = z_norm * std + mean  # back into the autoencoder's latent space

    with torch.no_grad():
        toks = ae.decoder.sample(z, greedy=False, temperature=temperature)

    for i in range(n):
        name = decode(toks[i].tolist())
        zz = z[i].numpy().round(2)
        print(f"{name:15s}  z={zz}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n=n)
