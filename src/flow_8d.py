"""
Velocity field for the 8-dimensional latent space.
Larger hidden size and condition embedding to match the richer latent.
"""

import torch
import torch.nn as nn
from train_flow_cond import NUM_CONDITIONS, NULL_TOKEN

LATENT_DIM = 8


class VelocityField8D(nn.Module):
    def __init__(self, hidden=256, n_layers=6, cond_embed_dim=32):
        super().__init__()
        self.cond_embed = nn.Embedding(NUM_CONDITIONS, cond_embed_dim)
        in_dim = LATENT_DIM + 1 + cond_embed_dim
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, LATENT_DIM)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, t, cond):
        return self.net(torch.cat([z, t, self.cond_embed(cond)], dim=-1))


@torch.no_grad()
def sample_ode_8d(v_field, n_samples, cond_token, guidance_scale=2.0,
                  n_steps=100, device="cpu", return_traj=False):
    z = torch.randn(n_samples, LATENT_DIM, device=device)
    dt = 1.0 / n_steps
    cond   = torch.full((n_samples,), cond_token, dtype=torch.long, device=device)
    uncond = torch.full((n_samples,), NULL_TOKEN,  dtype=torch.long, device=device)
    traj = [z.clone()] if return_traj else None

    for step in range(n_steps):
        t = torch.full((n_samples, 1), step * dt, device=device)
        v_c = v_field(z, t, cond)
        v_u = v_field(z, t, uncond)
        z = z + (v_u + guidance_scale * (v_c - v_u)) * dt
        if return_traj:
            traj.append(z.clone())

    if return_traj:
        return z, torch.stack(traj, dim=0)
    return z
