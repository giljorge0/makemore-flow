"""
Train the conditioned flow matching model in 8D latent space.

Same conditional flow matching setup as train_flow_cond.py but using:
  - 8D latent space (from autoencoder_8d.pt)
  - Larger velocity field (VelocityField8D from flow_8d.py)
  - 100 epochs for thorough coverage of the richer space

Outputs:
  checkpoints/flow_cond_8d.pt
"""

import time
import torch
from train_flow_cond import length_to_cond, NULL_TOKEN, COND_NAMES
from flow_8d import VelocityField8D, sample_ode_8d, LATENT_DIM
from model import AutoEncoder, decode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROP_PROB = 0.15


def main():
    torch.manual_seed(0)

    data    = torch.load("checkpoints/latents_8d.pt", weights_only=False)
    z1_all  = data["z"].to(DEVICE)
    names   = data["names"]
    mean    = data["mean"].to(DEVICE)
    std     = data["std"].to(DEVICE)

    conds_all = torch.tensor(
        [length_to_cond(len(n)) for n in names], dtype=torch.long, device=DEVICE
    )
    z1_norm = (z1_all - mean) / std

    v_field = VelocityField8D().to(DEVICE)
    opt     = torch.optim.Adam(v_field.parameters(), lr=1e-3)

    N = z1_norm.shape[0]; BATCH = 1024; EPOCHS = 100

    for epoch in range(EPOCHS):
        t0   = time.time()
        perm = torch.randperm(N, device=DEVICE)
        total = 0; nb = 0

        for i in range(0, N, BATCH):
            idx  = perm[i : i + BATCH]
            z1   = z1_norm[idx]
            cond = conds_all[idx].clone()
            B    = z1.shape[0]

            cond[torch.rand(B, device=DEVICE) < DROP_PROB] = NULL_TOKEN

            z0       = torch.randn_like(z1)
            t        = torch.rand(B, 1, device=DEVICE)
            zt       = (1 - t) * z0 + t * z1
            target_v = z1 - z0

            loss = ((v_field(zt, t, cond) - target_v) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  loss {total/nb:.4f}  ({time.time()-t0:.2f}s)")

    torch.save({"state_dict": v_field.state_dict(), "latent_dim": LATENT_DIM},
               "checkpoints/flow_cond_8d.pt")
    print("saved checkpoints/flow_cond_8d.pt")

    # sanity check
    ae = AutoEncoder(latent_dim=LATENT_DIM)
    ae.load_state_dict(torch.load("checkpoints/autoencoder_8d.pt",
                                  weights_only=False)["state_dict"])
    ae.eval()

    print("\nSanity check (guidance_scale=2.0):")
    for ct, label in COND_NAMES.items():
        torch.manual_seed(0)
        z_norm = sample_ode_8d(v_field, 200, ct, guidance_scale=2.0, device=DEVICE)
        z      = z_norm * std + mean
        with torch.no_grad():
            toks = ae.decoder.sample(z.cpu(), greedy=False, temperature=0.9)
        decoded  = [decode(t.tolist()) for t in toks if decode(t.tolist())]
        avg_len  = sum(len(d) for d in decoded) / len(decoded)
        print(f"  {label:20s}  avg_len={avg_len:.1f}  examples: {decoded[:6]}")


if __name__ == "__main__":
    main()
