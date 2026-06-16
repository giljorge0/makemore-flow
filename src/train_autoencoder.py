"""
Train the char-level autoencoder on names.txt.

This produces:
  - checkpoints/autoencoder.pt          (encoder+decoder weights)
  - checkpoints/latents.pt              (z for every name in the dataset, plus stats)

The resulting latent space (default 2D) is the "concept space" we'll later
learn a flow-matching velocity field over.
"""

import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import AutoEncoder, encode, decode, STOI, MAX_LEN, VOCAB_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 2


class NameDataset(Dataset):
    def __init__(self, path="data/names.txt"):
        self.names = [l.strip() for l in open(path) if l.strip()]
        self.encoded = [encode(n) for n in self.names]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        ids = self.encoded[idx]
        return torch.tensor(ids, dtype=torch.long), len(ids)


def collate(batch):
    seqs, lens = zip(*batch)
    lens = torch.tensor(lens, dtype=torch.long)
    maxlen = max(lens).item()
    padded = torch.zeros(len(seqs), maxlen, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return padded, lens


def main():
    torch.manual_seed(0)
    ds = NameDataset()
    dl = DataLoader(ds, batch_size=256, shuffle=True, collate_fn=collate)

    model = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    EPOCHS = 25
    for epoch in range(EPOCHS):
        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        for x, lens in dl:
            x, lens = x.to(DEVICE), lens.to(DEVICE)
            logits, z = model(x, lens)
            target = x[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), target.reshape(-1), ignore_index=-1
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch:2d}  loss {total_loss / n_batches:.4f}  ({time.time()-t0:.1f}s)")

    # --- quick qualitative check: reconstruct a few names ---
    model.eval()
    sample_names = ["emma", "olivia", "william", "sophia", "liam"]
    with torch.no_grad():
        for name in sample_names:
            ids = torch.tensor([encode(name)], dtype=torch.long, device=DEVICE)
            lens = torch.tensor([len(ids[0])], device=DEVICE)
            z = model.encoder(ids, lens)
            recon = model.decoder.sample(z, greedy=True)
            print(f"{name:10s} -> z={z.cpu().numpy().round(2)}  recon={decode(recon[0].tolist())}")

    # --- compute latents for the whole dataset (for flow matching + viz) ---
    print("computing latents for full dataset...")
    all_z = []
    with torch.no_grad():
        dl_full = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate)
        for x, lens in dl_full:
            x, lens = x.to(DEVICE), lens.to(DEVICE)
            z = model.encoder(x, lens)
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0)
    mean = all_z.mean(0)
    std = all_z.std(0)
    print("latent mean:", mean.numpy(), "std:", std.numpy())

    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/autoencoder.pt")
    torch.save(
        {"z": all_z, "names": ds.names, "mean": mean, "std": std, "latent_dim": LATENT_DIM},
        "checkpoints/latents.pt",
    )
    print("saved checkpoints/autoencoder.pt and checkpoints/latents.pt")


if __name__ == "__main__":
    main()
