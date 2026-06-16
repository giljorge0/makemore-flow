"""
Char-level sequence autoencoder.

Encoder: name (sequence of chars) -> single continuous latent vector z in R^latent_dim
Decoder: z -> autoregressively reconstructs the name, char by char.

This gives us a *continuous* space in which every name lives as a point.
That space is what we'll later learn a flow-matching velocity field over.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Vocabulary -------------------------------------------------------
# index 0 = '.' used as both BOS and EOS
CHARS = list("abcdefghijklmnopqrstuvwxyz")
ITOS = {0: "."}
for i, c in enumerate(CHARS):
    ITOS[i + 1] = c
STOI = {c: i for i, c in ITOS.items()}
VOCAB_SIZE = len(ITOS)  # 27
MAX_LEN = 17  # BOS + up to 15 chars + EOS


def encode(name: str) -> list[int]:
    return [STOI["."]] + [STOI[c] for c in name] + [STOI["."]]


def decode(ids: list[int]) -> str:
    out = []
    for i in ids:
        if i == STOI["."]:
            if out:  # stop at first EOS after we've started
                break
            else:
                continue  # skip leading BOS
        out.append(ITOS[i])
    return "".join(out)


# --- Model --------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=32, hidden=128, latent_dim=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent_dim)

    def forward(self, x, lengths):
        # x: (B, T) token ids (includes BOS/EOS, padded with 0)
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        z = self.to_latent(h.squeeze(0))
        return z


class Decoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=32, hidden=128, latent_dim=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.z_to_hidden = nn.Linear(latent_dim, hidden)
        self.rnn = nn.GRU(embed_dim + latent_dim, hidden, batch_first=True)
        self.to_logits = nn.Linear(hidden, vocab_size)
        self.latent_dim = latent_dim

    def forward(self, z, x_in):
        # z: (B, latent_dim), x_in: (B, T) teacher-forced input tokens (shifted)
        B, T = x_in.shape
        h0 = torch.tanh(self.z_to_hidden(z)).unsqueeze(0)  # (1, B, hidden)
        emb = self.embed(x_in)  # (B, T, E)
        z_rep = z.unsqueeze(1).expand(-1, T, -1)  # condition on z at every step
        rnn_in = torch.cat([emb, z_rep], dim=-1)
        out, _ = self.rnn(rnn_in, h0)
        logits = self.to_logits(out)
        return logits

    @torch.no_grad()
    def sample(self, z, max_len=MAX_LEN, greedy=False, temperature=1.0):
        # z: (B, latent_dim)
        B = z.shape[0]
        device = z.device
        h = torch.tanh(self.z_to_hidden(z)).unsqueeze(0)
        tok = torch.full((B, 1), STOI["."], dtype=torch.long, device=device)
        out_tokens = []
        for _ in range(max_len):
            emb = self.embed(tok)
            rnn_in = torch.cat([emb, z.unsqueeze(1)], dim=-1)
            o, h = self.rnn(rnn_in, h)
            logits = self.to_logits(o[:, -1, :]) / temperature
            if greedy:
                tok = logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                tok = torch.multinomial(probs, 1)
            out_tokens.append(tok)
        return torch.cat(out_tokens, dim=1)  # (B, max_len)


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=2, embed_dim=32, hidden=128):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, embed_dim=embed_dim, hidden=hidden)
        self.decoder = Decoder(latent_dim=latent_dim, embed_dim=embed_dim, hidden=hidden)
        self.latent_dim = latent_dim

    def forward(self, x, lengths):
        z = self.encoder(x, lengths)
        x_in = x[:, :-1]
        logits = self.decoder(z, x_in)
        return logits, z
