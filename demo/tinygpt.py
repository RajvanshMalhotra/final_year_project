"""
A tiny character-level GPT, its data, and the gradient helpers the nodes use.

Every node must build exactly the same batches and the same model, so:
  - the vocabulary is fixed (printable ASCII + newline), not learned from the text;
  - batch_for_step() draws from a torch generator seeded by (seed, step).
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEXT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CHARS = "\n" + "".join(chr(c) for c in range(32, 127))
VOCAB_SIZE = len(CHARS)
_INDEX = {c: i for i, c in enumerate(CHARS)}
_UNKNOWN = _INDEX["?"]


def load_text() -> str:
    """Tiny Shakespeare (1.1 MB, cached in demo/data/); falls back to the bundled sonnets offline."""
    path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(path):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            urllib.request.urlretrieve(TEXT_URL, path + ".part")
            os.replace(path + ".part", path)
        except OSError:
            path = os.path.join(DATA_DIR, "fallback.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def encode(text: str) -> torch.Tensor:
    return torch.tensor([_INDEX.get(c, _UNKNOWN) for c in text], dtype=torch.long)


def decode(ids: torch.Tensor) -> str:
    return "".join(CHARS[int(i)] for i in ids)


def batch_for_step(data: torch.Tensor, step: int, B: int, ctx: int,
                   seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """The global batch of step `step`: B windows of ctx+1 characters, identical on every node."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + step)
    starts = torch.randint(0, len(data) - ctx - 1, (B,), generator=g)
    rows = torch.stack([data[s:s + ctx + 1] for s in starts.tolist()]) if B else \
        torch.empty((0, ctx + 1), dtype=torch.long)
    return rows[:, :-1], rows[:, 1:]


class Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """About 0.8M parameters with the defaults."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d: int = 128, layers: int = 4,
                 heads: int = 4, ctx: int = 64) -> None:
        super().__init__()
        self.ctx = ctx
        self.tok, self.pos = nn.Embedding(vocab_size, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def loss_fn(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def accumulate_grads(model: nn.Module, x: torch.Tensor, y: torch.Tensor, cap: int) -> Optional[float]:
    """
    Gradient of the mean loss over all rows of x, computed in micro-batches of at most
    `cap` rows (gradient accumulation: what a GPU does when the slice doesn't fit in
    memory). Grads are zeroed first. Returns the mean loss, or None for an empty slice.
    """
    model.zero_grad(set_to_none=False)
    n = x.shape[0]
    if n == 0:
        for p in model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        return None
    total = 0.0
    for i in range(0, n, max(1, cap)):
        xb, yb = x[i:i + cap], y[i:i + cap]
        loss = loss_fn(model, xb, yb) * (xb.shape[0] / n)
        loss.backward()
        total += float(loss)
    return total


def flat_grads(model: nn.Module) -> torch.Tensor:
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()])


def set_flat_grads(model: nn.Module, flat: torch.Tensor) -> None:
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat[i:i + n].view_as(p).clone()
        i += n


def checksum(model: nn.Module) -> str:
    """First 12 hex digits of sha1 over all float32 parameters: equal => identical weights."""
    h = hashlib.sha1()
    for p in model.parameters():
        h.update(p.detach().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()[:12]
