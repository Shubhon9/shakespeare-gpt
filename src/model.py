import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# @dataclass
# class GPTConfig:
#     vocab_size: int
#     block_size: int
#     d_model: int = 256
#     n_heads: int = 4
#     n_layers: int = 4
#     d_ff: int = 1024
#     dropout: float = 0.1
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    d_model: int = 128
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.key, self.query, self.value = [nn.Linear(cfg.d_model, cfg.d_model) for _ in range(3)]
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.n_heads, self.dropout = cfg.n_heads, nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (C // self.n_heads) ** 0.5
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.ReLU(),
            nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout)
        )

    def forward(self, x):
        return x + self.ff(self.ln2(x + self.attn(self.ln1(x))))

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks    = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln        = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok, pos = self.token_emb(idx), self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(tok + pos)
        x = self.ln(x)
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
