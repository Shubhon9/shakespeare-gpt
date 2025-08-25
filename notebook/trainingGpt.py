# ====================================
# 1. Setup
# ====================================
# Import required libraries
# torch      -> deep learning framework
# nn, F      -> layers & functional API
# requests   -> fetch Shakespeare dataset from web
# tqdm       -> show progress bar
# dataclass  -> cleaner config object for GPT
import torch, torch.nn as nn, torch.nn.functional as F, requests
from dataclasses import dataclass
from tqdm import tqdm

# Pick device: "cuda" (GPU) if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ====================================
# 2. Dataset
# ====================================
# We use the Tiny Shakespeare dataset (~1MB text).
# This dataset is great for testing small language models.

# Download raw text
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text

# Build vocabulary: unique characters in text
chars = sorted(set(text))
stoi, itos = {ch:i for i,ch in enumerate(chars)}, {i:ch for i,ch in enumerate(chars)}

# Functions to encode (string -> int list) and decode (int list -> string)
def encode(s): return [stoi[c] for c in s]
def decode(t): return "".join([itos[i] for i in t])

# Convert entire dataset into tensor of ints
data = torch.tensor(encode(text), dtype=torch.long)

# Train/validation split (90% train, 10% val)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# Define context length & batch size
vocab_size, block_size, batch_size = len(chars), 128, 32
print("Dataset length:", len(text))
print("Vocab size:", vocab_size)

# Function to grab a random batch of sequences
def get_batch(split):
    """
    Returns (x, y) batch tensors:
    - x is input sequence (context of block_size)
    - y is the target (same sequence shifted by 1)
    """
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d)-block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# ====================================
# 3. Model
# ====================================
# GPTConfig controls model size & training complexity.
# You can use the "big" one (slower, better results)
# OR the "tiny" one (fast training, weaker results).

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    d_model: int = 256   # embedding dimension
    n_heads: int = 4     # number of attention heads
    n_layers: int = 4    # number of transformer blocks
    d_ff: int = 1024     # feedforward hidden size
    dropout: float = 0.1

# 💡 QUICKER TRAINING CONFIG (comment out above, use this instead)
# @dataclass
# class GPTConfig:
#     vocab_size: int
#     block_size: int
#     d_model: int = 128   # smaller embedding dim
#     n_heads: int = 2     # fewer heads
#     n_layers: int = 2    # fewer transformer blocks
#     d_ff: int = 256      # smaller feedforward size
#     dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """
    Implements masked multi-head self-attention.
    Each token attends only to previous tokens (causal mask).
    """
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        # Query, Key, Value projections
        self.key, self.query, self.value = [nn.Linear(cfg.d_model, cfg.d_model) for _ in range(3)]
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.n_heads, self.dropout = cfg.n_heads, nn.Dropout(cfg.dropout)
        # Register a lower-triangular mask (T x T), prevents looking ahead
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x):
        B,T,C = x.size()
        # Project and reshape into multiple heads
        k = self.key(x).view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
        q = self.query(x).view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
        v = self.value(x).view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)

        # Compute attention scores (scaled dot-product)
        att = (q @ k.transpose(-2,-1)) / (C//self.n_heads)**0.5
        # Apply mask so tokens can't attend to future
        att = att.masked_fill(self.mask[:T,:T]==0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Weighted sum of values
        y = att @ v
        # Reshape back and project
        return self.proj(y.transpose(1,2).contiguous().view(B,T,C))


class Block(nn.Module):
    """
    Transformer block = LayerNorm + Attention + FeedForward + Residual connections
    """
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn, self.ff = CausalSelfAttention(cfg), nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.ReLU(),
            nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout)
        )
    def forward(self, x):
        # Apply attention, then feedforward, with residuals
        return x + self.ff(self.ln2(x + self.attn(self.ln1(x))))


class GPT(nn.Module):
    """
    GPT Model = token embeddings + positional embeddings
              + stacked transformer blocks + final linear head
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb, self.pos_emb = nn.Embedding(cfg.vocab_size, cfg.d_model), nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln, self.head = nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B,T = idx.size()
        # Embed tokens + positions
        tok, pos = self.token_emb(idx), self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(tok + pos)
        x = self.ln(x)
        logits = self.head(x)

        # Compute loss only if targets are provided
        loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss


# ====================================
# 4. Training Loop with tqdm
# ====================================

cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size)
model = GPT(cfg).to(device)
print("Model params:", sum(p.numel() for p in model.parameters())/1e6, "M")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 🔑 For quick tests: set max_iters=1000 (instead of 5000)
max_iters, eval_interval = 1000, 200

progress = tqdm(range(max_iters+1), desc="Training")
for step in progress:
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    # Evaluate on validation set occasionally
    if step % eval_interval == 0:
        with torch.no_grad():
            xb_val, yb_val = get_batch("val")
            _, vloss = model(xb_val, yb_val)
        progress.set_postfix({"train_loss": loss.item(), "val_loss": vloss.item()})

# Save trained weights
torch.save(model.state_dict(), "gpt_shakespeare_ds.pt")
print("✅ Model saved as gpt_shakespeare_ds.pt")
