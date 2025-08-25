# ====================================
# 1. Reload Setup
# ====================================
import torch, torch.nn as nn, torch.nn.functional as F, requests
from dataclasses import dataclass

# Pick device automatically:
# - CUDA (GPU) if available for speed
# - CPU fallback if GPU not present
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------------------------
# Dataset (used here only to recreate tokenizer)
# ------------------------------------
# We don’t need the dataset for training anymore,
# but we need it to reconstruct the character-level vocabulary
# so we can map between text (chars) and token ids.
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text

# Build vocabulary mapping
chars = sorted(set(text))
stoi, itos = {ch: i for i, ch in enumerate(chars)}, {i: ch for i, ch in enumerate(chars)}

# Encode/decode functions:
# encode: text -> list of ints
# decode: list of ints -> text
def encode(s): return [stoi[c] for c in s]
def decode(t): return "".join([itos[i] for i in t])

# Vocab size (number of unique chars) and block size (context length)
vocab_size, block_size = len(chars), 128


# ====================================
# 2. Same Config & Model
# ====================================
@dataclass
class GPTConfig:
    """
    Config object that controls model size.
    Trade-offs:
    - Larger values = better text quality, but slower + needs more memory
    - Smaller values = faster training/inference, but weaker generations
    """
    vocab_size: int
    block_size: int
    d_model: int = 256   # embedding dimension per token
    n_heads: int = 4     # number of self-attention heads
    n_layers: int = 4    # number of transformer blocks
    d_ff: int = 1024     # size of feedforward hidden layer
    dropout: float = 0.1 # dropout for regularization

# ⚠️ Important:
# The architecture must EXACTLY match what was used in training,
# otherwise loading the checkpoint will fail.
# (Re-use same CausalSelfAttention, Block, GPT classes from training file)


# Load model with config and checkpoint
cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size)
model = GPT(cfg).to(device)
model.load_state_dict(torch.load("gpt_shakespeare_ds.pt", map_location=device))
model.eval()
print("✅ Model loaded")


# ====================================
# 3. Generation
# ====================================
def generate(prompt, max_new=200, top_k=50, top_p=0.9, temp=0.8):
    """
    Generate text from a prompt using nucleus/top-k sampling.
    
    Args:
      prompt (str): Input text to condition on
      max_new (int): Max new tokens to generate
      top_k (int): Keep only top-k most likely tokens
      top_p (float): Nucleus sampling threshold (keep minimal set with sum prob >= top_p)
      temp (float): Softens or sharpens distribution (lower = more deterministic)
    """
    # Encode prompt -> tensor of shape (1, T)
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]

    for _ in range(max_new):
        # Get last block_size tokens as context
        idx_cond = idx[:, -block_size:]

        # Forward pass (no targets, just logits)
        logits, _ = model(idx_cond)

        # Focus only on last token's logits
        logits = logits[:, -1, :] / temp

        # Convert to probability distribution
        probs = torch.softmax(logits, dim=-1)

        # --------------------------
        # Top-K filtering
        # --------------------------
        if top_k is not None:
            topk_vals, _ = torch.topk(probs, top_k)
            probs[probs < topk_vals[:, [-1]]] = 0.0

        # --------------------------
        # Top-P (nucleus) filtering
        # --------------------------
        if top_p is not None:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs > top_p
            # shift mask right to keep first prob above threshold
            mask[:, 1:] = mask[:, :-1].clone()
            sorted_probs[mask] = 0.0
            probs.zero_().scatter_(1, sorted_idx, sorted_probs)

        # Normalize again to valid distribution
        probs = probs.clamp(min=1e-9)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample next token
        idx_next = torch.multinomial(probs, 1)

        # Append sampled token to sequence
        idx = torch.cat([idx, idx_next], dim=1)

    # Decode token ids -> text
    return decode(idx[0].tolist())


# ====================================
# 4. Interactive Loop
# ====================================
# Simple REPL loop to keep generating text until user quits
while True:
    prompt = input("Enter a prompt (or 'quit' to exit): ")
    if prompt.lower() == "quit":
        break
    print("\n=== Model Output ===")
    print(generate(prompt, max_new=300))
    print("\n")
