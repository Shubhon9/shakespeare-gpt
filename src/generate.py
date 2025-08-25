import torch
from dataset import load_dataset
from model import GPT, GPTConfig

def generate_text(prompt, max_new=200, top_k=50, top_p=0.9, temp=0.8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Reload dataset (only for tokenizer)
    _, _, vocab_size, stoi, itos, encode, decode = load_dataset()
    block_size = 128

    # Load model
    cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size)
    model = GPT(cfg).to(device)
    model.load_state_dict(torch.load("checkpoints/gpt_shakespeare.pt", map_location=device))
    model.eval()
    print("✅ Model loaded")

    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]

    for _ in range(max_new):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temp
        probs = torch.softmax(logits, dim=-1)

        # Top-k filtering
        if top_k is not None:
            topk_vals, _ = torch.topk(probs, top_k)
            probs[probs < topk_vals[:, [-1]]] = 0.0

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            sorted_probs[mask] = 0.0
            probs.zero_().scatter_(1, sorted_idx, sorted_probs)

        probs = probs.clamp(min=1e-9)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        idx_next = torch.multinomial(probs, 1)
        idx = torch.cat([idx, idx_next], dim=1)

    return decode(idx[0].tolist())

if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt (or 'quit' to exit): ")
        if prompt.lower() == "quit":
            break
        print("\n=== Model Output ===")
        print(generate_text(prompt, max_new=300))
        print("\n")
