import torch
from tqdm import tqdm
from dataset import load_dataset, get_batch
from model import GPT, GPTConfig

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load dataset
    train_data, val_data, vocab_size, stoi, itos, encode, decode = load_dataset()
    block_size, batch_size = 128, 32

    # Init model
    cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size)
    model = GPT(cfg).to(device)
    print("Model params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    max_iters, eval_interval = 1000, 500
    # max_iters, eval_interval = 5000, 500

    progress = tqdm(range(max_iters + 1), desc="Training")
    for step in progress:
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            with torch.no_grad():
                xb_val, yb_val = get_batch(val_data, block_size, batch_size, device)
                _, vloss = model(xb_val, yb_val)
            progress.set_postfix({"train_loss": loss.item(), "val_loss": vloss.item()})

    # Save model
    torch.save(model.state_dict(), "checkpoints/gpt_shakespeare.pt")
    print("✅ Model saved to checkpoints/gpt_shakespeare.pt")

if __name__ == "__main__":
    train()
