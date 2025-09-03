import torch
import numpy as np
from model import GPT, GPTConfig
from dataset import load_dataset, get_batch

train_data, val_data, vocab_size, stoi, itos, encode, decode = load_dataset()
block_size, batch_size = 128, 32

# Init model
cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size)

model = GPT(cfg)   # use same GPTConfig + GPT class you trained
model.load_state_dict(torch.load(r"checkpoints/gpt_shakespeare.pt", map_location="cpu"))
model.eval()
embeddings = model.token_emb.weight.detach().cpu().numpy()
print(embeddings.shape)   # (vocab_size, d_model)

# Save embeddings
np.savetxt("embeddings.tsv", embeddings, delimiter="\t")

# Save metadata (chars corresponding to each row)
with open("metadata.tsv", "w") as f:
    for i in range(len(itos)):
        f.write(f"{itos[i]}\n")
