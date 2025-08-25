import torch, requests

def load_dataset(url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", split=0.9):
    """Download Tiny Shakespeare dataset and split into train/val sets."""
    text = requests.get(url).text
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s): return [stoi[c] for c in s]
    def decode(t): return "".join([itos[i] for i in t])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(split * len(data))
    train_data, val_data = data[:n], data[n:]

    vocab_size = len(chars)
    return train_data, val_data, vocab_size, stoi, itos, encode, decode

def get_batch(data, block_size, batch_size, device):
    """Return a random batch of sequences from data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
