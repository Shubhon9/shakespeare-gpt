# 📝 Mini-GPT: Train & Generate on Tiny Shakespeare

This repo contains two scripts to **train** and **generate text** using a GPT-style model on the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

- `src/trainingGpt.py` → Train the model and save weights  
- `src/generateGpt.py` → Load saved weights and interactively generate text  

---

## 🚀 Run in Google Colab

### 1. Setup Files
1. Upload both files (`trainingGpt.py`, `generateGpt.py`) to your Colab workspace
2. Create a `src` folder and place the files inside
3. Alternatively, use `!wget` or `!git clone` to get the files directly

### 2. Install Dependencies
```bash
!pip install torch tqdm requests
```

### 3. Training Options

#### **Short Training (Fast Demo)**
For a quick demo that runs in **~1–2 minutes** on Colab GPU, edit `trainingGpt.py` with these settings:

```python
# Quick demo settings
max_iters = 1000
cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                d_model=128, n_heads=2, n_layers=2, d_ff=256)
```

#### **Long Training (Better Quality)**
For better results that take **~10–15 minutes** on Colab GPU, use the default settings:

```python
# Better quality settings (default)
max_iters = 5000
cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                d_model=256, n_heads=4, n_layers=4, d_ff=1024)
```

### 4. Run Training
```bash
!python src/trainingGpt.py
```

This will:
- Download the Tiny Shakespeare dataset
- Train the model
- Save checkpoint as `gpt_shakespeare_ds.pt`

### 5. Generate Text
Once training is complete:

```bash
!python src/generateGpt.py
```

You'll get an interactive prompt:
```
Enter a prompt (or 'quit' to exit): To be or not to be
=== Model Output ===
To be or not to be, I woul'd spake...
```

---

## 💻 Run Locally

### 1. Setup Environment
```bash
git clone https://github.com/your-username/mini-gpt-shakespeare.git
cd mini-gpt-shakespeare
```

### 2. Install Dependencies
Ensure you have **Python 3.9+** installed, then:

```bash
pip install torch tqdm requests
```

For GPU support (recommended):
```bash
# For CUDA (NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For MPS (Apple Silicon Macs)
pip install torch torchvision torchaudio
```

### 3. Train the Model
```bash
python src/trainingGpt.py
```

### 4. Generate Text
```bash
python src/generateGpt.py
```

---

## ⚙️ Configuration Options

### Training Speed vs Quality Trade-offs

| Setting | Training Time | Model Size | Quality |
|---------|---------------|------------|---------|
| **Demo** | 1-2 min | Small (128d, 2 layers) | Basic |
| **Default** | 10-15 min | Medium (256d, 4 layers) | Good |
| **Extended** | 30+ min | Large (512d, 6+ layers) | Best |

### Key Parameters to Modify

In `trainingGpt.py`, you can adjust:

```python
# Training duration
max_iters = 5000  # Increase for better quality

# Model architecture
cfg = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    d_model=256,      # Model dimension (128/256/512)
    n_heads=4,        # Attention heads (2/4/8)
    n_layers=4,       # Transformer layers (2/4/6/8)
    d_ff=1024         # Feed-forward dimension
)

# Learning settings
learning_rate = 3e-4
batch_size = 64
```

---

## 📊 Expected Results

After training, you should see decreasing loss values:
```
Iter 1000: train loss 2.4521, val loss 2.4832
Iter 2000: train loss 2.1234, val loss 2.2156
Iter 3000: train loss 1.8567, val loss 1.9823
...
```

Generated text will improve from random characters to Shakespeare-like prose:
- **Early training**: Random gibberish
- **Mid training**: Recognizable words and patterns  
- **Late training**: Coherent Shakespeare-style sentences

---

## 🛠️ Troubleshooting

### Common Issues

**"CUDA out of memory"** → Reduce `batch_size` or model dimensions

**"Module not found"** → Make sure files are in `src/` folder

**Slow training on CPU** → Enable GPU in Colab (Runtime → Change runtime type → GPU)

**Poor generation quality** → Increase `max_iters` or model size

### File Structure
```
mini-gpt-shakespeare/
├── src/
│   ├── trainingGpt.py
│   └── generateGpt.py
├── gpt_shakespeare_ds.pt  (generated after training)
└── README.md
```

---

## 🎯 Quick Start Summary

1. **Colab Users**: Upload files → Install deps → Run training → Generate
2. **Local Users**: Clone → Install deps → Train → Generate  
3. **Quick Demo**: Edit training params for 1000 iterations
4. **Better Results**: Use default 5000+ iterations

Happy text generation! 🎭✨