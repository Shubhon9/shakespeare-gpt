# 📝 Mini-GPT: Train, Generate & Visualize on Tiny Shakespeare

This repo contains scripts to **train**, **generate text**, and **visualize embeddings** using a GPT-style model on the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

- `notebook/trainingGpt.py` → Train the model and save weights  
- `notebook/generateGpt.py` → Load saved weights and interactively generate text  
- `src/visual.py` → Extract token embeddings and export for visualization  
- `src/model.py` → GPT architecture definition  
- `src/dataset.py` → Dataset handling  
- `src/train.py` → Training loop logic  
- `src/generate.py` → Text generation logic  

---

## 🚀 Run in Google Colab

### 1. Setup Files
1. Upload the repo to Colab (or `!git clone <repo-url>`)  
2. Your folder structure will be preserved with `notebook/` and `src/`  

### 2. Install Dependencies
```bash
!pip install torch tqdm requests scikit-learn
```

### 3. Train the Model
```bash
!python notebook/trainingGpt.py
```
This saves a checkpoint like `checkpoints/gpt_shakespeare.pt`.

### 4. Generate Text
```bash
!python notebook/generateGpt.py
```

### 5. 📊 Visualize Embeddings in Colab

#### Option A → **TensorBoard (interactive inside Colab)**
```python
from torch.utils.tensorboard import SummaryWriter
import torch
from src.model import GPT, GPTConfig
from notebook.trainingGpt import vocab_size, block_size, itos

# Load model + weights
cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size)
model = GPT(cfg)
model.load_state_dict(torch.load("checkpoints/gpt_shakespeare.pt", map_location="cpu"))

# Extract embeddings
emb = model.token_emb.weight.detach().cpu()

# Save to TensorBoard
writer = SummaryWriter("runs/embeddings")
writer.add_embedding(emb, metadata=[itos[i] for i in range(len(itos))])
writer.close()

%load_ext tensorboard
%tensorboard --logdir runs/embeddings
```

#### Option B → **TensorFlow Projector**
```bash
!python src/visual.py
```
This generates:
- `embeddings.tsv`
- `metadata.tsv`

Download both and upload to [TensorFlow Projector](https://projector.tensorflow.org/) to explore in 2D/3D.

---

## 💻 Run Locally

### 1. Setup Environment
```bash
git clone https://github.com/your-username/mini-gpt-shakespeare.git
cd mini-gpt-shakespeare
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python notebook/trainingGpt.py
```

### 3. Generate Text
```bash
python notebook/generateGpt.py
```

### 4. Visualize Embeddings Locally

#### Option A → **TensorBoard**
```bash
python -m torch.utils.tensorboard.main --logdir runs/embeddings
```
Then open http://localhost:6006.

#### Option B → **TensorFlow Projector**
```bash
python src/visual.py
```
This creates `embeddings.tsv` + `metadata.tsv`. Upload them to https://projector.tensorflow.org/.

---

## ⚙️ Configuration Options

### Training Speed vs Quality Trade-offs

| Setting | Training Time | Model Size | Quality |
|---------|---------------|------------|---------|
| **Demo** | 1-2 min | Small (128d, 2 layers) | Basic |
| **Default** | 10-15 min | Medium (256d, 4 layers) | Good |
| **Extended** | 30+ min | Large (512d, 6+ layers) | Best |

### Key Parameters to Modify

In `notebook/trainingGpt.py`, you can adjust:

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

## 📂 File Structure

```
mini-gpt-shakespeare/
├── notebook/
│   ├── trainingGpt.py       # Train model
│   ├── generateGpt.py       # Generate Shakespeare text
│
├── src/
│   ├── dataset.py           # Dataset handling
│   ├── generate.py          # Generation logic
│   ├── model.py             # GPT architecture
│   ├── train.py             # Training loop
│   └── visual.py            # Export embeddings
│
├── checkpoints/             # Saved models (*.pt)
├── embeddings.tsv           # (created by visual.py)
├── metadata.tsv             # (created by visual.py)
├── requirements.txt
└── README.md
```

---

## 📊 Expected Results

### Training Progress
After training, you should see decreasing loss values:
```
Iter 1000: train loss 2.4521, val loss 2.4832
Iter 2000: train loss 2.1234, val loss 2.2156
Iter 3000: train loss 1.8567, val loss 1.9823
...
```

### Text Generation Quality
Generated text will improve from random characters to Shakespeare-like prose:
- **Early training**: Random gibberish
- **Mid training**: Recognizable words and patterns  
- **Late training**: Coherent Shakespeare-style sentences

### Embedding Visualization
The embedding visualizations will reveal:
- **Character clusters**: Similar characters (vowels, consonants) group together
- **Semantic relationships**: Related punctuation and common letter pairs
- **Language patterns**: Frequent Shakespeare vocabulary forms distinct clusters

---

## 🛠️ Troubleshooting

### Common Issues

**"CUDA out of memory"** → Reduce `batch_size` or model dimensions

**"Module not found"** → Ensure proper file structure with `notebook/` and `src/` folders

**Slow training on CPU** → Enable GPU in Colab (Runtime → Change runtime type → GPU)

**Poor generation quality** → Increase `max_iters` or model size

**TensorBoard not loading** → Check that logs are saved in `runs/embeddings/` directory

### Dependencies
Create `requirements.txt` with:
```
torch>=1.9.0
tqdm>=4.62.0
requests>=2.26.0
scikit-learn>=1.0.0
tensorboard>=2.7.0
```

---

## 🎯 Quick Start Summary

1. **Colab** → Run `trainingGpt.py` → Run `generateGpt.py` → Visualize embeddings via TensorBoard or Projector
2. **Local** → Same workflow using `python notebook/...` and `python src/visual.py`
3. **TensorBoard** → Best for interactive embedding exploration
4. **Projector** → Great for sharing embedding maps online

✨ Now you can not only generate Shakespeare-like text, but also **visualize how your GPT's embeddings cluster characters together**!

---

## 🔬 Understanding Your Embeddings

When you visualize the token embeddings, look for:
- **Vowel clustering**: a, e, i, o, u should group together
- **Punctuation patterns**: commas, periods, semicolons in similar regions
- **Common bigrams**: frequent letter pairs like "th", "he", "in" close to each other
- **Case sensitivity**: uppercase vs lowercase versions of same letters

This visualization helps you understand what your model has learned about character relationships in Shakespeare's writing style!