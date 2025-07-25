# Transformer Model for English-French Machine Translation

This repository contains the implementation of a Transformer model for English-to-French neural machine translation, built using PyTorch. The code is structured to allow for training, evaluation, and customization of the Transformer architecture.

---

## 📑 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Evaluation (Coming Soon)](#evaluation-coming-soon)
- [Project Structure](#project-structure)
- [Model Architecture Highlights](#model-architecture-highlights)
- [Troubleshooting & Performance Tips](#troubleshooting--performance-tips)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Introduction

This project implements the Transformer architecture, as described in the *"Attention Is All You Need"* paper, for translating text from English to French. It includes custom implementations of key components such as multi-head attention, positional encoding, and the encoder-decoder structure.

---

## ✨ Features

- **Custom Transformer Implementation:** Multi-Head Attention, Positional Encoding, Feed-Forward Networks.
- **Encoder-Decoder Architecture:** Stacked encoder and decoder layers.
- **Dataset Handling:** Utilizes Hugging Face's `datasets` library with a custom `BilingualDataset`.
- **Tokenization:** Uses `tokenizers` for `WordLevel` tokenization with special tokens.
- **Training Loop:** Basic script with TensorBoard logging support.
- **Model Checkpointing:** Periodic saving of weights and optimizer state.
- **Mixed Precision Training:** Support via `torch.cuda.amp` for speed and efficiency (currently commented out for clarity).

---

## ⚙️ Setup

### ✅ Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA (recommended)

### 📦 Installation

```bash
git clone https://github.com/your-username/your-transformer-repo.git
cd your-transformer-repo
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required libraries:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adjust for your CUDA version
pip install datasets tokenizers numpy tqdm tensorboard
```

---

## 🚀 Usage

### ⚙️ Configuration

All settings and hyperparameters are defined in `config.py`:

```python
# config.py
from pathlib import Path

def get_config():
    return {
        "batch_size": 4,
        "num_epochs": 50,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "multi_head_transformer",
        "preload": None,
        "tokenizer_file_en": "tokenizer_en.json",
        "tokenizer_file_fr": "tokenizer_fr.json",
        "experiment_name": "runs/model"
    }

def get_weights_file_path(config, epoch):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
```

Ensure you update the tokenizer paths in `Test.py` accordingly.

---

### 🏋️ Training

To train the model:

```bash
python Test.py
```

This script will:

- Load the **opus_books** English-French dataset
- Train/load tokenizers
- Prepare dataloaders
- Build and train the Transformer model
- Save checkpoints to the `weights/` folder
- Log training to TensorBoard

To monitor training:

```bash
tensorboard --logdir runs
```

Open your browser at: [http://localhost:6006](http://localhost:6006)

---

## 🧪 Evaluation (Coming Soon)

A BLEU score-based evaluation script will be included in a future release.

---

## 🗂 Project Structure

```
.
├── config.py          # Configuration and hyperparameters
├── Dataset.py         # Custom Dataset class and data handling
├── main.py            # Core Transformer architecture components
├── Test.py            # Training and tokenization pipeline
└── weights/           # Model checkpoints (auto-created)
```

---

## 🧱 Model Architecture Highlights

Implemented in `main.py`:

- `InputEmbeddings`: Converts tokens into dense vectors
- `PositionalEmbedding`: Sine-cosine positional encodings
- `LayerNormalization`: Stabilizes learning
- `FeedForward`: Two-layer MLP with ReLU
- `MultiHeadAttention`: Self and cross attention
- `ResidualConnection`: With layer norm
- `EncoderLayer` & `DecoderLayer`: Building blocks
- `Encoder` & `Decoder`: N-stacked layers
- `ProjectionLayer`: Output to vocab space
- `Transformer`: Full encoder-decoder pipeline

---

## 🛠️ Troubleshooting & Performance Tips

### 🔥 CUDA Out of Memory?

- Reduce `batch_size`, `seq_len`, `d_model`, or `N` (layers)
- Enable mixed-precision with `torch.cuda.amp`

### 🐌 Slow Training?

- Set `num_workers=0` temporarily in `DataLoader`
- Monitor GPU usage using `nvidia-smi`
- Limit dataset size:  
  Example:
  ```python
  ds_raw = ds_raw.select(range(10000))
  ```
- Save tokenizers after training to avoid retraining
- Disable tokenizer parallelism warning:
  ```bash
  export TOKENIZERS_PARALLELISM=false
  ```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request if you have suggestions for improvements or bug fixes.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
