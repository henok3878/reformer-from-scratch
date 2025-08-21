# Transformer with Rotary Positional Embeddings

This project extends [transformer-from-scratch](https://github.com/henok3878/transformer-from-scratch) by integrating Rotary Positional Embeddings (RoPE) as a drop-in replacement for sinusoidal position encodings. The goal is to improve relative position modeling while keeping the original architecture, model size (~65M parameters), and training recipe.

## Setup

This project uses Conda for environment management and pip for installing development tools and the core package. Both CPU and GPU environments are supported.

### Installation

```bash
git clone https://github.com/henok3878/reformer-from-scratch.git
cd reformer-from-scratch
```

Use the provided setup script to create either a CPU or GPU environment:

```bash
# for CPU:
./setup_env.sh cpu

# for GPU (with CUDA):
./setup_env.sh gpu
```

## Experiment Tracking

This project uses [Weights & Biases (wandb)](https://wandb.ai/) for experiment tracking. Key metrics like loss, perplexity, and BLEU scores are automatically logged, along with model configurations and checkpoints during training.

### Setup & Usage

1. **Sign up** for a free account at [wandb.ai](https://wandb.ai).
2. **Login** from the terminal:
   ```bash
   wandb login
   ```
3. Running `train.py` will automatically create a new run in your wandb project (`reformer-from-scratch`). Monitor experiments in the wandb dashboard.

## Project Structure

```
src/reformer/
├── components/
│   ├── multi_head_rope.py   # Multi-head attention with RoPE
│   └── rope.py              # Rotary embedding utilities
├── transformer_rope.py      # Complete Transformer model
└── __init__.py

configs/                     # YAML configs (e.g., config_de-en.yml)
train.py                     # Training script
run_dist.sh                  # Distributed training launcher
setup_env.sh                 # Convenience environment setup
```

## Model Training

This project supports both single-node single-GPU and distributed multi-GPU training.

### Single-Node Single-GPU Training

Run basic training on one GPU:

```bash
python train.py --config configs/config_de-en.yml
```

No `.env` or `hostfile` setup is required.

---

### Multi-GPU Training (Single-Node or Multi-Node)

For multi-GPU runs, create a `.env` file:

```properties
MASTER_ADDR=node001
MASTER_PORT=29500
NNODES=1
GPUS_PER_NODE=2
NCCL_DEBUG=INFO
```

For multi-node training, also create a `hostfile` listing each node, one per line (master first):

```
node001
node002
node003
```

Launch distributed training on each node:

```bash
bash run_dist.sh
```

Checkpoints are saved under `experiments/` and logged to wandb.

## Training & Evaluation Results

- **Final Test BLEU:** 25.97 on WMT14 En–De using beam search with beam size 4 (baseline: 25.53).
- [**View Project on W&B →**](https://wandb.ai/henokwondimu/reformer-from-scratch?nw=nwuserhenok3878)

## License

MIT
