# MiniTransformer

## 1. Project Overview

This project implements a decoder-only Transformer language model in PyTorch from first principles and investigates the effects of architectural and optimization hyperparameters on training dynamics and representational behavior.

The implementation includes:

- Multi-head self-attention
- Positional embeddings
- Residual pathways
- Layer normalization
- Feed-forward blocks
- Autoregressive training
- Configurable training pipelines

The project focuses on both implementation correctness and empirical analysis of transformer behavior.

## 2. Motivation

The goal of this project was to build a controlled Transformer system from first principles and use it to study the effect of architectural and optimization choices. The implementation is used to explore:

- core Transformer components implemented directly in PyTorch
- comparison of single-head, multi-head, and stacked architectures under controlled conditions
- sensitivity of learning dynamics to embedding dimension, head count, learning rate, and context length
- qualitative and structural analysis of learned representations and attention patterns

## 3. Architecture Section

### 3.1 Shared design

All variants are causal language models trained with next-token prediction. The input is a sequence of token IDs, and the model predicts the token distribution at each position under a causal mask so each position attends only to previous tokens via a causal mask.

The shared design elements are:

- token embeddings
- positional embeddings
- causal attention masking
- residual connections
- feed-forward networks
- cross-entropy loss over flattened token positions

### 3.2 Variant v1: Single-head Transformer

The v1 model in [minitransformer/models.py](minitransformer/models.py) implements a minimal causal self-attention block with one attention head.

Key components:

- one query, key, and value projection
- a lower-triangular attention mask
- a residual attention branch
- a feed-forward network with ReLU activation
- a final linear language-model head

This version is useful as a baseline because it exposes the mechanics of self-attention in the simplest possible form.

### 3.3 Variant v2: Multi-head Transformer

The v2 model extends the baseline with a reusable Transformer block:

- pre-norm layer normalization before attention
- multi-head attention implemented by reshaping the projected queries, keys, and values
- a learned output projection after attention
- a feed-forward sublayer with GELU activation
- tied token embedding and output weights

This variant is the first one that matches the standard Transformer block pattern more closely.

### 3.4 Variant v3: Stacked Transformer

The v3 model stacks multiple Transformer blocks, followed by a final layer normalization and tied output head.

This architecture is the most expressive model in the repository and is the main setting for controlled experiments on depth, context length, and optimization.

### 3.5 Model selection logic

The project exposes a simple factory function that chooses among the three variants based on a model label. The implementation also enforces the constraint that head_num × head_size = embed_dim, which keeps the multi-head projections compatible with the embedding width.

## 4. Experimental Setup

### 4.1 Data

The default training source is TinyStories, loaded from Hugging Face through the `datasets` library. The CLI also supports a local text file via `--source`.

The default training dataset is TinyStories, loaded via the Hugging Face `datasets` library. The CLI also supports training on a local text corpus through `--source`.

Preprocessing converts text into a character-level vocabulary for autoregressive next-token prediction.

### 4.2 Training procedure

Training is handled by the CLI in [minitransformer/cli.py](minitransformer/cli.py) and the loop in [minitransformer/training.py](minitransformer/training.py). The model is optimized with Adam, using teacher forcing on contiguous token blocks.

Default training settings:

- batch size: 32
- block size: 128
- learning rate: 1e-3
- training steps: 30,000 (v1–v3 experiments)
- embedding dimension: 256
- number of heads: 8
- number of layers (v3): 4

### 4.3 Reproducibility controls

The training script seeds Python and PyTorch through a helper that also covers CUDA and MPS when available. The CLI accepts a `--seed` argument, so the same experiment can be rerun with consistent initialization and sampling behavior.

## 5. Experiments

This section summarizes the experimental directions supported by the code and the saved artifacts in the repository. You can expand each subsection later with tables, quantitative results, or ablation summaries.

### 5.1 Architecture comparison

The three model variants represent increasing architectural capacity:

- v1: single-head causal attention baseline
- v2: multi-head attention within a single Transformer block
- v3: stacked multi-layer Transformer with pre-norm residual blocks

Training curves and checkpoints are saved for all variants under identical dataset and optimization settings.

### 5.2 Key empirical observations

Across controlled training runs, the following trends were observed:

- Increasing model depth improves convergence stability and reduces final training loss
- Multi-head attention improves optimization smoothness compared to single-head baseline
- Deeper models produce more syntactically consistent long-form generation
- Larger context windows improve coherence but slow early-stage convergence

## 6. Training Process & Result

### 6.1 Training process

Training was performed on an NVIDIA T4 GPU (Google Colab) and Apple M1 Pro (MPS backend).

Approximate training time for v3 (30k steps):

- T4 GPU: ~10–15 minutes
- M1 Pro: ~15–25 minutes

All models use identical optimization settings unless explicitly specified in ablation runs.
Each model use default config mentioned in section 4.2

### 6.2 Model info

Note: loss values are not directly comparable to standard token-level perplexity benchmarks due to character-level modeling and dataset scale.

| Model |  Size  | Final Train Loss | Sample Quality     |
| :---- | :----: | :--------------: | :----------------- |
| v1    | 3.2M   |     1.0586       | Incoherent         |
| v2    | 3.4M   |     0.8654       | Partially coherent |
| v3    | 13.1M  |     0.2484       | Mostly coherent    |

### 6.2 Loss curves

The following figures were generated by the CLI during training:

- v1: [model/loss_curve_v1_20260520_160016.png](model/loss_curve_v1_20260520_160016.png)
- v2: [model/loss_curve_v2_20260520_161125.png](model/loss_curve_v2_20260520_161125.png)
- v3: [model/loss_curve_v3_20260520_164209.png](model/loss_curve_v3_20260520_164209.png)

### 6.3 Sample generation comparison

v1 shows weak long-range structure and frequent syntactic breaks.

v2 improves local coherence and sentence continuity but still exhibits instability in narrative consistency.

v3 produces more stable sentence structure and improved global coherence, especially in maintaining subject consistency across paragraphs.

## 7. Limitations

The model is intentionally constrained to a small-scale setting:

- character-level tokenization rather than subword tokenization
- limited dataset scale compared to modern language model training
- no built-in evaluation benchmark or perplexity reporting pipeline
- no automated attention or activation visualization pipeline

These constraints define the scope of the project as an experimental and educational Transformer implementation.

## 8. Future Work

Planned analysis directions include:

- head-wise attention pattern comparison across layers
- locality vs long-range attention quantification
- representation similarity across layers
- embedding space analysis under different training configurations

Potential extensions include:

- validation split and perplexity evaluation
- attention map visualization pipeline
- systematic ablations on depth, width, and context length
- structured logging of training hyperparameters and metrics

## 9. References

- Vaswani et al., 2017, Attention Is All You Need
- Radford et al., 2019, Language Models are Unsupervised Multitask Learners

## Reproduction Notes

Train a model:

```bash
python3 mini_transformer.py --train --model v3 --model-path model/model3.pt --loss-curve-dir model
```

Evaluate a saved checkpoint:

```bash
python3 mini_transformer.py --eval --load-model --model v3 --model-path model/model3.pt --seed 100
```

If you want to use a local corpus instead of TinyStories, pass a text file through `--source`.
