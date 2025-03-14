# Genomic BERT with ALiBi Attention

A specialized framework for pre-training BERT models on genomic sequences with support for long sequence extrapolation using ALiBi attention. This project provides a complete solution for training language models on DNA data that can effectively handle sequences much longer than those seen during training.

## Overview

This project implements a transformer-based architecture for genomic sequence modeling, with special focus on:

1. **DNA-specific tokenization** using SentencePiece BPE
2. **Long sequence handling** with ALiBi (Attention with Linear Biases) attention mechanism
3. **Distributed training** support via PyTorch Accelerate
4. **Robust safety features** for handling genomic data quirks

The resulting models can be used for various genomic tasks such as:
- DNA sequence classification
- Motif discovery
- Sequence generation
- Anomaly detection in genomic sequences
- Transfer learning for genomic tasks

## Key Features

### ALiBi Attention

Unlike standard transformer attention which relies on positional embeddings limited to the training sequence length, ALiBi applies linear biases to attention scores based on the relative positions of tokens. This approach allows models to effectively process sequences much longer than those seen during training.

Benefits:
- **Improved extrapolation** to longer sequences
- **No position embedding limit** - handles sequences of arbitrary length
- **Better performance on long-range dependencies**

### Genomic Tokenization

Custom SentencePiece BPE tokenizer designed specifically for genomic sequences (A, T, G, C).

Benefits:
- **DNA-specific vocabulary** that captures biologically relevant patterns
- **Efficient representation** of genomic motifs and structures
- **Robust handling** of invalid characters and outlier sequences

### Distributed Training

Built with PyTorch Accelerate for efficient multi-GPU and multi-node training.

Benefits:
- **Faster training** on large genomic datasets
- **Seamless scaling** from single GPU to multi-node
- **Efficient checkpointing** and resumption

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/genomic-bert.git
cd genomic-bert

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
python train.py \
  --input_files path/to/your/genomic.fasta \
  --output_dir ./output \
  --attention_type alibi \
  --batch_size 16
```

## Command-line Options

### Input/Output

| Option | Description |
|--------|-------------|
| `--input_files` | Input genomic sequence files in FASTA format (required) |
| `--output_dir` | Output directory for model, tokenizer, and logs (required) |
| `--tokenizer_path` | Path to existing tokenizer (optional) |
| `--model_path` | Path to existing model checkpoint (optional) |
| `--resume_from_checkpoint` | Path to checkpoint to resume from, or 'latest' to find latest checkpoint |
| `--test_sequence_length` | Override sequence length for testing (useful for debugging) |

### GPU Options

| Option | Description |
|--------|-------------|
| `--force_gpu` | Force GPU usage even if distributed mode is active |
| `--gpu_ids` | Comma-separated list of GPU IDs to use (defaults to all available) |

### Tokenizer Options

| Option | Description |
|--------|-------------|
| `--vocab_size` | Vocabulary size for BPE tokenizer (default: 4096) |
| `--tokenizer_sample_size` | Number of sequences to sample for tokenizer training (default: 100000) |

### Model Architecture

| Option | Description |
|--------|-------------|
| `--hidden_size` | Hidden size of the model (default: 768) |
| `--num_hidden_layers` | Number of hidden layers (default: 12) |
| `--num_attention_heads` | Number of attention heads (default: 12) |
| `--dropout` | Dropout probability (default: 0.1) |
| `--attention_type` | Type of attention mechanism to use: "standard" or "alibi" (default: alibi) |

### Sequence Options

| Option | Description |
|--------|-------------|
| `--pre_training_length` | Sequence length during pre-training (default: 512) |
| `--max_inference_length` | Maximum sequence length for inference (default: None = unlimited) |
| `--mlm_probability` | Probability of masking a token for MLM (default: 0.15) |
| `--chunk_size` | Base size of sequence chunks (default: 2000) |
| `--stride` | Stride for overlapping chunks (default: 1000) |
| `--sample_long_sequences` | Include longer sequences during training to help with extrapolation |
| `--max_safe_sequence_length` | Maximum safe sequence length for processing (default: 50000) |
| `--max_supported_model_length` | Maximum sequence length the model should support (default: 16384) |

### Training Options

| Option | Description |
|--------|-------------|
| `--batch_size` | Batch size per GPU/TPU core/CPU (default: 16) |
| `--gradient_accumulation_steps` | Number of steps for gradient accumulation (default: 1) |
| `--epochs` | Number of training epochs (default: 3) |
| `--learning_rate` | Learning rate (default: 5e-5) |
| `--weight_decay` | Weight decay (default: 0.01) |
| `--warmup_steps` | Number of warmup steps (default: 10000) |
| `--max_grad_norm` | Maximum gradient norm (default: 1.0) |
| `--num_workers` | Number of data loader workers (default: 4) |
| `--lr_scheduler_type` | LR scheduler type: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup" (default: linear) |

### Logging and Checkpointing

| Option | Description |
|--------|-------------|
| `--logging_steps` | Log every X steps (default: 100) |
| `--checkpointing_steps` | Save a checkpoint every X steps (default: None) |
| `--save_steps` | Save a checkpoint every X steps (default: 1000) |
| `--save_total_limit` | Maximum number of checkpoints to keep (default: 3) |
| `--log_with_tensorboard` | Log training with TensorBoard |

### Other Options

| Option | Description |
|--------|-------------|
| `--seed` | Random seed (default: 42) |
| `--debug` | Enable debug output |

## Advanced Usage

### Training with Multiple GPUs

```bash
python train.py \
  --input_files data/*.fasta \
  --output_dir ./multi_gpu_output \
  --attention_type alibi \
  --batch_size 8 \
  --gradient_accumulation_steps 2
```

### Fine-tuning a Pre-trained Model

```bash
python train.py \
  --input_files path/to/your/genomic.fasta \
  --output_dir ./finetune_output \
  --model_path ./pretrained_model \
  --tokenizer_path ./pretrained_model/tokenizer \
  --learning_rate 2e-5 \
  --epochs 1
```

### Testing Extrapolation to Longer Sequences

The model automatically tests its ability to handle sequences of different lengths than those seen during training. You can customize this with:

```bash
python train.py \
  --input_files path/to/your/genomic.fasta \
  --output_dir ./output \
  --pre_training_length 512 \
  --max_supported_model_length 16384 \
  --sample_long_sequences
```

## Project Structure

```
.
├── data/
│   ├── __init__.py
│   └── data_collator.py         # Dataset and data collator implementations
├── modeling/
│   ├── __init__.py
│   └── alibi_attention.py       # ALiBi attention implementation
├── tokenization/
│   ├── __init__.py
│   └── genomic_tokenizer.py     # Genomic tokenizer implementation
├── training/
│   ├── __init__.py
│   ├── accelerate_utils.py      # Accelerate utilities
│   └── train_utils.py           # Training utilities
└── train.py                     # Main training script
```

## Customizing the Tokenizer

The tokenizer is automatically trained on your input data unless you provide an existing tokenizer. You can customize vocabulary size and other parameters:

```bash
python train.py \
  --input_files path/to/your/genomic.fasta \
  --output_dir ./output \
  --vocab_size 8192 \
  --tokenizer_sample_size 200000
```

## Performance Considerations

- **Memory Usage**: ALiBi attention has slightly higher memory requirements than standard attention.
- **Sequence Length**: Training with longer sequences requires more memory. Use `--gradient_accumulation_steps` to trade speed for memory.
- **Multi-GPU**: For large genomic datasets, multi-GPU training can significantly reduce training time.

## Citation

If you use this code in your research, please cite:

```
@article{tbd,
  title={TBD},
  author={TBD},
  journal={arXiv preprint},
  year={2025}
}
```

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
