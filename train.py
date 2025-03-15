#!/usr/bin/env python
"""
Main training script for genomic BERT with selectable attention mechanisms.

This script uses PyTorch Accelerate to train a BERT model on genomic sequences
with support for different attention mechanisms including standard BERT attention
and ALiBi attention for improved long-sequence handling.
"""

import os
import argparse
import logging
import torch
from accelerate.utils import set_seed

from training.accelerate_utils import train_with_accelerate, setup_accelerator

logger = logging.getLogger(__name__)


def main():
    """
    Parse arguments and start training with Accelerate.
    """
    # Auto-detect CUDA devices and configure environment
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if "CUDA_VISIBLE_DEVICES" not in os.environ and num_gpus > 0:
            # If not already set, use all available GPUs
            gpus = list(range(num_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
            logger.info(f"Auto-configured to use all {num_gpus} available GPUs")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a BERT model for genomic sequences using Accelerate"
    )

    # Input/Output
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--input_files", nargs="+", required=True,
                          help="Input genomic sequence files (FASTA format)")
    io_group.add_argument("--output_dir", required=True,
                          help="Output directory for model, tokenizer, and logs")
    io_group.add_argument("--tokenizer_path", required=True,
                          help="Path to pre-trained tokenizer directory (required)")
    io_group.add_argument("--model_path", default=None,
                          help="Path to existing model checkpoint (optional)")
    io_group.add_argument("--resume_from_checkpoint", type=str, default=None,
                          help="Path to checkpoint to resume from, or 'latest' to find latest checkpoint")
    io_group.add_argument("--test_sequence_length", type=int, default=None,
                          help="Override sequence length for testing (useful for debugging)")

    # GPU options
    gpu_group = parser.add_argument_group("GPU Options")
    gpu_group.add_argument("--force_gpu", action="store_true",
                           help="Force GPU usage even if distributed mode is active")
    gpu_group.add_argument("--gpu_ids", type=str, default=None,
                           help="Comma-separated list of GPU IDs to use (defaults to all available)")

    # Model architecture
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--hidden_size", type=int, default=768,
                             help="Hidden size of the model (default: 768)")
    model_group.add_argument("--num_hidden_layers", type=int, default=12,
                             help="Number of hidden layers (default: 12)")
    model_group.add_argument("--num_attention_heads", type=int, default=12,
                             help="Number of attention heads (default: 12)")
    model_group.add_argument("--dropout", type=float, default=0.1,
                             help="Dropout probability (default: 0.1)")
    # Add attention type selection
    model_group.add_argument("--attention_type", type=str, default="alibi", choices=["standard", "alibi"],
                             help="Type of attention mechanism to use (default: alibi)")

    # Sequence length options
    seq_group = parser.add_argument_group("Sequence Options")
    seq_group.add_argument("--pre_training_length", type=int, default=512,
                           help="Sequence length during pre-training (default: 512)")
    seq_group.add_argument("--max_inference_length", type=int, default=None,
                           help="Maximum sequence length for inference (default: None = unlimited)")
    seq_group.add_argument("--mlm_probability", type=float, default=0.15,
                           help="Probability of masking a token for MLM (default: 0.15)")
    seq_group.add_argument("--chunk_size", type=int, default=2000,
                           help="Base size of sequence chunks (default: 2000)")
    seq_group.add_argument("--stride", type=int, default=1000,
                           help="Stride for overlapping chunks (default: 1000)")
    seq_group.add_argument("--sample_long_sequences", action="store_true",
                           help="Include longer sequences during training to help with extrapolation")
    seq_group.add_argument("--max_safe_sequence_length", type=int, default=50000,
                           help="Maximum safe sequence length for processing (default: 50000)")
    seq_group.add_argument("--max_supported_model_length", type=int, default=16384,
                           help="Maximum sequence length the model should support (default: 16384)")
    seq_group.add_argument("--use_reverse_complement", action="store_true",
                           help="Include reverse complement sequences for augmentation (default: True)")
    seq_group.add_argument("--disable_reverse_complement", action="store_true",
                           help="Disable reverse complement augmentation")

    # Training options
    training_group = parser.add_argument_group("Training Options")
    training_group.add_argument("--batch_size", type=int, default=16,
                                help="Batch size per GPU/TPU core/CPU (default: 16)")
    training_group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                                help="Number of steps for gradient accumulation (default: 1)")
    training_group.add_argument("--epochs", type=int, default=3,
                                help="Number of training epochs (default: 3)")
    training_group.add_argument("--learning_rate", type=float, default=5e-5,
                                help="Learning rate (default: 5e-5)")
    training_group.add_argument("--weight_decay", type=float, default=0.01,
                                help="Weight decay (default: 0.01)")
    training_group.add_argument("--warmup_steps", type=int, default=10000,
                                help="Number of warmup steps (default: 10000)")
    training_group.add_argument("--max_grad_norm", type=float, default=1.0,
                                help="Maximum gradient norm (default: 1.0)")
    training_group.add_argument("--num_workers", type=int, default=4,
                                help="Number of data loader workers (default: 4)")
    training_group.add_argument("--lr_scheduler_type", type=str, default="linear",
                                choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                         "constant_with_warmup"],
                                help="LR scheduler type (default: linear)")

    # Logging and checkpointing
    logging_group = parser.add_argument_group("Logging and Checkpointing")
    logging_group.add_argument("--logging_steps", type=int, default=100,
                               help="Log every X steps (default: 100)")
    logging_group.add_argument("--checkpointing_steps", type=int, default=None,
                               help="Save a checkpoint every X steps (default: None)")
    logging_group.add_argument("--save_steps", type=int, default=1000,
                               help="Save a checkpoint every X steps (default: 1000)")
    logging_group.add_argument("--save_total_limit", type=int, default=3,
                               help="Maximum number of checkpoints to keep (default: 3)")
    logging_group.add_argument("--log_with_tensorboard", action="store_true",
                               help="Log training with TensorBoard")

    # Other options
    other_group = parser.add_argument_group("Other Options")
    other_group.add_argument("--seed", type=int, default=42,
                             help="Random seed (default: 42)")
    other_group.add_argument("--debug", action="store_true",
                             help="Enable debug output")

    args = parser.parse_args()

    # Configure Accelerate first
    accelerator = setup_accelerator(args)

    # Configure logging AFTER accelerator setup to avoid duplication
    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if not args.debug else logging.DEBUG
        )
    else:
        # Non-main processes should log only warnings and errors
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.WARNING
        )

    # Configure GPU usage based on arguments
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        logger.info(f"Using GPU IDs: {args.gpu_ids}")

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    # Run diagnostics before training
    if torch.cuda.is_available():
        logger.info(f"Found {torch.cuda.device_count()} CUDA devices")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("No CUDA devices found! Training will run on CPU.")

    # Set random seed early for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    # Verify that tokenizer exists
    if not os.path.exists(args.tokenizer_path):
        raise ValueError(f"Tokenizer path '{args.tokenizer_path}' does not exist. "
                         f"Please run train_tokenizer.py first to create the tokenizer.")

    # Start training with Accelerate
    train_with_accelerate(args, accelerator)


if __name__ == "__main__":
    main()