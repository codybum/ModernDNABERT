"""
Training utilities for genomic BERT model.

This module provides utility functions for model testing, evaluation,
and other training-related operations.
"""

import random
import logging
import torch
import copy
from typing import List, Dict, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)


def generate_test_sequences(lengths=[512, 1024, 2048, 4096, 8192]):
    """
    Generate test sequences of different lengths for extrapolation testing.

    Args:
        lengths: List of sequence lengths to generate

    Returns:
        List of DNA sequences
    """
    nucleotides = ['A', 'T', 'G', 'C']
    return [
        ''.join(random.choice(nucleotides) for _ in range(length))
        for length in lengths
    ]


def test_sequence_length_extrapolation(accelerator, model, tokenizer, test_sequences: List[str]):
    """
    Test model on sequences of different lengths with robust error handling.

    Args:
        accelerator: Accelerator instance
        model: Model to test
        tokenizer: Tokenizer to use
        test_sequences: List of test sequences

    Returns:
        List of test results
    """
    # Get the unwrapped model first
    unwrapped_model = accelerator.unwrap_model(model)

    # Create a CPU copy of the model for testing
    with torch.no_grad():
        # Deep copy the unwrapped model
        cpu_model = copy.deepcopy(unwrapped_model)
        # Move to CPU in one go, safely
        cpu_model.to('cpu').eval()

    logger.info("\n" + "=" * 50)
    logger.info("RUNNING LENGTH EXTRAPOLATION TEST ON CPU")
    logger.info("=" * 50)

    results = []
    for seq in test_sequences:
        try:
            # Validate sequence first
            if not seq or not isinstance(seq, str):
                logger.warning(f"Invalid sequence: {type(seq)}. Skipping.")
                continue

            # Filter sequence to only valid nucleotides
            seq = seq.upper()
            seq = ''.join(c for c in seq if c in 'ATGC')

            if not seq:
                logger.warning("Sequence contains no valid nucleotides. Skipping.")
                continue

            # Tokenize with error handling and truncation safety
            try:
                encoding = tokenizer(
                    seq,
                    truncation=True,  # Enable truncation for safety
                    padding=False,
                    return_tensors="pt"
                )
            except Exception as e:
                logger.error(f"Tokenization failed: {e}")
                continue

            # Verify encoding has expected keys
            if "input_ids" not in encoding:
                logger.error("Tokenizer didn't return input_ids")
                continue

            # Get tokenized length
            seq_length = encoding["input_ids"].shape[1]

            # Continue with evaluation...
            with torch.no_grad():
                try:
                    # Create labels for perplexity calculation
                    labels = encoding["input_ids"].clone()
                    encoding["labels"] = labels

                    # Ensure tensors are on CPU
                    encoding = {k: v.to('cpu') for k, v in encoding.items()}

                    # Run inference
                    outputs = cpu_model(**encoding)

                    # Record success and perplexity
                    perplexity = torch.exp(outputs.loss).item() if hasattr(outputs, 'loss') else None
                    results.append({
                        "length": seq_length,
                        "success": True,
                        "perplexity": perplexity,
                        "error": None
                    })

                    logger.info(f"✓ Successfully processed sequence of length {seq_length}")
                    if perplexity:
                        logger.info(f"  Perplexity: {perplexity:.4f}")

                except Exception as e:
                    logger.error(f"Model inference failed: {e}")
                    results.append({
                        "length": seq_length,
                        "success": False,
                        "perplexity": None,
                        "error": str(e)
                    })
        except Exception as outer_e:
            logger.error(f"Unexpected error in sequence testing: {outer_e}")

    # Summarize results
    success_lengths = [r["length"] for r in results if r["success"]]
    if success_lengths:
        logger.info(f"\nSuccessfully handled sequences from {min(success_lengths)} to {max(success_lengths)} tokens")
    else:
        logger.info("\nFailed to handle any test sequences")

    # Clean up memory
    del cpu_model
    torch.cuda.empty_cache()

    return results


def get_optimal_batch_size(seq_length, available_memory_gb, base_batch_size=16):
    """
    Calculate optimal batch size based on sequence length and available memory.

    Args:
        seq_length: Length of sequences
        available_memory_gb: Available GPU memory in GB
        base_batch_size: Base batch size for standard sequences

    Returns:
        Optimal batch size
    """
    # Memory usage approximately scales quadratically with sequence length
    # These are empirically determined values - would need to be adjusted based on model size
    memory_factor = (seq_length / 512) ** 1.5

    # Calculate adjusted batch size
    adjusted_batch_size = max(1, int(base_batch_size / memory_factor))

    # Further adjust based on available memory
    memory_scale = available_memory_gb / 16.0  # Assuming 16GB as reference memory
    memory_adjusted = max(1, int(adjusted_batch_size * memory_scale))

    return memory_adjusted


def calculate_perplexity(model, tokenizer, texts, device):
    """
    Calculate perplexity on a set of texts.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer to use
        texts: List of texts to evaluate
        device: Device to run evaluation on

    Returns:
        Perplexity value
    """
    model.eval()
    total_loss = 0.0
    total_length = 0

    with torch.no_grad():
        for text in texts:
            # Tokenize text
            encodings = tokenizer(text, return_tensors="pt").to(device)

            # Create labels for loss calculation
            labels = encodings.input_ids.clone()

            # Forward pass
            outputs = model(**encodings, labels=labels)

            # Add loss weighted by sequence length (excluding padding)
            seq_length = (encodings.attention_mask == 1).sum().item()
            total_loss += outputs.loss.item() * seq_length
            total_length += seq_length

    # Calculate perplexity from average loss
    avg_loss = total_loss / total_length
    perplexity = torch.exp(torch.tensor(avg_loss))

    return perplexity.item()


def log_gpu_memory_usage():
    """Log GPU memory usage for all available GPUs."""
    if not torch.cuda.is_available():
        logger.info("No CUDA devices available")
        return

    logger.info("GPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)

        device_name = torch.cuda.get_device_name(i)
        logger.info(f"  GPU {i} ({device_name}):")
        logger.info(f"    Allocated: {allocated:.2f} GB")
        logger.info(f"    Reserved:  {reserved:.2f} GB")
        logger.info(f"    Total:     {total:.2f} GB")
        logger.info(f"    Free:      {total - reserved:.2f} GB")


def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing for memory efficiency.

    Args:
        model: Model to modify

    Returns:
        Modified model
    """
    # Check if model has gradient checkpointing attribute
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled built-in gradient checkpointing")
    # For BERT models
    elif hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        model.bert.encoder.gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing for BERT encoder")
    else:
        logger.warning("Could not enable gradient checkpointing - not supported by model")

    return model