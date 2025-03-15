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

from data.data_collator import GenomicMLMDataCollator

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
    Test model on sequences of different lengths using the accelerator device.

    Args:
        accelerator: Accelerator instance
        model: Model to test
        tokenizer: Tokenizer to use
        test_sequences: List of test sequences

    Returns:
        List of test results
    """
    # Use the model on its current device through accelerator
    model.eval()

    logger.info("\n" + "=" * 50)
    logger.info(f"RUNNING LENGTH EXTRAPOLATION TEST ON {str(accelerator.device).upper()}")
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

                    # Move tensors to accelerator device
                    encoding = {k: v.to(accelerator.device) for k, v in encoding.items()}

                    # Run inference
                    outputs = model(**encoding)

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

                except RuntimeError as e:
                    # Handle CUDA OOM specifically
                    if "CUDA out of memory" in str(e):
                        logger.warning(f"GPU OOM for sequence length {seq_length}, falling back to CPU")
                        # Try on CPU as fallback
                        try:
                            # Move to CPU for this sequence only
                            cpu_encoding = {k: v.to('cpu') for k, v in encoding.items()}

                            cpu_model = accelerator.unwrap_model(model).to('cpu')
                            cpu_outputs = cpu_model(**cpu_encoding)

                            # Calculate perplexity and record
                            cpu_perplexity = torch.exp(cpu_outputs.loss).item() if hasattr(cpu_outputs,
                                                                                           'loss') else None
                            results.append({
                                "length": seq_length,
                                "success": True,
                                "perplexity": cpu_perplexity,
                                "error": None,
                                "used_fallback": True
                            })
                            logger.info(f"✓ Successfully processed sequence of length {seq_length} on CPU fallback")

                            # Move model back to original device
                            accelerator.unwrap_model(model).to(accelerator.device)
                        except Exception as cpu_e:
                            logger.error(f"CPU fallback also failed: {cpu_e}")
                            results.append({
                                "length": seq_length,
                                "success": False,
                                "perplexity": None,
                                "error": str(e) + " | CPU fallback: " + str(cpu_e)
                            })
                    else:
                        logger.error(f"Model inference failed: {e}")
                        results.append({
                            "length": seq_length,
                            "success": False,
                            "perplexity": None,
                            "error": str(e)
                        })
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


def calculate_perplexity(model, tokenizer, texts, accelerator):
    """
    Calculate perplexity on a set of texts.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer to use
        texts: List of texts to evaluate
        accelerator: Accelerator instance for device management

    Returns:
        Perplexity value
    """
    model.eval()
    total_loss = 0.0
    total_length = 0

    with torch.no_grad():
        for text in texts:
            # Tokenize text
            encodings = tokenizer(text, return_tensors="pt").to(accelerator.device)

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


def test_tokenizer_oov_handling(tokenizer):
    """
    Test that the tokenizer properly handles OOV tokens.
    Updated to work with PreTrainedTokenizerFast without SentencePiece.
    """
    logger.info("Testing tokenizer OOV handling...")

    # Get max valid token ID
    max_valid_id = tokenizer.vocab_size - 1

    # Test normal tokens
    for base in ['A', 'T', 'G', 'C']:
        # FastTokenizer uses convert_tokens_to_ids, not _convert_token_to_id
        token_id = tokenizer.convert_tokens_to_ids(base)
        logger.info(f"Token: {base}, ID: {token_id}")
        assert token_id <= max_valid_id, f"Token ID {token_id} exceeds max valid ID {max_valid_id}"

    # Test likely OOV tokens (unusual sequences)
    unusual_tokens = ['ZZZZZ', 'NNNNN', 'XXXXX']
    for token in unusual_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"OOV Token: {token}, ID: {token_id}")
        # Check if it's the unk_token_id or at least a valid token ID
        expected_id = tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else 0
        assert token_id == expected_id or token_id <= max_valid_id, f"OOV token '{token}' mapped to invalid ID {token_id}"

    logger.info("Tokenizer OOV handling test passed!")
    return True

def setup_mlm_data_collator(tokenizer, mlm_probability=0.15, model=None, max_seq_length=512):
    """
    Set up a data collator for masked language modeling with fallbacks.
    Updated to work with PreTrainedTokenizerFast without SentencePiece.

    Args:
        tokenizer: The tokenizer to use
        mlm_probability: Probability of masking a token
        model: Optional model reference to access its config for mask_token_id
        max_seq_length: Maximum sequence length

    Returns:
        A data collator for masked language modeling
    """
    logger = logging.getLogger(__name__)

    # Check if mask token exists
    if not hasattr(tokenizer, 'mask_token') or tokenizer.mask_token is None:
        logger.warning("Tokenizer has no mask token defined!")

        # Add a mask token if needed
        if not hasattr(tokenizer, 'mask_token') or tokenizer.mask_token is None:
            logger.info("Adding mask token to tokenizer")
            # Use a safe approach to add the token
            special_tokens_dict = {'mask_token': '<mask>'}
            tokenizer.add_special_tokens(special_tokens_dict)
            logger.info(f"Added mask token: {tokenizer.mask_token} with ID: {tokenizer.mask_token_id}")

    # Ensure mask token ID is valid
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        vocab_size = tokenizer.vocab_size
        if tokenizer.mask_token_id >= vocab_size:
            logger.warning(f"Mask token ID {tokenizer.mask_token_id} exceeds vocab size {vocab_size}")
            safe_id = min(1, vocab_size - 1)
            logger.info(f"Using safer mask token ID: {safe_id}")
            # Try to update tokenizer's mask_token_id if possible
            if hasattr(tokenizer, 'mask_token_id'):
                old_id = tokenizer.mask_token_id
                tokenizer.mask_token_id = safe_id
                logger.info(f"Updated mask_token_id in tokenizer from {old_id} to {safe_id}")

    # Create the data collator
    try:
        logger.info(
            f"Creating MLM data collator with mask token: {tokenizer.mask_token} "
            f"(ID: {tokenizer.mask_token_id}), max_seq_length: {max_seq_length}")

        data_collator = GenomicMLMDataCollator(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            max_seq_length=max_seq_length
        )

        # Store model reference
        if model is not None:
            data_collator.model = model

        return data_collator

    except ValueError as e:
        if "mask token" in str(e).lower():
            logger.error(f"Failed to create MLM data collator: {e}")
            logger.info("Attempting to create a safe version with custom mask handling")

            # Create safe custom version
            class SafeGenomicMLMDataCollator(GenomicMLMDataCollator):
                def __post_init__(self):
                    # Skip the parent class check for mask token
                    pass

                def completely_safe_mask_tokens(self, inputs):
                    """Safe implementation of token masking for MLM."""
                    if inputs.numel() == 0 or inputs.dim() != 2:
                        logger.error(f"Invalid inputs shape: {inputs.shape}")
                        return inputs, torch.full_like(inputs, -100)

                    # Get vocabulary size for verification
                    vocab_size = getattr(self.tokenizer, 'vocab_size', 100)

                    # Use model's mask token ID if available, otherwise fall back
                    if hasattr(self, 'model') and hasattr(self.model.config, 'mask_token_id'):
                        mask_token_id = self.model.config.mask_token_id
                    elif hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
                        mask_token_id = self.tokenizer.mask_token_id
                    else:
                        mask_token_id = 1  # Safe default

                    # Ensure ID is in range
                    if mask_token_id >= vocab_size:
                        logger.warning(f"Mask token ID {mask_token_id} exceeds vocab size {vocab_size}")
                        mask_token_id = min(1, vocab_size - 1)  # Use safer ID

                    labels = inputs.clone()

                    # Create probability matrix
                    probability_matrix = torch.full(labels.shape, self.mlm_probability)

                    # Create special tokens mask
                    special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)

                    # Mark pad tokens as special
                    if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                        special_tokens_mask = inputs.eq(self.tokenizer.pad_token_id)

                    # Don't mask special tokens
                    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

                    # Get indices to mask
                    masked_indices = torch.bernoulli(probability_matrix).bool()

                    # Set labels for unmasked tokens to -100 so they're ignored in loss
                    labels[~masked_indices] = -100

                    # Replace masked tokens with mask token
                    inputs[masked_indices] = mask_token_id

                    return inputs, labels

            collator = SafeGenomicMLMDataCollator(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=mlm_probability,
                max_seq_length=max_seq_length
            )

            # Store model reference
            if model is not None:
                collator.model = model

            return collator
        else:
            # Re-raise if it's not related to mask token
            raise

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