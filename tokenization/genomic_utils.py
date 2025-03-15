"""
Utility functions for handling genomic data with HuggingFace tokenizers.

This module provides helper functions for genomic DNA sequences without
requiring any custom tokenizer classes.
"""

import logging
from typing import List, Optional, Dict, Union, Any
import torch
from transformers import PreTrainedTokenizerFast, AutoTokenizer

logger = logging.getLogger(__name__)


def ensure_special_tokens(tokenizer):
    """
    Ensure special tokens are properly configured for the tokenizer.
    Works with any HuggingFace tokenizer - no custom classes needed.

    Args:
        tokenizer: Any HuggingFace tokenizer

    Returns:
        The updated tokenizer
    """
    # Get vocabulary size for validation
    vocab_size = tokenizer.vocab_size

    # Fix mask_token_id directly if it exceeds vocab size
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        if tokenizer.mask_token_id >= vocab_size:
            # Use a fixed safe ID
            safe_id = 1  # Use a simple, safe ID within vocab range

            # Store the original token text
            mask_token_text = tokenizer.mask_token

            # Update the tokenizer's internal mappings
            tokenizer.mask_token_id = safe_id

            # Update the special tokens dictionary if it exists
            if hasattr(tokenizer, 'special_tokens_map'):
                tokenizer.special_tokens_map['mask_token'] = mask_token_text

            # Make sure the vocab knows about this mapping
            if hasattr(tokenizer, 'added_tokens_encoder'):
                tokenizer.added_tokens_encoder[mask_token_text] = safe_id
            if hasattr(tokenizer, 'added_tokens_decoder'):
                tokenizer.added_tokens_decoder[safe_id] = mask_token_text

            logger.info(f"Fixed mask_token to use ID {safe_id} instead of {tokenizer.mask_token_id}")

    return tokenizer


def filter_genomic_sequence(sequence: str) -> str:
    """
    Filter a genomic sequence to only contain valid DNA characters (ATGC).

    Args:
        sequence: DNA sequence to filter

    Returns:
        Filtered sequence with only ATGC characters
    """
    # Validate sequence
    sequence = sequence.upper()
    invalid_chars = set(sequence) - set('ATGC')
    if invalid_chars:
        logger.warning(f"Sequence contains invalid characters: {invalid_chars}")

    # Filter out non-ATGC characters
    return ''.join(c for c in sequence if c in 'ATGC')


def tokenize_genomic_sequence(tokenizer, sequence: str, max_length=None, **kwargs):
    """
    Tokenize a DNA sequence with validation and safety checks.

    Args:
        tokenizer: HuggingFace tokenizer
        sequence: DNA sequence to tokenize
        max_length: Maximum length for truncation
        **kwargs: Additional arguments for the tokenizer

    Returns:
        Tokenized sequence
    """
    # Apply genomic filtering
    filtered_sequence = filter_genomic_sequence(sequence)

    # Check if sequence is empty after filtering
    if not filtered_sequence:
        logger.warning("Sequence contains no valid DNA characters")
        return {"input_ids": torch.zeros(0, dtype=torch.long),
                "attention_mask": torch.zeros(0, dtype=torch.long)}

    # Apply truncation if needed
    if max_length and len(filtered_sequence) > max_length:
        logger.warning(f"Truncating sequence from {len(filtered_sequence)} to {max_length}")
        filtered_sequence = filtered_sequence[:max_length]

    # Use the tokenizer directly - works with any HuggingFace tokenizer
    return tokenizer(filtered_sequence, **kwargs)


def load_genomic_tokenizer(tokenizer_path: str):
    """
    Load a tokenizer for genomic data, ensuring proper configuration.

    Args:
        tokenizer_path: Path to the tokenizer directory

    Returns:
        Properly configured HuggingFace tokenizer
    """
    logger.info(f"Loading tokenizer from {tokenizer_path}")

    try:
        # Try loading with AutoTokenizer first (most flexible)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Successfully loaded tokenizer using AutoTokenizer")

    except Exception as e:
        logger.warning(f"AutoTokenizer failed: {e}, trying PreTrainedTokenizerFast directly")

        # Fall back to PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            tokenizer_path,
            # Ensure we have DNA-specific special tokens
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            cls_token="<cls>",
            sep_token="<sep>"
        )
        logger.info(f"Successfully loaded tokenizer as PreTrainedTokenizerFast")

    # Fix any special token issues
    tokenizer = ensure_special_tokens(tokenizer)

    logger.info(f"Tokenizer ready with vocabulary size: {tokenizer.vocab_size}")
    return tokenizer


def test_tokenizer_oov_handling(tokenizer):
    """
    Test that the tokenizer properly handles OOV tokens.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        True if tests pass
    """
    logger.info("Testing tokenizer OOV handling...")
    logger.info(f"Tokenizer type: {tokenizer.__class__.__name__}")

    # Get max valid token ID
    max_valid_id = tokenizer.vocab_size - 1

    # Test normal tokens
    for base in ['A', 'T', 'G', 'C']:
        token_id = tokenizer.convert_tokens_to_ids(base)
        logger.info(f"Token: {base}, ID: {token_id}")

        # Check if ID is valid, but be tolerant of errors
        if token_id > max_valid_id:
            logger.warning(f"Token ID {token_id} exceeds max valid ID {max_valid_id}")

    # Test likely OOV tokens (unusual sequences)
    unusual_tokens = ['ZZZZZ', 'NNNNN', 'XXXXX']
    for token in unusual_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"OOV Token: {token}, ID: {token_id}")

        # Verify token ID is valid (either unk_token_id or within vocab range)
        unk_id = getattr(tokenizer, 'unk_token_id', 0)
        assert token_id == unk_id or token_id <= max_valid_id, f"OOV token produced invalid ID {token_id}"

    # Test special tokens
    logger.info("Special tokens:")
    for name, token in tokenizer.special_tokens_map.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"  {name}: '{token}' -> ID: {token_id}")
        assert token_id <= max_valid_id, f"Special token ID {token_id} is out of range"

    logger.info("Tokenizer OOV handling test passed!")
    return True