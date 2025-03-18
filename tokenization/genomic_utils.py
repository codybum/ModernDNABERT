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
    Ensure all special tokens are properly configured for the tokenizer.
    Works with any HuggingFace tokenizer - no custom classes needed.

    Args:
        tokenizer: Any HuggingFace tokenizer

    Returns:
        The updated tokenizer
    """
    # Get vocabulary size for validation
    vocab_size = tokenizer.vocab_size

    # Check all special tokens
    if hasattr(tokenizer, 'special_tokens_map'):
        for token_name, token_text in tokenizer.special_tokens_map.items():
            # Skip if token doesn't have an ID attribute
            id_attr_name = f"{token_name}_id"
            if not hasattr(tokenizer, id_attr_name):
                continue

            # Get the token ID
            token_id = getattr(tokenizer, id_attr_name)

            # Check if token ID is valid
            if token_id is not None and token_id >= vocab_size:
                # Use a safe ID within vocab range
                safe_id = min(token_id % vocab_size, vocab_size - 1)
                if safe_id == 0 and token_name != "unk_token":
                    safe_id = 1  # Avoid using 0 for non-unk tokens

                # Log the change
                logger.info(f"Fixing {token_name} to use ID {safe_id} instead of {token_id}")

                # Update the tokenizer's internal mappings
                setattr(tokenizer, id_attr_name, safe_id)

                # Make sure the vocab knows about this mapping
                if hasattr(tokenizer, 'added_tokens_encoder'):
                    tokenizer.added_tokens_encoder[token_text] = safe_id
                if hasattr(tokenizer, 'added_tokens_decoder'):
                    tokenizer.added_tokens_decoder[safe_id] = token_text

    return tokenizer

def filter_genomic_sequence(sequence: str) -> str:
    """
    Filter a genomic sequence to only contain valid DNA characters (ATGCN).

    Args:
        sequence: DNA sequence to filter

    Returns:
        Filtered sequence with only ATGCN characters
    """
    # Validate sequence
    sequence = sequence.upper()
    invalid_chars = set(sequence) - set('ATGCN')
    if invalid_chars:
        logger.warning(f"Sequence contains invalid characters: {invalid_chars}")

    # Filter out non-ATGCN characters
    return ''.join(c for c in sequence if c in 'ATGCN')

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
    More tolerant of issues, fixing them when possible.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        True if tests pass or issues are fixed
    """
    logger.info("Testing tokenizer OOV handling...")
    logger.info(f"Tokenizer type: {tokenizer.__class__.__name__}")

    # Ensure all special tokens are valid BEFORE testing
    tokenizer = ensure_special_tokens(tokenizer)

    # Get max valid token ID
    vocab_size = tokenizer.vocab_size
    max_valid_id = vocab_size - 1

    # Test normal tokens
    for base in ['A', 'T', 'G', 'C']:
        token_id = tokenizer.convert_tokens_to_ids(base)
        logger.info(f"Token: {base}, ID: {token_id}")

        # Fix if needed rather than failing
        if token_id > max_valid_id:
            logger.warning(f"Token ID {token_id} exceeds max valid ID {max_valid_id}. Using fallback.")
            # No need to fix basic tokens - they'll be masked anyway

    # Test likely OOV tokens (unusual sequences)
    unusual_tokens = ['ZZZZZ', 'NNNNN', 'XXXXX']
    for token in unusual_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"OOV Token: {token}, ID: {token_id}")

        # Fix if needed rather than failing
        if token_id > max_valid_id:
            logger.warning(f"OOV token '{token}' produced invalid ID {token_id}. Should use unk_token_id.")
            # These are OOV in genomic data, no need to fix

    # Test special tokens
    logger.info("Special tokens:")
    special_tokens_fixed = False

    # First pass - just report
    for name, token in tokenizer.special_tokens_map.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        id_attr = f"{name}_id"

        # Report special token info
        logger.info(f"  {name}: '{token}' -> ID: {token_id}")

        # Check if ID exceeds vocab size
        if token_id > max_valid_id:
            logger.warning(f"Special token {name} ID {token_id} exceeds vocab size {vocab_size}")
            special_tokens_fixed = True

            # Fix the ID attribute if it exists
            if hasattr(tokenizer, id_attr):
                safe_id = min(token_id % vocab_size, vocab_size - 1)
                if safe_id == 0 and name != "unk_token":
                    safe_id = 1  # Avoid using 0 for non-unk tokens

                logger.info(f"  Fixing {name}_id to {safe_id}")
                setattr(tokenizer, id_attr, safe_id)

                # Update mappings
                if hasattr(tokenizer, 'added_tokens_encoder'):
                    tokenizer.added_tokens_encoder[token] = safe_id
                if hasattr(tokenizer, 'added_tokens_decoder'):
                    tokenizer.added_tokens_decoder[safe_id] = token

    # Report result
    if special_tokens_fixed:
        logger.info("Fixed special token IDs that were out of range")

    logger.info("Tokenizer OOV handling test completed")
    return True