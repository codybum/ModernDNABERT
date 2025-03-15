"""
Genomic tokenizer implementation using SentencePiece for DNA sequences.
This module provides a tokenizer compatible with the HuggingFace ecosystem.
"""

import os
import logging
import shutil
from typing import List, Optional, Dict, Union, Tuple

import torch
from transformers import PreTrainedTokenizer
import os
import logging
import shutil
from typing import List, Optional, Dict, Union, Tuple

import torch
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

class GenomicTokenizer(PreTrainedTokenizerFast):
    """
    Genomic tokenizer based on HuggingFace Fast Tokenizer for DNA sequences,
    compatible with HuggingFace's ecosystem.
    """
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            tokenizer_file=None,
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
            **kwargs
    ):
        # Initialize parent class with tokenizer_file
        super().__init__(
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )

    def tokenize_with_safety(self, text, max_length=None, truncation=True, **kwargs):
        """Tokenize with explicit length safeguards to prevent indexing errors."""
        # Filter out invalid characters for genomic data
        if text:
            text = text.upper()
            text = ''.join(c for c in text if c in 'ATGC')

        # Check if text is empty after filtering
        if not text:
            return {"input_ids": torch.zeros(0, dtype=torch.long),
                    "attention_mask": torch.zeros(0, dtype=torch.long)}

        # Apply length limit if needed
        if max_length and len(text) > max_length and truncation:
            logger.warning(f"Truncating sequence from {len(text)} to {max_length}")
            text = text[:max_length]

        # Proceed with normal tokenization using the parent class
        return self(text, **kwargs)

    # Add genomic-specific methods
    def tokenize_genomic_sequence(self, sequence: str) -> List[str]:
        """Tokenize a DNA sequence with validation."""
        # Validate sequence
        sequence = sequence.upper()
        invalid_chars = set(sequence) - set('ATGC')
        if invalid_chars:
            logger.warning(f"Sequence contains invalid characters: {invalid_chars}")
            # Filter out non-ATGC characters
            sequence = ''.join(c for c in sequence if c in 'ATGC')

        return self.tokenize(sequence)

def _ensure_special_tokens(tokenizer):
    """
    Ensure special tokens are properly configured for the tokenizer.
    This function works with PreTrainedTokenizerFast and doesn't need SentencePiece.
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