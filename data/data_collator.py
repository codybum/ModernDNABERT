"""
Genomic dataset and data collator implementations.

This module provides dataset classes for genomic sequences and data collators
for masked language modeling with proper handling of long sequences.
"""

import random
import logging
import torch
from torch.utils.data import Dataset
from typing import List, Optional

from transformers import DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)


def reverse_complement(sequence):
    """Generate the reverse complement of a DNA sequence."""
    complement_map = {
        'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
        'a': 't', 't': 'a', 'g': 'c', 'c': 'g',
        'N': 'N', 'n': 'n',
        # Extended ambiguous nucleotides
        'R': 'Y', 'Y': 'R', 'M': 'K', 'K': 'M',
        'S': 'S', 'W': 'W', 'H': 'D', 'B': 'V',
        'V': 'B', 'D': 'H',
        'r': 'y', 'y': 'r', 'm': 'k', 'k': 'm',
        's': 's', 'w': 'w', 'h': 'd', 'b': 'v',
        'v': 'b', 'd': 'h'
    }

    return ''.join(complement_map.get(base, base) for base in reversed(sequence))

class GenomicDataset(Dataset):
    """
    Dataset for genomic sequences with variable length handling.

    This dataset loads genomic sequences from FASTA files, chunks them into
    appropriate sizes, and provides tokenized outputs for training.
    """

    def __init__(
            self,
            file_paths: List[str],
            tokenizer,
            pre_training_length: int = 512,
            max_inference_length: Optional[int] = None,
            mlm_probability: float = 0.15,
            chunk_size: int = 2000,
            stride: int = 1000,
            sample_long_sequences: bool = True,
            max_safe_sequence_length: int = 50000,
            use_reverse_complement: bool = True,  # New parameter
    ):
        """
        Initialize the dataset.

        Args:
            file_paths: List of FASTA file paths
            tokenizer: Tokenizer to use
            pre_training_length: Sequence length during pre-training
            max_inference_length: Maximum sequence length for inference
            mlm_probability: Probability of masking a token for MLM
            chunk_size: Base size of sequence chunks
            stride: Stride for overlapping chunks
            sample_long_sequences: Whether to include longer sequences for extrapolation
            max_safe_sequence_length: Maximum safe sequence length for processing
            use_reverse_complement: Whether to include reverse complements for augmentation
        """
        self.tokenizer = tokenizer
        self.pre_training_length = pre_training_length
        self.max_inference_length = max_inference_length
        self.mlm_probability = mlm_probability
        self.chunk_size = chunk_size
        self.stride = stride
        self.sample_long_sequences = sample_long_sequences
        self.max_safe_sequence_length = max_safe_sequence_length
        self.use_reverse_complement = use_reverse_complement  # Store the new parameter

        # Load and prepare sequences
        self.sequences = self._load_sequences(file_paths)
        logger.info(f"Loaded {len(self.sequences)} sequences")

        # Prepare chunks with variable lengths
        self.chunks = self._prepare_chunks()
        self.sequence_lengths = [len(chunk) for chunk in self.chunks]

        logger.info(f"Created {len(self.chunks)} chunks for training")
        if self.chunks:
            logger.info(f"Length distribution: min={min(self.sequence_lengths)}, "
                        f"max={max(self.sequence_lengths)}, "
                        f"avg={sum(self.sequence_lengths) / len(self.sequence_lengths):.1f}")

    def _load_sequences(self, file_paths: List[str]) -> List[str]:
        """
        Load genomic sequences from files - supports FASTA format and plain text.
        Now accepts ATGCN rather than just ATGC.
        """
        sequences = []
        total_files = 0
        fasta_files = 0

        for file_path in file_paths:
            logger.info(f"Loading sequences from {file_path}")
            total_files += 1

            with open(file_path, 'r') as f:
                # Detect format
                peek = [next(f, '').strip() for _ in range(5)]
                f.seek(0)  # Reset to beginning
                is_fasta = any(line.startswith('>') for line in peek if line)

                if is_fasta:
                    fasta_files += 1
                    logger.info(f"Detected FASTA format for {file_path}")
                    current_seq = ""
                    for line in f:
                        line = line.strip()
                        # Handle FASTA headers
                        if line.startswith('>'):
                            if current_seq:
                                sequences.append(current_seq)
                                current_seq = ""
                            continue
                        # Updated: Accept A, T, G, C, and N sequences
                        if set(line.upper()) <= set('ATGCN'):
                            current_seq += line.upper()

                    # Add the last sequence if exists
                    if current_seq:
                        sequences.append(current_seq)
                else:
                    logger.info(f"Detected plain text format for {file_path}")
                    # Process as plain text with one sequence per line
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            # Updated: Filter to keep A, T, G, C, and N characters
                            filtered_seq = ''.join(c for c in line.upper() if c in 'ATGCN')
                            if filtered_seq:
                                sequences.append(filtered_seq)

        logger.info(
            f"Loaded {len(sequences)} sequences from {total_files} files ({fasta_files} FASTA format, {total_files - fasta_files} plain text)")
        return sequences

    def _prepare_chunks(self) -> List[str]:
        """
        Split sequences into chunks with variable lengths, including reverse complements.
        Handles sequences shorter than chunk_size by including them directly if they meet
        the minimum length requirement.

        Returns:
            List of chunks
        """
        chunks = []

        # Track statistics for logging
        total_sequences = len(self.sequences)
        truncated_count = 0
        rc_count = 0  # Counter for reverse complements
        short_seq_count = 0  # Counter for sequences shorter than chunk_size

        for sequence in self.sequences:
            # Truncate very long sequences for safety
            if len(sequence) > self.max_safe_sequence_length:
                truncated_count += 1
                sequence = sequence[:self.max_safe_sequence_length]

            # Handle sequences shorter than chunk_size
            if len(sequence) < self.chunk_size:
                if len(sequence) >= 100:  # Keep the same minimum size check
                    chunks.append(sequence)
                    short_seq_count += 1

                    # Add reverse complement if enabled
                    if self.use_reverse_complement:
                        rc_seq = reverse_complement(sequence)
                        chunks.append(rc_seq)
                        rc_count += 1
            else:
                # Standard chunks for primary training - no change for longer sequences
                for i in range(0, len(sequence) - self.chunk_size + 1, self.stride):
                    chunk = sequence[i:i + self.chunk_size]
                    if len(chunk) >= 100:  # Only keep reasonably sized chunks
                        chunks.append(chunk)

                        # Add reverse complement if enabled
                        if self.use_reverse_complement:
                            rc_chunk = reverse_complement(chunk)
                            chunks.append(rc_chunk)
                            rc_count += 1

                # Optionally add longer chunks for extrapolation training
                if self.sample_long_sequences and len(sequence) > self.chunk_size * 2:
                    # Add a few longer chunks - up to 2-4x the normal chunk size
                    for _ in range(min(2, len(sequence) // (self.chunk_size * 2))):  # Add just a few longer samples
                        long_size = random.randint(int(self.chunk_size * 1.5),
                                                   min(len(sequence), int(self.chunk_size * 4)))
                        if len(sequence) > long_size:
                            start = random.randint(0, len(sequence) - long_size)
                            long_chunk = sequence[start:start + long_size]
                            chunks.append(long_chunk)

                            # Add reverse complement for long chunks too
                            if self.use_reverse_complement:
                                rc_long_chunk = reverse_complement(long_chunk)
                                chunks.append(rc_long_chunk)
                                rc_count += 1

        # Log statistics
        if truncated_count > 0:
            logger.warning(
                f"Truncated {truncated_count} out of {total_sequences} sequences that exceeded maximum safe length of {self.max_safe_sequence_length}")

        if short_seq_count > 0:
            logger.info(f"Included {short_seq_count} sequences shorter than chunk_size but meeting minimum length")

        if self.use_reverse_complement and rc_count > 0:
            logger.info(f"Added {rc_count} reverse complement sequences for augmentation")

        return chunks

    def __len__(self):
        """Return the number of chunks."""
        return len(self.chunks)

    def __getitem__(self, idx):
        """Get a tokenized sequence with appropriate length handling."""
        try:
            sequence = self.chunks[idx]

            # Use pre_training_length as the guide for maximum sequence length
            max_seq_length = self.pre_training_length

            # Apply sensible limit if sequence is extremely long
            very_long_threshold = min(50000, self.max_safe_sequence_length)
            if len(sequence) > very_long_threshold:
                start = random.randint(0, len(sequence) - very_long_threshold)
                sequence = sequence[start:start + very_long_threshold]
                logger.debug(f"Truncated very long sequence from original length to {very_long_threshold}")

            # Filter the sequence to include ATGCN
            sequence = ''.join(c for c in sequence.upper() if c in 'ATGCN')

            # Check if sequence is empty after filtering
            if not sequence:
                logger.warning("Empty sequence after filtering. Using dummy sequence.")
                sequence = "ATGC" * 32  # Use a dummy sequence

            # PROPER FIX: Don't request PyTorch tensors from the tokenizer
            # Let the collator handle tensor conversion with proper batching
            encoding = self.tokenizer(
                sequence,
                truncation=True,
                padding='max_length',
                max_length=max_seq_length,
                return_tensors=None  # Return Python lists instead of tensors
            )

            # Convert to tensors manually with correct shape
            encoding = {k: torch.tensor(v) for k, v in encoding.items()}

            # Ensure we have token_type_ids
            if 'token_type_ids' not in encoding:
                encoding['token_type_ids'] = torch.zeros_like(encoding['input_ids'])

            # Verify and clamp token IDs
            if 'input_ids' in encoding:
                vocab_size = self.tokenizer.vocab_size
                max_id = encoding['input_ids'].max().item()

                if max_id >= vocab_size:
                    logger.warning(f"Found token ID {max_id} exceeding vocab size {vocab_size}")
                    # Replace out-of-range IDs with UNK token ID
                    mask = encoding['input_ids'] >= vocab_size
                    unk_token_id = getattr(self.tokenizer, 'unk_token_id', 0)
                    encoding['input_ids'] = torch.where(
                        mask,
                        torch.tensor(unk_token_id, device=encoding['input_ids'].device,
                                     dtype=encoding['input_ids'].dtype),
                        encoding['input_ids']
                    )

            return encoding

        except Exception as e:
            logger.error(f"Error in dataset.__getitem__: {e}")
            return self._get_fallback_encoding(max_seq_length)

    def _get_fallback_encoding(self, length):
        """
        Create a fallback encoding when tokenization fails.

        Args:
            length: Length of the fallback encoding

        Returns:
            Fallback encoding
        """
        return {
            "input_ids": torch.ones(length, dtype=torch.long),
            "attention_mask": torch.ones(length, dtype=torch.long),
            "token_type_ids": torch.zeros(length, dtype=torch.long)
        }



class GenomicMLMDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator for genomic MLM that handles variable sequence lengths
    and ensures proper token_type_ids.
    """

    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15, max_seq_length=512, model=None):
        """Initialize with max_seq_length parameter"""
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.max_seq_length = max_seq_length
        self.model = model  # Store model reference

    def __call__(self, examples):
        """
        Create batches with consistent dimensions and ensure all token IDs are in valid range.
        """
        # Check and log length information
        input_lengths = [len(example["input_ids"]) for example in examples]
        # Use self.max_seq_length instead of hardcoded 512
        max_length = min(max(input_lengths), self.max_seq_length)

        # Warn if batch has inconsistent lengths
        if max(input_lengths) != min(input_lengths):
            logger.warning(f"Batch has inconsistent lengths: min={min(input_lengths)}, max={max(input_lengths)}")

        # Prepare batch
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }

        # Get vocabulary size for validation
        vocab_size = getattr(self.tokenizer, 'vocab_size', 100)

        # Process each example with explicit dimension control
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0
        unk_token_id = self.tokenizer.unk_token_id if hasattr(self.tokenizer, 'unk_token_id') else 0

        for example in examples:
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]

            # Validate input_ids are within vocab size range
            if input_ids.max() >= vocab_size:
                logger.warning(f"Found token ID {input_ids.max().item()} exceeding vocab size {vocab_size}")
                # Clip token IDs to valid range (0 to vocab_size-1)
                invalid_mask = input_ids >= vocab_size
                input_ids = torch.where(invalid_mask,
                                        torch.tensor(unk_token_id, device=input_ids.device, dtype=input_ids.dtype),
                                        input_ids)

            # Hard truncation for safety
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]

            # Pad to consistent length
            padding_length = max_length - len(input_ids)
            token_type_ids = torch.zeros_like(input_ids)

            if padding_length > 0:
                # Add padding
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), pad_token_id, dtype=torch.long, device=input_ids.device)
                ], dim=0)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=torch.long, device=attention_mask.device)
                ], dim=0)
                token_type_ids = torch.cat([
                    token_type_ids,
                    torch.zeros(padding_length, dtype=torch.long, device=token_type_ids.device)
                ], dim=0)

            # Verification check
            assert len(input_ids) == max_length, f"Expected length {max_length}, got {len(input_ids)}"

            # Final validation of token IDs
            if input_ids.max() >= vocab_size:
                logger.warning(
                    f"After processing, token ID {input_ids.max().item()} still exceeds vocab size {vocab_size}")
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["token_type_ids"].append(token_type_ids)

        # Create tensors from lists
        batch = {k: torch.stack(v) for k, v in batch.items()}

        # Apply MLM with completely rewritten safe masking
        inputs, labels = self.completely_safe_mask_tokens(batch["input_ids"])
        batch["input_ids"] = inputs
        batch["labels"] = labels

        return batch

    def completely_safe_mask_tokens(self, inputs):
        """
        Completely rewritten safe implementation of token masking for MLM.

        Args:
            inputs: Input token IDs

        Returns:
            Tuple of masked inputs and labels
        """
        if inputs.numel() == 0 or inputs.dim() != 2:
            logger.error(f"Invalid inputs shape: {inputs.shape}")
            return inputs, torch.full_like(inputs, -100)

        # Get vocabulary size for additional verification
        vocab_size = getattr(self.tokenizer, 'vocab_size', 100)

        # CRITICAL FIX: Make sure no token ID exceeds vocabulary size
        max_id = inputs.max().item()
        if max_id >= vocab_size:
            logger.warning(f"Batch contains token ID {max_id} exceeding vocab size {vocab_size}")
            # Clamp all tokens to valid range - use the unk_token_id for invalid tokens
            unk_token_id = getattr(self.tokenizer, 'unk_token_id', 0)
            mask = inputs >= vocab_size
            inputs = torch.where(mask, torch.tensor(unk_token_id, device=inputs.device, dtype=inputs.dtype), inputs)

        # CRITICAL FIX: Get mask token ID with better fallback handling
        # First check if we have a model reference with the correct mask token ID
        if hasattr(self, 'model') and hasattr(self.model.config, 'mask_token_id'):
            mask_token_id = self.model.config.mask_token_id
        elif hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
            mask_token_id = self.tokenizer.mask_token_id
        else:
            logger.warning("No valid mask token ID found, using a safe fallback")
            mask_token_id = 1  # Use a safe fallback (usually special token)

        # Verify mask token ID is in range
        if mask_token_id >= vocab_size:
            logger.warning(f"Mask token ID {mask_token_id} exceeds vocab size {vocab_size}, using fallback")
            mask_token_id = getattr(self.tokenizer, 'unk_token_id', 0)

        labels = inputs.clone()

        # Create probability matrix
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # CRITICAL FIX: Safer special tokens mask handling
        special_tokens_mask = []
        try:
            # Try the standard way first
            for val in inputs.cpu().tolist():
                # IMPORTANT: If get_special_tokens_mask fails, use an alternative
                try:
                    special_tokens = self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                    special_tokens_mask.append(special_tokens)
                except Exception as e:
                    logger.warning(f"Failed to get special tokens mask: {e}")
                    # Fallback: Mark UNK and PAD tokens as special
                    unk_id = getattr(self.tokenizer, 'unk_token_id', -1)
                    pad_id = getattr(self.tokenizer, 'pad_token_id', -1)
                    special_tokens = [(1 if (t == unk_id or t == pad_id) else 0) for t in val]
                    special_tokens_mask.append(special_tokens)
        except Exception as e:
            logger.error(f"Failed to create special tokens mask: {e}")
            # Default to empty special tokens mask
            special_tokens_mask = [[0] * inputs.size(1) for _ in range(inputs.size(0))]

        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=inputs.device)

        # Don't mask special tokens or padding
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Also explicitly don't mask padding tokens
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            padding_mask = inputs.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        # Get indices to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels for unmasked tokens to -100 so they're ignored in loss
        labels[~masked_indices] = -100

        # First, replace all masked tokens with mask token
        inputs[masked_indices] = mask_token_id

        # Create indices for tokens to replace with random tokens (10% of masked tokens)
        indices_random = torch.bernoulli(torch.full_like(probability_matrix, 0.1)).bool() & masked_indices

        # Generate random tokens safely, if needed
        if indices_random.sum() > 0:
            # CRITICAL FIX: Use a very small subset of vocab for safety - just 100 tokens
            # but ensure we don't exceed vocabulary size
            safe_upper_bound = min(100, vocab_size - 1)

            # CRITICAL FIX: Replace each token individually without creating a separate random token for each
            # This avoids creating a large tensor that might cause memory issues
            default_random_token = min(4, safe_upper_bound)  # A safe default (usually corresponds to a nucleotide)

            # Replace each token individually (very slow but safe)
            indices = torch.nonzero(indices_random)
            for idx in indices:
                batch_idx, token_idx = idx[0].item(), idx[1].item()
                # Use the same token for all replacements to avoid creating too many tensors
                inputs[batch_idx, token_idx] = default_random_token

        # CRITICAL FIX: Verify all token IDs are still in range after masking
        if inputs.max().item() >= vocab_size:
            logger.warning(f"After masking, found token ID exceeding vocab size. Clamping...")
            inputs = torch.clamp(inputs, 0, vocab_size - 1)

        return inputs, labels
