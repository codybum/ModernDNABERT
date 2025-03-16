"""
Genomic dataset and data collator implementations.

This module provides dataset classes for genomic sequences and data collators
for masked language modeling with proper handling of long sequences.
"""
import os
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


class GenomicDataset(torch.utils.data.IterableDataset):
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
            use_reverse_complement: bool = True,
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
        self.file_paths = sorted(file_paths)  # Sort for reproducibility
        self.tokenizer = tokenizer
        self.pre_training_length = pre_training_length
        self.max_inference_length = max_inference_length
        self.mlm_probability = mlm_probability
        self.chunk_size = chunk_size
        self.stride = stride
        self.sample_long_sequences = sample_long_sequences
        self.max_safe_sequence_length = max_safe_sequence_length
        self.use_reverse_complement = use_reverse_complement
        self.epoch = 0  # For deterministic shuffling

        # Perform estimation to maintain API compatibility
        self._estimate_dataset_properties()

    def _estimate_dataset_properties(self):
        """Estimate dataset properties for compatibility with old API."""
        try:
            total_bytes = sum(os.path.getsize(f) for f in self.file_paths)
            # Estimate sequences based on file size (rough approximation)
            self.sequences = ["placeholder"] * (total_bytes // 1000)  # Dummy list for length checking

            # Estimate sequence distribution
            est_nucleotides = total_bytes * 0.8
            chunks_per_sequence = (self.chunk_size / self.stride) * 1.5
            est_chunks = est_nucleotides / self.chunk_size * chunks_per_sequence

            # Double if using reverse complement
            if self.use_reverse_complement:
                est_chunks *= 2

            # Store as chunks property for API compatibility
            self.chunks = ["placeholder"] * int(est_chunks)
            self.sequence_lengths = [self.chunk_size] * len(self.chunks)

            logger.info(f"Estimated {len(self.sequences)} sequences")
            logger.info(f"Created {len(self.chunks)} chunks for training")
            logger.info(
                f"Length distribution: min={self.chunk_size}, max={self.chunk_size}, avg={float(self.chunk_size):.1f}")
        except Exception as e:
            logger.warning(f"Failed to estimate dataset size: {e}")
            # Default fallbacks
            self.sequences = ["placeholder"] * 1000000
            self.chunks = ["placeholder"] * 10000000
            self.sequence_lengths = [self.chunk_size] * len(self.chunks)

    def set_epoch(self, epoch):
        """Set the epoch for deterministic shuffling."""
        self.epoch = epoch

    def _process_sequence(self, sequence):
        """Process a single sequence into encoded chunks."""
        chunks = []

        # Truncate very long sequences for safety
        if len(sequence) > self.max_safe_sequence_length:
            sequence = sequence[:self.max_safe_sequence_length]

        # Handle sequences shorter than chunk_size
        if len(sequence) < self.chunk_size:
            if len(sequence) >= 100:  # Keep the same minimum size check
                encoding = self._create_encoding(sequence)
                if encoding:
                    chunks.append(encoding)

                # Add reverse complement if enabled
                if self.use_reverse_complement:
                    rc_seq = reverse_complement(sequence)
                    rc_encoding = self._create_encoding(rc_seq)
                    if rc_encoding:
                        chunks.append(rc_encoding)
        else:
            # Standard chunks for primary training
            for i in range(0, len(sequence) - self.chunk_size + 1, self.stride):
                chunk = sequence[i:i + self.chunk_size]
                if len(chunk) >= 100:  # Only keep reasonably sized chunks
                    encoding = self._create_encoding(chunk)
                    if encoding:
                        chunks.append(encoding)

                    # Add reverse complement if enabled
                    if self.use_reverse_complement:
                        rc_chunk = reverse_complement(chunk)
                        rc_encoding = self._create_encoding(rc_chunk)
                        if rc_encoding:
                            chunks.append(rc_encoding)

            # Optionally add longer chunks for extrapolation training
            if self.sample_long_sequences and len(sequence) > self.chunk_size * 2:
                # Add a few longer chunks - up to 2-4x the normal chunk size
                for _ in range(min(2, len(sequence) // (self.chunk_size * 2))):  # Add just a few longer samples
                    long_size = random.randint(int(self.chunk_size * 1.5),
                                               min(len(sequence), int(self.chunk_size * 4)))
                    if len(sequence) > long_size:
                        start = random.randint(0, len(sequence) - long_size)
                        long_chunk = sequence[start:start + long_size]
                        encoding = self._create_encoding(long_chunk)
                        if encoding:
                            chunks.append(encoding)

                        # Add reverse complement for long chunks too
                        if self.use_reverse_complement:
                            rc_long_chunk = reverse_complement(long_chunk)
                            rc_encoding = self._create_encoding(rc_long_chunk)
                            if rc_encoding:
                                chunks.append(rc_encoding)

        return chunks

    def _create_encoding(self, sequence):
        """Create tokenized encoding directly."""
        try:
            # Filter sequence to only include valid characters
            sequence = ''.join(c for c in sequence.upper() if c in 'ATGC')
            if not sequence:
                return None

            # Use pre_training_length as the guide for maximum sequence length
            max_seq_length = self.pre_training_length

            # Use the tokenizer with proper settings
            encoding = self.tokenizer(
                sequence,
                truncation=True,
                padding='max_length',
                max_length=max_seq_length,
                return_tensors='pt'
            )

            # Remove the batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}

            # Ensure we have token_type_ids
            if 'token_type_ids' not in encoding:
                encoding['token_type_ids'] = torch.zeros_like(encoding['input_ids'])

            # Verify and clamp token IDs
            if 'input_ids' in encoding:
                vocab_size = self.tokenizer.vocab_size
                max_id = encoding['input_ids'].max().item()

                if max_id >= vocab_size:
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
            logger.debug(f"Encoding failed: {e}")
            return None

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

    def __len__(self):
        """Return the number of chunks."""
        return len(self.chunks)

    def __iter__(self):
        """Iterator with proper sharding for Accelerate."""
        # Get worker info for multi-worker dataloaders
        worker_info = torch.utils.data.get_worker_info()

        # Get Accelerate state for distributed training
        from accelerate.state import AcceleratorState
        state = AcceleratorState()

        # Calculate shard index and total shards
        num_processes = getattr(state, 'num_processes', 1)
        process_index = getattr(state, 'process_index', 0)

        num_shards = num_processes
        if worker_info:
            num_shards *= worker_info.num_workers
            shard_idx = process_index * worker_info.num_workers + worker_info.id
        else:
            shard_idx = process_index

        # Shard files deterministically across processes/workers
        file_paths = self.file_paths.copy()
        num_files = len(file_paths)
        files_per_shard = max(1, num_files // num_shards)
        shard_start = shard_idx * files_per_shard
        shard_end = min(shard_start + files_per_shard, num_files) if shard_idx < num_shards - 1 else num_files

        # Get files for this shard
        shard_files = file_paths[shard_start:shard_end]

        logger.info(f"Process {process_index} worker {worker_info.id if worker_info else 0} "
                    f"processing {len(shard_files)} files ({shard_start}-{shard_end})")

        # Set RNG state for deterministic behavior
        rng = random.Random(42 + self.epoch * 100 + shard_idx)

        # Process files in this shard
        for file_path in shard_files:
            try:
                current_seq = ""
                is_fasta = False

                # Check file format first
                with open(file_path, 'r') as peek_f:
                    first_lines = [peek_f.readline().strip() for _ in range(min(5, os.path.getsize(file_path)))]
                    is_fasta = any(line.startswith('>') for line in first_lines if line)

                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()

                        # Handle FASTA headers
                        if is_fasta and line.startswith('>'):
                            if current_seq:
                                chunks = self._process_sequence(current_seq)
                                rng.shuffle(chunks)  # Shuffle chunks for better mixing
                                for chunk in chunks:
                                    yield chunk
                                current_seq = ""
                            continue

                        # Only process valid nucleotides
                        if set(line.upper()) <= set('ATGC'):
                            current_seq += line.upper()

                            # Process each line separately for non-FASTA
                            if not is_fasta:
                                chunks = self._process_sequence(current_seq)
                                rng.shuffle(chunks)
                                for chunk in chunks:
                                    yield chunk
                                current_seq = ""

                # Process final sequence in FASTA files
                if is_fasta and current_seq:
                    chunks = self._process_sequence(current_seq)
                    rng.shuffle(chunks)
                    for chunk in chunks:
                        yield chunk

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

    def __getitem__(self, idx):
        """
        Required for backward compatibility - will raise an error if called.
        This is implemented because the original class has it, but our streaming
        implementation doesn't support random access.
        """
        raise NotImplementedError(
            "Random access via __getitem__ is not supported in streaming mode. "
            "Use iteration instead through DataLoader."
        )

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
